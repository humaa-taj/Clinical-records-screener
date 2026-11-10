"""Greedy speculative decoding with a small draft model and a large target model.

The output must equal the target model's own greedy output, token for token.

Cache rule, checked after every round: both caches hold every token of the sequence except the last.
The last token is the one both models still have to read.

One round:
1. Draft: the draft model reads the last token and proposes up to k candidates, using its own cache.
2. Verify: the target reads the last token and all candidates in one forward pass. Its prediction at
   each position says what it would have generated there.
3. Accept: keep the longest prefix of candidates that match the target's predictions.
4. Correct: at the first mismatch, take the target's own token instead. If all candidates matched,
   the target's final prediction is a free bonus token. Every round therefore yields at least one
   token decided by the target.
5. Synchronize: crop the target's cache back to the accepted tokens, and crop the draft's too. After
   a fully accepted round with a bonus token, the draft has not read its last candidate, so that
   one token is fed to it.

Speedup depends on acceptance. A rejected candidate wastes draft time and part of a verification
pass, so a poorly matched draft or very short answers can make this slower than ordinary decoding.
"""
from dataclasses import dataclass, field

import torch
from transformers import DynamicCache

from screener.inference.common import DecodeResult, clock, positions_from_mask


@dataclass
class SpeculativeResult(DecodeResult):
    draft_seconds: float = 0.0
    verify_seconds: float = 0.0
    sync_seconds: float = 0.0
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    corrections: int = 0
    bonus_tokens: int = 0
    target_forwards: int = 0
    draft_forwards: int = 0
    state_log: list = field(default_factory=list)

    @property
    def acceptance_rate(self):
        return self.accepted / self.drafted if self.drafted else 0.0


def _advance(model, cache, tokens, full_mask):
    """Feed tokens through a model, extending its cache. Returns logits for each token fed."""
    seen = cache.get_seq_length()
    total = seen + tokens.shape[1]
    mask = full_mask[:, :total]
    logits = model(
        input_ids=tokens,
        attention_mask=mask,
        position_ids=positions_from_mask(mask)[:, seen:],
        past_key_values=cache,
        use_cache=True,
    ).logits
    if cache.get_seq_length() != total:
        raise RuntimeError(f"cache holds {cache.get_seq_length()} tokens, expected {total}")
    return logits


def _check_state(sequence_length, target_cache, draft_cache):
    expected = sequence_length - 1
    if target_cache.get_seq_length() != expected or draft_cache.get_seq_length() != expected:
        raise RuntimeError(
            f"caches out of sync: target={target_cache.get_seq_length()} "
            f"draft={draft_cache.get_seq_length()} expected={expected}"
        )


@torch.no_grad()
def greedy_speculative(target, draft, input_ids, attention_mask=None, max_new_tokens=32, eos_ids=(), draft_length=4, track_state=False):
    if input_ids.shape[0] != 1:
        raise ValueError("this decoder handles one sequence at a time")
    if max_new_tokens < 1 or draft_length < 1:
        raise ValueError("max_new_tokens and draft_length must be at least 1")

    device = input_ids.device
    draft_device = next(draft.parameters()).device
    eos = set(eos_ids)
    prompt_length = input_ids.shape[1]
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    padding = attention_mask.new_ones((1, max_new_tokens + draft_length + 1))
    full_mask = torch.cat([attention_mask, padding], dim=1)
    draft_mask = full_mask.to(draft_device)

    target_cache = DynamicCache(config=target.config)
    draft_cache = DynamicCache(config=draft.config)
    sequence = input_ids
    result = SpeculativeResult([], prompt_length, 0.0, 0.0)

    # Prefill both caches with everything except the last prompt token.
    start = clock(device)
    if prompt_length > 1:
        _advance(target, target_cache, sequence[:, :-1], full_mask)
        _advance(draft, draft_cache, sequence[:, :-1].to(draft_device), draft_mask)
    result.prefill_seconds = clock(device) - start
    _check_state(sequence.shape[1], target_cache, draft_cache)

    finished = False
    while not finished and len(result.tokens) < max_new_tokens:
        remaining = max_new_tokens - len(result.tokens)
        round_size = min(draft_length, remaining)

        # 1. Draft candidates.
        start = clock(draft_device)
        candidates = []
        current = sequence[:, -1:].to(draft_device)
        for _ in range(round_size):
            logits = _advance(draft, draft_cache, current, draft_mask)
            result.draft_forwards += 1
            candidate = logits[0, -1].argmax().item()
            candidates.append(candidate)
            if candidate in eos:
                break
            current = torch.tensor([[candidate]], device=draft_device)
        result.draft_seconds += clock(draft_device) - start

        # 2. Verify all candidates with the target in one pass.
        start = clock(device)
        candidate_tensor = torch.tensor([candidates], device=device)
        verify_input = torch.cat([sequence[:, -1:], candidate_tensor], dim=1)
        predictions = _advance(target, target_cache, verify_input, full_mask)[0].argmax(dim=-1).tolist()
        result.target_forwards += 1
        result.verify_seconds += clock(device) - start

        # 3. Accept the matching prefix.
        accepted = 0
        while accepted < len(candidates) and predictions[accepted] == candidates[accepted]:
            accepted += 1
        new_tokens = candidates[:accepted]

        # 4. Add the target's correction or bonus token.
        if accepted < len(candidates):
            new_tokens.append(predictions[accepted])
            result.corrections += 1
        elif len(candidates) < remaining and candidates[-1] not in eos:
            new_tokens.append(predictions[len(candidates)])
            result.bonus_tokens += 1

        result.rounds += 1
        result.drafted += len(candidates)
        result.accepted += accepted
        if new_tokens[-1] in eos:
            finished = True
        result.tokens.extend(new_tokens)
        sequence = torch.cat([sequence, torch.tensor([new_tokens], device=device)], dim=1)

        # 5. Synchronize both caches to the new sequence.
        start = clock(device)
        needed = sequence.shape[1] - 1
        target_cache.crop(needed)
        draft_cache.crop(needed)
        unread = draft_cache.get_seq_length()
        if unread < needed:
            _advance(draft, draft_cache, sequence[:, unread:needed].to(draft_device), draft_mask)
            result.draft_forwards += 1
        _check_state(sequence.shape[1], target_cache, draft_cache)
        result.sync_seconds += clock(device) - start
        if track_state:
            result.state_log.append((sequence.shape[1], target_cache.get_seq_length(), draft_cache.get_seq_length()))

    result.decode_seconds = result.draft_seconds + result.verify_seconds + result.sync_seconds
    return result
