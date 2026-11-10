"""Greedy decoder that reuses and updates the Transformer's key/value cache.

Prefill: the whole prompt goes through the model once. Each attention layer stores its keys and
values, of shape [batch, heads, length, head_dim], in the cache.

Decode: only the newest token goes through the model. Three things must then be kept consistent:

- Attention mask: it must still cover the full sequence, including cached tokens. The library
  sizes the causal mask as cache length + new tokens, so a mask covering only the new token breaks.
- Position IDs: the rotary embedding needs the new token's true position, one past the previous
  real token. Starting again from zero would give every new token the wrong rotation.
- Cache length: the layers append in place, so after each forward pass the cache must be exactly
  one token longer. This is asserted at every step.

Memory grows linearly with sequence length. Each update concatenates, so old and new tensors exist
together briefly, and the step's peak is higher than the cache's final size.

Cached and uncached runs can differ slightly in their floating-point values, since the matrices are
multiplied in different groupings. Tokens only change if the top two logits are nearly tied.
"""
import torch
from transformers import DynamicCache

from screener.inference.common import DecodeResult, clock, positions_from_mask


def cache_size_bytes(cache):
    return sum(
        layer.keys.numel() * layer.keys.element_size() + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )


def _check_length(cache, expected):
    actual = cache.get_seq_length()
    if actual != expected:
        raise RuntimeError(f"cache holds {actual} tokens, expected {expected}")


@torch.no_grad()
def greedy_cached(model, input_ids, attention_mask=None, max_new_tokens=32, eos_ids=(), record_logits=False, track_cache=False):
    if input_ids.shape[0] != 1:
        raise ValueError("this decoder handles one sequence at a time")
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    eos = set(eos_ids)
    cache = DynamicCache(config=model.config)
    mask = attention_mask
    positions = positions_from_mask(mask)
    processed = input_ids.shape[1]
    tokens, step_logits, cache_lengths, cache_bytes = [], [], [], []

    def record_state():
        if track_cache:
            cache_lengths.append(cache.get_seq_length())
            cache_bytes.append(cache_size_bytes(cache))

    # Prefill: run the prompt once and store its keys and values.
    start = clock(device)
    logits = model(
        input_ids=input_ids,
        attention_mask=mask,
        position_ids=positions,
        past_key_values=cache,
        use_cache=True,
    ).logits[:, -1, :]
    next_token = logits.argmax(dim=-1, keepdim=True)
    prefill_seconds = clock(device) - start
    _check_length(cache, processed)
    record_state()

    decode_seconds = 0.0
    while True:
        if record_logits:
            step_logits.append(logits[0].float().cpu())
        token = next_token.item()
        tokens.append(token)
        if token in eos or len(tokens) == max_new_tokens:
            break

        # Decode: send only the new token, with the full mask and its true position.
        mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
        positions = positions[:, -1:] + 1
        start = clock(device)
        logits = model(
            input_ids=next_token,
            attention_mask=mask,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
        ).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        decode_seconds += clock(device) - start
        processed += 1
        _check_length(cache, processed)
        record_state()

    return DecodeResult(tokens, input_ids.shape[1], prefill_seconds, decode_seconds, cache_lengths, cache_bytes, step_logits)
