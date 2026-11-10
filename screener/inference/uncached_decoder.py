"""Reference greedy decoder: no state is reused between steps.

Every step runs the whole sequence through the model, so the work per step grows with its length.
This is the slow but simple path that the cached decoder must match token for token.
"""
import torch

from screener.inference.common import DecodeResult, clock, positions_from_mask


@torch.no_grad()
def greedy_uncached(model, input_ids, attention_mask=None, max_new_tokens=32, eos_ids=(), record_logits=False):
    if input_ids.shape[0] != 1:
        raise ValueError("this decoder handles one sequence at a time")
    device = input_ids.device
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    eos = set(eos_ids)
    sequence, mask = input_ids, attention_mask
    tokens, step_logits = [], []
    prefill_seconds, decode_seconds = 0.0, 0.0

    for step in range(max_new_tokens):
        start = clock(device)
        logits = model(
            input_ids=sequence,
            attention_mask=mask,
            position_ids=positions_from_mask(mask),
            use_cache=False,
        ).logits[:, -1, :]
        next_token = logits.argmax(dim=-1, keepdim=True)
        elapsed = clock(device) - start
        if step == 0:
            prefill_seconds = elapsed
        else:
            decode_seconds += elapsed

        if record_logits:
            step_logits.append(logits[0].float().cpu())
        token = next_token.item()
        tokens.append(token)
        if token in eos:
            break
        sequence = torch.cat([sequence, next_token], dim=1)
        mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)

    return DecodeResult(tokens, input_ids.shape[1], prefill_seconds, decode_seconds, step_logits=step_logits)
