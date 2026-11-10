"""Run a real model on the tiled attention for an end-to-end check.

The model's own attention layers call a function looked up by name. This swaps that function for the
duration of a `with` block and restores it afterwards. The masks follow the same rules as the original:
a boolean mask if the library built one, otherwise a causal mask whenever more than one query is
processed together. A single new query with a cache may attend to everything cached.
"""
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from screener.attention.tiled_online import tiled_attention


@dataclass
class PatchCounter:
    calls: int = 0


@contextmanager
def use_tiled_attention(query_tile=64, key_tile=64):
    counter = PatchCounter()

    def tiled_forward(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        if dropout:
            raise ValueError("the tiled attention path does not implement attention dropout")
        counter.calls += 1
        groups = getattr(module, "num_key_value_groups", 1)
        if groups > 1:
            key = key.repeat_interleave(groups, dim=1)
            value = value.repeat_interleave(groups, dim=1)
        is_causal = is_causal if is_causal is not None else getattr(module, "is_causal", True)
        mask = attention_mask
        if mask is None and is_causal and query.shape[-2] > 1:
            mask = torch.ones(query.shape[-2], key.shape[-2], dtype=torch.bool, device=query.device).tril()
        output = tiled_attention(query, key, value, mask, query_tile, key_tile, scaling)
        return output.transpose(1, 2).contiguous(), None

    mapping = ALL_ATTENTION_FUNCTIONS._global_mapping
    original = mapping["sdpa"]
    mapping["sdpa"] = tiled_forward
    try:
        yield counter
    finally:
        mapping["sdpa"] = original
