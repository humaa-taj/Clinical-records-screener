"""Reference scaled dot-product attention, written out in full.

Shapes: query [..., n_q, d], key [..., n_k, d], value [..., n_k, d_v].
The mask has True where attention is allowed. It is applied before the softmax, so a masked
position gets exactly zero weight and cannot affect the normalizer.

The softmax subtracts each row's maximum first, so the largest exponent is 0. Very large scores
then cannot overflow. A row with no allowed position outputs zeros.
"""
import math

import torch


def reference_attention(query, key, value, mask=None, scale=None):
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-1, -2)) * scale
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)

    row_max = scores.max(dim=-1, keepdim=True).values
    has_allowed = row_max > -torch.inf
    weights = torch.exp(scores - row_max.where(has_allowed, torch.zeros_like(row_max)))
    normalizer = weights.sum(dim=-1, keepdim=True)
    weights = weights / normalizer.where(normalizer > 0, torch.ones_like(normalizer))
    return torch.matmul(weights, value)
