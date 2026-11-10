"""Tiled attention with online softmax, following the FlashAttention algorithm in plain PyTorch.

Queries and keys are split into tiles, and the full score matrix never exists in memory. Softmax
normally needs a whole row at once, because each weight is divided by the row's sum. Online softmax
removes that need by keeping three running values for each query row:

    m: the largest score seen so far
    l: the sum of exp(score - m) so far
    O: the sum of exp(score - m) * value so far

When a new tile raises m from m_old to m_new, the earlier terms were measured against the old
maximum, so l and O are multiplied by exp(m_old - m_new) before adding the new tile. After the last
tile, O / l equals the true softmax-weighted output. No backend attention kernel is called.
"""
import math
from dataclasses import dataclass

import torch


@dataclass
class TileStats:
    computed: int = 0
    skipped: int = 0
    largest_tile_scores: int = 0


def online_softmax_step(running_max, running_sum, running_output, scores, value_tile):
    """Fold one tile of scores into the running values for a block of query rows."""
    new_max = torch.maximum(running_max, scores.max(dim=-1).values)
    # A row with every position masked so far has no maximum yet. Using 0 keeps its terms at zero.
    anchor = new_max.where(new_max > -torch.inf, torch.zeros_like(new_max))
    rescale = torch.exp(running_max - anchor)
    weights = torch.exp(scores - anchor.unsqueeze(-1))
    new_sum = rescale * running_sum + weights.sum(dim=-1)
    new_output = rescale.unsqueeze(-1) * running_output + torch.matmul(weights, value_tile)
    return new_max, new_sum, new_output


def tiled_attention(query, key, value, mask=None, query_tile=64, key_tile=64, scale=None, stats=None):
    """Compute attention tile by tile. Fully masked tiles are skipped without computing scores."""
    if query_tile < 1 or key_tile < 1:
        raise ValueError("tile sizes must be positive")
    if scale is None:
        scale = 1.0 / math.sqrt(query.shape[-1])
    stats = stats if stats is not None else TileStats()
    n_query, n_key = query.shape[-2], key.shape[-2]
    blocks = []

    for q_start in range(0, n_query, query_tile):
        q_end = min(q_start + query_tile, n_query)
        query_block = query[..., q_start:q_end, :]
        row_shape = query_block.shape[:-1]
        running_max = torch.full(row_shape, -torch.inf, dtype=query.dtype, device=query.device)
        running_sum = torch.zeros(row_shape, dtype=query.dtype, device=query.device)
        running_output = torch.zeros((*row_shape, value.shape[-1]), dtype=value.dtype, device=value.device)

        for k_start in range(0, n_key, key_tile):
            k_end = min(k_start + key_tile, n_key)
            mask_tile = None if mask is None else mask[..., q_start:q_end, k_start:k_end]
            if mask_tile is not None and not mask_tile.any():
                stats.skipped += 1
                continue

            key_block = key[..., k_start:k_end, :]
            scores = torch.matmul(query_block, key_block.transpose(-1, -2)) * scale
            if mask_tile is not None:
                scores = scores.masked_fill(~mask_tile, -torch.inf)
            running_max, running_sum, running_output = online_softmax_step(
                running_max, running_sum, running_output, scores, value[..., k_start:k_end, :]
            )
            stats.computed += 1
            stats.largest_tile_scores = max(stats.largest_tile_scores, scores.numel())

        normalizer = running_sum.where(running_sum > 0, torch.ones_like(running_sum)).unsqueeze(-1)
        blocks.append(running_output / normalizer)

    return torch.cat(blocks, dim=-2)
