import inspect
import math

import pytest
import torch
import torch.nn.functional as F

import screener.attention.tiled_online as tiled_module
from screener.attention.model_patch import use_tiled_attention
from screener.attention.reference import reference_attention
from screener.attention.tiled_online import TileStats, online_softmax_step, tiled_attention
from screener.inference.cached_decoder import greedy_cached
from screener.inference.common import benchmark_prompt, load_for_inference
from screener.inference.uncached_decoder import greedy_uncached
from screener.train.train_lora import ROOT

MODEL = ROOT / "models" / "qwen2.5-0.5b"
TOLERANCE = {torch.float32: 2e-4, torch.float64: 1e-11}


def tensors(shape_q, shape_k, dtype=torch.float64, seed=0, magnitude=1.0):
    generator = torch.Generator().manual_seed(seed)
    make = lambda shape: torch.randn(shape, generator=generator, dtype=dtype) * magnitude
    return make(shape_q), make(shape_k), make(shape_k)


def causal(n, batch_dims=0):
    mask = torch.ones(n, n, dtype=torch.bool).tril()
    return mask[(None,) * batch_dims]


def assert_close(a, b, dtype):
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    assert (a - b).abs().max().item() <= TOLERANCE[dtype]


def test_reference_by_hand():
    query = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    value = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    first = math.exp(1 / math.sqrt(2)) / (math.exp(1 / math.sqrt(2)) + 1)
    expected = torch.tensor([[2 * first, 4 * (1 - first)]], dtype=torch.float64)
    assert (reference_attention(query, key, value) - expected).abs().max() < 1e-12


def test_reference_mask_gives_masked_keys_zero_weight():
    query, key, value = tensors((1, 4), (3, 4))
    mask = torch.tensor([[True, False, False]])
    assert (reference_attention(query, key, value, mask) - value[:1]).abs().max() < 1e-12


@pytest.mark.parametrize("n", [1, 7, 64])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_reference_matches_pytorch(n, dtype):
    query, key, value = tensors((2, 4, n, 16), (2, 4, n, 16), dtype)
    mask = causal(n, 2)
    expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
    assert_close(reference_attention(query, key, value, mask), expected, dtype)


def test_large_scores_do_not_overflow():
    query, key, value = tensors((1, 8, 16), (1, 8, 16), torch.float64, magnitude=1e4)
    assert torch.isfinite(reference_attention(query, key, value)).all()
    assert_close(tiled_attention(query, key, value, query_tile=3, key_tile=3), reference_attention(query, key, value), torch.float64)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("n", [1, 2, 7, 33, 128])
@pytest.mark.parametrize("tile", [1, 3, 16, 64, 1000])
def test_tiled_matches_reference_with_causal_mask(n, tile, dtype):
    query, key, value = tensors((2, 4, n, 16), (2, 4, n, 16), dtype, seed=n + tile)
    mask = causal(n, 2)
    assert_close(tiled_attention(query, key, value, mask, tile, tile), reference_attention(query, key, value, mask), dtype)


@pytest.mark.parametrize("q_tile,k_tile", [(1, 5), (5, 1), (4, 7), (8, 3)])
def test_tiled_handles_different_tile_shapes_and_rectangles(q_tile, k_tile):
    query, key, value = tensors((3, 21, 16), (3, 40, 16))
    assert_close(tiled_attention(query, key, value, None, q_tile, k_tile), reference_attention(query, key, value), torch.float64)


def test_tiled_handles_padding_style_masks():
    query, key, value = tensors((1, 2, 12, 16), (1, 2, 12, 16))
    keep = torch.tensor([False, False, False] + [True] * 9)
    mask = (causal(12) & keep[None, :])[None, None]
    assert_close(tiled_attention(query, key, value, mask, 5, 4), reference_attention(query, key, value, mask), torch.float64)


def test_tiled_handles_random_masks():
    query, key, value = tensors((2, 30, 16), (2, 30, 16))
    mask = torch.rand((2, 30, 30), generator=torch.Generator().manual_seed(1)) > 0.5
    assert_close(tiled_attention(query, key, value, mask, 7, 4), reference_attention(query, key, value, mask), torch.float64)


def test_fully_masked_rows_output_zero_without_nan():
    query, key, value = tensors((4, 16), (4, 16))
    mask = torch.ones(4, 4, dtype=torch.bool)
    mask[2] = False
    result = tiled_attention(query, key, value, mask, 2, 2)
    assert torch.isfinite(result).all() and (result[2] == 0).all()
    assert_close(result, reference_attention(query, key, value, mask), torch.float64)


def test_largest_score_in_a_later_tile_rescales_earlier_terms():
    """Fails if the exp(m_old - m_new) rescaling is removed."""
    query = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    key = torch.tensor([[1.0, 0, 0, 0], [2.0, 0, 0, 0], [40.0, 0, 0, 0], [0.5, 0, 0, 0]], dtype=torch.float64)
    value = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 1.0]], dtype=torch.float64)
    assert_close(tiled_attention(query, key, value, None, 1, 1), reference_attention(query, key, value), torch.float64)


def test_online_step_matches_direct_softmax_on_two_tiles():
    scores = torch.tensor([[0.5, -1.0, 3.0, 2.0]], dtype=torch.float64)
    values = torch.eye(4, dtype=torch.float64)
    running_max = torch.tensor([-torch.inf], dtype=torch.float64)
    running_sum = torch.zeros(1, dtype=torch.float64)
    output = torch.zeros(1, 4, dtype=torch.float64)
    for a, b in ((0, 2), (2, 4)):
        running_max, running_sum, output = online_softmax_step(running_max, running_sum, output, scores[:, a:b], values[a:b])
    assert running_max.item() == 3.0
    assert ((output / running_sum) - torch.softmax(scores, dim=-1)).abs().max() < 1e-12


def test_causal_upper_tiles_are_skipped():
    query, key, value = tensors((16, 8), (16, 8))
    stats = TileStats()
    tiled_attention(query, key, value, causal(16), 4, 4, stats=stats)
    assert stats.computed == 10 and stats.skipped == 6
    assert stats.largest_tile_scores == 16


def test_tiled_does_not_call_a_fused_backend():
    source = inspect.getsource(tiled_module)
    assert "scaled_dot_product_attention" not in source and "softmax(" not in source.replace("online_softmax_step(", "")


def test_invalid_tile_size():
    query, key, value = tensors((2, 4), (2, 4))
    with pytest.raises(ValueError):
        tiled_attention(query, key, value, None, 0, 2)


@pytest.fixture(scope="module")
def loaded():
    if not MODEL.exists():
        pytest.skip("model weights not present")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_for_inference(MODEL, device=device, dtype=torch.float32) + (device,)


@pytest.mark.parametrize("decoder", [greedy_cached, greedy_uncached])
@pytest.mark.parametrize("tile", [8, 64])
def test_real_model_generates_identical_tokens(loaded, decoder, tile):
    model, tokenizer, device = loaded
    ids = torch.tensor([benchmark_prompt(tokenizer, 96)], device=device)
    expected = decoder(model, ids, max_new_tokens=16).tokens
    with use_tiled_attention(tile, tile) as counter:
        tiled = decoder(model, ids, max_new_tokens=16).tokens
    assert tiled == expected
    assert counter.calls > 0


def test_patch_is_removed_afterwards(loaded):
    model, tokenizer, device = loaded
    ids = torch.tensor([benchmark_prompt(tokenizer, 16)], device=device)
    with use_tiled_attention() as counter:
        greedy_cached(model, ids, max_new_tokens=2)
    calls = counter.calls
    greedy_cached(model, ids, max_new_tokens=2)
    assert counter.calls == calls
