"""Decoder correctness on the real 0.5B Qwen model. Skipped if the weights are not downloaded."""
import pytest
import torch
from transformers import GenerationConfig

from screener.eval.predict import eos_token_ids
from screener.inference.cached_decoder import cache_size_bytes, greedy_cached
from screener.inference.common import benchmark_prompt, load_for_inference, positions_from_mask
from screener.inference.uncached_decoder import greedy_uncached
from screener.train.train_lora import ROOT

MODEL = ROOT / "models" / "qwen2.5-0.5b"


@pytest.fixture(scope="module")
def loaded():
    if not MODEL.exists():
        pytest.skip("model weights not present")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return load_for_inference(MODEL, device=device, dtype=torch.float32) + (device,)


def prompt(loaded, length):
    _, tokenizer, device = loaded
    return torch.tensor([benchmark_prompt(tokenizer, length)], device=device)


@pytest.mark.parametrize("input_length", [8, 64, 256])
@pytest.mark.parametrize("output_length", [1, 16, 48])
def test_cached_matches_uncached(loaded, input_length, output_length):
    model = loaded[0]
    ids = prompt(loaded, input_length)
    uncached = greedy_uncached(model, ids, max_new_tokens=output_length)
    cached = greedy_cached(model, ids, max_new_tokens=output_length)
    assert cached.tokens == uncached.tokens
    assert len(cached.tokens) == output_length


def test_logits_agree_closely(loaded):
    model = loaded[0]
    ids = prompt(loaded, 64)
    uncached = greedy_uncached(model, ids, max_new_tokens=16, record_logits=True)
    cached = greedy_cached(model, ids, max_new_tokens=16, record_logits=True)
    difference = (torch.stack(uncached.step_logits) - torch.stack(cached.step_logits)).abs().max().item()
    assert difference < 1e-2


def test_matches_library_generate(loaded):
    model, tokenizer, _ = loaded
    ids = prompt(loaded, 64)
    eos = eos_token_ids(model, tokenizer)
    config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=32,
                              eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    library = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=config)[0, ids.shape[1]:].tolist()
    assert greedy_cached(model, ids, max_new_tokens=32, eos_ids=eos).tokens == library


def test_cache_grows_one_token_per_step(loaded):
    model = loaded[0]
    ids = prompt(loaded, 32)
    result = greedy_cached(model, ids, max_new_tokens=10, track_cache=True)
    assert result.cache_lengths == list(range(32, 42))
    per_token = result.cache_bytes[0] / 32
    assert all(b == pytest.approx(per_token * n) for b, n in zip(result.cache_bytes, result.cache_lengths))


def test_left_padding_does_not_change_the_output(loaded):
    model, tokenizer, device = loaded
    ids = prompt(loaded, 64)
    pad = torch.full((1, 7), tokenizer.pad_token_id, device=device)
    padded = torch.cat([pad, ids], dim=1)
    mask = torch.cat([torch.zeros_like(pad), torch.ones_like(ids)], dim=1)
    plain = greedy_cached(model, ids, max_new_tokens=24)
    assert greedy_cached(model, padded, attention_mask=mask, max_new_tokens=24).tokens == plain.tokens
    assert greedy_uncached(model, padded, attention_mask=mask, max_new_tokens=24).tokens == plain.tokens


def test_positions_ignore_left_padding():
    mask = torch.tensor([[0, 0, 1, 1, 1]])
    assert positions_from_mask(mask).tolist() == [[0, 0, 0, 1, 2]]


def test_stops_at_eos_and_includes_it(loaded):
    model = loaded[0]
    ids = prompt(loaded, 32)
    free_run = greedy_cached(model, ids, max_new_tokens=12)
    stop_token = free_run.tokens[5]
    cut = free_run.tokens.index(stop_token)
    stopped = greedy_cached(model, ids, max_new_tokens=12, eos_ids=[stop_token])
    assert stopped.tokens == free_run.tokens[: cut + 1]
    assert greedy_uncached(model, ids, max_new_tokens=12, eos_ids=[stop_token]).tokens == stopped.tokens


def test_wrong_positions_are_detected_by_comparison(loaded):
    """A negative control: dropping the position update must change the tokens."""
    model = loaded[0]
    ids = prompt(loaded, 64)
    good = greedy_cached(model, ids, max_new_tokens=24).tokens
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        mask = torch.ones_like(ids)
        logits = model(input_ids=ids, attention_mask=mask, position_ids=positions_from_mask(mask), past_key_values=cache, use_cache=True).logits[:, -1]
        tokens, frozen = [], torch.tensor([[0]], device=ids.device)
        for _ in range(24):
            token = logits.argmax(-1, keepdim=True)
            tokens.append(token.item())
            mask = torch.cat([mask, mask.new_ones((1, 1))], dim=1)
            logits = model(input_ids=token, attention_mask=mask, position_ids=frozen, past_key_values=cache, use_cache=True).logits[:, -1]
    assert tokens != good


def test_cache_size_counts_keys_and_values(loaded):
    model = loaded[0]
    from transformers import DynamicCache

    ids = prompt(loaded, 8)
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        model(input_ids=ids, past_key_values=cache, use_cache=True)
    expected = sum((l.keys.numel() + l.values.numel()) * 4 for l in cache.layers)
    assert cache_size_bytes(cache) == expected
