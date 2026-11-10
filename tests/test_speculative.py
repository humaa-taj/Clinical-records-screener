"""Speculative decoding must equal ordinary greedy decoding of the target, whatever the draft does."""
import copy

import pytest
import torch
from transformers import GenerationConfig

from screener.eval.predict import eos_token_ids
from screener.inference.cached_decoder import greedy_cached
from screener.inference.common import benchmark_prompt, load_for_inference
from screener.inference.speculative import greedy_speculative
from screener.train.train_lora import ROOT

MODEL = ROOT / "models" / "qwen2.5-0.5b"


@pytest.fixture(scope="module")
def setup():
    if not MODEL.exists():
        pytest.skip("model weights not present")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    target, tokenizer = load_for_inference(MODEL, device=device, dtype=torch.float16)
    return target, tokenizer, device


def noisy_copy(model, strength, seed=0):
    """A draft that disagrees with the target, by an amount set by the noise strength."""
    clone = copy.deepcopy(model)
    generator = torch.Generator(device=next(model.parameters()).device).manual_seed(seed)
    with torch.no_grad():
        weight = clone.lm_head.weight
        noise = torch.randn(weight.shape, generator=generator, device=weight.device, dtype=weight.dtype)
        weight.copy_(weight + strength * weight.std() * noise)
    return clone


@pytest.fixture(scope="module")
def partly_wrong_draft(setup):
    return noisy_copy(setup[0], 0.05)


@pytest.fixture(scope="module")
def mostly_wrong_draft(setup):
    return noisy_copy(setup[0], 5.0)


def prompt(setup, length):
    return torch.tensor([benchmark_prompt(setup[1], length)], device=setup[2])


@pytest.mark.parametrize("draft_length", [1, 2, 4, 8])
@pytest.mark.parametrize("input_length,output_length", [(16, 12), (64, 40), (256, 64)])
def test_output_equals_greedy_with_imperfect_draft(setup, partly_wrong_draft, draft_length, input_length, output_length):
    target = setup[0]
    ids = prompt(setup, input_length)
    expected = greedy_cached(target, ids, max_new_tokens=output_length).tokens
    result = greedy_speculative(target, partly_wrong_draft, ids, max_new_tokens=output_length, draft_length=draft_length)
    assert result.tokens == expected
    assert len(result.tokens) == output_length


def test_identical_draft_accepts_everything(setup):
    target = setup[0]
    ids = prompt(setup, 64)
    expected = greedy_cached(target, ids, max_new_tokens=30).tokens
    result = greedy_speculative(target, target, ids, max_new_tokens=30, draft_length=4)
    assert result.tokens == expected
    assert result.acceptance_rate == 1.0 and result.corrections == 0
    assert result.accepted + result.bonus_tokens + result.corrections == len(result.tokens)


def test_rejections_are_corrected(setup, mostly_wrong_draft):
    target = setup[0]
    ids = prompt(setup, 64)
    expected = greedy_cached(target, ids, max_new_tokens=30).tokens
    result = greedy_speculative(target, mostly_wrong_draft, ids, max_new_tokens=30, draft_length=4)
    assert result.tokens == expected
    assert result.corrections > 0 and result.acceptance_rate < 0.5


def test_each_verification_replaces_at_least_one_target_step(setup, partly_wrong_draft):
    result = greedy_speculative(setup[0], partly_wrong_draft, prompt(setup, 64), max_new_tokens=40, draft_length=4)
    assert result.target_forwards == result.rounds <= 40
    assert 0 < result.accepted <= result.drafted


def test_caches_stay_synchronized_every_round(setup, partly_wrong_draft):
    result = greedy_speculative(setup[0], partly_wrong_draft, prompt(setup, 32), max_new_tokens=30, draft_length=4, track_state=True)
    assert result.state_log
    for sequence_length, target_length, draft_length in result.state_log:
        assert target_length == draft_length == sequence_length - 1


@pytest.mark.parametrize("draft_length,limit", [(4, 1), (4, 4), (4, 5), (8, 3), (1, 7)])
def test_never_exceeds_the_token_limit(setup, partly_wrong_draft, draft_length, limit):
    target = setup[0]
    ids = prompt(setup, 32)
    expected = greedy_cached(target, ids, max_new_tokens=limit).tokens
    assert greedy_speculative(target, partly_wrong_draft, ids, max_new_tokens=limit, draft_length=draft_length).tokens == expected


@pytest.mark.parametrize("draft", ["target", "noisy"])
def test_eos_stops_at_the_same_token(setup, partly_wrong_draft, draft):
    target = setup[0]
    ids = prompt(setup, 32)
    free_run = greedy_cached(target, ids, max_new_tokens=20).tokens
    stop_token = free_run[6]
    expected = greedy_cached(target, ids, max_new_tokens=20, eos_ids=[stop_token]).tokens
    helper = target if draft == "target" else partly_wrong_draft
    result = greedy_speculative(target, helper, ids, max_new_tokens=20, eos_ids=[stop_token], draft_length=4)
    assert result.tokens == expected and result.tokens[-1] == stop_token


def test_single_token_prompt(setup, partly_wrong_draft):
    target = setup[0]
    ids = prompt(setup, 1)
    expected = greedy_cached(target, ids, max_new_tokens=12).tokens
    assert greedy_speculative(target, partly_wrong_draft, ids, max_new_tokens=12, draft_length=3).tokens == expected


def test_left_padding(setup, partly_wrong_draft):
    target, tokenizer, device = setup
    ids = prompt(setup, 64)
    pad = torch.full((1, 5), tokenizer.pad_token_id, device=device)
    padded = torch.cat([pad, ids], dim=1)
    mask = torch.cat([torch.zeros_like(pad), torch.ones_like(ids)], dim=1)
    expected = greedy_cached(target, ids, max_new_tokens=24).tokens
    assert greedy_speculative(target, partly_wrong_draft, padded, attention_mask=mask, max_new_tokens=24, draft_length=4).tokens == expected


def test_draft_on_a_different_device_from_the_target(setup):
    target, _, device = setup
    if device.type != "cuda":
        pytest.skip("needs a GPU")
    draft_cpu, _ = load_for_inference(MODEL, device=torch.device("cpu"), dtype=torch.float32)
    ids = prompt(setup, 32)
    expected = greedy_cached(target, ids, max_new_tokens=12).tokens
    assert greedy_speculative(target, draft_cpu, ids, max_new_tokens=12, draft_length=3).tokens == expected


def test_matches_library_assisted_generation(setup, partly_wrong_draft):
    target, tokenizer, _ = setup
    ids = prompt(setup, 64)
    eos = eos_token_ids(target, tokenizer)
    config = GenerationConfig(do_sample=False, num_beams=1, repetition_penalty=1.0, max_new_tokens=32,
                              eos_token_id=eos, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    library = target.generate(ids, attention_mask=torch.ones_like(ids), generation_config=config,
                              assistant_model=partly_wrong_draft)[0, ids.shape[1]:].tolist()
    assert greedy_speculative(target, partly_wrong_draft, ids, max_new_tokens=32, eos_ids=eos, draft_length=4).tokens == library


@pytest.mark.parametrize("kwargs", [{"max_new_tokens": 0}, {"draft_length": 0}])
def test_invalid_arguments(setup, kwargs):
    with pytest.raises(ValueError):
        greedy_speculative(setup[0], setup[0], prompt(setup, 8), **kwargs)
