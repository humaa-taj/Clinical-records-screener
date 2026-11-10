import json

import pytest

from screener.data.prepare import load_split
from screener.train.train_lora import ROOT, collate, encode_example, load_tokenizer, resolve_config, sampling_weights


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(ROOT / "models" / "qwen2.5-1.5b")


@pytest.fixture(scope="module")
def example():
    return load_split("train")[0]


@pytest.mark.parametrize("order", ["label_first", "evidence_first"])
def test_loss_covers_only_the_answer(tokenizer, example, order):
    features = encode_example(tokenizer, example, order)
    answer = [t for t in features["labels"] if t != -100]
    text = tokenizer.decode(answer, skip_special_tokens=True)
    assert json.loads(text) == {"label": example.label, "evidence": list(example.evidence)}
    assert answer[-1] == tokenizer.eos_token_id
    assert features["input_ids"][-len(answer):] == answer
    prompt_length = len(features["labels"]) - len(answer)
    assert all(t == -100 for t in features["labels"][:prompt_length])


def test_prompt_contains_no_part_of_the_answer_marker(tokenizer, example):
    features = encode_example(tokenizer, example, "label_first")
    prompt_length = sum(t == -100 for t in features["labels"])
    prompt = tokenizer.decode(features["input_ids"][:prompt_length])
    assert prompt.endswith("<|im_start|>assistant\n")


def test_collate_pads_right_and_masks_padding():
    a = {"input_ids": [1, 2, 3], "labels": [-100, 2, 3], "attention_mask": [1, 1, 1], "label_name": "x"}
    b = {"input_ids": [4], "labels": [4], "attention_mask": [1], "label_name": "y"}
    batch = collate([a, b], pad_id=0)
    assert batch["input_ids"].tolist() == [[1, 2, 3], [4, 0, 0]]
    assert batch["labels"].tolist() == [[-100, 2, 3], [4, -100, -100]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]


def test_sampling_weights():
    assert sampling_weights(["a", "a", "a", "a", "b"], 0.0) == [1, 1, 1, 1, 1]
    assert sampling_weights(["a", "a", "a", "a", "b"], 0.5) == [0.5, 0.5, 0.5, 0.5, 1.0]


def test_run_configs_resolve():
    assert resolve_config("R0")["train"] is False
    assert resolve_config("R2")["oversample_power"] == 0.5
    assert resolve_config("R3")["lora_rank"] == 32
