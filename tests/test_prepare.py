import json

import pandas as pd
import pytest

from screener.data.prepare import ALLOWED_COLUMNS, load_split
from screener.data.prompts import LABELS, build_messages, build_target
from screener.data.sentences import format_numbered, parse_numbered_note, split_unseen_note
from screener.data.split import PARQUET

OPEN_SPLITS = ("train", "validation")


def prompt_text(example, order="label_first"):
    messages = build_messages(example.sentences, example.criterion_type, example.criterion_text, order)
    return "\n".join(m["content"] for m in messages)


def test_test_split_is_sealed():
    with pytest.raises(PermissionError):
        load_split("test")


def test_column_allowlist_excludes_model_outputs():
    assert not [c for c in ALLOWED_COLUMNS if c.startswith("gpt4") or c == "explanation_correctness"]


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_no_gpt4_text_reaches_a_prompt(split):
    examples = load_split(split)
    leaked_columns = ["annotation_id", "gpt4_explanation", "gpt4_eligibility", "explanation_correctness"]
    frame = pd.read_parquet(PARQUET, columns=leaked_columns).set_index("annotation_id")
    for example in examples:
        text = prompt_text(example)
        explanation = frame.loc[example.annotation_id, "gpt4_explanation"]
        assert "gpt4" not in text.lower()
        # Some GPT-4 explanations quote a note sentence verbatim, so the note itself is set aside.
        outside_note = text.replace(format_numbered(example.sentences), "")
        if isinstance(explanation, str) and len(explanation) > 20:
            assert explanation not in outside_note


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_notes_round_trip_and_evidence_ids_resolve(split):
    for example in load_split(split):
        assert parse_numbered_note(format_numbered(example.sentences)) == example.sentences
        assert list(example.sentences) == list(range(len(example.sentences)))
        assert all(i in example.sentences for i in example.evidence)


@pytest.mark.parametrize("split", OPEN_SPLITS)
def test_labels_are_valid_for_the_criterion_type(split):
    for example in load_split(split):
        assert example.label in LABELS[example.criterion_type]


@pytest.mark.parametrize("order", ["label_first", "evidence_first"])
def test_target_round_trips(order):
    assert json.loads(build_target("not excluded", [3, 1, 3], order)) == {"label": "not excluded", "evidence": [1, 3]}


def test_every_prompt_has_all_three_inputs():
    example = load_split("train")[0]
    text = prompt_text(example)
    assert example.criterion_text in text
    assert f"Criterion type: {example.criterion_type}" in text
    assert all(sentence in text for sentence in example.sentences.values())


def test_numbering_errors_are_rejected():
    with pytest.raises(ValueError):
        parse_numbered_note("0. first\n2. skipped one")
    with pytest.raises(ValueError):
        parse_numbered_note("not numbered")


def test_unseen_note_splitting_gives_sequential_ids():
    sentences = split_unseen_note("A 58-year-old woman has a cough.\nDr. Lee notes BP of 120/80. No fever.")
    assert list(sentences) == list(range(len(sentences)))
    assert sentences[1].startswith("Dr. Lee")
    assert len(sentences) == 3


def test_unseen_notes_lose_headings_and_list_marks():
    note = "History:\n- 71-year-old woman with atrial fibrillation.\n* Takes apixaban.\n1. Stroke two years ago.\nAllergies: none."
    assert list(split_unseen_note(note).values()) == [
        "71-year-old woman with atrial fibrillation.",
        "Takes apixaban.",
        "Stroke two years ago.",
        "Allergies: none.",
    ]
