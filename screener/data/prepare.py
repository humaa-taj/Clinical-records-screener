"""Load split examples, and audit prompt lengths.

Only allowlisted columns are ever read from the dataset, so GPT-4 predictions and explanations
cannot reach a prompt, a target or an evaluation. The test split requires an explicit opt-in.

Usage: python -m screener.data.prepare   (writes results/truncation.json)
"""
import ast
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from screener.data.prompts import build_messages, build_target
from screener.data.sentences import parse_numbered_note
from screener.data.split import MANIFEST, PARQUET, ROOT

INPUT_COLUMNS = ["annotation_id", "patient_id", "trial_id", "note", "criterion_type", "criterion_text"]
LABEL_COLUMNS = ["expert_eligibility", "expert_sentences"]
ALLOWED_COLUMNS = INPUT_COLUMNS + LABEL_COLUMNS
TARGET_MODEL = ROOT / "models" / "qwen2.5-1.5b"


@dataclass(frozen=True)
class Example:
    annotation_id: int
    patient_id: str
    trial_id: str
    sentences: dict
    criterion_type: str
    criterion_text: str
    label: str
    evidence: tuple


def _parse_ids(value):
    return tuple(sorted(int(i) for i in ast.literal_eval(value)))


def load_manifest():
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def load_split(name, allow_test=False):
    """Return the examples of one split. Reading the test split must be requested explicitly."""
    if name == "test" and not allow_test:
        raise PermissionError("the test split is sealed until final evaluation")
    manifest = load_manifest()
    removed = {row["annotation_id"] for row in manifest["removed_rows"]}
    frame = pd.read_parquet(PARQUET, columns=ALLOWED_COLUMNS)
    frame = frame[frame["patient_id"].isin(manifest["splits"][name]["patients"])]
    frame = frame[~frame["annotation_id"].isin(removed)].sort_values("annotation_id")
    examples = [
        Example(
            annotation_id=int(row.annotation_id),
            patient_id=row.patient_id,
            trial_id=row.trial_id,
            sentences=parse_numbered_note(row.note),
            criterion_type=row.criterion_type,
            criterion_text=row.criterion_text.strip(),
            label=row.expert_eligibility,
            evidence=_parse_ids(row.expert_sentences),
        )
        for row in frame.itertuples(index=False)
    ]
    if len(examples) != manifest["splits"][name]["num_usable_rows"]:
        raise ValueError(f"{name}: row count does not match the manifest")
    return examples


def render_prompt(tokenizer, example, order="label_first"):
    messages = build_messages(example.sentences, example.criterion_type, example.criterion_text, order)
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def truncation_audit(tokenizer, splits=("train", "validation")):
    """Measure full training-example lengths. Training uses no truncation at all."""
    report = {"policy": "no truncation in training or evaluation", "tokenizer": TARGET_MODEL.name, "splits": {}}
    for name in splits:
        lengths = []
        for example in load_split(name):
            prompt = render_prompt(tokenizer, example)
            target = build_target(example.label, example.evidence) + tokenizer.eos_token
            lengths.append(len(tokenizer(prompt + target, add_special_tokens=False)["input_ids"]))
        values = np.array(lengths)
        report["splits"][name] = {
            "examples": int(values.size),
            "truncated": 0,
            "tokens_median": int(np.median(values)),
            "tokens_p95": int(np.percentile(values, 95)),
            "tokens_max": int(values.max()),
        }
    return report


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL, local_files_only=True)
    report = truncation_audit(tokenizer)
    path = ROOT / "results" / "truncation.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
