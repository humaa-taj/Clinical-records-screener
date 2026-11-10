"""Check that every pulled resource is complete and usable. Prints no patient text."""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "models" / "qwen2.5-1.5b"
DRAFT = ROOT / "models" / "qwen2.5-0.5b"
OTHER = ROOT / "models" / "smollm2-1.7b"
failures = []


def check(name, passed, detail=""):
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}", flush=True)
    if not passed:
        failures.append(name)


def parse_ids(value):
    if isinstance(value, str):
        return list(ast.literal_eval(value))
    return list(value)


def numbered_sentences(note):
    count = 0
    position = 0
    while True:
        found = note.find(f"{count}. ", position)
        if found < 0 or (found > 0 and not note[found - 1].isspace()):
            return count
        position = found + 1
        count += 1


def check_dataset():
    frame = pd.read_parquet(ROOT / "data" / "raw" / "data" / "train-00000-of-00001.parquet")
    required = {"annotation_id", "patient_id", "note", "criterion_type", "criterion_text",
                "expert_sentences", "expert_eligibility"}
    check("dataset columns", required <= set(frame.columns), str(list(frame.columns)))
    check("dataset rows == 1015", len(frame) == 1015, str(len(frame)))
    patients = frame["patient_id"].nunique()
    check("unique patients == 53 (37+8+8)", patients == 53, str(patients))
    print(frame[list(required)].isna().sum().to_string())
    print(frame.groupby("criterion_type")["expert_eligibility"].value_counts().to_string())

    ids = frame["expert_sentences"].map(parse_ids)
    counts = frame["note"].map(numbered_sentences)
    out_of_range = sum(any(i >= c for i in row) for row, c in zip(ids, counts))
    check("evidence IDs exist in numbered notes", out_of_range == 0, f"{out_of_range} rows out of range")
    print(f"rows with empty expert evidence: {sum(len(r) == 0 for r in ids)}")
    print(f"sentences per note: min={counts.min()} median={int(counts.median())} max={counts.max()}")

    tokenizer = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    lengths = np.array([len(tokenizer(n)["input_ids"]) for n in frame["note"].unique()])
    print(f"note tokens: median={int(np.median(lengths))} p95={int(np.percentile(lengths, 95))} max={lengths.max()}")


def check_papers():
    lock = json.loads((ROOT / "resources.lock.json").read_text(encoding="utf-8"))
    good = 0
    for entry in lock["papers"].values():
        with open(ROOT / entry["file"], "rb") as handle:
            good += handle.read(5) == b"%PDF-"
    check("11 paper PDFs", good == 11, str(good))


def check_weights(folder):
    total = 0
    for path in folder.glob("*.safetensors"):
        with safe_open(path, framework="pt") as handle:
            for key in handle.keys():
                total += int(np.prod(handle.get_slice(key).get_shape()))
    config = AutoConfig.from_pretrained(folder, local_files_only=True)
    check(f"weights {folder.name}", total > 0,
          f"params={total/1e6:.1f}M arch={config.architectures} layers={config.num_hidden_layers}")


def check_tokenizers():
    target = AutoTokenizer.from_pretrained(TARGET, local_files_only=True)
    draft = AutoTokenizer.from_pretrained(DRAFT, local_files_only=True)
    check("draft/target vocabulary identical", target.get_vocab() == draft.get_vocab(), str(len(target)))
    check("draft/target special tokens identical",
          target.special_tokens_map == draft.special_tokens_map
          and target.eos_token_id == draft.eos_token_id
          and target.pad_token_id == draft.pad_token_id)
    check("draft/target chat template identical", target.chat_template == draft.chat_template)
    samples = ["Patient has HbA1c of 7.2% and eGFR < 30 mL/min.", '{"label": "not enough information", "ids": [0, 3]}']
    check("draft/target encodings identical", all(target(s)["input_ids"] == draft(s)["input_ids"] for s in samples))


def check_gpu():
    if not torch.cuda.is_available():
        check("CUDA available", False)
        return
    check("CUDA available", True, torch.cuda.get_device_name(0))
    for folder in (DRAFT, TARGET):
        torch.cuda.reset_peak_memory_stats()
        tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float16, local_files_only=True).to("cuda")
        inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        peak = torch.cuda.max_memory_allocated() / 1024**3
        check(f"GPU generate {folder.name}", output.shape[1] > inputs["input_ids"].shape[1], f"peak={peak:.2f} GB")
        del model
        torch.cuda.empty_cache()


def main():
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
    check_dataset()
    check_papers()
    for folder in (TARGET, DRAFT, OTHER):
        check_weights(folder)
    check_tokenizers()
    check_gpu()
    check("punkt_tab present", (ROOT / "data" / "nltk_data" / "tokenizers" / "punkt_tab").is_dir())
    print(f"\n{len(failures)} failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
