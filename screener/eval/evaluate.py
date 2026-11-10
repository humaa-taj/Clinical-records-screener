"""Evaluate a model, or a baseline, and write every required metric.

The test split can only be read with --final, and each such run is logged.

Usage:
  python -m screener.eval.evaluate --model models/qwen2.5-1.5b --adapter runs/R2/adapter \
      --split validation --output results/eval/after_validation
  python -m screener.eval.evaluate --baseline --split validation --output results/eval/baseline_validation
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from screener.data.prepare import load_split
from screener.eval.explain import explain
from screener.eval.metrics import bootstrap_intervals, full_report, majority_baseline
from screener.eval.predict import decoding_settings, predict_examples

ROOT = Path(__file__).resolve().parents[2]
ACCESS_LOG = ROOT / "results" / "test_access_log.jsonl"


def check_split_access(split, final):
    if split == "test" and not final:
        raise PermissionError("the test split requires --final")
    if final and split != "test":
        raise ValueError("--final applies only to the test split")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def log_test_access(details, path=ACCESS_LOG):
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time_utc": datetime.now(timezone.utc).isoformat(), "commit": git_commit(), **details}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def prediction_record(example, result):
    """Everything needed to trace one prediction back to the patient note."""
    if result.valid:
        explanation = explain(result.label, result.evidence, example.sentences)
    else:
        explanation = "The model output could not be verified: " + "; ".join(result.errors)
    return {
        "annotation_id": example.annotation_id,
        "patient_id": example.patient_id,
        "trial_id": example.trial_id,
        "criterion_type": example.criterion_type,
        "criterion": example.criterion_text,
        "gold_label": example.label,
        "predicted_label": result.label,
        "valid": result.valid,
        "gold_evidence": list(example.evidence),
        "cited_sentence_ids": list(result.evidence),
        "evidence_text": {str(i): example.sentences[i] for i in result.evidence},
        "explanation": explanation,
        "dropped_ids": [str(i) for i in result.dropped_ids],
        "errors": list(result.errors),
        "raw_output": result.raw,
    }


def write_outputs(output, examples, verified, details):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    report = {**details, "metrics": full_report(examples, verified), "uncertainty": bootstrap_intervals(examples, verified)}
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(output / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for example, result in zip(examples, verified):
            handle.write(json.dumps(prediction_record(example, result), ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--baseline", action="store_true", help="majority label per criterion type, citing nothing")
    parser.add_argument("--split", required=True, choices=["validation", "test"])
    parser.add_argument("--final", action="store_true", help="required to read the test split")
    parser.add_argument("--order", default="label_first", choices=["label_first", "evidence_first"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not args.baseline and not args.model:
        parser.error("give --model or --baseline")

    check_split_access(args.split, args.final)
    details = {"split": args.split, "model": args.model, "adapter": args.adapter, "order": args.order, "baseline": args.baseline}
    if args.final:
        log_test_access(details)
    examples = load_split(args.split, allow_test=args.final)[: args.limit]

    if args.baseline:
        verified, majority = majority_baseline(load_split("train"), examples)
        details["baseline_labels"] = majority
    else:
        from screener.inference.common import load_for_inference

        model, tokenizer = load_for_inference(ROOT / args.model, ROOT / args.adapter if args.adapter else None)
        details["decoding"] = decoding_settings(model, tokenizer, args.max_new_tokens)
        verified = predict_examples(model, tokenizer, examples, args.order, args.batch_size, args.max_new_tokens)

    report = write_outputs(args.output, examples, verified, details)
    overall = report["metrics"]["classification"]["overall"]
    print(f"rows={overall['rows']} accuracy={overall['accuracy']:.4f} macro_f1={overall['macro_f1']:.4f} invalid={report['metrics']['invalid_rate']:.4f}")


if __name__ == "__main__":
    main()
