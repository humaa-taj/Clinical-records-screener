"""Run the whole validation-only experiment list and choose the final configuration.

R1-R3 compare adapter size and rebalancing. R4 repeats the best with evidence listed first.
R5 and R6 apply the overall best to the competing model and the draft model.
Finished runs are skipped, so an interrupted session can continue.

Usage: python -m screener.train.run_experiments --output runs
"""
import argparse
import csv
import json
from pathlib import Path

from screener.eval.metrics import selection_key
from screener.train.train_lora import DEFAULT_OUTPUT, resolve_config, run_training

SMOLLM = "models/smollm2-1.7b"
DRAFT = "models/qwen2.5-0.5b"
TUNABLE_KEYS = ("oversample_power", "lora_rank", "lora_alpha", "order")


def run_once(name, config, output, results):
    existing = Path(output) / name / "result.json"
    if existing.exists():
        results[name] = json.loads(existing.read_text(encoding="utf-8"))
        print(f"{name}: already finished, skipping", flush=True)
    else:
        print(f"=== {name} ===", flush=True)
        results[name] = run_training(name, config, output)


def best_of(results, names):
    return max(names, key=lambda name: selection_key(results[name]["metrics"]))


def tuned_settings(results, name):
    return {key: results[name]["config"][key] for key in TUNABLE_KEYS}


def write_summary(output, results, chosen):
    rows = []
    for name, result in results.items():
        metrics = result["metrics"]
        rows.append({
            "run": name, "model": result["config"]["model"], "trained": result["config"]["train"],
            "lora_rank": result["config"]["lora_rank"], "oversample_power": result["config"]["oversample_power"],
            "order": result["config"]["order"], "best_epoch": result["best_epoch"],
            "accuracy": round(metrics["accuracy"], 4), "macro_f1": round(metrics["macro_f1"], 4),
            "invalid_rate": round(metrics["invalid_rate"], 4), "evidence_precision": round(metrics["evidence_precision"], 4),
            "evidence_recall": round(metrics["evidence_recall"], 4), "evidence_f1": round(metrics["evidence_f1"], 4),
            "seconds": round(result["elapsed_seconds"], 1), "peak_gpu_gb": result["peak_gpu_gb"],
        })
    with open(Path(output) / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (Path(output) / "selection.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")


def run_all(output):
    Path(output).mkdir(parents=True, exist_ok=True)
    results = {}
    for name in ("R0", "R0b", "R1", "R2", "R3"):
        run_once(name, resolve_config(name), output, results)

    best_tuning = tuned_settings(results, best_of(results, ("R1", "R2", "R3")))
    run_once("R4", resolve_config("R1", {**best_tuning, "order": "evidence_first"}), output, results)

    final = best_of(results, ("R1", "R2", "R3", "R4"))
    settings = tuned_settings(results, final)
    run_once("R5", resolve_config("R1", {**settings, "model": SMOLLM}), output, results)
    run_once("R6", resolve_config("R1", {**settings, "model": DRAFT}), output, results)

    chosen = {
        "final_target_run": final,
        "draft_run": "R6",
        "selection_rule": "highest validation macro-F1, then evidence F1, among Qwen2.5-1.5B runs R1-R4",
        "settings": settings,
    }
    write_summary(output, results, chosen)
    print(json.dumps(chosen, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    run_all(parser.parse_args().output)


if __name__ == "__main__":
    main()
