"""Compare ordinary cached greedy decoding with speculative decoding on the same prompts.

Scenarios cover the real short JSON answers on validation prompts, and longer free generations.
Token equality with ordinary greedy decoding is checked for every prompt and draft length.

Usage:
  python benchmarks/bench_speculative.py --target models/qwen2.5-1.5b --draft models/qwen2.5-0.5b \
      --label kaggle-t4 [--target-adapter runs/R2/adapter --draft-adapter runs/R6/adapter]
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.data.prepare import load_split, render_prompt  # noqa: E402
from screener.eval.predict import eos_token_ids  # noqa: E402
from screener.inference.cached_decoder import greedy_cached  # noqa: E402
from screener.inference.common import benchmark_prompt, clock, load_for_inference  # noqa: E402
from screener.inference.speculative import greedy_speculative  # noqa: E402


def scenarios(tokenizer, device, target, examples):
    eos = eos_token_ids(target, tokenizer)
    screening = [tokenizer(render_prompt(tokenizer, e), add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
                 for e in load_split("validation")[:examples]]
    make = lambda length: [torch.tensor([benchmark_prompt(tokenizer, length)], device=device)]
    return [
        {"name": "screening_json", "prompts": screening, "max_new_tokens": 96, "eos": eos},
        {"name": "long_256in_128out", "prompts": make(256), "max_new_tokens": 128, "eos": []},
        {"name": "long_1024in_128out", "prompts": make(1024), "max_new_tokens": 128, "eos": []},
    ]


def run_pass(decode, prompts, max_new_tokens, eos, device, draft_device):
    """Run every prompt once, returning per-prompt tokens and summed measurements."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    second_gpu = draft_device.type == "cuda" and draft_device != device
    if second_gpu:
        torch.cuda.reset_peak_memory_stats(draft_device)
    base = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    start = clock(device)
    results = [decode(prompt, max_new_tokens, eos) for prompt in prompts]
    total = clock(device) - start
    sum_of = lambda name: sum(getattr(r, name) for r in results)
    measured = {
        "total_s": total,
        "tokens": sum(len(r.tokens) for r in results),
        "peak_above_models_gb": (torch.cuda.max_memory_allocated(device) - base) / 1024**3 if device.type == "cuda" else 0.0,
        "prefill_s": sum_of("prefill_seconds"),
        "decode_s": sum_of("decode_seconds"),
    }
    if second_gpu:
        measured["draft_gpu_peak_gb"] = torch.cuda.max_memory_allocated(draft_device) / 1024**3
    if hasattr(results[0], "rounds"):
        for name in ("rounds", "drafted", "accepted", "corrections", "bonus_tokens", "target_forwards", "draft_forwards"):
            measured[name] = sum_of(name)
        for name in ("draft_seconds", "verify_seconds", "sync_seconds"):
            measured[name.replace("_seconds", "_s")] = sum_of(name)
    return [r.tokens for r in results], measured


def summarize(passes):
    summary = {}
    for key in passes[0]:
        values = [p[key] for p in passes]
        summary[key] = statistics.median(values)
        if key.endswith("_s"):
            summary[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    summary["tokens_per_s"] = summary["tokens"] / summary["total_s"]
    if "drafted" in summary:
        summary["acceptance_rate"] = summary["accepted"] / summary["drafted"] if summary["drafted"] else 0.0
        summary["tokens_per_round"] = summary["tokens"] / summary["rounds"] if summary["rounds"] else 0.0
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--target-adapter")
    parser.add_argument("--draft-adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--draft-device", default=None, help="defaults to the target's device")
    parser.add_argument("--draft-lengths", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    draft_device = torch.device(args.draft_device) if args.draft_device else device
    draft_dtype = torch.float16 if draft_device.type == "cuda" else torch.float32
    target, tokenizer = load_for_inference(ROOT / args.target, ROOT / args.target_adapter if args.target_adapter else None, device, torch.float16)
    draft, draft_tokenizer = load_for_inference(ROOT / args.draft, ROOT / args.draft_adapter if args.draft_adapter else None, draft_device, draft_dtype)
    if tokenizer.get_vocab() != draft_tokenizer.get_vocab():
        raise ValueError("draft and target tokenizers differ")

    decoders = {"greedy": lambda p, n, e: greedy_cached(target, p, max_new_tokens=n, eos_ids=e)}
    for length in args.draft_lengths:
        decoders[f"speculative_k{length}"] = (
            lambda p, n, e, k=length: greedy_speculative(target, draft, p, max_new_tokens=n, eos_ids=e, draft_length=k)
        )

    results = []
    output = ROOT / "results" / f"bench_speculative_{args.label}.json"

    def save_report():
        report = {
            "label": args.label, "target": args.target, "target_adapter": args.target_adapter, "draft": args.draft,
            "draft_adapter": args.draft_adapter, "target_device": str(device), "draft_device": str(draft_device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu", "repeats": args.repeats,
            "warm_up_passes_discarded": 2, "torch": torch.__version__, "results": results,
        }
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for scenario in scenarios(tokenizer, device, target, args.examples):
        name, prompts, limit, eos = scenario["name"], scenario["prompts"], scenario["max_new_tokens"], scenario["eos"]
        run = lambda decode: run_pass(decode, prompts, limit, eos, device, draft_device)
        try:
            expected, _ = run(decoders["greedy"])
            equality = {}
            for mode, decode in decoders.items():
                tokens, _ = run(decode)
                equality[mode] = tokens == expected
            passes = {mode: [] for mode in decoders}
            for _ in range(args.repeats):
                for mode, decode in decoders.items():
                    passes[mode].append(run(decode)[1])
        except torch.cuda.OutOfMemoryError as error:
            # Keep earlier scenarios: an out-of-memory result is still a finding worth reporting.
            torch.cuda.empty_cache()
            results.append({"scenario": name, "failed": "GPU out of memory", "detail": str(error)[:300]})
            save_report()
            print(f"{name}: GPU out of memory", flush=True)
            continue
        modes = {mode: summarize(values) for mode, values in passes.items()}
        baseline = modes["greedy"]["total_s"]
        for summary in modes.values():
            summary["speedup_vs_greedy"] = baseline / summary["total_s"]
        results.append({"scenario": name, "prompts": len(prompts), "max_new_tokens": limit, "stops_at_eos": bool(eos),
                        "tokens_equal_to_greedy": equality, "modes": modes})
        save_report()
        print(f"{name}: all equal={all(equality.values())}", flush=True)
        for mode, summary in modes.items():
            acceptance = f" accept={summary['acceptance_rate']:.3f}" if "acceptance_rate" in summary else ""
            print(f"  {mode}: {summary['total_s']:.3f}s {summary['tokens_per_s']:.1f} tok/s speedup={summary['speedup_vs_greedy']:.2f}x{acceptance}", flush=True)

    save_report()
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
