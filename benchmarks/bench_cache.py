"""Check and time the cached decoder against the uncached reference, under greedy decoding.

Each input and output length pair gets a token-equality and logit-difference check, a discarded
warm-up, then repeated runs that alternate modes so slow drift affects both alike.

Usage: python benchmarks/bench_cache.py --model models/qwen2.5-1.5b --label kaggle-t4 [--adapter runs/R2/adapter]
"""
import argparse
import json
import platform
import statistics
import sys
from pathlib import Path

import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.inference.cached_decoder import greedy_cached  # noqa: E402
from screener.inference.common import benchmark_prompt, load_for_inference  # noqa: E402
from screener.inference.uncached_decoder import greedy_uncached  # noqa: E402

DECODERS = {"uncached": greedy_uncached, "cached": greedy_cached}


def timed_run(decoder, model, input_ids, max_new_tokens, device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    base_gpu = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    result = decoder(model, input_ids, max_new_tokens=max_new_tokens)
    peak_gpu = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return {
        "prefill_s": result.prefill_seconds,
        "decode_s": result.decode_seconds,
        "total_s": result.total_seconds,
        "decode_tokens_per_s": result.decode_tokens_per_second,
        "total_tokens_per_s": result.total_tokens_per_second,
        "peak_gpu_gb": peak_gpu / 1024**3,
        "peak_above_model_gb": (peak_gpu - base_gpu) / 1024**3,
        "process_rss_gb": psutil.Process().memory_info().rss / 1024**3,
    }


def summarize(runs):
    summary = {}
    for key in runs[0]:
        values = [run[key] for run in runs]
        summary[key] = {"median": statistics.median(values), "stdev": statistics.stdev(values) if len(values) > 1 else 0.0}
    return summary


def check_equivalence(model, input_ids, max_new_tokens):
    reference = greedy_uncached(model, input_ids, max_new_tokens=max_new_tokens, record_logits=True)
    cached = greedy_cached(model, input_ids, max_new_tokens=max_new_tokens, record_logits=True, track_cache=True)
    difference = (torch.stack(reference.step_logits) - torch.stack(cached.step_logits)).abs()
    expected = list(range(input_ids.shape[1], input_ids.shape[1] + max_new_tokens))
    first_mismatch = next((i for i, (a, b) in enumerate(zip(reference.tokens, cached.tokens)) if a != b), None)
    return {
        "tokens_equal": reference.tokens == cached.tokens,
        "first_mismatch_step": first_mismatch,
        "max_abs_logit_difference": difference.max().item(),
        "mean_abs_logit_difference": difference.mean().item(),
        "cache_length_correct_each_step": cached.cache_lengths == expected,
        "cache_mb_after_prefill": cached.cache_bytes[0] / 1024**2,
        "cache_mb_at_end": cached.cache_bytes[-1] / 1024**2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--label", required=True)
    parser.add_argument("--inputs", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--outputs", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = getattr(torch, args.dtype)
    model, tokenizer = load_for_inference(ROOT / args.model, ROOT / args.adapter if args.adapter else None, device, dtype)
    results = []
    for input_length in args.inputs:
        input_ids = torch.tensor([benchmark_prompt(tokenizer, input_length)], device=device)
        for output_length in args.outputs:
            equivalence = check_equivalence(model, input_ids, output_length)
            for decoder in DECODERS.values():
                timed_run(decoder, model, input_ids, output_length, device)
            runs = {name: [] for name in DECODERS}
            for _ in range(args.repeats):
                for name, decoder in DECODERS.items():
                    runs[name].append(timed_run(decoder, model, input_ids, output_length, device))
            row = {
                "input_tokens": input_length,
                "output_tokens": output_length,
                "equivalence": equivalence,
                "modes": {name: summarize(values) for name, values in runs.items()},
                "raw_runs": runs,
            }
            results.append(row)
            uncached = row["modes"]["uncached"]["total_s"]["median"]
            cached = row["modes"]["cached"]["total_s"]["median"]
            print(f"in={input_length} out={output_length} equal={equivalence['tokens_equal']} "
                  f"max_diff={equivalence['max_abs_logit_difference']:.5f} uncached={uncached:.3f}s cached={cached:.3f}s "
                  f"speedup={uncached / cached:.2f}x", flush=True)

    report = {
        "label": args.label,
        "model": args.model,
        "adapter": args.adapter,
        "dtype": args.dtype,
        "decoding": "greedy, EOS ignored so output lengths are exact",
        "repeats": args.repeats,
        "warm_up_runs_discarded_per_mode": 1,
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "results": results,
    }
    output = ROOT / "results" / f"bench_cache_{args.label}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}; all equal: {all(r['equivalence']['tokens_equal'] for r in results)}")


if __name__ == "__main__":
    main()
