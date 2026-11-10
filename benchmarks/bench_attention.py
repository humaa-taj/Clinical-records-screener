"""Compare reference, tiled online-softmax and PyTorch's fused attention.

Reports numerical error against the reference, median runtime, and GPU peak memory. CPU allocations
are not tracked by PyTorch, so the size of the largest live score matrix is reported for both devices.

Usage: python benchmarks/bench_attention.py --label kaggle-t4
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from screener.attention.reference import reference_attention  # noqa: E402
from screener.attention.tiled_online import TileStats, tiled_attention  # noqa: E402
from screener.inference.common import clock  # noqa: E402

HEADS = 8
HEAD_DIM = 64


def inputs(length, device, dtype):
    generator = torch.Generator().manual_seed(length)
    shape = (1, HEADS, length, HEAD_DIM)
    tensors = [torch.randn(shape, generator=generator, dtype=dtype).to(device) for _ in range(3)]
    return (*tensors, torch.ones(length, length, dtype=torch.bool, device=device).tril()[None, None])


def timed(function, device, repeats):
    function()
    seconds, peaks = [], []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
        start = clock(device)
        output = function()
        seconds.append(clock(device) - start)
        peaks.append((torch.cuda.max_memory_allocated(device) - base) / 1024**2 if device.type == "cuda" else None)
        del output
    return {"median_s": statistics.median(seconds), "stdev_s": statistics.stdev(seconds) if repeats > 1 else 0.0,
            "peak_extra_gpu_mb": peaks[-1]}


def run_length(length, tiles, device, repeats, dtype):
    query, key, value, mask = inputs(length, device, dtype)
    element = torch.empty(0, dtype=dtype).element_size()
    with torch.no_grad():
        reference = reference_attention(query, key, value, mask)
        results = {
            "reference": {**timed(lambda: reference_attention(query, key, value, mask), device, repeats),
                          "largest_score_matrix_mb": HEADS * length**2 * element / 1024**2},
            "pytorch_fused": {**timed(lambda: F.scaled_dot_product_attention(query, key, value, attn_mask=mask), device, repeats),
                              "max_abs_error": (F.scaled_dot_product_attention(query, key, value, attn_mask=mask) - reference).abs().max().item()},
        }
        for tile in tiles:
            stats = TileStats()
            output = tiled_attention(query, key, value, mask, tile, tile, stats=stats)
            difference = (output - reference).abs()
            results[f"tiled_{tile}"] = {
                **timed(lambda: tiled_attention(query, key, value, mask, tile, tile), device, repeats),
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "tiles_computed": stats.computed,
                "tiles_skipped_by_causal_mask": stats.skipped,
                "largest_score_matrix_mb": stats.largest_tile_scores * element / 1024**2,
            }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--tiles", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    args = parser.parse_args()

    report = {"label": args.label, "dtype": "float32", "heads": HEADS, "head_dim": HEAD_DIM, "mask": "causal",
              "repeats": args.repeats, "warm_up_runs_discarded": 1, "torch": torch.__version__, "results": []}
    for name in args.devices:
        if name == "cuda" and not torch.cuda.is_available():
            continue
        device = torch.device(name)
        report[f"{name}_name"] = torch.cuda.get_device_name(0) if name == "cuda" else "cpu"
        for length in args.lengths:
            modes = run_length(length, args.tiles, device, args.repeats, torch.float32)
            report["results"].append({"device": name, "length": length, "modes": modes})
            line = " ".join(f"{mode}={values['median_s'] * 1000:.2f}ms" for mode, values in modes.items())
            errors = max(values.get("max_abs_error", 0.0) for mode, values in modes.items() if mode.startswith("tiled"))
            print(f"{name} n={length} {line} tiled_max_err={errors:.2e}", flush=True)

    output = ROOT / "results" / f"bench_attention_{args.label}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
