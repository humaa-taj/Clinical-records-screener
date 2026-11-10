"""Draw the report figures from the committed results. Usage: python scripts/make_figures.py

Every number comes from a file in results/; nothing is recomputed or typed in by hand.
Figures go to report/figures/ as vector PDFs sized for an IEEE column (3.5 in) or page (7.16 in).
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "figures"

# Three categorical slots that pass colour-blind separation as a set, plus text and grid inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLUMN, PAGE = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4,
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def load(path):
    return json.loads((RESULTS / path).read_text(encoding="utf-8"))


def history(run):
    with open(RESULTS / "training" / run / "history.csv", encoding="utf-8") as handle:
        return [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print(f"wrote report/figures/{name}.pdf")


def training_curves():
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE, 2.1))
    r3 = history("R3")
    epochs = [row["epoch"] for row in r3]
    left.plot(epochs, [row["train_loss"] for row in r3], color=BLUE, marker="o", label="Training loss")
    left.plot(epochs, [row["val_loss"] for row in r3], color=ORANGE, marker="s", linestyle="--", label="Validation loss")
    left.set(title="(a) R3 loss per epoch", xlabel="Epoch", ylabel="Cross-entropy (answer tokens)")
    left.legend()
    for run, color, marker, style in (("R1", ORANGE, "s", "--"), ("R2", AQUA, "^", ":"), ("R3", BLUE, "o", "-")):
        rows = history(run)
        right.plot([row["epoch"] for row in rows], [row["macro_f1"] for row in rows], color=color, marker=marker, linestyle=style, label=run)
    right.set(title="(b) Validation macro-F1 from greedy generation", xlabel="Epoch", ylabel="Macro-F1")
    right.legend(loc="lower right")
    for axis in (left, right):
        axis.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    save(fig, "training_curves")


def before_after():
    systems = (("baseline_test", "Majority baseline", BLUE, "////"), ("before_test", "Before fine-tuning", ORANGE, ".."), ("after_test", "After fine-tuning (R3)", AQUA, ""))
    metrics = (("Accuracy", lambda m: m["classification"]["overall"]["accuracy"]), ("Macro-F1", lambda m: m["classification"]["overall"]["macro_f1"]), ("Evidence micro-F1", lambda m: m["evidence"]["overall"]["micro"]["f1"]))
    fig, axis = plt.subplots(figsize=(COLUMN, 2.0))
    x = np.arange(len(metrics))
    width = 0.26
    for offset, (folder, label, color, hatch) in enumerate(systems):
        data = load(f"eval/{folder}/metrics.json")["metrics"]
        values = [get(data) for _, get in metrics]
        bars = axis.bar(x + (offset - 1) * width, values, width - 0.03, color=color, edgecolor="white", linewidth=0.6, hatch=hatch, label=label)
        for bar, value in zip(bars, values):
            axis.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.2f}", ha="center", va="bottom", fontsize=6, color=INK)
    axis.set_xticks(x, [name for name, _ in metrics])
    axis.set(ylim=(0, 1), ylabel="Score on test split")
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=3, fontsize=6.3, handlelength=1.4, columnspacing=0.8)
    save(fig, "before_after")


def confusion():
    data = load("eval/after_test/metrics.json")["metrics"]["classification"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.4))
    short = {"included": "incl.", "not included": "not incl.", "excluded": "excl.", "not excluded": "not excl.", "not enough information": "NEI", "not applicable": "N/A", "__invalid__": "invalid"}
    for axis, group, title in ((axes[0], "inclusion", "(a) Inclusion criteria (75 rows)"), (axes[1], "exclusion", "(b) Exclusion criteria (131 rows)")):
        table = data[group]["confusion_matrix"]
        labels = [label for label in table["columns"] if label != "__invalid__"]
        matrix = np.array([[table["matrix"][gold][pred] for pred in labels] for gold in labels])
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
        for i in range(len(labels)):
            for j in range(len(labels)):
                axis.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7, color="white" if matrix[i, j] > matrix.max() * 0.55 else INK)
        axis.set_xticks(range(len(labels)), [short[label] for label in labels])
        axis.set_yticks(range(len(labels)), [short[label] for label in labels])
        axis.set(title=title, xlabel="Predicted", ylabel="Expert (gold)")
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    save(fig, "confusion")


def kv_cache():
    data = load("bench_cache_kaggle-t4-finetuned.json")["results"]
    rows = [row for row in data if row["output_tokens"] == 64]
    lengths = [row["input_tokens"] for row in rows]
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE, 2.0))
    for mode, label, color, marker, style in (("uncached", "No cache (reference)", ORANGE, "s", "--"), ("cached", "KV cache", BLUE, "o", "-")):
        left.plot(lengths, [row["modes"][mode]["decode_tokens_per_s"]["median"] for row in rows], color=color, marker=marker, linestyle=style, label=label)
    left.set(title="(a) Decode throughput, 64 output tokens", xlabel="Input tokens", ylabel="Tokens per second", xscale="log", ylim=(0, 30))
    left.set_xticks(lengths, [str(value) for value in lengths])
    left.minorticks_off()
    left.legend()
    for output, color, marker, style in ((16, AQUA, "^", ":"), (64, BLUE, "o", "-"), (256, ORANGE, "s", "--")):
        subset = [row for row in data if row["output_tokens"] == output]
        right.plot([row["input_tokens"] for row in subset], [row["equivalence"]["cache_mb_at_end"] for row in subset], color=color, marker=marker, linestyle=style, label=f"{output} output tokens")
    right.set(title="(b) KV cache size after generation", xlabel="Input tokens", ylabel="Cache size (MB)", xscale="log")
    right.set_xticks(lengths, [str(value) for value in lengths])
    right.minorticks_off()
    right.legend()
    save(fig, "kv_cache")


def speculative():
    data = load("bench_speculative_kaggle-t4-finetuned-draft.json")["results"]
    names = {"screening_json": "Screening JSON", "long_256in_128out": "256 in / 128 out", "long_1024in_128out": "1024 in / 128 out"}
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE, 2.1))
    ks = [2, 4, 8]
    for (scenario, label), color, marker, style in zip(names.items(), (BLUE, ORANGE, AQUA), ("o", "s", "^"), ("-", "--", ":")):
        modes = next(row for row in data if row["scenario"] == scenario)["modes"]
        left.plot(ks, [modes[f"speculative_k{k}"]["speedup_vs_greedy"] for k in ks], color=color, marker=marker, linestyle=style, label=label)
        right.plot(ks, [modes[f"speculative_k{k}"]["accepted"] / modes[f"speculative_k{k}"]["drafted"] for k in ks], color=color, marker=marker, linestyle=style, label=label)
    left.axhline(1.0, color=MUTED, linewidth=0.9, linestyle="-")
    left.text(8, 1.01, "greedy = 1.0", ha="right", va="bottom", fontsize=6.5, color=MUTED)
    left.set(title="(a) Speed relative to greedy decoding", xlabel="Draft length k", ylabel="Speedup (x)", ylim=(0.4, 1.1))
    right.set(title="(b) Draft-token acceptance rate", xlabel="Draft length k", ylabel="Accepted / drafted", ylim=(0, 1.05))
    for axis in (left, right):
        axis.set_xticks(ks)
    left.legend(loc="lower left")
    save(fig, "speculative")


def attention():
    data = [row for row in load("bench_attention_kaggle-t4.json")["results"] if row["device"] == "cuda"]
    lengths = [row["length"] for row in data]
    fig, (left, right) = plt.subplots(1, 2, figsize=(PAGE, 2.0))
    series = (("reference", "Reference SDPA", ORANGE, "s", "--"), ("pytorch_fused", "PyTorch fused", AQUA, "^", ":"), ("tiled_256", "Tiled online softmax (256)", BLUE, "o", "-"))
    for mode, label, color, marker, style in series:
        left.plot(lengths, [row["modes"][mode]["median_s"] * 1000 for row in data], color=color, marker=marker, linestyle=style, label=label)
        right.plot(lengths, [row["modes"][mode]["peak_extra_gpu_mb"] for row in data], color=color, marker=marker, linestyle=style, label=label)
    left.set(title="(a) Runtime on Tesla T4, causal", xlabel="Sequence length", ylabel="Milliseconds", xscale="log", yscale="log")
    right.set(title="(b) Peak GPU memory above inputs", xlabel="Sequence length", ylabel="MB", xscale="log", yscale="log")
    for axis in (left, right):
        axis.set_xticks(lengths, [str(value) for value in lengths])
        axis.minorticks_off()
    left.legend(loc="upper left")
    save(fig, "attention")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    training_curves()
    before_after()
    confusion()
    kv_cache()
    speculative()
    attention()


if __name__ == "__main__":
    main()
