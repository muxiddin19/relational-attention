#!/usr/bin/env python3
"""
Cross-Task Accuracy Bar Chart.

Generates the grouped bar chart comparing Standard Transformer,
RelTransformer (125M), T5-Base, and T5+RelAttn across all five benchmarks.

Usage:
    python viz/cross_task_bar.py [--output viz/cross_task_bar.pdf]
                                  [--results-dir /nas/.../outputs]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Paper results (mean over 3 seeds) ────────────────────────────────────────
RESULTS = {
    "Std. Transformer": {
        "Spider EX": 62.3, "COGS": 35.0, "SCAN": 18.1, "CFQ": 37.4, "GSM8K": 18.2
    },
    "RelTransformer (125M)": {
        "Spider EX": 75.3, "COGS": 98.2, "SCAN": 99.8, "CFQ": 71.3, "GSM8K": 32.4
    },
    "T5-Base": {
        "Spider EX": None, "COGS": 81.0, "SCAN": 99.7, "CFQ": 61.6, "GSM8K": None
    },
    "T5-Base+RelAttn": {
        "Spider EX": None, "COGS": 96.4, "SCAN": 99.9, "CFQ": 74.8, "GSM8K": None
    },
}
COLORS = {
    "Std. Transformer":    "#aaaaaa",
    "RelTransformer (125M)": "#4472C4",
    "T5-Base":             "#ED7D31",
    "T5-Base+RelAttn":     "#C00000",
}
TASKS = ["Spider EX", "COGS", "SCAN", "CFQ", "GSM8K"]


def load_from_dir(results_dir: str) -> None:
    """Update RESULTS with actual eval JSON values if available."""
    for model_dir in Path(results_dir).iterdir():
        if not model_dir.is_dir():
            continue
        name = model_dir.name.lower()
        for task in TASKS:
            ev = model_dir / "eval_results.json"
            if ev.exists():
                try:
                    data = json.load(open(ev))
                    metrics = data.get("metrics", data)
                    acc = metrics.get("exact_match") or metrics.get("execution_accuracy")
                    if acc is None:
                        continue
                    if "relational" in name:
                        RESULTS["RelTransformer (125M)"][task] = round(acc * 100, 1)
                    elif "standard" in name:
                        RESULTS["Std. Transformer"][task] = round(acc * 100, 1)
                except Exception:
                    pass


def plot(output: str = "viz/cross_task_bar.pdf"):
    models = list(RESULTS.keys())
    n_tasks = len(TASKS)
    n_models = len(models)
    x = np.arange(n_tasks)
    w = 0.18
    offsets = np.linspace(-(n_models - 1) * w / 2, (n_models - 1) * w / 2, n_models)

    fig, ax = plt.subplots(figsize=(9.5, 4.8))

    for model, offset in zip(models, offsets):
        for j, (task, bx) in enumerate(zip(TASKS, x + offset)):
            v = RESULTS[model].get(task)
            if v is None:
                continue
            ax.bar(bx, v, w * 0.85, color=COLORS[model], alpha=0.87,
                   edgecolor="white", linewidth=0.4,
                   label=model if j == 0 else "")

    ax.set_xticks(x)
    ax.set_xticklabels(TASKS, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", linestyle=":", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    handles = [mpatches.Patch(color=COLORS[m], label=m) for m in models]
    ax.legend(handles=handles, fontsize=8.5, ncol=2,
              loc="upper left", framealpha=0.9, edgecolor="none")
    ax.set_title("Cross-Task Accuracy: Standard vs. RelTransformer vs. T5",
                 fontsize=12, pad=8)

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=200)
    print(f"Saved to {output}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default=None,
                   help="Optional: path to training outputs for live results")
    p.add_argument("--output", default="viz/cross_task_bar.pdf")
    args = p.parse_args()

    if args.results_dir:
        load_from_dir(args.results_dir)

    plot(args.output)


if __name__ == "__main__":
    main()
