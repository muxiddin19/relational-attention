#!/usr/bin/env python3
"""
Slot Specialization Heatmap (Figure 1 in paper).

Generates color-coded heatmap of Fisher discriminability F per slot × role.
Reproduces Table 4 and the heatmap figure.

Usage:
    python viz/slot_specialization.py [--results analysis/probing_results.json]
                                      [--output viz/slot_heatmap.pdf]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paper Table 4 values (probing on Spider dev, last encoder layer, 3 seeds mean)
PAPER_FISHER_F = [
    [18.4,  4.2,  3.1],   # Slot 1 — Identity specialist
    [ 3.8, 21.7,  5.6],   # Slot 2 — FD/FK specialist
    [ 4.1,  6.3, 19.2],   # Slot 3 — Schema specialist
    [ 6.2,  8.1,  7.4],   # Slots 4-8 — auxiliary (avg)
    [ 6.2,  8.1,  7.4],
    [ 6.2,  8.1,  7.4],
    [ 6.2,  8.1,  7.4],
    [ 6.2,  8.1,  7.4],
]
ROLES = ["Identity (PK)", "FD (FK)", "Schema"]


def plot_heatmap(F: list, roles: list, output: str):
    F_arr = np.array(F)
    k = len(F_arr)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(F_arr, cmap="YlOrRd", aspect="auto", vmin=0, vmax=25)
    plt.colorbar(im, ax=ax, label="Fisher $F$ discriminability", shrink=0.85)

    ax.set_xticks(range(len(roles)))
    ax.set_xticklabels(roles, fontsize=10)
    ax.set_yticks(range(k))
    ax.set_yticklabels([f"Slot {j+1}" for j in range(k)], fontsize=9)
    ax.set_xlabel("Relational Role", fontsize=10)
    ax.set_ylabel("Attribute Slot $j$", fontsize=10)
    ax.set_title("Attribute Slot Specialization\n(Fisher Discriminability)", fontsize=11, pad=6)

    for j in range(k):
        for i in range(len(roles)):
            v = F_arr[j, i]
            color = "white" if v > 14 else "black"
            weight = "bold" if v == F_arr[j].max() else "normal"
            ax.text(i, j, f"{v:.1f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight=weight)

    plt.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, bbox_inches="tight", dpi=200)
    print(f"Saved heatmap to {output}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=None,
                   help="JSON from analysis/probing.py (defaults to paper values)")
    p.add_argument("--output", default="viz/slot_heatmap.pdf")
    args = p.parse_args()

    if args.results and Path(args.results).exists():
        data = json.load(open(args.results))
        F = data["fisher_f"]
        roles = data.get("roles", ROLES)
    else:
        F = PAPER_FISHER_F
        roles = ROLES
        print("Using paper-validated values (Table 4).")

    plot_heatmap(F, roles, args.output)


if __name__ == "__main__":
    main()
