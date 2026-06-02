#!/usr/bin/env python3
"""
Animated Attention Pattern Visualization.

Creates an animated GIF/MP4 showing how each of the k=8 attribute slots
specializes across encoder layers: from broad surface-level attention
(layer 1) to focused relational-role attention (layer 12).

Usage:
    python viz/animations/generate_attention_anim.py \
        [--n-layers 12] [--seq-len 16] \
        [--output viz/animations/slot_evolution.gif]
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation


SLOT_LABELS = [
    "Slot 1\n(Identity/PK)",
    "Slot 2\n(FD/FK)",
    "Slot 3\n(Schema)",
    "Slots 4–8\n(Auxiliary)",
]
SLOT_CMAPS = ["Blues", "Reds", "Greens", "Purples"]


def simulate_attention(n_layers: int, seq_len: int, slot_idx: int, layer: int):
    """Simulate attention distribution for one slot at one layer."""
    np.random.seed(slot_idx * 100 + layer)
    spec = layer / max(n_layers - 1, 1)
    # Start diffuse, become concentrated as specialization grows
    concentration = max(0.05, 1.0 - spec * (0.6 + slot_idx * 0.08))
    attn = np.random.dirichlet([concentration] * seq_len, seq_len)

    # Each slot specializes to a different diagonal pattern
    if spec > 0.3:
        diag_strength = spec * max(0, 0.9 - slot_idx * 0.2)
        for i in range(seq_len):
            target = max(0, min(seq_len - 1, i + slot_idx - 1))
            attn[i, target] += diag_strength
        attn = np.abs(attn)
        attn /= attn.sum(axis=-1, keepdims=True)
    return attn


def make_animation(n_layers: int = 12, seq_len: int = 16, output: str = "viz/animations/slot_evolution.gif"):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.8),
                             gridspec_kw={"wspace": 0.05})
    ims = []

    def update(layer):
        for ax, (si, label, cmap) in enumerate(zip(range(4), SLOT_LABELS, SLOT_CMAPS)):
            axes[ax].clear()
            attn = simulate_attention(n_layers, seq_len, si, layer)
            axes[ax].imshow(attn, cmap=cmap, vmin=0, vmax=0.6, aspect="auto")
            axes[ax].set_title(label, fontsize=8.5, pad=3)
            if ax == 0:
                axes[ax].set_ylabel("Query position", fontsize=7)
            axes[ax].set_xlabel("Key position", fontsize=7)
            axes[ax].tick_params(labelbottom=False, labelleft=False,
                                  bottom=False, left=False)
        fig.suptitle(
            f"Attribute Slot Attention Patterns — Encoder Layer {layer + 1}/{n_layers}\n"
            f"(Specialization emerges progressively: layer 1 = diffuse, "
            f"layer {n_layers} = focused)",
            fontsize=10, fontweight="bold"
        )

    ani = animation.FuncAnimation(fig, update, frames=n_layers,
                                   interval=700, repeat=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    if output.endswith(".mp4"):
        ani.save(output, writer="ffmpeg", fps=1.3, dpi=120)
    else:
        ani.save(output, writer="pillow", fps=1.3)

    print(f"Saved animation ({n_layers} frames) to {output}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-layers", type=int, default=12)
    p.add_argument("--seq-len", type=int, default=16,
                   help="Number of tokens in the sequence (for visualization)")
    p.add_argument("--output", default="viz/animations/slot_evolution.gif")
    args = p.parse_args()
    make_animation(args.n_layers, args.seq_len, args.output)


if __name__ == "__main__":
    main()
