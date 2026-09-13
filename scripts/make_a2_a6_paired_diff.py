#!/usr/bin/env python3
"""Regenerate the A2-vs-A6 paired-difference figure (paper Fig: a2_a6_paired_diff.pdf).

One row per seed; each pair is the mean held-out Dice of A2 (fixed Haar, Q=I)
and A6 (SO(8), Q learned) on the *same* 250 BraTS2021 volumes. The difference
changes sign on the third seed (+0.79, +0.85, -0.85), so a load-bearing learned
factor produces no stable Dice direction -- the figure behind the paper's
"non-removability is necessary but not sufficient for a gain" claim.

The three per-seed means are taken verbatim from the paper's main table
(seeds 123 / 456 / 789).  Edit `A2` / `A6` below to regenerate from a fresh run.

Usage:
    python scripts/make_a2_a6_paired_diff.py [--out figs/a2_a6_paired_diff.pdf]
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# mean held-out Dice per seed (123, 456, 789) -- paper Table 1
A2 = [0.9017, 0.9016, 0.8936]   # fixed Haar, Q = I, groups = 8
A6 = [0.8938, 0.8931, 0.9021]   # SO(8), Q learned,   groups = 8
SEEDS = ["123", "456", "789"]

A2_COL = "#185FA5"   # blue  (circles)
A6_COL = "#D85A30"   # coral (squares)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="figs/a2_a6_paired_diff.pdf",
                    help="output PDF path")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(3.35, 2.7))
    y_ticks, y_labels = [], []
    for i, (a2, a6, s) in enumerate(zip(A2, A6, SEEDS)):
        y = -i
        d = a2 - a6
        # a short connector between the two points of the pair
        ax.plot([a2, a6], [y, y], color="#888888", lw=0.8, zorder=1)
        ax.scatter(a2, y, s=34, marker="o", color=A2_COL, zorder=3,
                   label="A2 (fixed Haar)" if i == 0 else None)
        ax.scatter(a6, y, s=34, marker="s", color=A6_COL, zorder=3,
                   label="A6 (SO(8), learned)" if i == 0 else None)
        # signed delta annotation in Dice POINTS (x100), matching the paper
        # Table 1 row "A2-A6" (+0.79 / +0.85 / -0.85).
        xmid = (a2 + a6) / 2
        ax.annotate(f"{d * 100:+.2f}", xy=(xmid, y), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.5, color="#333333")
        y_ticks.append(y)
        y_labels.append(f"seed {s}")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel("mean held-out Dice (TC/WT/ET)", fontsize=7)
    ax.set_xlim(0.88, 0.91)
    ax.set_xticks([0.88, 0.89, 0.90, 0.91])
    ax.tick_params(axis="x", labelsize=6.5)
    ax.set_title("Task utility where a generic Q is not removable\n"
                 "A2 vs A6, same 250 volumes", fontsize=7.5)
    ax.legend(fontsize=6, loc="lower right", framealpha=0.9)
    ax.grid(axis="x", ls=":", lw=0.4, color="#cccccc")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"[paired] wrote {args.out}  (A2-A6 per seed, Dice points: "
          f"{[round((a-b)*100, 2) for a, b in zip(A2, A6)]})")


if __name__ == "__main__":
    main()
