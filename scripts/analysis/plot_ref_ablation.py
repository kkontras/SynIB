#!/usr/bin/env python3
"""Figure 1 of the reference-distribution ablation: diagnostic KL curves.

One panel per dataset, one color per reference (fixed identity order), solid =
branch 1 (z2 masked, KL to p̂(·|x1)), dashed = branch 2 (symmetric). Curves are
fold-averaged validation-set values per validation step.

Usage: python scripts/analysis/plot_ref_ablation.py --curves <csv> --out docs/figures/fig_ref_ablation_diag.pdf
"""

import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical identity colors (validated: dataviz six-checks, light mode)
COLORS = {"uniform": "#2a78d6", "class_prior": "#eb6834", "unimodal_anchor": "#1baf7a"}
REF_ORDER = ["uniform", "class_prior", "unimodal_anchor"]
DS_LABEL = {"mustard": "MUStARD", "mosi": "MOSI", "urfunny": "UR-Funny",
            "hm": "Hateful Memes", "cremad": "CREMA-D-Irony (α=0.5)", "cremad_a01": "CREMA-D-Irony (α=0.1)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curves", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="val_diag_kl_1")
    ap.add_argument("--metric2", default="val_diag_kl_2")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.curves)))
    # (ds, ref, fold) -> [(step, kl1, kl2)]
    series = defaultdict(list)
    for r in rows:
        if not r.get(args.metric):
            continue
        series[(r["ds"], r["ref"], r["fold"])].append(
            (int(r["step"]), float(r[args.metric]), float(r[args.metric2] or "nan")))

    datasets = [d for d in ("mustard", "mosi", "urfunny", "hm", "cremad", "cremad_a01")
                if any(k[0] == d for k in series)]
    if not datasets:
        raise SystemExit("no data")

    fig, axes = plt.subplots(1, len(datasets), figsize=(3.1 * len(datasets), 2.8), sharey=False)
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        for ref in REF_ORDER:
            folds = [k for k in series if k[0] == ds and k[1] == ref]
            if not folds:
                continue
            # fold-average on the common step grid (truncate to shortest fold)
            per_fold = [sorted(series[k]) for k in folds]
            n = min(len(p) for p in per_fold)
            steps = [p[0] for p in per_fold[0][:n]]
            kl1 = np.mean([[v[1] for v in p[:n]] for p in per_fold], axis=0)
            kl2 = np.mean([[v[2] for v in p[:n]] for p in per_fold], axis=0)
            xs = np.arange(1, n + 1)
            ax.plot(xs, kl1, color=COLORS[ref], lw=1.8, label=ref.replace("_", " "))
            ax.plot(xs, kl2, color=COLORS[ref], lw=1.4, ls="--", alpha=0.85)
        ax.set_title(DS_LABEL.get(ds, ds), fontsize=10)
        ax.set_xlabel("validation step")
        ax.grid(alpha=0.25, lw=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(r"$D_{KL}\!\left(q(\cdot|\tilde x_i, x_{-i})\,\|\,\hat p(\cdot|x_{-i})\right)$")
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        for ax in axes[1:]:
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                break
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, 1.06))
    fig.text(0.99, 0.01, "solid: modality-2 masked   dashed: modality-1 masked",
             ha="right", fontsize=7, color="#666666")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
