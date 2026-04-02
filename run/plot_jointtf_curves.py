#!/usr/bin/env python3
"""
Plot train/val/test accuracy curves for the best lr-wd combo of JointTF and JointTF_IHA
across all 4 FactorCL datasets and 3 folds.

Layout: 4 rows (datasets) × 6 cols (fold0..2 JointTF, fold0..2 JointTF_IHA)
"""

import re
import math
import sys
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Dataset metadata ──────────────────────────────────────────────────────────
DATASETS = {
    "mustard": {
        "title":    "MUStARD",
        "ckpt_dir": REPO_ROOT / "artifacts/models/factorcl/mustard/vt",
        "log_dir_tf":  REPO_ROOT / "run/condor/logs/mustard_joint_tf",
        "log_dir_iha": REPO_ROOT / "run/condor/logs/mustard_joint_tf_iha",
    },
    "mosi": {
        "title":    "MOSI",
        "ckpt_dir": REPO_ROOT / "artifacts/models/factorcl/mosi/vt",
        "log_dir_tf":  REPO_ROOT / "run/condor/logs/mosi_joint_tf",
        "log_dir_iha": REPO_ROOT / "run/condor/logs/mosi_joint_tf_iha",
    },
    "mosei": {
        "title":    "MOSEI",
        "ckpt_dir": REPO_ROOT / "artifacts/models/factorcl/mosei/vt",
        "log_dir_tf":  REPO_ROOT / "run/condor/logs/mosei_joint_tf",
        "log_dir_iha": REPO_ROOT / "run/condor/logs/mosei_joint_tf_iha",
    },
    "urfunny": {
        "title":    "URFunny",
        "ckpt_dir": REPO_ROOT / "artifacts/models/factorcl/ur_funny/vt",
        "log_dir_tf":  REPO_ROOT / "run/condor/logs/urfunny_joint_tf",
        "log_dir_iha": REPO_ROOT / "run/condor/logs/urfunny_joint_tf_iha",
    },
}
DATASET_ORDER = ["mustard", "mosi", "mosei", "urfunny"]
N_FOLDS = 3

# ── Log parsing (same logic as show_jointtf_hparam_table.sh) ─────────────────
ANSI_RE       = re.compile(r"\x1b\[[0-9;]*m")
SAVE_DIR_RE   = re.compile(r"save_dir:\s*(\S+)")
FOLD_RE       = re.compile(r"_fold(\d+)_")
LR_RE         = re.compile(r"_lr([0-9.e+-]+)_wd")
WD_RE         = re.compile(r"_wd([0-9.e+-]+)_bs")
FINAL_VAL_RE  = re.compile(r"Val Epoch.*?step 0.*?Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
FINAL_TEST_RE = re.compile(r"Test Epoch.*?step 0.*?Acc_combined:\s*([0-9]+(?:\.[0-9]+)?)")
MODEL_PATH_RE = re.compile(r"Model saved successfully at:\s*(\S+)")


def parse_log(path):
    text = ANSI_RE.sub("", path.read_text(errors="ignore"))
    if "Job finished" not in text and "We are in the final state." not in text:
        return None
    m = SAVE_DIR_RE.search(text)
    if not m:
        return None
    sd = m.group(1)
    fold_m = FOLD_RE.search(sd)
    lr_m   = LR_RE.search(sd)
    wd_m   = WD_RE.search(sd)
    if not (fold_m and lr_m and wd_m):
        return None
    val_matches  = FINAL_VAL_RE.findall(text)
    test_matches = FINAL_TEST_RE.findall(text)
    if not val_matches or not test_matches:
        return None
    model_m = MODEL_PATH_RE.search(text)
    return {
        "fold":       int(fold_m.group(1)),
        "lr":         lr_m.group(1),
        "wd":         wd_m.group(1),
        "val":        float(val_matches[-1]),
        "test":       float(test_matches[-1]),
        "model_path": model_m.group(1) if model_m else None,
        "mtime":      path.stat().st_mtime,
    }


def collect_runs(log_dir):
    """Returns dict keyed by (lr, wd, fold) → record."""
    records = {}
    if not log_dir or not log_dir.exists():
        return records
    for path in log_dir.glob("*.out"):
        rec = parse_log(path)
        if rec is None:
            continue
        key = (rec["lr"], rec["wd"], rec["fold"])
        if key not in records or rec["mtime"] > records[key]["mtime"]:
            records[key] = rec
    return records


def best_lr_wd(records, n_folds=N_FOLDS):
    """Return (lr, wd) with highest avg val accuracy across all complete folds."""
    combos = sorted({(r["lr"], r["wd"]) for r in records.values()},
                    key=lambda x: (float(x[0]), float(x[1])))
    best_combo, best_avg = None, -1.0
    for lr, wd in combos:
        vals = [records[(lr, wd, f)]["val"] for f in range(n_folds)
                if (lr, wd, f) in records]
        if len(vals) < n_folds:
            continue
        avg = sum(vals) / len(vals)
        if avg > best_avg:
            best_avg, best_combo = avg, (lr, wd)
    return best_combo  # may be None if no complete set


# ── Checkpoint loading ────────────────────────────────────────────────────────
def resolve_ckpt(model_path_str, ckpt_dir):
    """Resolve checkpoint path (may be relative to repo root)."""
    if model_path_str is None:
        return None
    p = Path(model_path_str)
    if p.is_absolute() and p.exists():
        return p
    # relative to repo root
    candidate = REPO_ROOT / p
    if candidate.exists():
        return candidate
    # try just the filename in ckpt_dir
    candidate2 = ckpt_dir / p.name
    if candidate2.exists():
        return candidate2
    return None


def load_curves(ckpt_path):
    """
    Returns dict with keys 'steps', 'train', 'val', 'test'
    Each is a list of (step, acc*100) pairs sorted by step.
    """
    if ckpt_path is None or not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  Warning: could not load {ckpt_path}: {e}", file=sys.stderr)
        return None

    logs = ckpt.get("logs", {})
    best_step = logs.get("best_logs", {}).get("best_vaccuracy", {}).get("step")

    def extract(log_dict):
        pairs = []
        for step, entry in log_dict.items():
            acc = entry.get("acc", {}).get("combined")
            if acc is not None:
                pairs.append((int(step), float(acc) * 100))
        return sorted(pairs)

    train_pairs = extract(logs.get("train_logs", {}))
    val_pairs   = extract(logs.get("val_logs",   {}))
    test_pairs  = extract(logs.get("test_logs",  {}))

    return {
        "train":     train_pairs,
        "val":       val_pairs,
        "test":      test_pairs,
        "best_step": best_step,
    }


# ── Gather everything ─────────────────────────────────────────────────────────
print("Collecting runs and loading checkpoints …")
data = {}  # ds_key → { "tf": {fold: curves}, "iha": {fold: curves}, "best_tf": (lr,wd), "best_iha": (lr,wd) }

for ds_key in DATASET_ORDER:
    meta = DATASETS[ds_key]
    tf_records  = collect_runs(meta["log_dir_tf"])
    iha_records = collect_runs(meta["log_dir_iha"])

    best_tf  = best_lr_wd(tf_records)
    best_iha = best_lr_wd(iha_records)

    print(f"  {meta['title']:10s}  JointTF best={best_tf}  JointTF_IHA best={best_iha}")

    tf_curves, iha_curves = {}, {}

    for fold in range(N_FOLDS):
        # JointTF
        if best_tf and (best_tf[0], best_tf[1], fold) in tf_records:
            rec = tf_records[(best_tf[0], best_tf[1], fold)]
            ckpt = resolve_ckpt(rec["model_path"], meta["ckpt_dir"])
            tf_curves[fold] = load_curves(ckpt)
        else:
            tf_curves[fold] = None

        # JointTF_IHA
        if best_iha and (best_iha[0], best_iha[1], fold) in iha_records:
            rec = iha_records[(best_iha[0], best_iha[1], fold)]
            ckpt = resolve_ckpt(rec["model_path"], meta["ckpt_dir"])
            iha_curves[fold] = load_curves(ckpt)
        else:
            iha_curves[fold] = None

    data[ds_key] = {
        "title":    meta["title"],
        "best_tf":  best_tf,
        "best_iha": best_iha,
        "tf":       tf_curves,
        "iha":      iha_curves,
    }

# ── Plot ──────────────────────────────────────────────────────────────────────
print("Plotting …")

COLORS = {
    "train": "#4C9BE8",   # blue
    "val":   "#F5A623",   # orange
    "test":  "#7ED321",   # green
}
LINEWIDTH = 1.8
ALPHA_FILL = 0.08

fig, axes = plt.subplots(
    nrows=4, ncols=6,
    figsize=(34, 22),
    constrained_layout=False,
)
fig.subplots_adjust(
    left=0.06, right=0.99,
    top=0.91, bottom=0.07,
    wspace=0.32, hspace=0.42,
)

fig.suptitle("JointTF vs JointTF_IHA — Training Curves (best lr/wd per method)",
             fontsize=17, fontweight="bold", y=0.975)

# Method group banners above the two column groups
for spine_kw, label, xc, color in [
    (dict(), "JointTF",     0.275, "#2C6FAC"),
    (dict(), "JointTF_IHA", 0.735, "#8B3A8B"),
]:
    fig.text(xc, 0.945, label, ha="center", va="center",
             fontsize=14, fontweight="bold", color=color,
             bbox=dict(boxstyle="round,pad=0.35", fc=color+"18",
                       ec=color, linewidth=1.5))


def plot_curves(ax, curves, best_step=None, fold=None, method_color=None):
    if curves is None:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="grey")
        ax.set_axis_off()
        return

    for split in ("train", "val", "test"):
        pairs = curves[split]
        if not pairs:
            continue
        steps, accs = zip(*pairs)
        ax.plot(steps, accs, color=COLORS[split], linewidth=LINEWIDTH,
                label=split.capitalize(), zorder=3)

    # Mark best validation step
    bs = curves.get("best_step")
    if bs is not None:
        val_dict = dict(curves["val"])
        if bs in val_dict:
            ax.axvline(bs, color="grey", linewidth=1.0, linestyle="--",
                       alpha=0.7, zorder=2)
            ax.plot(bs, val_dict[bs], marker="*", markersize=9,
                    color=COLORS["val"], zorder=4)

    # Border colour by method
    for spine in ax.spines.values():
        spine.set_edgecolor(method_color or "black")
        spine.set_linewidth(1.4)

    ax.set_xlabel("Step", fontsize=8, labelpad=2)
    ax.set_ylabel("Accuracy (%)", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7.5)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


legend_handles = None

for row_idx, ds_key in enumerate(DATASET_ORDER):
    ds = data[ds_key]
    ax_row = axes[row_idx]

    lr_tf  = ds["best_tf"][0]  if ds["best_tf"]  else "?"
    wd_tf  = ds["best_tf"][1]  if ds["best_tf"]  else "?"
    lr_iha = ds["best_iha"][0] if ds["best_iha"] else "?"
    wd_iha = ds["best_iha"][1] if ds["best_iha"] else "?"

    for fold in range(N_FOLDS):
        # ── JointTF  → columns 0-2 ────────────────────────────────────────────
        ax = ax_row[fold]
        plot_curves(ax, ds["tf"][fold], fold=fold, method_color="#2C6FAC")

        # Fold title only on row 0 to avoid repetition
        if row_idx == 0:
            ax.set_title(f"Fold {fold}", fontsize=11, pad=4, color="#2C6FAC")

        # Dataset name on leftmost column
        if fold == 0:
            ax.set_ylabel(f"{ds['title']}\nAccuracy (%)",
                          fontsize=10, fontweight="bold", labelpad=6)
            # lr/wd annotation
            ax.annotate(f"lr={lr_tf}\nwd={wd_tf}", xy=(0.98, 0.03),
                        xycoords="axes fraction", fontsize=7, ha="right", va="bottom",
                        color="#2C6FAC",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec="#2C6FAC", alpha=0.85, linewidth=0.8))

        # ── JointTF_IHA  → columns 3-5 ───────────────────────────────────────
        ax2 = ax_row[fold + 3]
        plot_curves(ax2, ds["iha"][fold], fold=fold, method_color="#8B3A8B")

        if row_idx == 0:
            ax2.set_title(f"Fold {fold}", fontsize=11, pad=4, color="#8B3A8B")

        if fold == 0:
            ax2.annotate(f"lr={lr_iha}\nwd={wd_iha}", xy=(0.98, 0.03),
                         xycoords="axes fraction", fontsize=7, ha="right", va="bottom",
                         color="#8B3A8B",
                         bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                   ec="#8B3A8B", alpha=0.85, linewidth=0.8))

        if legend_handles is None and ds["tf"][fold] is not None:
            legend_handles = ax.get_legend_handles_labels()

# Shared legend at bottom
if legend_handles:
    fig.legend(*legend_handles, loc="lower center", ncol=3,
               fontsize=12, framealpha=0.95,
               bbox_to_anchor=(0.5, 0.005),
               handlelength=2.5, handleheight=1.2)

# Vertical separator between the two method groups
sep_x = 0.505
fig.add_artist(
    plt.Line2D([sep_x, sep_x], [0.07, 0.93],
               transform=fig.transFigure,
               color="#555555", linewidth=1.5, linestyle="--", alpha=0.5,
               zorder=10)
)

out_path = REPO_ROOT / "run" / "jointtf_curves.png"
fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
print(f"Saved → {out_path}")
plt.close(fig)
