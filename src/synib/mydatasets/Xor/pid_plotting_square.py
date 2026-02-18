import json
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe

EPS = 1e-12

# -------------------- Square plotting helpers --------------------
def draw_square(ax, xlabel="Redundancy (pred)", ylabel="Synergy (psyn)", note="Unique1=0, Unique2=0"):
    # boundary
    ax.plot([-0.02, 1.02], [-0.02, -0.02], lw=1.6, color="black")
    ax.plot([1.02, 1.02], [-0.02, 1.02], lw=1.6, color="black")
    ax.plot([1.02, -0.02], [1.02, 1.02], lw=1.6, color="black")
    ax.plot([-0.02, -0.02], [1.02, -0.02], lw=1.6, color="black")

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel, fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.tick_params(labelsize=6)

    outline = [pe.withStroke(linewidth=2, foreground="white")]
    ax.text(
        0.02, 1.04, note,
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=6,
        path_effects=outline
    )


# -------------------- Data loading (same structure as your file) --------------------
def load_records_from_three(file_path_main, file_path_synib, file_path_syniblearned):
    def read(path):
        with open(path, "r") as f:
            return json.load(f)

    data_main = read(file_path_main)
    data_synib = read(file_path_synib)
    data_learned = read(file_path_syniblearned)

    # index by (pu1, pred, psyn)
    def build_index(data):
        idx = {}
        for v in data.get("results", {}).values():
            probs = v.get("probs", {})
            key = (
                float(probs.get("pu1", np.nan)),
                float(probs.get("pred", np.nan)),
                float(probs.get("psyn", np.nan)),
            )
            idx[key] = v
        return idx

    idx_main = build_index(data_main)
    idx_synib = build_index(data_synib)
    idx_learned = build_index(data_learned)

    all_keys = set(idx_main.keys()) | set(idx_synib.keys()) | set(idx_learned.keys())

    def safe_get(v, *path, default=np.nan):
        cur = v
        for p in path:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    records = []
    for (pu1, pred, psyn) in sorted(all_keys):
        vm = idx_main.get((pu1, pred, psyn), {})
        vs = idx_synib.get((pu1, pred, psyn), {})
        vl = idx_learned.get((pu1, pred, psyn), {})

        records.append({
            "u1": pu1,
            "red": pred,
            "syn": psyn,

            # main metrics
            "acc_fusion": safe_get(vm, "summary_meanstd", "main", "test_tot_mean"),
            "acc_syn":    safe_get(vm, "summary_meanstd", "main", "test_syn_mean"),
            "acc_red":    safe_get(vm, "summary_meanstd", "main", "test_red_mean"),  # <-- NEW

            # synib metrics
            "synib_acc_fusion": safe_get(vs, "summary_meanstd", "synib_tuned", "test_tot_mean"),
            "synib_acc_syn":    safe_get(vs, "summary_meanstd", "synib_tuned", "test_syn_mean"),
            "synib_acc_red":    safe_get(vs, "summary_meanstd", "synib_tuned", "test_red_mean"),  # <-- NEW

            # learned metrics
            "synib_learned_acc_fusion": safe_get(vl, "summary_meanstd", "synib_RM_tuned", "test_tot_mean"),
            "synib_learned_acc_syn":    safe_get(vl, "summary_meanstd", "synib_RM_tuned", "test_syn_mean"),
            "synib_learned_acc_red":    safe_get(vl, "summary_meanstd", "synib_RM_tuned", "test_red_mean"),  # <-- NEW
        })

    return pd.DataFrame(records)


# -------------------- Paths --------------------
file_path_main = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_main_snr3_v3_overlapping.json"
file_path_synib = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib_snr3_v3_overlap.json"
file_path_synibleaned = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib_random03_snr3_v3_overlap.json"

df = load_records_from_three(file_path_main, file_path_synib, file_path_synibleaned)

# -------------------- Square coordinates --------------------
x = df["red"].to_numpy(dtype=float)
y = df["syn"].to_numpy(dtype=float)
sizes = 120


# -------------------- Figure: 3x3 grid (NEW third row) --------------------
fig, axes = plt.subplots(3, 3, figsize=(12, 14))
title_fs = 9

col_titles = [
    "(a) No Regularization",
    r"(b) SynIB $M^*_s$",
    r"(c) SynIB $M_{Random}$",
]

def plot_row(row_idx, metric_a, metric_b, metric_c, row_title):
    global_min = float(np.nanmin([df[metric_a].min(), df[metric_b].min(), df[metric_c].min()]))
    global_max = 1.0
    norm = Normalize(vmin=global_min, vmax=global_max)

    draw_square(axes[row_idx, 0])
    sc = axes[row_idx, 0].scatter(x, y, c=df[metric_a], s=sizes, cmap="RdPu", norm=norm,
                                  edgecolors="gray", linewidths=0.5)

    draw_square(axes[row_idx, 1])
    axes[row_idx, 1].scatter(x, y, c=df[metric_b], s=sizes, cmap="RdPu", norm=norm,
                             edgecolors="gray", linewidths=0.5)

    draw_square(axes[row_idx, 2])
    axes[row_idx, 2].scatter(x, y, c=df[metric_c], s=sizes, cmap="RdPu", norm=norm,
                             edgecolors="gray", linewidths=0.5)

    for j in range(3):
        axes[row_idx, j].set_title(col_titles[j], fontweight="semibold", fontsize=title_fs, pad=12)

    return sc


# Row 1: Total
sc1 = plot_row(
    0,
    "acc_fusion",
    "synib_acc_fusion",
    "synib_learned_acc_fusion",
    "Total Accuracy",
)
fig.text(0.5, 0.89, "Total Accuracy", ha="center", va="top",
         fontsize=11, fontweight="semibold")

# Row 2: Synergy
sc2 = plot_row(
    1,
    "acc_syn",
    "synib_acc_syn",
    "synib_learned_acc_syn",
    "Accuracy on Synergy",
)
fig.text(0.5, 0.62, "Accuracy on Synergy", ha="center", va="top",
         fontsize=11, fontweight="semibold")

# Row 3: Redundancy (NEW)
sc3 = plot_row(
    2,
    "acc_red",
    "synib_acc_red",
    "synib_learned_acc_red",
    "Accuracy on Redundancy",
)
fig.text(0.5, 0.35, "Accuracy on Redundancy", ha="center", va="top",
         fontsize=11, fontweight="semibold")

# Global note
# fig.text(0.12, 0.91, "PID: Unique1=0, Unique2=0; axes show (Redundancy, Synergy)",
#          ha="left", va="top", fontsize=6)

# ---------- Colorbars (one per row) ----------
fig.subplots_adjust(right=0.85)

cbar_ax1 = fig.add_axes([0.88, 0.675, 0.015, 0.18])
cb1 = fig.colorbar(sc1, cax=cbar_ax1)
cb1.ax.tick_params(labelsize=6)
cb1.set_label("Accuracy", fontweight="semibold")
cb1.formatter = FormatStrFormatter("%.2f")
cb1.update_ticks()

cbar_ax2 = fig.add_axes([0.88, 0.4, 0.015, 0.18])
cb2 = fig.colorbar(sc2, cax=cbar_ax2)
cb2.ax.tick_params(labelsize=6)
cb2.set_label("Accuracy", fontweight="semibold")
cb2.formatter = FormatStrFormatter("%.2f")
cb2.update_ticks()

cbar_ax3 = fig.add_axes([0.88, 0.13, 0.015, 0.18])
cb3 = fig.colorbar(sc3, cax=cbar_ax3)
cb3.ax.tick_params(labelsize=6)
cb3.set_label("Accuracy", fontweight="semibold")
cb3.formatter = FormatStrFormatter("%.2f")
cb3.update_ticks()

# tighter gaps
plt.subplots_adjust(hspace=0.18, wspace=0.25)

plt.savefig("pid_square_red_vs_syn_total_syn_red_v3.png", dpi=300, bbox_inches="tight")
print("Saved: pid_square_red_vs_syn_total_syn_red_v3.png")
