import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
import matplotlib.patheffects as pe
import json
import pandas as pd
import numpy as np


SQRT3_2 = np.sqrt(3) / 2
EPS = 1e-12

def ternary_xy(a, b, c):
    x = b + 0.5 * c
    y = SQRT3_2 * c
    return x, y

def draw_triangle(ax, left, right, top):
    # Triangle vertices in Cartesian coords (matching ternary_xy)
    v_left  = np.array([0.0, 0.0])                 # (a,b,c) = (1,0,0)
    v_right = np.array([1.0, 0.0])                 # (0,1,0)
    v_top   = np.array([0.5, SQRT3_2])             # (0,0,1)

    tri = np.array([v_left, v_right, v_top, v_left])
    ax.plot(tri[:, 0], tri[:, 1], lw=1.6, color="black")

    # light grid
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for pair in [
            ((t, 1 - t, 0), (t, 0, 1 - t)),
            ((1 - t, t, 0), (0, t, 1 - t)),
            ((1 - t, 0, t), (0, 1 - t, t)),
        ]:
            p1 = ternary_xy(*pair[0])
            p2 = ternary_xy(*pair[1])
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], lw=0.5, alpha=0.15, color="0.55")

    # --- Corner labels, centered on each corner (slightly offset outward) ---
    outline = [pe.withStroke(linewidth=2, foreground="white")]
    corner_fs = 6

    ax.annotate(left,  xy=v_left,  xytext=(-15, -4), textcoords="offset points",
                ha="left", va="top", fontweight="semibold", path_effects=outline, fontsize=corner_fs)
    ax.annotate(right, xy=v_right, xytext=(+15, -4), textcoords="offset points",
                ha="right",  va="top", fontweight="semibold", path_effects=outline, fontsize=corner_fs)
    ax.annotate(top,   xy=v_top,   xytext=(0, +3),   textcoords="offset points",
                ha="center", va="bottom", fontweight="semibold", path_effects=outline, fontsize=corner_fs)

    tick_fs = 6
    tick_len = 0.03

    def add_mid_tick_and_label(p_abc, rotation_deg, offset_pts):
        # point on triangle
        x, y = ternary_xy(*p_abc)
        p = np.array([x, y])

        if abs(p_abc[2]) < 1e-12:          # c == 0
            n = np.array([0.0, -1.0])
        elif abs(p_abc[0]) < 1e-12:        # a == 0
            n = np.array([+SQRT3_2, 0.5])  # roughly outward from right edge
        else:                               # b == 0
            n = np.array([-SQRT3_2, 0.5])  # roughly outward from left edge

        n = n / (np.linalg.norm(n) + 1e-12)

        p1 = p - 0.5 * tick_len * n
        p2 = p + 0.5 * tick_len * n
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", lw=1.0)

        ax.annotate(
            f"p=({p_abc[0]:.1f},{p_abc[1]:.1f},{p_abc[2]:.1f})",
            xy=p, xytext=offset_pts, textcoords="offset points",
            ha="center", va="center",
            rotation=rotation_deg, rotation_mode="anchor",
            fontweight="normal", fontsize=tick_fs,
            path_effects=outline
        )

    add_mid_tick_and_label((0.5, 0.5, 0), rotation_deg=0,   offset_pts=(0, -10))
    add_mid_tick_and_label((0, 0.5, 0.5), rotation_deg=-60,  offset_pts=(+12, 0))
    add_mid_tick_and_label((0.5, 0, 0.5), rotation_deg=60, offset_pts=(-12, 0))

    ax.set_aspect("equal")
    ax.axis("off")

def load_records_from_three(
    file_path_main,
    file_path_synib,
    file_path_syniblearned,
):
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

    # merge keys across all three
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
            "acc_syn":   safe_get(vm, "summary_meanstd", "main", "test_syn_mean"),

            # synib metrics
            "synib_acc_fusion": safe_get(vs, "summary_meanstd", "synib_tuned", "test_tot_mean"),
            "synib_acc_syn":    safe_get(vs, "summary_meanstd", "synib_tuned", "test_syn_mean"),

            # learned metrics
            # "synib_learned_acc_fusion": safe_get(vl, "summary_meanstd", "learned_tuned", "test_tot_mean"),
            # "synib_learned_acc_syn":    safe_get(vl, "summary_meanstd", "learned_tuned", "test_syn_mean"),

            # learned metrics
            "synib_learned_acc_fusion": safe_get(vl, "summary_meanstd", "main", "test_tot_mean"),
            # "synib_learned_acc_fusion": safe_get(vl, "summary_meanstd", "synib_RM_tuned", "test_tot_mean"),
            "synib_learned_acc_syn":    safe_get(vl, "summary_meanstd", "main", "test_syn_mean"),
            # "synib_learned_acc_syn":    safe_get(vl, "summary_meanstd", "synib_RM_tuned", "test_syn_mean"),
        })

    return pd.DataFrame(records)


# file_path_main = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_main.json"
# file_path_synib = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib.json"


file_path_main = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_main_snr3_v3.json"
file_path_synib = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib_snr3_v3.json"
file_path_synibleaned = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_mainmask03_snr3_v3.json"
# file_path_synibleaned = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib_random05_snr3_v3.json"

df = load_records_from_three(file_path_main, file_path_synib, file_path_synibleaned)

cols_to_check = ['synib_learned_acc_fusion']
df_filtered = df.dropna(subset=cols_to_check)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.width', 1000) # Increases the character width of the output
print(df_filtered.to_string())


# make a 2x3 grid (top: total acc, bottom: synergy acc)
fig, axes = plt.subplots(2, 3, figsize=(10, 7))

# ---------- ROW 1: Total Accuracy ----------
main_name = "acc_fusion"
synib_name = "synib_acc_fusion"
synib_learned_name = "synib_learned_acc_fusion"

global_min = min(df[main_name].min(), df[synib_name].min(), df[synib_learned_name].min())
global_max = 1.0
norm1 = Normalize(vmin=global_min, vmax=global_max)

keep = df["u1"] + df["red"] + df["syn"]
x, y = ternary_xy(df["u1"]/(keep+EPS), df["red"]/(keep+EPS), df["syn"]/(keep+EPS))
sizes = 30 + 100 * keep - 70

draw_triangle(axes[0, 0], "Unique 1", "Redundancy", "Synergy")
sc1 = axes[0, 0].scatter(x, y, c=df[main_name], s=sizes, cmap="RdPu", norm=norm1,
                         edgecolors="gray", linewidths=0.5)

draw_triangle(axes[0, 1], "Unique 1", "Redundancy", "Synergy")
axes[0, 1].scatter(x, y, c=df[synib_name], s=sizes, cmap="RdPu", norm=norm1,
                   edgecolors="gray", linewidths=0.5)

draw_triangle(axes[0, 2], "Unique 1", "Redundancy", "Synergy")
axes[0, 2].scatter(x, y, c=df[synib_learned_name], s=sizes, cmap="RdPu", norm=norm1,
                   edgecolors="gray", linewidths=0.5)

title_fs = 9
axes[0, 0].set_title("(a) No Regularization", fontweight="semibold", fontsize=title_fs, pad=12)
axes[0, 1].set_title(r"(b) SynIB $M^*_s$",          fontweight="semibold", fontsize=title_fs, pad=12)
axes[0, 2].set_title(r"(c) Masked Dropout $M_{Random}$",   fontweight="semibold", fontsize=title_fs, pad=12)

# ---------- ROW 2: Accuracy on Synergy ----------
main_name = "acc_syn"
synib_name = "synib_acc_syn"
synib_learned_name = "synib_learned_acc_syn"

global_min = min(df[main_name].min(), df[synib_name].min(), df[synib_learned_name].min())
global_max = 1.0
norm2 = Normalize(vmin=global_min, vmax=global_max)

draw_triangle(axes[1, 0], "Unique 1", "Redundancy", "Synergy")
sc2 = axes[1, 0].scatter(x, y, c=df[main_name], s=sizes, cmap="RdPu", norm=norm2,
                         edgecolors="gray", linewidths=0.5)

draw_triangle(axes[1, 1], "Unique 1", "Redundancy", "Synergy")
axes[1, 1].scatter(x, y, c=df[synib_name], s=sizes, cmap="RdPu", norm=norm2,
                   edgecolors="gray", linewidths=0.5)

draw_triangle(axes[1, 2], "Unique 1", "Redundancy", "Synergy")
axes[1, 2].scatter(x, y, c=df[synib_learned_name], s=sizes, cmap="RdPu", norm=norm2,
                   edgecolors="gray", linewidths=0.5)

axes[1, 0].set_title("(a) No Regularization", fontweight="semibold", fontsize=title_fs, pad=12)
axes[1, 1].set_title(r"(b) SynIB $M^*_s$",          fontweight="semibold", fontsize=title_fs, pad=12)
axes[1, 2].set_title(r"(c) Masked Dropout $M_{Random}$",   fontweight="semibold", fontsize=title_fs, pad=12)

# ---------- Row titles (general titles) ----------
fig.text(0.5, 0.905, "Total Accuracy", ha="center", va="top", fontsize=11, fontweight="semibold")
fig.text(0.5, 0.505,  "Accuracy on Synergy", ha="center", va="top", fontsize=11, fontweight="semibold")

# Optional annotation (keep if you want it once for the whole figure)
fig.text(0.12, 0.54, "PID p = (Unique1, Redundancy, Synergy)", ha="left", va="top", fontsize=6)

# ---------- Colorbars (one per row) ----------
fig.subplots_adjust(right=0.85)

# top row colorbar
cbar_ax1 = fig.add_axes([0.88, 0.59, 0.015, 0.23])
cb1 = fig.colorbar(sc1, cax=cbar_ax1)
cb1.ax.tick_params(labelsize=6)
cb1.set_label("Accuracy", fontweight="semibold")
cb1.formatter = FormatStrFormatter("%.2f")
cb1.update_ticks()

# bottom row colorbar
cbar_ax2 = fig.add_axes([0.88, 0.20, 0.015, 0.23])
cb2 = fig.colorbar(sc2, cax=cbar_ax2)
cb2.ax.tick_params(labelsize=6)
cb2.set_label("Accuracy", fontweight="semibold")
cb2.formatter = FormatStrFormatter("%.2f")
cb2.update_ticks()

# ---------- CLOSE THE GAP BETWEEN ROWS ----------
# Smaller hspace => smaller gap. Try 0.05, 0.02, 0.0
plt.subplots_adjust(hspace=0.05, wspace=0.25)

plt.savefig("pid_mainvssynib_total_and_syn_v3.png", dpi=300, bbox_inches="tight")