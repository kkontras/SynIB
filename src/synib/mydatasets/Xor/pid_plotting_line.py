import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import json
import numpy as np

PU1 = 0.0
PU2 = 0.0

file_path_main = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_main_snr3_v3.json"
file_path_synib = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib_snr3_v3.json"
file_path_synibleaned = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib_random05_snr3_v3.json"
file_path_synibrandommask = "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy/mydatasets/Xor/runs_refactor/sweep_nonoverlap_probs_tunedkl_synib_random05_snr3_v3.json"

def _parse_key_probs(key: str) -> dict:
    # key example: "pu1=0.2|pu2=0.0|psyn=0.4|pred=..."
    parts = {}
    for p in key.split("|"):
        k, v = p.split("=")
        parts[k] = float(v)
    return parts

def _collect_from_file(json_path: str, summary_key: str, PU1: float, PU2: float):
    """
    Returns dict: {"tot": [(psyn, mean, std), ...], "syn": [(psyn, mean, std), ...]}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    out = {"tot": [], "syn": []}
    results = data.get("results", {})

    for key, content in results.items():
        parts = _parse_key_probs(key)

        if parts.get("pu1", 0.0) != PU1 or parts.get("pu2", 0.0) != PU2:
            continue

        psyn = parts.get("psyn", 0.0)
        summary = content.get("summary_meanstd", {})

        if summary_key not in summary:
            continue

        s = summary[summary_key]

        # tot
        mean_tot = s.get("test_tot_mean", 0.0)
        std_tot  = s.get("test_tot_std", 0.0)
        if isinstance(std_tot, float) and np.isnan(std_tot):
            std_tot = 0.0
        out["tot"].append((psyn, mean_tot, std_tot))

        # syn
        mean_syn = s.get("test_syn_mean", 0.0)
        std_syn  = s.get("test_syn_std", 0.0)
        if isinstance(std_syn, float) and np.isnan(std_syn):
            std_syn = 0.0
        out["syn"].append((psyn, mean_syn, std_syn))

    return out

def load_them(file_path_main, file_path_synib, file_path_synibleaned, file_path_synibrandommask, PU1=PU1, PU2=PU2):
    plot_data = {
        "main":   {"tot": [], "syn": []},
        "synib":  {"tot": [], "syn": []},
        "learned":{"tot": [], "syn": []},
        "random_mask":{"tot": [], "syn": []},
    }

    # main file provides "main"
    plot_data["main"] = _collect_from_file(
        json_path=file_path_main,
        summary_key="main",
        PU1=PU1,
        PU2=PU2,
    )

    # synib file provides "synib_tuned"
    plot_data["synib"] = _collect_from_file(
        json_path=file_path_synib,
        summary_key="synib_tuned",
        PU1=PU1,
        PU2=PU2,
    )

    # learned file provides "learned_tuned"
    plot_data["learned"] = _collect_from_file(
        json_path=file_path_synibleaned,
        # summary_key="learned_tuned",
        summary_key="synib_RM_tuned",
        PU1=PU1,
        PU2=PU2,
    )

    # learned file provides "learned_tuned"
    plot_data["random_mask"] = _collect_from_file(
        json_path=file_path_synibrandommask,
        # summary_key="learned_tuned",
        summary_key="synib_RM_tuned",
        PU1=PU1,
        PU2=PU2,
    )
    print(plot_data.keys())

    return plot_data

# Usage:


def load_and_plot():
    plot_data = load_them(file_path_main, file_path_synib, file_path_synibleaned, file_path_synibrandommask, PU1=PU1, PU2=PU2)

    # 3. Figure Setup
    plt.rcParams.update({'font.size': 14})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True)

    colors = {"main": "#2c3e50", "synib": "#2980b9", "learned": "#c0392b", "random_mask": "lightgreen"}
    names = {"main": "No Regularization", "synib": r"SynIB $M^*_s$", "learned": "SynIB Learned M", "random_mask": r"SynIB $M_{Random}$"}
    all_x = []

    for label in ["synib", "random_mask", "main"]:
        for ax, metric, mkr in zip([ax1, ax2], ["tot", "syn"], ['o', 's']):
            v = sorted(plot_data[label][metric], key=lambda x: x[0])
            if v:
                x_coords, means, stds = map(np.array, zip(*v))
                all_x = x_coords
                ax.plot(x_coords, means, label=names[label], color=colors[label], marker=mkr, linewidth=3, markersize=8)
                ax.fill_between(x_coords, means - stds, means + stds, color=colors[label], alpha=0.15)

    # 4. Formatting and Special Annotations
    xtick_labels = [f"({x:.2f}, {1.0 - PU1 - x:.2f})" for x in all_x]

    for ax in [ax1, ax2]:
        ax.set_ylabel("Accuracy", fontsize=16, fontweight='bold')
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="center left", frameon=True, fontsize=12)  # Moved legend to avoid overlapping corner box
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xticks(all_x)
        ax.set_xticklabels(xtick_labels, fontsize=5)
        ax.tick_params(labelbottom=True)

        # Lowered intuitive boxes
        ax.text(0.02, 0.02, "Less Synergy / More Redundancy", transform=ax.transAxes,
                fontsize=11, color='gray', style='italic', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        ax.text(0.98, 0.02, "More Synergy / Less Redundancy", transform=ax.transAxes,
                fontsize=11, color='gray', style='italic', ha='right',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        # NEW: Unique Information Status Box
        ax.text(0.95, 1.05, fr"$PU1 = {PU1}, PU2 = {PU2}$", transform=ax.transAxes,
                fontsize=13, fontweight='bold', ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', edgecolor='#dee2e6', alpha=0.9))

    ax1.set_title("Total Accuracy Transition", fontweight='bold', fontsize=18, pad=25)
    ax2.set_title("Synergy Accuracy Transition", fontweight='bold', fontsize=18, pad=25)
    ax2.set_xlabel(r"Information Regime Pairs $(P_{synergy}, P_{redundancy})$", fontsize=16, labelpad=20)
    ax1.set_ylim(0.85,1.0)
    ax2.set_ylim(0.5,1.0)
    plt.tight_layout()
    save_path = f"pid_accuracy_comparison_plot_u1_{PU1}_u2_{PU2}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Plot saved successfully with PID corner box: {save_path}")


if __name__ == "__main__":
    load_and_plot()