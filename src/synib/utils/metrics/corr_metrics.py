from itertools import combinations
from sklearn.cross_decomposition import CCA
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from colorama import Fore
from openTSNE import TSNE
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -------------------------------------------------------
# Feature comparison metrics
# -------------------------------------------------------

def compute_pairwise_dist_corr(X, Y):
    return pearsonr(pdist(X), pdist(Y))[0]

def compute_cka(X, Y):
    Kx, Ky = linear_kernel(X), linear_kernel(Y)
    Kx -= Kx.mean(axis=0, keepdims=True)
    Ky -= Ky.mean(axis=0, keepdims=True)
    return np.sum(Kx * Ky) / (np.linalg.norm(Kx, 'fro') * np.linalg.norm(Ky, 'fro'))

def compute_cosine_mean(X, Y):
    return cosine_similarity(X, Y).mean()

def compute_cca_mean_corr(X, Y, n_components=10):
    n = min(n_components, X.shape[1], Y.shape[1])
    try:
        cca = CCA(n_components=n).fit(X, Y)
        Xc, Yc = cca.transform(X, Y)
        return np.mean([np.corrcoef(Xc[:, i], Yc[:, i])[0, 1] for i in range(n)])
    except:
        return 1.0


def compute_procrustes(X, Y):
    _, _, disp = procrustes(X, Y)
    return disp


# -------------------------------------------------------
# Flatten metrics for W&B logging
# -------------------------------------------------------

def log_feature_results(prefix, metrics_dict):
    """Convert a dict of metrics into W&B log-ready flattened dict."""
    base = f"{prefix}"
    return {
        f"{base}/dist_corr": metrics_dict["dist_corr"],
        f"{base}/cka": metrics_dict["cka"],
        f"{base}/cosine_sim": metrics_dict["cosine_sim"],
        f"{base}/cca_mean_corr": metrics_dict["cca_mean_corr"],
        f"{base}/procrustes_disp": metrics_dict["procrustes_disp"]
    }


# -------------------------------------------------------
# Main function for feature comparisons
# -------------------------------------------------------

def compare_feature_spaces(features: dict):
    """
    Returns ONE flattened dict ready for wandb.log().
    """
    flat_results = {}

    for a, b in combinations(features.keys(), 2):
        Xa, Xb = features[a], features[b]

        metrics = {
            "dist_corr": compute_pairwise_dist_corr(Xa, Xb),
            "cka": compute_cka(Xa, Xb),
            "cosine_sim": compute_cosine_mean(Xa, Xb),
            "cca_mean_corr": compute_cca_mean_corr(Xa, Xb),
            "procrustes_disp": compute_procrustes(Xa, Xb),
        }

        prefix = f"{a}_vs_{b}"
        flat_results.update(log_feature_results(prefix, metrics))

    return flat_results


# -------------------------------------------------------
# Pretty printing
# -------------------------------------------------------

def print_feature_comparisons(results_dict):
    # Extract all pair names
    pairs = sorted({k.split("/")[0] for k in results_dict.keys()})

    metrics = ["dist_corr", "cka", "cosine_sim", "cca_mean_corr", "procrustes_disp"]

    # Header
    header = f"{'Pair':<22} " + " ".join([f"{m:>15}" for m in metrics])
    print("\n" + header)
    print("-" * len(header))

    # Rows
    for pair in pairs:
        row = f"{pair:<22} "
        for m in metrics:
            key = f"{pair}/{m}"
            val = results_dict.get(key, float('nan'))
            row += f"{val:>15.4f}"
        print(row)

    print("-" * len(header))

# -------------------------------------------------------
# TSNE plot
# -------------------------------------------------------

def tsne_plot(z1, z2, z1_tilde, z2_tilde, y, perplexity=30, separate_tsne=True):
    print(Fore.CYAN + "\n🔹 Running t-SNE…" + Fore.RESET)

    ts1 = TSNE(n_components=2, perplexity=perplexity).fit(z1)
    ts2 = TSNE(n_components=2, perplexity=perplexity).fit(z2)

    z1_e = np.asarray(ts1)
    z2_e = np.asarray(ts2)

    if separate_tsne:
        z1_t_e = TSNE(n_components=2, perplexity=perplexity).fit(z1_tilde)
        z2_t_e = TSNE(n_components=2, perplexity=perplexity).fit(z2_tilde)
    else:
        z1_t_e = ts1.transform(z1_tilde)
        z2_t_e = ts2.transform(z2_tilde)

    cmap = ListedColormap([
        "#E69F00", "#56B4E9", "#009E73", "#F0E442",
        "#0072B2", "#D55E00", "#CC79A7", "#999999"
    ])

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    labels = [
        "z₁", "z₂",
        "ẑ₁ (separate t-SNE)" if separate_tsne else "ẑ₁ → z₁ t-SNE",
        "ẑ₂ (separate t-SNE)" if separate_tsne else "ẑ₂ → z₂ t-SNE"
    ]

    data = [z1_e, z2_e, z1_t_e, z2_t_e]

    for ax, emb, title in zip(axs.ravel(), data, labels):
        ax.scatter(emb[:, 0], emb[:, 1], c=y, cmap=cmap, s=8)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("t-SNE embeddings", fontsize=16)
    plt.tight_layout()
    plt.show()
