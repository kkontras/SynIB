"""rebuttal_entangled_probe.py — Arm B: measured entanglement verification.

For each dataset variant {identity, tanh∘Q (rot_seed 0), frozen-MLP mixer (seed 0)}:
  1. Per-coordinate screen: max over input coordinates of |point-biserial corr| and a
     16-quantile-bin MI estimate between x[j] and each source's latent bit.
  2. L1-logistic sparse-probe curve (accuracy vs support size); minimal support
     achieving >= 90% of the unregularized probe's held-out accuracy.

Sources probed (each on the examples where that source is active, train split, seed-0
data, standardized inputs exactly as the encoder sees them):
  unique  (modality 0, target y on u1 examples)
  red     (modality 0, target y on red examples)
  syn b0  (modality 0, target latent bit b_s0 on syn examples)
  syn b1  (modality 1, target latent bit b_s1 on syn examples)

Usage: python scripts/analysis/rebuttal_entangled_probe.py
Writes: artifacts/rebuttal_entangled_xor/probe_table.{json,md},
        docs/figures/rebuttal_entangled_xor/fig3_sparse_probes.{pdf,png}
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

ANALYSIS = Path(__file__).resolve().parent
sys.path.insert(0, str(ANALYSIS))
import rebuttal_entangled_xor as rx  # noqa: E402  (reuses hooks, dirs, style)

pid_mod = rx._load_module(ANALYSIS / "Xor_PID3Main_MaskSynIB_Search.py", "pid_mod_probe")

VARIANTS = ["identity", "tanh_Q", "mlp_mixer"]
SEED = 0
BINS = 16
C_GRID = np.logspace(-2.5, 2.0, 25)


def build_variant(variant: str):
    cfg = pid_mod.Config()
    cfg.device = "cpu"
    cfg.seed = SEED
    pid_mod._set_nonoverlap_signal_probs(cfg, 0.45, 0.0, 0.45, 0.10, pnone=0.0)
    if variant == "tanh_Q":
        cfg.rotation_Q0 = rx.get_rotation(pid_mod, "full", 0, 0, cfg.dim0, cfg)
        cfg.rotation_Q1 = rx.get_rotation(pid_mod, "full", 0, 1, cfg.dim1, cfg)
        cfg.rotation_nonlinearity = "tanh"
    elif variant == "mlp_mixer":
        cfg.mlp_mixer0 = rx.get_mlp_mixer(pid_mod, cfg, 0, 0, cfg.dim0, 2.5)
        cfg.mlp_mixer1 = rx.get_mlp_mixer(pid_mod, cfg, 0, 1, cfg.dim1, 2.5)
    rx.set_global_seed(SEED)
    _, _, train_l, _, _ = pid_mod.build_loaders(cfg, verbose=False)
    ds = train_l.dataset.dataset
    idx = np.asarray(train_l.dataset.indices)
    x0 = ds.x0[idx].numpy()
    x1 = ds.x1[idx].numpy()
    y = ds.y[idx].numpy().ravel()
    src = ds.source[idx].numpy()  # [n,4] u1,u2,red,syn
    b0 = ds.synbit0[idx].numpy()
    b1 = ds.synbit1[idx].numpy()
    return cfg, x0, x1, y, src, b0, b1


def point_biserial_max(X: np.ndarray, t: np.ndarray) -> float:
    t = t.astype(float)
    tc = t - t.mean()
    Xc = X - X.mean(0)
    num = (Xc * tc[:, None]).mean(0)
    den = X.std(0) * t.std() + 1e-12
    return float(np.abs(num / den).max())


def binned_mi_max(X: np.ndarray, t: np.ndarray) -> float:
    from sklearn.metrics import mutual_info_score
    best = 0.0
    for j in range(X.shape[1]):
        q = np.quantile(X[:, j], np.linspace(0, 1, BINS + 1)[1:-1])
        xb = np.digitize(X[:, j], q)
        best = max(best, mutual_info_score(t.astype(int), xb))
    return float(best)


def sparse_probe(X: np.ndarray, t: np.ndarray, rng_seed: int = 0) -> Dict[str, Any]:
    """L1-logistic path: (support size, held-out acc) per C + unregularized ref."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(rng_seed)
    n = len(t)
    perm = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    Xtr, Xte, ttr, tte = X[tr], X[te], t[tr], t[te]
    ref = LogisticRegression(penalty=None, max_iter=5000).fit(Xtr, ttr)
    ref_acc = float(ref.score(Xte, tte))
    curve = []
    for C in C_GRID:
        clf = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=5000)
        clf.fit(Xtr, ttr)
        supp = int((np.abs(clf.coef_) > 1e-6).sum())
        acc = float(clf.score(Xte, tte))
        curve.append({"C": float(C), "support": supp, "acc": acc})
    thresh = 0.9 * ref_acc
    ok = [c for c in curve if c["acc"] >= thresh and c["support"] > 0]
    min_support = min((c["support"] for c in ok), default=X.shape[1])
    return {"ref_acc": ref_acc, "min_support_90": int(min_support), "curve": curve,
            "n_examples": int(n)}


def destruction_support(X: np.ndarray, t: np.ndarray, chance_margin: float = 0.60,
                        rng_seed: int = 0) -> Dict[str, Any]:
    """Minimal DESTRUCTION support: greedily remove (destroy) coordinates, retraining
    an unregularized probe on the REMAINING coordinates each step; report the smallest
    number of destroyed coordinates that drives the best remaining-coordinate reader
    to <= chance_margin held-out accuracy. This is the masking-relevant notion of
    entanglement: on axis-aligned data a small block suffices; under mixing every
    coordinate carries a projection of the (1-D) source signal, so nearly all must go."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.RandomState(rng_seed)
    n = len(t)
    perm = rng.permutation(n)
    n_tr = int(0.8 * n)
    tr, te = perm[:n_tr], perm[n_tr:]

    def fit_on(keep: List[int]):
        clf = LogisticRegression(penalty=None, max_iter=3000)
        clf.fit(X[tr][:, keep], t[tr])
        return clf, float(clf.score(X[te][:, keep], t[te]))

    # Inverse RFE: at each step retrain on the remaining coordinates and destroy the
    # highest-|coefficient| one. A min-over-single-removals greedy stalls on redundant
    # pairs (removing either member alone loses nothing, so ties send it off destroying
    # noise coordinates); targeting max|coef| dismantles redundant groups member by member.
    remaining = list(range(X.shape[1]))
    clf, acc0 = fit_on(remaining)
    curve = [{"n_destroyed": 0, "acc": acc0}]
    while remaining:
        j_local = int(np.abs(clf.coef_[0]).argmax())
        remaining.pop(j_local)
        if not remaining:
            curve.append({"n_destroyed": X.shape[1], "acc": 0.5})
            break
        clf, a = fit_on(remaining)
        curve.append({"n_destroyed": X.shape[1] - len(remaining), "acc": a})
        if a <= chance_margin:
            break
    k = curve[-1]["n_destroyed"] if curve[-1]["acc"] <= chance_margin else X.shape[1]
    return {"min_destroy_to_chance": int(k), "chance_margin": chance_margin,
            "destroy_curve": curve}


def main() -> None:
    results: Dict[str, Any] = {}
    for variant in VARIANTS:
        cfg, x0, x1, y, src, b0, b1 = build_variant(variant)
        probes = {
            "unique (m0)": (x0[src[:, 0] > 0.5], y[src[:, 0] > 0.5]),
            "red (m0)": (x0[src[:, 2] > 0.5], y[src[:, 2] > 0.5]),
            "syn b0 (m0)": (x0[src[:, 3] > 0.5], b0[src[:, 3] > 0.5]),
            "syn b1 (m1)": (x1[src[:, 3] > 0.5], b1[src[:, 3] > 0.5]),
        }
        results[variant] = {}
        for name, (X, t) in probes.items():
            assert set(np.unique(t)).issubset({0.0, 1.0}), f"bad targets for {name}"
            r = sparse_probe(X, t)
            r["max_point_biserial"] = point_biserial_max(X, t)
            r["max_binned_mi_bits"] = binned_mi_max(X, t) / np.log(2)
            r.update(destruction_support(X, t))
            results[variant][name] = r
            print(f"[{variant:10s}] {name:12s} n={r['n_examples']:4d} "
                  f"ref_acc={r['ref_acc']:.3f} read_support={r['min_support_90']:2d}/32 "
                  f"destroy_support={r['min_destroy_to_chance']:2d}/32 "
                  f"max|pb|={r['max_point_biserial']:.3f} maxMI={r['max_binned_mi_bits']:.3f}b")

    # table
    lines = ["| variant | source | n | unreg. probe acc | read support (≥90% ref) | destroy support (→chance) | max |point-biserial| | max binned MI (bits) |",
             "|---|---|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for name, r in results[v].items():
            lines.append(f"| {v} | {name} | {r['n_examples']} | {r['ref_acc']:.3f} "
                         f"| {r['min_support_90']}/32 | {r['min_destroy_to_chance']}/32 "
                         f"| {r['max_point_biserial']:.3f} "
                         f"| {r['max_binned_mi_bits']:.3f} |")
    md = "\n".join(lines)
    (rx.ART_DIR / "probe_table.md").write_text(md + "\n")
    (rx.ART_DIR / "probe_table.json").write_text(json.dumps(results, indent=1))
    print(md)

    # figure: accuracy vs support-size curves
    rx._apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.7), sharey=True)
    vcolor = {"identity": "#2563EB", "tanh_Q": "#16A34A", "mlp_mixer": "#DC2626"}
    vlabel = {"identity": "identity", "tanh_Q": r"tanh$\circ$Q", "mlp_mixer": "frozen MLP"}
    for ax, name in zip(axes, ["unique (m0)", "red (m0)", "syn b0 (m0)", "syn b1 (m1)"]):
        for v in VARIANTS:
            r = results[v][name]
            pts = sorted({(c["support"], c["acc"]) for c in r["curve"] if c["support"] > 0})
            xs = [p[0] for p in pts]
            best = np.maximum.accumulate([p[1] for p in pts])
            ax.plot(xs, best, marker="o", ms=2.5, lw=1.3, color=vcolor[v], label=vlabel[v])
            ax.axhline(0.9 * r["ref_acc"], color=vcolor[v], lw=0.6, linestyle=(0, (1, 2)))
        ax.set_title(name)
        ax.set_xlabel("L1-probe support size")
        ax.set_xlim(0, 33)
    axes[0].set_ylabel("Held-out probe accuracy")
    axes[0].legend(loc="lower right", fontsize=7)
    fig.suptitle("Sparse READ probes (L1): a single coordinate suffices even under mixing "
                 "(the source is a 1-D direction)", y=1.04, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(rx.FIG_DIR / f"fig3_sparse_probes.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {rx.FIG_DIR}/fig3_sparse_probes.pdf/.png")

    fig2, axes2 = plt.subplots(1, 4, figsize=(11.5, 2.7), sharey=True)
    for ax, name in zip(axes2, ["unique (m0)", "red (m0)", "syn b0 (m0)", "syn b1 (m1)"]):
        for v in VARIANTS:
            r = results[v][name]
            xs = [c["n_destroyed"] for c in r["destroy_curve"]]
            ys = [c["acc"] for c in r["destroy_curve"]]
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.3, color=vcolor[v], label=vlabel[v])
        ax.axhline(0.6, color="#999999", lw=0.6, linestyle=(0, (1, 2)))
        ax.set_title(name)
        ax.set_xlabel("Coordinates destroyed (greedy)")
        ax.set_xlim(0, 33)
    axes2[0].set_ylabel("Best remaining-coord probe acc")
    axes2[0].legend(loc="lower left", fontsize=7)
    fig2.suptitle("DESTRUCTION support: under mixing, nearly all coordinates must be destroyed "
                  "to remove a source — the masking-relevant entanglement", y=1.04, fontsize=9)
    fig2.tight_layout()
    for ext in ("pdf", "png"):
        fig2.savefig(rx.FIG_DIR / f"fig4_destroy_support.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {rx.FIG_DIR}/fig4_destroy_support.pdf/.png")


if __name__ == "__main__":
    main()
