"""figs_pid_ntk_dynamics.py
Generate the 2x3 NTK-diagnostic figure for PID-XOR (vanilla vs SynIB-learned).

Three diagnostics, repeated for two training methods:
    - lambda_g  (per-source NTK signal strength)
    - cos(g, h) (gradient alignment between source pairs)
    - BCE loss by source (train, dashed validation)

Bands are mean +/- 1 std across 3 seeds. Style matches the appendix
training-dynamics figures (see figs_training_dynamics.py).

Usage:
    python scripts/analysis/figs_pid_ntk_dynamics.py [--mode all|render|smoke] [--device cuda]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import types as _types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.modules.setdefault("wandb", _types.SimpleNamespace(run=None, init=lambda **kw: None))

if not hasattr(torch, "isin"):
    def _isin(elements, test_elements):
        elements = elements.unsqueeze(-1)
        test_elements = test_elements.view(*([1] * (elements.dim() - 1)), -1)
        return (elements == test_elements).any(dim=-1)
    torch.isin = _isin  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "scripts" / "analysis"
ART_DIR = REPO / "artifacts" / "training_dynamics"
FIG_DIR = REPO / "docs" / "figures" / "xor_ntk"
ART_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

PID_NAMES = ["U1", "Red", "Syn"]
COS_KEYS = ["U1-Syn", "Red-Syn"]

COLORS = {
    "U1":      "#2563EB",
    "Red":     "#16A34A",
    "Syn":     "#DC2626",
    "U1-Syn":  "#2563EB",
    "Red-Syn": "#F59E0B",
}
LABELS = {
    "U1": r"$U_1$", "Red": r"$R$", "Syn": r"$S$",
    "U1-Syn": r"$U_1$-$S$", "Red-Syn": r"$R$-$S$",
}


def _apply_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "font.size": 7.5,
        "axes.titlesize": 8.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.5,
        "axes.labelpad": 2.0,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "legend.fontsize": 6.5,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#000000",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "legend.frameon": False,
    })


def _load_module(path: Path, name: str):
    src = path.read_text()
    head = src.split("if __name__ == ")[0]
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)
    mod.__name__ = name
    sys.modules[name] = mod
    exec(compile(head, str(path), "exec"), mod.__dict__)
    return mod


def set_global_seed(seed: int) -> None:
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================================
# Training with full NTK + per-source logging
# =====================================================================================

def _train_with_ntk_logging(
    pid_mod, cfg, train_loader, val_loader, device: str,
    method: str, lambda_kl: float = 10.0, lam_sparsity: float = 1.0,
) -> Dict[str, Any]:
    """Run training (vanilla or SynIB-learned), logging per step:
       - lambda/U1, lambda/Red, lambda/Syn
       - cos/U1-Syn, cos/Red-Syn
       - pidloss/U1, pidloss/Red, pidloss/Syn (train BCE, only over the batch slice for that source)
    Plus per-epoch validation per-source BCE and sample step bookkeeping.
    method in {"vanilla", "mlearned"}.
    """
    model = pid_mod.FusionModel(cfg.dim0, cfg.dim1, cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    series: List[Dict[str, Any]] = []
    val_series: List[Dict[str, Any]] = []
    opt_steps = 0

    def _to_finite(x):
        try:
            v = float(x.item()) if hasattr(x, "item") else float(x)
            return v if np.isfinite(v) else None
        except Exception:
            return None

    for epoch in range(cfg.epochs):
        model.train()
        for b in train_loader:
            x0 = b["x0"].to(device); x1 = b["x1"].to(device); y = b["y"].to(device)
            m0 = b["mask0"].to(device); m1 = b["mask1"].to(device)
            source = b["source"].to(device)

            f, u0, u1 = model.forward_logits(x0, x1)
            lf  = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            ltot = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            if method == "mlearned":
                with torch.enable_grad():
                    masks = pid_mod.learn_destroy_gates(
                        model=model, x0=x0.detach(), x1=x1.detach(), y=y.detach(),
                        method=cfg.learned_mask_method, device=device,
                        gate_shape=cfg.learned_mask_gate_shape,
                        steps=cfg.learned_mask_steps, lr=cfg.learned_mask_lr,
                        tau=cfg.learned_mask_tau, noise_std=cfg.learned_mask_noise_std,
                        lam_sparsity=lam_sparsity,
                        alpha_unimodal=cfg.learned_mask_alpha_unimodal,
                        hard=cfg.learned_mask_hard, hard_thresh=cfg.learned_mask_hard_thresh,
                        m0=m0.detach(), m1=m1.detach(), iou_target="syn",
                        print_every=10**9, label="",
                    )
                g0, g1 = masks["g0"].detach(), masks["g1"].detach()
                x0_t = pid_mod.destroy_block(x0, g0, 1, noise_std=1.0)
                x1_t = pid_mod.destroy_block(x1, g1, 1, noise_std=1.0)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                l_cf = (pid_mod.bern_kl_to_uniform_from_logits(f_t0)
                        + pid_mod.bern_kl_to_uniform_from_logits(f_t1))
                ltot = ltot + lambda_kl * l_cf
            elif method == "vanilla":
                pass
            else:
                raise ValueError(f"Unknown method {method}")

            opt.zero_grad(set_to_none=True)
            ltot.backward(retain_graph=True)

            lambdas, _vJt, stats = pid_mod.ntk_strengths_onehot_source_debug(
                model, f, y, source, print_debug=False, steps=opt_steps,
            )
            lps = pid_mod.loss_per_source(f, y, source, steps=opt_steps)

            opt.step()

            row: Dict[str, Any] = {"step": int(opt_steps), "epoch": int(epoch)}
            for k in PID_NAMES:
                v = _to_finite(lambdas.get(k, None))
                if v is not None:
                    row[f"lambda/{k}"] = v
            cos_map = stats.get("cosines", {}) if isinstance(stats, dict) else {}
            for ck in COS_KEYS:
                v = _to_finite(cos_map.get(ck, None))
                if v is not None:
                    row[f"cos/{ck}"] = v
            for k in PID_NAMES:
                c = lps.get(f"count/{k}", 0)
                if c is None or int(c) == 0:
                    continue
                lv = _to_finite(lps.get(f"loss/{k}", None))
                if lv is not None:
                    row[f"pidloss/{k}"] = lv
            series.append(row)
            opt_steps += 1

        # epoch-end validation (per-source BCE on val loader)
        val_metrics = pid_mod.eval_loss_per_source(model, val_loader, device)
        val_metrics["epoch"] = int(epoch)
        val_metrics["step"] = int(opt_steps)
        val_series.append(val_metrics)

    return {"series": series, "val": val_series, "total_steps": opt_steps}


def run_experiments(simplex: Tuple[float, float, float, float], seeds: List[int],
                    device: str, methods: List[str],
                    lambda_kl: float, lam_sparsity: float,
                    epochs: int) -> Dict[str, Any]:
    pid_mod = _load_module(ANALYSIS / "Xor_PID3Main_MaskSynIB_Search.py", "pid_mod_ntk")
    pu1, pu2, pred, psyn = simplex
    out = {"simplex": list(simplex), "lambda_kl": lambda_kl, "lam_sparsity": lam_sparsity,
           "epochs": epochs, "seeds": seeds, "methods": {}}
    for method in methods:
        out["methods"][method] = []
        for seed in seeds:
            t0 = time.time()
            cfg = pid_mod.Config()
            cfg.device = device; cfg.seed = seed
            cfg.lr = 3e-4; cfg.weight_decay = 1e-5; cfg.dropout = 0.0
            cfg.epochs = epochs
            pid_mod._set_nonoverlap_signal_probs(cfg, pu1, pu2, pred, psyn, pnone=0.0)
            set_global_seed(seed)
            _, _splits, train_l, val_l, test_l = pid_mod.build_loaders(cfg, verbose=False)
            hist = _train_with_ntk_logging(
                pid_mod, cfg, train_l, val_l, device=device,
                method=method, lambda_kl=lambda_kl, lam_sparsity=lam_sparsity,
            )
            elapsed = time.time() - t0
            last_val = hist["val"][-1]
            print(f"[NTK {method:9s} seed={seed} simplex={simplex}] "
                  f"{elapsed:6.1f}s; final val/perf_loss_Syn={last_val.get('val/perf_loss_Syn', float('nan')):.3f}")
            out["methods"][method].append(hist)
    return out


# =====================================================================================
# Aggregation: cross-seed mean +/- std with light smoothing
# =====================================================================================

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean along the last axis, ignoring NaNs.
    Returns array of same shape as arr.
    """
    if window <= 1:
        return arr
    n = arr.shape[-1]
    half = window // 2
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half); hi = min(n, i + half + 1)
        win = arr[..., lo:hi]
        # nanmean over the window (last axis)
        with np.errstate(invalid="ignore"):
            mean = np.where(
                np.isfinite(win).any(axis=-1),
                np.nanmean(np.where(np.isfinite(win), win, np.nan), axis=-1),
                np.nan,
            )
        out[..., i] = mean
    return out


def _agg_seeds_step(seed_runs: List[Dict[str, Any]], key: str,
                    smooth_window: int = 51) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack a per-step diagnostic across seeds and aggregate.

    Returns (steps, mean, std) where mean/std are computed across seeds AFTER
    per-seed rolling smoothing (which fills holes for sparse signals like
    pidloss when a source is absent from a batch).
    """
    # All seeds are deterministic -> same step set; collect union just in case.
    step_sets = [set(r["step"] for r in run["series"]) for run in seed_runs]
    all_steps = sorted(set().union(*step_sets))
    arrays = []
    for run in seed_runs:
        m = {r["step"]: r.get(key, np.nan) for r in run["series"]}
        arrays.append([float(m.get(s, np.nan)) if m.get(s, np.nan) is not None else np.nan
                       for s in all_steps])
    arr = np.asarray(arrays, dtype=float)  # [n_seeds, n_steps]
    arr_smoothed = _rolling_mean(arr, smooth_window)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(arr_smoothed)
        n = finite.sum(axis=0)
        mean = np.where(n > 0, np.nanmean(arr_smoothed, axis=0), np.nan)
        std = np.where(
            n > 1,
            np.nanstd(arr_smoothed, axis=0, ddof=1),
            0.0,
        )
    return np.asarray(all_steps, dtype=float), mean, std


def _agg_seeds_step_normalized(
    seed_runs: List[Dict[str, Any]], target_key: str, denom_keys: List[str],
    smooth_window: int = 51,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step normalized share: target_key / sum(denom_keys), aggregated across seeds.

    For each seed and each step, compute target / total, where total is the sum of
    finite values across `denom_keys` at that step. Skip steps where total <= 0.
    Then aggregate across seeds (smooth per seed first, then mean +/- std).
    """
    step_sets = [set(r["step"] for r in run["series"]) for run in seed_runs]
    all_steps = sorted(set().union(*step_sets))
    arrays = []
    for run in seed_runs:
        step_map = {r["step"]: r for r in run["series"]}
        seed_vals = []
        for s in all_steps:
            r = step_map.get(s)
            if r is None:
                seed_vals.append(np.nan); continue
            denom = 0.0
            for dk in denom_keys:
                v = r.get(dk, None)
                if v is not None:
                    fv = float(v)
                    if np.isfinite(fv):
                        denom += fv
            num = r.get(target_key, None)
            if num is None or denom <= 0.0 or not np.isfinite(float(num)):
                seed_vals.append(np.nan)
            else:
                seed_vals.append(float(num) / denom)
        arrays.append(seed_vals)
    arr = np.asarray(arrays, dtype=float)
    arr_smoothed = _rolling_mean(arr, smooth_window)
    with np.errstate(invalid="ignore"):
        n = np.isfinite(arr_smoothed).sum(axis=0)
        mean = np.where(n > 0, np.nanmean(arr_smoothed, axis=0), np.nan)
        std = np.where(n > 1, np.nanstd(arr_smoothed, axis=0, ddof=1), 0.0)
    return np.asarray(all_steps, dtype=float), mean, std


def _agg_seeds_val(seed_runs: List[Dict[str, Any]], src: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-epoch validation BCE for `src` across seeds (mean +/- std)."""
    key = f"val/perf_loss_{src}"
    arrays, x_steps_ref = [], None
    for run in seed_runs:
        x_steps = [v.get("step", np.nan) for v in run["val"]]
        if x_steps_ref is None:
            x_steps_ref = x_steps
        arrays.append([float(v.get(key, np.nan)) if v.get(key, np.nan) is not None else np.nan
                       for v in run["val"]])
    arr = np.asarray(arrays, dtype=float)
    with np.errstate(invalid="ignore"):
        n = np.isfinite(arr).sum(axis=0)
        mean = np.where(n > 0, np.nanmean(arr, axis=0), np.nan)
        std = np.where(n > 1, np.nanstd(arr, axis=0, ddof=1), 0.0)
    return np.asarray(x_steps_ref, dtype=float), mean, std


# =====================================================================================
# Plotting
# =====================================================================================

def _style_axes(ax) -> None:
    ax.spines["left"].set_color("#000000")
    ax.spines["bottom"].set_color("#000000")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(axis="both", which="major", direction="out",
                   length=3.0, width=0.6, color="#000000")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.4, zorder=0)
    ax.xaxis.grid(False)


def _load_test_accs_from_pid_full(simplex: Tuple[float, float, float, float]
                                  ) -> Dict[str, Dict[str, float]]:
    """Pull mean test (total, synergy) accuracy per method from the cached
    pid_full.json artifact (same simplex / seeds / hyperparams as our runs).
    Returns {} if cache is unavailable or simplex doesn't match."""
    cache = ART_DIR / "pid_full.json"
    if not cache.exists():
        return {}
    try:
        d = json.loads(cache.read_text())
    except Exception:
        return {}
    cached_simplex = tuple(d.get("simplex", []))
    if cached_simplex and tuple(simplex) != cached_simplex:
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for method, runs in d.get("methods", {}).items():
        if not runs or "test" not in runs[0][-1] if False else not runs:
            continue
        try:
            ns, accs, syns = [], [], []
            for r in runs:
                last = r["test"][-1]
                n = sum(int(last[s]["n"]) for s in ("u1", "u2", "red", "syn") if s in last)
                tot = sum(float(last[s]["acc"]) * int(last[s]["n"])
                          for s in ("u1", "u2", "red", "syn") if s in last)
                ns.append(n); accs.append(tot / max(n, 1))
                syns.append(float(last["syn"]["acc"]))
            out[method] = {"acc_total": float(np.mean(accs)),
                            "acc_syn":   float(np.mean(syns))}
        except Exception:
            continue
    return out


def render_figure(results: Dict[str, Any], basename: str = "pid_ntk_history",
                  smooth_window: int = 51) -> Dict[str, Any]:
    """Build two 1x3 figures (one per method: Vanilla, SynIB) with mean +/- 1 std bands.

    Saves `{basename}_vanilla.{pdf,png}` and `{basename}_synib.{pdf,png}`.
    Y-limits for the lambda and cosine columns are shared across the two
    figures so vanilla and SynIB are visually comparable.
    """
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods = [
        ("vanilla",  "Vanilla",                          "vanilla"),
        ("mlearned", r"SynIB ($M_{\mathrm{Learned}}$)",  "synib"),
    ]

    # Pre-compute aggregated arrays for both methods so we can determine
    # shared y-limits across figures. Lambda panel uses NORMALIZED shares:
    # lambda_g / sum_h lambda_h, so the per-source *relative* weight is
    # comparable between vanilla and SynIB (raw lambdas differ by ~100x).
    lambda_denom_keys = [f"lambda/{k}" for k in PID_NAMES]
    agg = {}  # agg[(method, panel, key)] = (x, mean, std)
    for method, _label, _suffix in methods:
        runs = results["methods"][method]
        for k in PID_NAMES:
            agg[(method, "lambda", k)] = _agg_seeds_step_normalized(
                runs, f"lambda/{k}", lambda_denom_keys, smooth_window,
            )
            agg[(method, "lambda_raw", k)] = _agg_seeds_step(runs, f"lambda/{k}", smooth_window)
            agg[(method, "pidloss", k)] = _agg_seeds_step(runs, f"pidloss/{k}", smooth_window)
            agg[(method, "val_loss", k)] = _agg_seeds_val(runs, k)
        for ck in COS_KEYS:
            agg[(method, "cos", ck)] = _agg_seeds_step(runs, f"cos/{ck}", smooth_window)

    def _ylim_from(values: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                   pad: float = 0.05, lo_floor: float = None, hi_ceiling: float = None,
                   symmetric_zero: bool = False) -> Tuple[float, float]:
        los, his = [], []
        for _x, m, s in values:
            up = m + s; dn = m - s
            up = up[np.isfinite(up)]; dn = dn[np.isfinite(dn)]
            if up.size == 0 or dn.size == 0:
                continue
            los.append(float(np.nanmin(dn))); his.append(float(np.nanmax(up)))
        if not los:
            return -1.0, 1.0
        lo, hi = min(los), max(his)
        rng = max(hi - lo, 1e-9)
        lo -= pad * rng; hi += pad * rng
        if symmetric_zero:
            m = max(abs(lo), abs(hi))
            lo, hi = -m, m
        if lo_floor is not None: lo = max(lo, lo_floor) if lo > lo_floor else lo
        if hi_ceiling is not None: hi = min(hi, hi_ceiling) if hi < hi_ceiling else hi
        return lo, hi

    # Lambda column (normalized share -> bounded in [0, 1] but pad slightly)
    lam_y = _ylim_from([agg[(m, "lambda", k)] for m, _, _ in methods for k in PID_NAMES],
                       lo_floor=0.0, hi_ceiling=1.0)
    lam_y = (max(lam_y[0], 0.0), min(lam_y[1], 1.0))
    # Cosine column: symmetric around zero, clipped to [-1, 1].
    cos_y = _ylim_from([agg[(m, "cos", ck)] for m, _, _ in methods for ck in COS_KEYS],
                       symmetric_zero=True)
    cos_y = (max(cos_y[0], -1.0), min(cos_y[1], 1.0))

    panel_titles = [
        r"Learning Signal Strength $\lambda_g$",
        r"Gradient Alignment $\cos(\cdot, \cdot)$",
        "Fusion Losses by Source",
    ]

    # Pull cached test accuracies for the dataset/accuracy stats box.
    test_accs = _load_test_accs_from_pid_full(tuple(results.get("simplex", (0.45, 0, 0.45, 0.10))))
    pu1, pu2, pred, psyn = results.get("simplex", (0.45, 0.0, 0.45, 0.10))

    band_alpha = 0.18
    final: Dict[str, Dict[str, float]] = {}

    for method, row_label, suffix in methods:
        # Per-figure loss y-limit (independent so SynIB's flat plateau isn't
        # crushed by vanilla's diverging val curve).
        pairs = []
        for k in PID_NAMES:
            pairs.append(agg[(method, "pidloss", k)])
            pairs.append(agg[(method, "val_loss", k)])
        ly = _ylim_from(pairs)
        loss_y = (max(ly[0], 0.0), ly[1])

        runs = results["methods"][method]
        x_lim = (0.0, float(max(int(run["total_steps"]) for run in runs)))

        fig, axes = plt.subplots(1, 3, figsize=(6.75, 1.95),
                                 sharex=False, sharey=False, constrained_layout=False)

        # Panel 1: lambdas
        ax = axes[0]
        for k in PID_NAMES:
            x, m, s = agg[(method, "lambda", k)]
            mask = np.isfinite(m)
            if not mask.any():
                continue
            ax.fill_between(x[mask], (m - s)[mask], (m + s)[mask],
                            color=COLORS[k], alpha=band_alpha, edgecolor="none", zorder=2)
            ax.plot(x[mask], m[mask], color=COLORS[k], linewidth=1.5,
                    label=LABELS[k], zorder=3)
        ax.set_ylim(*lam_y)
        ax.set_xlim(*x_lim)
        ax.set_ylabel("Normalized Strength")
        ax.axhline(1.0 / 3.0, color="#999999", linewidth=0.6, linestyle=(0, (1, 2)),
                   zorder=1)
        ax.set_title(panel_titles[0])
        _style_axes(ax)
        # SynIB's lambda lines plateau low, so the lower-right corner is empty
        # — but the user prefers the legend lifted off the x-axis a touch.
        _lambda_legend_kwargs = dict(
            handles=[
                Line2D([0], [0], color=COLORS["U1"],  lw=1.3, label=r"$U_1$"),
                Line2D([0], [0], color=COLORS["Red"], lw=1.3, label=r"$R$"),
                Line2D([0], [0], color=COLORS["Syn"], lw=1.3, label=r"$S$"),
            ],
            loc="lower right", frameon=False, fontsize=6,
            handlelength=1.1, handletextpad=0.3, labelspacing=0.2,
            ncol=3, columnspacing=0.6,
        )
        if method == "mlearned":
            _lambda_legend_kwargs["bbox_to_anchor"] = (1.0, 0.10)
        ax.legend(**_lambda_legend_kwargs)

        # Panel 2: cosines
        ax = axes[1]
        ax.axhline(0.0, color="#666666", linewidth=0.6, zorder=1)
        for ck in COS_KEYS:
            x, m, s = agg[(method, "cos", ck)]
            mask = np.isfinite(m)
            if not mask.any():
                continue
            ax.fill_between(x[mask], (m - s)[mask], (m + s)[mask],
                            color=COLORS[ck], alpha=band_alpha, edgecolor="none", zorder=2)
            ax.plot(x[mask], m[mask], color=COLORS[ck], linewidth=1.5,
                    label=LABELS[ck], zorder=3)
        ax.set_ylim(*cos_y)
        ax.set_xlim(*x_lim)
        ax.set_ylabel(r"$\cos(\cdot, \cdot)$")
        ax.set_title(panel_titles[1])
        _style_axes(ax)
        ax.legend(
            handles=[
                Line2D([0], [0], color=COLORS["U1-Syn"],  lw=1.3, label=r"$U_1$-$S$"),
                Line2D([0], [0], color=COLORS["Red-Syn"], lw=1.3, label=r"$R$-$S$"),
            ],
            loc="lower right", frameon=False, fontsize=6,
            handlelength=1.1, handletextpad=0.3, labelspacing=0.2,
            ncol=2, columnspacing=0.6,
        )

        # Panel 3: fusion loss by source (train solid + val dashed)
        ax = axes[2]
        for k in PID_NAMES:
            xt, mt, st = agg[(method, "pidloss", k)]
            mask = np.isfinite(mt)
            if mask.any():
                ax.fill_between(xt[mask], (mt - st)[mask], (mt + st)[mask],
                                color=COLORS[k], alpha=band_alpha, edgecolor="none", zorder=2)
                ax.plot(xt[mask], mt[mask], color=COLORS[k], linewidth=1.5,
                        label=f"{LABELS[k]} train", zorder=3)
            xv, mv, sv = agg[(method, "val_loss", k)]
            maskv = np.isfinite(mv)
            if maskv.any():
                ax.fill_between(xv[maskv], (mv - sv)[maskv], (mv + sv)[maskv],
                                color=COLORS[k], alpha=band_alpha * 0.6, edgecolor="none", zorder=2)
                ax.plot(xv[maskv], mv[maskv], color=COLORS[k], linewidth=1.4,
                        linestyle=(0, (1.2, 1.0)), label=f"{LABELS[k]} val", zorder=3)
        ax.set_ylim(*loss_y)
        ax.set_xlim(*x_lim)
        ax.set_ylabel("BCE")
        ax.set_title(panel_titles[2])
        _style_axes(ax)
        ax.legend(
            handles=[
                Line2D([0], [0], color=COLORS["U1"],  lw=1.3, linestyle="-",         label=r"$U_1$ tr"),
                Line2D([0], [0], color=COLORS["U1"],  lw=1.2, linestyle=(0, (1.2, 1.0)), label=r"$U_1$ val"),
                Line2D([0], [0], color=COLORS["Red"], lw=1.3, linestyle="-",         label=r"$R$ tr"),
                Line2D([0], [0], color=COLORS["Red"], lw=1.2, linestyle=(0, (1.2, 1.0)), label=r"$R$ val"),
                Line2D([0], [0], color=COLORS["Syn"], lw=1.3, linestyle="-",         label=r"$S$ tr"),
                Line2D([0], [0], color=COLORS["Syn"], lw=1.2, linestyle=(0, (1.2, 1.0)), label=r"$S$ val"),
            ],
            loc="upper left", frameon=False, fontsize=5.5,
            ncol=3, columnspacing=0.5,
            handlelength=1.2, handletextpad=0.3, labelspacing=0.15,
        )

        # Test-accuracy box, anchored bottom-right.
        accs = test_accs.get(method, {})
        if accs:
            def _pct(x): return f"{float(x)*100:5.1f}%"
            acc_lines = [
                r"$\bf{Test\ accuracy}$",
                f"Total:   {_pct(accs.get('acc_total', float('nan')))}",
                f"Synergy: {_pct(accs.get('acc_syn', float('nan')))}",
            ]
            ax.text(0.97, 0.05, "\n".join(acc_lines),
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=5.5, color="#222222",
                    fontfamily="DejaVu Sans Mono",
                    bbox=dict(boxstyle="round,pad=0.25",
                              facecolor="gainsboro", edgecolor="#94A3B8",
                              linewidth=0.5, alpha=0.95),
                    zorder=5)

        for c in range(3):
            axes[c].set_xlabel("Optimization steps")

        # Method label on the left, sitting just outside the y-axis label.
        # Shifted right (-0.30 -> -0.25) so it hugs the panel more tightly
        # without colliding with the y-axis label. Skipped for vanilla — the
        # caption / filename already identifies it.
        if method != "vanilla":
            axes[0].text(-0.25, 0.5, row_label, transform=axes[0].transAxes,
                         ha="center", va="center", rotation=90, fontsize=8.5,
                         fontweight="bold", color="#000000")

        plt.subplots_adjust(left=0.095, right=0.995, top=0.86, bottom=0.20,
                            wspace=0.28)

        # Nudge the Fusion Losses panel horizontally; vanilla goes a touch
        # left to tighten its gap with the cosine panel, SynIB sits a hair
        # further right so its right-side x-tick labels don't crowd the edge.
        _pos = axes[2].get_position()
        _shift = -0.025 if method == "vanilla" else -0.005
        axes[2].set_position([_pos.x0 + _shift, _pos.y0, _pos.width, _pos.height])

        # Composition descriptor sits inside the Fusion Losses panel, tucked
        # just under its train/val legend (upper-left). Italic, no box —
        # keeps the corner of the figure tidy.
        def _pctb(x): return f"{float(x)*100:.0f}%"
        # Per-method placement: vanilla's val curves climb on the right side
        # so the note hugs the legend; SynIB's plateau frees up more room
        # so the note can sit slightly higher and a touch indented.
        if method == "vanilla":
            comp_x, comp_y = 0.02, 0.78
        else:  # SynIB
            comp_x, comp_y = 0.10, 0.78
        axes[2].text(
            comp_x, comp_y,
            rf"PID-XOR  ·  $U_1$ {_pctb(pu1)}  $U_2$ {_pctb(pu2)}  "
            rf"$R$ {_pctb(pred)}  $S$ {_pctb(psyn)}",
            transform=axes[2].transAxes,
            ha="left", va="top", fontsize=6.0, color="#000000",
            style="italic", zorder=5,
        )

        pdf = FIG_DIR / f"{basename}_{suffix}.pdf"
        png = FIG_DIR / f"{basename}_{suffix}.png"
        fig.savefig(pdf, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        print(f"Saved {pdf} and {png}")
        plt.close(fig)

        final[method] = {}
        for k in PID_NAMES:
            xt, mt, _ = agg[(method, "pidloss", k)]
            xv, mv, _ = agg[(method, "val_loss", k)]
            final[method][f"train/{k}_last"] = float(mt[np.isfinite(mt)][-1]) if np.any(np.isfinite(mt)) else float("nan")
            final[method][f"val/{k}_last"]   = float(mv[np.isfinite(mv)][-1]) if np.any(np.isfinite(mv)) else float("nan")
        for k in PID_NAMES:
            xL, mL, _ = agg[(method, "lambda", k)]
            final[method][f"lambda_share/{k}_last"] = float(mL[np.isfinite(mL)][-1]) if np.any(np.isfinite(mL)) else float("nan")
            xR, mR, _ = agg[(method, "lambda_raw", k)]
            final[method][f"lambda/{k}_last"] = float(mR[np.isfinite(mR)][-1]) if np.any(np.isfinite(mR)) else float("nan")
        for ck in COS_KEYS:
            xC, mC, _ = agg[(method, "cos", ck)]
            final[method][f"cos/{ck}_last"] = float(mC[np.isfinite(mC)][-1]) if np.any(np.isfinite(mC)) else float("nan")
    return final


# =====================================================================================
# Main
# =====================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all", "render", "smoke"], default="all")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--simplex", nargs=4, type=float, default=[0.45, 0.0, 0.45, 0.10],
                    help="(p_u1, p_u2, p_red, p_syn) summing to 1.0")
    ap.add_argument("--lambda_kl", type=float, default=10.0)
    ap.add_argument("--lam_sparsity", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--smooth_window", type=int, default=51)
    return ap.parse_args()


def _print_summary(method: str, final: Dict[str, float]) -> None:
    print(f"  {method}:")
    print(f"    final lambda      : U1={final.get('lambda/U1_last', float('nan')):.1f}  "
          f"Red={final.get('lambda/Red_last', float('nan')):.1f}  "
          f"Syn={final.get('lambda/Syn_last', float('nan')):.1f}")
    print(f"    final share (l/Sum): U1={final.get('lambda_share/U1_last', float('nan')):.3f}  "
          f"Red={final.get('lambda_share/Red_last', float('nan')):.3f}  "
          f"Syn={final.get('lambda_share/Syn_last', float('nan')):.3f}")
    print(f"    final cos     : U1-Syn={final.get('cos/U1-Syn_last', float('nan')):+.3f}  "
          f"Red-Syn={final.get('cos/Red-Syn_last', float('nan')):+.3f}")
    print(f"    final train BCE: U1={final.get('train/U1_last', float('nan')):.3f}  "
          f"Red={final.get('train/Red_last', float('nan')):.3f}  "
          f"Syn={final.get('train/Syn_last', float('nan')):.3f}")
    print(f"    final val BCE  : U1={final.get('val/U1_last', float('nan')):.3f}  "
          f"Red={final.get('val/Red_last', float('nan')):.3f}  "
          f"Syn={final.get('val/Syn_last', float('nan')):.3f}")


def main() -> None:
    args = parse_args()
    simplex = tuple(args.simplex)
    cache_path = ART_DIR / "pid_ntk_2row.json"

    if args.mode == "render":
        results = json.loads(cache_path.read_text())
        if isinstance(results.get("simplex"), list):
            results["simplex"] = tuple(results["simplex"])
        final = render_figure(results, smooth_window=args.smooth_window)
        for method in ["vanilla", "mlearned"]:
            _print_summary(method, final[method])
        return

    if args.mode == "smoke":
        # Fast pipeline check: 5 epochs, 1 seed, both methods
        print(f"\n=== SMOKE: 5 epochs, 1 seed ({args.seeds[:1]}), both methods ===")
        results = run_experiments(simplex, args.seeds[:1], args.device,
                                  methods=["vanilla", "mlearned"],
                                  lambda_kl=args.lambda_kl,
                                  lam_sparsity=args.lam_sparsity, epochs=5)
        smoke_path = ART_DIR / "pid_ntk_2row_smoke.json"
        smoke_path.write_text(json.dumps(results, indent=2, default=float))
        final = render_figure(results, basename="pid_ntk_history_smoke",
                              smooth_window=11)
        for method in ["vanilla", "mlearned"]:
            _print_summary(method, final[method])
        return

    # Full mode
    print(f"\n=== Running PID-XOR NTK 2-row ===")
    print(f"   simplex={simplex}, seeds={args.seeds}, epochs={args.epochs}, "
          f"lambda_kl={args.lambda_kl}, lam_sparsity={args.lam_sparsity}, device={args.device}")
    t0 = time.time()
    results = run_experiments(simplex, args.seeds, args.device,
                              methods=["vanilla", "mlearned"],
                              lambda_kl=args.lambda_kl,
                              lam_sparsity=args.lam_sparsity,
                              epochs=args.epochs)
    print(f"\nTotal training time: {time.time() - t0:.1f}s")

    cache_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"Cached results to {cache_path}")

    final = render_figure(results, smooth_window=args.smooth_window)

    print("\n=== Final-step summary (mean across 3 seeds) ===")
    for method in ["vanilla", "mlearned"]:
        _print_summary(method, final[method])

    # Sanity flags vs expected behavior
    print("\n=== Expected-behavior checks (SynIB row) ===")
    syn = final["mlearned"]
    van = final["vanilla"]
    print(f"  [a] SynIB synergy lambda_g remains non-zero? "
          f"lambda/Syn = {syn['lambda/Syn_last']:.1f}  -> "
          f"{'PASS' if syn['lambda/Syn_last'] > 1.0 else 'FAIL'}")
    print(f"  [b] SynIB cos near zero (|cos| < 0.5)?  "
          f"cos/U1-Syn = {syn['cos/U1-Syn_last']:+.3f}, "
          f"cos/Red-Syn = {syn['cos/Red-Syn_last']:+.3f}  -> "
          f"{'PASS' if max(abs(syn['cos/U1-Syn_last']), abs(syn['cos/Red-Syn_last'])) < 0.5 else 'FAIL'}")
    syn_gap_synib = syn['val/Syn_last'] - syn['train/Syn_last']
    syn_gap_van   = van['val/Syn_last'] - van['train/Syn_last']
    print(f"  [c] Synergy val-train BCE gap closes under SynIB? "
          f"vanilla gap={syn_gap_van:+.3f} -> SynIB gap={syn_gap_synib:+.3f}  -> "
          f"{'PASS' if syn_gap_synib < syn_gap_van else 'FAIL'}")


if __name__ == "__main__":
    main()
