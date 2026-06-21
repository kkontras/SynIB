"""figs_training_dynamics.py
Generate appendix Figure 1 (PID-XOR per-source train/val loss dynamics, vanilla vs SynIB-learned)
and Figure 2 (synergy-restricted test accuracy over training, Spurious-XOR + PID-XOR).

Loads dataset/model classes from the existing scripts via importlib (avoiding their __main__),
adds the logging hooks the spec requires, runs experiments, caches results to JSON, and renders.

Usage:
    python scripts/analysis/figs_training_dynamics.py [--mode sanity|all] [--device cpu|cuda]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

# The PID3Main script imports `wandb` for optional run tracking but never inits
# it (only catches an exception inside try/except). Provide a no-op stub so we
# don't have to install it.
import types as _types
sys.modules.setdefault("wandb", _types.SimpleNamespace(run=None, init=lambda **kw: None))

# torch.isin was added in 1.10; the on-host install is 1.9.1. Polyfill so the
# PID3Main destroy_block helper works.
if not hasattr(torch, "isin"):
    def _isin(elements, test_elements):
        elements = elements.unsqueeze(-1)
        test_elements = test_elements.view(*([1] * (elements.dim() - 1)), -1)
        return (elements == test_elements).any(dim=-1)
    torch.isin = _isin  # type: ignore[attr-defined]

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "scripts" / "analysis"
ART_DIR = REPO / "artifacts" / "training_dynamics"
FIG_DIR = REPO / "docs" / "figures" / "training_dynamics"
ART_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# --- Source-keyed colors for Figure 1 (from Xor_PIDExamineNTK.py:503-509) ---
COLORS = {
    "u1":  "#2563EB",  # blue
    "red": "#16A34A",  # green
    "syn": "#DC2626",  # red
}
LABELS = {"u1": r"$U_1$", "red": "$R$", "syn": "$S$"}

# --- Method colors for Figure 2 (colorblind-safe per spec) ---
METHOD_COLORS = {
    "vanilla":       "#7a7a7a",  # gray
    "synib_oracle":  "#1f4e79",  # dark blue
    "synib_random":  "#2c9e9e",  # medium teal
    "synib_learned": "#d55e00",  # warm orange
}
METHOD_LABELS = {
    "vanilla":       "Vanilla",
    "synib_oracle":  r"SynIB $M^*$",
    "synib_random":  r"SynIB $M_\mathrm{Random}$",
    "synib_learned": r"SynIB $M_\mathrm{Learned}$",
}


def _apply_style() -> None:
    """Production style for NeurIPS appendix figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,           # embed TrueType (avoids NeurIPS font warnings)
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
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
    """Load script module while skipping its __main__ block."""
    src = path.read_text()
    # Truncate at the __main__ guard so its sweep code doesn't run on import.
    head = src.split('if __name__ == ')[0]
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)
    mod.__name__ = name  # dataclass introspection looks this up in sys.modules
    sys.modules[name] = mod
    exec(compile(head, str(path), 'exec'), mod.__dict__)
    return mod


def set_global_seed(seed: int) -> None:
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================================
# PID-XOR section
# =====================================================================================

def _per_source_loss_acc(model, loader, device: str) -> Dict[str, Dict[str, float]]:
    """Per-source mean BCE loss + accuracy on a loader (sources keyed 'u1','u2','red','syn')."""
    model.eval()
    sums = {k: {"loss_sum": 0.0, "acc_sum": 0.0, "n": 0} for k in ["u1", "u2", "red", "syn"]}
    with torch.no_grad():
        for b in loader:
            x0 = b["x0"].to(device); x1 = b["x1"].to(device); y = b["y"].to(device)
            src = b["source"].to(device)  # [B, 4] multi-hot order [u1, u2, red, syn]
            f, _, _ = model.forward_logits(x0, x1)
            losses = F.binary_cross_entropy_with_logits(f, y, reduction='none').view(-1)  # [B]
            preds = (f.view(-1) > 0).float()
            correct = (preds == y.view(-1)).float()
            for i, k in enumerate(["u1", "u2", "red", "syn"]):
                mask = src[:, i].bool()
                if mask.any():
                    sums[k]["loss_sum"] += float(losses[mask].sum().item())
                    sums[k]["acc_sum"]  += float(correct[mask].sum().item())
                    sums[k]["n"]        += int(mask.sum().item())
    out = {}
    for k, s in sums.items():
        n = max(1, s["n"])
        out[k] = {"loss": s["loss_sum"]/n, "acc": s["acc_sum"]/n, "n": s["n"]}
    return out


def _instrumented_train_pid(pid_mod, cfg, train_loader, val_loader, test_loader,
                            device: str, method: str,
                            lambda_kl: float = 10.0, lam_sparsity: float = 1.0) -> Dict[str, Any]:
    """Train one PID-XOR model with per-epoch per-source loss/acc on train, val, AND test.

    method ∈ {'vanilla', 'mstar', 'mrand', 'mlearned'}.
    Mirrors the loss structure of train_main / train_synib_mstar but adds the train-side hook.
    """
    model = pid_mod.FusionModel(cfg.dim0, cfg.dim1, cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = {"epochs": [], "train": [], "val": [], "test": []}

    for epoch in range(cfg.epochs):
        model.train()
        for b in train_loader:
            x0 = b["x0"].to(device); x1 = b["x1"].to(device); y = b["y"].to(device)
            m0 = b["mask0"].to(device); m1 = b["mask1"].to(device)
            f, u0, u1 = model.forward_logits(x0, x1)
            lf  = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            ltot = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            if method == "vanilla":
                pass
            elif method == "mstar":
                x0_t = pid_mod.destroy_block(x0, m0, pid_mod.DESTROY_MASK, noise_std=1.0)
                x1_t = pid_mod.destroy_block(x1, m1, pid_mod.DESTROY_MASK, noise_std=1.0)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                l_cf = pid_mod.bern_kl_to_uniform_from_logits(f_t0) + pid_mod.bern_kl_to_uniform_from_logits(f_t1)
                ltot = ltot + lambda_kl * l_cf
            elif method == "mrand":
                m0_r = (torch.rand_like(x0) < cfg.random_mask_proportion).float()
                m1_r = (torch.rand_like(x1) < cfg.random_mask_proportion).float()
                x0_t = pid_mod.destroy_block(x0, m0_r, 1, noise_std=1.0)
                x1_t = pid_mod.destroy_block(x1, m1_r, 1, noise_std=1.0)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                l_cf = pid_mod.bern_kl_to_uniform_from_logits(f_t0) + pid_mod.bern_kl_to_uniform_from_logits(f_t1)
                ltot = ltot + lambda_kl * l_cf
            elif method == "mlearned":
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
                l_cf = pid_mod.bern_kl_to_uniform_from_logits(f_t0) + pid_mod.bern_kl_to_uniform_from_logits(f_t1)
                ltot = ltot + lambda_kl * l_cf
            else:
                raise ValueError(f"Unknown method: {method}")

            opt.zero_grad(set_to_none=True)
            ltot.backward()
            opt.step()

        # ---- per-source eval at epoch end ----
        history["epochs"].append(epoch)
        history["train"].append(_per_source_loss_acc(model, train_loader, device))
        history["val"].append(_per_source_loss_acc(model, val_loader, device))
        history["test"].append(_per_source_loss_acc(model, test_loader, device))

    return history


def run_pid_experiments(simplex: Tuple[float, float, float, float], seeds: List[int],
                        device: str, methods: List[str],
                        lambda_kl: float, lam_sparsity: float,
                        epochs: int = 30) -> Dict[str, Any]:
    pid_mod = _load_module(ANALYSIS / "Xor_PID3Main_MaskSynIB_Search.py", "pid_mod")
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
            _, splits, train_l, val_l, test_l = pid_mod.build_loaders(cfg, verbose=False)
            hist = _instrumented_train_pid(pid_mod, cfg, train_l, val_l, test_l,
                                           device=device, method=method,
                                           lambda_kl=lambda_kl, lam_sparsity=lam_sparsity)
            elapsed = time.time() - t0
            print(f"[PID  {method:9s} seed={seed} simplex={simplex}] {elapsed:6.1f}s; "
                  f"final test syn acc = {hist['test'][-1]['syn']['acc']:.3f}")
            out["methods"][method].append(hist)
    return out


# =====================================================================================
# Spurious-XOR section
# =====================================================================================

def _instrumented_train_spur(spur_mod, cfg, params, alpha: float, seed: int, method: str,
                             device: str, log_train_acc: bool) -> Dict[str, Any]:
    """Train one Spurious-XOR run; log per-epoch val + test accuracy (+ optional train acc)."""
    spur_mod.set_seed(seed)
    train_l, val_l, test_l = spur_mod._make_loaders(alpha, seed, cfg, include_val=True)
    model = spur_mod._build_model(cfg, params)
    opt = spur_mod._build_optimizer(model, params)

    history = {"epochs": [], "train_acc": [], "val_acc": [], "test_acc": []}
    anneal = int(getattr(params, "l_anneal_epochs", 0))

    for ep in range(cfg.epochs):
        l_eff = params.lambda_kl * min(1.0, (ep + 1) / anneal) if anneal > 0 else params.lambda_kl
        model.train()
        for batch in train_l:
            x0 = batch["data"][0].to(device); x1 = batch["data"][1].to(device); y = batch["y"].to(device)
            _, _, _, l_base = spur_mod.base_loss(model, x0, x1, y, cfg.lambda_uni)
            if method == "baseline":
                loss = l_base
            elif method == "synib_star":
                x0_t = spur_mod.make_oracle_counterfactual(x0, cfg.signal_dims, cfg.comp_noise_std)
                x1_t = spur_mod.make_oracle_counterfactual(x1, cfg.signal_dims, cfg.comp_noise_std)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                loss = l_base + params.lambda_kl * (spur_mod.kl_to_uniform(f_t0) + spur_mod.kl_to_uniform(f_t1))
            elif method == "synib_random":
                B = x0.size(0)
                g0 = torch.bernoulli(torch.full((B, cfg.dim0), params.mask_prob, device=device))
                g1 = torch.bernoulli(torch.full((B, cfg.dim1), params.mask_prob, device=device))
                x0_t = spur_mod.apply_destroy(x0, g0, cfg.comp_noise_std)
                x1_t = spur_mod.apply_destroy(x1, g1, cfg.comp_noise_std)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                loss = l_base + params.lambda_kl * (spur_mod.kl_to_uniform(f_t0) + spur_mod.kl_to_uniform(f_t1))
            elif method == "synib_learned":
                with torch.enable_grad():
                    g0 = spur_mod.learn_one_gate(model, x0.detach(), x1.detach(), y.detach(), "x0", params, cfg)
                    g1 = spur_mod.learn_one_gate(model, x0.detach(), x1.detach(), y.detach(), "x1", params, cfg)
                x0_t = spur_mod.apply_destroy(x0, 1.0 - g0, cfg.mask_noise_std)
                x1_t = spur_mod.apply_destroy(x1, 1.0 - g1, cfg.mask_noise_std)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                loss = l_base + l_eff * (spur_mod.kl_to_uniform(f_t0) + spur_mod.kl_to_uniform(f_t1))
            else:
                raise ValueError(f"Unknown method: {method}")
            opt.zero_grad(); loss.backward(); opt.step()

        history["epochs"].append(ep)
        history["val_acc"].append(spur_mod.eval_acc(model, val_l, device))
        history["test_acc"].append(spur_mod.eval_acc(model, test_l, device))
        if log_train_acc:
            history["train_acc"].append(spur_mod.eval_acc(model, train_l, device))
    return history


def run_spurious_experiments(alpha: float, seeds: List[int], device: str,
                             methods: List[str], epochs: int = 100) -> Dict[str, Any]:
    spur_mod = _load_module(ANALYSIS / "xor_spurious_pub.py", "spur_mod")
    out = {"alpha": alpha, "epochs": epochs, "seeds": seeds, "methods": {}}
    for method in methods:
        out["methods"][method] = []
        params = copy.deepcopy(spur_mod.ARCHIVED_BEST[method])
        for seed in seeds:
            t0 = time.time()
            cfg = spur_mod.Cfg(epochs=epochs, device=device)
            log_train = (method == "baseline")  # only vanilla gets train-acc curve
            hist = _instrumented_train_spur(spur_mod, cfg, params, alpha, seed, method, device, log_train)
            elapsed = time.time() - t0
            print(f"[SPUR {method:14s} seed={seed} alpha={alpha}] {elapsed:6.1f}s; "
                  f"final test acc = {hist['test_acc'][-1]:.3f}")
            out["methods"][method].append(hist)
    return out


# =====================================================================================
# Plotting
# =====================================================================================

def _stack_per_source(method_runs: List[Dict[str, Any]], split: str, source: str, key: str) -> np.ndarray:
    """Return [n_seeds, n_epochs] array of `key` for given split+source."""
    arr = []
    for run in method_runs:
        arr.append([epoch_d[source][key] for epoch_d in run[split]])
    return np.asarray(arr, dtype=float)


def _mean_sem(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard error of the mean (std/sqrt(N)) over axis 0 (seeds)."""
    n = max(1, stack.shape[0])
    return stack.mean(axis=0), stack.std(axis=0) / np.sqrt(n)


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


def _annotation_box(ax, *, right: float = 0.985, top: float = 0.50,
                    width: float = 0.30, height: float = 0.36) -> None:
    """Draw a semi-transparent white box in axes-fraction coordinates,
    used as a background for the side annotation block."""
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (right - width, top - height), width, height,
        transform=ax.transAxes,
        facecolor="white", edgecolor="#cccccc", linewidth=0.5,
        alpha=0.78, zorder=4, clip_on=False,
    ))


def _save(fig, basename: str) -> None:
    pdf = FIG_DIR / f"{basename}.pdf"
    png = FIG_DIR / f"{basename}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"Saved {pdf} and {png}")


def render_figure_1(pid_results: Dict[str, Any], out_basename: str) -> None:
    """Per-source train (dashed) and val (solid) accuracy across 4 methods.
    Final held-out test accuracy annotated in lower-right of each panel.
    Bands = standard error of mean.
    """
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    sources = ["u1", "red", "syn"]
    methods = [
        ("vanilla",  "Vanilla fusion"),
        ("mstar",    r"SynIB $M^*$"),
        ("mrand",    r"SynIB $M_{\mathrm{Random}}$"),
        ("mlearned", r"SynIB $M_{\mathrm{Learned}}$"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.4), sharey=True,
                             constrained_layout=False)
    dash = (0, (3, 2))
    for (method, title), ax in zip(methods, axes):
        runs = pid_results["methods"][method]
        epochs = np.asarray(runs[0]["epochs"])
        annot_lines = []
        for src in sources:
            tm, te = _mean_sem(_stack_per_source(runs, "train", src, "acc"))
            vm, ve = _mean_sem(_stack_per_source(runs, "val",   src, "acc"))
            # Train = dashed, val = solid (per user spec)
            ax.plot(epochs, tm, color=COLORS[src], linestyle=dash, lw=1.4, zorder=3,
                    label=f"{LABELS[src]} train")
            ax.fill_between(epochs, tm-te, tm+te,
                            color=COLORS[src], alpha=0.12, edgecolor="none", zorder=2)
            ax.plot(epochs, vm, color=COLORS[src], linestyle="-", lw=1.6, zorder=3,
                    label=f"{LABELS[src]} val")
            ax.fill_between(epochs, vm-ve, vm+ve,
                            color=COLORS[src], alpha=0.18, edgecolor="none", zorder=2)
            test_final = _stack_per_source(runs, "test", src, "acc").mean(axis=0)[-1]
            annot_lines.append((src, float(test_final)))
        ax.axhline(0.5, color="#999999", lw=0.6, linestyle=(0, (1, 2)), zorder=1)
        # Right-side annotation (empty band between vanilla S at chance and others at 1.0)
        _annotation_box(ax, right=0.985, top=0.50, width=0.32, height=0.34)
        ax.text(0.97, 0.45, "Test acc", transform=ax.transAxes,
                ha="right", va="top", fontsize=7.0, color="#222222",
                fontweight="bold", zorder=5)
        for i, (src, val_) in enumerate(annot_lines):
            ax.text(0.97, 0.38 - i*0.07,
                    f"{LABELS[src]}: {val_:.2f}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7.0, color=COLORS[src], zorder=5)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Epoch")
        ax.set_ylim(0.4, 1.04)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        _style_axes(ax)
    axes[0].set_ylabel("Per-source accuracy")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False,
               handlelength=2.0, columnspacing=1.2, handletextpad=0.4)
    plt.subplots_adjust(left=0.06, right=0.995, top=0.78, bottom=0.20, wspace=0.06)
    _save(fig, out_basename)
    plt.close(fig)


def render_figure_2(spur_results: Dict[str, Any], pid_results: Dict[str, Any],
                    out_basename: str) -> None:
    """Curves: validation (PID-XOR) and out-of-distribution test (Spurious-XOR has no val).
    Final test accuracy annotated on the right side of each panel."""
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, 2, figsize=(6.75, 2.6), constrained_layout=False)
    dash = (0, (3, 2))

    # --- Left: Spurious XOR (test = OOD probe; no val concept here) ---
    ax = axes[0]
    spur_color = {
        "baseline":       METHOD_COLORS["vanilla"],
        "synib_star":     METHOD_COLORS["synib_oracle"],
        "synib_random":   METHOD_COLORS["synib_random"],
        "synib_learned":  METHOD_COLORS["synib_learned"],
    }
    method_order = ["baseline", "synib_star", "synib_random", "synib_learned"]
    annot_lines_spur = []
    for method in method_order:
        runs = spur_results["methods"][method]
        epochs = np.asarray(runs[0]["epochs"])
        # Curve: VAL accuracy (in-distribution validation; spur=y in train+val)
        val = np.asarray([r["val_acc"] for r in runs]) if "val_acc" in runs[0] else \
              np.asarray([r["test_acc"] for r in runs])
        m, e = _mean_sem(val)
        c = spur_color[method]
        ax.plot(epochs, m, color=c, lw=1.6, zorder=3)
        ax.fill_between(epochs, m-e, m+e, color=c, alpha=0.18, edgecolor="none", zorder=2)
        if method == "baseline" and runs[0].get("train_acc"):
            tr = np.asarray([r["train_acc"] for r in runs])
            tm, te = _mean_sem(tr)
            ax.plot(epochs, tm, color=c, lw=1.2, linestyle=dash, zorder=3)
            ax.fill_between(epochs, tm-te, tm+te, color=c, alpha=0.10, edgecolor="none", zorder=2)
        # Final test acc annotation = OOD test (spur decorrelated)
        test_final = np.asarray([r["test_acc"][-1] for r in runs]).mean()
        annot_lines_spur.append((method, float(test_final)))
    ax.axhline(0.5, color="#999999", lw=0.6, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val accuracy")
    ax.set_ylim(0.4, 1.04); ax.set_title("Spurious XOR")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    _style_axes(ax)
    # All val curves saturate at 1.0 by epoch ~60; bottom-right at y=0.05-0.40 is empty
    _annotation_box(ax, right=0.985, top=0.44, width=0.28, height=0.36)
    ax.text(0.98, 0.40, "Final test acc", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#222222",
            fontweight="bold", zorder=5)
    for i, (m, v) in enumerate(annot_lines_spur):
        c = spur_color[m]
        short = {"baseline": "Vanilla", "synib_star": "$M^*$",
                 "synib_random": "$M_{\\mathrm{Random}}$", "synib_learned": "$M_{\\mathrm{Learned}}$"}[m]
        ax.text(0.98, 0.33 - i*0.06,
                f"{short}: {v:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=c, zorder=5)

    # --- Right: PID-XOR (validation curves, synergy subset) ---
    ax = axes[1]
    pid_color = {
        "vanilla":   METHOD_COLORS["vanilla"],
        "mstar":     METHOD_COLORS["synib_oracle"],
        "mrand":     METHOD_COLORS["synib_random"],
        "mlearned":  METHOD_COLORS["synib_learned"],
    }
    pid_method_order = ["vanilla", "mstar", "mrand", "mlearned"]
    annot_lines_pid = []
    for method in pid_method_order:
        runs = pid_results["methods"][method]
        epochs = np.asarray(runs[0]["epochs"])
        # Curve: VAL synergy accuracy (in-distribution)
        val_syn = np.asarray([[ep["syn"]["acc"] for ep in r["val"]] for r in runs])
        m, e = _mean_sem(val_syn)
        c = pid_color[method]
        ax.plot(epochs, m, color=c, lw=1.6, zorder=3)
        ax.fill_between(epochs, m-e, m+e, color=c, alpha=0.18, edgecolor="none", zorder=2)
        # Annotation: final TEST synergy accuracy (held out)
        test_syn = np.asarray([[ep["syn"]["acc"] for ep in r["test"]] for r in runs]).mean(0)
        annot_lines_pid.append((method, float(test_syn[-1])))
    ax.axhline(0.5, color="#999999", lw=0.6, linestyle=(0, (1, 2)), zorder=1)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Val synergy accuracy")
    ax.set_ylim(0.4, 1.0); ax.set_title("PID-Controlled XOR")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
    _style_axes(ax)
    # Middle-right: vanilla sits at y=0.5 (axes 0.17), others saturate at ~0.85+
    # (axes 0.75+). Place block in the empty band 0.30-0.65.
    _annotation_box(ax, right=0.985, top=0.55, width=0.28, height=0.36)
    ax.text(0.98, 0.51, "Final test acc", transform=ax.transAxes,
            ha="right", va="top", fontsize=7.5, color="#222222",
            fontweight="bold", zorder=5)
    for i, (m, v) in enumerate(annot_lines_pid):
        c = pid_color[m]
        short = {"vanilla": "Vanilla", "mstar": "$M^*$",
                 "mrand": "$M_{\\mathrm{Random}}$", "mlearned": "$M_{\\mathrm{Learned}}$"}[m]
        ax.text(0.98, 0.44 - i*0.06,
                f"{short}: {v:.3f}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7.5, color=c, zorder=5)

    # Single legend below: one entry per method + the vanilla-train (Spurious only) marker
    legend_handles = [
        Line2D([0], [0], color=METHOD_COLORS["vanilla"], lw=1.6,    label=METHOD_LABELS["vanilla"]),
        Line2D([0], [0], color=METHOD_COLORS["synib_oracle"],  lw=1.6, label=METHOD_LABELS["synib_oracle"]),
        Line2D([0], [0], color=METHOD_COLORS["synib_random"],  lw=1.6, label=METHOD_LABELS["synib_random"]),
        Line2D([0], [0], color=METHOD_COLORS["synib_learned"], lw=1.6, label=METHOD_LABELS["synib_learned"]),
        Line2D([0], [0], color=METHOD_COLORS["vanilla"], lw=1.2, linestyle=dash,
               label="Vanilla train (Spurious)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=5, frameon=False,
               handlelength=2.0, columnspacing=1.2, handletextpad=0.4)
    plt.subplots_adjust(left=0.085, right=0.99, top=0.90, bottom=0.24, wspace=0.20)
    _save(fig, out_basename)
    plt.close(fig)


# =====================================================================================
# Main
# =====================================================================================

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sanity", "all", "render", "fig1_lambda"], default="all")
    ap.add_argument("--fig1_lambda_kl", type=float, default=1.0,
                    help="lambda_kl for Figure 1 (BCE-loss panels). Decoupled from --lambda_kl which drives Figure 2.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--simplex", nargs=4, type=float, default=[0.45, 0.0, 0.45, 0.10],
                    help="(p_u1, p_u2, p_red, p_syn) summing to 1.0")
    ap.add_argument("--lambda_kl", type=float, default=10.0)
    ap.add_argument("--lam_sparsity", type=float, default=1.0)
    ap.add_argument("--pid_epochs", type=int, default=30)
    ap.add_argument("--spur_epochs", type=int, default=100)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    simplex = tuple(args.simplex)

    if args.mode == "render":
        # Re-render from cached JSON without re-running anything.
        # Figure 1 prefers pid_fig1.json (decoupled lambda) when present, else falls back to pid_full.json.
        fig1_path = ART_DIR / "pid_fig1.json"
        pid_full_path = ART_DIR / "pid_full.json"
        pid_fig1 = json.loads(fig1_path.read_text()) if fig1_path.exists() else json.loads(pid_full_path.read_text())
        pid_full = json.loads(pid_full_path.read_text())
        spur_full = json.loads((ART_DIR / "spurious_full.json").read_text())
        for d in (pid_fig1, pid_full):
            if isinstance(d.get("simplex"), list):
                d["simplex"] = tuple(d["simplex"])
        render_figure_1(pid_fig1, "pid_xor_training_dynamics")
        render_figure_2(spur_full, pid_full, "synergy_acc_over_training")
        return

    if args.mode == "fig1_lambda":
        # Re-run only mlearned at the chosen lambda (vanilla doesn't depend on it; reuse cache).
        fig1_lambda = args.fig1_lambda_kl
        cached = json.loads((ART_DIR / "pid_full.json").read_text())
        if isinstance(cached.get("simplex"), list):
            cached["simplex"] = tuple(cached["simplex"])
        print(f"\n=== Figure 1 re-run: mlearned at lambda_kl={fig1_lambda} (vanilla cached) ===")
        new_ml = run_pid_experiments(simplex, args.seeds, args.device,
                                     methods=["mlearned"],
                                     lambda_kl=fig1_lambda,
                                     lam_sparsity=args.lam_sparsity,
                                     epochs=args.pid_epochs)
        pid_fig1 = {
            "simplex": list(simplex),
            "lambda_kl": fig1_lambda,
            "lam_sparsity": args.lam_sparsity,
            "epochs": args.pid_epochs,
            "seeds": args.seeds,
            "methods": {
                "vanilla": cached["methods"]["vanilla"],
                "mlearned": new_ml["methods"]["mlearned"],
            },
        }
        (ART_DIR / "pid_fig1.json").write_text(json.dumps(pid_fig1, indent=2, default=float))
        pid_fig1["simplex"] = tuple(pid_fig1["simplex"])
        render_figure_1(pid_fig1, "pid_xor_training_dynamics")
        return

    # Gap check (sanity step in the spec): run vanilla only, look at synergy train vs test loss
    sanity_path = ART_DIR / "pid_vanilla_sanity.json"
    print(f"\n=== STEP 1: vanilla sanity check at simplex={simplex} ===")
    pid_vanilla = run_pid_experiments(simplex, args.seeds, args.device,
                                      methods=["vanilla"],
                                      lambda_kl=args.lambda_kl,
                                      lam_sparsity=args.lam_sparsity,
                                      epochs=args.pid_epochs)
    sanity_path.write_text(json.dumps(pid_vanilla, indent=2, default=float))

    # Compute the synergy gap
    runs = pid_vanilla["methods"]["vanilla"]
    syn_train_loss = np.mean([[ep["syn"]["loss"] for ep in r["train"]] for r in runs], axis=0)
    syn_test_loss  = np.mean([[ep["syn"]["loss"] for ep in r["test"]]  for r in runs], axis=0)
    syn_train_acc  = np.mean([[ep["syn"]["acc"]  for ep in r["train"]] for r in runs], axis=0)
    syn_test_acc   = np.mean([[ep["syn"]["acc"]  for ep in r["test"]]  for r in runs], axis=0)
    gap_loss = float(syn_test_loss[-1] - syn_train_loss[-1])
    gap_acc  = float(syn_train_acc[-1] - syn_test_acc[-1])
    print(f"\n[sanity] synergy train→test loss gap at last epoch: "
          f"train={syn_train_loss[-1]:.3f} test={syn_test_loss[-1]:.3f} (gap={gap_loss:+.3f})")
    print(f"[sanity] synergy train→test acc gap: "
          f"train={syn_train_acc[-1]:.3f} test={syn_test_acc[-1]:.3f} (gap={gap_acc:+.3f})")
    print(f"[sanity] synergy train loss trajectory: "
          f"first={syn_train_loss[0]:.3f}, mid={syn_train_loss[len(syn_train_loss)//2]:.3f}, last={syn_train_loss[-1]:.3f}")
    print(f"[sanity] synergy test  loss trajectory: "
          f"first={syn_test_loss[0]:.3f}, mid={syn_test_loss[len(syn_test_loss)//2]:.3f}, last={syn_test_loss[-1]:.3f}")

    if args.mode == "sanity":
        print("\nSanity-only mode; stopping. Inspect the trajectories above before running full mode.")
        return

    # Full mode: run all SynIB variants on PID-XOR + Spurious-XOR
    print(f"\n=== STEP 2: PID-XOR full sweep (mstar, mrand, mlearned) ===")
    pid_full = run_pid_experiments(simplex, args.seeds, args.device,
                                   methods=["mstar", "mrand", "mlearned"],
                                   lambda_kl=args.lambda_kl,
                                   lam_sparsity=args.lam_sparsity,
                                   epochs=args.pid_epochs)
    # Merge vanilla into full results
    pid_full["methods"]["vanilla"] = pid_vanilla["methods"]["vanilla"]
    (ART_DIR / "pid_full.json").write_text(json.dumps(pid_full, indent=2, default=float))

    print(f"\n=== STEP 3: Spurious-XOR sweep at alpha={args.alpha} ===")
    spur_full = run_spurious_experiments(args.alpha, args.seeds, args.device,
                                         methods=["baseline", "synib_star", "synib_random", "synib_learned"],
                                         epochs=args.spur_epochs)
    (ART_DIR / "spurious_full.json").write_text(json.dumps(spur_full, indent=2, default=float))

    print(f"\n=== STEP 4: rendering figures ===")
    render_figure_1(pid_full, "pid_xor_training_dynamics")
    render_figure_2(spur_full, pid_full, "synergy_acc_over_training")
    print("\nDone.")


if __name__ == "__main__":
    main()
