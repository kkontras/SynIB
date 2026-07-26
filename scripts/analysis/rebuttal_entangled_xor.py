"""rebuttal_entangled_xor.py — Entangled PID-XOR ablation (Reviewer RpxH, W5).

Tests whether SynIB's learned masking survives when no axis-aligned sparse
structure exists: inputs are rotated by a fixed random orthogonal matrix Q per
modality, applied AFTER block construction and BEFORE standardization
(opt-in hook in Xor_PID3Main_MaskSynIB_Search.PID4BlockDataset).

Experiment 1 (rotation sweep): rotation in {identity, full, partial};
methods vanilla / mstar (identity only) / mstar_srcbasis (oracle in source
basis, rotated arms) / mrand / mlearned; per-source train/val/test curves per
epoch as in paper Fig. 8. Paper hyperparameters throughout
(lambda_kl=10, lam_sparsity=1.0, inner loop 20 Adam steps lr=0.1 tau=1.0).

Experiment 2 (mask-oracle overlap, identity arm): per epoch, learn gates on a
fixed held-out val batch and log IoU / precision / recall of the predicted
unimodal support (kept coordinates) vs the ground-truth unique+redundant block
positions, plus a Bernoulli(0.5) random-mask baseline.

Usage:
  python scripts/analysis/rebuttal_entangled_xor.py run --config run/configs/rebuttal_entangled_xor/identity.json
  python scripts/analysis/rebuttal_entangled_xor.py render
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# wandb stub (pid module imports it)
import types as _types
sys.modules.setdefault("wandb", _types.SimpleNamespace(run=None, init=lambda **kw: None))

REPO = Path(__file__).resolve().parents[2]
ANALYSIS = REPO / "scripts" / "analysis"
ART_DIR = REPO / "artifacts" / "rebuttal_entangled_xor"
RUN_DIR = ART_DIR / "runs"
Q_DIR = ART_DIR / "Q"
MIXER_DIR = ART_DIR / "mixers"
FIG_DIR = REPO / "docs" / "figures" / "rebuttal_entangled_xor"
for d in (ART_DIR, RUN_DIR, Q_DIR, MIXER_DIR, FIG_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _load_module(path: Path, name: str):
    """Load a script module while skipping its __main__ block."""
    src = path.read_text()
    head = src.split('if __name__ == ')[0]
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(path)
    mod.__name__ = name
    sys.modules[name] = mod
    exec(compile(head, str(path), 'exec'), mod.__dict__)
    return mod


def set_global_seed(seed: int) -> None:
    import random as _random
    _random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================================
# Rotations
# =====================================================================================

def _canonical_orthogonal(A: torch.Tensor) -> torch.Tensor:
    """QR with sign fix so the result is deterministic and canonical."""
    Q, R = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(R)).unsqueeze(0)
    return Q


def _full_rotation(dim: int, gen_seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(gen_seed)
    A = torch.randn(dim, dim, generator=g, dtype=torch.float64)
    return _canonical_orthogonal(A).to(torch.float32)


def _block_index_sets(pid_mod, dim: int, cfg) -> List[torch.Tensor]:
    """Contiguous block coordinate sets [unique, red, syn, noise] (paper layout,
    random_block_positions=False)."""
    u, r, s, _ = pid_mod._block_sizes(dim, cfg.frac_unique, cfg.frac_red, cfg.frac_syn)
    return [torch.arange(0, u),
            torch.arange(u, u + r),
            torch.arange(u + r, u + r + s),
            torch.arange(u + r + s, dim)]


def _partial_rotation(dim: int, gen_seed: int, blocks: List[torch.Tensor]) -> Tuple[torch.Tensor, List[List[int]]]:
    """Block-diagonal partial rotation: randomly pair the 4 blocks into 2 pairs,
    rotate within the union of each pair (intermediate entanglement)."""
    g = torch.Generator().manual_seed(gen_seed)
    perm = torch.randperm(4, generator=g).tolist()
    pairs = [[perm[0], perm[1]], [perm[2], perm[3]]]
    Q = torch.eye(dim, dtype=torch.float32)
    for a, b in pairs:
        idx = torch.cat([blocks[a], blocks[b]])
        k = idx.numel()
        A = torch.randn(k, k, generator=g, dtype=torch.float64)
        q = _canonical_orthogonal(A).to(torch.float32)
        Q[idx.unsqueeze(1), idx.unsqueeze(0)] = q
    return Q, pairs


def get_rotation(pid_mod, rotation: str, rot_seed: int, mod_idx: int, dim: int, cfg) -> Optional[torch.Tensor]:
    """Build (or load from disk) the fixed rotation matrix for one modality.
    Saved to Q_DIR so runs are exactly reproducible."""
    if rotation == "identity":
        return None
    path = Q_DIR / f"Q_{rotation}_rotseed{rot_seed}_mod{mod_idx}_d{dim}.npy"
    meta_path = Q_DIR / f"Q_{rotation}_rotseed{rot_seed}_mod{mod_idx}_d{dim}.json"
    if path.exists():
        return torch.from_numpy(np.load(path))
    gen_seed = 7700 + 100 * rot_seed + mod_idx
    meta = {"rotation": rotation, "rot_seed": rot_seed, "mod_idx": mod_idx,
            "dim": dim, "gen_seed": gen_seed}
    if rotation == "full":
        Q = _full_rotation(dim, gen_seed)
    elif rotation == "partial":
        blocks = _block_index_sets(pid_mod, dim, cfg)
        Q, pairs = _partial_rotation(dim, gen_seed, blocks)
        meta["block_pairs"] = pairs
    else:
        raise ValueError(f"Unknown rotation: {rotation}")
    err = float((Q @ Q.T - torch.eye(dim)).abs().max().item())
    assert err < 1e-5, f"Q not orthogonal (max err {err})"
    np.save(path, Q.numpy())
    meta_path.write_text(json.dumps(meta, indent=2))
    return Q


# =====================================================================================
# Frozen random MLP mixer (rebuttal Arm A)
# =====================================================================================

MIXER_HIDDEN = 64  # x in R^32 -> tanh(W1 x) in R^64 -> W2 back to R^32


def _raw_train_inputs(pid_mod, cfg) -> Tuple[torch.Tensor, torch.Tensor]:
    """Raw (pre-standardization) identity-basis train inputs for gain calibration.
    Uses data seed 0 regardless of the training seed so the mixer is one fixed
    object shared by every run."""
    cal_cfg = copy.deepcopy(cfg)
    cal_cfg.seed = 0
    for attr in ("rotation_Q0", "rotation_Q1", "rotation_nonlinearity",
                 "mlp_mixer0", "mlp_mixer1"):
        if hasattr(cal_cfg, attr):
            delattr(cal_cfg, attr)
    set_global_seed(0)
    _, _, train_l, _, _ = pid_mod.build_loaders(cal_cfg, verbose=False)
    ds = train_l.dataset.dataset  # Subset -> PID4BlockDataset
    idx = train_l.dataset.indices
    raw0 = ds.x0[idx] * ds.stats["x0"]["s"] + ds.stats["x0"]["m"]
    raw1 = ds.x1[idx] * ds.stats["x1"]["s"] + ds.stats["x1"]["m"]
    return raw0, raw1


def get_mlp_mixer(pid_mod, cfg, mixer_seed: int, mod_idx: int, dim: int,
                  target_preact_std: float) -> Dict[str, Any]:
    """Build (or load) one frozen per-modality MLP mixer x <- W2 tanh(W1 x).
    W1 entries are i.i.d. Gaussian with gain calibrated so tanh pre-activations
    have std ~= target_preact_std on the seed-0 training data. Saved to disk with
    full provenance; identical across splits/methods/training seeds."""
    stem = f"mlp_mixer_seed{mixer_seed}_mod{mod_idx}_d{dim}_h{MIXER_HIDDEN}_t{target_preact_std:g}"
    npz_path = MIXER_DIR / f"{stem}.npz"
    meta_path = MIXER_DIR / f"{stem}.json"
    if npz_path.exists():
        z = np.load(npz_path)
        return {"W1": torch.from_numpy(z["W1"]), "W2": torch.from_numpy(z["W2"]),
                "meta_file": meta_path.name, "npz_file": npz_path.name,
                "realized_preact_std": float(json.loads(meta_path.read_text())["realized_preact_std"])}
    gen_seed = 8800 + 100 * mixer_seed + mod_idx
    g = torch.Generator().manual_seed(gen_seed)
    W1_raw = torch.randn(MIXER_HIDDEN, dim, generator=g)
    W2 = torch.randn(dim, MIXER_HIDDEN, generator=g) / np.sqrt(MIXER_HIDDEN)
    raw0, raw1 = _raw_train_inputs(pid_mod, cfg)
    raw = raw0 if mod_idx == 0 else raw1
    pre_raw_std = float((raw @ W1_raw.T).std().item())
    W1 = W1_raw * (target_preact_std / pre_raw_std)
    realized = float((raw @ W1.T).std().item())
    np.savez(npz_path, W1=W1.numpy(), W2=W2.numpy())
    meta_path.write_text(json.dumps({
        "mixer_seed": mixer_seed, "mod_idx": mod_idx, "dim": dim,
        "hidden": MIXER_HIDDEN, "gen_seed": gen_seed,
        "target_preact_std": target_preact_std,
        "realized_preact_std": realized,
        "calibration": "seed-0 identity train split, raw (pre-standardization) inputs",
    }, indent=2))
    print(f"[mixer] built {stem}: realized pre-act std = {realized:.3f}")
    return {"W1": W1, "W2": W2, "meta_file": meta_path.name, "npz_file": npz_path.name,
            "realized_preact_std": realized}


# =====================================================================================
# Eval helpers
# =====================================================================================

def _per_source_loss_acc(model, loader, device: str) -> Dict[str, Dict[str, float]]:
    """Per-source mean BCE loss + accuracy, plus overall ('_overall')."""
    model.eval()
    sums = {k: {"loss_sum": 0.0, "acc_sum": 0.0, "n": 0}
            for k in ["u1", "u2", "red", "syn", "_overall"]}
    with torch.no_grad():
        for b in loader:
            x0 = b["x0"].to(device); x1 = b["x1"].to(device); y = b["y"].to(device)
            src = b["source"].to(device)
            f, _, _ = model.forward_logits(x0, x1)
            losses = F.binary_cross_entropy_with_logits(f, y, reduction='none').view(-1)
            preds = (f.view(-1) > 0).float()
            correct = (preds == y.view(-1)).float()
            sums["_overall"]["loss_sum"] += float(losses.sum().item())
            sums["_overall"]["acc_sum"] += float(correct.sum().item())
            sums["_overall"]["n"] += int(y.numel())
            for i, k in enumerate(["u1", "u2", "red", "syn"]):
                mask = src[:, i].bool()
                if mask.any():
                    sums[k]["loss_sum"] += float(losses[mask].sum().item())
                    sums[k]["acc_sum"] += float(correct[mask].sum().item())
                    sums[k]["n"] += int(mask.sum().item())
    out = {}
    for k, s in sums.items():
        n = max(1, s["n"])
        out[k] = {"loss": s["loss_sum"] / n, "acc": s["acc_sum"] / n, "n": s["n"]}
    return out


def _support_metrics(g: torch.Tensor, gt_support: torch.Tensor) -> Dict[str, float]:
    """Agreement of predicted unimodal support (kept coords, g==0) with the
    ground-truth unique+redundant block positions. Per-example, then averaged.
    Accepts soft gates (keep-prob semantics): hard metrics use a 0.5 threshold;
    auroc measures whether the SOFT gate ranks support coords as more
    destruction-relevant (lower keep-prob) than non-support coords."""
    if g.dim() == 1:
        g = g.unsqueeze(0)
    pred = (g < 0.5)
    gt = gt_support.to(pred.device)
    if gt.dim() == 1:
        gt = gt.unsqueeze(0).expand_as(pred)
    inter = (pred & gt).sum(1).float()
    union = (pred | gt).sum(1).float()
    psz = pred.sum(1).float()
    gsz = gt.sum(1).float()
    # per-example rank AUROC of (1 - g) for support membership (Mann-Whitney)
    scores = (1.0 - g).float()
    ranks = scores.argsort(dim=1).argsort(dim=1).float() + 1.0
    n_pos = gt.sum(1).float()
    n_neg = (~gt).sum(1).float()
    r_pos = (ranks * gt.float()).sum(1)
    u = r_pos - n_pos * (n_pos + 1) / 2
    auroc = (u / (n_pos * n_neg).clamp(min=1)).mean()
    return {
        "iou": float((inter / union.clamp(min=1)).mean().item()),
        "precision": float((inter / psz.clamp(min=1)).mean().item()),
        "recall": float((inter / gsz.clamp(min=1)).mean().item()),
        "destroy_frac": float((g >= 0.5).float().mean().item()),
        "auroc": float(auroc.item()),
    }


def mask_oracle_diagnostic(pid_mod, model, cfg, fixed_batch, gt_support0, gt_support1,
                           device: str, lam_sparsity: float, epoch: int) -> Dict[str, Any]:
    """Experiment 2: learn gates on the fixed held-out batch with the CURRENT
    model (paper mask HPs) and measure agreement with the oracle support."""
    x0 = fixed_batch["x0"].to(device); x1 = fixed_batch["x1"].to(device)
    y = fixed_batch["y"].to(device)
    m0 = fixed_batch["mask0"].to(device); m1 = fixed_batch["mask1"].to(device)
    was_training = model.training
    model.eval()
    with torch.enable_grad():
        masks = pid_mod.learn_destroy_gates(
            model=model, x0=x0.detach(), x1=x1.detach(), y=y.detach(),
            method=cfg.learned_mask_method, device=device,
            gate_shape=cfg.learned_mask_gate_shape,
            steps=cfg.learned_mask_steps, lr=cfg.learned_mask_lr,
            tau=cfg.learned_mask_tau, noise_std=cfg.learned_mask_noise_std,
            lam_sparsity=lam_sparsity,
            alpha_unimodal=cfg.learned_mask_alpha_unimodal,
            hard=False,  # keep gates SOFT for the ranking (AUROC) diagnostic;
            hard_thresh=cfg.learned_mask_hard_thresh,  # hard metrics threshold at 0.5 inside _support_metrics
            m0=m0.detach(), m1=m1.detach(), iou_target="syn",
            print_every=10**9, label="",
        )
    if was_training:
        model.train()
    g0, g1 = masks["g0"].detach(), masks["g1"].detach()
    out = {"epoch": epoch,
           "mod0": _support_metrics(g0, gt_support0),
           "mod1": _support_metrics(g1, gt_support1)}
    # Random-mask baseline: Bernoulli(0.5) destroy pattern, same shapes,
    # deterministic per epoch.
    g_rand = torch.Generator(device="cpu").manual_seed(4242 + epoch)
    r0 = (torch.rand(g0.shape, generator=g_rand) < 0.5).float().to(device)
    r1 = (torch.rand(g1.shape, generator=g_rand) < 0.5).float().to(device)
    out["rand0"] = _support_metrics(r0, gt_support0)
    out["rand1"] = _support_metrics(r1, gt_support1)
    return out


# =====================================================================================
# Training (mirrors figs_training_dynamics._instrumented_train_pid + extensions)
# =====================================================================================

def destroy_syn_source_basis(pid_mod, x_std, mask_block, Q, mean, std, noise_std=1.0):
    """Oracle in SOURCE basis under rotation: unstandardize, rotate back to the
    source basis, destroy the synergistic block there, re-rotate, re-standardize."""
    x_rot = x_std * std + mean
    x_src = x_rot @ Q          # rows: x_rot = x_src @ Q.T  =>  x_src = x_rot @ Q
    x_src_t = pid_mod.destroy_block(x_src, mask_block, pid_mod.DESTROY_MASK, noise_std=noise_std)
    x_rot_t = x_src_t @ Q.T
    return (x_rot_t - mean) / std


def train_one(pid_mod, cfg, train_loader, val_loader, test_loader, device: str,
              method: str, lambda_kl: float, lam_sparsity: float,
              rotation_ctx: Optional[Dict[str, Any]] = None,
              mask_diag_cfg: Optional[Dict[str, Any]] = None,
              destroy_fix: bool = False) -> Dict[str, Any]:
    """Train one PID-XOR model; per-epoch per-source loss/acc on train/val/test.
    method in {'vanilla','mstar','mrand','mlearned','mstar_srcbasis'}.
    Final model = last epoch (paper Fig. 8 protocol: no checkpoint selection).

    destroy_fix: the paper scripts call destroy_block(x, m, 1) with an int
    block_list for mrand/mlearned; destroy_block returns the input UNCHANGED for
    non-list block_list, so the counterfactual KL is applied to INTACT inputs
    (a confidence penalty; mrand and mlearned are computationally identical).
    destroy_fix=False reproduces that published behavior exactly;
    destroy_fix=True passes [1] so the sampled/learned mask actually corrupts."""
    destroy_ids: Any = [1] if destroy_fix else 1
    model = pid_mod.FusionModel(cfg.dim0, cfg.dim1, cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history: Dict[str, Any] = {"epochs": [], "train": [], "val": [], "test": [],
                               "mask_diag": [], "train_mask_stats": []}

    if method == "mstar_srcbasis":
        assert rotation_ctx is not None, "mstar_srcbasis needs rotation context"
        Q0 = rotation_ctx["Q0"].to(device); Q1 = rotation_ctx["Q1"].to(device)
        m0_stats = rotation_ctx["stats"]["x0"]
        m1_stats = rotation_ctx["stats"]["x1"]
        mean0, std0 = m0_stats["m"].to(device), m0_stats["s"].to(device)
        mean1, std1 = m1_stats["m"].to(device), m1_stats["s"].to(device)

    for epoch in range(cfg.epochs):
        model.train()
        ep_destroy0, ep_destroy1, ep_nb = 0.0, 0.0, 0
        for b in train_loader:
            x0 = b["x0"].to(device); x1 = b["x1"].to(device); y = b["y"].to(device)
            m0 = b["mask0"].to(device); m1 = b["mask1"].to(device)
            f, u0, u1 = model.forward_logits(x0, x1)
            lf = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            ltot = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            if method == "vanilla":
                pass
            else:
                if method == "mstar":
                    x0_t = pid_mod.destroy_block(x0, m0, pid_mod.DESTROY_MASK, noise_std=1.0)
                    x1_t = pid_mod.destroy_block(x1, m1, pid_mod.DESTROY_MASK, noise_std=1.0)
                elif method == "mstar_srcbasis":
                    x0_t = destroy_syn_source_basis(pid_mod, x0, m0, Q0, mean0, std0)
                    x1_t = destroy_syn_source_basis(pid_mod, x1, m1, Q1, mean1, std1)
                elif method == "mfull":
                    # Destroy-all control: corrupt the entire modality (one at a
                    # time), i.e. what mlearned reduces to if its gates saturate.
                    x0_t = pid_mod.destroy_block(x0, torch.ones_like(x0), destroy_ids, noise_std=1.0)
                    x1_t = pid_mod.destroy_block(x1, torch.ones_like(x1), destroy_ids, noise_std=1.0)
                elif method == "mrand":
                    m0_r = (torch.rand_like(x0) < cfg.random_mask_proportion).float()
                    m1_r = (torch.rand_like(x1) < cfg.random_mask_proportion).float()
                    x0_t = pid_mod.destroy_block(x0, m0_r, destroy_ids, noise_std=1.0)
                    x1_t = pid_mod.destroy_block(x1, m1_r, destroy_ids, noise_std=1.0)
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
                    ep_destroy0 += float((g0 >= 0.5).float().mean().item())
                    ep_destroy1 += float((g1 >= 0.5).float().mean().item())
                    ep_nb += 1
                    x0_t = pid_mod.destroy_block(x0, g0, destroy_ids, noise_std=1.0)
                    x1_t = pid_mod.destroy_block(x1, g1, destroy_ids, noise_std=1.0)
                else:
                    raise ValueError(f"Unknown method: {method}")
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                l_cf = (pid_mod.bern_kl_to_uniform_from_logits(f_t0)
                        + pid_mod.bern_kl_to_uniform_from_logits(f_t1))
                ltot = ltot + lambda_kl * l_cf

            opt.zero_grad(set_to_none=True)
            ltot.backward()
            opt.step()

        history["epochs"].append(epoch)
        history["train"].append(_per_source_loss_acc(model, train_loader, device))
        history["val"].append(_per_source_loss_acc(model, val_loader, device))
        history["test"].append(_per_source_loss_acc(model, test_loader, device))
        if method == "mlearned" and ep_nb > 0:
            history["train_mask_stats"].append(
                {"epoch": epoch, "destroy_frac0": ep_destroy0 / ep_nb,
                 "destroy_frac1": ep_destroy1 / ep_nb})
        if mask_diag_cfg is not None and method == "mlearned":
            history["mask_diag"].append(mask_oracle_diagnostic(
                pid_mod, model, cfg, mask_diag_cfg["batch"],
                mask_diag_cfg["gt0"], mask_diag_cfg["gt1"],
                device, lam_sparsity, epoch))

    return history


# =====================================================================================
# Run driver
# =====================================================================================

def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO).decode().strip()
    except Exception:
        return "unknown"


def run_arm(config_path: Path, device: str, methods_override: Optional[List[str]],
            seeds_override: Optional[List[int]], force: bool) -> None:
    conf = json.loads(config_path.read_text())
    tag = conf["tag"]
    rotation = conf.get("rotation", "identity")
    rot_seed = int(conf.get("rot_seed", 0))
    simplex = tuple(conf["simplex"])
    methods = methods_override or conf["methods"]
    seeds = seeds_override or conf["seeds"]
    lambda_kl = float(conf.get("lambda_kl", 10.0))
    lam_sparsity = float(conf.get("lam_sparsity", 1.0))
    epochs = int(conf.get("epochs", 30))
    destroy_fix = bool(conf.get("destroy_fix", False))
    config_hash = hashlib.sha256(json.dumps(conf, sort_keys=True).encode()).hexdigest()[:12]

    pid_mod = _load_module(ANALYSIS / "Xor_PID3Main_MaskSynIB_Search.py", "pid_mod")
    pu1, pu2, pred, psyn = simplex

    for method in methods:
        for seed in seeds:
            out_path = RUN_DIR / f"{tag}__{method}__seed{seed}.json"
            if out_path.exists() and not force:
                print(f"[SKIP] {out_path.name} exists")
                continue
            t0 = time.time()
            cfg = pid_mod.Config()
            cfg.device = device; cfg.seed = seed
            cfg.lr = 3e-4; cfg.weight_decay = 1e-5; cfg.dropout = 0.0
            cfg.epochs = epochs
            if "mask_steps" in conf:
                cfg.learned_mask_steps = int(conf["mask_steps"])
            pid_mod._set_nonoverlap_signal_probs(cfg, pu1, pu2, pred, psyn, pnone=0.0)

            # rotation (fixed across train/val/test and across seeds/methods)
            Q0 = get_rotation(pid_mod, rotation, rot_seed, 0, cfg.dim0, cfg)
            Q1 = get_rotation(pid_mod, rotation, rot_seed, 1, cfg.dim1, cfg)
            if Q0 is not None:
                cfg.rotation_Q0 = Q0
                cfg.rotation_Q1 = Q1
            nonlinearity = conf.get("nonlinearity", None)
            if nonlinearity:
                cfg.rotation_nonlinearity = nonlinearity
            if "tanh_scale" in conf:
                cfg.rotation_tanh_scale = float(conf["tanh_scale"])

            # Frozen MLP mixer (Arm A) — built once, shared by all methods/seeds
            mixer_info: Dict[str, Any] = {}
            if conf.get("mixer", None) == "mlp":
                mixer_seed = int(conf.get("mixer_seed", 0))
                target_std = float(conf.get("mixer_target_preact_std", 2.5))
                mix0 = get_mlp_mixer(pid_mod, cfg, mixer_seed, 0, cfg.dim0, target_std)
                mix1 = get_mlp_mixer(pid_mod, cfg, mixer_seed, 1, cfg.dim1, target_std)
                cfg.mlp_mixer0 = mix0
                cfg.mlp_mixer1 = mix1
                mixer_info = {
                    "mixer": "mlp", "mixer_seed": mixer_seed,
                    "mixer_target_preact_std": target_std,
                    "mixer_realized_preact_std": [mix0["realized_preact_std"],
                                                  mix1["realized_preact_std"]],
                    "mixer_files": [mix0["npz_file"], mix1["npz_file"]],
                }

            set_global_seed(seed)
            _, splits, train_l, val_l, test_l = pid_mod.build_loaders(cfg, verbose=False)

            rotation_ctx = None
            if method == "mstar_srcbasis":
                full_train_ds = train_l.dataset.dataset  # Subset -> PID4BlockDataset
                rotation_ctx = {"Q0": Q0, "Q1": Q1, "stats": full_train_ds.stats}

            # Experiment 2 diagnostic: fixed held-out val batch + global support gt
            mask_diag_cfg = None
            if method == "mlearned" and conf.get("mask_diag", False):
                # Collate the fixed batch straight from the val dataset: creating a
                # DataLoader iterator would consume one global RNG draw and shift
                # all subsequent epoch shuffles vs the other methods.
                from torch.utils.data import default_collate
                val_ds = val_l.dataset
                nb = min(64, len(val_ds))
                fixed_batch = default_collate([val_ds[i] for i in range(nb)])
                blocks0 = _block_index_sets(pid_mod, cfg.dim0, cfg)
                blocks1 = _block_index_sets(pid_mod, cfg.dim1, cfg)
                gt0 = torch.zeros(cfg.dim0, dtype=torch.bool)
                gt0[blocks0[0]] = True; gt0[blocks0[1]] = True   # unique + red
                gt1 = torch.zeros(cfg.dim1, dtype=torch.bool)
                gt1[blocks1[0]] = True; gt1[blocks1[1]] = True
                mask_diag_cfg = {"batch": fixed_batch, "gt0": gt0, "gt1": gt1}

            hist = train_one(pid_mod, cfg, train_l, val_l, test_l, device,
                             method=method, lambda_kl=lambda_kl,
                             lam_sparsity=lam_sparsity,
                             rotation_ctx=rotation_ctx,
                             mask_diag_cfg=mask_diag_cfg,
                             destroy_fix=destroy_fix)
            elapsed = time.time() - t0
            final = hist["test"][-1]
            print(f"[{tag:24s} {method:15s} seed={seed}] {elapsed:6.1f}s | "
                  f"test total={final['_overall']['acc']:.3f} "
                  f"syn={final['syn']['acc']:.3f} u1={final['u1']['acc']:.3f} "
                  f"red={final['red']['acc']:.3f}")
            payload = {
                "meta": {
                    "tag": tag, "method": method, "seed": seed,
                    "rotation": rotation, "rot_seed": rot_seed,
                    "nonlinearity": conf.get("nonlinearity", None),
                    "tanh_scale": float(conf.get("tanh_scale", 1.0)),
                    **mixer_info,
                    "mask_steps": int(conf.get("mask_steps", 20)),
                    "destroy_fix": destroy_fix,
                    "simplex": list(simplex),
                    "lambda_kl": lambda_kl, "lam_sparsity": lam_sparsity,
                    "epochs": epochs, "splits": splits,
                    "git_commit": _git_commit(),
                    "config_file": (str(config_path.relative_to(REPO))
                                    if config_path.is_relative_to(REPO) else str(config_path)),
                    "config_hash": config_hash,
                    "command": " ".join(sys.argv),
                    "hostname": socket.gethostname(),
                    "device": device,
                    "torch_version": torch.__version__,
                    "timestamp": time.time(),
                    "Q_files": ([f"Q_{rotation}_rotseed{rot_seed}_mod{i}_d32.npy" for i in (0, 1)]
                                if rotation != "identity" else []),
                    "elapsed_s": elapsed,
                },
                "history": hist,
            }
            out_path.write_text(json.dumps(payload, indent=1, default=float))


# =====================================================================================
# Render: Table 1 + Figure 1 (dynamics) + Figure 2 (mask-oracle overlap) + summary
# =====================================================================================

COLORS = {"u1": "#2563EB", "red": "#16A34A", "syn": "#DC2626"}
LABELS = {"u1": r"$U_1$", "red": "$R$", "syn": "$S$"}
METHOD_LABELS = {
    "vanilla": "Vanilla fusion",
    "mstar": r"SynIB $M^*$",
    "mstar_srcbasis": r"SynIB $M^*$ (source basis)",
    "mrand": r"SynIB $M_{\mathrm{Random}}$",
    "mlearned": r"SynIB $M_{\mathrm{Learned}}$",
    "mfull": r"SynIB $M_{\mathrm{Full}}$ (destroy all)",
}


def _apply_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
    })


def _load_runs() -> Dict[str, List[Dict[str, Any]]]:
    """Group run payloads by (tag, method)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted(RUN_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        key = f"{d['meta']['tag']}__{d['meta']['method']}"
        groups.setdefault(key, []).append(d)
    return groups


def _mean_sem(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) <= 1:
        return float(a.mean()), 0.0
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def _stack_curve(runs, split, source, key="acc") -> np.ndarray:
    return np.asarray([[ep[source][key] for ep in r["history"][split]] for r in runs])


def render_table(groups) -> str:
    rows = []
    for key in sorted(groups.keys()):
        runs = groups[key]
        tag, method = key.split("__")
        tot = [r["history"]["test"][-1]["_overall"]["acc"] for r in runs]
        syn = [r["history"]["test"][-1]["syn"]["acc"] for r in runs]
        u1 = [r["history"]["test"][-1]["u1"]["acc"] for r in runs]
        red = [r["history"]["test"][-1]["red"]["acc"] for r in runs]
        tm, ts = _mean_sem(tot); sm, ss = _mean_sem(syn)
        um, us = _mean_sem(u1); rm, rs = _mean_sem(red)
        rows.append({"tag": tag, "method": method, "n_seeds": len(runs),
                     "total": [tm, ts], "syn": [sm, ss], "u1": [um, us], "red": [rm, rs]})
    lines = ["| Arm | Method | Total acc | U1 acc | R acc | S acc |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['tag']} | {r['method']} (n={r['n_seeds']}) "
            f"| {r['total'][0]:.3f} ± {r['total'][1]:.3f} "
            f"| {r['u1'][0]:.3f} ± {r['u1'][1]:.3f} "
            f"| {r['red'][0]:.3f} ± {r['red'][1]:.3f} "
            f"| {r['syn'][0]:.3f} ± {r['syn'][1]:.3f} |")
    md = "\n".join(lines)
    (ART_DIR / "table1.md").write_text(md + "\n")
    (ART_DIR / "table1.json").write_text(json.dumps(rows, indent=2))
    return md


def render_fig1(groups, arm_tags: List[str], out_name: str) -> None:
    """Fig. 8-style per-source dynamics, one row per arm, one column per method."""
    _apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    method_order = ["vanilla", "mstar", "mstar_srcbasis", "mrand", "mlearned", "mfull"]
    dash = (0, (3, 2))
    panel_keys = []
    for tag in arm_tags:
        for m in method_order:
            if f"{tag}__{m}" in groups:
                panel_keys.append((tag, m))
    ncols = max(sum(1 for t, _ in panel_keys if t == tag) for tag in arm_tags)
    nrows = len(arm_tags)
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols + 0.8, 2.3 * nrows),
                             sharey=True, squeeze=False)
    for i, tag in enumerate(arm_tags):
        cols = [(t, m) for t, m in panel_keys if t == tag]
        for j in range(ncols):
            ax = axes[i][j]
            if j >= len(cols):
                ax.axis("off"); continue
            _, method = cols[j]
            runs = groups[f"{tag}__{method}"]
            epochs = np.asarray(runs[0]["history"]["epochs"])
            annot = []
            for src in ["u1", "red", "syn"]:
                tr = _stack_curve(runs, "train", src)
                va = _stack_curve(runs, "val", src)
                tm, te = tr.mean(0), tr.std(0, ddof=1) / np.sqrt(max(1, len(runs)))
                vm, ve = va.mean(0), va.std(0, ddof=1) / np.sqrt(max(1, len(runs)))
                ax.plot(epochs, tm, color=COLORS[src], linestyle=dash, lw=1.2)
                ax.fill_between(epochs, tm - te, tm + te, color=COLORS[src], alpha=0.10, edgecolor="none")
                ax.plot(epochs, vm, color=COLORS[src], linestyle="-", lw=1.5)
                ax.fill_between(epochs, vm - ve, vm + ve, color=COLORS[src], alpha=0.16, edgecolor="none")
                te_fin = _stack_curve(runs, "test", src).mean(0)[-1]
                annot.append((src, float(te_fin)))
            ax.axhline(0.5, color="#999999", lw=0.6, linestyle=(0, (1, 2)))
            for k, (src, v) in enumerate(annot):
                ax.text(0.97, 0.42 - k * 0.09, f"{LABELS[src]}: {v:.2f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=7, color=COLORS[src])
            ax.text(0.97, 0.51, "Test acc", transform=ax.transAxes, ha="right",
                    va="top", fontsize=7, fontweight="bold", color="#222222")
            title = METHOD_LABELS.get(method, method)
            if i == 0 or nrows == 1:
                ax.set_title(title, fontsize=8.5)
            else:
                ax.set_title(title, fontsize=8.5)
            if j == 0:
                ax.set_ylabel(f"{tag}\nPer-source accuracy", fontsize=8)
            if i == nrows - 1:
                ax.set_xlabel("Epoch")
            ax.set_ylim(0.38, 1.04)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    handles = []
    import matplotlib.lines as mlines
    for src in ["u1", "red", "syn"]:
        handles.append(mlines.Line2D([], [], color=COLORS[src], linestyle=dash, lw=1.2,
                                     label=f"{LABELS[src]} train"))
        handles.append(mlines.Line2D([], [], color=COLORS[src], linestyle="-", lw=1.5,
                                     label=f"{LABELS[src]} val"))
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{out_name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_DIR / out_name}.pdf/.png")
    import matplotlib.pyplot as plt2
    plt2.close(fig)


def render_fig2(groups, tag: str, out_name: str) -> Dict[str, Any]:
    """Mask-oracle IoU / precision / recall curves (Experiment 2, identity arm)."""
    _apply_style()
    import matplotlib.pyplot as plt
    key = f"{tag}__mlearned"
    runs = [r for r in groups.get(key, []) if r["history"].get("mask_diag")]
    if not runs:
        print(f"[render_fig2] no mask_diag runs for {key}")
        return {}
    metrics = ["iou", "precision", "recall"]
    if "auroc" in runs[0]["history"]["mask_diag"][0]["mod0"]:
        metrics = ["iou", "precision", "recall", "auroc"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.85 * len(metrics), 2.6), sharey=True)
    summary = {}
    for ax, met in zip(axes, metrics):
        curves, rand_curves = [], []
        for r in runs:
            md = r["history"]["mask_diag"]
            ep = [d["epoch"] for d in md]
            # average the two modalities
            c = [(d["mod0"][met] + d["mod1"][met]) / 2 for d in md]
            rc = [(d["rand0"][met] + d["rand1"][met]) / 2 for d in md]
            curves.append(c); rand_curves.append(rc)
            ax.plot(ep, c, color="#d55e00", alpha=0.35, lw=0.9)
        cm = np.asarray(curves).mean(0)
        rm = np.asarray(rand_curves).mean(0)
        ax.plot(ep, cm, color="#d55e00", lw=2.0, label="Learned mask (mean)")
        ax.plot(ep, rm, color="#7a7a7a", lw=1.4, linestyle="--", label="Random mask (π=0.5)")
        ax.set_title(met.capitalize())
        ax.set_xlabel("Epoch"); ax.set_ylim(0, 1.02)
        summary[met] = {"learned_final": float(cm[-1]), "random_final": float(rm[-1]),
                        "learned_per_seed_final": [float(c[-1]) for c in curves]}
    axes[0].set_ylabel("Support agreement")
    axes[-1].legend(loc="lower right", fontsize=7)
    fig.suptitle("Learned mask vs oracle unimodal support (unique+redundant blocks)", y=1.04, fontsize=9)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{out_name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {FIG_DIR / out_name}.pdf/.png")
    plt.close(fig)
    return summary


def render_all() -> None:
    groups = _load_runs()
    if not groups:
        print("No runs found."); return
    md = render_table(groups)
    print(md)
    tags = sorted({k.split("__")[0] for k in groups})
    arm_tags = [t for t in ["identity_fixed", "rotated_full_fixed", "rotated_partial_fixed"]
                if t in tags]
    if arm_tags:
        render_fig1(groups, arm_tags, "fig1_entangled_dynamics")
    if "identity" in tags:
        render_fig1(groups, ["identity"], "fig1_identity_as_published")
    fig2_summary = {}
    if "identity_fixed" in tags:
        fig2_summary["identity_fixed"] = render_fig2(groups, "identity_fixed",
                                                     "fig2_mask_oracle_overlap")
    if "identity_fixed_steps100" in tags:
        fig2_summary["identity_fixed_steps100"] = render_fig2(
            groups, "identity_fixed_steps100", "fig2_mask_oracle_overlap_steps100")
    if "identity" in tags:
        fig2_summary["identity_as_published"] = render_fig2(groups, "identity",
                                                            "fig2_mask_oracle_overlap_as_published")
    (ART_DIR / "fig2_summary.json").write_text(json.dumps(fig2_summary, indent=2))


# =====================================================================================
# Main
# =====================================================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "render"])
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.mode == "render":
        render_all()
        return

    assert args.config, "run mode needs --config"
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_arm(Path(args.config), device, args.methods, args.seeds, args.force)


if __name__ == "__main__":
    main()
