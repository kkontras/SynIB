# synib_learned_experiment_procedural.py
# Procedural refactor: Synthetic block dataset + fusion model + SynIB-Learned (learned destroy masks) + KL sweep
# Goal: make it easier to tweak and understand, with fewer nested closures and clearer phases.

from __future__ import annotations

import os, time, json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# ============================================================
# Types / constants
# ============================================================

MaskMethod = Literal["kl_uniform_fusion", "flip_fusion", "fusion_more_than_unimodal", "unimodal", "kl_uniform_unimodal", "adv_unimodal"]
SelectBy = Literal["loss", "acc"]
GateShape = Literal["global", "per_example"]
IoUTarget = Literal["syn", "uni"]

_SOURCES = ("u1", "u2", "red", "syn")
_SRC2IDX = {s: i for i, s in enumerate(_SOURCES)}

MASK_NOISE, MASK_UNIQUE, MASK_RED, MASK_SYN = 0, 1, 2, 3


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    # data
    n_train: int = 2000
    n_test: int = 4000
    dim0: int = 32
    dim1: int = 32
    frac_unique: float = 0.20
    frac_red: float = 0.20
    frac_syn: float = 0.20
    random_block_positions: bool = False
    latent_u: int = 4
    latent_r: int = 4
    latent_s: int = 4
    unique_strength: float = 3.0
    red_strength: float = 3.0
    syn_strength: float = 3.0
    noise_std: float = 1.0
    signal_probs: Dict[str, float] = field(default_factory=lambda: {"none": 0.0, "u1": 0.0, "u2": 0.0, "red": 0.9, "syn": 0.1})
    val_frac: float = 0.10

    # train
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-5
    hidden: int = 256
    lambda_uni: float = 1.0
    lambda_kl: float = 0.0

    # runtime
    device: Optional[str] = None
    out_dir: str = "runs_refactor"

    # learned mask params
    learned_mask_method: MaskMethod = "adv_unimodal"
    learned_mask_steps: int = 100
    learned_mask_lr: float = 1e-1
    learned_mask_tau: float = 1.0
    learned_mask_noise_std: float = 1.0
    learned_mask_lam_sparsity: float = 5
    learned_mask_alpha_unimodal: float = 1.0
    learned_mask_hard: bool = True
    learned_mask_hard_thresh: float = 0.5
    learned_mask_gate_shape: GateShape = "per_example"


# ============================================================
# Small utilities
# ============================================================

def set_seed(seed: int) -> None:
    seed = int(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def mkdirp(path: str) -> None: os.makedirs(path, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)

def load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path): return {}
    with open(path, "r") as f: return json.load(f)

def mean_std(xs: List[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), float(a.std())

def acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    p = (torch.sigmoid(logits) > 0.5).float()
    return float((p == y).float().mean().item())

def prob_key(pu1: float, pu2: float, pred: float, psyn: float, pnone: float = 0.0) -> str:
    return f"pu1={pu1:.3f}|pu2={pu2:.3f}|pred={pred:.3f}|psyn={psyn:.3f}|pnone={pnone:.3f}"

def set_nonoverlap_signal_probs(cfg: Config, pu1: float, pu2: float, pred: float, psyn: float, pnone: float = 0.0) -> None:
    tot = pu1 + pu2 + pred + psyn + pnone
    if abs(tot - 1.0) > 1e-5: raise ValueError(f"probs must sum to 1, got {tot}")
    cfg.signal_probs = {"none": float(pnone), "u1": float(pu1), "u2": float(pu2), "red": float(pred), "syn": float(psyn)}

def sign_from_bit(bit01: float) -> float: return 1.0 if float(bit01) == 1.0 else -1.0

def freeze_model_(m: nn.Module) -> None:
    for p in m.parameters(): p.requires_grad_(False)

def _iou_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.bool(); gt = gt.bool()
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return float((inter / (union + eps)).item())


def _overlap_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    # Convert to boolean for logical operations
    pred = pred.bool()
    gt = gt.bool()

    # Intersection: pixels that are both in pred and gt
    inter = (pred & gt).sum().float()

    # Target: total pixels in the ground truth
    target_area = gt.sum().float()

    # If GT is empty, overlap is 1.0 (it's technically fully covered)
    if target_area < eps:
        return 1.0

    return float((inter / target_area).item())

# ============================================================
# Data generation helpers
# ============================================================

def block_sizes(dim: int, fu: float, fr: float, fs: float) -> Tuple[int, int, int, int]:
    u, r, s = int(round(dim * fu)), int(round(dim * fr)), int(round(dim * fs))
    used = u + r + s
    if used > dim:
        overflow = used - dim
        take = min(s, overflow); s -= take; overflow -= take
        take = min(r, overflow); r -= take; overflow -= take
        take = min(u, overflow); u -= take; overflow -= take
    noise = dim - (u + r + s)
    return u, r, s, noise

def choose_blocks(dim: int, u: int, r: int, s: int, g: torch.Generator, randomize: bool) -> Dict[int, torch.Tensor]:
    if u + r + s > dim: raise ValueError("block sizes exceed dim")
    if randomize:
        perm = torch.randperm(dim, generator=g)
        idx_u, idx_r, idx_s, idx_n = perm[:u], perm[u:u+r], perm[u+r:u+r+s], perm[u+r+s:]
    else:
        idx_u, idx_r, idx_s, idx_n = torch.arange(0, u), torch.arange(u, u+r), torch.arange(u+r, u+r+s), torch.arange(u+r+s, dim)
    return {MASK_UNIQUE: idx_u, MASK_RED: idx_r, MASK_SYN: idx_s, MASK_NOISE: idx_n}

def parse_subset_key(key: str) -> List[str]:
    key = key.strip().lower()
    if key in ("", "none"): return []
    return [p.strip() for p in key.split("+") if p.strip()]

def sample_subset_key(g: torch.Generator, probs: Dict[str, float]) -> str:
    keys = list(probs.keys())
    p = torch.tensor([float(probs[k]) for k in keys], dtype=torch.float32)
    if (p < 0).any(): raise ValueError("negative probs")
    if float(p.sum().item()) <= 0: raise ValueError("prob sum <= 0")
    idx = torch.multinomial(p, 1, replacement=True, generator=g).item()
    return keys[idx]

def multihot(active: List[str]) -> torch.Tensor:
    v = torch.zeros(len(_SOURCES), dtype=torch.float32)
    for s in active:
        if s not in _SRC2IDX: raise ValueError(f"unknown source {s}")
        v[_SRC2IDX[s]] = 1.0
    return v


# ============================================================
# Dataset
# ============================================================

class PID4BlockDataset(Dataset):
    def __init__(self, cfg: Config, n: int, seed: int, split: str, train_stats: Optional[Dict[str, Any]] = None, verbose: bool = True):
        super().__init__()
        self.cfg, self.n = cfg, int(n)
        g = torch.Generator().manual_seed(int(seed))

        u0, r0, s0, _ = block_sizes(cfg.dim0, cfg.frac_unique, cfg.frac_red, cfg.frac_syn)
        u1, r1, s1, _ = block_sizes(cfg.dim1, cfg.frac_unique, cfg.frac_red, cfg.frac_syn)

        # fixed projections
        with torch.random.fork_rng():
            torch.manual_seed(999)
            self.proj_u0 = torch.randn(cfg.dim0, cfg.latent_u) * 0.5
            self.proj_u1 = torch.randn(cfg.dim1, cfg.latent_u) * 0.5
            self.proj_r0 = torch.randn(cfg.dim0, cfg.latent_r) * 0.5
            self.proj_r1 = torch.randn(cfg.dim1, cfg.latent_r) * 0.5
            self.proj_s0 = torch.randn(cfg.dim0, cfg.latent_s) * 0.5
            self.proj_s1 = torch.randn(cfg.dim1, cfg.latent_s) * 0.5

        self.x0 = torch.randn(self.n, cfg.dim0, generator=g) * float(cfg.noise_std)
        self.x1 = torch.randn(self.n, cfg.dim1, generator=g) * float(cfg.noise_std)
        self.mask0 = torch.zeros(self.n, cfg.dim0, dtype=torch.long)
        self.mask1 = torch.zeros(self.n, cfg.dim1, dtype=torch.long)
        self.y = torch.zeros(self.n, 1)
        self.source = torch.zeros(self.n, len(_SOURCES), dtype=torch.float32)

        # validate sampling + reset generator
        _ = sample_subset_key(g, cfg.signal_probs); g.manual_seed(int(seed))

        counts = {k: 0 for k in cfg.signal_probs.keys()}

        for i in range(self.n):
            b0 = choose_blocks(cfg.dim0, u0, r0, s0, g, cfg.random_block_positions)
            b1 = choose_blocks(cfg.dim1, u1, r1, s1, g, cfg.random_block_positions)

            key = sample_subset_key(g, cfg.signal_probs)
            active = parse_subset_key(key)
            counts[key] = counts.get(key, 0) + 1
            self.source[i] = multihot(active)

            y_i = float(torch.rand(1, generator=g).item() > 0.5)
            self.y[i] = y_i

            if "u1" in active:
                z = torch.randn(cfg.latent_u, generator=g) * float(cfg.unique_strength)
                z = z.abs() * sign_from_bit(y_i)
                x_full = self.proj_u0 @ z
                self.x0[i, b0[MASK_UNIQUE]] = x_full[b0[MASK_UNIQUE]]
                self.mask0[i, b0[MASK_UNIQUE]] = MASK_UNIQUE

            if "u2" in active:
                z = torch.randn(cfg.latent_u, generator=g) * float(cfg.unique_strength)
                z = z.abs() * sign_from_bit(y_i)
                x_full = self.proj_u1 @ z
                self.x1[i, b1[MASK_UNIQUE]] = x_full[b1[MASK_UNIQUE]]
                self.mask1[i, b1[MASK_UNIQUE]] = MASK_UNIQUE

            if "red" in active:
                z = torch.randn(cfg.latent_r, generator=g) * float(cfg.red_strength)
                z = z.abs() * sign_from_bit(y_i)
                x0_full, x1_full = self.proj_r0 @ z, self.proj_r1 @ z
                self.x0[i, b0[MASK_RED]] = x0_full[b0[MASK_RED]]
                self.x1[i, b1[MASK_RED]] = x1_full[b1[MASK_RED]]
                self.mask0[i, b0[MASK_RED]] = MASK_RED
                self.mask1[i, b1[MASK_RED]] = MASK_RED

            if "syn" in active:
                # XOR-like: b0 random, b1 = b0 XOR y
                b_s0 = float(torch.rand(1, generator=g).item() > 0.5)
                b_s1 = float(b_s0 != y_i)
                z0 = (torch.randn(cfg.latent_s, generator=g) * float(cfg.syn_strength)).abs() * sign_from_bit(b_s0)
                z1 = (torch.randn(cfg.latent_s, generator=g) * float(cfg.syn_strength)).abs() * sign_from_bit(b_s1)
                x0_full, x1_full = self.proj_s0 @ z0, self.proj_s1 @ z1
                self.x0[i, b0[MASK_SYN]] = x0_full[b0[MASK_SYN]]
                self.x1[i, b1[MASK_SYN]] = x1_full[b1[MASK_SYN]]
                self.mask0[i, b0[MASK_SYN]] = MASK_SYN
                self.mask1[i, b1[MASK_SYN]] = MASK_SYN

        self.stats = self._normalize(split, train_stats)

        if verbose:
            top = ", ".join([f"{k}={counts[k]/self.n:.3f}" for k in sorted(counts.keys())])
            print(f"[DATA:{split}] {top}")

    def _normalize(self, split: str, train_stats: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if split == "train":
            train_stats = {
                "x0": {"m": self.x0.mean(0), "s": self.x0.std(0) + 1e-7},
                "x1": {"m": self.x1.mean(0), "s": self.x1.std(0) + 1e-7},
            }
        assert train_stats is not None
        self.x0 = (self.x0 - train_stats["x0"]["m"]) / train_stats["x0"]["s"]
        self.x1 = (self.x1 - train_stats["x1"]["m"]) / train_stats["x1"]["s"]
        return train_stats

    def __len__(self) -> int: return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {"x0": self.x0[i], "x1": self.x1[i], "y": self.y[i], "mask0": self.mask0[i], "mask1": self.mask1[i], "source": self.source[i]}


def build_loaders(cfg: Config, seed: int, verbose: bool = True) -> Tuple[str, Dict[str, int], DataLoader, DataLoader, DataLoader]:
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    full_train = PID4BlockDataset(cfg, cfg.n_train, seed=seed, split="train", verbose=verbose)
    test_ds = PID4BlockDataset(cfg, cfg.n_test, seed=seed + 1, split="test", train_stats=full_train.stats, verbose=verbose)

    n = len(full_train)
    n_val = max(1, int(round(cfg.val_frac * n)))
    n_tr = n - n_val

    g = torch.Generator().manual_seed(seed + 12345)
    perm = torch.randperm(n, generator=g).tolist()
    tr_idx, va_idx = perm[:n_tr], perm[n_tr:]

    tr_ds, va_ds = Subset(full_train, tr_idx), Subset(full_train, va_idx)
    tr_loader = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False)
    te_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    return device, {"n_train": n_tr, "n_val": n_val, "n_test": len(test_ds)}, tr_loader, va_loader, te_loader


# ============================================================
# Model
# ============================================================

class FusionModel(nn.Module):
    def __init__(self, dim0: int, dim1: int, hidden: int = 256, fuse_hidden: int = 128):
        super().__init__()
        self.enc0 = nn.Sequential(nn.Linear(dim0, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.enc1 = nn.Sequential(nn.Linear(dim1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.score0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.score1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.fuse = nn.Sequential(nn.Linear(2 * hidden, fuse_hidden), nn.ReLU(), nn.Linear(fuse_hidden, fuse_hidden), nn.ReLU(), nn.Linear(fuse_hidden, 1))

    def forward_logits(self, x0: torch.Tensor, x1: torch.Tensor, not_detached=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0, h1 = self.enc0(x0), self.enc1(x1)
        if not_detached:
            u0, u1 = self.score0(h0), self.score1(h1)  # detach unimodal heads
        else:
            u0, u1 = self.score0(h0.detach()), self.score1(h1.detach())  # detach unimodal heads
        f = self.fuse(torch.cat([h0, h1], dim=-1))
        return f, u0, u1


# ============================================================
# Mask learning primitives
# ============================================================

def bern_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p, q = p.clamp(eps, 1 - eps), q.clamp(eps, 1 - eps)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()

def bern_kl_to_uniform(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1 - eps)
    return (p * (2 * p).log() + (1 - p) * (2 * (1 - p)).log()).mean()

def apply_destroy(x: torch.Tensor, g: torch.Tensor, noise_std: float = 1.0, inv_mask=False) -> torch.Tensor:
    if g.dim() == 1: g = g.view(1, -1)
    g = g.to(x.device).type_as(x)
    eps = torch.randn_like(x) * float(noise_std)
    if inv_mask:
        return g * x + (1 - g) * eps
    return (1 - g) * x + g * eps

def mask_objective(method: MaskMethod, which: Literal["x0", "x1"], p_f_clean: torch.Tensor, p_u0_clean: torch.Tensor, p_u1_clean: torch.Tensor, p_f_t: torch.Tensor, p_u_t: torch.Tensor, y: torch.Tensor, sparsity: torch.Tensor, lam_sparsity: float, alpha_unimodal: float) -> torch.Tensor:
    if method == "kl_uniform_fusion": return bern_kl_to_uniform(p_f_t) + lam_sparsity * sparsity
    if method == "kl_uniform_unimodal": return bern_kl_to_uniform(p_u_t) + lam_sparsity * sparsity
    if method == "ce_unimodal": return F.binary_cross_entropy_with_logits(p_u_t, y) + float(lam_sparsity) * sparsity
    if method == "advbern_unimodal": return bern_kl_to_uniform(p_u_t)
    if method == "adv_unimodal": return -F.binary_cross_entropy_with_logits(p_u_t, y) - float(lam_sparsity) * sparsity
    if method == "adv_fusion":
        # ce_fusion = F.binary_cross_entropy_with_logits(p_f_t, y)
        ce_unimodal = F.binary_cross_entropy_with_logits(p_u_t, y)
        return bern_kl_to_uniform(p_f_t) + ce_unimodal + float(lam_sparsity) * sparsity
    if method == "flip_fusion": return -bern_kl(p_f_clean, p_f_t) + lam_sparsity * sparsity
    if method == "fusion_more_than_unimodal":
        div_f = bern_kl(p_f_clean, p_f_t)
        div_u = bern_kl(p_u0_clean, p_u_t) if which == "x0" else bern_kl(p_u1_clean, p_u_t)
        return -(div_f - alpha_unimodal * div_u) + lam_sparsity * sparsity
    if method == "unimodal":
        div_u = bern_kl(p_u0_clean, p_u_t) if which == "x0" else bern_kl(p_u1_clean, p_u_t)
        return -div_u + lam_sparsity * sparsity
    raise ValueError(f"Unknown method {method}")

def init_gate_logits(B: int, d: int, gate_shape: GateShape, device: str) -> torch.Tensor:
    return (torch.ones((B,d), device=device)).requires_grad_(True)

def gate_probs_from_logits(ell: torch.Tensor, tau: float, gate_shape: GateShape, B: int, d: int) -> torch.Tensor:
    if gate_shape == "global":
        return torch.sigmoid(ell / float(tau)).view(1, d)
    return torch.sigmoid(ell / float(tau)).view(B, d)

def gate_to_hard_mask(g: torch.Tensor, hard_thresh: float, ref_shape: torch.Size, inv_mask:bool=False) -> torch.Tensor:
    # returns bool mask aligned to ref_shape [B,D]
    pred = (g >= float(hard_thresh)) if g.dtype.is_floating_point else g.bool()
    if pred.dim() == 1: pred = pred.view(1, -1)
    if pred.shape[0] == 1 and ref_shape[0] > 1: pred = pred.expand(ref_shape[0], ref_shape[1])
    if inv_mask: pred = ~pred
    return pred

def oracle_mask_from_block(mask_block: Optional[torch.Tensor], target: IoUTarget) -> Optional[torch.Tensor]:
    if mask_block is None: return None
    if target == "syn": return (mask_block == MASK_SYN)
    # "uni": unique OR red
    return (mask_block == MASK_UNIQUE) | (mask_block == MASK_RED)

@torch.no_grad()
def forward_clean_probs(model: nn.Module, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    f, u0, u1 = model.forward_logits(x0, x1)
    return f, u0, u1

def forward_destroyed_probs(model: nn.Module, x0: torch.Tensor, x1: torch.Tensor, which: Literal["x0", "x1"], g: torch.Tensor, noise_std: float, inv_mask=False) -> Tuple[torch.Tensor, torch.Tensor]:
    if which == "x0": x0_t, x1_t = apply_destroy(x0, g, noise_std, inv_mask=inv_mask), x1
    else: x0_t, x1_t = x0, apply_destroy(x1, g, noise_std, inv_mask=inv_mask)
    f_t, u0_t, u1_t = model.forward_logits(x0_t, x1_t, not_detached=True)
    u_t = u0_t if which == "x0" else u1_t
    return f_t, u_t

def learn_one_gate(model: nn.Module, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor, which: Literal["x0", "x1"], method: MaskMethod, device: str, gate_shape: GateShape, steps: int, lr: float, tau: float, noise_std: float, lam_sparsity: float, alpha_unimodal: float, hard: bool, hard_thresh: float, mask_block: Optional[torch.Tensor], iou_target: IoUTarget, print_every: int = 10, label: str = "") -> torch.Tensor:
    # model.eval(); freeze_model_(model)
    x0, x1 = x0.to(device), x1.to(device)
    B, d0 = x0.shape
    _, d1 = x1.shape
    d = d0 if which == "x0" else d1

    p_f_clean, p_u0_clean, p_u1_clean = model.forward_logits(x0, x1)
    # f, u0, u1 = model.forward_logits(x0, x1)
    # steps = 100
    # lam_sparsity = 0.1
    # method = "adv_unimodal"
    # print_every = 100
    # lr = 0.1

    ell = torch.nn.Parameter(init_gate_logits(B, d, gate_shape, device), requires_grad=True)
    opt = torch.optim.Adam([ell], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps), eta_min=lr * 0.1)

    gt = oracle_mask_from_block(mask_block.to(device) if mask_block is not None else None, iou_target)
    gt_unires = oracle_mask_from_block(mask_block.to(device) if mask_block is not None else None, "rest")
    prev, stall = None, 0

    for t in range(int(steps)):
        g = gate_probs_from_logits(ell, tau, gate_shape, B, d)
        p_f_t, p_u_t = forward_destroyed_probs(model, x0, x1, which, g, noise_std, inv_mask=True)
        sparsity = g.mean()
        # obj = -F.binary_cross_entropy_with_logits(p_u_t, y)+ lam_sparsity*sparsity
        # obj = bern_kl_to_uniform(p_u_t)
        obj = mask_objective(method, which, p_f_clean, p_u0_clean, p_u1_clean, p_f_t, p_u_t, y, sparsity, float(lam_sparsity), float(alpha_unimodal))
        opt.zero_grad(set_to_none=True)
        obj.backward()

        torch.nn.utils.clip_grad_norm_([ell], 1.0)
        opt.step(); sched.step()

        if gt is not None and print_every > 0 and (t == steps - 1): #t % print_every == 0 or
            with torch.no_grad():
                g_now = torch.sigmoid(ell / float(tau))
                pred = gate_to_hard_mask(g_now, hard_thresh, gt.shape)
                iou = _overlap_binary(pred, gt)
                iou_unires = _overlap_binary(pred, gt_unires)
                sp = float(pred.float().mean().item())
                tag = f"{label} " if label else ""
                p_f_gt, p_u_gt = forward_destroyed_probs(model, x0, x1, which, gt_unires, noise_std)

                obj_gt = mask_objective(method, which, p_f_clean, p_u0_clean, p_u1_clean, p_f_gt, p_u_gt, y, sparsity,
                                     float(0.0), float(alpha_unimodal))

                # obj_gt = -F.binary_cross_entropy_with_logits(p_u_gt, y)

                acc_u_gt = acc_from_logits(p_u_gt, y)
                acc_u_t = acc_from_logits(p_u_t, y)
                acc_u_clean = acc_from_logits(p_f_clean, y)

                # print(f"        [mask]{tag}{which}/{method} step={t:03d} obj={obj:.3f} obj_gt={obj_gt:.3f} overlap({iou_target})={iou:.3f} overlap(uni/res)={iou_unires:.3f} sparsity={sp:.3f} acc_g={acc_u_t:.3f} acc_gt={acc_u_gt:.3f} acc_clean={acc_u_clean:.3f}")

        with torch.no_grad():
            now = torch.sigmoid(ell / float(tau))
            if prev is not None:
                dg = (now - prev).abs().mean().item()
                stall = stall + 1 if dg < 1e-4 else 0
                if stall >= 10: break
            prev = now.clone()

    g_final = torch.sigmoid(ell / float(tau)).detach()
    if hard: g_final = (g_final > float(hard_thresh)).float()
    return g_final

def learn_destroy_gates(model: nn.Module, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor, method: MaskMethod, device: str, gate_shape: GateShape, steps: int, lr: float, tau: float, noise_std: float, lam_sparsity: float, alpha_unimodal: float, hard: bool, hard_thresh: float, m0: Optional[torch.Tensor] = None, m1: Optional[torch.Tensor] = None, iou_target: IoUTarget = "syn", print_every: int = 10, label: str = "") -> Dict[str, torch.Tensor]:
    # main method masks
    g0 = learn_one_gate(model, x0, x1, y, "x0", method, device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m0, iou_target, print_every, label)
    g1 = learn_one_gate(model, x0, x1, y, "x1", method, device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m1, iou_target, print_every, label)
    # unimodal-ablation masks (used for your counterfactual KL)
    # g0_uni = learn_one_gate(model, x0, x1, "x0", "unimodal", device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m0, "uni", print_every, label)
    # g1_uni = learn_one_gate(model, x0, x1, "x1", "unimodal", device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m1, "uni", print_every, label)
    return {"g0": g0, "g1": g1}


# ============================================================
# Eval
# ============================================================

@torch.no_grad()
def eval_stats(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    n = 0
    sum_lf = sum_lu0 = sum_lu1 = 0.0
    sum_af = sum_au0 = sum_au1 = 0.0

    for b in loader:
        x0, x1, y = b["x0"].to(device), b["x1"].to(device), b["y"].to(device)

        f, u0, u1 = model.forward_logits(x0, x1)

        lf  = F.binary_cross_entropy_with_logits(f,  y)
        lu0 = F.binary_cross_entropy_with_logits(u0, y)
        lu1 = F.binary_cross_entropy_with_logits(u1, y)

        bs = y.size(0)
        n += bs

        sum_lf  += float(lf.item())  * bs
        sum_lu0 += float(lu0.item()) * bs
        sum_lu1 += float(lu1.item()) * bs

        sum_af  += acc_from_logits(f,  y) * bs
        sum_au0 += acc_from_logits(u0, y) * bs
        sum_au1 += acc_from_logits(u1, y) * bs

    denom = max(1, n)
    return {
        "loss_fusion": sum_lf / denom,
        "acc_fusion":  sum_af / denom,
        "loss_u0":     sum_lu0 / denom,
        "acc_u0":      sum_au0 / denom,
        "loss_u1":     sum_lu1 / denom,
        "acc_u1":      sum_au1 / denom,
    }

@torch.no_grad()
def eval_by_source(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, Dict[str, float]]:
    model.eval()
    out = {s: {"n": 0, "acc": 0.0} for s in _SOURCES}
    for b in loader:
        x0, x1, y = b["x0"].to(device), b["x1"].to(device), b["y"].to(device)
        src = b["source"].to(device)  # [B,4]
        f, _, _ = model.forward_logits(x0, x1)
        pred = (torch.sigmoid(f) > 0.5).float()
        correct = (pred == y).float().view(-1)
        for i, s in enumerate(_SOURCES):
            mask = (src[:, i] > 0.5)
            if mask.any():
                n = int(mask.sum().item())
                out[s]["n"] += n
                out[s]["acc"] += float(correct[mask].sum().item())
    for s in _SOURCES:
        n = out[s]["n"]
        out[s]["acc"] = (out[s]["acc"] / n) if n > 0 else float("nan")
    return out


# ============================================================
# Training (SynIB-Learned)
# ============================================================

def base_losses(model: FusionModel, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor, lambda_uni: float) -> Tuple[torch.Tensor, torch.Tensor]:
    f, u0, u1 = model.forward_logits(x0, x1)
    lf = F.binary_cross_entropy_with_logits(f, y)
    lu0 = F.binary_cross_entropy_with_logits(u0, y)
    lu1 = F.binary_cross_entropy_with_logits(u1, y)
    return f, u0, u1, lf + float(lambda_uni) * (lu0 + lu1)

def synib_counterfactual_kl(model: FusionModel, x0: torch.Tensor, x1: torch.Tensor, g0_uni: torch.Tensor, g1_uni: torch.Tensor, noise_std: float) -> torch.Tensor:
    x0_t = apply_destroy(x0, g0_uni, noise_std)
    x1_t = apply_destroy(x1, g1_uni, noise_std)
    f_t0, _, _ = model.forward_logits(x0_t, x1)
    f_t1, _, _ = model.forward_logits(x0, x1_t)
    return bern_kl_to_uniform(torch.sigmoid(f_t0).view(-1)) + bern_kl_to_uniform(torch.sigmoid(f_t1).view(-1))

def train_synib_learned(cfg: Config, train_loader: DataLoader, val_loader: DataLoader, device: str, verbose: bool = True) -> Tuple[nn.Module, Dict[str, Any]]:
    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # warmup (stabilize supervised heads)
    # for _ in range(int(5)):
    #     for b in train_loader:
    #         x0, x1, y = b["x0"].to(device), b["x1"].to(device), b["y"].to(device)
    #         _, _, _, l_base = base_losses(model, x0, x1, y, cfg.lambda_uni)
    #         opt.zero_grad(set_to_none=True); l_base.backward(); opt.step()

    best_state, best_val = None, -1.0
    hist = {"train": [], "val": [], "best_val_acc": float("-inf"), "best_val_loss": float("inf"), "best_epoch": None}

    for ep in range(int(cfg.epochs)):
        model.train()
        n, sum_l, sum_acc = 0, 0.0, 0.0

        nb_iou = 0
        sum_iou_syn0 = sum_iou_syn1 = 0.0
        sum_iou_uni0 = sum_iou_uni1 = 0.0
        sum_g0 = sum_g1 = sum_g0u = sum_g1u = 0.0

        for b in train_loader:
            x0, x1, y = b["x0"].to(device), b["x1"].to(device), b["y"].to(device)
            m0, m1 = b["mask0"].to(device), b["mask1"].to(device)

            # supervised loss
            f, u1, u2, l_base = base_losses(model, x0, x1, y, cfg.lambda_uni)

            # learn gates with frozen model
            # req = [p.requires_grad for p in model.parameters()]
            # was_train = model.training
            # for p in model.parameters(): p.requires_grad_(False)
            # model.eval()
            # try:
            with torch.enable_grad():
                masks = learn_destroy_gates(
                    model=model,
                    x0=x0.detach(), x1=x1.detach(),
                    y=y.detach(),
                    method=cfg.learned_mask_method,
                    device=device,
                    gate_shape=cfg.learned_mask_gate_shape,
                    steps=cfg.learned_mask_steps,
                    lr=cfg.learned_mask_lr,
                    tau=cfg.learned_mask_tau,
                    noise_std=cfg.learned_mask_noise_std,
                    lam_sparsity=cfg.learned_mask_lam_sparsity,
                    alpha_unimodal=cfg.learned_mask_alpha_unimodal,
                    hard=cfg.learned_mask_hard,
                    hard_thresh=cfg.learned_mask_hard_thresh,
                    m0=m0.detach(), m1=m1.detach(),
                    iou_target="syn",  # change to "uni" if you want to track that instead
                    print_every=cfg.learned_mask_steps,
                    label=f"ep{ep}",
                )
            # finally:
            #     if was_train: model.train()
            #     for p, r in zip(model.parameters(), req): p.requires_grad_(r)

            # IoU bookkeeping (optional)
            thr = float(cfg.learned_mask_hard_thresh)
            g0, g1 = masks["g0"].detach(), masks["g1"].detach()

            pred0 = gate_to_hard_mask(g0, thr, m0.shape)
            pred1 = gate_to_hard_mask(g1, thr, m1.shape)

            gt_syn0 = (m0 == MASK_SYN); gt_syn1 = (m1 == MASK_SYN)
            gt_uni0 = (m0 == MASK_UNIQUE) | (m0 == MASK_RED)
            gt_uni1 = (m1 == MASK_UNIQUE) | (m1 == MASK_RED)

            sum_iou_syn0 += _overlap_binary(pred0, gt_syn0)
            sum_iou_syn1 += _overlap_binary(pred1, gt_syn1)
            sum_iou_uni0 += _overlap_binary(pred0, gt_uni0)
            sum_iou_uni1 += _overlap_binary(pred1, gt_uni1)

            nb_iou += 1

            sum_g0 += float(g0.float().mean().item())
            sum_g1 += float(g1.float().mean().item())

            # counterfactual KL regularizer
            l_cf = synib_counterfactual_kl(model, x0, x1, g0, g1, cfg.learned_mask_noise_std)
            loss = l_base + float(cfg.lambda_kl) * l_cf

            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

            bs = y.size(0); n += bs
            sum_l += float(loss.item()) * bs
            sum_acc += acc_from_logits(f.detach(), y) * bs

        tr = {"epoch": ep, "loss_total": sum_l / max(1, n), "acc_fusion": sum_acc / max(1, n)}
        if nb_iou > 0:
            tr["iou_syn_avg"] = 0.5 * ((sum_iou_syn0 / nb_iou) + (sum_iou_syn1 / nb_iou))
            tr["iou_uni_avg"] = 0.5 * ((sum_iou_uni0 / nb_iou) + (sum_iou_uni1 / nb_iou))
            tr["mask_g0_mean"] = sum_g0 / nb_iou
            tr["mask_g1_mean"] = sum_g1 / nb_iou
            tr["mask_g0u_mean"] = sum_g0u / nb_iou
            tr["mask_g1u_mean"] = sum_g1u / nb_iou

        va_stats = eval_stats(model, val_loader, device)
        va = {"epoch": ep, **va_stats}
        hist["train"].append(tr); hist["val"].append(va)

        if va["acc_fusion"] > best_val:
            best_val = float(va["acc_fusion"])
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            hist["best_val_acc"] = float(va["acc_fusion"])
            hist["best_val_loss"] = float(va["loss_fusion"])
            hist["best_epoch"] = ep

        if verbose:
            mark = "*" if hist["best_epoch"] == ep else ""
            extra = ""
            if "iou_syn_avg" in tr:
                extra = f" | overlap_syn={tr['iou_syn_avg']:.3f} overlap_uni={tr['iou_uni_avg']:.3f} g0u={tr['mask_g0u_mean']:.3f} g1u={tr['mask_g1u_mean']:.3f}"
            # print(f"[E{ep:03d}]{mark} train_loss={tr['loss_total']:.3f} train_acc={tr['acc_fusion']:.3f} | va_loss={va['loss_fusion']:.3f} va_acc={va['acc_fusion']:.3f} va_u0={va['acc_u0']:.3f} va_u1={va['acc_u1']:.3f} {extra}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model, hist

def generate_prob_settings(step: float, prior_u2: float, targets_psyn: List[float]) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    def r(x: float) -> float: return float(round(x, 3))

    for psyn in targets_psyn:
        for pu1 in np.arange(0, 1.0 - prior_u2 - psyn + 1e-12, step):
            pred = 1.0 - pu1 - prior_u2 - psyn
            if pred >= -1e-9: out.append((r(pu1), r(prior_u2), r(psyn), r(pred)))

    target_set = {float(round(t, 3)) for t in targets_psyn}
    for psyn in np.arange(0, 1.0 - prior_u2 + 1e-12, step):
        if float(round(psyn, 3)) in target_set: continue
        for pu1 in np.arange(0, 1.0 - prior_u2 - psyn + 1e-12, step):
            pred = 1.0 - pu1 - prior_u2 - psyn
            if pred >= -1e-9: out.append((r(pu1), r(prior_u2), r(psyn), r(pred)))
    return out

def select_best_lambda_kl_lamsparse(cfg0: Config, seeds: List[int], kl_vals: List[float], lam_sparsity_vals: List[float], select_by: SelectBy = "acc", verbose: bool = False) -> Dict[str, Any]:
    """
    Same return format as before, but "best" (lambda_kl, lam_sparsity) is chosen by VAL performance
    averaged across seeds (mean over seeds). Uses no-retrain: it trains each combo for each seed once.

    Returns (same keys as before):
      best_lambda_kl, best_lam_sparsity, best_val_loss, best_val_acc, best_test, best_by, best_hist, table, split
    Where:
      - best_val_* are MEAN across seeds for the selected combo
      - best_test is a stats dict with acc_fusion/loss_fusion = MEAN across seeds (plus *_std fields)
      - best_by is per-source acc/n aggregated across seeds for the selected combo
      - best_hist is a dict containing per-seed hists + per-seed rows for the selected combo
      - table is per-(kl,lsp) aggregated across seeds (mean/std of val/test)
    """
    def mean_std(xs: List[float]) -> Tuple[float, float]:
        arr = np.asarray(xs, dtype=float)
        return float(arr.mean()), float(arr.std())

    # ---- run full grid across seeds ----
    all_rows: List[Dict[str, Any]] = []
    split_any = None

    for seed in seeds:
        device, split, tr_loader, va_loader, te_loader = build_loaders(cfg0, seed, verbose=True)
        split_any = split_any or split

        for lam_kl in kl_vals:
            for lam_sp in lam_sparsity_vals:
                cfg = Config(**asdict(cfg0))
                cfg.lambda_kl = float(lam_kl)
                cfg.learned_mask_lam_sparsity = float(lam_sp)

                model, hist = train_synib_learned(cfg, tr_loader, va_loader, device, verbose=verbose)
                val_best = max(hist["val"], key=lambda r: r["acc_fusion"])
                test_stats = eval_stats(model, te_loader, device)
                by = eval_by_source(model, te_loader, device)

                row = {
                    "seed": int(seed),
                    "lambda_kl": float(lam_kl),
                    "lam_sparsity": float(lam_sp),
                    "val_best_acc": float(val_best["acc_fusion"]),
                    "val_best_loss": float(val_best["loss_fusion"]),
                    "test_acc": float(test_stats["acc_fusion"]),
                    "test_loss": float(test_stats["loss_fusion"]),
                    "test_stats": test_stats,
                    "by": by,
                    "hist": hist,
                }
                all_rows.append(row)

                if verbose:
                    syn_acc = float(by["syn"]["acc"]) if by and "syn" in by else float("nan")
                    u1_acc = float(by["u1"]["acc"]) if by and "u1" in by else float("nan")
                    red_acc = float(by["red"]["acc"]) if by and "red" in by else float("nan")
                    print(f"[seed={seed} l_ib={lam_kl:8.1e} l_sp={lam_sp:8.1e}] val_acc={row['val_best_acc']:.3f} val_loss={row['val_best_loss']:.3f} | test_acc={row['test_acc']:.3f} syn={syn_acc:.3f} u1={u1_acc:.3f} red={red_acc:.3f}")

    # ---- aggregate per combo over seeds ----
    combo2rows: Dict[Tuple[float, float], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (float(r["lambda_kl"]), float(r["lam_sparsity"]))
        combo2rows.setdefault(key, []).append(r)

    table: List[Dict[str, Any]] = []
    for (lam_kl, lam_sp), rows in combo2rows.items():
        val_accs = [float(r["val_best_acc"]) for r in rows]
        val_losses = [float(r["val_best_loss"]) for r in rows]
        te_accs = [float(r["test_acc"]) for r in rows]
        te_losses = [float(r["test_loss"]) for r in rows]
        m_va, s_va = mean_std(val_accs)
        m_vl, s_vl = mean_std(val_losses)
        m_ta, s_ta = mean_std(te_accs)
        m_tl, s_tl = mean_std(te_losses)

        table.append({
            "lambda_kl": float(lam_kl),
            "lam_sparsity": float(lam_sp),
            "val_best_acc": m_va, "val_best_acc_std": s_va,
            "val_best_loss": m_vl, "val_best_loss_std": s_vl,
            "test_acc": m_ta, "test_acc_std": s_ta,
            "test_loss": m_tl, "test_loss_std": s_tl,
            "n_seeds": len(rows),
        })

    # ---- pick best combo by MEAN val metric across seeds ----
    if select_by == "loss":
        best_row = min(table, key=lambda r: float(r["val_best_loss"]))
    else:
        best_row = max(table, key=lambda r: float(r["val_best_acc"]))

    best_kl = float(best_row["lambda_kl"])
    best_lsp = float(best_row["lam_sparsity"])

    chosen_rows = combo2rows[(best_kl, best_lsp)]
    # best_val_* should be MEAN across seeds for chosen combo
    best_val_acc = float(best_row["val_best_acc"])
    best_val_loss = float(best_row["val_best_loss"])

    # ---- aggregate test stats dict in same "shape-ish" as before ----
    te_accs = [float(r["test_stats"]["acc_fusion"]) for r in chosen_rows]
    te_losses = [float(r["test_stats"]["loss_fusion"]) for r in chosen_rows]
    m_ta, s_ta = mean_std(te_accs)
    m_tl, s_tl = mean_std(te_losses)
    best_test = {"acc_fusion": m_ta, "loss_fusion": m_tl, "acc_fusion_std": s_ta, "loss_fusion_std": s_tl}

    # ---- aggregate by_source over seeds for chosen combo ----
    best_by: Dict[str, Dict[str, float]] = {s: {"n": 0, "acc": 0.0} for s in _SOURCES}
    for r in chosen_rows:
        by = r["by"]
        for s in _SOURCES:
            if s not in by:
                continue
            n = int(by[s].get("n", 0))
            acc = float(by[s].get("acc", float("nan")))
            if n > 0 and not np.isnan(acc):
                best_by[s]["n"] += n
                best_by[s]["acc"] += acc * n  # weight by count

    for s in _SOURCES:
        n = int(best_by[s]["n"])
        best_by[s]["acc"] = float(best_by[s]["acc"] / n) if n > 0 else float("nan")

    # ---- keep "best_hist" key: store per-seed hists for chosen combo ----
    best_hist = {
        "per_seed": [
            {
                "seed": int(r["seed"]),
                "val_best_acc": float(r["val_best_acc"]),
                "val_best_loss": float(r["val_best_loss"]),
                "test_acc": float(r["test_acc"]),
                "test_loss": float(r["test_loss"]),
                "hist": r["hist"],
            }
            for r in chosen_rows
        ],
        "selected_by": select_by,
        "chosen_combo": {"lambda_kl": best_kl, "lam_sparsity": best_lsp},
    }

    return {
        "best_lambda_kl": best_kl,
        "best_lam_sparsity": best_lsp,
        "best_val_loss": best_val_loss,
        "best_val_acc": best_val_acc,
        "best_test": best_test,
        "best_by": best_by,
        "best_hist": best_hist,
        "table": table,
        "split": split_any,
    }

def sweep_nonoverlap(cfg0: Config, seeds: List[int], kl_vals: List[float], lam_sparsity_vals: List[float], step: float = 0.1, prior_u2: float = 0.0, targets_psyn: List[float] = [0.1, 0.2, 0.3], select_by: SelectBy = "acc", save_path: Optional[str] = None) -> Dict[str, Any]:
    device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg0.device = device
    mkdirp(cfg0.out_dir)

    db = load_json(save_path) if save_path else {}
    db.setdefault("results", {})

    settings = generate_prob_settings(step, prior_u2, targets_psyn)
    print(f"[SWEEP] settings={len(settings)} seeds={seeds} kl_vals={len(kl_vals)} lam_sparsity_vals={len(lam_sparsity_vals)} device={cfg0.device}")
    print("pu1 pu2 pred psyn | tuned_test_acc(mean±std) | tuned_kl | tuned_lsp | syn_acc(mean±std) | status")

    for pu1, pu2, psyn, pred in settings:
        key = prob_key(pu1, pu2, pred, psyn, 0.0)
        if key in db["results"]:
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | ... | ... | ... | ... | SKIP")
            continue

        cfg = Config(**asdict(cfg0))
        set_nonoverlap_signal_probs(cfg, pu1, pu2, pred, psyn, 0.0)

        # NEW: choose (lambda_kl, lam_sparsity) by VAL averaged across seeds
        best = select_best_lambda_kl_lamsparse(cfg, seeds, kl_vals, lam_sparsity_vals, select_by=select_by, verbose=True)

        best_kl = float(best["best_lambda_kl"])
        best_lsp = float(best["best_lam_sparsity"])

        # per-seed results for the chosen combo are stored inside best["best_hist"]["per_seed"]
        per_seed = []
        for r in best["best_hist"]["per_seed"]:
            sd = int(r["seed"])
            by = r.get("by", None)  # may not exist if you didn't store it there
            syn_acc = float("nan")
            if by and "syn" in by:
                syn_acc = float(by["syn"]["acc"])
            per_seed.append({
                "seed": sd,
                "best_lambda_kl": best_kl,
                "best_lam_sparsity": best_lsp,
                "test_acc": float(r["test_acc"]),
                "test_syn_acc": syn_acc,
            })

        # If you prefer the aggregated best_by (weighted across seeds), use it for syn_acc summary:
        # syn_acc_weighted = float(best["best_by"]["syn"]["acc"]) if best.get("best_by") else float("nan")

        accs = [r["test_acc"] for r in per_seed]
        syns = [r["test_syn_acc"] for r in per_seed if not np.isnan(r["test_syn_acc"])]

        m_acc, s_acc = mean_std(accs)
        m_syn, s_syn = mean_std(syns) if len(syns) else (float("nan"), float("nan"))

        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | {m_acc:.3f}±{s_acc:.3f} | {best_kl:.2e} | {best_lsp:.2e} | {m_syn:.3f}±{s_syn:.3f} | ADD")

        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "lam_sparsity_vals": list(lam_sparsity_vals),
            "cfg": asdict(cfg0),
            "per_seed": per_seed,
            "best_combo": {"lambda_kl": best_kl, "lam_sparsity": best_lsp, "best_val_acc": float(best["best_val_acc"]), "best_val_loss": float(best["best_val_loss"])},
            "combo_table": best["table"],        # aggregated over seeds for each (kl,lsp)
            "best_by": best["best_by"],          # aggregated over seeds (weighted by n)
            "best_hist": best["best_hist"],      # contains per-seed histories for chosen combo
            "summary": {
                "learned_tuned":{
                    "test_acc_mean": m_acc, "test_acc_std": s_acc,
                    "best_kl": best_kl,
                    "best_lsp": best_lsp,
                    "test_syn_acc_mean": m_syn, "test_syn_acc_std": s_syn,}
            },
        }
        if save_path:
            save_json(save_path, db)

    if save_path:
        save_json(save_path, db)
    return db


# ============================================================
# Entry point
# ============================================================

def main() -> None:
    cfg = Config()
    cfg.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg.out_dir)

    # quick sanity single run (no sweep)
    # set_nonoverlap_signal_probs(cfg, pu1=0.0, pu2=0.0, pred=0.9, psyn=0.1, pnone=0.0)
    # best = select_best_lambda_kl(cfg, seed=0, kl_vals=[1e-2, 1e-1, 1e0], select_by="acc", verbose=True)
    # print("BEST:", best["best_lambda_kl"], best["best_val_acc"], best["best_test"])

    save_path = os.path.join(cfg.out_dir, "synib_learneduni_sweep.json")
    sweep_nonoverlap(cfg, seeds=[0], kl_vals=[1e-1, 1e0, 1e1, 1e2], lam_sparsity_vals=[0.01, 0.1, 1, 5, 10],
                     step=0.05, targets_psyn=[0.05, 0.1, 0.2], select_by="acc", save_path=save_path)
    print(f"[DONE] saved to {save_path}")

if __name__ == "__main__":
    main()
