
"""
SynIB PID XOR Benchmark (Refactored)
====================================

What this script provides
- Synthetic multimodal dataset with 4 blocks PER DATAPOINT:
    (Noise | Unique | Redundant | Synergy)
  * Unique differs by modality (unique-to-mod0 vs unique-to-mod1).
  * Redundant shares the same latent injected into both modalities.
  * Synergy uses XOR between modality-specific synergy latents.
- Two training methods:
    (A) Main: supervised fusion + unimodal heads
    (B) SynIB: Main + counterfactual KL-to-uniform when SYNERGY block is destroyed
- Sanity checks:
    1) Unimodal head performance on XOR should be ~0.5 when synergy dominates
    2) Block ablations at test time: destroy Unique / Red / Syn / (Unique+Red)
    3) Perturbation detectability test (can a small probe detect "perturbed" inputs?)
- Ready for sweeps:
    - SNR sweeps via (syn_strength, noise_std)
    - Dimension/block-ratio sweeps via (dim, block_fracs)

Run:
  python synib_pid_refactor.py

Outputs:
  - prints per-run metrics and sanity checks
  - writes CSVs for sweeps (if enabled)

Notes:
  - "Destroy block" replaces that block with fresh noise sampled from N(0, noise_std).
  - To avoid easy "coordinate position" shortcuts, enable random_block_positions=True
    which randomizes the block coordinate sets per-sample (mask returned to preserve IDs).
"""
from __future__ import annotations
import os
from datetime import datetime
import copy
from dataclasses import dataclass, asdict, field
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Dict, Any, Optional, Set, List, Tuple, Literal
import json
import time
import numpy as np
from copy import deepcopy

import os
import math
from copy import deepcopy

import numpy as np

MASK_NOISE = 0
MASK_UNIQUE = 1
MASK_RED = 2
MASK_SYN = 3
# DESTROY_MASK = [MASK_NOISE, MASK_UNIQUE_1, MASK_RED, MASK_SYN]
DESTROY_MASK = [MASK_SYN]
INV_DESTROY_MASK = [MASK_RED, MASK_UNIQUE]
Method = Literal["kl_uniform_fusion", "flip_fusion", "fusion_more_than_unimodal"]

_SOURCES = ("u1", "u2", "red", "syn")
_SRC2IDX = {s: i for i, s in enumerate(_SOURCES)}


def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)
def set_global_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def pretty_float(x: float) -> str:
    x = float(x)
    if x == 0.0:
        return "0"
    if 1e-2 <= abs(x) < 1e3:
        s = f"{x:.4f}"
        return s.rstrip("0").rstrip(".")
    return f"{x:.1e}"
def bern_kl_to_uniform_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """KL( Bern(sigmoid(logits)) || Bern(0.5) ), averaged over batch."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    return (p * torch.log(2 * p) + (1 - p) * torch.log(2 * (1 - p))).mean()
@torch.no_grad()
def entropy_from_logits_binary(logits: torch.Tensor) -> torch.Tensor:
    """Binary entropy H(Bern(sigmoid(logits))) averaged over batch."""
    p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
    return (-(p * torch.log(p) + (1 - p) * torch.log(1 - p))).mean()
def _block_sizes(dim: int, frac_unique: float, frac_red: float, frac_syn: float) -> Tuple[int, int, int, int]:
    u = int(round(dim * frac_unique))
    r = int(round(dim * frac_red))
    s = int(round(dim * frac_syn))
    used = u + r + s
    if used > dim:
        overflow = used - dim
        take = min(s, overflow); s -= take; overflow -= take
        if overflow > 0:
            take = min(r, overflow); r -= take; overflow -= take
        if overflow > 0:
            take = min(u, overflow); u -= take; overflow -= take
    noise = dim - (u + r + s)
    return u, r, s, noise
def _choose_blocks(dim: int, u: int, r: int, s: int, *, rng: torch.Generator, randomize: bool) -> Dict[int, torch.Tensor]:
    if (u + r + s) > dim:
        raise ValueError("Block sizes exceed dim.")
    if randomize:
        perm = torch.randperm(dim, generator=rng)
        idx_u = perm[:u]
        idx_r = perm[u:u+r]
        idx_s = perm[u+r:u+r+s]
        idx_n = perm[u+r+s:]
    else:
        idx_u = torch.arange(0, u)
        idx_r = torch.arange(u, u+r)
        idx_s = torch.arange(u+r, u+r+s)
        idx_n = torch.arange(u+r+s, dim)
    return {MASK_UNIQUE: idx_u, MASK_RED: idx_r, MASK_SYN: idx_s, MASK_NOISE: idx_n}
def _parse_subset_key(key: str) -> Set[str]:
    key = key.strip().lower()
    if key in ("none", ""):
        return set()
    return {part.strip() for part in key.split("+") if part.strip()}
def _sample_subset_key(rng: torch.Generator, probs: Dict[str, float]) -> str:
    keys = list(probs.keys())
    if len(keys) == 0:
        raise ValueError("cfg.signal_probs is empty.")
    p = torch.tensor([float(probs[k]) for k in keys])
    if (p < 0).any():
        raise ValueError("cfg.signal_probs contains negative probabilities.")
    s = float(p.sum().item())
    # if abs(s - 1.0) > 1e-6:
    #     raise ValueError(f"cfg.signal_probs must sum to 1. Got sum={s}.")
    idx = torch.multinomial(p, num_samples=1, replacement=True, generator=rng).item()
    return keys[idx]
def _sign_from_bit(bit01: float) -> float:
    return 1.0 if float(bit01) == 1.0 else -1.0
def _multihot_from_sources(active: Set[str]) -> torch.Tensor:
    v = torch.zeros(len(_SOURCES), dtype=torch.float32)
    for s in active:
        if s not in _SRC2IDX:
            raise ValueError(f"Unknown source '{s}'. Allowed: {_SOURCES}")
        v[_SRC2IDX[s]] = 1.0
    return v
def _sources_str(active: Set[str]) -> str:
    if not active:
        return "none"
    return "+".join(sorted(active))
class PID4BlockDataset(Dataset):
    """
    Stores per-sample active sources in:
      - self.source: FloatTensor [n,4] multi-hot in order [u1,u2,red,syn]  (batchable)
      - self.source_str: list[str] human-readable like "u1+syn" (debug only)
    Prints distribution at end of __init__.
    """

    def __init__(self, cfg, n: int, *, seed: int, split: str, train_stats: Optional[Dict[str, Any]] = None, verbose: bool = True):
        super().__init__()
        self.cfg = cfg
        self.n = int(n)

        g = torch.Generator()
        g.manual_seed(int(seed))

        u0, r0, s0, _ = _block_sizes(cfg.dim0, cfg.frac_unique, cfg.frac_red, cfg.frac_syn)
        u1, r1, s1, _ = _block_sizes(cfg.dim1, cfg.frac_unique, cfg.frac_red, cfg.frac_syn)

        with torch.random.fork_rng():
            torch.manual_seed(999)
            self.proj_u0 = torch.randn(cfg.dim0, cfg.latent_u) * 0.5
            self.proj_u1 = torch.randn(cfg.dim1, cfg.latent_u) * 0.5
            self.proj_r0 = torch.randn(cfg.dim0, cfg.latent_r) * 0.5
            self.proj_r1 = torch.randn(cfg.dim1, cfg.latent_r) * 0.5
            self.proj_s0 = torch.randn(cfg.dim0, cfg.latent_s) * 0.5
            self.proj_s1 = torch.randn(cfg.dim1, cfg.latent_s) * 0.5

        # base noise
        self.x0 = torch.randn(self.n, cfg.dim0, generator=g) * float(cfg.noise_std)
        self.x1 = torch.randn(self.n, cfg.dim1, generator=g) * float(cfg.noise_std)

        # masks
        self.mask0 = torch.zeros(self.n, cfg.dim0, dtype=torch.long)
        self.mask1 = torch.zeros(self.n, cfg.dim1, dtype=torch.long)

        # labels
        self.y = torch.zeros(self.n, 1)

        # NEW: sources (batchable + debug)
        self.source = torch.zeros(self.n, len(_SOURCES), dtype=torch.float32)  # [n,4]
        self.source_str: List[str] = [""] * self.n

        # validate probs
        if not hasattr(cfg, "signal_probs") or cfg.signal_probs is None:
            raise ValueError("Config must define cfg.signal_probs (joint table over sources).")
        _ = _sample_subset_key(g, cfg.signal_probs)  # validates
        g.manual_seed(int(seed))  # reset

        # bookkeeping for printing
        subset_counter = Counter()
        marginal_counts = Counter()

        for i in range(self.n):
            b0 = _choose_blocks(cfg.dim0, u0, r0, s0, rng=g, randomize=cfg.random_block_positions)
            b1 = _choose_blocks(cfg.dim1, u1, r1, s1, rng=g, randomize=cfg.random_block_positions)

            key = _sample_subset_key(g, cfg.signal_probs)
            A = _parse_subset_key(key)

            # store sources
            self.source[i] = _multihot_from_sources(A)
            sstr = _sources_str(A)
            self.source_str[i] = sstr

            subset_counter[sstr] += 1
            for s in A:
                marginal_counts[s] += 1

            # sample label first
            y_i = float(torch.rand(1, generator=g).item() > 0.5)
            self.y[i] = y_i

            # u1 -> modality 0 UNIQUE
            if "u1" in A:
                if cfg.unique_strength <= 0:
                    raise ValueError("unique_strength must be > 0 if 'u1' can be active.")
                z_u0 = torch.randn(cfg.latent_u, generator=g) * float(cfg.unique_strength)
                z_u0 = z_u0.abs() * _sign_from_bit(y_i)
                x_u0_full = (self.proj_u0 @ z_u0)
                self.x0[i, b0[MASK_UNIQUE]] = x_u0_full[b0[MASK_UNIQUE]]
                self.mask0[i, b0[MASK_UNIQUE]] = MASK_UNIQUE

            # u2 -> modality 1 UNIQUE
            if "u2" in A:
                if cfg.unique_strength <= 0:
                    raise ValueError("unique_strength must be > 0 if 'u2' can be active.")
                z_u1 = torch.randn(cfg.latent_u, generator=g) * float(cfg.unique_strength)
                z_u1 = z_u1.abs() * _sign_from_bit(y_i)
                x_u1_full = (self.proj_u1 @ z_u1)
                self.x1[i, b1[MASK_UNIQUE]] = x_u1_full[b1[MASK_UNIQUE]]
                self.mask1[i, b1[MASK_UNIQUE]] = MASK_UNIQUE

            # red -> BOTH modalities RED
            if "red" in A:
                if cfg.red_strength <= 0:
                    raise ValueError("red_strength must be > 0 if 'red' can be active.")
                z_r = torch.randn(cfg.latent_r, generator=g) * float(cfg.red_strength)
                z_r = z_r.abs() * _sign_from_bit(y_i)
                x_r0_full = (self.proj_r0 @ z_r)
                x_r1_full = (self.proj_r1 @ z_r)
                self.x0[i, b0[MASK_RED]] = x_r0_full[b0[MASK_RED]]
                self.x1[i, b1[MASK_RED]] = x_r1_full[b1[MASK_RED]]
                self.mask0[i, b0[MASK_RED]] = MASK_RED
                self.mask1[i, b1[MASK_RED]] = MASK_RED

            # syn -> XOR in BOTH modalities SYN
            if "syn" in A:
                if cfg.syn_strength <= 0:
                    raise ValueError("syn_strength must be > 0 if 'syn' can be active.")

                b_s0 = float(torch.rand(1, generator=g).item() > 0.5)
                b_s1 = float(b_s0 != y_i)

                z_s0 = torch.randn(cfg.latent_s, generator=g) * float(cfg.syn_strength)
                z_s1 = torch.randn(cfg.latent_s, generator=g) * float(cfg.syn_strength)
                z_s0 = z_s0.abs() * _sign_from_bit(b_s0)
                z_s1 = z_s1.abs() * _sign_from_bit(b_s1)

                x_s0_full = (self.proj_s0 @ z_s0)
                x_s1_full = (self.proj_s1 @ z_s1)
                self.x0[i, b0[MASK_SYN]] = x_s0_full[b0[MASK_SYN]]
                self.x1[i, b1[MASK_SYN]] = x_s1_full[b1[MASK_SYN]]
                self.mask0[i, b0[MASK_SYN]] = MASK_SYN
                self.mask1[i, b1[MASK_SYN]] = MASK_SYN
        self.stats = self._normalize(split, train_stats)

        # -------------------- print distribution (one-liners) --------------------
        top = ", ".join([f"{k}={c/self.n:.3f}" for k, c in subset_counter.most_common(10)])
        marg = ", ".join([f"{s}={marginal_counts.get(s,0)/self.n:.3f}" for s in _SOURCES])
        if verbose:
            print(f"[DATA:{split}] subsets(top10): {top} | marginals: {marg}")


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

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Dict[str, Any]:
        return {
            "x0": self.x0[i],
            "x1": self.x1[i],
            "y": self.y[i],
            "mask0": self.mask0[i],
            "mask1": self.mask1[i],
            "source": self.source[i],         # FloatTensor shape [4] (batchable)
            "source_str": self.source_str[i], # string (debug; collate will make list[str], fine if you don't tensorize it)
        }

class FusionModel(nn.Module):
    """
    XOR-friendly fusion model:
    - Each modality is encoded.
    - Each modality produces a scalar score t0, t1.
    - Fusion head sees [t0, t1, t0*t1, |t0-t1|] so parity/XOR becomes easy.

    Returns:
      f: fusion logit [B,1]
      u0: unimodal0 logit [B,1]
      u1: unimodal1 logit [B,1]
    """
    def __init__(self, dim0: int, dim1: int, hidden: int = 256, fuse_hidden: int = 128, dropout: float = 0.0):
        super().__init__()

        self.enc0 = nn.Sequential(
            nn.Linear(dim0, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.enc1 = nn.Sequential(
            nn.Linear(dim1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Produce scalar "bit-like" evidence per modality
        self.score0 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.score1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Fusion sees interaction features
        # [t0, t1, t0*t1, |t0-t1|] -> logit
        self.fuse = nn.Sequential(
            nn.Linear(2*hidden, fuse_hidden),
            nn.ReLU(),
            nn.Linear(fuse_hidden, fuse_hidden),
            nn.ReLU(),
            nn.Linear(fuse_hidden, 1),
        )

        self.secondenc0 = nn.Sequential(
            nn.Linear(dim0, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.secondenc1 = nn.Sequential(
            nn.Linear(dim1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.secondfuse = nn.Sequential(
            nn.Linear(2*hidden, fuse_hidden),
            nn.ReLU(),
            nn.Linear(fuse_hidden, fuse_hidden),
            nn.ReLU(),
            nn.Linear(fuse_hidden, 1),
        )

    def forward_logits(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.enc0(x0)
        h1 = self.enc1(x1)

        u0 = self.score0(h0.detach())  # unimodal logit
        u1 = self.score1(h1.detach())  # unimodal logit

        z = torch.cat([h0, h1], dim=-1)
        f = self.fuse(z)  # fusion logit

        return f, u0, u1

    def forward_secondfusion(self, x0: torch.Tensor, x1: torch.Tensor):
        h0 = self.secondenc0(x0)
        h1 = self.secondenc1(x1)
        z = torch.cat([h0, h1], dim=-1)
        f = self.secondfuse(z)  # fusion logit
        return f

@torch.no_grad()
def eval_clean(model: FusionModel, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    hits_f = hits_u0 = hits_u1 = 0
    n = 0
    for b in loader:
        x0, x1 = b["x0"].to(device), b["x1"].to(device)
        y = b["y"].to(device).view(-1)
        f, u0, u1 = model.forward_logits(x0, x1)
        pf, p0, p1 = (f.view(-1) > 0).float(), (u0.view(-1) > 0).float(), (u1.view(-1) > 0).float()
        hits_f += int((pf == y).sum().item())
        hits_u0 += int((p0 == y).sum().item())
        hits_u1 += int((p1 == y).sum().item())
        n += y.numel()
    return {"acc_fusion": hits_f / max(1, n), "acc_uni0": hits_u0 / max(1, n), "acc_uni1": hits_u1 / max(1, n)}
@torch.no_grad()
def eval_block_ablations(model: FusionModel, loader: DataLoader, device: str, *, noise_std: float = 1.0) -> Dict[str, Dict[str, float]]:
    model.eval()
    conds = {
        "clean": None,
        "destroy_unique": (MASK_UNIQUE,),
        "destroy_red": (MASK_RED,),
        "destroy_syn": (MASK_SYN,),
        "destroy_unique+red": (MASK_UNIQUE, MASK_RED),
    }
    out = {}
    for name, blocks in conds.items():
        hits = 0; n = 0; ent = 0.0; nb = 0
        for b in loader:
            x0, x1 = b["x0"].to(device), b["x1"].to(device)
            y = b["y"].to(device).view(-1)
            m0, m1 = b["mask0"].to(device), b["mask1"].to(device)
            if blocks is not None:
                for bid in blocks:
                    x0 = destroy_block(x0, m0, [bid], noise_std=noise_std)
                    x1 = destroy_block(x1, m1, [bid], noise_std=noise_std)
            f, _, _ = model.forward_logits(x0, x1)
            pred = (f.view(-1) > 0).float()
            hits += int((pred == y).sum().item())
            n += y.numel()
            ent += float(entropy_from_logits_binary(f).item())
            nb += 1
        out[name] = {"acc": hits / max(1, n), "entropy": ent / max(1, nb)}
    return out

def _acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Binary accuracy from logits. y expected shape [B,1] or [B].
    """
    yv = y.view(-1)
    pred = (logits.view(-1) > 0).float()
    return float((pred == yv).float().mean().item())
@torch.no_grad()
def _eval_epoch_main( model, loader: DataLoader, device: str, lambda_uni: float ) -> Dict[str, float]:
    model.eval()
    n = 0

    sum_loss_total = 0.0
    sum_loss_f = 0.0
    sum_loss_u0 = 0.0
    sum_loss_u1 = 0.0

    sum_acc_f = 0.0
    sum_acc_u0 = 0.0
    sum_acc_u1 = 0.0

    for b in loader:
        x0 = b["x0"].to(device)
        x1 = b["x1"].to(device)
        y = b["y"].to(device)

        f, u0, u1 = model.forward_logits(x0, x1)

        lf = F.binary_cross_entropy_with_logits(f, y)
        lu0 = F.binary_cross_entropy_with_logits(u0, y)
        lu1 = F.binary_cross_entropy_with_logits(u1, y)
        ltot = lf + float(lambda_uni) * (lu0 + lu1)

        bs = y.size(0)
        n += bs

        sum_loss_total += float(ltot.item()) * bs
        sum_loss_f += float(lf.item()) * bs
        sum_loss_u0 += float(lu0.item()) * bs
        sum_loss_u1 += float(lu1.item()) * bs

        sum_acc_f += _acc_from_logits(f, y) * bs
        sum_acc_u0 += _acc_from_logits(u0, y) * bs
        sum_acc_u1 += _acc_from_logits(u1, y) * bs

    return {
        "loss_total": sum_loss_total / max(1, n),
        "loss_fusion": sum_loss_f / max(1, n),
        "loss_uni0": sum_loss_u0 / max(1, n),
        "loss_uni1": sum_loss_u1 / max(1, n),
        "acc_fusion": sum_acc_f / max(1, n),
        "acc_uni0": sum_acc_u0 / max(1, n),
        "acc_uni1": sum_acc_u1 / max(1, n),
    }
def train_main(cfg, train_loader: DataLoader, device: str, val_loader: Optional[DataLoader] = None ) -> Tuple["FusionModel", Dict[str, Any]]:
    """
    Adds:
      - optional val_dataloader
      - keeps best model based on VAL fusion loss (BCE on fusion logits)
      - tracks train/val losses + accuracies per epoch

    Returns:
      model_best, history
    """
    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "train": [],  # list of dicts
        "val": [],    # list of dicts (empty if no val_loader)
        "best_epoch": None,
        "best_val_fusion_loss": float("inf"),
        "best_val_fusion_acc": 0.0,
    }

    best_state = None

    for epoch in range(cfg.epochs):
        # -------------------- train epoch --------------------
        model.train()

        n = 0
        sum_ltot = 0.0
        sum_lf = 0.0
        sum_lu0 = 0.0
        sum_lu1 = 0.0

        sum_acc_f = 0.0
        sum_acc_u0 = 0.0
        sum_acc_u1 = 0.0

        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y = b["y"].to(device)

            f, u0, u1 = model.forward_logits(x0, x1)

            lf = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            ltot = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            opt.zero_grad()
            ltot.backward()
            opt.step()

            bs = y.size(0)
            n += bs

            sum_ltot += float(ltot.item()) * bs
            sum_lf += float(lf.item()) * bs
            sum_lu0 += float(lu0.item()) * bs
            sum_lu1 += float(lu1.item()) * bs

            sum_acc_f += _acc_from_logits(f, y) * bs
            sum_acc_u0 += _acc_from_logits(u0, y) * bs
            sum_acc_u1 += _acc_from_logits(u1, y) * bs

        train_metrics = {
            "epoch": epoch,
            "loss_total": sum_ltot / max(1, n),
            "loss_fusion": sum_lf / max(1, n),
            "loss_uni0": sum_lu0 / max(1, n),
            "loss_uni1": sum_lu1 / max(1, n),
            "acc_fusion": sum_acc_f / max(1, n),
            "acc_uni0": sum_acc_u0 / max(1, n),
            "acc_uni1": sum_acc_u1 / max(1, n),
        }
        history["train"].append(train_metrics)

        # -------------------- val epoch + checkpoint --------------------
        if val_loader is not None:
            val_metrics = _eval_epoch_main(model, val_loader, device, cfg.lambda_uni)
            val_metrics["epoch"] = epoch
            history["val"].append(val_metrics)

            # Best model based ONLY on fusion loss (not total)
            cur = float(val_metrics["acc_fusion"])
            if cur > history["best_val_fusion_acc"]:
                history["best_val_fusion_acc"] = cur
                history["best_val_fusion_loss"] = val_metrics["loss_fusion"]
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(model.state_dict())

        # Optional: keep your printing style here if you want
        # e.g., print(epoch, train_metrics, val_metrics)

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
def train_mainmask(cfg, train_loader: DataLoader, device: str, val_loader: Optional[DataLoader] = None ) -> Tuple["FusionModel", Dict[str, Any]]:
    """
    Adds:
      - optional val_dataloader
      - keeps best model based on VAL fusion loss (BCE on fusion logits)
      - tracks train/val losses + accuracies per epoch

    Returns:
      model_best, history
    """
    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr)

    history = {
        "train": [],  # list of dicts
        "val": [],    # list of dicts (empty if no val_loader)
        "best_epoch": None,
        "best_val_fusion_loss": float("inf"),
        "best_val_fusion_acc": 0.0,
    }

    best_state = None

    for epoch in range(cfg.epochs):
        # -------------------- train epoch --------------------
        model.train()

        n = 0
        sum_ltot = 0.0
        sum_lf = 0.0
        sum_lu0 = 0.0
        sum_lu1 = 0.0

        sum_acc_f = 0.0
        sum_acc_u0 = 0.0
        sum_acc_u1 = 0.0

        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y = b["y"].to(device)

            # m0_random = torch.rand_like(x0)
            # m1_random = torch.rand_like(x1)
            # m0_random = (m0_random > cfg.random_mask_proportion).float()#Inverse mask
            # m1_random = (m1_random > cfg.random_mask_proportion).float()#Inverse mask
            # x0_t = destroy_block(x0, m0_random, 1, noise_std=1.0)
            # x1_t = destroy_block(x1, m1_random, 1, noise_std=1.0)


            m0 = b["mask0"].to(device)
            m1 = b["mask1"].to(device)
            x0_t = destroy_block(x0, m0, INV_DESTROY_MASK, noise_std=1.0)
            x1_t = destroy_block(x1, m1, INV_DESTROY_MASK, noise_std=1.0)

            # x0_t = torch.concatenate([x0, x0_t],dim=0)
            # x1_t = torch.concatenate([x1, x1_t],dim=0)
            # y_t = torch.concatenate([y,y],dim=0)


            x0_t = torch.concatenate([x0, x0_t, x0],dim=0)
            x1_t = torch.concatenate([x1, x1, x1_t],dim=0)
            y_t = torch.concatenate([y,y,y],dim=0)

            f, u0, u1 = model.forward_logits(x0_t, x1_t)

            lf = F.binary_cross_entropy_with_logits(f, y_t)
            lu0 = F.binary_cross_entropy_with_logits(u0, y_t)
            lu1 = F.binary_cross_entropy_with_logits(u1, y_t)
            ltot = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            opt.zero_grad()
            ltot.backward()
            opt.step()

            bs = y_t.size(0)
            n += bs

            sum_ltot += float(ltot.item()) * bs
            sum_lf += float(lf.item()) * bs
            sum_lu0 += float(lu0.item()) * bs
            sum_lu1 += float(lu1.item()) * bs

            sum_acc_f += _acc_from_logits(f, y_t) * bs
            sum_acc_u0 += _acc_from_logits(u0, y_t) * bs
            sum_acc_u1 += _acc_from_logits(u1, y_t) * bs

        train_metrics = {
            "epoch": epoch,
            "loss_total": sum_ltot / max(1, n),
            "loss_fusion": sum_lf / max(1, n),
            "loss_uni0": sum_lu0 / max(1, n),
            "loss_uni1": sum_lu1 / max(1, n),
            "acc_fusion": sum_acc_f / max(1, n),
            "acc_uni0": sum_acc_u0 / max(1, n),
            "acc_uni1": sum_acc_u1 / max(1, n),
        }
        history["train"].append(train_metrics)

        # -------------------- val epoch + checkpoint --------------------
        if val_loader is not None:
            val_metrics = _eval_epoch_main(model, val_loader, device, cfg.lambda_uni)
            val_metrics["epoch"] = epoch
            history["val"].append(val_metrics)

            # Best model based ONLY on fusion loss (not total)
            cur = float(val_metrics["acc_fusion"])
            if cur > history["best_val_fusion_acc"]:
                history["best_val_fusion_acc"] = cur
                history["best_val_fusion_loss"] = val_metrics["loss_fusion"]
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(model.state_dict())

        # Optional: keep your printing style here if you want
        # e.g., print(epoch, train_metrics, val_metrics)

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

@torch.no_grad()
def _eval_epoch_synib( model,    loader: DataLoader,    device: str,    cfg) -> Dict[str, float]:
    model.eval()
    n = 0

    sum_l_base = 0.0
    sum_l_f = 0.0
    sum_l_u0 = 0.0
    sum_l_u1 = 0.0
    sum_l_cf = 0.0
    sum_l_total = 0.0

    sum_acc_f = 0.0
    sum_acc_u0 = 0.0
    sum_acc_u1 = 0.0

    for b in loader:
        x0 = b["x0"].to(device)
        x1 = b["x1"].to(device)
        y  = b["y"].to(device)
        m0 = b["mask0"].to(device)
        m1 = b["mask1"].to(device)

        f, u0, u1 = model.forward_logits(x0, x1)

        lf  = F.binary_cross_entropy_with_logits(f, y)
        lu0 = F.binary_cross_entropy_with_logits(u0, y)
        lu1 = F.binary_cross_entropy_with_logits(u1, y)
        l_base = lf + float(cfg.lambda_uni) * (lu0 + lu1)

        # counterfactual: destroy SYNERGY and push fusion uncertain
        x0_t = destroy_block(x0, m0, DESTROY_MASK, noise_std=cfg.noise_std)
        x1_t = destroy_block(x1, m1, DESTROY_MASK, noise_std=cfg.noise_std)
        f_t0, _, _ = model.forward_logits(x0_t, x1)
        f_t1, _, _ = model.forward_logits(x0, x1_t)
        l_cf = bern_kl_to_uniform_from_logits(f_t0) + bern_kl_to_uniform_from_logits(f_t1)

        l_total = l_base + float(cfg.lambda_kl) * l_cf

        bs = y.size(0)
        n += bs

        sum_l_base += float(l_base.item()) * bs
        sum_l_f += float(lf.item()) * bs
        sum_l_u0 += float(lu0.item()) * bs
        sum_l_u1 += float(lu1.item()) * bs
        sum_l_cf += float(l_cf.item()) * bs
        sum_l_total += float(l_total.item()) * bs

        sum_acc_f += _acc_from_logits(f, y) * bs
        sum_acc_u0 += _acc_from_logits(u0, y) * bs
        sum_acc_u1 += _acc_from_logits(u1, y) * bs

    return {
        "loss_total": sum_l_total / max(1, n),
        "loss_base":  sum_l_base / max(1, n),
        "loss_fusion": sum_l_f / max(1, n),
        "loss_uni0":  sum_l_u0 / max(1, n),
        "loss_uni1":  sum_l_u1 / max(1, n),
        "loss_cf":    sum_l_cf / max(1, n),
        "acc_fusion": sum_acc_f / max(1, n),
        "acc_uni0":   sum_acc_u0 / max(1, n),
        "acc_uni1":   sum_acc_u1 / max(1, n),
    }
def train_synib( cfg, train_loader: DataLoader, device: str, val_loader: Optional[DataLoader] = None ) -> Tuple["FusionModel", Dict[str, Any]]:
    """
    Adds:
      - optional val_dataloader
      - keeps best model based on VAL fusion loss (BCE on fusion logits, clean)
      - tracks train/val losses + accuracies per epoch (including cf/inv terms)

    Returns:
      model_best, history
    """
    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay_synib)

    history = {
        "train": [],
        "val": [],
        "best_epoch": None,
        "best_val_fusion_acc": 0.0,
        "best_val_fusion_loss": float("inf"),
    }
    best_state = None

    for epoch in range(cfg.epochs):
        # -------------------- train epoch --------------------
        model.train()

        n = 0
        sum_l_base = 0.0
        sum_l_f = 0.0
        sum_l_u0 = 0.0
        sum_l_u1 = 0.0
        sum_l_cf = 0.0
        sum_l_inv = 0.0
        sum_l_total = 0.0

        sum_acc_f = 0.0
        sum_acc_u0 = 0.0
        sum_acc_u1 = 0.0

        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y  = b["y"].to(device)
            m0 = b["mask0"].to(device)
            m1 = b["mask1"].to(device)

            f, u0, u1 = model.forward_logits(x0, x1)

            lf  = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            l_base = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            # counterfactual: destroy SYNERGY block => uncertainty
            x0_t = destroy_block(x0, m0, DESTROY_MASK, noise_std=1.0)
            x1_t = destroy_block(x1, m1, DESTROY_MASK, noise_std=1.0)
            f_t0, _, _ = model.forward_logits(x0_t, x1)
            f_t1, _, _ = model.forward_logits(x0, x1_t)
            l_cf = bern_kl_to_uniform_from_logits(f_t0) + bern_kl_to_uniform_from_logits(f_t1)

            # optional invariance to shortcut destruction
            l_inv = torch.tensor(0.0, device=device)
            if float(cfg.lambda_shortcut_inv) > 0:
                with torch.no_grad():
                    f_clean = f.detach()
                x0_inv, x1_inv = x0, x1
                if cfg.inv_destroy_unique:
                    x0_inv = destroy_block(x0_inv, m0, MASK_UNIQUE, noise_std=1.0)
                    x1_inv = destroy_block(x1_inv, m1, MASK_UNIQUE, noise_std=1.0)
                if cfg.inv_destroy_red:
                    x0_inv = destroy_block(x0_inv, m0, MASK_RED, noise_std=1.0)
                    x1_inv = destroy_block(x1_inv, m1, MASK_RED, noise_std=1.0)
                f_inv, _, _ = model.forward_logits(x0_inv, x1_inv)
                l_inv = F.mse_loss(f_inv, f_clean)

            l_total = l_base + float(cfg.lambda_kl) * l_cf #+ float(cfg.lambda_shortcut_inv) * l_inv

            opt.zero_grad()
            l_total.backward()
            opt.step()

            bs = y.size(0)
            n += bs

            sum_l_base += float(l_base.item()) * bs
            sum_l_f += float(lf.item()) * bs
            sum_l_u0 += float(lu0.item()) * bs
            sum_l_u1 += float(lu1.item()) * bs
            sum_l_cf += float(l_cf.item()) * bs
            sum_l_inv += float(l_inv.item()) * bs
            sum_l_total += float(l_total.item()) * bs

            sum_acc_f += _acc_from_logits(f, y) * bs
            sum_acc_u0 += _acc_from_logits(u0, y) * bs
            sum_acc_u1 += _acc_from_logits(u1, y) * bs

        train_metrics = {
            "epoch": epoch,
            "loss_total": sum_l_total / max(1, n),
            "loss_base":  sum_l_base / max(1, n),
            "loss_fusion": sum_l_f / max(1, n),
            "loss_uni0":  sum_l_u0 / max(1, n),
            "loss_uni1":  sum_l_u1 / max(1, n),
            "loss_cf":    sum_l_cf / max(1, n),
            "loss_inv":   sum_l_inv / max(1, n),
            "acc_fusion": sum_acc_f / max(1, n),
            "acc_uni0":   sum_acc_u0 / max(1, n),
            "acc_uni1":   sum_acc_u1 / max(1, n),
        }
        history["train"].append(train_metrics)

        # -------------------- val epoch + checkpoint --------------------
        if val_loader is not None:
            val_metrics = _eval_epoch_synib(model, val_loader, device, cfg)
            val_metrics["epoch"] = epoch
            history["val"].append(val_metrics)

            # Best model based ONLY on clean fusion loss
            cur = float(val_metrics["acc_fusion"])
            if cur > history["best_val_fusion_acc"]:
                history["best_val_fusion_acc"] = cur
                history["best_val_fusion_loss"] = val_metrics["loss_fusion"]
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(model.state_dict())

        # Optional: print progress here if you want
        # print(f"Epoch {epoch}: train_fusion_loss={train_metrics['loss_fusion']:.4f} "
        #       f"train_acc_f={train_metrics['acc_fusion']:.3f} "
        #       f"val_fusion_loss={history['val'][-1]['loss_fusion'] if val_loader else float('nan'):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
def train_synib_random( cfg, train_loader: DataLoader, device: str, val_loader: Optional[DataLoader] = None ) -> Tuple["FusionModel", Dict[str, Any]]:

    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay_synib)

    history = {
        "train": [],
        "val": [],
        "best_epoch": None,
        "best_val_fusion_acc": 0.0,
        "best_val_fusion_loss": float("inf"),
    }
    best_state = None

    for epoch in range(cfg.epochs):
        # -------------------- train epoch --------------------
        model.train()

        n = 0
        sum_l_base = 0.0
        sum_l_f = 0.0
        sum_l_u0 = 0.0
        sum_l_u1 = 0.0
        sum_l_cf = 0.0
        sum_l_inv = 0.0
        sum_l_total = 0.0

        sum_acc_f = 0.0
        sum_acc_u0 = 0.0
        sum_acc_u1 = 0.0

        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y  = b["y"].to(device)
            m0 = b["mask0"].to(device)
            m1 = b["mask1"].to(device)

            f, u0, u1 = model.forward_logits(x0, x1)

            lf  = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            l_base = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            # counterfactual: destroy SYNERGY block => uncertainty
            m0_random = torch.rand_like(x0)
            m1_random = torch.rand_like(x1)
            m0_random = (m0_random<cfg.random_mask_proportion).float()
            m1_random = (m1_random<cfg.random_mask_proportion).float()

            x0_t = destroy_block(x0, m0_random, 1, noise_std=1.0)
            x1_t = destroy_block(x1, m1_random, 1, noise_std=1.0)
            f_t0, _, _ = model.forward_logits(x0_t, x1)
            f_t1, _, _ = model.forward_logits(x0, x1_t)
            l_cf = bern_kl_to_uniform_from_logits(f_t0) + bern_kl_to_uniform_from_logits(f_t1)

            l_total = l_base + float(cfg.lambda_kl) * l_cf #+ float(cfg.lambda_shortcut_inv) * l_inv

            opt.zero_grad()
            l_total.backward()
            opt.step()

            bs = y.size(0)
            n += bs

            sum_l_base += float(l_base.item()) * bs
            sum_l_f += float(lf.item()) * bs
            sum_l_u0 += float(lu0.item()) * bs
            sum_l_u1 += float(lu1.item()) * bs
            sum_l_cf += float(l_cf.item()) * bs
            sum_l_total += float(l_total.item()) * bs

            sum_acc_f += _acc_from_logits(f, y) * bs
            sum_acc_u0 += _acc_from_logits(u0, y) * bs
            sum_acc_u1 += _acc_from_logits(u1, y) * bs

        train_metrics = {
            "epoch": epoch,
            "loss_total": sum_l_total / max(1, n),
            "loss_base":  sum_l_base / max(1, n),
            "loss_fusion": sum_l_f / max(1, n),
            "loss_uni0":  sum_l_u0 / max(1, n),
            "loss_uni1":  sum_l_u1 / max(1, n),
            "loss_cf":    sum_l_cf / max(1, n),
            "acc_fusion": sum_acc_f / max(1, n),
            "acc_uni0":   sum_acc_u0 / max(1, n),
            "acc_uni1":   sum_acc_u1 / max(1, n),
        }
        history["train"].append(train_metrics)

        # -------------------- val epoch + checkpoint --------------------
        if val_loader is not None:
            val_metrics = _eval_epoch_synib(model, val_loader, device, cfg)
            val_metrics["epoch"] = epoch
            history["val"].append(val_metrics)

            # Best model based ONLY on clean fusion loss
            cur = float(val_metrics["acc_fusion"])
            if cur > history["best_val_fusion_acc"]:
                history["best_val_fusion_acc"] = cur
                history["best_val_fusion_loss"] = val_metrics["loss_fusion"]
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(model.state_dict())

        # Optional: print progress here if you want
        # print(f"Epoch {epoch}: train_fusion_loss={train_metrics['loss_fusion']:.4f} "
        #       f"train_acc_f={train_metrics['acc_fusion']:.3f} "
        #       f"val_fusion_loss={history['val'][-1]['loss_fusion'] if val_loader else float('nan'):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
def train_synib_randomdiff( cfg, train_loader: DataLoader, device: str, val_loader: Optional[DataLoader] = None ) -> Tuple["FusionModel", Dict[str, Any]]:

    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay_synib)

    history = {
        "train": [],
        "val": [],
        "best_epoch": None,
        "best_val_fusion_acc": 0.0,
        "best_val_fusion_loss": float("inf"),
    }
    best_state = None

    for epoch in range(cfg.epochs):
        # -------------------- train epoch --------------------
        model.train()

        n = 0
        sum_l_base = 0.0
        sum_l_f = 0.0
        sum_l_u0 = 0.0
        sum_l_u1 = 0.0
        sum_l_cf = 0.0
        sum_l_inv = 0.0
        sum_l_total = 0.0

        sum_acc_f = 0.0
        sum_acc_u0 = 0.0
        sum_acc_u1 = 0.0

        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y  = b["y"].to(device)
            m0 = b["mask0"].to(device)
            m1 = b["mask1"].to(device)

            f, u0, u1 = model.forward_logits(x0, x1)

            lf  = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            l_base = lf + float(cfg.lambda_uni) * (lu0 + lu1)


            def make_tilde_diff(z):
                K = cfg.K
                cosine_s = cfg.cosine_s
                p_min = cfg.p_min
                p_max = cfg.p_max

                def _get_diff_p(t):
                    """
                    Returns p(t) shaped as a scalar tensor.
                    p(t) ~= alpha_bar(t) with cosine schedule:
                      alpha_bar(t) = cos^2( ((t/T)+s)/(1+s) * pi/2 )
                    This decreases from ~1 to ~0, matching increasing noise over time.

                    Scale to [p_min, p_max].
                    """
                    u = (torch.as_tensor(t) / (K - 1)).clamp(0.0, 1.0)

                    s = cosine_s
                    alpha_bar = torch.cos(((u + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2  # ~1 -> ~0
                    p = p_min + (p_max - p_min) * alpha_bar
                    return p

                def make_keep(z, p):
                    return (torch.rand_like(z) < p).to(z.dtype)


                keep = torch.cat([make_keep(z, _get_diff_p(k)) for k in range(K)], dim=0)
                return keep

            def repeat_k(z, K):
                return z.unsqueeze(0).expand(K, *z.shape).reshape(K * z.shape[0], *z.shape[1:])

            m0_random = make_tilde_diff(x0)
            m1_random = make_tilde_diff(x1)
            x0 = repeat_k(x0, cfg.K)
            x1 = repeat_k(x1, cfg.K)

            x0_t = destroy_block(x0, m0_random, 1, noise_std=1.0)
            x1_t = destroy_block(x1, m1_random, 1, noise_std=1.0)

            f_t0, _, _ = model.forward_logits(x0_t, x1)
            f_t1, _, _ = model.forward_logits(x0, x1_t)
            l_cf = bern_kl_to_uniform_from_logits(f_t0) + bern_kl_to_uniform_from_logits(f_t1)

            # f_t, _, _ = model.forward_logits(x0_t, x1_t)
            # l_cf = bern_kl_to_uniform_from_logits(f_t)

            l_total = l_base + float(cfg.lambda_kl) * l_cf #+ float(cfg.lambda_shortcut_inv) * l_inv

            opt.zero_grad()
            l_total.backward()
            opt.step()

            bs = y.size(0)
            n += bs

            sum_l_base += float(l_base.item()) * bs
            sum_l_f += float(lf.item()) * bs
            sum_l_u0 += float(lu0.item()) * bs
            sum_l_u1 += float(lu1.item()) * bs
            sum_l_cf += float(l_cf.item()) * bs
            sum_l_total += float(l_total.item()) * bs

            sum_acc_f += _acc_from_logits(f, y) * bs
            sum_acc_u0 += _acc_from_logits(u0, y) * bs
            sum_acc_u1 += _acc_from_logits(u1, y) * bs

        train_metrics = {
            "epoch": epoch,
            "loss_total": sum_l_total / max(1, n),
            "loss_base":  sum_l_base / max(1, n),
            "loss_fusion": sum_l_f / max(1, n),
            "loss_uni0":  sum_l_u0 / max(1, n),
            "loss_uni1":  sum_l_u1 / max(1, n),
            "loss_cf":    sum_l_cf / max(1, n),
            "acc_fusion": sum_acc_f / max(1, n),
            "acc_uni0":   sum_acc_u0 / max(1, n),
            "acc_uni1":   sum_acc_u1 / max(1, n),
        }
        history["train"].append(train_metrics)

        # -------------------- val epoch + checkpoint --------------------
        if val_loader is not None:
            val_metrics = _eval_epoch_synib(model, val_loader, device, cfg)
            val_metrics["epoch"] = epoch
            history["val"].append(val_metrics)

            # Best model based ONLY on clean fusion loss
            cur = float(val_metrics["acc_fusion"])
            if cur > history["best_val_fusion_acc"]:
                history["best_val_fusion_acc"] = cur
                history["best_val_fusion_loss"] = val_metrics["loss_fusion"]
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(model.state_dict())

        # Optional: print progress here if you want
        # print(f"Epoch {epoch}: train_fusion_loss={train_metrics['loss_fusion']:.4f} "
        #       f"train_acc_f={train_metrics['acc_fusion']:.3f} "
        #       f"val_fusion_loss={history['val'][-1]['loss_fusion'] if val_loader else float('nan'):.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history




def _bern_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(Bern(p) || Bern(q)), elementwise then mean."""
    p = p.clamp(eps, 1 - eps)
    q = q.clamp(eps, 1 - eps)
    return (p * (p / q).log() + (1 - p) * ((1 - p) / (1 - q)).log()).mean()
def _bern_kl_to_uniform(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """KL(Bern(p) || Bern(0.5)) = p log(2p) + (1-p) log(2(1-p))."""
    p = p.clamp(eps, 1 - eps)
    return (p * (2 * p).log() + (1 - p) * (2 * (1 - p)).log()).mean()
@torch.no_grad()
def _freeze_model(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad_(False)
def get_learned_destroy_mask( model: torch.nn.Module, x0: torch.Tensor, x1: torch.Tensor,method: Method, device: str, *, steps: int = 80, lr: float = 5e-2,  tau: float = 1.0,  noise_std: float = 1.0,  lam_sparsity: float = 1e-2,  alpha_unimodal: float = 1.0,  hard: bool = False,  hard_thresh: float = 0.5,) -> Dict[str, torch.Tensor]:
    """
    Sequential mask learning:
      1) learn g0 while only x0 is destroyed (x1 clean) -> unimodal penalty uses u0 only
      2) learn g1 while only x1 is destroyed (x0 clean) -> unimodal penalty uses u1 only
    """
    model = model.to(device)
    model.eval()
    _freeze_model(model)

    x0 = x0.to(device)
    x1 = x1.to(device)
    B, d0 = x0.shape
    _, d1 = x1.shape

    with torch.no_grad():
        f_clean, u0_clean, u1_clean = model.forward_logits(x0, x1)
        p_f_clean = torch.sigmoid(f_clean).view(-1)
        p_u0_clean = torch.sigmoid(u0_clean).view(-1)
        p_u1_clean = torch.sigmoid(u1_clean).view(-1)

    def _apply_destroy(x: torch.Tensor, g_row: torch.Tensor) -> torch.Tensor:
        ns = float(noise_std) if noise_std is not None else 1.0
        eps = torch.randn_like(x) * ns
        return (1.0 - g_row) * x + g_row * eps

    def _obj_one(
        method,
        which: str,                 # "x0" or "x1"
        p_f_t: torch.Tensor,        # [B]
        p_u_t: torch.Tensor,        # [B] (corresponding unimodal only)
        sparsity: torch.Tensor,     # scalar
        lam_sparsity
    ) -> torch.Tensor:
        if method == "kl_uniform_fusion":
            return _bern_kl_to_uniform(p_f_t) + float(lam_sparsity) * sparsity

        if method == "flip_fusion":
            div = _bern_kl(p_f_clean, p_f_t)
            return -div + float(lam_sparsity) * sparsity

        if method == "fusion_more_than_unimodal":
            div_f = _bern_kl(p_f_clean, p_f_t)

            if which == "x0":
                div_u = _bern_kl(p_u0_clean, p_u_t)
            else:
                div_u = _bern_kl(p_u1_clean, p_u_t)

            score = div_f - float(alpha_unimodal) * div_u
            return -score + float(lam_sparsity) * sparsity

        if method == "unimodal":
            if which == "x0":
                div_u = _bern_kl(p_u0_clean, p_u_t)
            else:
                div_u = _bern_kl(p_u1_clean, p_u_t)

            score = div_u
            return -score + float(lam_sparsity) * sparsity

        raise ValueError(f"Unknown method: {method}")

    def _learn_one(which: str, method: str) -> torch.Tensor:
        d = d0 if which == "x0" else d1
        ell = torch.zeros(d, device=device, requires_grad=True)
        opt = torch.optim.Adam([ell], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
        patience = 10
        small_change = 1e-4
        prev_g = None
        eta_lam = 0.02  # 0.01–0.05 works well
        s_target = 0.20  # choose desired sparsity
        lam = eta_lam

        for _ in range(int(steps)):
            g = torch.sigmoid(ell / float(tau)).view(1, d)  # [1,d]
            if which == "x0":
                x0_t, x1_t = _apply_destroy(x0, g), x1
            else:
                x0_t, x1_t = x0, _apply_destroy(x1, g)

            f_t, u0_t, u1_t = model.forward_logits(x0_t, x1_t)
            f_t = model.forward_secondfusion(x0_t, x1_t)
            p_f_t = torch.sigmoid(f_t).view(-1)

            # only the corresponding unimodal
            p_u_t = torch.sigmoid(u0_t).view(-1) if which == "x0" else torch.sigmoid(u1_t).view(-1)

            sparsity = g.mean()
            obj = _obj_one(method=method, which=which, p_f_t=p_f_t, p_u_t=p_u_t, sparsity=sparsity, lam_sparsity=lam)

            opt.zero_grad(set_to_none=True)
            obj.backward()
            torch.nn.utils.clip_grad_norm_([ell], 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                g_now = torch.sigmoid(ell / tau)
                if prev_g is not None:
                    dg = (g_now - prev_g).abs().mean().item()
                    stall = stall + 1 if dg < small_change else 0
                    if stall >= patience:
                        break
                prev_g = g_now.clone()
                lam = max(0.0, lam + eta_lam * (sparsity.item() - s_target))

        return torch.sigmoid(ell / float(tau)).detach()

    g0 = _learn_one("x0", "fusion_more_than_unimodal")
    g1 = _learn_one("x1", "fusion_more_than_unimodal")

    g0_uni = _learn_one("x0", "unimodal")
    g1_uni = _learn_one("x1", "unimodal")

    if hard:
        thr = float(hard_thresh)
        g0 = (g0 > thr).float()
        g1 = (g1 > thr).float()
        g0_uni = (g0_uni > thr).float()
        g1_uni = (g1_uni > thr).float()

    return {"g0": g0, "g1": g1, "g0_uni": g0_uni, "g1_uni": g1_uni}
def get_learned_destroy_unimask( model: torch.nn.Module, x0: torch.Tensor, x1: torch.Tensor,method: Method, device: str, *, steps: int = 80, lr: float = 5e-2,  tau: float = 1.0,  noise_std: float = 1.0,  lam_sparsity: float = 1e-2,  alpha_unimodal: float = 1.0,  hard: bool = False,  hard_thresh: float = 0.5,) -> Dict[str, torch.Tensor]:
    """
    Sequential mask learning:
      1) learn g0 while only x0 is destroyed (x1 clean) -> unimodal penalty uses u0 only
      2) learn g1 while only x1 is destroyed (x0 clean) -> unimodal penalty uses u1 only
    """
    model = model.to(device)
    model.eval()
    _freeze_model(model)

    x0 = x0.to(device)
    x1 = x1.to(device)
    B, d0 = x0.shape
    _, d1 = x1.shape

    with torch.no_grad():
        f_clean, u0_clean, u1_clean = model.forward_logits(x0, x1)
        p_f_clean = torch.sigmoid(f_clean).view(-1)
        p_u0_clean = torch.sigmoid(u0_clean).view(-1)
        p_u1_clean = torch.sigmoid(u1_clean).view(-1)

    def _apply_destroy(x: torch.Tensor, g_row: torch.Tensor) -> torch.Tensor:
        ns = float(noise_std) if noise_std is not None else 1.0
        eps = torch.randn_like(x) * ns
        return (1.0 - g_row) * x + g_row * eps

    def _obj_one(
        method,
        which: str,                 # "x0" or "x1"
        p_f_t: torch.Tensor,        # [B]
        p_u_t: torch.Tensor,        # [B] (corresponding unimodal only)
        sparsity: torch.Tensor,     # scalar
        lam_sparsity
    ) -> torch.Tensor:
        if method == "kl_uniform_fusion":
            return _bern_kl_to_uniform(p_f_t) + float(lam_sparsity) * sparsity

        if method == "flip_fusion":
            div = _bern_kl(p_f_clean, p_f_t)
            return -div + float(lam_sparsity) * sparsity

        if method == "fusion_more_than_unimodal":
            div_f = _bern_kl(p_f_clean, p_f_t)

            if which == "x0":
                div_u = _bern_kl(p_u0_clean, p_u_t)
            else:
                div_u = _bern_kl(p_u1_clean, p_u_t)

            score = div_f - float(alpha_unimodal) * div_u
            return -score + float(lam_sparsity) * sparsity

        if method == "unimodal":
            if which == "x0":
                div_u =  _bern_kl_to_uniform(p_u_t)
                # div_u = _bern_kl(p_u0_clean, p_u_t)
            else:
                div_u =  _bern_kl_to_uniform(p_u_t)
                # div_u = _bern_kl(p_u1_clean, p_u_t)

            score = div_u
            return -score + float(lam_sparsity) * sparsity

        raise ValueError(f"Unknown method: {method}")

    def _learn_one(which: str, method: str) -> torch.Tensor:
        d = d0 if which == "x0" else d1
        ell = torch.rand((B,d), device=device, requires_grad=True)
        opt = torch.optim.Adam([ell], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=lr * 0.1)
        patience = 10
        small_change = 1e-4
        prev_g = None
        lam = 0.1
        # eta_lam = 0.02  # 0.01–0.05 works well
        # s_target = 0.20  # choose desired sparsity
        # lam = eta_lam

        for _ in range(int(steps)):
            g = torch.sigmoid(ell / float(tau)).view(B, d)  # [1,d]
            if which == "x0":
                x0_t, x1_t = _apply_destroy(x0, g), x1
            else:
                x0_t, x1_t = x0, _apply_destroy(x1, g)

            f_t, u0_t, u1_t = model.forward_logits(x0_t, x1_t)
            p_f_t = torch.sigmoid(f_t).view(-1)

            # only the corresponding unimodal
            p_u_t = torch.sigmoid(u0_t).view(-1) if which == "x0" else torch.sigmoid(u1_t).view(-1)

            sparsity = g.mean()
            obj = _obj_one(method=method, which=which, p_f_t=p_f_t, p_u_t=p_u_t, sparsity=sparsity, lam_sparsity=lam)

            opt.zero_grad(set_to_none=True)
            obj.backward()
            torch.nn.utils.clip_grad_norm_([ell], 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                g_now = torch.sigmoid(ell / tau)
                if prev_g is not None:
                    dg = (g_now - prev_g).abs().mean().item()
                    stall = stall + 1 if dg < small_change else 0
                    if stall >= patience:
                        break
                prev_g = g_now.clone()
                # lam = max(0.0, lam + eta_lam * (sparsity.item() - s_target))

        return torch.sigmoid(ell / float(tau)).detach()

    g0 = _learn_one("x0", "fusion_more_than_unimodal")
    g1 = _learn_one("x1", "fusion_more_than_unimodal")

    g0_uni = _learn_one("x0", "unimodal")
    g1_uni = _learn_one("x1", "unimodal")

    if hard:
        thr = float(hard_thresh)
        g0 = (g0 > thr).float()
        g1 = (g1 > thr).float()
        g0_uni = (g0_uni > thr).float()
        g1_uni = (g1_uni > thr).float()

    return {"g0": g0, "g1": g1, "g0_uni": g0_uni, "g1_uni": g1_uni}
def _apply_learned_destroy(x: torch.Tensor, g: torch.Tensor, *, noise_std: float = 1.0) -> torch.Tensor:
    """
    Differentiable "destroy" using a learned soft gate g in [0,1]^D.
    x: [B,D], g: [D] or [1,D]
    returns x_t: [B,D]
    """
    if g.dim() == 1:
        g = g.view(1, -1)
    g = g.to(x.device).type_as(x)
    eps = torch.randn_like(x) * float(noise_std)
    return (1 - g) * x + g * eps
def _freeze_model_params(model: torch.nn.Module):
    """
    Temporarily disable grads for model params (for mask optimization), then restore.
    """
    req = [p.requires_grad for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(False)
    return req
def _restore_model_params(model: torch.nn.Module, req_list):
    for p, r in zip(model.parameters(), req_list):
        p.requires_grad_(r)
def _gt_synergy_from_blockmask(block_mask: torch.Tensor, mask=MASK_SYN) -> torch.Tensor:
    # block_mask: int tensor, same shape as x (e.g., (B,D) or (B,T,D))
    if isinstance(mask, int):
        return (block_mask == mask)
    elif isinstance(mask, list):
        # Initialize a boolean mask of all False
        synergy_mask = torch.zeros_like(block_mask, dtype=torch.bool)
        for m in mask:
            # Union of all specified mask indices
            synergy_mask |= (block_mask == m)
        return synergy_mask
    else:
        raise ValueError(f"Unsupported mask type: {type(mask)}")
@torch.no_grad()
def _iou_binary(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> float:
    # pred/gt: bool tensors, same shape
    inter = (pred & gt).sum().float()
    union = (pred | gt).sum().float()
    return float((inter / (union + eps)).item())
def _eval_epoch_synib_learned(    model: "FusionModel",    loader: DataLoader,    device: str,    cfg,) -> Dict[str, float]:
    model.eval()

    n = n_syn = 0
    sum_l_f = sum_l_u0 = sum_l_u1 = 0.0
    sum_l_cf = sum_l_inv = 0.0
    sum_l_base = sum_l_total = 0.0
    sum_acc_f = sum_acc_f_syn = sum_acc_u0 = sum_acc_u1 = 0.0

    for b in loader:
        x0 = b["x0"].to(device)
        x1 = b["x1"].to(device)
        y  = b["y"].to(device)

        with torch.no_grad():
            f, u0, u1 = model.forward_logits(x0, x1)
            lf  = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            l_base = lf + float(cfg.lambda_uni) * (lu0 + lu1)

        req = _freeze_model_params(model)
        was_training = model.training
        model.eval()
        try:
            with torch.enable_grad():
                masks = get_learned_destroy_unimask(
                    model,
                    x0.detach(),
                    x1.detach(),
                    method=cfg.learned_mask_method,
                    device=device,
                    steps=cfg.learned_mask_steps,
                    lr=cfg.learned_mask_lr,
                    tau=cfg.learned_mask_tau,
                    noise_std=cfg.learned_mask_noise_std,
                    lam_sparsity=cfg.learned_mask_lam_sparsity,
                    alpha_unimodal=cfg.learned_mask_alpha_unimodal,
                    hard=cfg.learned_mask_hard,
                    hard_thresh=cfg.learned_mask_hard_thresh,
                )
        finally:
            if was_training:
                model.train()
            _restore_model_params(model, req)

        g0 = masks["g0_uni"].detach()
        g1 = masks["g1_uni"].detach()

        x0_t = _apply_learned_destroy(x0, g0, noise_std=getattr(cfg, "learned_mask_noise_std", 1.0))
        x1_t = _apply_learned_destroy(x1, g1, noise_std=getattr(cfg, "learned_mask_noise_std", 1.0))

        with torch.no_grad():
            f_t0, _, _ = model.forward_logits(x0_t, x1)
            f_t1, _, _ = model.forward_logits(x0, x1_t)
            l_cf = bern_kl_to_uniform_from_logits(f_t0) + bern_kl_to_uniform_from_logits(f_t1)
            l_total = l_base + float(cfg.lambda_kl) * l_cf
            bs = y.size(0)
            n += bs

            subset_syn_flag = (b["mask0"]==MASK_SYN).any(dim=1)
            subset_syn_bs = subset_syn_flag.sum()
            n_syn += subset_syn_bs

            sum_l_f += float(lf.item()) * bs
            sum_l_u0 += float(lu0.item()) * bs
            sum_l_u1 += float(lu1.item()) * bs
            sum_l_cf += float(l_cf.item()) * bs
            sum_l_base += float(l_base.item()) * bs
            sum_l_total += float(l_total.item()) * bs

            sum_acc_f += _acc_from_logits(f, y) * bs
            sum_acc_f_syn += _acc_from_logits(f[subset_syn_flag], y[subset_syn_flag]) * subset_syn_bs if subset_syn_bs>0 else 0
            sum_acc_u0 += _acc_from_logits(u0, y) * bs
            sum_acc_u1 += _acc_from_logits(u1, y) * bs

    return {
        "loss_total":  sum_l_total / max(1, n),
        "loss_base":   sum_l_base / max(1, n),
        "loss_fusion": sum_l_f / max(1, n),
        "loss_uni0":   sum_l_u0 / max(1, n),
        "loss_uni1":   sum_l_u1 / max(1, n),
        "loss_cf":     sum_l_cf / max(1, n),
        "acc_fusion":  sum_acc_f / max(1, n),
        "acc_fusion_syn":  sum_acc_f_syn / max(1, n_syn),
        "acc_uni0":    sum_acc_u0 / max(1, n),
        "acc_uni1":    sum_acc_u1 / max(1, n),
    }
def train_synib_learned( cfg, train_loader: DataLoader, device: str, val_loader: Optional[DataLoader] = None,) -> Tuple["FusionModel", Dict[str, Any]]:
    """
    SynIB variant where the counterfactual destruction mask is learned on-the-fly
    (mask-only optimization; model is not retrained inside mask learning).

    Notes:
      - Counterfactual term uses learned feature-level gates (not oracle block ids)
      - Shortcut invariance term can still use oracle block ids if desired
    """
    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay_synib)

    # printing controls
    print_every = int(getattr(cfg, "print_every", 1))
    print_masks = bool(getattr(cfg, "print_mask_stats", True))

    history = {
        "train": [],
        "val": [],
        "best_epoch": None,
        "best_val_fusion_acc": 0.0,
        "best_val_fusion_loss": float("inf"),
    }
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()

        n = 0
        sum_l_base = 0.0
        sum_l_f = 0.0
        sum_l_u0 = 0.0
        sum_l_u1 = 0.0
        sum_l_cf = 0.0
        sum_lf_uni = 0.0
        sum_l_total = 0.0

        sum_acc_f = 0.0
        sum_acc_syn_f = 0.0
        sum_acc_u0 = 0.0
        sum_acc_u1 = 0.0

        # mask diagnostics (averaged over batches)
        sum_g0 = 0.0
        sum_g1 = 0.0
        nb = 0
        n_syn = 0

        sum_iou_syn0 = 0.0
        sum_iou_syn1 = 0.0
        sum_iou_nonsyn0 = 0.0
        sum_iou_nonsyn1 = 0.0
        sum_iou_uni0 = 0.0
        sum_iou_uni1 = 0.0
        sum_iou_nonuni0 = 0.0
        sum_iou_nonuni1 = 0.0
        nb_iou = 0


        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y = b["y"].to(device)
            m0 = b["mask0"].to(device)
            m1 = b["mask1"].to(device)

            # ---- clean forward ----
            f, u0, u1 = model.forward_logits(x0, x1)
            lf = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            l_base = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            # ---- learn destroy mask (model frozen during mask optimization) ----
            req = _freeze_model_params(model)
            was_training = model.training
            model.eval()

            try:
                with torch.enable_grad():  # robust even if caller wraps in no_grad
                    masks = get_learned_destroy_unimask(
                        model,
                        x0.detach(),
                        x1.detach(),
                        method=cfg.learned_mask_method,
                        device=device,
                        steps=cfg.learned_mask_steps,
                        lr=cfg.learned_mask_lr,
                        tau=cfg.learned_mask_tau,
                        noise_std=cfg.learned_mask_noise_std,
                        lam_sparsity=cfg.learned_mask_lam_sparsity,
                        alpha_unimodal=cfg.learned_mask_alpha_unimodal,
                        hard=cfg.learned_mask_hard,
                        hard_thresh=cfg.learned_mask_hard_thresh,
                    )
            finally:
                if was_training:
                    model.train()
                _restore_model_params(model, req)


            g0 = masks["g0"].detach()
            g1 = masks["g1"].detach()
            g0_uni = masks["g0_uni"].detach()
            g1_uni = masks["g1_uni"].detach()

            gt0 = _gt_synergy_from_blockmask(m0)
            gt1 = _gt_synergy_from_blockmask(m1)
            gt0_uni = _gt_synergy_from_blockmask(m0, [MASK_RED, MASK_UNIQUE])
            gt1_uni = _gt_synergy_from_blockmask(m1, [MASK_RED, MASK_UNIQUE])

            thr = float(getattr(cfg, "learned_mask_hard_thresh", 0.5))
            pred0 = (g0 >= thr)
            pred1 = (g1 >= thr)
            gt0_nonsyn = ~gt0
            gt1_nonsyn = ~gt1
            iou_syn0 = _iou_binary(pred0, gt0)
            iou_syn1 = _iou_binary(pred1, gt1)
            iou_nonsyn0 = _iou_binary(pred0, gt0_nonsyn)
            iou_nonsyn1 = _iou_binary(pred1, gt1_nonsyn)

            thr = float(getattr(cfg, "learned_mask_hard_thresh", 0.5))
            pred0 = (g0_uni >= thr)
            pred1 = (g1_uni >= thr)
            gt0_nonuni = ~gt0_uni
            gt1_nonuni = ~gt1_uni
            iou_uni0 = _iou_binary(pred0, gt0)
            iou_uni1 = _iou_binary(pred1, gt1)
            iou_nonuni0 = _iou_binary(pred0, gt0_nonuni)
            iou_nonuni1 = _iou_binary(pred1, gt1_nonuni)

            sum_iou_syn0 += iou_syn0
            sum_iou_syn1 += iou_syn1
            sum_iou_nonsyn0 += iou_nonsyn0
            sum_iou_nonsyn1 += iou_nonsyn1

            sum_iou_uni0 += iou_uni0
            sum_iou_uni1 += iou_uni1
            sum_iou_nonuni0 += iou_nonuni0
            sum_iou_nonuni1 += iou_nonuni1
            nb_iou += 1

            if print_masks:
                sum_g0 += float(g0.mean().item())
                sum_g1 += float(g1.mean().item())
                nb += 1

            # ---- counterfactual: destroy learned mask => fusion uncertainty ----
            x0_t = _apply_learned_destroy(x0, ~g0_uni.bool(), noise_std=getattr(cfg, "learned_mask_noise_std", 1.0))
            x1_t = _apply_learned_destroy(x1, ~g0_uni.bool(), noise_std=getattr(cfg, "learned_mask_noise_std", 1.0))
            # x0_t_uni = _apply_learned_destroy(x0, g0_uni, noise_std=getattr(cfg, "learned_mask_noise_std", 1.0))
            # x1_t_uni = _apply_learned_destroy(x1, g1_uni, noise_std=getattr(cfg, "learned_mask_noise_std", 1.0))

            # f_t0_uni = model.forward_secondfusion(x0_t_uni, x1_t_uni)
            # lf_uni = F.binary_cross_entropy_with_logits(f_t0_uni, y)


            f_t0, _, _ = model.forward_logits(x0_t, x1)
            f_t1, _, _ = model.forward_logits(x0, x1_t)
            l_cf = bern_kl_to_uniform_from_logits(f_t0) + bern_kl_to_uniform_from_logits(f_t1)
            l_total = l_base + float(cfg.lambda_kl) * l_cf

            opt.zero_grad()
            l_total.backward()
            opt.step()

            bs = y.size(0)
            n += bs


            sum_l_base += float(l_base.item()) * bs
            sum_l_f += float(lf.item()) * bs
            sum_l_u0 += float(lu0.item()) * bs
            sum_l_u1 += float(lu1.item()) * bs
            sum_l_cf += float(l_cf.item()) * bs
            # sum_lf_uni += float(lf_uni.item()) * bs
            sum_l_total += float(l_total.item()) * bs

            subset_syn_flag = (b["mask0"]==MASK_SYN).any(dim=1)
            subset_syn_bs = subset_syn_flag.sum()
            n_syn += subset_syn_bs

            sum_acc_f += _acc_from_logits(f, y) * bs
            sum_acc_syn_f += _acc_from_logits(f[subset_syn_flag], y[subset_syn_flag]) * subset_syn_bs if subset_syn_bs>0 else 0
            sum_acc_u0 += _acc_from_logits(u0, y) * bs
            sum_acc_u1 += _acc_from_logits(u1, y) * bs

        train_metrics = {
            "epoch": epoch,
            "loss_total": sum_l_total / max(1, n),
            "loss_base": sum_l_base / max(1, n),
            "loss_fusion": sum_l_f / max(1, n),
            "loss_uni0": sum_l_u0 / max(1, n),
            "loss_uni1": sum_l_u1 / max(1, n),
            "loss_cf": sum_l_cf / max(1, n),
            "loss_fusion_uni": sum_lf_uni / max(1, n),
            "acc_fusion": sum_acc_f / max(1, n),
            "acc_fusion_syn": sum_acc_syn_f / max(1, n_syn),
            "acc_uni0": sum_acc_u0 / max(1, n),
            "acc_uni1": sum_acc_u1 / max(1, n),
        }
        if print_masks and nb > 0:
            train_metrics["mask_g0_mean"] = sum_g0 / nb
            train_metrics["mask_g1_mean"] = sum_g1 / nb

        if nb_iou > 0:
            train_metrics["iou_syn0"] = sum_iou_syn0 / nb_iou
            train_metrics["iou_syn1"] = sum_iou_syn1 / nb_iou
            train_metrics["iou_syn_avg"] = 0.5 * (train_metrics["iou_syn0"] + train_metrics["iou_syn1"])
            train_metrics["iou_nonsyn0"] = sum_iou_nonsyn0 / nb_iou
            train_metrics["iou_nonsyn1"] = sum_iou_nonsyn1 / nb_iou
            train_metrics["iou_nonsyn_avg"] = 0.5 * (train_metrics["iou_nonsyn0"] + train_metrics["iou_nonsyn1"])

            train_metrics["iou_uni0"] = sum_iou_uni0 / nb_iou
            train_metrics["iou_uni1"] = sum_iou_uni1 / nb_iou
            train_metrics["iou_uni_avg"] = 0.5 * (train_metrics["iou_uni0"] + train_metrics["iou_uni1"])
            train_metrics["iou_nonuni0"] = sum_iou_nonuni0 / nb_iou
            train_metrics["iou_nonuni1"] = sum_iou_nonuni1 / nb_iou
            train_metrics["iou_nonuni_avg"] = 0.5 * (train_metrics["iou_nonuni0"] + train_metrics["iou_nonuni1"])

        history["train"].append(train_metrics)

        # -------------------- val epoch + checkpoint --------------------
        val_metrics = None
        if val_loader is not None:
            val_metrics = _eval_epoch_synib_learned(model, val_loader, device, cfg)
            val_metrics["epoch"] = epoch
            history["val"].append(val_metrics)

            cur = float(val_metrics["acc_fusion"])
            if cur > history["best_val_fusion_acc"]:
                history["best_val_fusion_acc"] = cur
                history["best_val_fusion_loss"] = val_metrics["loss_fusion"]
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(model.state_dict())

        # -------------------- prints --------------------
        if print_every > 0 and (epoch % print_every == 0 or epoch == cfg.epochs - 1):
            best_ep = history["best_epoch"]
            mark = "*" if (best_ep is not None and best_ep == epoch) else ""

            tr = train_metrics
            tr_str = (f"tr: l={tr['loss_fusion']:.3f} ib={tr['loss_cf']:.3f} "
                      f"l_tot={tr['loss_total']:.3f} acc={tr['acc_fusion']:.3f} acc_syn={tr['acc_fusion_syn']:.3f}")
            if print_masks and nb > 0:
                tr_str += f" g0={tr['mask_g0_mean']:.3f} g1={tr['mask_g1_mean']:.3f}"

            if "iou_syn_avg" in tr:
                tr_str += (f" iou_syn={tr['iou_syn_avg']:.3f}"
                           f" (m0={tr['iou_syn0']:.3f}, m1={tr['iou_syn1']:.3f})")
                tr_str += f" iou_nonsyn={tr['iou_nonsyn_avg']:.3f}"
            if "iou_uni_avg" in tr:
                tr_str += (f" iou_uni={tr['iou_uni_avg']:.3f}"
                           f" (m0={tr['iou_uni0']:.3f}, m1={tr['iou_uni1']:.3f})")
                tr_str += f" iou_nonuni={tr['iou_nonuni_avg']:.3f}"

            if val_metrics is not None:
                va = val_metrics
                va_str = (f"va: f={va['loss_fusion']:.3f} cf={va['loss_cf']:.3f} "
                          f"tot={va['loss_total']:.3f} acc={va['acc_fusion']:.3f} acc_syn={va['acc_fusion_syn']:.3f}")
                best_str = f"best_va_f-acc={history['best_val_fusion_loss']:.3f}-{history['best_val_fusion_acc']:.3f}@{history['best_epoch']}"
                print(f"[E{epoch:03d}]{mark} {tr_str} | {va_str} | {best_str}")
            else:
                print(f"[E{epoch:03d}] {tr_str}")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history
def run_once_synib_learned( cfg: Config, *, seed: int, verbose: bool = True,) -> Dict[str, Any]:
    """
    Same scaffold as your run_once, but ONLY trains/evaluates SynIB-Learned.
    - train/val split from TRAIN dataset
    - val_loader passed into train_synib_learned
    - returns history for synib_learned
    - best checkpoint selection is done inside train_synib_learned (based on VAL fusion loss)
    """
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)

    # -------------------- build datasets --------------------
    full_train_ds = PID4BlockDataset(cfg, cfg.n_train, seed=seed, split="train", verbose=verbose)
    test_ds = PID4BlockDataset(cfg, cfg.n_test, seed=seed + 1, split="test", train_stats=full_train_ds.stats, verbose=verbose)

    # -------------------- split train into train/val --------------------
    val_frac = getattr(cfg, "val_frac", 0.10)
    n = len(full_train_ds)
    n_val = max(1, int(round(val_frac * n)))
    n_train = n - n_val

    g = torch.Generator()
    g.manual_seed(seed + 12345)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_ds = Subset(full_train_ds, train_idx)
    val_ds = Subset(full_train_ds, val_idx)

    # -------------------- loaders --------------------
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    # -------------------- detectability sanity check --------------------
    det = run_detectability_probe(
        train_loader,
        device,
        dim0=cfg.dim0,
        dim1=cfg.dim1,
        destroy_block_list=DESTROY_MASK,  # if your probe still uses block-id destruction
        noise_std=1.0,
    )

    # -------------------- train SynIB-Learned --------------------
    synibL_model, synibL_hist = train_synib_learned(cfg, train_loader, device, val_loader=val_loader)

    # -------------------- final eval on test --------------------
    synibL_stats = eval_clean(synibL_model, test_loader, device)
    synibL_ab    = eval_block_ablations(synibL_model, test_loader, device, noise_std=1.0)

    synibL_by = eval_by_source(synibL_model, test_loader, device)
    if verbose:
        print_by_source("SynIB-Learned", synibL_by, min_n=10)

    return {
        "cfg": asdict(cfg),
        "seed": seed,
        "split": {"n_train": n_train, "n_val": n_val, "n_test": len(test_ds)},
        "detectability_acc": det,
        "synib_learned": synibL_stats,
        "ablations": synibL_ab,
        "history": {"synib_learned": synibL_hist},
        "by_source": {"synib_learned": synibL_by},
    }


@torch.no_grad()
def eval_by_source(model, loader, device: str) -> Dict[str, Any]:
    """
    Computes fusion accuracy grouped by the per-sample `source` indicator.
    Expects batches with:
      - b["x0"], b["x1"], b["y"]
      - b["source"] as Tensor [B,4] (multi-hot)
    Returns:
      {
        "overall_acc": float,
        "by_source": {source_key: {"acc": float, "n": int}}
      }
    """
    model.eval()
    correct_total, n_total = 0, 0
    correct = defaultdict(int)
    count = defaultdict(int)

    for b in loader:
        x0 = b["x0"].to(device)
        x1 = b["x1"].to(device)
        y  = b["y"].to(device).view(-1)  # [B]
        src = b["source"]                # [B,4] on CPU by default

        f, _, _ = model.forward_logits(x0, x1)
        pred = (f.view(-1) > 0).float()

        # overall
        correct_total += int((pred == y).sum().item())
        n_total += int(y.numel())

        # by-source
        B = y.numel()
        for i in range(B):
            k = _decode_source_key(src[i])
            count[k] += 1
            correct[k] += int((pred[i].item() == y[i].item()))

    by_source = {
        k: {"acc": (correct[k] / count[k]) if count[k] > 0 else 0.0, "n": int(count[k])}
        for k in sorted(count.keys())
    }
    return {
        "overall_acc": (correct_total / n_total) if n_total > 0 else 0.0,
        "by_source": by_source
    }
def _decode_source_key(source_tensor_1d: torch.Tensor) -> str:
    """
    source_tensor_1d: shape [4] multi-hot float/bool in order [u1,u2,red,syn]
    returns e.g. "u1", "syn+red", or "none"
    """
    active = [name for j, name in enumerate(_SOURCES) if float(source_tensor_1d[j].item()) > 0.5]
    return "none" if len(active) == 0 else "+".join(active)
def print_by_source(tag: str, stats: Dict[str, Any], min_n: int = 1) -> None:
    """
    Pretty prints eval_by_source() output in one compact block.
    """
    items = []
    for k, v in stats["by_source"].items():
        if v["n"] >= min_n:
            items.append(f"{k}:{v['acc']:.3f}(n={v['n']})")
    s = " | ".join(items)
    print(f"[BY-SOURCE:{tag}] overall={stats['overall_acc']:.3f} | {s}")

def print_config(cfg: Config, title: str = "CONFIG") -> None:
    u0, r0, s0, n0 = _block_sizes(cfg.dim0, cfg.frac_unique, cfg.frac_red, cfg.frac_syn)
    u1, r1, s1, n1 = _block_sizes(cfg.dim1, cfg.frac_unique, cfg.frac_red, cfg.frac_syn)
    print("" + "=" * 110)
    print(f"[{title}]")
    print(f"  dims: dim0={cfg.dim0} dim1={cfg.dim1}")
    print(f"  blocks0: unique={u0} red={r0} syn={s0} noise={n0} | random_pos={cfg.random_block_positions}")
    print(f"  blocks1: unique={u1} red={r1} syn={s1} noise={n1} | random_pos={cfg.random_block_positions}")
    print(f"  SNR: unique_strength={pretty_float(cfg.unique_strength)} red_strength={pretty_float(cfg.red_strength)} syn_strength={pretty_float(cfg.syn_strength)} noise_std={pretty_float(cfg.noise_std)}")
    print(f"  train: n_train={cfg.n_train} n_test={cfg.n_test} batch={cfg.batch_size} epochs={cfg.epochs} lr={pretty_float(cfg.lr)}")
    print(f"  loss: lambda_uni={pretty_float(cfg.lambda_uni)} lambda_kl={pretty_float(cfg.lambda_kl)} lambda_shortcut_inv={pretty_float(cfg.lambda_shortcut_inv)}")
    print(f"  probs: u1={cfg.signal_probs['u1']}, u2={cfg.signal_probs['u2']}, red={cfg.signal_probs['red']}, syn={cfg.signal_probs['syn']}")
    print("=" * 110)
def print_run_header() -> None:
    header = "Seed | Meth   | AccF  | AccU0 | AccU1"
    print(header); print("-" * len(header))
def print_row(seed: int, meth: str, stats: Dict[str, float]) -> None:
    print(f"{seed:<4d} | {meth:<6} | {stats['acc_fusion']:<5.3f} | {stats['acc_uni0']:<5.3f} | {stats['acc_uni1']:<5.3f}")
def _series(hist: Dict[str, Any], split: str, key: str) -> Tuple[List[int], List[float]]:
    """
    hist: history dict returned by train_main/train_synib
      hist["train"] is a list of dicts with {"epoch": int, key: float, ...}
      hist["val"]   is a list of dicts with {"epoch": int, key: float, ...}
    """
    rows = hist.get(split, [])
    xs = [int(r["epoch"]) for r in rows]
    ys = [float(r[key]) for r in rows]
    return xs, ys
def _plot_loss_curves(rep: Dict[str, Any], out_dir: str) -> str:
    main_hist = rep["history"]["main"]
    syn_hist  = rep["history"]["synib"]

    # We plot fusion loss (clean BCE on fusion logits)
    x_mt, y_mt = _series(main_hist, "train", "loss_fusion")
    x_mv, y_mv = _series(main_hist, "val",   "loss_fusion")
    x_st, y_st = _series(syn_hist,  "train", "loss_fusion")
    x_sv, y_sv = _series(syn_hist,  "val",   "loss_fusion")

    plt.figure()
    if x_mt: plt.plot(x_mt, y_mt, label="Main train loss_fusion")
    if x_mv: plt.plot(x_mv, y_mv, label="Main val loss_fusion")
    if x_st: plt.plot(x_st, y_st, label="SynIB train loss_fusion")
    if x_sv: plt.plot(x_sv, y_sv, label="SynIB val loss_fusion")

    # mark best epochs (based on val fusion loss)
    if main_hist.get("best_epoch") is not None and x_mv:
        be = int(main_hist["best_epoch"])
        bv = float(main_hist["best_val_fusion_loss"])
        plt.scatter([be], [bv], marker="o", label="Main best val (fusion loss)")
    if syn_hist.get("best_epoch") is not None and x_sv:
        be = int(syn_hist["best_epoch"])
        bv = float(syn_hist["best_val_fusion_loss"])
        plt.scatter([be], [bv], marker="o", label="SynIB best val (fusion loss)")

    plt.xlabel("epoch")
    plt.ylabel("fusion loss (BCE)")
    plt.title("Fusion loss curves (train/val)")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "curves_loss_fusion.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path
def _plot_acc_curves(rep: Dict[str, Any], out_dir: str) -> str:
    main_hist = rep["history"]["main"]
    syn_hist  = rep["history"]["synib"]

    # Fusion accuracy during training/validation
    x_mt, y_mt = _series(main_hist, "train", "acc_fusion")
    x_mv, y_mv = _series(main_hist, "val",   "acc_fusion")
    x_st, y_st = _series(syn_hist,  "train", "acc_fusion")
    x_sv, y_sv = _series(syn_hist,  "val",   "acc_fusion")

    plt.figure()
    if x_mt: plt.plot(x_mt, y_mt, label="Main train acc_fusion")
    if x_mv: plt.plot(x_mv, y_mv, label="Main val acc_fusion")
    if x_st: plt.plot(x_st, y_st, label="SynIB train acc_fusion")
    if x_sv: plt.plot(x_sv, y_sv, label="SynIB val acc_fusion")

    # Add TEST fusion accuracy as horizontal reference lines
    main_test = float(rep["main"]["acc_fusion"])
    syn_test  = float(rep["synib"]["acc_fusion"])
    plt.axhline(main_test, linestyle="--", label=f"Main test acc_fusion = {main_test:.3f}")
    plt.axhline(syn_test,  linestyle="--", label=f"SynIB test acc_fusion = {syn_test:.3f}")

    plt.xlabel("epoch")
    plt.ylabel("fusion accuracy")
    plt.title("Fusion accuracy curves (train/val) + test reference")
    plt.ylim(0.0, 1.1)
    plt.legend()
    plt.tight_layout()

    path = os.path.join(out_dir, "curves_acc_fusion.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path
def print_sanity(rep: Dict[str, Any]) -> None:
    print("[SANITY]")
    print(f"  detectability probe (destroy SYNERGY) acc = {rep['detectability_acc']:.3f}  (~0.5 ideal)")
    for k in ["clean", "destroy_unique", "destroy_red", "destroy_syn", "destroy_unique+red"]:
        m = rep["ablations"][k]
        print(f"  {k:<18} acc={m['acc']:.3f}  entropy={m['entropy']:.3f}")


def _syn_acc_from_by_source(by: dict) -> float:
    """Weighted avg acc over all groups that include 'syn' in the key."""
    num, den = 0.0, 0
    for k, v in by["by_source"].items():
        parts = k.split("+")
        if "syn" in parts:
            n = int(v["n"])
            num += float(v["acc"]) * n
            den += n
    return (num / den) if den > 0 else float("nan")
def _plot_acc_vs_kl(out_dir: str, kl_vals, main_total, main_syn, synib_total, synib_syn) -> str:
    plt.figure()
    plt.plot(kl_vals, main_total, marker="o", label="Main total acc")
    plt.plot(kl_vals, synib_total, marker="o", label="SynIB total acc")
    plt.plot(kl_vals, main_syn, marker="o", label="Main syn-slice acc")
    plt.plot(kl_vals, synib_syn, marker="o", label="SynIB syn-slice acc")
    plt.xscale("log")
    plt.ylim(0.0, 1.0)
    plt.xlabel("lambda_kl (log scale)")
    plt.ylabel("accuracy")
    plt.title("Accuracy vs lambda_kl (total + syn-slice)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "sweep_lambda_kl_acc.png")
    plt.savefig(path, dpi=160)
    plt.close()
    return path
def main_sweep_lambda_kl():
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    # log-scale from 1e-3 to 1e+3
    kl_vals = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]

    main_total, synib_total = [], []
    main_syn, synib_syn = [], []

    print_config(cfg0, "DEFAULT CONFIG")
    print("[RUN] sweep lambda_kl")
    print("lambda_kl | MainTot | MainSyn | SynIBTot | SynIBSyn | det")
    print("----------------------------------------------------------")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg0.out_dir, f"sweep_lambda_kl_{ts}")
    mkdirp(out_dir)
    cfg0.out_dir = out_dir

    first_time = True
    for lam in kl_vals:
        cfg = deepcopy(cfg0)
        cfg.lambda_kl = float(lam)
        this_out_dir = os.path.join(out_dir, f"lambda_kl_{lam:g}")
        mkdirp(this_out_dir)
        cfg.out_dir = this_out_dir


        dont_compute_main = False if first_time else True

        rep = run_once(cfg, seed=0, verbose=False, dont_compute_main=dont_compute_main)
        if not dont_compute_main:
            first_time=False
            m_history = rep["history"]["main"]
            m_main = rep["main"]
            mt = float(rep["main"]["acc_fusion"])
            ms = _syn_acc_from_by_source(rep["by_source"]["main"])
        else:
            rep["history"]["main"] = m_history
            rep["main"] = m_main


        st = float(rep["synib"]["acc_fusion"])
        ss = _syn_acc_from_by_source(rep["by_source"]["synib"])
        det = float(rep["detectability_acc"])

        main_total.append(mt); synib_total.append(st)
        main_syn.append(ms); synib_syn.append(ss)

        print(f"{lam:8.1e} | {mt:7.3f} | {ms:7.3f} | {st:8.3f} | {ss:8.3f} | {det:5.3f}")

        # optional: per-run plots you already had
        _plot_loss_curves(rep, cfg.out_dir)
        _plot_acc_curves(rep, cfg.out_dir)

    # one aggregate plot for the sweep
    sweep_plot = _plot_acc_vs_kl(cfg0.out_dir, kl_vals, main_total, main_syn, synib_total, synib_syn)
    print(f"[SAVED] {sweep_plot}")


def main_sweep_lambda_kl_learned():
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    # log-scale from 1e-3 to 1e+3
    # kl_vals = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    kl_vals = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    kl_vals = [1e1]

    main_total, learned_total = [], []
    main_syn, learned_syn = [], []

    print_config(cfg0, "DEFAULT CONFIG (LEARNED)")
    print("[RUN] sweep lambda_kl (SynIB-Learned)")
    print("lambda_kl | MainTot | MainSyn | LearnedTot | LearnedSyn | det")
    print("--------------------------------------------------------------")

    # --- build loaders + probe once ---
    device, split, train_loader, val_loader, test_loader = build_loaders(cfg0, seed=0, verbose=False)
    # det = run_detectability_probe(
    #     train_loader, device,
    #     dim0=cfg0.dim0, dim1=cfg0.dim1,
    #     destroy_block_list=DESTROY_MASK,
    #     noise_std=1.0,
    # )
    main_out = run_main(cfg0, train_loader, val_loader, test_loader, device, verbose=False)

    for lam in kl_vals:
        cfg = deepcopy(cfg0)
        cfg.lambda_kl = float(lam)
        # (optional) speed knobs for learned mask during sweep
        # cfg.learned_mask_steps = getattr(cfg, "learned_mask_steps", 5)  # reduce if needed
        # cfg.learned_mask_method = getattr(cfg, "learned_mask_method", "fusion_more_than_unimodal")
        out_dir = os.path.join(cfg0.out_dir, f"sweep_lambda_kl_learned_{lam:g}")
        mkdirp(out_dir)
        cfg.out_dir = out_dir

        # --- SynIB-Learned ---
        learned_model, learned_hist = train_synib_learned(cfg, train_loader, device, val_loader=val_loader)
        learned_stats = eval_clean(learned_model, test_loader, device)
        learned_by = eval_by_source(learned_model, test_loader, device)

        # package rep for your plotting helpers (same keys as before)
        rep = {
            "cfg": asdict(cfg),
            "seed": 0,
            "split": split,
            "main": main_out["stats"],
            "synib": learned_stats,  # keep name "synib" so _plot_* works
            "history": {"main": main_out["history"], "synib": learned_hist},
            "by_source": {"main": main_out["by_source"], "synib": learned_by},
        }

        mt = float(rep["main"]["acc_fusion"])
        lt = float(rep["synib"]["acc_fusion"])  # learned total

        ms = _syn_acc_from_by_source(rep["by_source"]["main"])
        ls = _syn_acc_from_by_source(rep["by_source"]["synib"])

        main_total.append(mt); learned_total.append(lt)
        main_syn.append(ms); learned_syn.append(ls)

        print(f"{lam:8.1e} | {mt:7.3f} | {ms:7.3f} | {lt:9.3f} | {ls:9.3f}")

        _plot_loss_curves(rep, cfg.out_dir)
        _plot_acc_curves(rep, cfg.out_dir)

    sweep_plot = _plot_acc_vs_kl(cfg0.out_dir, kl_vals, main_total, main_syn, learned_total, learned_syn)
    print(f"[SAVED] {sweep_plot}")
def main_sweep_lambda_kl_both():
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg0.out_dir, f"sweep_lambda_kl_both_{ts}")
    mkdirp(out_dir)
    cfg0.out_dir = out_dir
    mkdirp(cfg0.out_dir)


    # log-scale from 1e-3 to 1e+3
    # kl_vals = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    kl_vals = [1e0, 1e1]

    main_total = []
    main_syn   = []

    synib_total = []
    synib_syn   = []

    learned_total = []
    learned_syn   = []

    print_config(cfg0, "DEFAULT CONFIG (BOTH)")
    print("[RUN] sweep lambda_kl (SynIB oracle + SynIB-Learned)")
    print("lambda_kl | MainTot MainSyn | SynIBTot SynIBSyn | LearnTot LearnSyn | det")
    print("-------------------------------------------------------------------------")

    # Build loaders ONCE (same split across the sweep)
    device, split, train_loader, val_loader, test_loader = build_loaders(cfg0, seed=0, verbose=False)

    # Detectability probe ONCE (same dataset; purely a dataset sanity check)
    det = run_detectability_probe(
        train_loader, device,
        dim0=cfg0.dim0, dim1=cfg0.dim1,
        destroy_block_list=DESTROY_MASK,
        noise_std=1.0,
    )

    # Main baseline ONCE (independent of lambda_kl)
    main_out = run_main(cfg0, train_loader, val_loader, test_loader, device, verbose=False)
    mt0 = float(main_out["stats"]["acc_fusion"])
    ms0 = _syn_acc_from_by_source(main_out["by_source"])
    # store once and repeat in arrays for plotting convenience
    for _ in kl_vals:
        main_total.append(mt0)
        main_syn.append(ms0)

    for lam in kl_vals:
        cfg = deepcopy(cfg0)
        cfg.lambda_kl = float(lam)

        # (optional) speed knobs for learned mask during sweep
        # cfg.learned_mask_steps  = getattr(cfg, "learned_mask_steps", 5)
        # cfg.learned_mask_method = getattr(cfg, "learned_mask_method", "fusion_more_than_unimodal")

        this_out_dir = os.path.join(out_dir, f"lambda_kl_{lam:g}")
        mkdirp(this_out_dir)
        cfg.out_dir = this_out_dir

        # -------------------- SynIB (oracle mask) --------------------
        synib_out = run_synib(cfg, train_loader, val_loader, test_loader, device, verbose=False)
        st = float(synib_out["stats"]["acc_fusion"])
        ss = _syn_acc_from_by_source(synib_out["by_source"])
        synib_total.append(st)
        synib_syn.append(ss)

        # -------------------- SynIB-Learned --------------------
        learned_model, learned_hist = train_synib_learned(cfg, train_loader, device, val_loader=val_loader)
        learned_stats = eval_clean(learned_model, test_loader, device)
        learned_by = eval_by_source(learned_model, test_loader, device)

        lt = float(learned_stats["acc_fusion"])
        ls = _syn_acc_from_by_source(learned_by)
        learned_total.append(lt)
        learned_syn.append(ls)

        print(f"{lam:8.1e} | "
              f"{mt0:7.3f} {ms0:7.3f} | "
              f"{st:7.3f} {ss:7.3f} | "
              f"{lt:7.3f} {ls:7.3f} | "
              f"{float(det):5.3f}")

        # ---- optional per-lam plots (use learned rep, but include all three in by_source if you want) ----
        rep_learned = {
            "cfg": asdict(cfg),
            "seed": 0,
            "split": split,
            "detectability_acc": det,
            "main": main_out["stats"],
            "synib": learned_stats,  # keep for existing plot helpers
            "history": {"main": main_out["history"], "synib": learned_hist},
            "by_source": {"main": main_out["by_source"], "synib": learned_by},
        }
        _plot_loss_curves(rep_learned, cfg.out_dir)
        _plot_acc_curves(rep_learned, cfg.out_dir)

    plot_path = _plot_acc_vs_kl_three(
        cfg0.out_dir, kl_vals,
        main_total, main_syn,
        synib_total, synib_syn,
        learned_total, learned_syn,
    )
    print(f"[SAVED] {plot_path}")

    return {
        "kl_vals": kl_vals,
        "det": det,
        "main_total": main_total, "main_syn": main_syn,
        "synib_total": synib_total, "synib_syn": synib_syn,
        "learned_total": learned_total, "learned_syn": learned_syn,
    }
def main_sweep_lambda_kl_both_multiseed( seeds=(0, 1, 2, 3, 4), learned_mask_steps_default: int = 5,):
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    kl_vals = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    S = list(seeds)
    K = len(kl_vals)

    print_config(cfg0, "DEFAULT CONFIG (BOTH, multiseed)")
    print("[RUN] sweep lambda_kl (SynIB oracle + SynIB-Learned) -- multi-seed")
    print("seeds:", S)
    print("lambda_kl | MainTot±std MainSyn±std | SynIBTot±std SynIBSyn±std | LearnTot±std LearnSyn±std")
    print("------------------------------------------------------------------------------------------------")

    # store per-kl arrays of shape [K, n_seeds]
    main_tot_mat    = np.full((K, len(S)), np.nan, dtype=float)
    main_syn_mat    = np.full((K, len(S)), np.nan, dtype=float)
    synib_tot_mat   = np.full((K, len(S)), np.nan, dtype=float)
    synib_syn_mat   = np.full((K, len(S)), np.nan, dtype=float)
    learn_tot_mat   = np.full((K, len(S)), np.nan, dtype=float)
    learn_syn_mat   = np.full((K, len(S)), np.nan, dtype=float)
    det_mat         = np.full((len(S),), np.nan, dtype=float)

    for si, seed in enumerate(S):
        # Build loaders per-seed (keeps fairness: each seed corresponds to a dataset draw + split)
        device, split, train_loader, val_loader, test_loader = build_loaders(cfg0, seed=seed, verbose=False)

        # probe per seed (dataset-dependent sanity)
        det = run_detectability_probe(
            train_loader, device,
            dim0=cfg0.dim0, dim1=cfg0.dim1,
            destroy_block_list=DESTROY_MASK,
            noise_std=1.0,
        )
        det_mat[si] = float(det)

        # Main baseline per seed (independent of lambda_kl but depends on data split)
        main_out = run_main(cfg0, train_loader, val_loader, test_loader, device, verbose=False)
        mt0 = float(main_out["stats"]["acc_fusion"])
        ms0 = float(_syn_acc_from_by_source(main_out["by_source"]))

        for ki, lam in enumerate(kl_vals):
            cfg = deepcopy(cfg0)
            cfg.lambda_kl = float(lam)

            # SynIB oracle (uses same train/val/test loaders)
            synib_out = run_synib(cfg, train_loader, val_loader, test_loader, device, verbose=False)
            st = float(synib_out["stats"]["acc_fusion"])
            ss = float(_syn_acc_from_by_source(synib_out["by_source"]))

            # SynIB-Learned
            learned_model, _learned_hist = train_synib_learned(cfg, train_loader, device, val_loader=val_loader)
            learned_stats = eval_clean(learned_model, test_loader, device)
            learned_by = eval_by_source(learned_model, test_loader, device)
            lt = float(learned_stats["acc_fusion"])
            ls = float(_syn_acc_from_by_source(learned_by))

            main_tot_mat[ki, si]  = mt0
            main_syn_mat[ki, si]  = ms0
            synib_tot_mat[ki, si] = st
            synib_syn_mat[ki, si] = ss
            learn_tot_mat[ki, si] = lt
            learn_syn_mat[ki, si] = ls

    # aggregate mean/std across seeds for each kl
    def _ms(mat):
        return np.nanmean(mat, axis=1).tolist(), np.nanstd(mat, axis=1).tolist()

    main_tot_mean, main_tot_std = _ms(main_tot_mat)
    main_syn_mean, main_syn_std = _ms(main_syn_mat)
    syn_tot_mean,  syn_tot_std  = _ms(synib_tot_mat)
    syn_syn_mean,  syn_syn_std  = _ms(synib_syn_mat)
    lea_tot_mean,  lea_tot_std  = _ms(learn_tot_mat)
    lea_syn_mean,  lea_syn_std  = _ms(learn_syn_mat)

    # print table (mean±std) per kl
    for ki, lam in enumerate(kl_vals):
        print(f"{lam:8.1e} | "
              f"{main_tot_mean[ki]:.3f}±{main_tot_std[ki]:.3f} {main_syn_mean[ki]:.3f}±{main_syn_std[ki]:.3f} | "
              f"{syn_tot_mean[ki]:.3f}±{syn_tot_std[ki]:.3f} {syn_syn_mean[ki]:.3f}±{syn_syn_std[ki]:.3f} | "
              f"{lea_tot_mean[ki]:.3f}±{lea_tot_std[ki]:.3f} {lea_syn_mean[ki]:.3f}±{lea_syn_std[ki]:.3f}")

    print(f"[SANITY] detectability acc over seeds: mean={float(np.nanmean(det_mat)):.3f} std={float(np.nanstd(det_mat)):.3f}")

    plot_path = _plot_acc_vs_kl_three_ms(
        cfg0.out_dir, kl_vals,
        (main_tot_mean, main_tot_std), (main_syn_mean, main_syn_std),
        (syn_tot_mean, syn_tot_std),   (syn_syn_mean, syn_syn_std),
        (lea_tot_mean, lea_tot_std),   (lea_syn_mean, lea_syn_std),
        fname="sweep_acc_vs_kl_three_meanstd.png",
    )
    print(f"[SAVED] {plot_path}")

    return {
        "kl_vals": kl_vals,
        "seeds": S,
        "det_per_seed": det_mat.tolist(),
        "main_total": {"mean": main_tot_mean, "std": main_tot_std},
        "main_syn":   {"mean": main_syn_mean, "std": main_syn_std},
        "synib_total": {"mean": syn_tot_mean, "std": syn_tot_std},
        "synib_syn":   {"mean": syn_syn_mean, "std": syn_syn_std},
        "learned_total": {"mean": lea_tot_mean, "std": lea_tot_std},
        "learned_syn":   {"mean": lea_syn_mean, "std": lea_syn_std},
        "plot": plot_path,
    }

def select_best_kl_on_val(cfg0: Config,    *, seed: int, kl_vals: list, method: str, verbose: bool = False,):
    """
    Returns:
      best_lam, val_table(list of dicts), and also the loaders/datasets so caller can retrain.
    """
    device, split, train_loader, val_loader, test_loader = build_loaders(cfg0, seed=seed, verbose=False)

    rows = []
    for lam in kl_vals:
        cfg = deepcopy(cfg0)
        cfg.lambda_kl = float(lam)

        if method == "synib":
            out = run_synib(cfg, train_loader, val_loader, test_loader, device, verbose=False)
            val_hist = out["history"]["val"]
            best_val_fusion_loss = float(np.min([v["loss_fusion"] for v in val_hist])) if len(val_hist) else float("inf")
            best_val_acc = float(np.max([v["acc_fusion"] for v in val_hist])) if len(val_hist) else float("-inf")

        elif method == "learned":
            model, hist = train_synib_learned(cfg, train_loader, device, val_loader=val_loader)
            # hist["best_val_fusion_loss"] already exists (from your checkpoint logic)
            best_val_fusion_loss = float(hist.get("best_val_fusion_loss", float("inf")))
            best_val_acc = float(np.max([v["acc_fusion"] for v in hist.get("val", [])])) if len(hist.get("val", [])) else float("-inf")
        else:
            raise ValueError(method)

        rows.append({
            "lambda_kl": float(lam),
            "val_best_loss_fusion": best_val_fusion_loss,
            "val_best_acc_fusion": best_val_acc,
        })

    # select by VAL fusion loss (recommended; matches your checkpoint criterion)
    best_row = min(rows, key=lambda r: r["val_best_loss_fusion"])
    best_lam = best_row["lambda_kl"]

    if verbose:
        print(f"[SEL:{method}] seed={seed} best lambda_kl={best_lam:g} "
              f"(val_fusion_loss={best_row['val_best_loss_fusion']:.4f}, val_acc={best_row['val_best_acc_fusion']:.3f})")

    return best_lam, rows, (device, split, train_loader, val_loader, test_loader)
def retrain_and_test_best_kl( cfg0: Config, *, seed: int, best_lam: float, method: str,  verbose: bool = False,):
    device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)

    # rebuild datasets the same way as build_loaders does
    full_train_ds = PID4BlockDataset(cfg0, cfg0.n_train, seed=seed, split="train", verbose=False)
    test_ds = PID4BlockDataset(cfg0, cfg0.n_test, seed=seed + 1, split="test", train_stats=full_train_ds.stats, verbose=False)

    train_loader_full = DataLoader(full_train_ds, batch_size=cfg0.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=cfg0.batch_size, shuffle=False)

    cfg = deepcopy(cfg0)
    cfg.lambda_kl = float(best_lam)

    if method == "synib":
        # train without val_loader now (or keep val_loader=None)
        syn_model, syn_hist = train_synib(cfg, train_loader_full, device, val_loader=None)
        test_stats = eval_clean(syn_model, test_loader, device)
        by = eval_by_source(syn_model, test_loader, device)

    elif method == "learned":
        syn_model, syn_hist = train_synib_learned(cfg, train_loader_full, device, val_loader=None)
        test_stats = eval_clean(syn_model, test_loader, device)
        by = eval_by_source(syn_model, test_loader, device)
    else:
        raise ValueError(method)

    if verbose:
        print(f"[TEST:{method}] seed={seed} lambda_kl={best_lam:g} "
              f"test_acc={float(test_stats['acc_fusion']):.3f} syn_acc={float(_syn_acc_from_by_source(by)):.3f}")

    return test_stats, by, syn_hist

def select_best_lambda_kl_on_val( cfg0: Config,  *, seed: int, kl_vals: list, method: str, verbose: bool = False ) -> dict:
    """
    Trains for each lambda_kl on the SAME train/val split (for this seed),
    selects best lambda_kl by VAL fusion loss, and returns:
      - best_lambda_kl
      - per-lambda val summary table
      - also returns the best-val model's TEST performance (no retrain).
        (Because the model already has a best-val checkpoint loaded.)

    Returns dict with keys:
      best_lambda_kl, table, best_test_stats, best_test_by_source, best_val_metric
    """
    device, split, train_loader, val_loader, test_loader = build_loaders(cfg0, seed=seed, verbose=False)

    table = []
    best = {
        "best_lambda_kl": None,
        "best_val_fusion_loss": float("inf"),
        "best_val_fusion_acc": float(0.0),
        "best_test_stats": None,
        "best_test_by_source": None,
    }

    for lam in kl_vals:
        cfg = deepcopy(cfg0)
        cfg.lambda_kl = float(lam)

        if method == "synib":
            # run_synib should train + keep best checkpoint based on VAL fusion loss
            syn_out = run_synib(cfg, train_loader, val_loader, test_loader, device, verbose=False)
            # best val metric is in history (preferred) if available
            val_hist = syn_out.get("history", {}).get("val", [])
            best_val_loss = float(min([v["loss_fusion"] for v in val_hist])) if len(val_hist) else float("inf")
            best_val_acc  = float(max([v["acc_fusion"] for v in val_hist])) if len(val_hist) else float("-inf")

            test_stats = syn_out["stats"]
            test_by    = syn_out["by_source"]

        elif method == "synib_random":
            # run_synib should train + keep best checkpoint based on VAL fusion loss
            syn_out = run_synib_random(cfg, train_loader, val_loader, test_loader, device, verbose=False)
            # best val metric is in history (preferred) if available
            val_hist = syn_out.get("history", {}).get("val", [])
            best_val_loss = float(min([v["loss_fusion"] for v in val_hist])) if len(val_hist) else float("inf")
            best_val_acc  = float(max([v["acc_fusion"] for v in val_hist])) if len(val_hist) else float("-inf")

            test_stats = syn_out["stats"]
            test_by    = syn_out["by_source"]
        elif method == "synib_randomdiff":
            # run_synib should train + keep best checkpoint based on VAL fusion loss
            syn_out = run_synib_randomdiff(cfg, train_loader, val_loader, test_loader, device, verbose=False)
            # best val metric is in history (preferred) if available
            val_hist = syn_out.get("history", {}).get("val", [])
            best_val_loss = float(min([v["loss_fusion"] for v in val_hist])) if len(val_hist) else float("inf")
            best_val_acc  = float(max([v["acc_fusion"] for v in val_hist])) if len(val_hist) else float("-inf")

            test_stats = syn_out["stats"]
            test_by    = syn_out["by_source"]

        elif method == "learned":
            model, hist = train_synib_learned(cfg, train_loader, device, val_loader=val_loader)
            best_val_loss = float(hist.get("best_val_fusion_loss", float("inf")))
            best_val_acc  = float(max([v["acc_fusion"] for v in hist.get("val", [])])) if len(hist.get("val", [])) else float("-inf")

            test_stats = eval_clean(model, test_loader, device)
            test_by    = eval_by_source(model, test_loader, device)

        else:
            raise ValueError(f"Unknown method={method}")

        row = {
            "lambda_kl": float(lam),
            "val_best_loss_fusion": best_val_loss,
            "val_best_acc_fusion": best_val_acc,
            "test_acc_fusion": float(test_stats["acc_fusion"]),
            "test_acc_syn": float(_syn_acc_from_by_source(test_by)),
        }
        table.append(row)

        if best_val_acc > best["best_val_fusion_acc"]:
            best["best_val_fusion_acc"] = best_val_acc
            best["best_val_fusion_loss"] = best_val_loss
            best["best_lambda_kl"] = float(lam)
            best["best_test_stats"] = test_stats
            best["best_test_by_source"] = test_by

        if verbose:
            print(f"[{method}] seed={seed} lam={lam:8.1e} "
                  f"val_f={best_val_loss:.4f} val_acc={best_val_acc:.3f} "
                  f"test_acc={row['test_acc_fusion']:.3f} test_syn={row['test_acc_syn']:.3f}")

    best["table"] = table
    return best
def main_tune_lambda_kl_no_retrain( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]):
    """
    For each seed:
      - sweeps lambda_kl
      - selects best lambda_kl by VAL fusion loss
      - reports TEST performance of the *selected run* (no retraining)

    Prints per-seed selection and summary mean±std across seeds.
    """
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    print_config(cfg0, "DEFAULT CONFIG (tune kl on val, no retrain)")
    print("seed | bestKL_synib | testTot_synib testSyn_synib | bestKL_learn | testTot_learn testSyn_learn")
    print("-----------------------------------------------------------------------------------------------")

    out = {"synib": [], "learned": []}

    for seed in seeds:
        syn = select_best_lambda_kl_on_val(cfg0, seed=seed, kl_vals=kl_vals, method="synib", verbose=False)
        lea = select_best_lambda_kl_on_val(cfg0, seed=seed, kl_vals=kl_vals, method="learned", verbose=False)

        syn_tot = float(syn["best_test_stats"]["acc_fusion"])
        syn_syn = float(_syn_acc_from_by_source(syn["best_test_by_source"]))
        lea_tot = float(lea["best_test_stats"]["acc_fusion"])
        lea_syn = float(_syn_acc_from_by_source(lea["best_test_by_source"]))

        out["synib"].append({"seed": seed, "best_kl": syn["best_lambda_kl"], "test_tot": syn_tot, "test_syn": syn_syn})
        out["learned"].append({"seed": seed, "best_kl": lea["best_lambda_kl"], "test_tot": lea_tot, "test_syn": lea_syn})

        print(f"{seed:4d} | {syn['best_lambda_kl']:12.1e} | {syn_tot:11.3f} {syn_syn:11.3f} | "
              f"{lea['best_lambda_kl']:11.1e} | {lea_tot:11.3f} {lea_syn:11.3f}")

    # summary
    syn_tot_arr = np.array([r["test_tot"] for r in out["synib"]], float)
    syn_syn_arr = np.array([r["test_syn"] for r in out["synib"]], float)
    lea_tot_arr = np.array([r["test_tot"] for r in out["learned"]], float)
    lea_syn_arr = np.array([r["test_syn"] for r in out["learned"]], float)

    print("-----------------------------------------------------------------------------------------------")
    print(f"[AVG] SynIB   testTot={syn_tot_arr.mean():.3f}±{syn_tot_arr.std():.3f}  testSyn={syn_syn_arr.mean():.3f}±{syn_syn_arr.std():.3f}")
    print(f"[AVG] Learned testTot={lea_tot_arr.mean():.3f}±{lea_tot_arr.std():.3f}  testSyn={lea_syn_arr.mean():.3f}±{lea_syn_arr.std():.3f}")

    return out
def main_sweep_nonoverlap_probs_tuned_kl( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-1, 1e0, 1e1, 1e2],):
    """
    For each non-overlap prob setting:
      - runs Main once per seed (val-checkpointed) and reports test
      - tunes lambda_kl for SynIB on val (no retrain), reports test of selected run
      - tunes lambda_kl for SynIB-Learned on val (no retrain), reports test of selected run
      - saves EVERYTHING to a JSON (append/dedupe, never overwrites previous entries)

    JSON entry contains:
      - probs
      - per-seed selected lambda_kl and test metrics (total + parts)
      - mean/std across seeds
      - (optional) the per-lambda validation tables for reproducibility
    """
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_tunedkl_v2.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG (tuned kl: synib + learned)")
    print("[RUN] sweep non-overlap source probs + tune lambda_kl (val) for SynIB + SynIB-Learned")
    print(f"[LOG] append/dedupe json: {results_path}")
    print(f"[SEEDS] {seeds}")
    print("pu1 pu2 pred psyn | MainTot | SynIB(tuned)Tot | Learn(tuned)Tot | MainSyn | SynIBSyn | LearnSyn | status")
    print("--------------------------------------------------------------------------------------------------------")

    prob_settings = [
    (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.1, 0.9), (0.0, 0.0, 0.2, 0.8), (0.0, 0.0, 0.3, 0.7),
    (0.0, 0.0, 0.4, 0.6), (0.0, 0.0, 0.5, 0.5), (0.0, 0.0, 0.6, 0.4), (0.0, 0.0, 0.7, 0.3),
    (0.0, 0.0, 0.8, 0.2), (0.0, 0.0, 0.9, 0.1), (0.0, 0.0, 1.0, 0.0), (0.1, 0.0, 0.0, 0.9),
    (0.1, 0.0, 0.1, 0.8), (0.1, 0.0, 0.2, 0.7), (0.1, 0.0, 0.3, 0.6), (0.1, 0.0, 0.4, 0.5),
    (0.1, 0.0, 0.5, 0.4), (0.1, 0.0, 0.6, 0.3), (0.1, 0.0, 0.7, 0.2), (0.1, 0.0, 0.8, 0.1),
    (0.1, 0.0, 0.9, 0.0), (0.2, 0.0, 0.0, 0.8), (0.2, 0.0, 0.1, 0.7), (0.2, 0.0, 0.2, 0.6),
    (0.2, 0.0, 0.3, 0.5), (0.2, 0.0, 0.4, 0.4), (0.2, 0.0, 0.5, 0.3), (0.2, 0.0, 0.6, 0.2),
    (0.2, 0.0, 0.7, 0.1), (0.2, 0.0, 0.8, 0.0), (0.3, 0.0, 0.0, 0.7), (0.3, 0.0, 0.1, 0.6),
    (0.3, 0.0, 0.2, 0.5), (0.3, 0.0, 0.3, 0.4), (0.3, 0.0, 0.4, 0.3), (0.3, 0.0, 0.5, 0.2),
    (0.3, 0.0, 0.6, 0.1), (0.3, 0.0, 0.7, 0.0), (0.4, 0.0, 0.0, 0.6), (0.4, 0.0, 0.1, 0.5),
    (0.4, 0.0, 0.2, 0.4), (0.4, 0.0, 0.3, 0.3), (0.4, 0.0, 0.4, 0.2), (0.4, 0.0, 0.5, 0.1),
    (0.4, 0.0, 0.6, 0.0), (0.5, 0.0, 0.0, 0.5), (0.5, 0.0, 0.1, 0.4), (0.5, 0.0, 0.2, 0.3),
    (0.5, 0.0, 0.3, 0.2), (0.5, 0.0, 0.4, 0.1), (0.5, 0.0, 0.5, 0.0), (0.6, 0.0, 0.0, 0.4),
    (0.6, 0.0, 0.1, 0.3), (0.6, 0.0, 0.2, 0.2), (0.6, 0.0, 0.3, 0.1), (0.6, 0.0, 0.4, 0.0),
    (0.7, 0.0, 0.0, 0.3), (0.7, 0.0, 0.1, 0.2), (0.7, 0.0, 0.2, 0.1), (0.7, 0.0, 0.3, 0.0),
    (0.8, 0.0, 0.0, 0.2), (0.8, 0.0, 0.1, 0.1), (0.8, 0.0, 0.2, 0.0), (0.9, 0.0, 0.0, 0.1),
    (0.9, 0.0, 0.1, 0.0), (1.0, 0.0, 0.0, 0.0)
]

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std())

    def _acc_parts_from_by(by_source_dict):
        # returns dict with u1,u2,red,syn (NaN if missing)
        out = {"u1": float("nan"), "u2": float("nan"), "red": float("nan"), "syn": float("nan")}
        for k in out.keys():
            if k in by_source_dict and by_source_dict[k].get("n", 0) > 0:
                out[k] = float(by_source_dict[k].get("acc", float("nan")))
        return out

    def _tune_one_method(cfg_base, *, seed, method):
        """
        method in {"synib","learned"}
        returns dict:
          best_kl, best_test_stats, best_test_by_source, table
        """
        return select_best_lambda_kl_on_val(
            cfg_base,
            seed=seed,
            kl_vals=kl_vals,
            method=method,
            verbose=False,
        )

    for (pu1, pu2, pred, psyn) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)  # reuse your stable key function

        if key in db["results"]:
            summ = db["results"][key]["summary_meanstd"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{summ['main']['test_tot_mean']:.3f} | "
                  f"{summ['synib_tuned']['test_tot_mean']:.3f} | "
                  f"{summ['learned_tuned']['test_tot_mean']:.3f} | "
                  f"{summ['main']['test_syn_mean']:.3f} | "
                  f"{summ['synib_tuned']['test_syn_mean']:.3f} | "
                  f"{summ['learned_tuned']['test_syn_mean']:.3f} | SKIP(existing)")
            continue

        cfg_base = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg_base, pu1, pu2, pred, psyn, pnone=0.0)

        # store per-seed results
        per_seed = {
            "main": [],
            "synib_tuned": [],
            "learned_tuned": [],
        }

        # run over seeds
        for seed in seeds:
            # --- build loaders once per seed for this prob setting ---
            device, split, train_loader, val_loader, test_loader = build_loaders(cfg_base, seed=seed, verbose=False)

            # --- main ---
            main_out = run_main(cfg_base, train_loader, val_loader, test_loader, device, verbose=False)
            main_stats = main_out["stats"]
            main_by = main_out["by_source"]

            main_rec = {
                "seed": int(seed),
                "test_acc_fusion": float(main_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(main_by)),
                "test_acc_parts": _acc_parts_from_by(main_by),
            }
            per_seed["main"].append(main_rec)

            # --- tune synib (oracle) ---
            syn_best = _tune_one_method(cfg_base, seed=seed, method="synib")
            syn_stats = syn_best["best_test_stats"]
            syn_by = syn_best["best_test_by_source"]
            per_seed["synib_tuned"].append({
                "seed": int(seed),
                "best_lambda_kl": float(syn_best["best_lambda_kl"]),
                "val_best_fusion_loss": float(syn_best["best_val_fusion_loss"]),
                "test_acc_fusion": float(syn_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(syn_by)),
                "test_acc_parts": _acc_parts_from_by(syn_by),
                "val_table": syn_best["table"],  # keep for reproducibility
            })

            # --- tune learned ---
            lea_best = _tune_one_method(cfg_base, seed=seed, method="learned")
            lea_stats = lea_best["best_test_stats"]
            lea_by = lea_best["best_test_by_source"]
            per_seed["learned_tuned"].append({
                "seed": int(seed),
                "best_lambda_kl": float(lea_best["best_lambda_kl"]),
                "val_best_fusion_loss": float(lea_best["best_val_fusion_loss"]),
                "test_acc_fusion": float(lea_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(lea_by)),
                "test_acc_parts": _acc_parts_from_by(lea_by),
                "val_table": lea_best["table"],
            })

        # ---- aggregate mean/std across seeds ----
        def _agg(block, field):
            vals = [r[field] for r in per_seed[block]]
            return _mean_std(vals)

        def _agg_part(block, part):
            vals = [r["test_acc_parts"][part] for r in per_seed[block]]
            return _mean_std(vals)

        summary = {"main": {}, "synib_tuned": {}, "learned_tuned": {}}
        for block in ["main", "synib_tuned", "learned_tuned"]:
            m_tot, s_tot = _agg(block, "test_acc_fusion")
            m_syn, s_syn = _agg(block, "test_acc_syn")
            summary[block]["test_tot_mean"] = m_tot
            summary[block]["test_tot_std"] = s_tot
            summary[block]["test_syn_mean"] = m_syn
            summary[block]["test_syn_std"] = s_syn
            summary[block]["parts_meanstd"] = {
                p: {"mean": _agg_part(block, p)[0], "std": _agg_part(block, p)[1]}
                for p in ["u1", "u2", "red", "syn"]
            }

        # one-line print (means only)
        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{summary['main']['test_tot_mean']:.3f} | "
              f"{summary['synib_tuned']['test_tot_mean']:.3f} | "
              f"{summary['learned_tuned']['test_tot_mean']:.3f} | "
              f"{summary['main']['test_syn_mean']:.3f} | "
              f"{summary['synib_tuned']['test_syn_mean']:.3f} | "
              f"{summary['learned_tuned']['test_syn_mean']:.3f} | ADD")

        # save
        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "cfg": getattr(cfg_base, "__dict__", {}),   # lightweight; or use asdict if cfg is dataclass
            "per_seed": per_seed,
            "summary_meanstd": summary,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return db
def main_sweep_nonoverlap_probs_synib_learned( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],):
    """
    For each non-overlap prob setting:
      - runs Main once per seed (val-checkpointed) and reports test
      - tunes lambda_kl for SynIB on val (no retrain), reports test of selected run
      - tunes lambda_kl for SynIB-Learned on val (no retrain), reports test of selected run
      - saves EVERYTHING to a JSON (append/dedupe, never overwrites previous entries)

    JSON entry contains:
      - probs
      - per-seed selected lambda_kl and test metrics (total + parts)
      - mean/std across seeds
      - (optional) the per-lambda validation tables for reproducibility
    """
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_tunedkl_synibleaneduni_snr3_v3_vf.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG (tuned kl: synib + learned)")
    print("[RUN] sweep non-overlap source probs + tune lambda_kl (val) for SynIB + SynIB-Learned")
    print(f"[LOG] append/dedupe json: {results_path}")
    print(f"[SEEDS] {seeds}")
    print("pu1 pu2 pred psyn | MainTot | SynIB(tuned)Tot | Learn(tuned)Tot | MainSyn | SynIBSyn | LearnSyn | status")
    print("--------------------------------------------------------------------------------------------------------")

    step = 0.05  # <--- smaller than 0.1
    prior_u2 = 0.0
    targets = [0.1, 0.2, 0.3]  # prioritize these values for element index 2 (psyn)

    prob_settings = (
            [(round(u1, 3), prior_u2, t, round(1 - u1 - prior_u2 - t, 3))
             for t in targets
             for u1 in np.arange(0, 1 - prior_u2 - t + 1e-12, step)]
            +
            [(round(u1, 3), prior_u2, round(psyn, 3), round(1 - u1 - prior_u2 - psyn, 3))
             for psyn in np.arange(0, 1 - prior_u2 + 1e-12, step)
             for u1 in np.arange(0, 1 - prior_u2 - psyn + 1e-12, step)
             if round(psyn, 3) not in targets]
    )

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std())

    def _acc_parts_from_by(by_source_dict):
        # returns dict with u1,u2,red,syn (NaN if missing)
        out = {"u1": float("nan"), "u2": float("nan"), "red": float("nan"), "syn": float("nan")}
        for k in out.keys():
            if k in by_source_dict and by_source_dict[k].get("n", 0) > 0:
                out[k] = float(by_source_dict[k].get("acc", float("nan")))
        return out

    def _tune_one_method(cfg_base, *, seed, method):
        """
        method in {"synib","learned"}
        returns dict:
          best_kl, best_test_stats, best_test_by_source, table
        """
        return select_best_lambda_kl_on_val(
            cfg_base,
            seed=seed,
            kl_vals=kl_vals,
            method=method,
            verbose=False,
        )

    for (pu1, pu2, psyn, pred) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)  # reuse your stable key function

        if key in db["results"]:
            summ = db["results"][key]["summary_meanstd"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{summ['learned_tuned']['test_tot_mean']:.3f} | "
                  f"{summ['learned_tuned']['test_syn_mean']:.3f} | SKIP(existing)")
            continue

        cfg_base = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg_base, pu1, pu2, pred, psyn, pnone=0.0)

        # store per-seed results
        per_seed = {
            "learned_tuned": [],
        }

        # run over seeds
        for seed in seeds:
            # --- tune learned ---
            lea_best = _tune_one_method(cfg_base, seed=seed, method="learned")
            lea_stats = lea_best["best_test_stats"]
            lea_by = lea_best["best_test_by_source"]
            per_seed["learned_tuned"].append({
                "seed": int(seed),
                "best_lambda_kl": float(lea_best["best_lambda_kl"]),
                "val_best_fusion_loss": float(lea_best["best_val_fusion_loss"]),
                "val_best_fusion_acc": float(lea_best["best_val_fusion_acc"]),
                "test_acc_fusion": float(lea_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(lea_by)),
                "test_acc_parts": _acc_parts_from_by(lea_by),
                "val_table": lea_best["table"],
            })

        # ---- aggregate mean/std across seeds ----
        def _agg(block, field):
            vals = [r[field] for r in per_seed[block]]
            return _mean_std(vals)

        def _agg_part(block, part):
            vals = [r["test_acc_parts"][part] for r in per_seed[block]]
            return _mean_std(vals)

        summary = {"learned_tuned": {}}
        for block in ["learned_tuned"]:
            m_tot, s_tot = _agg(block, "test_acc_fusion")
            m_syn, s_syn = _agg(block, "test_acc_syn")
            summary[block]["test_tot_mean"] = m_tot
            summary[block]["test_tot_std"] = s_tot
            summary[block]["test_syn_mean"] = m_syn
            summary[block]["test_syn_std"] = s_syn
            summary[block]["parts_meanstd"] = {
                p: {"mean": _agg_part(block, p)[0], "std": _agg_part(block, p)[1]}
                for p in ["u1", "u2", "red", "syn"]
            }

        # one-line print (means only)
        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{summary['learned_tuned']['test_tot_mean']:.3f} | "
              f"{summary['learned_tuned']['test_syn_mean']:.3f} | ADD")

        # save
        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "cfg": getattr(cfg_base, "__dict__", {}),   # lightweight; or use asdict if cfg is dataclass
            "per_seed": per_seed,
            "summary_meanstd": summary,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return db
def main_sweep_nonoverlap_probs_synib( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],):
    """
    For each non-overlap prob setting:
      - runs Main once per seed (val-checkpointed) and reports test
      - tunes lambda_kl for SynIB on val (no retrain), reports test of selected run
      - tunes lambda_kl for SynIB-Learned on val (no retrain), reports test of selected run
      - saves EVERYTHING to a JSON (append/dedupe, never overwrites previous entries)

    JSON entry contains:
      - probs
      - per-seed selected lambda_kl and test metrics (total + parts)
      - mean/std across seeds
      - (optional) the per-lambda validation tables for reproducibility
    """
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_tunedkl_synib_snr3_v3.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG (tuned kl: synib)")
    print("[RUN] sweep non-overlap source probs + tune lambda_kl (val) for SynIB")
    print(f"[LOG] append/dedupe json: {results_path}")
    print(f"[SEEDS] {seeds}")
    print("pu1 pu2 pred psyn | SynIB(tuned)Tot | SynIBSyn | status")
    print("--------------------------------------------------------------------------------------------------------")

    step = 0.05  # <--- smaller than 0.1
    prior_u2 = 0.0
    targets = [0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35, 0.45]  # prioritize these values for element index 2 (psyn)

    prob_settings = (
            [(round(u1, 3), prior_u2, t, round(1 - u1 - prior_u2 - t, 3))
             for t in targets
             for u1 in np.arange(0, 1 - prior_u2 - t + 1e-12, step)]
            +
            [(round(u1, 3), prior_u2, round(psyn, 3), round(1 - u1 - prior_u2 - psyn, 3))
             for psyn in np.arange(0, 1 - prior_u2 + 1e-12, step)
             for u1 in np.arange(0, 1 - prior_u2 - psyn + 1e-12, step)
             if round(psyn, 3) not in targets]
    )

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std())

    def _acc_parts_from_by(by_source_dict):
        # returns dict with u1,u2,red,syn (NaN if missing)
        out = {"u1": float("nan"), "u2": float("nan"), "red": float("nan"), "syn": float("nan")}
        for k in out.keys():
            if k in by_source_dict and by_source_dict[k].get("n", 0) > 0:
                out[k] = float(by_source_dict[k].get("acc", float("nan")))
        return out

    def _tune_one_method(cfg_base, *, seed, method):
        """
        method in {"synib","learned"}
        returns dict:
          best_kl, best_test_stats, best_test_by_source, table
        """
        return select_best_lambda_kl_on_val(
            cfg_base,
            seed=seed,
            kl_vals=kl_vals,
            method=method,
            verbose=False,
        )

    for (pu1, pu2, psyn, pred) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)  # reuse your stable key function

        if key in db["results"]:
            summ = db["results"][key]["summary_meanstd"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{summ['synib_tuned']['test_tot_mean']:.3f} | "
                  f"{summ['synib_tuned']['test_syn_mean']:.3f} | "
                  f" SKIP(existing)")
            continue

        cfg_base = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg_base, pu1, pu2, pred, psyn, pnone=0.0)

        # store per-seed results
        per_seed = {
            "main": [],
            "synib_tuned": [],
            "learned_tuned": [],
        }

        # run over seeds
        for seed in seeds:

            # --- tune synib (oracle) ---
            syn_best = _tune_one_method(cfg_base, seed=seed, method="synib")
            syn_stats = syn_best["best_test_stats"]
            syn_by = syn_best["best_test_by_source"]
            per_seed["synib_tuned"].append({
                "seed": int(seed),
                "best_lambda_kl": float(syn_best["best_lambda_kl"]),
                "val_best_fusion_loss": float(syn_best["best_val_fusion_loss"]),
                "test_acc_fusion": float(syn_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(syn_by)),
                "test_acc_parts": _acc_parts_from_by(syn_by),
                "val_table": syn_best["table"],  # keep for reproducibility
            })

        # ---- aggregate mean/std across seeds ----
        def _agg(block, field):
            vals = [r[field] for r in per_seed[block]]
            return _mean_std(vals)

        def _agg_part(block, part):
            vals = [r["test_acc_parts"][part] for r in per_seed[block]]
            return _mean_std(vals)

        summary = {"main": {}, "synib_tuned": {}, "learned_tuned": {}}
        for block in ["synib_tuned"]:
            m_tot, s_tot = _agg(block, "test_acc_fusion")
            m_syn, s_syn = _agg(block, "test_acc_syn")
            summary[block]["test_tot_mean"] = m_tot
            summary[block]["test_tot_std"] = s_tot
            summary[block]["test_syn_mean"] = m_syn
            summary[block]["test_syn_std"] = s_syn
            summary[block]["parts_meanstd"] = {
                p: {"mean": _agg_part(block, p)[0], "std": _agg_part(block, p)[1]}
                for p in ["u1", "u2", "red", "syn"]
            }

        # one-line print (means only)
        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{summary['synib_tuned']['test_tot_mean']:.3f} | "
              f"{summary['synib_tuned']['test_syn_mean']:.3f} | "
              f"ADD")

        # save
        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "cfg": getattr(cfg_base, "__dict__", {}),   # lightweight; or use asdict if cfg is dataclass
            "per_seed": per_seed,
            "summary_meanstd": summary,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return db
def main_sweep_nonoverlap_probs_synib_random( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],):

    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_tunedkl_synib_random03_snr3_v3.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG (tuned kl: synib random M)")
    print("[RUN] sweep non-overlap source probs + tune lambda_kl (val) for SynIB RM")
    print(f"[LOG] append/dedupe json: {results_path}")
    print(f"[SEEDS] {seeds}")
    print("pu1 pu2 pred psyn | SynIB(tuned)Tot RM | SynIBSyn RM | status")
    print("--------------------------------------------------------------------------------------------------------")

    step = 0.05  # <--- smaller than 0.1
    prior_u2 = 0.0
    targets = [0.1, 0.2, 0.3, 0.4]  # prioritize these values for element index 2 (psyn)

    prob_settings = (
            [(round(u1, 3), prior_u2, t, round(1 - u1 - prior_u2 - t, 3))
             for t in targets
             for u1 in np.arange(0, 1 - prior_u2 - t + 1e-12, step)]
            +
            [(round(u1, 3), prior_u2, round(psyn, 3), round(1 - u1 - prior_u2 - psyn, 3))
             for psyn in np.arange(0, 1 - prior_u2 + 1e-12, step)
             for u1 in np.arange(0, 1 - prior_u2 - psyn + 1e-12, step)
             if round(psyn, 3) not in targets]
    )

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std())

    def _acc_parts_from_by(by_source_dict):
        # returns dict with u1,u2,red,syn (NaN if missing)
        out = {"u1": float("nan"), "u2": float("nan"), "red": float("nan"), "syn": float("nan")}
        for k in out.keys():
            if k in by_source_dict and by_source_dict[k].get("n", 0) > 0:
                out[k] = float(by_source_dict[k].get("acc", float("nan")))
        return out

    def _tune_one_method(cfg_base, *, seed, method):
        """
        method in {"synib","learned"}
        returns dict:
          best_kl, best_test_stats, best_test_by_source, table
        """
        return select_best_lambda_kl_on_val(
            cfg_base,
            seed=seed,
            kl_vals=kl_vals,
            method=method,
            verbose=False,
        )

    for (pu1, pu2, psyn, pred) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)  # reuse your stable key function

        if key in db["results"]:
            summ = db["results"][key]["summary_meanstd"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{summ['synib_tuned']['test_tot_mean']:.3f} | "
                  f"{summ['synib_tuned']['test_syn_mean']:.3f} | "
                  f" SKIP(existing)")
            continue

        cfg_base = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg_base, pu1, pu2, pred, psyn, pnone=0.0)

        # store per-seed results
        per_seed = {
            "main": [],
            "synib_tuned": [],
            "synib_RM_tuned": [],
            "learned_tuned": [],
        }

        for seed in seeds:
            syn_best = _tune_one_method(cfg_base, seed=seed, method="synib_random")
            syn_stats = syn_best["best_test_stats"]
            syn_by = syn_best["best_test_by_source"]
            per_seed["synib_RM_tuned"].append({
                "seed": int(seed),
                "best_lambda_kl": float(syn_best["best_lambda_kl"]),
                "val_best_fusion_loss": float(syn_best["best_val_fusion_loss"]),
                "test_acc_fusion": float(syn_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(syn_by)),
                "test_acc_parts": _acc_parts_from_by(syn_by),
                "val_table": syn_best["table"],  # keep for reproducibility
            })

        # ---- aggregate mean/std across seeds ----
        def _agg(block, field):
            vals = [r[field] for r in per_seed[block]]
            return _mean_std(vals)

        def _agg_part(block, part):
            vals = [r["test_acc_parts"][part] for r in per_seed[block]]
            return _mean_std(vals)

        summary = {"synib_RM_tuned": {}}
        for block in ["synib_RM_tuned"]:
            m_tot, s_tot = _agg(block, "test_acc_fusion")
            m_syn, s_syn = _agg(block, "test_acc_syn")
            summary[block]["test_tot_mean"] = m_tot
            summary[block]["test_tot_std"] = s_tot
            summary[block]["test_syn_mean"] = m_syn
            summary[block]["test_syn_std"] = s_syn
            summary[block]["parts_meanstd"] = {
                p: {"mean": _agg_part(block, p)[0], "std": _agg_part(block, p)[1]}
                for p in ["u1", "u2", "red", "syn"]
            }

        # one-line print (means only)
        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{summary['synib_RM_tuned']['test_tot_mean']:.3f} | "
              f"{summary['synib_RM_tuned']['test_syn_mean']:.3f} | "
              f"ADD")

        # save
        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "cfg": getattr(cfg_base, "__dict__", {}),   # lightweight; or use asdict if cfg is dataclass
            "per_seed": per_seed,
            "summary_meanstd": summary,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return db
def main_sweep_nonoverlap_probs_synib_randomdiff( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],):

    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_tunedkl_synib_random03diff_10-07-1_snr3_v3.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG (tuned kl: synib random M)")
    print("[RUN] sweep non-overlap source probs + tune lambda_kl (val) for SynIB RM")
    print(f"[LOG] append/dedupe json: {results_path}")
    print(f"[SEEDS] {seeds}")
    print("pu1 pu2 pred psyn | SynIB(tuned)Tot RM | SynIBSyn RM | status")
    print("--------------------------------------------------------------------------------------------------------")

    step = 0.05  # <--- smaller than 0.1
    prior_u2 = 0.0
    targets = [0.05, 0.1, 0.2, 0.3, 0.4]  # prioritize these values for element index 2 (psyn)

    prob_settings = (
            [(round(u1, 3), prior_u2, t, round(1 - u1 - prior_u2 - t, 3))
             for t in targets
             for u1 in np.arange(0, 1 - prior_u2 - t + 1e-12, step)]
            +
            [(round(u1, 3), prior_u2, round(psyn, 3), round(1 - u1 - prior_u2 - psyn, 3))
             for psyn in np.arange(0, 1 - prior_u2 + 1e-12, step)
             for u1 in np.arange(0, 1 - prior_u2 - psyn + 1e-12, step)
             if round(psyn, 3) not in targets]
    )

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std())

    def _acc_parts_from_by(by_source_dict):
        # returns dict with u1,u2,red,syn (NaN if missing)
        out = {"u1": float("nan"), "u2": float("nan"), "red": float("nan"), "syn": float("nan")}
        for k in out.keys():
            if k in by_source_dict and by_source_dict[k].get("n", 0) > 0:
                out[k] = float(by_source_dict[k].get("acc", float("nan")))
        return out

    def _tune_one_method(cfg_base, *, seed, method):
        """
        method in {"synib","learned"}
        returns dict:
          best_kl, best_test_stats, best_test_by_source, table
        """
        return select_best_lambda_kl_on_val(
            cfg_base,
            seed=seed,
            kl_vals=kl_vals,
            method=method,
            verbose=False,
        )

    for (pu1, pu2, psyn, pred) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)  # reuse your stable key function

        if key in db["results"]:
            summ = db["results"][key]["summary_meanstd"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{summ['synib_RM_tuned']['test_tot_mean']:.3f} | "
                  f"{summ['synib_RM_tuned']['test_syn_mean']:.3f} | "
                  f" SKIP(existing)")
            continue

        cfg_base = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg_base, pu1, pu2, pred, psyn, pnone=0.0)

        # store per-seed results
        per_seed = {
            "main": [],
            "synib_tuned": [],
            "synib_RM_tuned": [],
            "learned_tuned": [],
        }

        for seed in seeds:
            syn_best = _tune_one_method(cfg_base, seed=seed, method="synib_randomdiff")
            syn_stats = syn_best["best_test_stats"]
            syn_by = syn_best["best_test_by_source"]
            per_seed["synib_RM_tuned"].append({
                "seed": int(seed),
                "best_lambda_kl": float(syn_best["best_lambda_kl"]),
                "val_best_fusion_loss": float(syn_best["best_val_fusion_loss"]),
                "test_acc_fusion": float(syn_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(syn_by)),
                "test_acc_parts": _acc_parts_from_by(syn_by),
                "val_table": syn_best["table"],  # keep for reproducibility
            })

        # ---- aggregate mean/std across seeds ----
        def _agg(block, field):
            vals = [r[field] for r in per_seed[block]]
            return _mean_std(vals)

        def _agg_part(block, part):
            vals = [r["test_acc_parts"][part] for r in per_seed[block]]
            return _mean_std(vals)

        summary = {"synib_RM_tuned": {}}
        for block in ["synib_RM_tuned"]:
            m_tot, s_tot = _agg(block, "test_acc_fusion")
            m_syn, s_syn = _agg(block, "test_acc_syn")
            summary[block]["test_tot_mean"] = m_tot
            summary[block]["test_tot_std"] = s_tot
            summary[block]["test_syn_mean"] = m_syn
            summary[block]["test_syn_std"] = s_syn
            summary[block]["parts_meanstd"] = {
                p: {"mean": _agg_part(block, p)[0], "std": _agg_part(block, p)[1]}
                for p in ["u1", "u2", "red", "syn"]
            }

        # one-line print (means only)
        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{summary['synib_RM_tuned']['test_tot_mean']:.3f} | "
              f"{summary['synib_RM_tuned']['test_syn_mean']:.3f} | "
              f"ADD")

        # save
        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "cfg": getattr(cfg_base, "__dict__", {}),   # lightweight; or use asdict if cfg is dataclass
            "per_seed": per_seed,
            "summary_meanstd": summary,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return db

def main_sweep_nonoverlap_probs_main( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-1, 1e0, 1e1, 1e2],):
    """
    For each non-overlap prob setting:
      - runs Main once per seed (val-checkpointed) and reports test
      - tunes lambda_kl for SynIB on val (no retrain), reports test of selected run
      - tunes lambda_kl for SynIB-Learned on val (no retrain), reports test of selected run
      - saves EVERYTHING to a JSON (append/dedupe, never overwrites previous entries)

    JSON entry contains:
      - probs
      - per-seed selected lambda_kl and test metrics (total + parts)
      - mean/std across seeds
      - (optional) the per-lambda validation tables for reproducibility
    """
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_main_snr3_v3.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG (tuned kl: synib + learned)")
    print("[RUN] sweep non-overlap source probs + tune lambda_kl (val) for SynIB + SynIB-Learned")
    print(f"[LOG] append/dedupe json: {results_path}")
    print(f"[SEEDS] {seeds}")
    print("pu1 pu2 pred psyn | MainTot | MainSyn |")
    print("--------------------------------------------------------------------------------------------------------")

    step = 0.05  # <--- smaller than 0.1
    prior_u2 = 0.0
    targets = [0.3, 0.4, 0.1, 0.2]  # prioritize these values for element index 2 (psyn)

    prob_settings = (
            [(round(u1, 3), prior_u2, t, round(1 - u1 - prior_u2 - t, 3))
             for t in targets
             for u1 in np.arange(0, 1 - prior_u2 - t + 1e-12, step)]
            +
            [(round(u1, 3), prior_u2, round(psyn, 3), round(1 - u1 - prior_u2 - psyn, 3))
             for psyn in np.arange(0, 1 - prior_u2 + 1e-12, step)
             for u1 in np.arange(0, 1 - prior_u2 - psyn + 1e-12, step)
             if round(psyn, 3) not in targets]
    )

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std())

    def _acc_parts_from_by(by_source_dict):
        # returns dict with u1,u2,red,syn (NaN if missing)
        out = {"u1": float("nan"), "u2": float("nan"), "red": float("nan"), "syn": float("nan")}
        for k in out.keys():
            if k in by_source_dict and by_source_dict[k].get("n", 0) > 0:
                out[k] = float(by_source_dict[k].get("acc", float("nan")))
        return out

    for (pu1, pu2, psyn, pred) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)  # reuse your stable key function

        if key in db["results"]:
            summ = db["results"][key]["summary_meanstd"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{summ['main']['test_tot_mean']:.3f} | "
                  f"{summ['main']['test_syn_mean']:.3f} | "
                  f"SKIP(existing)")
            continue

        cfg_base = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg_base, pu1, pu2, pred, psyn, pnone=0.0)

        # store per-seed results
        per_seed = {
            "main": []
        }

        # run over seeds
        for seed in seeds:
            # --- build loaders once per seed for this prob setting ---
            device, split, train_loader, val_loader, test_loader = build_loaders(cfg_base, seed=seed, verbose=False)

            # --- main ---
            main_out = run_main(cfg_base, train_loader, val_loader, test_loader, device, verbose=False)
            main_stats = main_out["stats"]
            main_by = main_out["by_source"]

            main_rec = {
                "seed": int(seed),
                "test_acc_fusion": float(main_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(main_by)),
                "test_acc_parts": _acc_parts_from_by(main_by),
            }
            per_seed["main"].append(main_rec)

        # ---- aggregate mean/std across seeds ----
        def _agg(block, field):
            vals = [r[field] for r in per_seed[block]]
            return _mean_std(vals)

        def _agg_part(block, part):
            vals = [r["test_acc_parts"][part] for r in per_seed[block]]
            return _mean_std(vals)

        summary = {"main": {}, "synib_tuned": {}, "learned_tuned": {}}
        for block in ["main"]:
            m_tot, s_tot = _agg(block, "test_acc_fusion")
            m_syn, s_syn = _agg(block, "test_acc_syn")
            summary[block]["test_tot_mean"] = m_tot
            summary[block]["test_tot_std"] = s_tot
            summary[block]["test_syn_mean"] = m_syn
            summary[block]["test_syn_std"] = s_syn
            summary[block]["parts_meanstd"] = {
                p: {"mean": _agg_part(block, p)[0], "std": _agg_part(block, p)[1]}
                for p in ["u1", "u2", "red", "syn"]
            }

        # one-line print (means only)
        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{summary['main']['test_tot_mean']:.3f} | "
              f"{summary['main']['test_syn_mean']:.3f} | ")

        # save
        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "cfg": getattr(cfg_base, "__dict__", {}),   # lightweight; or use asdict if cfg is dataclass
            "per_seed": per_seed,
            "summary_meanstd": summary,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return db
def main_sweep_nonoverlap_probs_mainmask( *, seeds: list = [0, 1, 2, 3, 4], kl_vals: list = [1e-1, 1e0, 1e1, 1e2],):
    """
    For each non-overlap prob setting:
      - runs Main once per seed (val-checkpointed) and reports test
      - tunes lambda_kl for SynIB on val (no retrain), reports test of selected run
      - tunes lambda_kl for SynIB-Learned on val (no retrain), reports test of selected run
      - saves EVERYTHING to a JSON (append/dedupe, never overwrites previous entries)

    JSON entry contains:
      - probs
      - per-seed selected lambda_kl and test metrics (total + parts)
      - mean/std across seeds
      - (optional) the per-lambda validation tables for reproducibility
    """
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_mainmask03star_snr3_v3_v2.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG (tuned kl: synib + learned)")
    print("[RUN] sweep non-overlap source probs + tune lambda_kl (val) for SynIB + SynIB-Learned")
    print(f"[LOG] append/dedupe json: {results_path}")
    print(f"[SEEDS] {seeds}")
    print("pu1 pu2 pred psyn | MainTot | MainSyn |")
    print("--------------------------------------------------------------------------------------------------------")

    step = 0.05  # <--- smaller than 0.1
    prior_u2 = 0.0
    targets = [0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35, 0.45]  # prioritize these values for element index 2 (psyn)

    prob_settings = (
            [(round(u1, 3), prior_u2, t, round(1 - u1 - prior_u2 - t, 3))
             for t in targets
             for u1 in np.arange(0, 1 - prior_u2 - t + 1e-12, step)]
            +
            [(round(u1, 3), prior_u2, round(psyn, 3), round(1 - u1 - prior_u2 - psyn, 3))
             for psyn in np.arange(0, 1 - prior_u2 + 1e-12, step)
             for u1 in np.arange(0, 1 - prior_u2 - psyn + 1e-12, step)
             if round(psyn, 3) not in targets]
    )

    def _mean_std(xs):
        xs = np.asarray(xs, dtype=float)
        return float(xs.mean()), float(xs.std())

    def _acc_parts_from_by(by_source_dict):
        # returns dict with u1,u2,red,syn (NaN if missing)
        out = {"u1": float("nan"), "u2": float("nan"), "red": float("nan"), "syn": float("nan")}
        for k in out.keys():
            if k in by_source_dict and by_source_dict[k].get("n", 0) > 0:
                out[k] = float(by_source_dict[k].get("acc", float("nan")))
        return out

    for (pu1, pu2, psyn, pred) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)  # reuse your stable key function

        if key in db["results"]:
            summ = db["results"][key]["summary_meanstd"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{summ['main']['test_tot_mean']:.3f} | "
                  f"{summ['main']['test_syn_mean']:.3f} | "
                  f"SKIP(existing)")
            continue

        cfg_base = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg_base, pu1, pu2, pred, psyn, pnone=0.0)

        # store per-seed results
        per_seed = {
            "main": []
        }

        # run over seeds
        for seed in seeds:
            # --- build loaders once per seed for this prob setting ---
            device, split, train_loader, val_loader, test_loader = build_loaders(cfg_base, seed=seed, verbose=False)

            # --- main ---
            main_out = run_mainmask(cfg_base, train_loader, val_loader, test_loader, device, verbose=False)
            main_stats = main_out["stats"]
            main_by = main_out["by_source"]

            main_rec = {
                "seed": int(seed),
                "test_acc_fusion": float(main_stats["acc_fusion"]),
                "test_acc_syn": float(_syn_acc_from_by_source(main_by)),
                "test_acc_parts": _acc_parts_from_by(main_by),
            }
            per_seed["main"].append(main_rec)

        # ---- aggregate mean/std across seeds ----
        def _agg(block, field):
            vals = [r[field] for r in per_seed[block]]
            return _mean_std(vals)

        def _agg_part(block, part):
            vals = [r["test_acc_parts"][part] for r in per_seed[block]]
            return _mean_std(vals)

        summary = {"main": {}, "synib_tuned": {}, "learned_tuned": {}}
        for block in ["main"]:
            m_tot, s_tot = _agg(block, "test_acc_fusion")
            m_syn, s_syn = _agg(block, "test_acc_syn")
            summary[block]["test_tot_mean"] = m_tot
            summary[block]["test_tot_std"] = s_tot
            summary[block]["test_syn_mean"] = m_syn
            summary[block]["test_syn_std"] = s_syn
            summary[block]["parts_meanstd"] = {
                p: {"mean": _agg_part(block, p)[0], "std": _agg_part(block, p)[1]}
                for p in ["u1", "u2", "red", "syn"]
            }

        # one-line print (means only)
        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{summary['main']['test_tot_mean']:.3f} | "
              f"{summary['main']['test_syn_mean']:.3f} | ")

        # save
        db["results"][key] = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "kl_vals": list(kl_vals),
            "cfg": getattr(cfg_base, "__dict__", {}),   # lightweight; or use asdict if cfg is dataclass
            "per_seed": per_seed,
            "summary_meanstd": summary,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return db

def _set_nonoverlap_signal_probs(cfg, pu1: float, pu2: float, pred: float, psyn: float, pnone: float = 0.0):
    s = pu1 + pu2 + pred + psyn + pnone
    if abs(s - 1.0) > 1e-8:
        raise ValueError(f"Non-overlap probs must sum to 1. Got {s}.")
    cfg.signal_probs = {
        "u1": float(pu1),
        "u2": float(pu2),
        "red": float(pred),
        "syn": float(psyn),
    }
    if pnone > 0:
        cfg.signal_probs["none"] = float(pnone)
def _prob_key(pu1: float, pu2: float, pred: float, psyn: float, pnone: float = 0.0) -> str:
    return f"pu1={pu1:.4f}|pu2={pu2:.4f}|pred={pred:.4f}|psyn={psyn:.4f}|pnone={pnone:.4f}"
def _load_results_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"schema": "sweep_nonoverlap_probs_v2", "results": {}}
    with open(path, "r") as f:
        return json.load(f)
def _save_results_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    os.replace(tmp, path)
def _slice_acc_from_by_source(by: Dict[str, Any], want: str) -> float:
    """
    Non-overlap case: keys are exactly "u1", "u2", "red", "syn", or "none".
    Returns weighted acc on the `want` slice, or NaN if not present.
    """
    if "by_source" not in by:
        return float("nan")
    if want not in by["by_source"]:
        return float("nan")
    v = by["by_source"][want]
    n = int(v.get("n", 0))
    return float(v.get("acc", float("nan"))) if n > 0 else float("nan")
def _count_from_by_source(by: Dict[str, Any], want: str) -> int:
    if "by_source" not in by:
        return 0
    return int(by["by_source"].get(want, {}).get("n", 0))
def _summarize_rep(rep: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact summary:
      - overall test acc
      - per-part test acc for u1/u2/red/syn (for both Main & SynIB)
      - counts per part
    Requires run_once to include rep["by_source"]["main"/"synib"] from eval_by_source.
    """
    out = {
        "seed": rep.get("seed"),
        "detectability_acc": float(rep["detectability_acc"]),
        "main": {"acc_fusion": float(rep["main"]["acc_fusion"])},
        "synib": {"acc_fusion": float(rep["synib"]["acc_fusion"])},
    }

    if "by_source" in rep:
        main_by = rep["by_source"]["main"]
        syn_by  = rep["by_source"]["synib"]

        out["main"]["acc_parts"] = {k: _slice_acc_from_by_source(main_by, k) for k in ("u1", "u2", "red", "syn")}
        out["synib"]["acc_parts"] = {k: _slice_acc_from_by_source(syn_by, k) for k in ("u1", "u2", "red", "syn")}

        # counts (same for main/synib since same test set)
        out["counts"] = {k: _count_from_by_source(main_by, k) for k in ("u1", "u2", "red", "syn", "none")}

    return out
def main_sweep_nonoverlap_probs():
    cfg0 = Config()
    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg0.out_dir)

    # NEW file (keeps old one untouched)
    results_path = os.path.join(cfg0.out_dir, "sweep_nonoverlap_probs_parts.json")
    db = _load_results_json(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG")
    print("[RUN] sweep non-overlap source probs (u1/u2/red/syn)")
    print(f"[LOG] append/dedupe json: {results_path}")
    print("pu1 pu2 pred psyn | MainTot SynIBTot | Main(u1,u2,red,syn) | SynIB(u1,u2,red,syn) | det | status")
    print("---------------------------------------------------------------------------------------------------")

    prob_settings = [
        (0.30, 0.30, 0.00, 0.40),
        (0.20, 0.20, 0.20, 0.40),
        (0.15, 0.15, 0.30, 0.40),
        (0.10, 0.10, 0.40, 0.40),
        (0.10, 0.10, 0.20, 0.60),
        (0.40, 0.40, 0.10, 0.10),
    ]

    reps = []
    for (pu1, pu2, pred, psyn) in prob_settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)

        if key in db["results"]:
            summ = db["results"][key]["summary"]
            mt, st, det = summ["main"]["acc_fusion"], summ["synib"]["acc_fusion"], summ["detectability_acc"]
            mparts = summ["main"]["acc_parts"]
            sparts = summ["synib"]["acc_parts"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{mt:.3f}  {st:.3f}  | "
                  f"{mparts['u1']:.3f},{mparts['u2']:.3f},{mparts['red']:.3f},{mparts['syn']:.3f} | "
                  f"{sparts['u1']:.3f},{sparts['u2']:.3f},{sparts['red']:.3f},{sparts['syn']:.3f} | "
                  f"{det:.3f} | SKIP(existing)")
            continue

        cfg = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg, pu1, pu2, pred, psyn, pnone=0.0)

        tag = f"pu1_{pu1:.2f}_pu2_{pu2:.2f}_pred_{pred:.2f}_psyn_{psyn:.2f}"
        cfg.out_dir = os.path.join(cfg0.out_dir, f"sweep_probs_{tag}")
        mkdirp(cfg.out_dir)

        rep = run_once(cfg, seed=0, verbose=False)  # must return by_source
        reps.append(((pu1, pu2, pred, psyn), rep))

        summ = _summarize_rep(rep)  # now includes acc_parts
        mt, st, det = summ["main"]["acc_fusion"], summ["synib"]["acc_fusion"], summ["detectability_acc"]
        mparts = summ["main"]["acc_parts"]
        sparts = summ["synib"]["acc_parts"]

        print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
              f"{mt:.3f}  {st:.3f}  | "
              f"{mparts['u1']:.3f},{mparts['u2']:.3f},{mparts['red']:.3f},{mparts['syn']:.3f} | "
              f"{sparts['u1']:.3f},{sparts['u2']:.3f},{sparts['red']:.3f},{sparts['syn']:.3f} | "
              f"{det:.3f} | ADD")

        _plot_loss_curves(rep, cfg.out_dir)
        _plot_acc_curves(rep, cfg.out_dir)

        db["results"][key] = {
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "cfg": rep.get("cfg", {}),
            "split": rep.get("split", {}),
            "summary": summ,
        }
        _save_results_json(results_path, db)

    _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return reps

def build_loaders(cfg: Config, *, seed: int, verbose: bool = True):
    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(seed)

    full_train_ds = PID4BlockDataset(cfg, cfg.n_train, seed=seed, split="train", verbose=verbose)
    test_ds = PID4BlockDataset(cfg, cfg.n_test, seed=seed + 1, split="test", train_stats=full_train_ds.stats, verbose=verbose)

    val_frac = getattr(cfg, "val_frac", 0.10)
    n = len(full_train_ds)
    n_val = max(1, int(round(val_frac * n)))
    n_train = n - n_val

    g = torch.Generator()
    g.manual_seed(seed + 12345)
    perm = torch.randperm(n, generator=g).tolist()
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_ds = Subset(full_train_ds, train_idx)
    val_ds   = Subset(full_train_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False)

    split = {"n_train": n_train, "n_val": n_val, "n_test": len(test_ds)}
    return device, split, train_loader, val_loader, test_loader
def run_main(cfg: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: str, *, verbose: bool = True):
    main_model, main_hist = train_main(cfg, train_loader, device, val_loader=val_loader)
    main_stats = eval_clean(main_model, test_loader, device)
    main_by = eval_by_source(main_model, test_loader, device)
    if verbose:
        print_by_source("Main", main_by, min_n=10)
    return {
        "model": main_model,
        "stats": main_stats,
        "history": main_hist,
        "by_source": main_by,
    }
def run_mainmask(cfg: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: str, *, verbose: bool = True):
    main_model, main_hist = train_mainmask(cfg, train_loader, device, val_loader=val_loader)
    main_stats = eval_clean(main_model, test_loader, device)
    main_by = eval_by_source(main_model, test_loader, device)
    if verbose:
        print_by_source("Main", main_by, min_n=10)
    return {
        "model": main_model,
        "stats": main_stats,
        "history": main_hist,
        "by_source": main_by,
    }
def run_synib(cfg: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: str, *, verbose: bool = True):
    synib_model, synib_hist = train_synib(cfg, train_loader, device, val_loader=val_loader)
    synib_stats = eval_clean(synib_model, test_loader, device)
    ab = eval_block_ablations(synib_model, test_loader, device, noise_std=1.0)
    syn_by = eval_by_source(synib_model, test_loader, device)
    if verbose:
        print_by_source("SynIB", syn_by, min_n=10)
    return {
        "model": synib_model,
        "stats": synib_stats,
        "history": synib_hist,
        "by_source": syn_by,
        "ablations": ab,
    }

def run_synib_random(cfg: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: str, *, verbose: bool = True):
    synib_model, synib_hist = train_synib_random(cfg, train_loader, device, val_loader=val_loader)
    synib_stats = eval_clean(synib_model, test_loader, device)
    ab = eval_block_ablations(synib_model, test_loader, device, noise_std=1.0)
    syn_by = eval_by_source(synib_model, test_loader, device)
    if verbose:
        print_by_source("SynIB", syn_by, min_n=10)
    return {
        "model": synib_model,
        "stats": synib_stats,
        "history": synib_hist,
        "by_source": syn_by,
        "ablations": ab,
    }
def run_synib_randomdiff(cfg: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: str, *, verbose: bool = True):
    synib_model, synib_hist = train_synib_randomdiff(cfg, train_loader, device, val_loader=val_loader)
    synib_stats = eval_clean(synib_model, test_loader, device)
    ab = eval_block_ablations(synib_model, test_loader, device, noise_std=1.0)
    syn_by = eval_by_source(synib_model, test_loader, device)
    if verbose:
        print_by_source("SynIB", syn_by, min_n=10)
    return {
        "model": synib_model,
        "stats": synib_stats,
        "history": synib_hist,
        "by_source": syn_by,
        "ablations": ab,
    }
def run_once(cfg: Config, *, seed: int, dont_compute_main: bool = False, verbose: bool = True) -> Dict[str, Any]:
    """
    Orchestrates:
      - data + loaders
      - detectability probe
      - (optional) main
      - synib
      - aggregates results
    """
    device, split, train_loader, val_loader, test_loader = build_loaders(cfg, seed=seed, verbose=verbose)

    det = run_detectability_probe(
        train_loader,
        device,
        dim0=cfg.dim0,
        dim1=cfg.dim1,
        destroy_block_list=DESTROY_MASK,
        noise_std=1.0,
    )

    out: Dict[str, Any] = {
        "cfg": asdict(cfg),
        "seed": seed,
        "split": split,
        "detectability_acc": det,
        "history": {},
        "by_source": {},
    }

    if not dont_compute_main:
        main_out = run_main(cfg, train_loader, val_loader, test_loader, device, verbose=verbose)
        out["main"] = main_out["stats"]
        out["history"]["main"] = main_out["history"]
        out["by_source"]["main"] = main_out["by_source"]

    synib_out = run_synib(cfg, train_loader, val_loader, test_loader, device, verbose=verbose)
    out["synib"] = synib_out["stats"]
    out["ablations"] = synib_out["ablations"]
    out["history"]["synib"] = synib_out["history"]
    out["by_source"]["synib"] = synib_out["by_source"]

    return out

class DetectProbe(nn.Module):
    def __init__(self, dim0: int, dim1: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim0 + dim1, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
@torch.no_grad()
def make_detect_dataset(loader: DataLoader, device: str, *, destroy_block_list: list, noise_std: float = 1.0, max_batches: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    nb = 0
    for b in loader:
        x0, x1 = b["x0"].to(device), b["x1"].to(device)
        m0, m1 = b["mask0"].to(device), b["mask1"].to(device)

        xs.append(torch.cat([x0, x1], dim=-1)); ys.append(torch.zeros(x0.size(0), 1, device=device))
        x0p = destroy_block(x0, m0, destroy_block_list, noise_std=noise_std)
        x1p = destroy_block(x1, m1, destroy_block_list, noise_std=noise_std)
        xs.append(torch.cat([x0p, x1p], dim=-1)); ys.append(torch.ones(x0.size(0), 1, device=device))

        nb += 1
        if nb >= max_batches: break
    X = torch.cat(xs, dim=0); Y = torch.cat(ys, dim=0)
    perm = torch.randperm(X.size(0), device=device)
    return X[perm], Y[perm]
def destroy_block(x: torch.Tensor, mask: torch.Tensor, block_list: List[int], *, noise_std: float = 1.0) -> torch.Tensor:
    """
    Replace coordinates whose mask id is in `block_list` with N(0, noise_std^2).

    Args:
      x:          [B,D] or [D]
      mask:       [B,D] or [D] integer block ids
      block_list: e.g. [2] or [2,3]
      noise_std:  std of replacement noise
    """
    if not isinstance(block_list, list) or len(block_list) == 0:
        return x.clone()

    x_t = x.clone()

    # normalize shapes to [B,D]
    squeezed = False
    if x_t.dim() == 1:
        x_t = x_t.unsqueeze(0)
        squeezed = True
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)

    if mask.shape != x_t.shape:
        mask = mask.expand_as(x_t)

    ids = torch.tensor(block_list, device=mask.device, dtype=mask.dtype)
    m = torch.isin(mask, ids)

    if m.any():
        x_t[m] = torch.randn_like(x_t[m]) * float(noise_std)

    return x_t.squeeze(0) if squeezed else x_t
def run_detectability_probe(train_loader: DataLoader, device: str, *, dim0: int, dim1: int, destroy_block_list: list, noise_std: float = 1.0, steps: int = 150, lr: float = 3e-4) -> float:
    X, Y = make_detect_dataset(train_loader, device, destroy_block_list=destroy_block_list, noise_std=noise_std)
    probe = DetectProbe(dim0, dim1).to(device)
    opt = optim.Adam(probe.parameters(), lr=lr)
    n = X.size(0); ntr = int(0.8 * n)
    Xtr, Ytr = X[:ntr], Y[:ntr]; Xte, Yte = X[ntr:], Y[ntr:]
    for _ in range(steps):
        idx = torch.randint(0, Xtr.size(0), (256,), device=device)
        xb, yb = Xtr[idx], Ytr[idx]
        opt.zero_grad()
        logits = probe(xb)
        loss = F.binary_cross_entropy_with_logits(logits, yb)
        loss.backward(); opt.step()
    with torch.no_grad():
        pred = (probe(Xte) > 0).float()
        return float((pred == Yte).float().mean().item())

def main():
    cfg = Config()
    cfg.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg.out_dir)

    print_config(cfg, "DEFAULT CONFIG")
    print("[RUN] single benchmark")
    print_run_header()

    # -------------------- loaders + probe --------------------
    device, split, train_loader, val_loader, test_loader = build_loaders(cfg, seed=0, verbose=True)

    det = run_detectability_probe(
        train_loader,
        device,
        dim0=cfg.dim0,
        dim1=cfg.dim1,
        destroy_block_list=DESTROY_MASK,
        noise_std=1.0,
    )

    # -------------------- run models --------------------
    main_out  = run_main(cfg, train_loader, val_loader, test_loader, device, verbose=True)
    synib_out = run_synib(cfg, train_loader, val_loader, test_loader, device, verbose=True)

    # -------------------- pack rep (same shape expected by plotting/sanity) --------------------
    rep = {
        "cfg": asdict(cfg),
        "seed": 0,
        "split": split,
        "detectability_acc": det,
        "main": main_out["stats"],
        "synib": synib_out["stats"],
        "ablations": synib_out["ablations"],
        "history": {"main": main_out["history"], "synib": synib_out["history"]},
        "by_source": {"main": main_out["by_source"], "synib": synib_out["by_source"]},
    }

    # -------------------- prints --------------------
    print_row(0, "Main", rep["main"])
    print_row(0, "SynIB", rep["synib"])
    print_sanity(rep)

    # -------------------- plots --------------------
    loss_path = _plot_loss_curves(rep, cfg.out_dir)
    acc_path  = _plot_acc_curves(rep, cfg.out_dir)
    print(f"[SAVED] {loss_path}")
    print(f"[SAVED] {acc_path}")

def main_learned():
    cfg = Config()
    cfg.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    mkdirp(cfg.out_dir)

    print_config(cfg, "DEFAULT CONFIG (LEARNED)")
    print("[RUN] single benchmark (SynIB-Learned)")
    print_run_header()

    # -------------------- loaders + probe --------------------
    device, split, train_loader, val_loader, test_loader = build_loaders(cfg, seed=0, verbose=True)

    det = run_detectability_probe(
        train_loader,
        device,
        dim0=cfg.dim0,
        dim1=cfg.dim1,
        destroy_block_list=DESTROY_MASK,  # probe can stay oracle-based; optional
        noise_std=1.0,
    )

    # -------------------- optional baseline (Main) --------------------
    main_out = run_main(cfg, train_loader, val_loader, test_loader, device, verbose=True)

    # -------------------- SynIB-Learned --------------------
    synibL_model, synibL_hist = train_synib_learned(cfg, train_loader, device, val_loader=val_loader)
    synibL_stats = eval_clean(synibL_model, test_loader, device)
    synibL_ab = eval_block_ablations(synibL_model, test_loader, device, noise_std=1.0)
    synibL_by = eval_by_source(synibL_model, test_loader, device)
    print_by_source("SynIB-Learned", synibL_by, min_n=10)

    rep = {
        "cfg": asdict(cfg),
        "seed": 0,
        "split": split,
        "detectability_acc": det,
        "main": main_out["stats"],
        "synib": synibL_stats,                 # keep key name "synib" so print_row/plots work unchanged
        "ablations": synibL_ab,
        "history": {"main": main_out["history"], "synib": synibL_hist},
        "by_source": {"main": main_out["by_source"], "synib": synibL_by},
    }

    print_row(0, "Main", rep["main"])
    print_row(0, "SynIB-Learned", rep["synib"])
    print_sanity(rep)

    loss_path = _plot_loss_curves(rep, cfg.out_dir)
    acc_path  = _plot_acc_curves(rep, cfg.out_dir)
    print(f"[SAVED] {loss_path}")
    print(f"[SAVED] {acc_path}")

@dataclass
class Config:
    # Data sizes
    n_train: int = 2000
    n_test: int = 4000

    # Total feature dims per modality
    dim0: int = 32
    dim1: int = 32

    # Block fractions (per modality); remaining goes to noise.
    frac_unique: float = 0.20
    frac_red: float = 0.20
    frac_syn: float = 0.20

    # Block position strategy
    random_block_positions: bool = False

    # Latent dims for each signal type (projected into block dims)
    latent_u: int = 4
    latent_r: int = 4
    latent_s: int = 4

    # Signal strengths (SNR knobs)
    unique_strength: float = 3.0
    red_strength: float = 3.0
    syn_strength: float = 3.0
    noise_std: float = 1.0

    # Signal prob to be correlated with label
    signal_probs: Dict[str, float] = field(default_factory=lambda: {
        "none": 0.0, "u1": 0.0, "u2": 0.0, "red": 0.9, "syn": 0.1,
        # "red+syn": 0.10, "u1+u2": 0.05, "u1+red": 0.03, "u2+red": 0.02,
        "u1+u2+red+syn": 0.0,
    })

    # Training
    batch_size: int = 64
    epochs: int = 10
    lr: float = 3e-4
    weight_decay_synib: float = 1e-5
    hidden: int = 256

    # Loss weights
    lambda_uni: float = 1.0
    lambda_kl: float = 10.0

    # # Optional: discourage reliance on shortcuts by enforcing invariance to destroying them
    lambda_shortcut_inv: float = 0.0
    inv_destroy_unique: bool = False
    inv_destroy_red: bool = False

    K = 10
    p_min = 0.7
    p_max = 1.0
    cosine_s = 0.008

    # Device / outputs
    device: Optional[str] = None
    out_dir: str = "runs_refactor"

    learned_mask_method="fusion_more_than_unimodal"
    learned_mask_steps = 30
    learned_mask_lr = 5e-2
    learned_mask_tau = 1.0
    learned_mask_noise_std = 1.0
    learned_mask_lam_sparsity = 0.05
    learned_mask_alpha_unimodal = 1.0
    learned_mask_hard = True
    learned_mask_hard_thresh = 0.5

    random_mask_proportion = 0.3


if __name__ == "__main__":
    # main()
    # main_learned()
    # main_sweep_lambda_kl()

    # main_sweep_lambda_kl_learned()
    # main_sweep_lambda_kl_both()
    # main_sweep_lambda_kl_both_multiseed(seeds=[0, 1, 2], learned_mask_steps_default=5)

    # main_sweep_nonoverlap_probs_main(seeds=[0, 1, 2])
    # main_sweep_nonoverlap_probs_mainmask(seeds=[0, 1, 2])
    # main_sweep_nonoverlap_probs_synib(seeds=[0, 1, 2])
    # main_sweep_nonoverlap_probs_synib_random(seeds=[0, 1, 2])
    # main_sweep_nonoverlap_probs_synib_randomdiff(seeds=[0, 1, 2])
    main_sweep_nonoverlap_probs_synib_learned(seeds=[0, 1, 2])

    # main_sweep_nonoverlap_probs_tuned_kl(seeds=[0, 1, 2])
    # main_sweep_nonoverlap_probs()


# 1   - 1.0e+01 |   0.953 |   0.538 |     0.978 |     0.889
# 5   - 1.0e+01 |   0.951 |   0.510 |     0.984 |     0.869
# 10  - 1.0e+01 |   0.953 |   0.538 |     0.979 |     0.896
# 50  - 1.0e+01 |   0.951 |   0.510 |     0.987 |     0.881
# 500 - 1.0e+01 |   0.953 |   0.538 |     0.982 |     0.833

