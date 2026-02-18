
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
import matplotlib.pyplot as plt

import torch
import wandb
import random
import numpy as np
import json, os, time, socket

MASK_NOISE = 0
MASK_UNIQUE = 1
MASK_RED = 2
MASK_SYN = 3
# DESTROY_MASK = [MASK_NOISE, MASK_UNIQUE_1, MASK_RED, MASK_SYN]
DESTROY_MASK = [MASK_SYN]
INV_DESTROY_MASK = [MASK_RED, MASK_UNIQUE]
Method = Literal["kl_uniform_fusion", "flip_fusion", "fusion_more_than_unimodal"]
MaskMethod = Literal["kl_uniform_fusion", "flip_fusion", "fusion_more_than_unimodal", "unimodal", "kl_uniform_unimodal", "adv_unimodal"]
GateShape = Literal["global", "per_example"]
IoUTarget = Literal["syn", "uni"]


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

    def __init__(self, cfg, n: int, *, seed: int, split: str, train_stats: Optional[Dict[str, Any]] = None, verbose: bool = False):
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
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.enc1 = nn.Sequential(
            nn.Linear(dim1, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
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
            nn.Dropout(dropout),
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

    def forward_logits(self, x0: torch.Tensor, x1: torch.Tensor, not_detached=False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0, h1 = self.enc0(x0), self.enc1(x1)
        if not_detached:
            u0, u1 = self.score0(h0), self.score1(h1)  # detach unimodal heads
        else:
            u0, u1 = self.score0(h0.detach()), self.score1(h1.detach())  # detach unimodal heads
        f = self.fuse(torch.cat([h0, h1], dim=-1))
        return f, u0, u1




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


def plot_pid_ntk_history_beautiful(
    history: dict,
    smooth_window: int = 101,
    max_points: int = 4500,
    figsize=(15.2, 4.2),
    savepath: str = None,
    show: bool = False,
    band: str = "mad",
    band_alpha: float = 0.12,
    raw_alpha: float = 0.05,
    ylim_lambda=(0, 5000),
    ylim_cos=(-0.4, 0.4),
    ylim_loss=None,
    annotation: dict = None,  # <-- NEW
):
    rows = history.get("series", {}).get("steps", [])
    val_rows = history.get("val", {})
    if not rows:
        raise ValueError('history["series"]["steps"] is empty.')

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "axes.facecolor": "#FBFBFD",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    pid = ["U1", "Red", "Syn"]
    cos_keys = ["U1-Syn", "Red-Syn"]

    COLORS = {
        "U1": "#2563EB",
        "Red": "#16A34A",
        "Syn": "#DC2626",
        "U1-Syn": "#2563EB",
        "Red-Syn": "#F59E0B",
    }
    LABELS = {"U1": "U1", "Red": "Red", "Syn": "Syn", "U1-Syn": "U1–Syn", "Red-Syn": "Red–Syn"}

    x = np.asarray([r.get("step", i) for i, r in enumerate(rows)], dtype=int)
    lambdas = {k: np.asarray([r.get(f"lambda/{k}", np.nan) for r in rows], dtype=float) for k in pid}
    cosines = {k: np.asarray([r.get(f"cos/{k}", np.nan) for r in rows], dtype=float) for k in cos_keys}
    pidloss = {k: np.asarray([r.get(f"pidloss/{k}", np.nan) for r in rows], dtype=float) for k in pid}
    x_val = np.asarray([r.get("steps", i) for i, r in enumerate(val_rows)], dtype=int)
    pidloss_val = {k: np.asarray([r.get(f"val/perf_loss_{k}", np.nan) for r in val_rows], dtype=float) for k in pid}

    def _finite_any(a):
        a = np.asarray(a, dtype=float)
        return np.isfinite(a).any()

    def _downsample(x, series_dict, max_points):
        n = len(x)
        if max_points is None or n <= max_points:
            return x, series_dict
        idx = np.linspace(0, n - 1, max_points).astype(int)
        return x[idx], {k: v[idx] for k, v in series_dict.items()}

    def _rolling_stats(y, w, mode="mad"):
        y = np.asarray(y, dtype=float)
        n = len(y)
        if w is None or w <= 1 or n == 0:
            return y, None, None

        half = w // 2
        center = np.full(n, np.nan, dtype=float)
        spread = np.full(n, np.nan, dtype=float)

        min_support = max(5, w // 10)
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            win = y[lo:hi]
            win = win[np.isfinite(win)]
            if win.size < min_support:
                continue

            if mode == "std":
                c = float(np.mean(win))
                s = float(np.std(win))
            else:
                c = float(np.median(win))
                mad = float(np.median(np.abs(win - c)))
                s = 1.4826 * mad

            center[i] = c
            spread[i] = s

        return center, center - spread, center + spread

    def _style_axes(ax):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#111827")
        ax.spines["bottom"].set_color("#111827")
        ax.spines["left"].set_alpha(0.55)
        ax.spines["bottom"].set_alpha(0.55)
        ax.tick_params(axis="both", which="major", length=4, width=1, color="#111827")
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_color("#111827")
            label.set_alpha(0.85)
        ax.grid(True, axis="y", alpha=0.22, linewidth=0.9, color="#CBD5E1")
        ax.grid(False, axis="x")
        ax.set_axisbelow(True)

    def _plot(ax, x, raw, center, lo, hi, key):
        c = COLORS[key]
        if lo is not None and hi is not None and _finite_any(lo) and _finite_any(hi):
            ax.fill_between(x, lo, hi, color=c, alpha=band_alpha, linewidth=0)
        if raw_alpha and raw_alpha > 0 and _finite_any(raw):
            ax.plot(x, raw, color=c, linewidth=0.9, alpha=raw_alpha)
        if _finite_any(center):
            ax.plot(x, center, color=c, linewidth=2.6, label=LABELS[key])

    # downsample
    x, lambdas = _downsample(x, lambdas, max_points)
    _, cosines = _downsample(x, cosines, max_points)
    _, pidloss = _downsample(x, pidloss, max_points)
    # _, pidloss_val = _downsample(x, pidloss_val, max_points)

    # smooth
    lambdas_sb = {k: _rolling_stats(v, smooth_window, mode=band) for k, v in lambdas.items()}
    cosines_sb = {k: _rolling_stats(v, smooth_window, mode=band) for k, v in cosines.items()}
    pidloss_sb = {k: _rolling_stats(v, smooth_window, mode=band) for k, v in pidloss.items()}
    pidloss_val_sb = {k: _rolling_stats(v, 1, mode=band) for k, v in pidloss_val.items()}

    # for k, v in pidloss_sb.items():
    #     print(v.shape)
    # for k, v in pidloss_val.items():
    #     print(v.shape)
    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Panel 1: lambdas (y tick labels / 1000 + "x1000" note)
    ax = axes[0]
    for k in pid:
        center, lo, hi = lambdas_sb[k]
        _plot(ax, x, lambdas[k], center, lo, hi, k)
    ax.set_title(r"NTK strength $\lambda_g$", fontweight="semibold")
    ax.set_ylabel(r"$\lambda$")
    ax.set_ylim(*ylim_lambda)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=3.2)

    # rescale y tick labels by 1000 (display-only)
    yt = ax.get_yticks()
    ax.set_yticklabels([f"{t/1000:g}" for t in yt])
    ax.text(
        0.01, 1.0, r"$\times 10^3$",
        transform=ax.transAxes,
        ha="left", va="top",
        fontsize=10,
        color="#111827",
        alpha=0.75,
    )

    axes[0].text(
        0.3, 0.005, r"Optimization Steps",
        transform=axes[0].transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="#111827",
        alpha=0.75,
    )
    axes[1].text(
        0.3, 0.005, r"Optimization Steps",
        transform=axes[1].transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="#111827",
        alpha=0.75,
    )
    axes[2].text(
        0.3, 0.005, r"Optimization Steps",
        transform=axes[2].transAxes,
        ha="right", va="bottom",
        fontsize=10,
        color="#111827",
        alpha=0.75,
    )

    # Panel 2: cosines
    ax = axes[1]
    for k in cos_keys:
        center, lo, hi = cosines_sb[k]
        _plot(ax, x, cosines[k], center, lo, hi, k)
    ax.axhline(0.0, color="#111827", linewidth=1.0, alpha=0.55)
    ax.set_title("Update alignment", fontweight="semibold")
    ax.set_ylabel(r"$\cos(v_g, v_h)$")
    ax.set_ylim(*ylim_cos)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=3.2)

    # Panel 3: pid loss (train + val)
    ax = axes[2]
    plotted = False

    for k in pid:
        # ---- train (solid) ----
        center, lo, hi = pidloss_sb[k]
        if _finite_any(pidloss[k]) or _finite_any(center):
            _plot(ax, x, pidloss[k], center, lo, hi, k)
            plotted = True

        v_center, v_lo, v_hi = pidloss_val_sb[k]
        if _finite_any(pidloss_val[k]) or _finite_any(v_center):
            c = COLORS[k]
            ax.plot(x_val, v_center, color=c, linewidth=2.0, linestyle="dotted", alpha=0.95, label=f"{LABELS[k]} (val)")
            plotted = True

    ax.set_title("Fusion loss by PID source", fontweight="semibold")
    ax.set_ylabel("BCE")
    if ylim_loss is not None:
        ax.set_ylim(*ylim_loss)
    _style_axes(ax)

    if plotted:
        ax.legend(frameon=False, fontsize=8.5, loc="upper left", handlelength=3.2)

        # leg = ax.legend(
        #     frameon=True,  # Enable the frame
        #     loc="upper left",
        #     handlelength=3.2,
        #     fontsize=8.5,
        #     facecolor="gainsboro",  # Matches your text box color (the "tint")
        #     # edgecolor="#94A3B8",  # Matching border color
        #     framealpha=0.4,  # 0 is transparent, 1 is fully opaque
        # )
        # # leg.get_frame().set_linewidth(0.8)
        # # ax.legend(frameon=False, loc="upper left", handlelength=3.2)
    else:
        ax.text(0.5, 0.5, "No PID-loss data logged", ha="center", va="center",
                transform=ax.transAxes, color="#6B7280")

    # Consistent x-lims
    xmin, xmax = int(np.min(x)), int(np.max(x))
    for ax in axes:
        ax.set_xlim(xmin, xmax)

    # Put x-axis label on top of the right-most panel only
    axes[0].set_xlabel("")  # remove bottom xlabels
    axes[1].set_xlabel("")
    axes[2].set_xlabel("")
    # axes[2].set_xlabel("Optimization Steps", fontweight="semibold")
    axes[2].xaxis.set_label_position("top")
    axes[2].xaxis.tick_bottom()     # ticks stay at bottom
    axes[2].xaxis.set_ticks_position("bottom")

    # ---- stats box annotation ----
    if annotation:
        # ---- Build a neat, aligned block (monospace) ----
        p = annotation.get("data_pct_train", {})
        acc_total = annotation.get("acc_total", float("nan"))
        acc_syn = annotation.get("acc_syn", float("nan"))

        def pct(x):
            return f"{float(x):4.1f}%" if np.isfinite(x) else "  n/a "

        def num(x):
            return f"{float(x)*100:.1f}%" if np.isfinite(x) else "n/a"

        lines = [
            r"$\bf{Dataset\ composition}$",
            f"  U1  {pct(p.get('U1', np.nan))}  U2  {pct(p.get('U2', np.nan))}",
            f"  Red {pct(p.get('Red', np.nan))}  Syn {pct(p.get('Syn', np.nan))}",
            "",
            r"$\bf{Accuracy}$",
            f"  Total:   {num(acc_total)}",
            f"  Synergy: {num(acc_syn)}",
        ]

        axes[2].text(
            0.58, 0.35, "\n".join(lines),
            ha="left", va="center",
            transform=axes[2].transAxes,  # Ensures coordinates are relative to the subplot (0 to 1)
            fontsize=9.5,
            fontweight=1.4,
            fontfamily="DejaVu Sans Mono",  # key: column alignment
            color="darkslategray",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="gainsboro", edgecolor="#94A3B8", linewidth=0.8, alpha=0.96),
        )

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight", dpi=300)
        fig.savefig(savepath.replace("pdf","png"), bbox_inches="tight", dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axes





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
import copy
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@torch.no_grad()
def loss_per_source(f, y, source, eps=1e-12, steps=None):
    """
    f: logits, shape [B] or [B,1] (or [B,...] broadcastable with y)
    y: labels broadcastable to f (for BCEWithLogits)
    source: one-hot [B,4] for (U1,U2,Red,Syn)

    Returns a dict with:
      - loss/<name>: mean BCE loss over that subset (None if empty)
      - acc/<name>: accuracy over that subset (None if empty)
      - count/<name>: number of samples in subset
      - loss_total: mean BCE loss over full batch
      - acc_total: accuracy over full batch
    """
    names = ["U1", "U2", "Red", "Syn"]

    per_elem = F.binary_cross_entropy_with_logits(f, y, reduction="none")
    while per_elem.dim() > 1:
        per_elem = per_elem.mean(dim=-1)

    p = torch.sigmoid(f)
    while p.dim() > 1:
        p = p.mean(dim=-1)

    while y.dim() > 1:
        y = y.mean(dim=-1)

    y_bin = (y > 0.5)
    pred_bin = (p > 0.5)

    out: Dict[str, Any] = {}
    out["loss_total"] = float(per_elem.mean().item())
    out["acc_total"] = float((pred_bin == y_bin).float().mean().item())

    for g, name in enumerate(names):
        idx = source[:, g].bool()
        c = int(idx.sum().item())
        out[f"count/{name}"] = c
        if c == 0:
            out[f"loss/{name}"] = None
            out[f"acc/{name}"] = None
        else:
            out[f"loss/{name}"] = float(per_elem[idx].mean().item())
            out[f"acc/{name}"] = float((pred_bin[idx] == y_bin[idx]).float().mean().item())

    return out


def train_main(
    cfg,
    train_loader: DataLoader,
    device: str,
    val_loader: Optional[DataLoader] = None,
) -> Tuple["FusionModel", Dict[str, Any]]:
    """
    Batch-level logging (one row per batch) for plots:
      1) lambda_g evolution over steps
      2) cosine evolution: U1-Syn, Red-Syn over steps
      3) PID-decomposed fusion loss over steps

    Returns:
      model_best, history with history["series"]["steps"] for plotting.
    """
    model = FusionModel(cfg.dim0, cfg.dim1, cfg.hidden, dropout=cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    pid_names = ["U1", "U2", "Red", "Syn"]
    cosine_keys = ["U1-Syn", "Red-Syn"]

    history: Dict[str, Any] = {
        "train": [],  # epoch-level metrics (optional)
        "val": [],
        "best_epoch": None,
        "best_val_fusion_loss": float("inf"),
        "best_val_fusion_acc": 0.0,
        "best_val_fusion_syn_acc": 0.0,
        "series": {"steps": []},  # batch-level rows
    }

    best_state = None
    opt_steps = 0

    def _to_float_or_nan(x):
        try:
            if x is None:
                return float("nan")
            if hasattr(x, "item"):
                x = x.item()
            x = float(x)
            return x if np.isfinite(x) else float("nan")
        except Exception:
            return float("nan")

    for epoch in range(cfg.epochs):
        model.train()

        # epoch-level bookkeeping (optional, kept for tables)
        n = 0
        sum_ltot = sum_lf = sum_lu0 = sum_lu1 = 0.0
        sum_acc_f = sum_acc_u0 = sum_acc_u1 = 0.0

        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y = b["y"].to(device)
            source = b["source"].to(device)  # one-hot [B,4]

            f, u0, u1 = model.forward_logits(x0, x1)

            lf = F.binary_cross_entropy_with_logits(f, y)
            lu0 = F.binary_cross_entropy_with_logits(u0, y)
            lu1 = F.binary_cross_entropy_with_logits(u1, y)
            ltot = lf + float(cfg.lambda_uni) * (lu0 + lu1)

            opt.zero_grad(set_to_none=True)
            ltot.backward(retain_graph=True)

            # ---- NTK diagnostics ----
            lambdas, vJt, stats = ntk_strengths_onehot_source_debug(
                model, f, y, source, print_debug=False, steps=opt_steps
            )

            # ---- PID loss decomposition (fusion loss per PID group) ----
            lps = loss_per_source(f, y, source, steps=opt_steps)

            opt.step()

            # ---- epoch bookkeeping ----
            bs = int(y.size(0))
            n += bs
            sum_ltot += float(ltot.item()) * bs
            sum_lf += float(lf.item()) * bs
            sum_lu0 += float(lu0.item()) * bs
            sum_lu1 += float(lu1.item()) * bs
            sum_acc_f += _acc_from_logits(f, y) * bs
            sum_acc_u0 += _acc_from_logits(u0, y) * bs
            sum_acc_u1 += _acc_from_logits(u1, y) * bs

            # ---- batch-level row for plotting ----
            row: Dict[str, Any] = {"step": int(opt_steps), "epoch": int(epoch), "batch_size": bs}

            # lambdas
            for k in pid_names:
                v = _to_float_or_nan(lambdas.get(k, None))
                if np.isfinite(v):
                    row[f"lambda/{k}"] = v  # skip if non-finite

            # cosines
            cos_map = stats.get("cosines", {}) if isinstance(stats, dict) else {}
            for ck in cosine_keys:
                v = _to_float_or_nan(cos_map.get(ck, None))
                if np.isfinite(v):
                    row[f"cos/{ck}"] = v  # skip if non-finite

            # PID loss decomposition: ignore empty groups (do NOT store NaN)
            if isinstance(lps, dict):
                for k in pid_names:
                    c = lps.get(f"count/{k}", None)
                    if c is not None and int(c) == 0:
                        continue
                    lv = _to_float_or_nan(lps.get(f"loss/{k}", None))
                    if np.isfinite(lv):
                        row[f"pidloss/{k}"] = lv

            history["series"]["steps"].append(row)
            opt_steps += 1

        # epoch-level metrics (optional)
        history["train"].append({
            "epoch": epoch,
            "loss_total": sum_ltot / max(1, n),
            "loss_fusion": sum_lf / max(1, n),
            "loss_uni0": sum_lu0 / max(1, n),
            "loss_uni1": sum_lu1 / max(1, n),
            "acc_fusion": sum_acc_f / max(1, n),
            "acc_uni0": sum_acc_u0 / max(1, n),
            "acc_uni1": sum_acc_u1 / max(1, n),
        })

        # -------------------- val epoch + checkpoint --------------------
        if val_loader is not None:
            val_metrics = _eval_epoch_main(model, val_loader, device, cfg.lambda_uni)
            val_metrics["epoch"] = epoch
            val_metrics["steps"] = opt_steps
            history["val"].append(val_metrics)

            val_src = eval_loss_per_source(model, val_loader, device)
            val_acc = eval_by_source(model, val_loader, device)
            val_metrics.update(val_src)
            val_metrics.update(val_acc)

            # selection logic
            if cfg.val_method == "val_loss":
                cur = float(val_metrics["loss_fusion"])
                check_with = history["best_val_fusion_loss"]
                better = cur < check_with
            elif cfg.val_method == "val_acc":
                cur = float(val_metrics["acc_fusion"])
                check_with = history["best_val_fusion_acc"]
                better = cur > check_with
            elif cfg.val_method == "val_syn_acc":
                if "syn" in val_metrics.get("by_source", {}):
                    cur = float(val_metrics["by_source"]["syn"]["acc"])
                    check_with = history["best_val_fusion_syn_acc"]
                    better = cur > check_with
                else:
                    cur = float(val_metrics["acc_fusion"])
                    check_with = history["best_val_fusion_acc"]
                    better = cur > check_with
            else:
                better = False

            if better:
                history["best_val_fusion_acc"] = float(val_metrics.get("acc_fusion", np.nan))
                history["best_val_fusion_syn_acc"] = (
                    float(val_metrics["by_source"]["syn"]["acc"])
                    if "by_source" in val_metrics and "syn" in val_metrics["by_source"]
                    else float("nan")
                )
                history["best_val_fusion_loss"] = float(val_metrics.get("loss_fusion", np.nan))
                history["best_epoch"] = epoch
                best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history

def train_synib_mstar( cfg, train_loader: DataLoader, device: str, val_loader: Optional[DataLoader] = None,) -> Tuple["FusionModel", Dict[str, Any]]:

    class _Meter:
        __slots__ = ("s", "n")

        def __init__(self) -> None:
            self.s = 0.0
            self.n = 0

        def add(self, v: float, k: int) -> None:
            self.s += float(v) * int(k)
            self.n += int(k)

        def mean(self) -> float:
            return self.s / max(1, self.n)

    def _make_meters(names):
        return defaultdict(_Meter, {name: _Meter() for name in names})

    def _run_one_kl(kl: float):
        # clone cfg so caller cfg isn't mutated
        cfg_run = copy.deepcopy(cfg)
        cfg_run.lambda_kl = float(kl)

        model = FusionModel(cfg_run.dim0, cfg_run.dim1, cfg_run.hidden, dropout=cfg_run.dropout).to(device)
        opt = optim.Adam(model.parameters(), lr=cfg_run.lr, weight_decay=cfg_run.weight_decay)

        history = {
            "train": [],
            "val": [],
            "best_epoch": None,
            "best_val_fusion_loss": float("inf"),
            "best_val_fusion_acc": 0.0,
            "best_val_fusion_syn_acc": 0.0,
            # additions (non-breaking)
            "kl_val": float(kl),
            "aggregate_masks": defaultdict(list),
        }

        best_state = None
        opt_steps = 0

        for epoch in range(cfg_run.epochs):
            model.train()

            meters = _make_meters(
                [
                    "loss_total",
                    "loss_fusion",
                    "loss_uni0",
                    "loss_uni1",
                    "loss_cf",
                    "acc_fusion",
                    "acc_uni0",
                    "acc_uni1",
                    "iou_syn0",
                    "iou_syn1",
                    "iou_unired0",
                    "iou_unired1",
                ]
            )

            for b in train_loader:
                x0 = b["x0"].to(device)
                x1 = b["x1"].to(device)
                y = b["y"].to(device)
                m0, m1 = b["mask0"].to(device), b["mask1"].to(device)
                source = b["source"].to(device)
                bs = y.size(0)

                f, u0, u1 = model.forward_logits(x0, x1)

                lf = F.binary_cross_entropy_with_logits(f, y)
                lu0 = F.binary_cross_entropy_with_logits(u0, y)
                lu1 = F.binary_cross_entropy_with_logits(u1, y)

                if cfg.train_method == "synib_mstar":
                    # ---- your KL counterfactual term (exact lines you gave) ----
                    x0_t = destroy_block(x0, m0, DESTROY_MASK, noise_std=1.0)
                    x1_t = destroy_block(x1, m1, DESTROY_MASK, noise_std=1.0)
                elif cfg.train_method == "synib_mrand":
                    m0_random = torch.rand_like(x0)
                    m1_random = torch.rand_like(x1)
                    m0_random = (m0_random < cfg.random_mask_proportion).float()
                    m1_random = (m1_random < cfg.random_mask_proportion).float()
                    x0_t = destroy_block(x0, m0_random, 1, noise_std=1.0)
                    x1_t = destroy_block(x1, m1_random, 1, noise_std=1.0)
                elif cfg.train_method == "synib_mlearned":

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
                            label=f"ep{epoch}",
                        )

                        g0, g1 = masks["g0"].detach(), masks["g1"].detach()
                        x0_t = destroy_block(x0, g0, 1, noise_std=1.0)
                        x1_t = destroy_block(x1, g1, 1, noise_std=1.0)

                        pred0 = gate_to_hard_mask(g0, 0.5, m0.shape)
                        pred1 = gate_to_hard_mask(g1, 0.5, m1.shape)

                        gt_syn0 = (m0 == MASK_SYN)
                        gt_syn1 = (m1 == MASK_SYN)
                        gt_uni0 = (m0 == MASK_UNIQUE) | (m0 == MASK_RED)
                        gt_uni1 = (m1 == MASK_UNIQUE) | (m1 == MASK_RED)

                        meters["iou_syn0"].add(_overlap_binary(pred0, gt_syn0), bs)
                        meters["iou_syn1"].add(_overlap_binary(pred1, gt_syn1), bs)
                        meters["iou_unired0"].add(_overlap_binary(pred0, gt_uni0), bs)
                        meters["iou_unired1"].add(_overlap_binary(pred1, gt_uni1), bs)


                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                l_cf = bern_kl_to_uniform_from_logits(f_t0) + bern_kl_to_uniform_from_logits(f_t1)
                # ---------------------------------------------

                ltot = (
                    lf
                    + float(cfg_run.lambda_uni) * (lu0 + lu1)
                    + float(cfg_run.lambda_kl) * l_cf
                )

                opt.zero_grad(set_to_none=True)
                ltot.backward()
                opt.step()

                bs = int(y.size(0))
                meters["loss_total"].add(ltot.item(), bs)
                meters["loss_fusion"].add(lf.item(), bs)
                meters["loss_uni0"].add(lu0.item(), bs)
                meters["loss_uni1"].add(lu1.item(), bs)
                meters["loss_cf"].add(l_cf.item(), bs)

                meters["acc_fusion"].add(_acc_from_logits(f, y), bs)
                meters["acc_uni0"].add(_acc_from_logits(u0, y), bs)
                meters["acc_uni1"].add(_acc_from_logits(u1, y), bs)

                opt_steps += 1

            out = { "epoch": epoch,
                    "loss_total": meters["loss_total"].mean(),
                    "loss_fusion": meters["loss_fusion"].mean(),
                    "loss_uni0": meters["loss_uni0"].mean(),
                    "loss_uni1": meters["loss_uni1"].mean(),
                    # addition (harmless)
                    "loss_cf": meters["loss_cf"].mean(),
                    "acc_fusion": meters["acc_fusion"].mean(),
                    "acc_uni0": meters["acc_uni0"].mean(),
                    "acc_uni1": meters["acc_uni1"].mean(),
                }
            if cfg.train_method == "synib_mlearned":
                out["iou_syn0"] = meters["iou_syn0"].mean()
                out["iou_syn1"] = meters["iou_syn1"].mean()
                out["iou_unired0"] = meters["iou_unired0"].mean()
                out["iou_unired1"] = meters["iou_unired1"].mean()
            history["train"].append( out )

            if val_loader is not None:
                val_metrics = _eval_epoch_main(model, val_loader, device, cfg_run.lambda_uni)
                val_metrics["epoch"] = epoch

                # keep your per-source epoch-level diagnostics merged into the same dict
                val_metrics.update(eval_loss_per_source(model, val_loader, device))
                val_metrics.update(eval_by_source(model, val_loader, device))

                history["val"].append(val_metrics)
                try:
                    if wandb.run is None:
                        wandb.init(project="synergy-ntk", reinit=False)
                except Exception:
                    pass

                # ---- your selection logic (as in your code) ----
                if cfg_run.val_method == "val_loss":
                    cur = float(val_metrics["loss_fusion"])
                    check_with = history["best_val_fusion_loss"]
                elif cfg_run.val_method == "val_acc":
                    cur = float(val_metrics["acc_fusion"])
                    check_with = history["best_val_fusion_acc"]
                elif cfg_run.val_method == "val_syn_acc":
                    if "syn" not in val_metrics["by_source"]:
                        cur = float(val_metrics["acc_fusion"])
                        check_with = history["best_val_fusion_acc"]
                    else:
                        cur = float(val_metrics["by_source"]["syn"]["acc"])
                        check_with = history["best_val_fusion_syn_acc"]
                else:
                    cur = float(val_metrics["by_source"]["syn"]["acc"])
                    check_with = history["best_val_fusion_syn_acc"]

                if cur > check_with:
                    history["best_val_fusion_acc"] = float(val_metrics["acc_fusion"])
                    history["best_val_fusion_syn_acc"] = float(val_metrics["by_source"]["syn"]["acc"]) if "syn" in val_metrics["by_source"] else np.nan
                    history["best_val_fusion_loss"] = float(val_metrics["loss_fusion"])
                    history["best_epoch"] = epoch
                    best_state = copy.deepcopy(model.state_dict())
                # ---------------------------------------------

        if best_state is not None:
            model.load_state_dict(best_state)

        # score for choosing best KL across runs
        if cfg_run.val_method == "val_loss":
            score = float(history["best_val_fusion_loss"])
        elif cfg_run.val_method == "val_acc":
            score = float(history["best_val_fusion_acc"])
        else:
            if history["best_val_fusion_syn_acc"] is None:
                score = float(history["best_val_fusion_acc"])
            else:
                score = float(history["best_val_fusion_syn_acc"])

        return model, history, score

    kl_vals = cfg.kl_vals
    lsp_vals = cfg.lsp_vals
    if lsp_vals is None or len(lsp_vals) == 0:
        lsp_vals = [0.0]
    best_model = None
    best_history = None
    best_score = float("-inf")
    best_kl = None

    kl_runs: Dict[str, Any] = {}

    for lsp in lsp_vals:
        for kl in kl_vals:
            model_kl, hist_kl, score_kl = _run_one_kl(float(kl))
            kl_runs[str(float(kl))+"-"+str(float(lsp))] = hist_kl

            if score_kl > best_score:
                best_score = score_kl
                best_model = model_kl
                best_history = hist_kl
                best_kl = float(kl)

    # top-level history mirrors the chosen run, with added sweep logs
    out_history = dict(best_history) if best_history is not None else {"train": [], "val": []}
    out_history["kl_vals"] = [float(x) for x in kl_vals]
    out_history["best_kl"] = best_kl
    out_history["kl_runs"] = kl_runs

    return best_model, out_history


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
def acc_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    p = (torch.sigmoid(logits) > 0.5).float()
    return float((p == y).float().mean().item())
def apply_destroy(x: torch.Tensor, g: torch.Tensor, noise_std: float = 1.0, inv_mask=False) -> torch.Tensor:
    if g.dim() == 1: g = g.view(1, -1)
    g = g.to(x.device).type_as(x)
    eps = torch.randn_like(x) * float(noise_std)
    if inv_mask:
        return g * x + (1 - g) * eps
    return (1 - g) * x + g * eps
def oracle_mask_from_block(mask_block: Optional[torch.Tensor], target: IoUTarget) -> Optional[torch.Tensor]:
    if mask_block is None: return None
    if target == "syn": return (mask_block == MASK_SYN)
    # "uni": unique OR red
    return (mask_block == MASK_UNIQUE) | (mask_block == MASK_RED)
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

    ell = torch.nn.Parameter(init_gate_logits(B, d, gate_shape, device), requires_grad=True)
    opt = torch.optim.Adam([ell], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(steps), eta_min=lr * 0.1)

    gt = oracle_mask_from_block(mask_block.to(device) if mask_block is not None else None, iou_target)
    gt_unires = oracle_mask_from_block(mask_block.to(device) if mask_block is not None else None, "rest")
    prev, stall = None, 0

    aggregate_metrics = defaultdict(list)
    for t in range(int(steps)):
        g = gate_probs_from_logits(ell, tau, gate_shape, B, d)
        p_f_t, p_u_t = forward_destroyed_probs(model, x0, x1, which, g, noise_std, inv_mask=True)
        sparsity = g.mean()
        obj = mask_objective(method, which, p_f_clean, p_u0_clean, p_u1_clean, p_f_t, p_u_t, y, sparsity, float(lam_sparsity), float(alpha_unimodal))
        opt.zero_grad(set_to_none=True)
        obj.backward()

        torch.nn.utils.clip_grad_norm_([ell], 1.0)
        opt.step(); sched.step()

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

            acc_u_gt = acc_from_logits(p_u_gt, y)
            acc_u_t = acc_from_logits(p_u_t, y)
            acc_u_clean = acc_from_logits(p_f_clean, y)

            aggregate_metrics["iou"].append(iou)
            aggregate_metrics["iou_unires"].append(iou_unires)
            aggregate_metrics["sp"].append(sp)
            aggregate_metrics["obj"].append(obj)
            aggregate_metrics["obj_gt"].append(obj_gt)
            aggregate_metrics["acc_u_t"].append(acc_u_t)
            aggregate_metrics["acc_u_gt"].append(acc_u_gt)
            aggregate_metrics["acc_u_clean"].append(acc_u_clean)


        with torch.no_grad():
            now = torch.sigmoid(ell / float(tau))
            if prev is not None:
                dg = (now - prev).abs().mean().item()
                stall = stall + 1 if dg < 1e-4 else 0
                if stall >= 10: break
            prev = now.clone()

    # aggregate_metrics = {k: torch.tensor(v).mean().item() for k, v in aggregate_metrics.items()}
    # print(f"        [mask]{tag}{which}/{method} obj={aggregate_metrics['obj']:.3f} obj_gt={aggregate_metrics['obj_gt']:.3f} overlap({iou_target})={aggregate_metrics['iou']:.3f} overlap(uni/res)={aggregate_metrics['iou_unires']:.3f} sparsity={aggregate_metrics['sp']:.3f} acc_g={aggregate_metrics['acc_u_t']:.3f} acc_gt={aggregate_metrics['acc_u_gt']:.3f} acc_clean={aggregate_metrics['acc_u_clean']:.3f}")

    g_final = torch.sigmoid(ell / float(tau)).detach()
    if hard: g_final = (g_final > float(hard_thresh)).float()
    return g_final, aggregate_metrics
def learn_destroy_gates(model: nn.Module, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor, method: MaskMethod, device: str, gate_shape: GateShape, steps: int, lr: float, tau: float, noise_std: float, lam_sparsity: float, alpha_unimodal: float, hard: bool, hard_thresh: float, m0: Optional[torch.Tensor] = None, m1: Optional[torch.Tensor] = None, iou_target: IoUTarget = "syn", print_every: int = 10, label: str = "") -> Dict[str, torch.Tensor]:
    # main method masks
    g0, aggr_metrics_0 = learn_one_gate(model, x0, x1, y, "x0", method, device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m0, iou_target, print_every, label)
    g1, aggr_metrics_1 = learn_one_gate(model, x0, x1, y, "x1", method, device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m1, iou_target, print_every, label)
    # unimodal-ablation masks (used for your counterfactual KL)
    # g0_uni = learn_one_gate(model, x0, x1, "x0", "unimodal", device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m0, "uni", print_every, label)
    # g1_uni = learn_one_gate(model, x0, x1, "x1", "unimodal", device, gate_shape, steps, lr, tau, noise_std, lam_sparsity, alpha_unimodal, hard, hard_thresh, m1, "uni", print_every, label)
    return {"g0": g0, "g1": g1, "aggr_metrics":{"g0": aggr_metrics_0, "g1": aggr_metrics_1}}

@torch.no_grad()
def source_marginals_from_loader(loader, device=None):
    """
    Returns marginal percentages of each source being active in the dataset behind `loader`.
    Works with multi-hot `b["source"]` of shape [B,4] in order [u1,u2,red,syn].
    """
    counts = torch.zeros(4, dtype=torch.float64)
    n = 0
    for b in loader:
        src = b["source"]
        if device is not None:
            src = src.to(device)
        counts += src.double().sum(dim=0)
        n += int(src.size(0))

    if n == 0:
        return {"U1": float("nan"), "U2": float("nan"), "Red": float("nan"), "Syn": float("nan")}

    pct = 100.0 * (counts / float(n))
    return {"U1": float(pct[0].item()), "U2": float(pct[1].item()),
            "Red": float(pct[2].item()), "Syn": float(pct[3].item())}


def run_main(cfg: Config, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader, device: str, *, verbose: bool = True):

    if not hasattr(cfg, "train_method"): cfg.train_method = "main"

    if cfg.train_method == "main":
        model, hist = train_main(cfg, train_loader, device, val_loader=val_loader)
    else:
        model, hist = train_synib_mstar(cfg, train_loader, device, val_loader=val_loader)


    main_stats = eval_clean(model, test_loader, device)
    main_by = eval_by_source(model, test_loader, device)

    data_pct_train = source_marginals_from_loader(train_loader)
    data_pct_val = source_marginals_from_loader(val_loader)
    data_pct_test = source_marginals_from_loader(test_loader)

    annotation = {
        "data_pct_train": data_pct_train,
        "data_pct_val": data_pct_val,
        "data_pct_test": data_pct_test,
        "acc_total": float(main_stats["acc_fusion"]),
        "acc_syn": float(_syn_acc_from_by_source_any(main_by)),
    }

    plot_pid_ntk_history_beautiful(
        hist,
        smooth_window=101,
        band="mad",
        savepath="pid_ntk_batches_beautiful.pdf",
        annotation=annotation,  # <-- NEW
    )


    if verbose:
        print_by_source("Main", main_by, min_n=10)
    return {
        "model": model,
        "test_stats": main_stats,
        "history": hist,
        "test_by_source": main_by,
    }

def ntk_strengths_onehot_source_debug(
    model, f, y, source_onehot,
    eps=1e-12,
    denom_min=1e-10,
    max_norm_sq=1e30,
    print_debug=False,
        steps=0
):
    names = ["U1", "U2", "Red", "Syn"]

    with torch.no_grad():
        p = torch.sigmoid(f)
        s = (p - y).detach()

    B = source_onehot.size(0)
    assert source_onehot.shape == (B, 4), (source_onehot.shape, B)

    params = [p for p in model.parameters() if p.requires_grad]

    lambdas, vJt, stats = {}, {}, {}

    for g, name in enumerate(names):
        idx = source_onehot[:, g].bool()
        count = int(idx.sum().item())

        if count == 0:
            lambdas[name] = torch.tensor(float("nan"), device=f.device)
            vJt[name] = None
            stats[name] = {"count": 0, "denom": float("nan"), "num": float("nan"), "status": "empty"}
            continue

        mask_g = idx.float()
        while mask_g.dim() < s.dim():
            mask_g = mask_g.unsqueeze(-1)

        s_g = s * mask_g
        denom_raw = s_g.pow(2).sum()

        if denom_raw.item() < denom_min:
            lambdas[name] = torch.tensor(float("nan"), device=f.device)
            vJt[name] = None
            stats[name] = {
                "count": count,
                "denom": float(denom_raw.item()),
                "num": float("nan"),
                "status": "no_signal_denom_tiny",
            }
            continue

        denom = denom_raw.clamp_min(eps)

        Lg = (s_g * f).sum()

        grads = torch.autograd.grad(
            Lg, params,
            retain_graph=True,
            create_graph=False,
            allow_unused=True
        )

        flat = []
        any_nan = False
        any_inf = False
        for gi in grads:
            if gi is None:
                continue
            any_nan |= torch.isnan(gi).any().item()
            any_inf |= torch.isinf(gi).any().item()
            flat.append(gi.reshape(-1))

        v = torch.cat(flat) if flat else torch.zeros(0, device=f.device)
        num = v.pow(2).sum()

        status = "ok"
        if any_nan or torch.isnan(num):
            status = "grad_nan"
        if any_inf or torch.isinf(num):
            status = "grad_inf"
        if num.item() > max_norm_sq:
            status = "grad_huge"

        if status != "ok":
            lambdas[name] = torch.tensor(float("nan"), device=f.device)
            vJt[name] = None
        else:
            lambdas[name] = (num / denom).detach()
            vJt[name] = v.detach()

        stats[name] = {
            "count": count,
            "denom": float(denom_raw.item()),
            "num": float(num.item()) if status == "ok" else float("nan"),
            "status": status,
        }

    if print_debug:
        for k in ["Red", "Syn"]:
            if k in stats:
                print(k, stats[k], "lambda", lambdas[k])

    # cosines
    cosines = {}

    def _is_finite_vec(v):
        return (v is not None) and v.numel() > 0 and torch.isfinite(v).all() and (v.norm() > 0)

    def _cos(a, b, eps=1e-12):
        return ((a @ b) / (a.norm() * b.norm() + eps)).detach()

    pairs = [("Red", "Syn"), ("U1", "Red"), ("U2", "Red"), ("U1", "Syn"), ("U2", "Syn")]
    for a, b in pairs:
        if a in vJt and b in vJt and _is_finite_vec(vJt[a]) and _is_finite_vec(vJt[b]):
            cosines[f"{a}-{b}"] = _cos(vJt[a], vJt[b])
        else:
            cosines[f"{a}-{b}"] = torch.tensor(float("nan"), device=f.device)

    stats["cosines"] = {k: float(v.item()) for k, v in cosines.items()}

    # wandb logging (optional, side-effect free)
    # try:
    #     if wandb.run is None:
    #         wandb.init(project="synergy-ntk", reinit=False)
    #     wandb.log({
    #         "ntk/lambda_red": float(lambdas["Red"].item()) if torch.isfinite(lambdas["Red"]) else float("nan"),
    #         "ntk/lambda_syn": float(lambdas["Syn"].item()) if torch.isfinite(lambdas["Syn"]) else float("nan"),
    #         "ntk/lambda_u1": float(lambdas["U1"].item()) if torch.isfinite(lambdas["U1"]) else float("nan"),
    #         "ntk/lambda_u2": float(lambdas["U2"].item()) if torch.isfinite(lambdas["U2"]) else float("nan"),
    #         "ntk/num_red": float(stats["Red"]["num"]),
    #         "ntk/num_syn": float(stats["Syn"]["num"]),
    #         "ntk/num_u1": float(stats["U1"]["num"]),
    #         "ntk/num_u2": float(stats["U2"]["num"]),
    #         "ntk/ratio_syn_over_red": float((lambdas["Syn"] / (lambdas["Red"] + 1e-12)).item())
    #         if torch.isfinite(lambdas["Syn"]) and torch.isfinite(lambdas["Red"]) else float("nan"),
    #         "ntk/count_red": stats.get("Red", {}).get("count", float("nan")),
    #         "ntk/count_u1": stats.get("U1", {}).get("count", float("nan")),
    #         "ntk/count_u2": stats.get("U2", {}).get("count", float("nan")),
    #         "ntk/count_syn": stats.get("Syn", {}).get("count", float("nan")),
    #         "ntk/cos_red_syn": float(stats["cosines"].get("Red-Syn", float("nan"))),
    #         "ntk/cos_u1_syn": float(stats["cosines"].get("U1-Syn", float("nan"))),
    #         "ntk/cos_u2_syn": float(stats["cosines"].get("U2-Syn", float("nan"))),
    #     }, step=steps)
    # except Exception:
    #     pass

    return lambdas, vJt, stats

# @torch.no_grad()
# def loss_per_source(f, y, source, eps=1e-12, steps=None):
#     """
#     f: logits, shape [B] or [B,1] (or [B,...] broadcastable with y)
#     y: labels broadcastable to f (for BCEWithLogits)
#     source: one-hot [B,4] for (U1,U2,Red,Syn)
#
#     Returns a dict with:
#       - loss/<name>: mean BCE loss over that subset (NaN if empty)
#       - acc/<name>: accuracy over that subset (NaN if empty)  (binary threshold 0.5)
#       - count/<name>: number of samples in subset
#       - loss_total: mean BCE loss over full batch
#       - acc_total: accuracy over full batch
#     """
#     names = ["U1", "U2", "Red", "Syn"]
#
#     # BCE loss per element, then reduce per-sample (keep batch dimension)
#     # This supports f being [B] or [B,1] or [B, ...] (batch first).
#     per_elem = F.binary_cross_entropy_with_logits(f, y, reduction="none")
#
#     # reduce all non-batch dims to get per-sample losses: [B]
#     while per_elem.dim() > 1:
#         per_elem = per_elem.mean(dim=-1)
#
#     # predictions for accuracy (binary)
#     p = torch.sigmoid(f)
#     # reduce non-batch dims for prediction similarly
#     while p.dim() > 1:
#         p = p.mean(dim=-1)
#     while y.dim() > 1:
#         y_ = y.mean(dim=-1)
#         y = y_
#
#     y_bin = (y > 0.5)
#     pred_bin = (p > 0.5)
#
#     out = {}
#     out["loss_total"] = float(per_elem.mean().item())
#     out["acc_total"] = float((pred_bin == y_bin).float().mean().item())
#
#     for g, name in enumerate(names):
#         idx = source[:, g].bool()
#         c = int(idx.sum().item())
#         out[f"count/{name}"] = c
#         if c == 0:
#             out[f"loss/{name}"] = float("nan")
#             out[f"acc/{name}"] = float("nan")
#         else:
#             out[f"loss/{name}"] = float(per_elem[idx].mean().item())
#             out[f"acc/{name}"] = float((pred_bin[idx] == y_bin[idx]).float().mean().item())
#
#     try:
#         if wandb.run is None:
#             wandb.init(project="synergy-ntk", reinit=False)
#     except Exception:
#         pass
#     return out

def _last_finite_from_series(rows, key):
    for r in reversed(rows):
        v = r.get(key, None)
        if v is None:
            continue
        try:
            v = float(v)
        except Exception:
            continue
        if np.isfinite(v):
            return v
    return float("nan")

def _syn_acc_from_by_source_any(by):
    # weighted accuracy over all groups that include 'syn' in the key
    num, den = 0.0, 0
    for k, v in by.get("by_source", {}).items():
        parts = k.split("+")
        if "syn" in parts:
            n = int(v.get("n", 0))
            num += float(v.get("acc", 0.0)) * n
            den += n
    return (num / den) if den > 0 else float("nan")

@torch.no_grad()
def eval_loss_per_source(model, loader, device, max_batches=None):
    """
    Returns epoch-level (val) per-source losses/accs + counts.
    Uses b["source"] one-hot [B,4] and b["y"] labels.
    Assumes model.forward_logits(x0,x1) returns fusion logits f.
    """
    names = ["U1", "U2", "Red", "Syn"]

    sum_loss_total = 0.0
    sum_acc_total = 0.0
    n_total = 0

    sum_loss = {n: 0.0 for n in names}
    sum_acc  = {n: 0.0 for n in names}
    count    = {n: 0   for n in names}

    for bi, b in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break

        x0 = b["x0"].to(device)
        x1 = b["x1"].to(device)
        y  = b["y"].to(device).float()
        src = b["source"].to(device).float()  # [B,4]

        f, _, _ = model.forward_logits(x0, x1)

        per_elem = F.binary_cross_entropy_with_logits(f, y, reduction="none")
        while per_elem.dim() > 1:
            per_elem = per_elem.mean(dim=-1)  # [B]

        p = torch.sigmoid(f)
        while p.dim() > 1:
            p = p.mean(dim=-1)
        yy = y
        while yy.dim() > 1:
            yy = yy.mean(dim=-1)

        y_bin = (yy > 0.5)
        pred_bin = (p > 0.5)

        B = per_elem.size(0)
        sum_loss_total += float(per_elem.sum().item())
        sum_acc_total  += float((pred_bin == y_bin).float().sum().item())
        n_total += int(B)

        for gi, name in enumerate(names):
            idx = src[:, gi].bool()
            c = int(idx.sum().item())
            if c == 0:
                continue
            count[name] += c
            sum_loss[name] += float(per_elem[idx].sum().item())
            sum_acc[name]  += float((pred_bin[idx] == y_bin[idx]).float().sum().item())

    out = {
        "val/perf_loss_total": sum_loss_total / max(n_total, 1),
        "val/perf_acc_total":  sum_acc_total  / max(n_total, 1),
    }

    for name in names:
        out[f"val/perf_count_{name}"] = int(count[name])
        if count[name] == 0:
            out[f"val/perf_loss_{name}"] = float("nan")
            out[f"val/perf_acc_{name}"]  = float("nan")
        else:
            out[f"val/perf_loss_{name}"] = sum_loss[name] / count[name]
            out[f"val/perf_acc_{name}"]  = sum_acc[name]  / count[name]

    return out

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
# def _load_results_json(path: str) -> Dict[str, Any]:
#     if not os.path.exists(path):
#         return {"schema": "sweep_nonoverlap_probs_v2", "results": {}}
#     with open(path, "r") as f:
#         return json.load(f)
# def _save_results_json(path: str, obj: Dict[str, Any]) -> None:
#     tmp = path + ".tmp"
#     with open(tmp, "w") as f:
#         json.dump(obj, f, indent=2, sort_keys=True)
#     os.replace(tmp, path)

def _acquire_lock(lock_path: str, timeout: float = 120.0, poll: float = 0.05, stale_seconds: float = 600.0) -> str:
    token = str(os.getpid()) + "_" + str(int(time.time() * 1e6))
    t0 = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            try:
                os.write(fd, token.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)
            return token
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(lock_path) > stale_seconds:
                    try:
                        os.remove(lock_path)
                    except FileNotFoundError:
                        pass
                    continue
            except FileNotFoundError:
                continue
            if time.time() - t0 > timeout:
                raise TimeoutError("Could not acquire lock: " + lock_path)
            time.sleep(poll)

def _release_lock(lock_path: str, token: str) -> None:
    try:
        with open(lock_path, "r", encoding="utf-8") as f:
            cur = f.read()
        if cur == token:
            os.remove(lock_path)
    except Exception:
        pass

def save_result(path: str, key: str, value) -> dict:
    ap = os.path.abspath(path)
    if not isinstance(key, str):
        key = str(key)
    v = dict(value) if isinstance(value, dict) else {"value": value}
    ts = v.get("timestamp")
    if ts is None:
        ts = time.time()
        v["timestamp"] = ts
    host = socket.gethostname()
    pid = os.getpid()
    rec = {"key": key, "ts": float(ts), "host": host, "pid": pid, "value": v}
    line_str = json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
    line = (line_str + "\n").encode("utf-8")

    lock_path = ap + ".lock"
    token = _acquire_lock(lock_path)
    try:
        fd = os.open(ap, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            n = os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        _release_lock(lock_path, token)

    try:
        st = os.stat(ap)
        sz = st.st_size
        mt = st.st_mtime
    except Exception:
        sz = None
        mt = None

    return {
        "abs_path": ap,
        "lock_path": lock_path,
        "host": host,
        "pid": pid,
        "bytes_written": n if "n" in locals() else None,
        "file_size": sz,
        "file_mtime": mt,
        "line_len": len(line_str),
        "key": key,
        "ts": float(ts),
    }

def load_results(path: str, return_debug: bool = False, verbose: bool = True, max_bad_decode_samples: int = 5) -> dict:
    ap = os.path.abspath(path)
    dbg = {
        "abs_path": ap,
        "exists": os.path.exists(ap),
        "mode": None,
        "read_lines": 0,
        "kept": 0,
        "unique_keys": 0,
        "skipped_json": 0,
        "skipped_shape": 0,
        "skipped_key": 0,
        "skipped_decode": 0,
        "skipped_empty": 0,
        "first_nonempty": None,
        "head": None,
        "tail": None,
        "bad_decode_samples": [],
    }

    if verbose:
        print(f"[LOAD] path={path}")
        print(f"[LOAD] abs_path={ap}")

    if not dbg["exists"]:
        if verbose:
            print("[LOAD] file does not exist -> returning empty")
        out = {"results": {}}
        return (out, dbg) if return_debug else out

    try:
        st = os.stat(ap)
        if verbose:
            print(f"[LOAD] size_bytes={st.st_size} mtime={st.st_mtime}")
    except Exception as e:
        if verbose:
            print(f"[LOAD] stat failed: {repr(e)}")

    try:
        with open(ap, "rb") as f:
            raw = f.read(4096)
        dbg["head"] = raw.decode("utf-8", errors="replace")
        if verbose:
            head_preview = dbg["head"][:400].replace("\n", "\\n")
            print(f"[LOAD] head_preview(400)={head_preview}")
    except Exception as e:
        if verbose:
            print(f"[LOAD] head read failed: {repr(e)}")
        out = {"results": {}}
        return (out, dbg) if return_debug else out

    try:
        with open(ap, "rb") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - 4096), os.SEEK_SET)
            raw = f.read(4096)
        dbg["tail"] = raw.decode("utf-8", errors="replace")
        if verbose:
            tail_preview = dbg["tail"][-400:].replace("\n", "\\n")
            print(f"[LOAD] tail_preview(400)={tail_preview}")
    except Exception as e:
        if verbose:
            print(f"[LOAD] tail read failed: {repr(e)}")

    first = None
    try:
        with open(ap, "r", encoding="utf-8", errors="ignore") as f:
            for _ in range(200):
                line = f.readline()
                if not line:
                    break
                s = line.strip()
                if s:
                    first = s
                    break
    except Exception as e:
        first = None
        if verbose:
            print(f"[LOAD] first_nonempty scan failed: {repr(e)}")
    dbg["first_nonempty"] = first
    if verbose:
        if first is None:
            print("[LOAD] first_nonempty=None (file may be empty or non-text)")
        else:
            print(f"[LOAD] first_nonempty_prefix={first[:200]}")

    results = {}
    latest_ts = {}
    dbg["mode"] = "jsonl"

    if verbose:
        print("[LOAD] mode=jsonl (line-by-line json.loads)")
        print("[LOAD] counters: read_lines kept unique_keys skipped_empty skipped_decode skipped_json skipped_key skipped_shape")

    with open(ap, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            dbg["read_lines"] += 1

            if not line or line == "\n":
                dbg["skipped_empty"] += 1
                continue

            s = line.strip()
            if not s:
                dbg["skipped_empty"] += 1
                continue

            try:
                rec = json.loads(s)
            except Exception:
                dbg["skipped_decode"] += 1
                if len(dbg["bad_decode_samples"]) < max_bad_decode_samples:
                    dbg["bad_decode_samples"].append(s[:300])
                continue

            if not isinstance(rec, dict):
                dbg["skipped_json"] += 1
                continue

            k = rec.get("key", None)
            v = rec.get("value", None)

            if not isinstance(k, str):
                dbg["skipped_key"] += 1
                continue

            if v is None:
                dbg["skipped_shape"] += 1
                continue

            ts = rec.get("ts", None)
            if ts is None and isinstance(v, dict):
                ts = v.get("timestamp", v.get("_ts", None))

            if ts is None:
                results[k] = v
                latest_ts[k] = float("inf")
                dbg["kept"] += 1
                continue

            try:
                ts_f = float(ts)
            except Exception:
                results[k] = v
                latest_ts[k] = float("inf")
                dbg["kept"] += 1
                continue

            if ts_f >= latest_ts.get(k, float("-inf")):
                latest_ts[k] = ts_f
                results[k] = v
                dbg["kept"] += 1

    dbg["unique_keys"] = len(results)

    if verbose:
        print(f"[LOAD] done: read_lines={dbg['read_lines']} kept={dbg['kept']} unique_keys={dbg['unique_keys']} "
              f"skipped_empty={dbg['skipped_empty']} skipped_decode={dbg['skipped_decode']} skipped_json={dbg['skipped_json']} "
              f"skipped_key={dbg['skipped_key']} skipped_shape={dbg['skipped_shape']}")
        if dbg["skipped_decode"] > 0:
            print(f"[LOAD] bad_decode_samples(n={len(dbg['bad_decode_samples'])}):")
            for i, samp in enumerate(dbg["bad_decode_samples"], 1):
                print(f"  [{i}] {samp}")
        if dbg["unique_keys"] > 0:
            some = list(results.keys())[:5]
            print(f"[LOAD] example_keys={some}")

    out = {"results": results}
    return (out, dbg) if return_debug else out


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
    out = {
        "seed": rep.get("seed"),
        "main": {"acc_fusion": float(rep["main"]["acc_fusion"])},
    }

    if "by_source" in rep:
        main_by = rep["by_source"]["main"]

        out["main"]["acc_parts"] = {k: _slice_acc_from_by_source(main_by, k) for k in ("u1", "u2", "red", "syn")}

        # counts (same for main/synib since same test set)
        out["counts"] = {k: _count_from_by_source(main_by, k) for k in ("u1", "u2", "red", "syn", "none")}

    return out
def generate_prob_settings(step: float, prior_u2: float, targets_psyn: List[float], scramble:bool=True) -> List[Tuple[float, float, float, float]]:
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

    if scramble:
        random.shuffle(out)
    # pu1, pu2, psyn, pred
    # return [(0.475, 0.0, 0.05, 0.475)]
    return out

def main_examinetnk(cfg0: Config, seeds: List[int], lr_vals: List[float], wd_vals: List[float], dropout_vals: List[float],
                     step: float = 0.1, prior_u2: float = 0.0, targets_psyn: List[float] = [0.1, 0.2, 0.3],
                     select_by= "val_loss", save_path: Optional[str] = None) -> Dict[str, Any]:

    cfg0.device = cfg0.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # NEW file (keeps old one untouched)
    results_path = save_path
    db = load_results(results_path)
    if "results" not in db:
        db["results"] = {}

    print_config(cfg0, "DEFAULT CONFIG")
    print("[RUN] sweep non-overlap source probs (u1/u2/red/syn)")
    print(f"[LOG] append/dedupe json: {results_path}")
    print("pu1 pu2 pred psyn | MainTot SynIBTot | Main(u1,u2,red,syn) | SynIB(u1,u2,red,syn) | det | status")
    print("---------------------------------------------------------------------------------------------------")

    settings = generate_prob_settings(step, prior_u2, targets_psyn, cfg0.scramble)
    settings = [(0.2, 0.0, 0.4, 0.4)]
    settings = [(0.45, 0.0, 0.1, 0.45)]
    reps = []
    for (pu1, pu2, psyn, pred) in settings:
        key = _prob_key(pu1, pu2, pred, psyn, 0.0)

        if key in db["results"]:
            summ = db["results"][key]["summary"]
            mt = summ["main_tuned"]["acc_fusion_mean"]
            mt_syn = summ["main_tuned"]["acc_fusion_syn_mean"]
            print(f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
                  f"{mt:.3f} | {mt_syn:.3f} | "
                  " SKIP(existing)")
            continue

        cfg = deepcopy(cfg0)
        _set_nonoverlap_signal_probs(cfg, pu1, pu2, pred, psyn, pnone=0.0)

        this_best = select_best_lr_wd_dropout(
            cfg_base=cfg,
            seeds=seeds,
            lr_vals=lr_vals,
            wd_vals=wd_vals,
            dropout_vals=dropout_vals,
            select_by=select_by,  # "val_acc" or "val_loss"
            larger_is_better=(select_by == "val_acc"),
            verbose=True,
        )

        message = f"{pu1:.2f} {pu2:.2f} {pred:.2f} {psyn:.2f} | "
        for k in ["val_acc", "val_syn_acc", "test_acc", "test_syn_acc"]:
            m, s = _mean_std([r[k] for r in this_best["per_seed"]])
            message += f"{m:.3f}±{s:.3f} | "
        message += (f"best lr={this_best['lr']:.2e} wd={this_best['wd']:.2e} dr={this_best['dropout']:.2f} | ADD")
        print(message)

        payload = {
            "timestamp": time.time(),
            "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
            "seeds": list(seeds),
            "lr_vals": list(lr_vals),
            "wd_vals": list(wd_vals),
            "dropout_vals": list(dropout_vals),
            "cfg0": (cfg0.__dict__.copy() if hasattr(cfg0, "__dict__") else {}),
            "per_seed": this_best["per_seed"],
            "summary": {
                "main_tuned": {
                    "acc_fusion_mean": this_best["test_acc"],
                    "acc_fusion_syn_mean": this_best["test_syn_acc"],
                    "best_lr": this_best["lr"],
                    "best_wd": this_best["wd"],
                    "best_dropout": this_best["dropout"],
                },
            },
        }

        # save_result(results_path, (pu1, pu2, pred, psyn), payload)

        # db["results"][key] = {
        #     "timestamp": time.time(),
        #     "probs": {"pu1": pu1, "pu2": pu2, "pred": pred, "psyn": psyn, "pnone": 0.0},
        #     "seeds": list(seeds),
        #     "lr_vals": list(lr_vals),
        #     "wd_vals": list(wd_vals),
        #     "dropout_vals": list(dropout_vals),
        #     "cfg0": asdict(cfg0) if hasattr(cfg0, "__dict__") else {},
        #     "per_seed": this_best["per_seed"],  # per-seed *final* summaries after rerun at best combo
        #     "summary": {
        #         # keep your old structure; here’s a simple aggregated add-on:
        #         "main_tuned": {
        #             "acc_fusion_mean": this_best["test_acc"],
        #             "acc_fusion_syn_mean": this_best["test_syn_acc"],
        #             "best_lr": this_best['lr'],
        #             "best_wd": this_best['wd'],
        #             "best_dropout": this_best['dropout'],
        #         },
        #     },
        # }
        # _save_results_json(results_path, db)

    # _save_results_json(results_path, db)
    print(f"[LOG] saved: {results_path} (total entries: {len(db['results'])})")
    return reps

import math
import numpy as np
from copy import deepcopy
from dataclasses import asdict
from itertools import product

def _mean_std(xs):
    xs = list(xs)
    if len(xs) == 0:
        return float("nan"), float("nan")
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return m, s

def _agg_weighted(metrics_list, key, weight_key="n"):
    """
    metrics_list: list of dicts like {"val_acc":..., "val_loss":..., "n":...}
    Returns weighted mean for a metric `key` using `weight_key`.
    """
    num = 0.0
    den = 0.0
    for m in metrics_list:
        if key not in m:
            continue
        w = float(m.get(weight_key, 1.0))
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        v = float(m[key])
        if not np.isfinite(v):
            continue
        num += w * v
        den += w
    return (num / den) if den > 0 else float("nan")

def select_best_lr_wd_dropout(
    cfg_base,
    seeds,
    lr_vals,
    wd_vals,
    dropout_vals,
    select_by="val_acc",          # or "val_loss"
    larger_is_better=True,        # True for acc, False for loss
    verbose=False,
):

    table = []  # one row per combo

    best_score = -float("inf") if larger_is_better else float("inf")
    best = None

    for lr, wd, dr in product(lr_vals, wd_vals, dropout_vals):
        per_seed = []

        for sd in seeds:
            cfg = deepcopy(cfg_base)

            # Set HPs on cfg (adapt these field names to your Config)
            cfg.seed = int(sd)
            cfg.lr = float(lr)
            cfg.weight_decay = float(wd)
            cfg.dropout = float(dr)

            device, split, train_loader, val_loader, test_loader = build_loaders(cfg, verbose=False)
            rep = run_main(cfg, train_loader, val_loader, test_loader, device, verbose=False)

            # ---- Pull VAL metrics (adapt to your rep structure) ----

            v_acc = rep["history"].get("best_val_fusion_acc", np.nan)
            v_syn_acc = rep["history"].get("best_val_fusion_syn_acc", np.nan)
            v_loss = rep["history"].get("best_val_fusion_loss", np.nan)
            v_n = rep["history"].get("best_epoch", np.nan)

            t_acc = rep["test_stats"].get("acc_fusion", np.nan)
            t_syn_acc = rep["test_by_source"]["by_source"]["syn"].get("acc", np.nan) if "syn" in rep["test_by_source"]["by_source"] else np.nan


            per_seed.append({
                "seed": int(sd),
                "lr": float(lr),
                "wd": float(wd),
                "dropout": float(dr),
                "val_acc": float(v_acc) if v_acc is not None else float("nan"),
                "test_acc": float(t_acc) if t_acc is not None else float("nan"),
                "val_loss": float(v_loss) if v_loss is not None else float("nan"),
                "val_syn_acc": float(v_syn_acc) if v_syn_acc is not None else float("nan"),
                "test_syn_acc": float(t_syn_acc) if t_syn_acc is not None else float("nan"),
                "val_n": float(v_n) if v_n is not None else 1.0,
            })

        # Aggregate across seeds (weighted by n if provided)
        val_acc = _agg_weighted(per_seed, "val_acc", weight_key="n")
        val_syn_acc = _agg_weighted(per_seed, "val_syn_acc", weight_key="n")
        test_acc = _agg_weighted(per_seed, "test_acc", weight_key="n")
        test_syn_acc = _agg_weighted(per_seed, "test_syn_acc", weight_key="n")
        val_loss = _agg_weighted(per_seed, "val_loss", weight_key="n")

        # define "score" for selection
        score_for_cmp = float(val_syn_acc)

        if select_by == "val_loss":
            score_for_cmp = -float(val_loss)
        elif select_by == "val_acc":
            score_for_cmp = float(val_acc)
        elif select_by == "val_syn_acc":
            score_for_cmp = float(val_syn_acc)

        row = {
            "lr": float(lr),
            "wd": float(wd),
            "dropout": float(dr),
            "val_acc": float(val_acc),
            "val_syn_acc": float(val_syn_acc),
            "val_loss": float(val_loss),
            "test_acc": float(test_acc),
            "test_syn_acc": float(test_syn_acc),
            "score": float(score_for_cmp),
            "per_seed": per_seed,  # optionally omit if table too big
        }
        table.append(row)
        improved = (score_for_cmp > best_score)
        if improved:
            # print(f"[TUNE main] New best found! {table[-1]}")
            best_score = score_for_cmp
            best = row
        if best is None:
            best_score = score_for_cmp
            best = row

        if verbose:
            print(f"[TUNE main] lr={lr:.2e} wd={wd:.2e} dr={dr:.2f} | "
                  f"val_acc={val_acc:.4f} val_syn_cc={val_syn_acc:.4f} val_loss={val_loss:.4f} | test_acc={test_acc:.4f} test_syn_cc={test_syn_acc:.4f}| score={score_for_cmp:.4f}")
        break

    best.update({"table": table})
    return best

def build_loaders(cfg: Config, *, verbose: bool = True):
    seed = cfg.seed
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
        "none": 0.0, "u1": 0.3, "u2": 0.3, "red": 0.3, "syn": 0.1,
        # "red+syn": 0.10, "u1+u2": 0.05, "u1+red": 0.03, "u2+red": 0.02,
        "u1+u2+red+syn": 0.0,
    })

    # Training
    batch_size: int = 64
    epochs: int = 60
    val_method: str = "val_syn_acc"
    lr: float = 3e-4
    weight_decay: float = 3e-2
    hidden: int = 1024
    dropout: float = 0.5

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
    out_dir: str = "pid_synthetic_tuning"

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

    random_mask_proportion = 0.5


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
    # main_sweep_nonoverlap_probs_synib_learned(seeds=[0, 1, 2])

    # main_sweep_nonoverlap_probs_tuned_kl(seeds=[0, 1, 2])
    # main_sweep_nonoverlap_probs()


    cfg = Config()
    cfg.device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg.scramble = True
    mkdirp(cfg.out_dir)

    # lr_vals = [1e-2, 1e-3, 1e-4]
    # wd_vals = [1e-2, 1e-3, 1e-4]
    # dropout_vals = [0.0, 0.3, 0.5]

    lr_vals = [3e-4]
    wd_vals = [1e-5]
    dropout_vals = [0.3]


    # sweep_nonoverlap(cfg, seeds=[0,1,2], lr_vals=lr_vals, wd_vals=wd_vals, dropout_vals=dropout_vals, step=0.05, targets_psyn=[0.05, 0.1, 0.15, 0.2], select_by=cfg.val_method, save_path=os.path.join(cfg.out_dir, "maintuned_valsynacc.jsonl"))
    main_examinetnk(cfg, seeds=[0,1,2], lr_vals=lr_vals, wd_vals=wd_vals, dropout_vals=dropout_vals, step=0.05, targets_psyn=[0.05, 0.1, 0.15, 0.2], select_by="val_acc", save_path=os.path.join(cfg.out_dir, "maintuned_valacc_ntkk.jsonl"))
    # sweep_nonoverlap(cfg, seeds=[0,1,2], lr_vals=lr_vals, wd_vals=wd_vals, dropout_vals=dropout_vals, step=0.05, targets_psyn=[0.05, 0.1, 0.15, 0.2], select_by="val_loss", save_path=os.path.join(cfg.out_dir, "maintuned_valacc.jsonl"))

    # save_path = os.path.join(cfg.out_dir, "maintuned_valacc.jsonl")
    # sweep_nonoverlap(cfg, seeds=[0,1,2], lr_vals=[1e-2, 1e-3, 1e-4], wd_vals=[1e-2, 1e-3, 1e-4], dropout_vals=[0.0,0.3,0.5],
    #                  step=0.05, targets_psyn=[0.05, 0.1, 0.15, 0.2], select_by="val_acc", save_path=save_path)
    #
    # save_path = os.path.join(cfg.out_dir, "maintuned_valloss.jsonl")
    # sweep_nonoverlap(cfg, seeds=[0,1,2], lr_vals=[1e-2, 1e-3, 1e-4], wd_vals=[1e-2, 1e-3, 1e-4], dropout_vals=[0.0,0.3,0.5],
    #                  step=0.05, targets_psyn=[0.05, 0.1, 0.15, 0.2], select_by="val_loss", save_path=save_path)



    # cfg.scramble = True
    # cfg.kl_vals = [0.01,0.1, 1.0, 10.0, 100.0]
    # # cfg.train_method = "synib_mstar"
    # # cfg.train_method = "synib_mrand"
    # cfg.train_method = "synib_mlearned"
    #
    # # cfg.train_method = random.choice([
    # #     "synib_mstar",
    # #     "synib_mrand",
    # #     "synib_mlearned",
    # # ])
    # if cfg.train_method=="synib_mlearned": cfg.lsp_vals = [0.01,0.1, 1.0, 10.0, 100.0]
    # else: cfg.lsp_vals = [0.0]
    #
    # print(f"######################################Training method: {cfg.train_method}######################################")
    # save_path = os.path.join(cfg.out_dir, f"{cfg.train_method}_valsynacc.jsonl")
    # sweep_nonoverlap(cfg, seeds=[0,1,2], lr_vals=lr_vals, wd_vals=wd_vals, dropout_vals=dropout_vals, step=0.05, targets_psyn=[0.05, 0.1, 0.15, 0.2], select_by=cfg.val_method, save_path=save_path)
    #
    # save_path = os.path.join(cfg.out_dir, f"{cfg.train_method}_valacc.jsonl")
    # sweep_nonoverlap(cfg, seeds=[0,1,2], lr_vals=lr_vals, wd_vals=wd_vals, dropout_vals=dropout_vals, step=0.05, targets_psyn=[0.05, 0.1, 0.15, 0.2], select_by="val_acc", save_path=save_path)


