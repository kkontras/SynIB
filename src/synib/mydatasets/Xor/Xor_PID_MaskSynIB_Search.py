"""
XOR PID Mask + SynIB Benchmark (FULL, SELF-CONTAINED, WITH ALL PLOTS)
====================================================================

This single file includes:
  - Config
  - Synthetic PID dataset with EXACTLY-ONE source per datapoint:
      Noise / U1 / U2 / Redundant / Synergy (XOR)
    with p_u1 + p_u2 + p_red + p_syn + p_noise = 1
  - Two models/methods:
      (A) Main baseline: supervised fusion + unimodal losses
      (B) SynIB: supervised + unimodal + counterfactual KL-to-uniform after removing synergy block
  - Evaluation:
      * total accuracy
      * per-source conditional accuracy
      * per-source global contribution (hits/N_total) like your previous "Syn" global plot
  - Nice printing:
      * per-seed x per-ratio rows for Main and SynIB
      * aggregate mean±std summary
      * sweep ranking table
  - Plots:
      * per-run 1x2: Main vs SynIB (Acc + Syn global)
      * per-value plots in a sweep
      * overlay plot: all sweep values as multiple lines (+ std shading)
      * overlay + ranking table in the SAME image (what you asked for)

How you’ll use it:
  - Run a single benchmark:
      run_dual_benchmarks(Config(), out_path="synib_default.png")
  - Sweep lambda_kl and get:
      * individual plots per lambda_kl
      * OVERLAY_lambda_kl_synib.png
      * OVERLAY_TABLE_lambda_kl_synib.png  (overlay + table side-by-side)

Notes:
  - Mask is a per-feature ID map (0..4). DataLoader stacks it to [B, D].
  - make_tilde destroys ONLY synergy features (mask==4) by resampling noise.
"""

from __future__ import annotations

import os
import copy
import itertools
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")  # safe on headless clusters
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# =============================================================================
# Utilities
# =============================================================================

def mkdirp(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_global_seed(seed: int) -> None:
    seed = int(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def as_list(x):
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def pretty_float(x: float) -> str:
    x = float(x)
    if x == 0.0:
        return "0"
    if 1e-2 <= abs(x) < 1e3:
        s = f"{x:.4f}"
        return s.rstrip("0").rstrip(".")
    return f"{x:.1e}"


# =============================================================================
# Config
# =============================================================================

@dataclass
class Config:
    # dataset sizes
    n_train: int = 500
    n_test: int = 2000

    # feature dims
    dim1: int = 25
    dim2: int = 25

    # block sizes (disjoint blocks per modality)
    u1_dim: int = 5
    u2_dim: int = 5
    red_dim: int = 5
    syn_dim: int = 5

    # mixture probabilities (exactly one source per datapoint)
    # p_noise is implied if allow_none=True
    p_u1: float = 0.2
    p_u2: float = 0.0
    p_red: float = 0.3
    p_syn: float = 0.3
    allow_none: bool = True

    # signal/noise knobs
    signal_strength: float = 1.2
    noise_std: float = 1.0

    # optional label flips inside each source (stress-testing shortcuts)
    flip_u1_prob: float = 0.0
    flip_u2_prob: float = 0.0
    flip_red_prob: float = 0.0

    # training
    batch_size: int = 32
    seed: int = 0

    pred_hidden: int = 256
    pred_epochs: int = 50
    pred_lr: float = 1e-4
    weight_decay: float = 1e-4  # applied to SynIB optimizer (baseline uses no wd by default)

    # losses
    lambda_uni: float = 1.0
    lambda_kl: float = 10.0


# =============================================================================
# Dataset
# =============================================================================

SOURCE_NAMES = {0: "Noise", 1: "U1", 2: "U2", 3: "Red", 4: "Syn"}


class PIDDataset(Dataset):
    """
    Exactly ONE source per datapoint.

    source ids:
      0 = Noise
      1 = unique-in-x0 (U1)
      2 = unique-in-x1 (U2)
      3 = redundant (Red)        : same cue in both
      4 = synergy (Syn, XOR)     : x0 has z, x1 has (y XOR z)

    Returned dict:
      x0, x1: modalities
      y: (1,) float in {0,1}
      source: int
      mask0, mask1: global feature-to-component map (0..4). DataLoader stacks -> [B, D].
    """

    def __init__(self, cfg: Config, n: int, seed: int):
        super().__init__()
        self.n = int(n)
        self.dim1 = int(cfg.dim1)
        self.dim2 = int(cfg.dim2)

        rng = np.random.RandomState(seed)
        g = torch.Generator().manual_seed(seed)

        # base noise
        self.x0 = torch.randn(self.n, self.dim1, generator=g) * float(cfg.noise_std)
        self.x1 = torch.randn(self.n, self.dim2, generator=g) * float(cfg.noise_std)

        # labels (0/1)
        self.y = torch.randint(0, 2, (self.n, 1), generator=g, dtype=torch.float32).contiguous()
        y = self.y.squeeze(1)  # (n,)

        # allocate disjoint blocks
        u1_dim, u2_dim, red_dim, syn_dim = int(cfg.u1_dim), int(cfg.u2_dim), int(cfg.red_dim), int(cfg.syn_dim)
        assert u1_dim + red_dim + syn_dim <= self.dim1, "dim1 too small for u1+red+syn blocks"
        assert u2_dim + red_dim + syn_dim <= self.dim2, "dim2 too small for u2+red+syn blocks"

        idx0 = torch.arange(self.dim1)
        idx1 = torch.arange(self.dim2)

        self.u1_idx = idx0[:u1_dim]
        self.u2_idx = idx1[:u2_dim]
        self.red0_idx = idx0[u1_dim: u1_dim + red_dim]
        self.red1_idx = idx1[u2_dim: u2_dim + red_dim]
        self.syn0_idx = idx0[u1_dim + red_dim: u1_dim + red_dim + syn_dim]
        self.syn1_idx = idx1[u2_dim + red_dim: u2_dim + red_dim + syn_dim]

        # global per-feature masks (0..4)
        self.mask0 = torch.zeros(self.dim1, dtype=torch.long)
        self.mask1 = torch.zeros(self.dim2, dtype=torch.long)
        if self.u1_idx.numel() > 0:
            self.mask0[self.u1_idx] = 1
        if self.u2_idx.numel() > 0:
            self.mask1[self.u2_idx] = 2
        if self.red0_idx.numel() > 0:
            self.mask0[self.red0_idx] = 3
        if self.red1_idx.numel() > 0:
            self.mask1[self.red1_idx] = 3
        if self.syn0_idx.numel() > 0:
            self.mask0[self.syn0_idx] = 4
        if self.syn1_idx.numel() > 0:
            self.mask1[self.syn1_idx] = 4

        # choose exactly one source per datapoint
        p_u1, p_u2, p_red, p_syn = float(cfg.p_u1), float(cfg.p_u2), float(cfg.p_red), float(cfg.p_syn)
        if cfg.allow_none:
            p_noise = max(0.0, 1.0 - (p_u1 + p_u2 + p_red + p_syn))
            p = np.array([p_noise, p_u1, p_u2, p_red, p_syn], dtype=np.float64)
            p = p / p.sum()
            src = rng.choice([0, 1, 2, 3, 4], size=self.n, p=p)
        else:
            p = np.array([p_u1, p_u2, p_red, p_syn], dtype=np.float64)
            if p.sum() <= 0:
                raise ValueError("If allow_none=False, at least one of p_u1/p_u2/p_red/p_syn must be > 0.")
            p = p / p.sum()
            src = rng.choice([1, 2, 3, 4], size=self.n, p=p)

        self.sources = torch.from_numpy(src).long()
        m_u1 = (self.sources == 1)
        m_u2 = (self.sources == 2)
        m_red = (self.sources == 3)
        m_syn = (self.sources == 4)

        # optional flips (change label used for that source injection)
        def maybe_flip(mask_rows: torch.Tensor, y_vec: torch.Tensor, flip_prob: float) -> torch.Tensor:
            if float(flip_prob) <= 0:
                return y_vec
            rows = mask_rows.nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                return y_vec
            flips = (torch.rand(rows.numel(), generator=g) < float(flip_prob))
            y_out = y_vec.clone()
            y_out[rows[flips]] = 1.0 - y_out[rows[flips]]
            return y_out

        y_u1 = maybe_flip(m_u1, y, cfg.flip_u1_prob)
        y_u2 = maybe_flip(m_u2, y, cfg.flip_u2_prob)
        y_red = maybe_flip(m_red, y, cfg.flip_red_prob)

        s = float(cfg.signal_strength)

        def inject_block(X: torch.Tensor, rows_mask: torch.Tensor, cols_idx: torch.Tensor, values_1d: torch.Tensor) -> None:
            if cols_idx.numel() == 0:
                return
            rows = rows_mask.nonzero(as_tuple=True)[0]
            if rows.numel() == 0:
                return
            V = (values_1d[rows].unsqueeze(1) * s).expand(rows.numel(), cols_idx.numel())
            X[rows[:, None], cols_idx[None, :]] = V

        # Unique cues
        inject_block(self.x0, m_u1, self.u1_idx, y_u1)
        inject_block(self.x1, m_u2, self.u2_idx, y_u2)

        # Redundant cue (same in both)
        inject_block(self.x0, m_red, self.red0_idx, y_red)
        inject_block(self.x1, m_red, self.red1_idx, y_red)

        # Synergy XOR: x0 has z, x1 has y XOR z
        rows = m_syn.nonzero(as_tuple=True)[0]
        if rows.numel() > 0:
            z = torch.randint(0, 2, (rows.numel(),), generator=g, dtype=torch.float32)
            y_rows = y[rows]
            y_xor_z = (y_rows + z) % 2.0

            if self.syn0_idx.numel() > 0:
                V0 = (z.unsqueeze(1) * s).expand(rows.numel(), self.syn0_idx.numel())
                self.x0[rows[:, None], self.syn0_idx[None, :]] = V0

            if self.syn1_idx.numel() > 0:
                V1 = (y_xor_z.unsqueeze(1) * s).expand(rows.numel(), self.syn1_idx.numel())
                self.x1[rows[:, None], self.syn1_idx[None, :]] = V1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {
            "x0": self.x0[idx],
            "x1": self.x1[idx],
            "y": self.y[idx],
            "source": self.sources[idx],
            "mask0": self.mask0,  # global mask (DataLoader stacks -> [B, D])
            "mask1": self.mask1,
        }


# =============================================================================
# Models
# =============================================================================

class FusionModel(nn.Module):
    """Two encoders + two unimodal heads + fusion head."""

    def __init__(self, dim0: int, dim1: int, hidden: int):
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
        self.clf0 = nn.Linear(hidden, 1)
        self.clf1 = nn.Linear(hidden, 1)
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward_logits(self, x0: torch.Tensor, x1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f0 = self.enc0(x0)
        f1 = self.enc1(x1)
        u0 = self.clf0(f0)
        u1 = self.clf1(f1)
        f = self.fusion(torch.cat([f0, f1], dim=1))
        return f, u0, u1


class SynIBHelpers:
    @staticmethod
    def kl_bern_to_uniform(logits: torch.Tensor) -> torch.Tensor:
        """KL( Bern(sigmoid(logits)) || Bern(0.5) ), averaged over batch."""
        p = torch.sigmoid(logits).clamp(1e-6, 1 - 1e-6)
        return (p * torch.log(2 * p) + (1 - p) * torch.log(2 * (1 - p))).mean()

    @staticmethod
    def make_tilde(x: torch.Tensor, mask: torch.Tensor, noise_std: float) -> torch.Tensor:
        """
        Destroy synergy block (mask==4) by resampling noise.
        mask arrives either as [D] or [B, D]; handle both.
        """
        x_t = x.clone()
        if mask.dim() == 1:
            mask = mask.unsqueeze(0).expand_as(x_t)
        destroy = (mask == 4)
        destroy = (mask == 1) | (mask == 2) | (mask == 3) | (mask == 4)
        destroy = (mask == 1) | (mask == 2) | (mask == 4)
        destroy = (mask == 1) | (mask == 2)
        x_t[destroy] = torch.randn_like(x_t[destroy]) * float(noise_std)
        return x_t


class SynIBModel(FusionModel):
    """Same architecture; SynIB uses SynIBHelpers during training."""
    pass


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_pid(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, Any]:
    """
    Returns:
      - total_acc
      - per-source conditional accuracy: cond_{Name}
      - global contributions: Name = hits(Name)/N_total  (used for "Syn global contribution" plots)
      - counts: n_{Name}, empirical probs: p_emp_{Name}
    """
    model.eval()

    per = {name: {"hits": 0, "n": 0} for name in SOURCE_NAMES.values()}
    total_hits = 0
    total_n = 0

    for b in loader:
        x0 = b["x0"].to(device)
        x1 = b["x1"].to(device)
        y = b["y"].to(device)

        logits, _, _ = model.forward_logits(x0, x1)
        pred = (torch.sigmoid(logits) > 0.5).float()
        correct = (pred == y).squeeze(1).cpu()

        B = y.size(0)
        total_n += B
        total_hits += int(correct.sum().item())

        src = b["source"].cpu().numpy().tolist()
        for i, sid in enumerate(src):
            name = SOURCE_NAMES[int(sid)]
            per[name]["n"] += 1
            per[name]["hits"] += int(correct[i].item())

    out: Dict[str, Any] = {}
    out["total_acc"] = total_hits / max(1, total_n)

    # global contributions
    for name in SOURCE_NAMES.values():
        out[name] = per[name]["hits"] / max(1, total_n)

    # conditional + counts + empirical proportion
    for name in SOURCE_NAMES.values():
        n = per[name]["n"]
        hits = per[name]["hits"]
        out[f"cond_{name}"] = (hits / n) if n > 0 else 0.0
        out[f"n_{name}"] = n
        out[f"p_emp_{name}"] = (n / total_n) if total_n > 0 else 0.0

    return out


# =============================================================================
# Training
# =============================================================================

def train_main(cfg: Config, device: str) -> Dict[str, Any]:
    """Baseline: supervised fusion + unimodal losses."""
    train_loader = DataLoader(PIDDataset(cfg, cfg.n_train, cfg.seed), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(PIDDataset(cfg, cfg.n_test, cfg.seed + 1), batch_size=cfg.batch_size)

    model = FusionModel(cfg.dim1, cfg.dim2, cfg.pred_hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.pred_lr)  # baseline: no weight_decay

    for _ in range(cfg.pred_epochs):
        model.train()
        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y = b["y"].to(device)

            opt.zero_grad()
            f, u0, u1 = model.forward_logits(x0, x1)
            loss = F.binary_cross_entropy_with_logits(f, y) + cfg.lambda_uni * (
                F.binary_cross_entropy_with_logits(u0, y) + F.binary_cross_entropy_with_logits(u1, y)
            )
            loss.backward()
            opt.step()

    return evaluate_pid(model, test_loader, device)


def train_synib(cfg: Config, device: str) -> Dict[str, Any]:
    """SynIB: supervised + unimodal + counterfactual KL-to-uniform after removing syn features."""
    train_loader = DataLoader(PIDDataset(cfg, cfg.n_train, cfg.seed), batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(PIDDataset(cfg, cfg.n_test, cfg.seed + 1), batch_size=cfg.batch_size)

    model = SynIBModel(cfg.dim1, cfg.dim2, cfg.pred_hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.pred_lr, weight_decay=cfg.weight_decay)

    for _ in range(cfg.pred_epochs):
        model.train()
        for b in train_loader:
            x0 = b["x0"].to(device)
            x1 = b["x1"].to(device)
            y = b["y"].to(device)
            mask0 = b["mask0"].to(device)
            mask1 = b["mask1"].to(device)

            # base supervised loss
            f, u0, u1 = model.forward_logits(x0, x1)
            l_base = F.binary_cross_entropy_with_logits(f, y) + cfg.lambda_uni * (
                F.binary_cross_entropy_with_logits(u0, y) + F.binary_cross_entropy_with_logits(u1, y)
            )

            # counterfactuals: destroy syn block in each modality separately
            x0_t = SynIBHelpers.make_tilde(x0, mask0, noise_std=cfg.noise_std)
            x1_t = SynIBHelpers.make_tilde(x1, mask1, noise_std=cfg.noise_std)

            f_t0, _, _ = model.forward_logits(x0_t, x1)
            f_t1, _, _ = model.forward_logits(x0, x1_t)
            l_cf = SynIBHelpers.kl_bern_to_uniform(f_t0) + SynIBHelpers.kl_bern_to_uniform(f_t1)

            loss = l_base + cfg.lambda_kl * l_cf

            opt.zero_grad()
            loss.backward()
            opt.step()

    return evaluate_pid(model, test_loader, device)


# =============================================================================
# Printing helpers
# =============================================================================

def print_config(cfg: Config, title: str = "CONFIG") -> None:
    if cfg.allow_none:
        p_noise = max(0.0, 1.0 - (cfg.p_u1 + cfg.p_u2 + cfg.p_red + cfg.p_syn))
    else:
        p_noise = 0.0

    lines = [
        f"[{title}]",
        f"  dims: dim1={cfg.dim1} dim2={cfg.dim2} | blocks: u1={cfg.u1_dim} u2={cfg.u2_dim} red={cfg.red_dim} syn={cfg.syn_dim}",
        f"  probs: p_u1={pretty_float(cfg.p_u1)} p_u2={pretty_float(cfg.p_u2)} p_red={pretty_float(cfg.p_red)} p_syn={pretty_float(cfg.p_syn)}"
        + (f" p_noise={pretty_float(p_noise)}" if cfg.allow_none else ""),
        f"  data: n_train={cfg.n_train} n_test={cfg.n_test} batch={cfg.batch_size}",
        f"  signal: strength={pretty_float(cfg.signal_strength)} noise_std={pretty_float(cfg.noise_std)}",
        f"  flips: u1={pretty_float(cfg.flip_u1_prob)} u2={pretty_float(cfg.flip_u2_prob)} red={pretty_float(cfg.flip_red_prob)}",
        f"  model: hidden={cfg.pred_hidden}",
        f"  opt: lr={pretty_float(cfg.pred_lr)} epochs={cfg.pred_epochs} wd(SynIB)={pretty_float(cfg.weight_decay)}",
        f"  loss: lambda_uni={pretty_float(cfg.lambda_uni)} lambda_kl={pretty_float(cfg.lambda_kl)}",
    ]
    print("\n" + "=" * 120)
    for ln in lines:
        print(ln)
    print("=" * 120)


def print_run_header() -> None:
    cols = [
        ("Seed", 4),
        ("Ratio", 5),
        ("Meth", 5),
        ("Tot", 5),
        ("U1", 5),
        ("U2", 5),
        ("Red", 6),
        ("Syn", 6),
        ("Noi", 6),
        ("GlobSyn", 7),
        ("pSyn", 5),
        ("nSyn", 5),
    ]
    header = " | ".join([f"{n:<{w}}" for n, w in cols])
    print(header)
    print("-" * len(header))


def format_row(seed: int, ratio: float, method: str, res: Dict[str, Any]) -> str:
    return (
        f"{seed:<4d} | "
        f"{ratio:<5.2f} | "
        f"{method:<5} | "
        f"{res['total_acc']:<5.2f} | "
        f"{res['cond_U1']:<5.2f} | "
        f"{res['cond_U2']:<5.2f} | "
        f"{res['cond_Red']:<6.2f} | "
        f"{res['cond_Syn']:<6.2f} | "
        f"{res['cond_Noise']:<6.2f} | "
        f"{res['Syn']:<7.3f} | "
        f"{res['p_emp_Syn']:<5.2f} | "
        f"{res['n_Syn']:<5d}"
    )


# =============================================================================
# Plotting
# =============================================================================

def plot_dual_panels_single(
    ratios: List[float],
    main_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    synib_stats: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    p_signal: float,
    out_path: str,
    title: str = "",
) -> None:
    """
    Standard 1x2 plot:
      Panel 1: Total accuracy (Main vs SynIB)
      Panel 2: Syn global contribution (Main vs SynIB + theoretical max)
    """
    r = np.array(ratios, dtype=float)
    m_acc_a, s_acc_a = main_stats["acc"]
    m_syn_a, s_syn_a = main_stats["syn"]
    m_acc_b, s_acc_b = synib_stats["acc"]
    m_syn_b, s_syn_b = synib_stats["syn"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    if title:
        fig.suptitle(title, fontsize=11)

    # Panel 1: Total accuracy
    ax1.plot(r, m_acc_a, "o-", label="Main (Baseline)", color="#7f8c8d")
    ax1.fill_between(r, m_acc_a - s_acc_a, m_acc_a + s_acc_a, color="#7f8c8d", alpha=0.2)
    ax1.plot(r, m_acc_b, "s-", label="SynIB (Proposed)", color="#2980b9")
    ax1.fill_between(r, m_acc_b - s_acc_b, m_acc_b + s_acc_b, color="#2980b9", alpha=0.2)
    ax1.axhline(0.5, color="r", linestyle="--", alpha=0.3, label="Random Chance")
    ax1.set_title("Generalization Stress Test (Total Acc)", fontsize=12)
    ax1.set_xlabel("Ratio of Synergy (0=Redundant, 1=Synergistic)")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_ylim(0.4, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: Syn global contribution
    ax2.plot(r, m_syn_a, "o--", label="Main Syn Contrib", color="#7f8c8d")
    ax2.fill_between(r, m_syn_a - s_syn_a, m_syn_a + s_syn_a, color="#7f8c8d", alpha=0.2)
    ax2.plot(r, m_syn_b, "s--", label="SynIB Syn Contrib", color="#2980b9")
    ax2.fill_between(r, m_syn_b - s_syn_b, m_syn_b + s_syn_b, color="#2980b9", alpha=0.2)
    ax2.plot(r, p_signal * r, "k:", alpha=0.6, label="Theoretical Max")
    ax2.set_title("Synergy Extraction Quality (Global Contrib)", fontsize=12)
    ax2.set_xlabel("Ratio of Synergy")
    ax2.set_ylabel("Global Contribution")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    mkdirp(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_param_sweep_overlay(
    sweep_results: Dict[Any, Dict[str, Any]],
    param_name: str,
    *,
    method: str = "synib",          # "synib" | "main" | "both"
    p_signal: float = 0.6,
    include_std: bool = True,
    out_path: str = "OVERLAY.png",
    title: str = "",
    legend_ncol: int = 2,
) -> str:
    """
    Overlay plot: many parameter values as different lines (+ std shading).
    Uses the same acc/syn panels as before.

    method:
      - "synib": 1x2 overlay for SynIB only
      - "main":  1x2 overlay for Main only
      - "both":  2x2 overlay: rows=(Main,SynIB), cols=(Acc,Syn)
    """
    if len(sweep_results) == 0:
        raise ValueError("sweep_results is empty.")

    values = list(sweep_results.keys())
    try:
        values = sorted(values, key=lambda v: float(v))
    except Exception:
        values = sorted(values, key=lambda v: str(v))

    first = sweep_results[values[0]]
    ratios = np.array(first["ratios"], dtype=float)

    def label(v):
        if isinstance(v, (float, np.floating)):
            return f"{param_name}={pretty_float(float(v))}"
        return f"{param_name}={v}"

    def plot_one(ax_acc, ax_syn, bench, which: str, lbl: str):
        acc_m = np.array(bench[which]["acc_mean"], dtype=float)
        acc_s = np.array(bench[which]["acc_std"], dtype=float)
        syn_m = np.array(bench[which]["syn_mean"], dtype=float)
        syn_s = np.array(bench[which]["syn_std"], dtype=float)

        ax_acc.plot(ratios, acc_m, linestyle="-", label=lbl)
        if include_std:
            ax_acc.fill_between(ratios, acc_m - acc_s, acc_m + acc_s, alpha=0.15)

        ax_syn.plot(ratios, syn_m, linestyle="--", label=lbl)
        if include_std:
            ax_syn.fill_between(ratios, syn_m - syn_s, syn_m + syn_s, alpha=0.15)

    if method in ("synib", "main"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        if title:
            fig.suptitle(title, fontsize=11)

        for v in values:
            plot_one(ax1, ax2, sweep_results[v], method, label(v))

        ax1.axhline(0.5, linestyle="--", alpha=0.3, label="Random Chance")
        ax1.set_title(f"{method.upper()}: Total Acc (overlay)", fontsize=12)
        ax1.set_xlabel("Ratio of Synergy")
        ax1.set_ylabel("Test Accuracy")
        ax1.set_ylim(0.4, 1.05)
        ax1.grid(True, alpha=0.3)
        ax1.legend(ncol=legend_ncol)

        ax2.plot(ratios, p_signal * ratios, "k:", alpha=0.6, label="Theoretical Max")
        ax2.set_title(f"{method.upper()}: Syn Global (overlay)", fontsize=12)
        ax2.set_xlabel("Ratio of Synergy")
        ax2.set_ylabel("Global Contribution")
        ax2.grid(True, alpha=0.3)
        ax2.legend(ncol=legend_ncol)

        plt.tight_layout()
        mkdirp(os.path.dirname(out_path) or ".")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    if method == "both":
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        if title:
            fig.suptitle(title, fontsize=11)

        ax_acc_main, ax_syn_main = axs[0, 0], axs[0, 1]
        ax_acc_synb, ax_syn_synb = axs[1, 0], axs[1, 1]

        for v in values:
            plot_one(ax_acc_main, ax_syn_main, sweep_results[v], "main", label(v))
            plot_one(ax_acc_synb, ax_syn_synb, sweep_results[v], "synib", label(v))

        for ax in (ax_acc_main, ax_acc_synb):
            ax.axhline(0.5, linestyle="--", alpha=0.3)
            ax.set_xlabel("Ratio of Synergy")
            ax.set_ylabel("Test Accuracy")
            ax.set_ylim(0.4, 1.05)
            ax.grid(True, alpha=0.3)

        for ax in (ax_syn_main, ax_syn_synb):
            ax.plot(ratios, p_signal * ratios, "k:", alpha=0.6)
            ax.set_xlabel("Ratio of Synergy")
            ax.set_ylabel("Global Contribution")
            ax.grid(True, alpha=0.3)

        ax_acc_main.set_title("MAIN: Total Acc (overlay)", fontsize=12)
        ax_syn_main.set_title("MAIN: Syn Global (overlay)", fontsize=12)
        ax_acc_synb.set_title("SYNIB: Total Acc (overlay)", fontsize=12)
        ax_syn_synb.set_title("SYNIB: Syn Global (overlay)", fontsize=12)

        ax_acc_main.legend(ncol=legend_ncol)
        ax_acc_synb.legend(ncol=legend_ncol)

        plt.tight_layout()
        mkdirp(os.path.dirname(out_path) or ".")
        plt.savefig(out_path, dpi=300)
        plt.close(fig)
        return out_path

    raise ValueError("method must be 'synib', 'main', or 'both'")


def plot_param_sweep_overlay_with_table(
    sweep_results: Dict[Any, Dict[str, Any]],
    ranking_rows: List[Tuple[Any, float, float, str]],
    param_name: str,
    *,
    method: str = "synib",          # "synib" | "main" | "both"
    p_signal: float = 0.6,
    include_std: bool = True,
    out_path: str = "OVERLAY_TABLE.png",
    title: str = "",
) -> str:
    """
    Single figure with:
      - overlay curves on the left
      - ranking table on the right

    ranking_rows: list of (value, syn_gap_mae, acc_auc, per_value_plot_path), sorted best->worst
    Curves are labeled with IDs [1], [2], ... matching table rows.
    """
    if len(sweep_results) == 0:
        raise ValueError("sweep_results is empty.")
    if len(ranking_rows) == 0:
        raise ValueError("ranking_rows is empty.")

    values_in_order = [row[0] for row in ranking_rows]
    first = sweep_results[values_in_order[0]]
    ratios = np.array(first["ratios"], dtype=float)

    def plot_line(ax_acc, ax_syn, bench, which: str, line_id: str):
        acc_m = np.array(bench[which]["acc_mean"], dtype=float)
        acc_s = np.array(bench[which]["acc_std"], dtype=float)
        syn_m = np.array(bench[which]["syn_mean"], dtype=float)
        syn_s = np.array(bench[which]["syn_std"], dtype=float)

        ax_acc.plot(ratios, acc_m, linestyle="-", label=line_id)
        if include_std:
            ax_acc.fill_between(ratios, acc_m - acc_s, acc_m + acc_s, alpha=0.15)

        ax_syn.plot(ratios, syn_m, linestyle="--", label=line_id)
        if include_std:
            ax_syn.fill_between(ratios, syn_m - syn_s, syn_m + syn_s, alpha=0.15)

    # Layout
    if method in ("synib", "main"):
        fig = plt.figure(figsize=(20, 6))
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.0, 1.15])
        ax_acc = fig.add_subplot(gs[0, 0])
        ax_syn = fig.add_subplot(gs[0, 1])
        ax_tbl = fig.add_subplot(gs[0, 2])
        axes_to_plot = [(method, ax_acc, ax_syn)]
    elif method == "both":
        fig = plt.figure(figsize=(20, 10))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.0, 1.0])
        ax_acc_main = fig.add_subplot(gs[0, 0])
        ax_syn_main = fig.add_subplot(gs[0, 1])
        ax_acc_synb = fig.add_subplot(gs[1, 0])
        ax_syn_synb = fig.add_subplot(gs[1, 1])
        ax_tbl = fig.add_subplot(gs[:, 2])
        axes_to_plot = [
            ("main", ax_acc_main, ax_syn_main),
            ("synib", ax_acc_synb, ax_syn_synb),
        ]
    else:
        raise ValueError("method must be 'synib', 'main', or 'both'.")

    if title:
        fig.suptitle(title, fontsize=12)

    # Plot lines
    for rank, v in enumerate(values_in_order, start=1):
        bench = sweep_results[v]
        line_id = f"[{rank}]"
        for which, axA, axS in axes_to_plot:
            plot_line(axA, axS, bench, which=which, line_id=line_id)

    # Decorate axes
    for which, axA, axS in axes_to_plot:
        axA.axhline(0.5, linestyle="--", alpha=0.3)
        axA.set_xlabel("Ratio of Synergy")
        axA.set_ylabel("Test Accuracy")
        axA.set_ylim(0.4, 1.05)
        axA.grid(True, alpha=0.3)

        axS.plot(ratios, p_signal * ratios, "k:", alpha=0.6)
        axS.set_xlabel("Ratio of Synergy")
        axS.set_ylabel("Global Contribution")
        axS.grid(True, alpha=0.3)

        if method == "both":
            if which == "main":
                axA.set_title("MAIN: Total Acc (overlay)", fontsize=12)
                axS.set_title("MAIN: Syn Global (overlay)", fontsize=12)
            else:
                axA.set_title("SYNIB: Total Acc (overlay)", fontsize=12)
                axS.set_title("SYNIB: Syn Global (overlay)", fontsize=12)
        else:
            axA.set_title(f"{which.upper()}: Total Acc (overlay)", fontsize=12)
            axS.set_title(f"{which.upper()}: Syn Global (overlay)", fontsize=12)

    # Legend only on first acc axis (keeps clutter low)
    axes_to_plot[0][1].legend(title="Line ID", ncol=6, fontsize=9)

    # Build table
    ax_tbl.axis("off")
    ax_tbl.set_title("Ranking (best at top)", fontsize=12, pad=10)

    def basename(p: str) -> str:
        return os.path.basename(p)

    col_labels = ["ID", param_name, "syn_gap_mae", "acc_auc", "plot"]
    cell_text = []
    for rank, (v, gap, auc, path) in enumerate(ranking_rows, start=1):
        cell_text.append([f"[{rank}]", f"{v}", f"{gap:.4f}", f"{auc:.4f}", basename(path)])

    table = ax_tbl.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.25)

    plt.tight_layout()
    mkdirp(os.path.dirname(out_path) or ".")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


# =============================================================================
# Benchmark runner
# =============================================================================

def run_dual_benchmarks(
    cfg: Optional[Config] = None,
    *,
    num_seeds: int = 3,
    p_signal: float = 0.6,
    ratios: Optional[List[float]] = None,
    device: Optional[str] = None,
    out_path: str = "synib_dual_benchmark.png",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Sweep ratios; for each ratio:
      cfg.p_syn = p_signal * ratio
      cfg.p_red = p_signal * (1 - ratio)
    """

    if cfg is None:
        cfg = Config()
    if ratios is None:
        ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
    ratios = list(ratios)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # per-ratio lists over seeds
    main_acc = [[] for _ in ratios]
    main_syn = [[] for _ in ratios]
    synb_acc = [[] for _ in ratios]
    synb_syn = [[] for _ in ratios]

    if verbose:
        print_config(cfg, title="BASE CONFIG (before ratio update)")
        print_run_header()

    for seed in range(num_seeds):
        for i, ratio in enumerate(ratios):
            cfg_i = copy.deepcopy(cfg)
            cfg_i.seed = int(seed)
            cfg_i.p_syn = float(p_signal) * float(ratio)
            cfg_i.p_red = float(p_signal) * (1.0 - float(ratio))

            # same init randomness for both methods at this (seed, ratio)
            set_global_seed(seed * 1000 + int(ratio * 100) + 7)

            res_main = train_main(cfg_i, device)

            set_global_seed(seed * 1000 + int(ratio * 100) + 7)
            res_synb = train_synib(cfg_i, device)

            if verbose:
                print(format_row(seed, ratio, "Main", res_main))
                print(format_row(seed, ratio, "SynIB", res_synb))
                print("")

            main_acc[i].append(res_main["total_acc"])
            main_syn[i].append(res_main["Syn"])
            synb_acc[i].append(res_synb["total_acc"])
            synb_syn[i].append(res_synb["Syn"])

    def mean_std(x: List[List[float]]) -> Tuple[np.ndarray, np.ndarray]:
        m = np.array([np.mean(v) for v in x], dtype=float)
        s = np.array([np.std(v) for v in x], dtype=float)
        return m, s

    m_acc_a, s_acc_a = mean_std(main_acc)
    m_syn_a, s_syn_a = mean_std(main_syn)
    m_acc_b, s_acc_b = mean_std(synb_acc)
    m_syn_b, s_syn_b = mean_std(synb_syn)

    title = (
        f"lr={cfg.pred_lr:.0e} ep={cfg.pred_epochs} hid={cfg.pred_hidden} | "
        f"uni={cfg.lambda_uni} kl={cfg.lambda_kl} wd={cfg.weight_decay} | n_train={cfg.n_train}"
    )

    plot_dual_panels_single(
        ratios=ratios,
        main_stats={"acc": (m_acc_a, s_acc_a), "syn": (m_syn_a, s_syn_a)},
        synib_stats={"acc": (m_acc_b, s_acc_b), "syn": (m_syn_b, s_syn_b)},
        p_signal=float(p_signal),
        out_path=out_path,
        title=title,
    )

    if verbose:
        print("\n" + "=" * 120)
        print("[AGGREGATE] mean ± std over seeds")
        print(" ratios:", ratios)
        print(" Main  Acc:", np.round(m_acc_a, 3), "±", np.round(s_acc_a, 3))
        print(" SynIB Acc:", np.round(m_acc_b, 3), "±", np.round(s_acc_b, 3))
        print(" Main  Syn:", np.round(m_syn_a, 3), "±", np.round(s_syn_a, 3))
        print(" SynIB Syn:", np.round(m_syn_b, 3), "±", np.round(s_syn_b, 3))
        print(" plot:", out_path)
        print("=" * 120)

    return {
        "ratios": ratios,
        "main": {"acc_mean": m_acc_a, "acc_std": s_acc_a, "syn_mean": m_syn_a, "syn_std": s_syn_a},
        "synib": {"acc_mean": m_acc_b, "acc_std": s_acc_b, "syn_mean": m_syn_b, "syn_std": s_syn_b},
        "cfg": copy.deepcopy(cfg),
        "p_signal": float(p_signal),
        "plot_path": out_path,
    }


# =============================================================================
# Sweeps
# =============================================================================

DEFAULT_SWEEPS = {
    # objective/optimization knobs
    "lambda_uni": [0.0, 0.1, 0.3, 1.0],
    "lambda_kl": [0.0, 0.25, 0.5, 1.0, 2.0, 4.0],
    "pred_lr": [3e-5, 1e-4, 3e-4],
    "pred_epochs": [25, 50, 100, 200],
    "weight_decay": [0.0, 1e-4, 1e-3],

    # data/SNR
    "n_train": [500, 2000, 10000],
    "signal_strength": [0.6, 1.2, 2.0],
    "noise_std": [0.5, 1.0, 2.0],

    # capacity
    "pred_hidden": [128, 256, 512],
}


def score_setting(bench: Dict[str, Any]) -> Dict[str, float]:
    """
    Scalar scores for ranking sweep values:
      - syn_gap_mae: mean abs gap between SynIB syn curve and theoretical max (lower is better)
      - acc_auc: AUC of SynIB accuracy curve vs ratio (higher is better)
    """
    ratios = np.array(bench["ratios"], dtype=float)
    p_signal = float(bench["p_signal"])
    ideal = p_signal * ratios

    syn = np.array(bench["synib"]["syn_mean"], dtype=float)
    acc = np.array(bench["synib"]["acc_mean"], dtype=float)

    syn_gap_mae = float(np.mean(np.abs(syn - ideal)))
    acc_auc = float(np.trapz(acc, ratios))
    return {"syn_gap_mae": syn_gap_mae, "acc_auc": acc_auc}


def run_param_sweep(
    param_name: str,
    values: Optional[List[Any]] = None,
    *,
    base_cfg: Optional[Config] = None,
    num_seeds: int = 3,
    p_signal: float = 0.6,
    ratios: Optional[List[float]] = None,
    out_dir: str = "param_sweeps",
    verbose_runs: bool = False,
    overlay_method: str = "synib",  # "synib" | "main" | "both"
) -> Tuple[Dict[Any, Dict[str, Any]], List[Tuple[Any, float, float, str]]]:
    """
    Sweep one parameter; for each value:
      - runs run_dual_benchmarks()
      - saves per-value plot
    After sweep:
      - saves overlay plot
      - saves overlay+table plot (overlay next to ranked table)
      - prints ranked table

    Returns:
      (sweep_results, ranking_rows)
    where:
      sweep_results[value] = benchmark dict
      ranking_rows = [(value, syn_gap_mae, acc_auc, per_value_plot_path), ...] sorted best->worst
    """
    if base_cfg is None:
        base_cfg = Config()
    if values is None:
        if param_name not in DEFAULT_SWEEPS:
            raise ValueError(f"No default sweep values for '{param_name}'. Provide values=[...].")
        values = DEFAULT_SWEEPS[param_name]
    values = as_list(values)

    mkdirp(out_dir)
    sweep_results: Dict[Any, Dict[str, Any]] = {}
    ranking_rows: List[Tuple[Any, float, float, str]] = []

    print("\n" + "#" * 120)
    print(f"[PARAM SWEEP] {param_name} over {values} | seeds={num_seeds} | overlay_method={overlay_method}")
    print("#" * 120)

    for v in values:
        cfg = copy.deepcopy(base_cfg)
        setattr(cfg, param_name, v)

        safe = str(v).replace(".", "p").replace("-", "m")
        per_value_plot = os.path.join(out_dir, f"{param_name}={safe}.png")

        bench = run_dual_benchmarks(
            cfg,
            num_seeds=num_seeds,
            p_signal=p_signal,
            ratios=ratios,
            out_path=per_value_plot,
            verbose=verbose_runs,
        )
        sweep_results[v] = bench

        sc = score_setting(bench)
        ranking_rows.append((v, sc["syn_gap_mae"], sc["acc_auc"], per_value_plot))

        print(
            f"[DONE] {param_name}={v:<8} | syn_gap_mae={sc['syn_gap_mae']:.4f} | "
            f"acc_auc={sc['acc_auc']:.4f} | plot={per_value_plot}"
        )

    # Sort best->worst by syn_gap_mae
    ranking_rows.sort(key=lambda t: t[1])

    # Overlay plot (all lines)
    overlay_path = os.path.join(out_dir, f"OVERLAY_{param_name}_{overlay_method}.png")
    plot_param_sweep_overlay(
        sweep_results,
        param_name,
        method=overlay_method,
        p_signal=float(p_signal),
        include_std=True,
        out_path=overlay_path,
        title=f"Overlay sweep: {param_name} | method={overlay_method} | seeds={num_seeds}",
        legend_ncol=2,
    )
    print(f"[OVERLAY SAVED] {overlay_path}")

    # Overlay + ranking table (side-by-side)  ✅ what you asked for
    overlay_table_path = os.path.join(out_dir, f"OVERLAY_TABLE_{param_name}_{overlay_method}.png")
    plot_param_sweep_overlay_with_table(
        sweep_results=sweep_results,
        ranking_rows=ranking_rows,
        param_name=param_name,
        method=overlay_method,
        p_signal=float(p_signal),
        include_std=True,
        out_path=overlay_table_path,
        title=f"{param_name} sweep | method={overlay_method} | seeds={num_seeds}",
    )
    print(f"[OVERLAY+TABLE SAVED] {overlay_table_path}")

    # Print ranked table
    print("\n" + "-" * 120)
    print(f"[SUMMARY RANKING] {param_name}  (lower syn_gap_mae is better; higher acc_auc is better)")
    print(f"{param_name:<18} | {'syn_gap_mae':<12} | {'acc_auc':<8} | plot")
    print("-" * 120)
    for v, gap, auc, path in ranking_rows:
        print(f"{str(v):<18} | {gap:<12.4f} | {auc:<8.4f} | {path}")
    print("-" * 120)

    return sweep_results, ranking_rows


# =============================================================================
# Optional: grid sweep (lambda_kl, lr, epochs)
# =============================================================================

def sweep_benchmarks_grid(
    *,
    num_seeds: int = 3,
    lambda_kls=(0.0, 0.5, 1.0, 2.0, 4.0),
    lrs=(3e-5, 1e-4, 3e-4),
    epochs=(25, 50, 100),
    base_cfg: Optional[Config] = None,
    p_signal: float = 0.6,
    ratios: Optional[List[float]] = None,
    out_dir: str = "grid_sweeps",
    verbose_runs: bool = False,
) -> Dict[Tuple[float, float, int], Dict[str, Any]]:
    if base_cfg is None:
        base_cfg = Config()

    lambda_kls = as_list(lambda_kls)
    lrs = as_list(lrs)
    epochs = as_list(epochs)

    mkdirp(out_dir)
    results: Dict[Tuple[float, float, int], Dict[str, Any]] = {}

    print("\n" + "#" * 120)
    print(f"[GRID SWEEP] |kl|={len(lambda_kls)} |lr|={len(lrs)} |ep|={len(epochs)} => {len(lambda_kls)*len(lrs)*len(epochs)} runs")
    print("#" * 120)

    for lam, lr, ep in itertools.product(lambda_kls, lrs, epochs):
        cfg = copy.deepcopy(base_cfg)
        cfg.lambda_kl = float(lam)
        cfg.pred_lr = float(lr)
        cfg.pred_epochs = int(ep)

        tag = f"kl{float(lam):g}_lr{float(lr):.0e}_ep{int(ep)}"
        plot_path = os.path.join(out_dir, f"{tag}.png")

        bench = run_dual_benchmarks(
            cfg,
            num_seeds=num_seeds,
            p_signal=p_signal,
            ratios=ratios,
            out_path=plot_path,
            verbose=verbose_runs,
        )
        results[(float(lam), float(lr), int(ep))] = bench

        sc = score_setting(bench)
        print(f"[SAVED] {tag:<22} syn_gap_mae={sc['syn_gap_mae']:.4f} acc_auc={sc['acc_auc']:.4f} -> {plot_path}")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    np.random.seed(42)

    # 1) Default single benchmark
    run_dual_benchmarks(
        cfg=Config(),
        num_seeds=3,
        p_signal=0.6,
        ratios=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        out_path="synib_default_masku.png",
        verbose=True,
    )

    # 2) Example: lambda_kl sweep with the overlay+table plot you requested
    #    This will create:
    #      - per-value plots:   param_sweeps/lambda_kl/lambda_kl=4p0.png, ...
    #      - overlay plot:      param_sweeps/lambda_kl/OVERLAY_lambda_kl_synib.png
    #      - overlay + table:  param_sweeps/lambda_kl/OVERLAY_TABLE_lambda_kl_synib.png
    # run_param_sweep(
    #     "lambda_kl",
    #     values=[0.1, 0.5, 1.0, 5.0, 10.0],
    #     base_cfg=Config(),
    #     num_seeds=3,
    #     p_signal=0.6,
    #     ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    #     out_dir="param_sweeps/lambda_kl",
    #     verbose_runs=False,
    #     overlay_method="both",
    # )


if __name__ == "__main__":
    main()
