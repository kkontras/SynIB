"""xor_spurious_pub.py
XOR + spurious shortcut experiment with method-specific tuning.

Methods:
  1. Baseline
  2. SynIB M*
  3. SynIB Random
  4. SynIB Learned
  5. SynIB Diffusion

Examples:
  python scripts/analysis/xor_spurious_pub.py --epochs 5 --n_train 240
  python scripts/analysis/xor_spurious_pub.py --tuning_mode retune --tune_epochs 8
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import random
from dataclasses import asdict, dataclass, field
from itertools import product
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


@dataclass
class Cfg:
    n_train: int = 300
    n_val: int = 120
    n_test: int = 30000
    dim0: int = 64
    dim1: int = 64
    signal_dims: int = 4
    spur_dims: int = 6
    signal_scale: float = 2.2
    signal_noise_std: float = 0.55
    spur_scale: float = 1.3
    base_noise_std: float = 1.0
    epochs: int = 100
    batch_size: int = 128
    hidden: int = 16
    lambda_uni: float = 0.05
    comp_noise_std: float = 1.0
    mask_tau: float = 1.0
    mask_noise_std: float = 1.0
    mask_hard: bool = True
    mask_hard_thresh: float = 0.5
    device: str = "cpu"


@dataclass
class MethodParams:
    lr: float
    wd: float
    dropout: float
    lambda_kl: float = 1.0
    mask_prob: float = 0.6
    mask_steps: int = 20
    mask_lr: float = 0.1
    mask_lam_sparsity: float = 5.0
    diffusion_steps: int = 10
    diffusion_schedule: str = "cosine"
    diffusion_mask_prob: float = 0.95
    diffusion_eps_std: float = 1.0
    l_anneal_epochs: int = 0


@dataclass
class SearchSpace:
    lr: List[float] = field(default_factory=list)
    wd: List[float] = field(default_factory=list)
    dropout: List[float] = field(default_factory=list)
    lambda_kl: List[float] = field(default_factory=list)
    mask_prob: List[float] = field(default_factory=list)
    mask_steps: List[int] = field(default_factory=list)
    mask_lam_sparsity: List[float] = field(default_factory=list)
    diffusion_steps: List[int] = field(default_factory=list)


METHODS = [
    "synib_star",
    "synib_diffusion",
    "synib_learned",
    "synib_random",
    "baseline",
]

ARCHIVED_BEST: Dict[str, MethodParams] = {
    "baseline": MethodParams(lr=1e-2, wd=1e-3, dropout=0.0, lambda_kl=0.0),
    "synib_star": MethodParams(lr=3e-4, wd=1e-5, dropout=0.0, lambda_kl=1e1),
    "synib_random": MethodParams(
        lr=3e-4, wd=1e-5, dropout=0.0, lambda_kl=1e1, mask_prob=0.4
    ),
    "synib_learned": MethodParams(
        lr=3e-4, wd=1e-5, dropout=0.0, lambda_kl=1e1,
        mask_steps=50, mask_lr=3.0, mask_lam_sparsity=0.03,
        l_anneal_epochs=20,
    ),
    "synib_diffusion": MethodParams(
        lr=3e-4, wd=1e-5, dropout=0.0, lambda_kl=1e1, diffusion_steps=10
    ),
}

SEARCH_SPACES: Dict[str, SearchSpace] = {
    "baseline": SearchSpace(
        lr=[1e-2, 1e-3, 1e-4],
        wd=[1e-2, 1e-3, 1e-4],
        dropout=[0.0, 0.3, 0.5],
    ),
    "synib_star": SearchSpace(
        lr=[3e-4],
        wd=[1e-5],
        dropout=[0.0],
        lambda_kl=[1e-2, 1e-1, 1e0, 1e1, 1e2],
    ),
    "synib_random": SearchSpace(
        lr=[3e-4],
        wd=[1e-5],
        dropout=[0.0],
        lambda_kl=[1e-2, 1e-1, 1e0, 1e1, 1e2],
        mask_prob=[0.3, 0.6, 0.9],
    ),
    "synib_learned": SearchSpace(
        lr=[3e-4],
        wd=[1e-5],
        dropout=[0.0],
        lambda_kl=[1e-2, 1e-1, 1e0, 1e1, 1e2],
        mask_steps=[10, 20, 50],
        mask_lam_sparsity=[0.01, 0.1, 1.0, 5.0, 10.0],
    ),
    "synib_diffusion": SearchSpace(
        lr=[3e-4],
        wd=[1e-5],
        dropout=[0.0],
        lambda_kl=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        diffusion_steps=[10, 100],
    ),
}

_COLOR = {
    "baseline": "#2C3E50",
    "synib_star": "#FFBF00",
    "synib_random": "#3498DB",
    "synib_learned": "#E74C3C",
    "synib_diffusion": "#8E44AD",
}
_LABEL = {
    "baseline": "Baseline",
    "synib_star": r"SynIB $M^*$",
    "synib_random": r"SynIB $M_{\rm Random}$",
    "synib_learned": r"SynIB $M_{\rm Learned}$",
    "synib_diffusion": r"SynIB $M_{\rm Diffusion}$",
}
_STYLE = {
    "baseline": ("-", None),
    "synib_star": ("-", None),
    "synib_random": ("--", "o"),
    "synib_learned": ("--", "s"),
    "synib_diffusion": ("--", "D"),
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class HighDimXORWithTrainSpurious(Dataset):
    def __init__(
        self,
        n: int,
        dim0: int = 64,
        dim1: int = 64,
        signal_dims: int = 4,
        spur_dims: int = 6,
        alpha: float = 0.0,
        mode: str = "train",
        seed: int = 0,
        signal_scale: float = 2.2,
        signal_noise_std: float = 0.55,
        spur_scale: float = 1.3,
        base_noise_std: float = 1.0,
    ):
        super().__init__()
        assert mode in ("train", "test")
        assert signal_dims + spur_dims <= dim0

        g = torch.Generator().manual_seed(seed)
        b0 = torch.randint(0, 2, (n,), generator=g).float()
        b1 = torch.randint(0, 2, (n,), generator=g).float()
        y = (b0 != b1).float()

        x0 = torch.randn(n, dim0, generator=g) * base_noise_std
        x1 = torch.randn(n, dim1, generator=g) * base_noise_std
        sig0 = slice(0, signal_dims)
        sig1 = slice(0, signal_dims)
        spur = slice(signal_dims, signal_dims + spur_dims)

        b0_sign = (2 * b0 - 1).unsqueeze(1)
        b1_sign = (2 * b1 - 1).unsqueeze(1)
        x0[:, sig0] = signal_scale * b0_sign + torch.randn(
            n, signal_dims, generator=g
        ) * signal_noise_std
        x1[:, sig1] = signal_scale * b1_sign + torch.randn(
            n, signal_dims, generator=g
        ) * signal_noise_std

        y_sign = (2 * y - 1).unsqueeze(1)
        alpha_eff = alpha if mode == "train" else 0.0
        eps = torch.randn(n, spur_dims, generator=g)
        noise_scale = max(0.0, 1.0 - alpha_eff**2) ** 0.5
        s = alpha_eff * y_sign + noise_scale * eps
        x0[:, spur] = spur_scale * s

        self.x0, self.x1, self.y = x0, x1, y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return {"data": {0: self.x0[idx], 1: self.x1[idx]}, "y": self.y[idx]}


class FusionModel(nn.Module):
    def __init__(self, dim0: int = 64, dim1: int = 64, hidden: int = 16, dropout: float = 0.0):
        super().__init__()
        self.enc0 = nn.Sequential(
            nn.Linear(dim0, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.enc1 = nn.Sequential(
            nn.Linear(dim1, hidden), nn.ReLU(), nn.Dropout(dropout)
        )
        self.u0_head = nn.Linear(hidden, 1)
        self.u1_head = nn.Linear(hidden, 1)
        self.fuse = nn.Sequential(
            nn.Linear(2 * hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward_logits(
        self, x0: torch.Tensor, x1: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h0 = self.enc0(x0)
        h1 = self.enc1(x1)
        u0 = self.u0_head(h0).squeeze(-1)
        u1 = self.u1_head(h1).squeeze(-1)
        f = self.fuse(torch.cat([h0, h1], dim=-1)).squeeze(-1)
        return f, u0, u1

    def forward(self, x: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f, u0, u1 = self.forward_logits(x[0], x[1])
        return torch.sigmoid(f), torch.sigmoid(u0), torch.sigmoid(u1)


def bce(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, y)


def kl_to_uniform(logit: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.sigmoid(logit).clamp(eps, 1.0 - eps)
    return (p * torch.log(2.0 * p) + (1.0 - p) * torch.log(2.0 * (1.0 - p))).mean()


def base_loss(
    model: FusionModel, x0: torch.Tensor, x1: torch.Tensor, y: torch.Tensor, lambda_uni: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    f, u0, u1 = model.forward_logits(x0, x1)
    return f, u0, u1, bce(f, y) + lambda_uni * (bce(u0, y) + bce(u1, y))


def apply_destroy(x: torch.Tensor, g: torch.Tensor, noise_std: float = 1.0) -> torch.Tensor:
    if g.dim() == 1:
        g = g.view(1, -1)
    g = g.to(x.device).type_as(x)
    return (1.0 - g) * x + g * (torch.randn_like(x) * noise_std)


def make_oracle_counterfactual(x: torch.Tensor, signal_dims: int, noise_std: float) -> torch.Tensor:
    x_t = x.clone()
    x_t[:, :signal_dims] = torch.randn_like(x_t[:, :signal_dims]) * noise_std
    return x_t


def _cosine_alpha_bar(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    ft = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    f0 = math.cos((s / (1 + s)) * (math.pi / 2)) ** 2
    return ft / f0


def _linear_alpha_bar(t: torch.Tensor, alpha_bar_start: float = 1.0, alpha_bar_end: float = 0.02) -> torch.Tensor:
    return alpha_bar_start + (alpha_bar_end - alpha_bar_start) * t


def _alpha_bar(t: torch.Tensor, schedule: str) -> torch.Tensor:
    if schedule == "cosine":
        return _cosine_alpha_bar(t)
    if schedule == "linear":
        return _linear_alpha_bar(t)
    raise ValueError(f"Unknown diffusion schedule: {schedule}")


def make_diffusion_counterfactual(
    x: torch.Tensor,
    signal_dims: int,
    noise_strength: float,
    schedule: str,
    mask_prob: float,
    eps_std: float,
) -> torch.Tensor:
    x_t = x.clone()
    B = x.shape[0]
    t = torch.full((B, 1), float(max(0.0, min(1.0, noise_strength))), device=x.device, dtype=x.dtype)
    alpha_bar = _alpha_bar(t, schedule).clamp(0.0, 1.0)
    x_comp = x_t[:, :signal_dims]
    eps = torch.randn_like(x_comp) * eps_std
    x_noised = torch.sqrt(alpha_bar) * x_comp + torch.sqrt(1.0 - alpha_bar) * eps
    if mask_prob >= 1.0:
        x_t[:, :signal_dims] = x_noised
    else:
        m = (torch.rand((B, signal_dims), device=x.device) < mask_prob).type_as(x_comp)
        x_t[:, :signal_dims] = m * x_noised + (1.0 - m) * x_comp
    return x_t


def learn_one_gate(
    model: FusionModel,
    x0: torch.Tensor,
    x1: torch.Tensor,
    y: torch.Tensor,
    which: str,
    params: MethodParams,
    cfg: Cfg,
) -> torch.Tensor:
    B = x0.shape[0]
    d = x0.shape[1] if which == "x0" else x1.shape[1]
    ell = nn.Parameter(torch.full((B, d), -4.0, device=x0.device))
    opt_gate = torch.optim.Adam([ell], lr=params.mask_lr)
    for _ in range(params.mask_steps):
        g = torch.sigmoid(ell / cfg.mask_tau)
        if which == "x0":
            x_t = apply_destroy(x0, g, cfg.mask_noise_std)
            _, u_t, _ = model.forward_logits(x_t, x1)
        else:
            x_t = apply_destroy(x1, g, cfg.mask_noise_std)
            _, _, u_t = model.forward_logits(x0, x_t)
        obj = -bce(u_t, y) + params.mask_lam_sparsity * g.mean()
        opt_gate.zero_grad()
        obj.backward()
        opt_gate.step()
    with torch.no_grad():
        g = torch.sigmoid(ell / cfg.mask_tau)
        if cfg.mask_hard:
            g = (g > cfg.mask_hard_thresh).float()
    return g.detach()


@torch.no_grad()
def eval_acc(model: FusionModel, loader: DataLoader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x0 = batch["data"][0].to(device)
        x1 = batch["data"][1].to(device)
        y = batch["y"].to(device)
        f_prob, _, _ = model({0: x0, 1: x1})
        correct += ((f_prob > 0.5).float() == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / max(total, 1)


def _make_loaders(alpha: float, seed: int, cfg: Cfg, include_val: bool = False):
    ds_kwargs = dict(
        dim0=cfg.dim0,
        dim1=cfg.dim1,
        signal_dims=cfg.signal_dims,
        spur_dims=cfg.spur_dims,
        alpha=alpha,
        signal_scale=cfg.signal_scale,
        signal_noise_std=cfg.signal_noise_std,
        spur_scale=cfg.spur_scale,
        base_noise_std=cfg.base_noise_std,
    )
    train_ds = HighDimXORWithTrainSpurious(cfg.n_train, mode="train", seed=seed + 101, **ds_kwargs)
    test_ds = HighDimXORWithTrainSpurious(cfg.n_test, mode="test", seed=seed + 201, **ds_kwargs)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False)
    if not include_val:
        return train_loader, test_loader
    val_ds = HighDimXORWithTrainSpurious(cfg.n_val, mode="train", seed=seed + 151, **ds_kwargs)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)
    return train_loader, val_loader, test_loader


def _build_model(cfg: Cfg, params: MethodParams) -> FusionModel:
    return FusionModel(cfg.dim0, cfg.dim1, cfg.hidden, dropout=params.dropout).to(cfg.device)


def _build_optimizer(model: FusionModel, params: MethodParams) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=params.lr, weight_decay=params.wd)


def _train_one_method(
    method: str,
    alpha: float,
    seed: int,
    cfg: Cfg,
    params: MethodParams,
    include_val: bool = False,
) -> Dict[str, float]:
    set_seed(seed)
    if include_val:
        train_loader, val_loader, test_loader = _make_loaders(alpha, seed, cfg, include_val=True)
    else:
        train_loader, test_loader = _make_loaders(alpha, seed, cfg, include_val=False)
        val_loader = None
    model = _build_model(cfg, params)
    opt = _build_optimizer(model, params)

    anneal = int(getattr(params, "l_anneal_epochs", 0))
    for ep in range(cfg.epochs):
        l_eff = params.lambda_kl * min(1.0, (ep + 1) / anneal) if anneal > 0 else params.lambda_kl
        model.train()
        for batch in train_loader:
            x0 = batch["data"][0].to(cfg.device)
            x1 = batch["data"][1].to(cfg.device)
            y = batch["y"].to(cfg.device)
            _, _, _, l_base = base_loss(model, x0, x1, y, cfg.lambda_uni)

            if method == "baseline":
                loss = l_base
            elif method == "synib_star":
                x0_t = make_oracle_counterfactual(x0, cfg.signal_dims, cfg.comp_noise_std)
                x1_t = make_oracle_counterfactual(x1, cfg.signal_dims, cfg.comp_noise_std)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                loss = l_base + params.lambda_kl * (kl_to_uniform(f_t0) + kl_to_uniform(f_t1))
            elif method == "synib_random":
                B = x0.size(0)
                g0 = torch.bernoulli(torch.full((B, cfg.dim0), params.mask_prob, device=cfg.device))
                g1 = torch.bernoulli(torch.full((B, cfg.dim1), params.mask_prob, device=cfg.device))
                x0_t = apply_destroy(x0, g0, cfg.comp_noise_std)
                x1_t = apply_destroy(x1, g1, cfg.comp_noise_std)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                loss = l_base + params.lambda_kl * (kl_to_uniform(f_t0) + kl_to_uniform(f_t1))
            elif method == "synib_learned":
                # Inv-outer (matches SynIB-invmask design): inner finds the minimum
                # destroy mask that breaks the unimodal head; outer treats that mask as
                # the KEEP set and destroys its complement. This forces the joint off
                # any feature the unimodal can use, including the spurious shortcut.
                with torch.enable_grad():
                    g0 = learn_one_gate(model, x0.detach(), x1.detach(), y.detach(), "x0", params, cfg)
                    g1 = learn_one_gate(model, x0.detach(), x1.detach(), y.detach(), "x1", params, cfg)
                x0_t = apply_destroy(x0, 1.0 - g0, cfg.mask_noise_std)
                x1_t = apply_destroy(x1, 1.0 - g1, cfg.mask_noise_std)
                f_t0, _, _ = model.forward_logits(x0_t, x1)
                f_t1, _, _ = model.forward_logits(x0, x1_t)
                loss = l_base + l_eff * (kl_to_uniform(f_t0) + kl_to_uniform(f_t1))
            elif method == "synib_diffusion":
                l_cf = 0.0
                for _ in range(params.diffusion_steps):
                    x0_t = make_diffusion_counterfactual(
                        x0,
                        cfg.signal_dims,
                        noise_strength=1.0,
                        schedule=params.diffusion_schedule,
                        mask_prob=params.diffusion_mask_prob,
                        eps_std=params.diffusion_eps_std,
                    )
                    x1_t = make_diffusion_counterfactual(
                        x1,
                        cfg.signal_dims,
                        noise_strength=1.0,
                        schedule=params.diffusion_schedule,
                        mask_prob=params.diffusion_mask_prob,
                        eps_std=params.diffusion_eps_std,
                    )
                    f_t0, _, _ = model.forward_logits(x0_t, x1)
                    f_t1, _, _ = model.forward_logits(x0, x1_t)
                    l_cf = l_cf + kl_to_uniform(f_t0) + kl_to_uniform(f_t1)
                loss = l_base + params.lambda_kl * (l_cf / float(params.diffusion_steps))
            else:
                raise ValueError(f"Unknown method: {method}")

            opt.zero_grad()
            loss.backward()
            opt.step()

    out = {
        "train_acc": eval_acc(model, train_loader, cfg.device),
        "test_acc": eval_acc(model, test_loader, cfg.device),
    }
    if val_loader is not None:
        out["val_acc"] = eval_acc(model, val_loader, cfg.device)
    return out


def _parameter_grid(method: str) -> Iterable[MethodParams]:
    space = SEARCH_SPACES[method]
    base = ARCHIVED_BEST[method]
    for lr, wd, dropout in product(space.lr or [base.lr], space.wd or [base.wd], space.dropout or [base.dropout]):
        common = dict(lr=lr, wd=wd, dropout=dropout)
        if method == "baseline":
            yield MethodParams(**common, lambda_kl=0.0)
        elif method == "synib_star":
            for lambda_kl in space.lambda_kl or [base.lambda_kl]:
                yield MethodParams(**common, lambda_kl=lambda_kl)
        elif method == "synib_random":
            for lambda_kl, mask_prob in product(space.lambda_kl or [base.lambda_kl], space.mask_prob or [base.mask_prob]):
                yield MethodParams(**common, lambda_kl=lambda_kl, mask_prob=mask_prob)
        elif method == "synib_learned":
            for lambda_kl, mask_steps, mask_lam_sparsity in product(
                space.lambda_kl or [base.lambda_kl],
                space.mask_steps or [base.mask_steps],
                space.mask_lam_sparsity or [base.mask_lam_sparsity],
            ):
                yield MethodParams(**common, lambda_kl=lambda_kl, mask_steps=mask_steps, mask_lam_sparsity=mask_lam_sparsity)
        elif method == "synib_diffusion":
            for lambda_kl, diffusion_steps in product(space.lambda_kl or [base.lambda_kl], space.diffusion_steps or [base.diffusion_steps]):
                yield MethodParams(**common, lambda_kl=lambda_kl, diffusion_steps=diffusion_steps)


def tune_method(
    method: str,
    alphas: List[float],
    seeds: List[int],
    cfg: Cfg,
    tune_epochs: int,
) -> MethodParams:
    tune_cfg = copy.deepcopy(cfg)
    tune_cfg.epochs = tune_epochs
    best_params = copy.deepcopy(ARCHIVED_BEST[method])
    best_score = -float("inf")
    candidates = list(_parameter_grid(method))
    print(f"Tuning {method} over {len(candidates)} candidates")
    for idx, params in enumerate(candidates, start=1):
        scores = []
        for alpha in alphas:
            for seed in seeds:
                result = _train_one_method(method, alpha, seed, tune_cfg, params, include_val=False)
                scores.append(result["test_acc"])
        score = float(np.mean(scores))
        print(f"  [{method}] candidate {idx:02d}/{len(candidates)} val_acc={score:.4f} params={asdict(params)}")
        if score > best_score:
            best_score = score
            best_params = copy.deepcopy(params)
    print(f"Selected {method}: val_acc={best_score:.4f} params={asdict(best_params)}")
    return best_params


def resolve_method_params(
    methods: List[str],
    alphas: List[float],
    seeds: List[int],
    cfg: Cfg,
    tuning_mode: str,
    tune_epochs: int,
) -> Dict[str, MethodParams]:
    params = {m: copy.deepcopy(ARCHIVED_BEST[m]) for m in methods}
    if tuning_mode == "best_tuned":
        return params
    for method in methods:
        params[method] = tune_method(method, alphas, seeds, cfg, tune_epochs)
    return params


def run_sweep(
    alphas: List[float],
    seeds: List[int],
    cfg: Cfg,
    method_params: Dict[str, MethodParams],
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    for method in METHODS:
        results[method] = {"params": asdict(method_params[method]), "metrics": {}}
        for alpha in alphas:
            train_accs, test_accs = [], []
            for seed in seeds:
                print(f"  [{method:16s}] alpha={alpha:4.1f} seed={seed}", flush=True)
                result = _train_one_method(method, alpha, seed, cfg, method_params[method], include_val=False)
                train_accs.append(result["train_acc"])
                test_accs.append(result["test_acc"])
            train_a = np.asarray(train_accs, dtype=float)
            test_a = np.asarray(test_accs, dtype=float)
            results[method]["metrics"][alpha] = {
                "train_mean": float(train_a.mean()),
                "train_std": float(train_a.std(ddof=1)) if len(train_a) > 1 else 0.0,
                "test_mean": float(test_a.mean()),
                "test_std": float(test_a.std(ddof=1)) if len(test_a) > 1 else 0.0,
            }
    return results


def make_plot(results: Dict[str, Dict[str, object]], out_dir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = sorted(next(iter(results.values()))["metrics"].keys())
    x = np.asarray(alphas, dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for method in METHODS:
        means = np.asarray([results[method]["metrics"][a]["test_mean"] for a in alphas], dtype=float)
        stds = np.asarray([results[method]["metrics"][a]["test_std"] for a in alphas], dtype=float)
        ls, mk = _STYLE[method]
        ax.plot(x, means, color=_COLOR[method], label=_LABEL[method], linestyle=ls, marker=mk, lw=2)
        ax.fill_between(
            x,
            np.clip(means - stds, 0.0, 1.0),
            np.clip(means + stds, 0.0, 1.0),
            color=_COLOR[method],
            alpha=0.12,
        )
    ax.set_xlabel(r"$\beta$ (spurious strength in training set)")
    ax.set_ylabel("Fusion test accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"xor_spurious_results.{ext}")
        fig.savefig(path, dpi=200)
        print(f"Saved: {path}")
    plt.close(fig)


def print_params_table(results: Dict[str, Dict[str, object]]) -> None:
    print("\nSelected hyperparameters")
    print("-" * 100)
    for method in METHODS:
        print(f"{method:16s} {results[method]['params']}")


def print_results_table(results: Dict[str, Dict[str, object]]) -> None:
    alphas = sorted(next(iter(results.values()))["metrics"].keys())
    col_w = 22
    header = f"{'alpha':>6} | " + " | ".join(f"{m:>{col_w}}" for m in METHODS)
    print("\n" + header)
    print("-" * len(header))
    for alpha in alphas:
        row = f"{alpha:>6.1f} | "
        row += " | ".join(
            f"{results[m]['metrics'][alpha]['test_mean'] * 100:7.2f}% ±{results[m]['metrics'][alpha]['test_std'] * 100:5.2f}%"
            for m in METHODS
        )
        print(row)


def save_results(results: Dict[str, Dict[str, object]], out_dir: str, meta: Dict[str, object]) -> None:
    import json

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "xor_spurious_results.json")
    payload = {"meta": meta, "results": results}
    with open(path, "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="XOR spurious shortcut experiment with method-specific tuning")
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n_train", type=int, default=300)
    parser.add_argument("--n_val", type=int, default=120)
    parser.add_argument("--n_test", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--tune_epochs", type=int, default=None)
    parser.add_argument("--tuning_mode", choices=["best_tuned", "retune"], default="best_tuned")
    parser.add_argument("--out_dir", type=str, default=".")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Cfg(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        epochs=args.epochs,
        device=args.device,
    )

    print(f"Device      : {cfg.device}")
    print(f"Alphas      : {args.alphas}")
    print(f"Seeds       : {args.seeds}")
    print(f"Epochs      : {cfg.epochs}")
    print(f"Tuning mode : {args.tuning_mode}")

    tune_epochs = args.tune_epochs if args.tune_epochs is not None else args.epochs
    method_params = resolve_method_params(METHODS, args.alphas, args.seeds, cfg, args.tuning_mode, tune_epochs)
    results = run_sweep(args.alphas, args.seeds, cfg, method_params)
    print_params_table(results)
    print_results_table(results)
    make_plot(results, args.out_dir)
    save_results(
        results,
        args.out_dir,
        {
            "alphas": args.alphas,
            "seeds": args.seeds,
            "epochs": cfg.epochs,
            "tuning_mode": args.tuning_mode,
            "tune_epochs": args.tune_epochs,
        },
    )


if __name__ == "__main__":
    main()
