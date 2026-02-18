import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from synib.models.vlm.qwen_synergy_decoder_models import *
# ============================================================
# Utilities
# ============================================================
import math
from typing import Any, Dict, List, Optional, Literal, Tuple
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
from synib.models.model_utils.backbone import resnet18

def _cfg(args, key, default=None):
    """Support args as dict-like or namespace-like."""
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _as_tensor_features(enc_out):
    """
    Accept either:
      - tensor (B,d)
      - dict with enc_out["features"]["combined"] (B,d)
      - dict with enc_out["combined"] (B,d)
    """
    if torch.is_tensor(enc_out):
        return enc_out
    if isinstance(enc_out, dict):
        features, non_aggr_features = None, None
        if "features" in enc_out and isinstance(enc_out["features"], dict) and "combined" in enc_out["features"]:
            features = enc_out["features"]["combined"]
        if "nonaggr_features" in enc_out and isinstance(enc_out["nonaggr_features"], dict) and "combined" in enc_out["nonaggr_features"]:
            non_aggr_features = enc_out["nonaggr_features"]["combined"]
        return features, non_aggr_features
    raise ValueError("Encoder output must be a Tensor or a dict with ['features']['combined'].")

def _as_tensor_preds(enc_out):
    """
    Accept either:
      - tensor (B,d)
      - dict with enc_out["features"]["combined"] (B,d)
      - dict with enc_out["combined"] (B,d)
    """
    if torch.is_tensor(enc_out):
        return enc_out
    if isinstance(enc_out, dict):
        preds = None
        if "preds" in enc_out and isinstance(enc_out["preds"], dict) and "combined" in enc_out["preds"]:
            preds = enc_out["preds"]["combined"]
        return preds
    raise ValueError("Encoder output must be a Tensor or a dict with ['preds']['combined'].")


def _kl_normal(mu, logvar):
    """KL(q||p) with q=N(mu,diag(exp(logvar))) and p=N(0,I)."""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


def _reparam(mu, logvar):
    std = (0.5 * logvar).exp()
    eps = torch.randn_like(std)
    return mu + eps * std


# ============================================================
# Encoders / Blocks (ALL with __init__(args, encs, **kwargs))
# ============================================================

# class SVAEGenerator(nn.Module):
#     """
#     VAE-based perturbation generator: z -> z_tilde
#     (use as SynIB.gen_x1 or SynIB.gen_x2)
#
#     args:
#       d_z, d_hidden, d_latent, dropout
#       rec_loss: "mse" (default)
#     """
#     def __init__(self, args, encs=None, **kwargs):
#         super().__init__()
#         d_z = int(_cfg(args, "d_z"))
#         d_hidden = int(_cfg(args, "d_hidden"))
#         d_latent = int(_cfg(args, "d_latent"))
#         dropout = float(_cfg(args, "dropout", 0.0))
#
#         # You can optionally pass [posterior, decoder] in encs for modular saving/loading
#         if encs is not None and len(encs) == 2:
#             self.posterior = encs[0]
#             self.decoder = encs[1]
#         else:
#             self.posterior = VAEPosterior(
#                 {"d_z": d_z, "d_hidden": d_hidden, "d_latent": d_latent, "dropout": dropout},
#                 encs=None,
#             )
#             self.decoder = VAEDecoder(
#                 {"d_latent": d_latent, "d_hidden": d_hidden, "d_z": d_z, "dropout": dropout},
#                 encs=None,
#             )
#
#         self.rec_loss = str(_cfg(args, "rec_loss", "mse")).lower()
#
#     def forward(self, zt, zc, compute_loss: bool = False, add_noise: bool = False, **kwargs):
#         # zc is passed through only for API compatibility with other generators
#         mu, logvar = self.posterior(zt)
#         u = _reparam(mu, logvar)
#         if add_noise:
#             u = u + 0.01 * torch.randn_like(u)
#
#         z_tilde = self.decoder(u)
#
#         out = {"tilde": z_tilde, "z_c": zc, "mu": mu, "logvar": logvar}
#
#         if compute_loss:
#             if self.rec_loss == "mse":
#                 rec = F.mse_loss(z_tilde, zt, reduction="mean")
#             else:
#                 raise ValueError(f"Unknown rec_loss={self.rec_loss}")
#
#             kl = _kl_normal(mu, logvar)
#             out["losses"] = {"rec": rec, "kl": kl}
#
#         return out

class FusionTrunkLinear(nn.Module):
    """
    enc_2: z1,z2 -> feat
    args must contain: d_model, fc_inner, dropout
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_model = int(_cfg(args, "d_model"))
        fc_inner = int(_cfg(args, "fc_inner"))
        dropout = float(_cfg(args, "dropout", 0.1))

        self.net = nn.Sequential(
            nn.Linear(2 * d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, z1, z2):
        return self.net(torch.cat([z1, z2], dim=1))


class FusionConformer(nn.Module):
    """
    enc_2: z1,z2 -> feat
    args must contain: d_model, fc_inner, dropout
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_model = int(_cfg(args, "d_model"))
        fc_inner = int(_cfg(args, "fc_inner"))
        dropout = float(_cfg(args, "dropout", 0.1))

        self.net = nn.Sequential(
            nn.Linear(2 * d_model, fc_inner),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, z1, z2):
        return self.net(torch.cat([z1, z2], dim=1))


class LinearHead(nn.Module):
    """
    enc_3: feat -> logits
    args must contain: in_dim, num_classes
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        in_dim = args.get("fc_inner",64)
        num_classes = args.get("num_classes")
        self.proj = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.proj(x)


class MLPHead(nn.Module):
    """
    enc_4 / enc_5: z -> logits
    args must contain: in_dim, hidden_dim, num_classes, dropout
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        in_dim = int(_cfg(args, "in_dim"))
        hidden_dim = int(_cfg(args, "hidden_dim"))
        num_classes = int(_cfg(args, "num_classes"))
        dropout = float(_cfg(args, "dropout", 0.1))

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z)


class VAEPosterior(nn.Module):
    """
    enc_6 / enc_8: z -> (mu, logvar)
    args must contain: d_z, d_hidden, d_latent, dropout
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_z = int(_cfg(args, "d_z"))
        d_hidden = int(_cfg(args, "d_hidden"))
        d_latent = int(_cfg(args, "d_latent"))
        dropout = float(_cfg(args, "dropout", 0.0))

        self.backbone = nn.Sequential(
            nn.Linear(d_z, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)

    def forward(self, z):
        h = self.backbone(z)
        return self.mu(h), self.logvar(h)
class VAEDecoder(nn.Module):
    """
    enc_7 / enc_9: u -> z_tilde
    args must contain: d_latent, d_hidden, d_z, dropout
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_latent = int(_cfg(args, "d_latent"))
        d_hidden = int(_cfg(args, "d_hidden"))
        d_z = int(_cfg(args, "d_z"))
        dropout = float(_cfg(args, "dropout", 0.0))

        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_z),
        )

    def forward(self, u):
        x = self.net(u)
        x = F.layer_norm(x, (x.shape[-1],))
        return x


class _MLPBlock(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.0, act="gelu", norm=True):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out) if norm else None
        self.drop = nn.Dropout(dropout)
        if act == "gelu":
            self.act = nn.GELU()
        elif act == "relu":
            self.act = nn.ReLU()
        elif act == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown act={act}")

    def forward(self, x):
        x = self.fc(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        return x
class VAEPosterior_XL(nn.Module):
    """
    Bigger posterior: z -> (mu, logvar)

    args:
      d_z, d_hidden, d_latent, dropout
      n_layers: int (default 4)
      act: "gelu"|"relu"|"silu" (default "gelu")
      norm: bool (default True)
      residual: bool (default True)  # residual between hidden layers
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_z = int(_cfg(args, "d_z"))
        d_hidden = int(_cfg(args, "d_hidden"))
        d_latent = int(_cfg(args, "d_latent"))
        dropout = float(_cfg(args, "dropout", 0.0))
        n_layers = int(_cfg(args, "n_layers", 4))
        act = str(_cfg(args, "act", "gelu")).lower()
        norm = bool(_cfg(args, "norm", True))
        residual = bool(_cfg(args, "residual", True))

        self.inp = _MLPBlock(d_z, d_hidden, dropout=dropout, act=act, norm=norm)
        self.hid = nn.ModuleList([
            _MLPBlock(d_hidden, d_hidden, dropout=dropout, act=act, norm=norm)
            for _ in range(max(0, n_layers - 1))
        ])
        self.residual = residual

        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)

        # mild init that helps avoid crazy logvars early
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, z):
        h = self.inp(z)
        for blk in self.hid:
            h2 = blk(h)
            h = (h + h2) if self.residual else h2
        return self.mu(h), self.logvar(h)
class VAEDecoder_XL(nn.Module):
    """
    Bigger decoder: u -> z_out  OR  u -> delta (if delta_mode=True)

    args:
      d_latent, d_hidden, d_z, dropout
      n_layers: int (default 4)
      act: "gelu"|"relu"|"silu" (default "gelu")
      norm: bool (default True)
      out_norm: "layernorm"|"none" (default "layernorm")
      delta_mode: bool (default False)   # if True, returns bounded delta instead of full z
      delta_scale: float (default 1.0)   # scale for delta_mode
      residual: bool (default True)
    """
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_latent = int(_cfg(args, "d_latent"))
        d_hidden = int(_cfg(args, "d_hidden"))
        d_z = int(_cfg(args, "d_z"))
        dropout = float(_cfg(args, "dropout", 0.0))
        n_layers = int(_cfg(args, "n_layers", 4))
        act = str(_cfg(args, "act", "gelu")).lower()
        norm = bool(_cfg(args, "norm", True))
        out_norm = str(_cfg(args, "out_norm", "layernorm")).lower()
        self.delta_mode = bool(_cfg(args, "delta_mode", False))
        self.delta_scale = float(_cfg(args, "delta_scale", 1.0))
        residual = bool(_cfg(args, "residual", True))

        self.inp = _MLPBlock(d_latent, d_hidden, dropout=dropout, act=act, norm=norm)
        self.hid = nn.ModuleList([
            _MLPBlock(d_hidden, d_hidden, dropout=dropout, act=act, norm=norm)
            for _ in range(max(0, n_layers - 1))
        ])
        self.residual = residual

        self.out = nn.Linear(d_hidden, d_z)
        self.out_ln = nn.LayerNorm(d_z) if out_norm == "layernorm" else None

    def forward(self, u):
        h = self.inp(u)
        for blk in self.hid:
            h2 = blk(h)
            h = (h + h2) if self.residual else h2

        x = self.out(h)

        # If you want "small, controlled edits", use delta_mode and add outside: z_tilde = z + delta
        if self.delta_mode:
            x = self.delta_scale * torch.tanh(x)  # bounded delta

        if self.out_ln is not None:
            x = self.out_ln(x)

        return x


class VAEPosterior_TF(nn.Module):
    def __init__(self, args, encs=[]):
        super().__init__()
        d_z = int(_cfg(args, "d_z"))
        d_hidden = int(_cfg(args, "d_hidden"))  # Must be divisible by nhead
        d_latent = int(_cfg(args, "d_latent"))
        n_layers = int(_cfg(args, "n_layers", 4))
        n_heads = int(_cfg(args, "n_heads", 8))
        dropout = float(_cfg(args, "dropout", 0.1))

        # Project input to the Transformer dimension
        self.proj_in = nn.Linear(d_z, d_hidden)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_hidden))
        self.pos_embed = nn.Parameter(torch.randn(1, 150 + 1, d_hidden))

        # Native PyTorch Transformer Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.mu = nn.Linear(d_hidden, d_latent)
        self.logvar = nn.Linear(d_hidden, d_latent)

        # Init logvar to zeros to avoid unstable KL early on
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, z, na_z=None):
        if na_z is not None:
            h = self.proj_in(z).unsqueeze(1)
        else:
            h = self.proj_in(na_z.permute(0, 2, 1))

        h = h + self.pos_embed[:, :h.size(1), :]
        h = self.transformer(h)

        return self.mu(h), self.logvar(h)
class VAEDecoder_TF(nn.Module):
    def __init__(self, args, encs=[]):
        super().__init__()
        d_latent = int(_cfg(args, "d_latent"))
        d_hidden = int(_cfg(args, "d_hidden"))
        d_z = int(_cfg(args, "d_z"))
        n_layers = int(_cfg(args, "n_layers", 4))
        n_heads = int(_cfg(args, "n_heads", 8))
        self.delta_mode = bool(_cfg(args, "delta_mode", False))
        self.delta_scale = float(_cfg(args, "delta_scale", 1.0))

        self.proj_in = nn.Linear(d_latent, d_hidden)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_hidden,
            nhead=n_heads,
            dim_feedforward=d_hidden * 4,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=n_layers)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_hidden))
        self.pos_embed = nn.Parameter(torch.randn(1, 150 + 1, d_hidden))

        self.out = nn.Linear(d_hidden, d_z)
        self.out_agg = nn.Linear(d_hidden, d_z)
        self.out_ln = nn.LayerNorm(d_z)

    def forward(self, u):
        h = self.proj_in(u)

        cls_tokens = self.cls_token.expand(h.shape[0], -1, -1)
        h = torch.cat((cls_tokens, h), dim=1)
        h = h + self.pos_embed[:, :h.size(1), :]

        hout = self.transformer(h)

        x = self.out(hout[:, 0, :])
        x_nonaggr = self.out(hout)
        if self.delta_mode:
            x = self.delta_scale * torch.tanh(x)

        return self.out_ln(x), self.out_ln(x_nonaggr)

class SynIB_VAE(nn.Module):
    """
    Same interface as SynIB, but uses pretrained VAEs as skew samplers.
    IMPORTANT: VAEs are used under torch.no_grad(); no VAE losses are returned.

    Expected encs layout:
      encs[0] = modality-1 encoder (main model uses)
      encs[1] = modality-2 encoder (main model uses)
      encs[2] = VAE posterior for z1
      encs[3] = VAE decoder   for z1
      encs[4] = VAE posterior for z2
      encs[5] = VAE decoder   for z2
    """
    def __init__(self, args, encs, main):
        super().__init__()
        object.__setattr__(self, "main", main)

        self.perturb = _cfg(args, "perturb", {}) or {}
        self.reestimate_features = bool(self.perturb.get("reestimate_features", False))

        bias = _cfg(args, "bias_infusion", {}) or {}
        self.synergy_weight = float(bias.get("l", 0.0))
        self.contrastive_weight = float(bias.get("contrcoeff", 0.0) or 0.0)
        self.synergy_type = getattr(args, "synergy_type", "gaussian")  # "gaussian" or "dirichlet"

        vae_w = _cfg(args, "vae_w", {}) or {}
        self.w_rec = vae_w.get("rec", 1.0)
        self.w_erase = vae_w.get("erase", 1.0)
        self.w_prior_kl = vae_w.get("prior_kl", 0.01)
        self.temperature = vae_w.get("temperature", 1.0)

        fc_inner = int(_cfg(args, "fc_inner"))
        num_classes = int(_cfg(args, "num_classes"))

        if self.synergy_type == "gaussian":
            self.logvar_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = None
        elif self.synergy_type == "dirichlet":
            self.evidence_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = float(_cfg(args, "dirichlet_prior_conc", 1.0))
        else:
            raise ValueError(f"Unknown synergy_type: {self.synergy_type}")

        # pretrained VAEs (samplers)
        self.vae_z1_post = encs[0]
        self.vae_z1_dec  = encs[1]
        self.vae_z2_post = encs[2]
        self.vae_z2_dec  = encs[3]

        self.has_vae_z1 = (self.vae_z1_post is not None) and (self.vae_z1_dec is not None)
        self.has_vae_z2 = (self.vae_z2_post is not None) and (self.vae_z2_dec is not None)

    # ------------------ KL utils ------------------

    @staticmethod
    def _gaussian_kl(mu, logvar):
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, dim=1).mean()

    @staticmethod
    def _dirichlet_kl(alpha, prior_conc=1.0):
        alpha0 = torch.full_like(alpha, prior_conc) if isinstance(prior_conc, float) else prior_conc
        alpha0_sum = alpha0.sum(dim=1, keepdim=True)
        alpha_sum = alpha.sum(dim=1, keepdim=True)

        lgamma = torch.lgamma
        digamma = torch.digamma

        logB_alpha = torch.sum(lgamma(alpha), dim=1) - lgamma(alpha_sum.squeeze(1))
        logB_alpha0 = torch.sum(lgamma(alpha0), dim=1) - lgamma(alpha0_sum.squeeze(1))

        term1 = logB_alpha0 - logB_alpha
        term2 = torch.sum((alpha - alpha0) * (digamma(alpha) - digamma(alpha_sum)), dim=1)
        return (term1 + term2).mean()

    def _encode_and_perturb(self, x, features, px1, px2, **kwargs):
        self.main.eval()

        # if self.reestimate_features:
        #     a, v = self.main._get_features(x, **kwargs)
        #     z1, z2 = a["features"]["combined"], v["features"]["combined"]
        losses = {}
        if px1:
            out = self._perturb(features, "x1", **kwargs)
            z1, z2, y = out["tilde"], out["z_c"].detach(), out["ey"]
            losses.update(out.get("losses",{}))
        if px2:
            out = self._perturb(features, "x2", **kwargs)
            z2, z1, y = out["tilde"], out["z_c"].detach(), out["ey"]
            losses.update(out.get("losses",{}))


        self.main.train()
        return z1, z2, y, losses

    def _repeat_tensor(self, t, K):
        if t is None: return None
        return t.repeat_interleave(K, dim=0)

    def _logit_kl(self, pred_logits, target_logits):
        return F.kl_div(
            F.log_softmax(pred_logits, dim=-1),
            F.softmax(target_logits, dim=-1),
            reduction="batchmean"
        )

    def _perturb(self, features, direction, **kwargs):
        """
        Returns dict with:
          tilde: (K*B, D) if K>1 else (B, D)
          z_c:  (K*B, D) if K>1 else (B, D)   (expanded conditioning)
        """
        ptype = self.perturb.get("type", "none")
        if ptype != "svae":
            raise ValueError(f"SynIB_VAE expects perturb.type='svae', got {ptype}")

        K = int(self.perturb.get("num_samples", 1))
        deterministic = bool(self.perturb.get("deterministic", False))

        if direction == "x1":
            if not self.has_vae_z1:
                raise RuntimeError("Need pretrained VAE z1 (posterior+decoder) in encs[2:4].")
            post, dec = self.vae_z1_post, self.vae_z1_dec
            zt = features["z1"].detach()
            na_zt = features.get("na_z1", None).detach()
            na_zc = features.get("na_z2", None).detach()
            zc = features["z2"].detach()
        else:
            if not self.has_vae_z2:
                raise RuntimeError("Need pretrained VAE z2 (posterior+decoder) in encs[4:6].")
            post, dec = self.vae_z2_post, self.vae_z2_dec
            zt = features["z2"].detach()
            na_zt = features.get("na_z2", None).detach()
            na_zc = features.get("na_z1", None).detach()
            zc = features["z1"].detach()


        losses = {}

        mu, logvar = post(zt, na_zt)  # (B, L)
        if K > 1 and not deterministic:
            mu, logvar, zc, na_zt, na_zc, zt, label = [self._repeat_tensor(v, K) for v in [mu, logvar, zc, na_zt, na_zc, zt, kwargs["label"]]]
        u = mu if deterministic else _reparam(mu, logvar)
        tilde, na_tilde = dec(u)

        with torch.no_grad():
            if direction == "x1":
                fused_logits, _ = self.main._compute_logits(zt, zc, na_zt, na_zc)
                unimodal_target_logits = self.main._compute_logits_unimodal(zc, na_zc, direction="z2")
                other_unimodal_target_logits = self.main._compute_logits_unimodal(zt, na_zt, direction="z1")
            else:
                fused_logits, _ = self.main._compute_logits(zc, zt, na_zc, na_zt)
                unimodal_target_logits = self.main._compute_logits_unimodal(zc, na_zc, direction="z1")
                other_unimodal_target_logits = self.main._compute_logits_unimodal(zt, na_zt, direction="z2")

            ensemble_target_logits = 0.5 * (unimodal_target_logits + other_unimodal_target_logits)

        # cls_heads = [self.main.cls_1, self.main.cls_2, self.main.cls_joint]
        cls_heads = [self.main.enc_2, self.main.enc_3, self.main.enc_4, self.main.enc_5]
        for head in cls_heads:
            head.requires_grad_(False)
            head.eval()

        if direction == "x1":
            tilde_fused_logits, _ = self.main._compute_logits(tilde, zc, na_tilde, na_zc)
        else:
            tilde_fused_logits, _ = self.main._compute_logits(zc, tilde, na_zc, na_tilde)

        # 4. Shaping: Erasure & Ensemble Matching
        losses[f"mse_~fused_unimodal"] = F.mse_loss(tilde_fused_logits, unimodal_target_logits.detach()) * self.w_erase
        losses[f"kl_~fused_unimodal"] = self._logit_kl(tilde_fused_logits,
                                                       unimodal_target_logits.detach()) * self.w_erase

        losses["mse_~fused_ens"] = F.mse_loss(tilde_fused_logits, ensemble_target_logits.detach()) * self.w_erase
        losses["kl_~fused_ens"] = self._logit_kl(tilde_fused_logits, ensemble_target_logits.detach()) * self.w_erase

        # 5. Generative & Prior Losses
        losses[f"r_~{direction}_{direction}"] = F.mse_loss(tilde, zt.detach()) * self.w_rec
        losses["kl_u"] = self._gaussian_kl(mu, logvar) * self.w_prior_kl

        for head in cls_heads:
            head.requires_grad_(True)
            head.train()

        # --- Generative & Prior - --
        losses[f"r_~{direction}_{direction}"] = F.mse_loss(tilde, zt.detach(), reduction="mean") * self.w_rec
        losses["kl_u"] = self._gaussian_kl(mu, logvar) * self.w_prior_kl


        return {"tilde": tilde, "z_c": zc, "ey": label, "losses":losses}

    def _kl_loss(self, mu, feat):
        if self.synergy_type == "gaussian":
            logvar = self.logvar_head(feat)
            kl = self._gaussian_kl(mu, logvar)
        else:
            evidence = F.softplus(self.evidence_head(feat))
            alpha = evidence + 1.0
            kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
        return kl * self.synergy_weight

    def _kl_pass(self, x, features, px1, px2, **kwargs):

        z1, z2, y, losses = self._encode_and_perturb(x, features, px1, px2, **kwargs)
        mu, feat = self.main._compute_logits(z1, z2, features["na_z1"], features["na_z2"])
        kl_loss = self._kl_loss(mu, feat)
        ce_loss = F.cross_entropy(mu, y)

        losses.update({"kl_synergy": kl_loss, "ce_loss": ce_loss.detach()})
        return losses, z1, z2, y

    def compute_training_losses(self, x, base_output, **kwargs):
        z1, z2 = base_output["features"]["z1"], base_output["features"]["z2"]

        losses_1, tz1, ez2, ey1 = self._kl_pass(x, base_output["features"], px1=True,  px2=False, **kwargs)
        losses_2, ez1, tz2, ey2 = self._kl_pass(x, base_output["features"], px1=False, px2=True,  **kwargs)

        features = {"tz1": tz1, "tz2": tz2, "ez1": ez1, "ez2": ez2, "ey":ey1}

        # infonce = nt_xent_loss(z1, z2, temperature=1.0)

        if self.training:
            if "current_step" in kwargs:
                wandb.log({"sl1": losses_1, "sl2": losses_2}, step=kwargs.get("current_step", 0)+1)

        losses = {}
        losses.update({"sl1": losses_1})
        losses.update({"sl2": losses_2})
        # losses["sl_sqdiff"] = (kl1 - kl2).pow(2.0).mean() * self.synergy_weight*10000
        # losses["infonce"] = infonce * self.contrastive_weight
        return losses, features

class FusionIBModel_VAE(nn.Module):
    """
    Renamed version of your FusionIBModel:
      enc_0, enc_1 : modality encoders (same as before)
      enc_2        : fusion trunk (was common_fc_1)
      enc_3        : fusion projector (was common_fc_2)
      enc_4        : fusion head (was mu_head)
      synergy      : SynIB_VAE (or SynIB) using main=self
    """
    def __init__(self, args, encs):
        super().__init__()
        self.args = args
        self.cls_type = _cfg(args, "cls_type")
        self.norm_decision = _cfg(args, "norm_decision", False)

        self.num_classes = int(_cfg(args, "num_classes"))
        d_model = int(_cfg(args, "d_model"))
        fc_inner = int(_cfg(args, "fc_inner"))
        dropout = float(_cfg(args, "dropout", 0.1))

        self.synergy_weight = float(_cfg(args, "bias_infusion", {}).get("l", 0.0))

        # main encoders
        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        self.enc_2 = encs[2]  # mlp fusion trunk
        self.enc_3 = encs[3]  # cls projector
        self.enc_4 = encs[4]  # unimodal cls head
        self.enc_5 = encs[5]  # unimodal cls head

        # # fusion trunk + projector
        # if self.cls_type == "mlp":
        #     self.cls_1 = nn.Sequential(
        #         nn.Linear(d_model * 2, d_model),
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(d_model, d_model),
        #     )
        #     self.cls_2 = nn.Sequential(
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(d_model, fc_inner),
        #     )
        # elif self.cls_type == "conformer":
        #     self.cls_1 = TF_Fusion(
        #         input_dim=d_model,
        #         dim=d_model,
        #         layers=6,
        #         output_dim=d_model,
        #     )
        #     self.cls_2 = nn.Sequential(
        #         nn.ReLU(),
        #         nn.Dropout(dropout),
        #         nn.Linear(d_model, fc_inner),
        #     )
        # else:
        #     raise ValueError("Unknown cls_type")

        # #add linear unimodal classifiers
        # self.cls_1_uni = nn.Linear(d_model, self.num_classes)
        # self.cls_2_uni = nn.Linear(d_model, self.num_classes)
        #
        # # fusion head
        # self.cls_joint = nn.Linear(fc_inner, self.num_classes)

        # synergy module: VAE sampler variant
        # (expects encs[2:6] = z1_post,z1_dec,z2_post,z2_dec)
        new_encs = [i for i in encs[6:]]
        self.synergy = SynIB_VAE(args, new_encs, main=self)

    # -------------------------
    # original interfaces kept
    # -------------------------
    def _get_features(self, x, **kwargs):
        out0 = self.enc_0(x, **kwargs)
        out1 = self.enc_1(x, **kwargs)
        z1, na_z1 = _as_tensor_features(out0)
        z2, na_z2 = _as_tensor_features(out1)
        preds1 = _as_tensor_preds(out0)
        preds2 = _as_tensor_preds(out1)
        z1 = F.layer_norm(z1, (z1.shape[-1],))
        z2 = F.layer_norm(z2, (z2.shape[-1],))
        na_z1 = F.layer_norm(na_z1, (na_z1.shape[-1],))
        na_z2 = F.layer_norm(na_z2, (na_z2.shape[-1],))
        return preds1, preds2, z1, z2, na_z1, na_z2

    def _compute_logits(self, z1, z2, na_z1, na_z2, **kwargs):
        # if self.cls_type == "mlp":
        #     aggr_feat = self.cls_1(torch.cat((z1, z2), dim=1))
        #     feat = self.cls_2(aggr_feat)
        #
        # elif self.cls_type == "conformer":
        #     pred_fused, aggr_feat, feat = self.cls_1(
        #         [na_z1, na_z2],
        #         return_all=True,
        #     )
        #     feat = self.cls_2(aggr_feat)
        #
        # pred = self.cls_joint(feat)
        # return pred, feat

        try:
            feat = self.enc_2(z1, z2)
        except TypeError:
            feat = self.enc_2(torch.cat([z1, z2], dim=1))
        logits = self.enc_3(feat)
        return logits, feat

    def _compute_logits_unimodal(self, z, na_z, direction, **kwargs):
        if direction == "z1":
            # pred = self.cls_1_uni(z.detach())
            pred = self.enc_4(z.detach())
        elif direction == "z2":
            # pred = self.cls_2_uni(z.detach())
            pred = self.enc_5(z.detach())

        return pred

    def _base_forward(self, x, **kwargs):
        uni_pred_1, uni_pred_2, z1, z2, na_z1, na_z2 = self._get_features(x, **kwargs)
        pred, feat = self._compute_logits(z1, z2, na_z1, na_z2)

        return {
            "preds": {
                "combined": pred,
                "c":uni_pred_1,
                "g":uni_pred_2
            },
            "features": {
                "z1": z1,
                "z2": z2,
                "na_z1":na_z1,
                "na_z2":na_z2
            },
            "losses": {},
        }

    def forward(self, x, **kwargs):
        output = self._base_forward(x, **kwargs)

        if self.synergy_weight > 0:
            synergy_losses, syn_features = self.synergy.compute_training_losses(x, base_output=output, **kwargs)
            output["features"].update(syn_features)
            output["losses"].update(synergy_losses)

        return output

class FusionIBModel_SVAE_Phased(nn.Module):
    """
    Phase-controlled model to:
      - warmup: train heads with CE
      - vae:    train one VAE side (z1 or z2)
      - after:  do nothing

    IMPORTANT: your trainer optimizes whatever is in output["losses"].
    This model returns ONLY {"loss": loss_tensor} inside "losses".

    Expected encs (by default enc_ids):
      enc_0: modality-1 encoder (pretrained)
      enc_1: modality-2 encoder (pretrained)
      enc_2: fusion trunk  (FusionTrunkLinear or your impl)
      enc_3: fusion head   (LinearHead)
      enc_4: unimodal head z1 (MLPHead)
      enc_5: unimodal head z2 (MLPHead)
      enc_6: VAE posterior z1 (VAEPosterior)
      enc_7: VAE decoder   z1 (VAEDecoder)
      enc_8: VAE posterior z2 (VAEPosterior)
      enc_9: VAE decoder   z2 (VAEDecoder)

    Config compatibility:
      - args.train_mode: "phased" | "warmup" | "vae" | "after"
      - args.schedule: { warmup_epochs, vae_start_epoch, vae_end_epoch, after_mode }
      - kwargs["current_epoch"] is provided by your trainer
      - kwargs["label"] is used in warmup (CE)
      - args.vae_train_side: "z1" or "z2"
      - args.multi_loss.multi_supervised_w: {"combined":.., "c":.., "g":..}
      - args.vae_loss: {z1_rec_w,z1_kl_w,z2_rec_w,z2_kl_w}
      - args.svae_shaping: {syn_kl_w,temp}  (optional; used in VAE phase)
    """

    def __init__(self, args, encs, **kwargs):
        super().__init__()
        self.args = args

        # mapping from name -> index in encs
        enc_ids = _cfg(args, "enc_ids", None)
        if enc_ids is None:
            enc_ids = {
                "enc_0": 0, "enc_1": 1, "enc_2": 2, "enc_3": 3, "enc_4": 4,
                "enc_5": 5, "enc_6": 6, "enc_7": 7, "enc_8": 8, "enc_9": 9,
            }

        def _get_enc(name):
            idx = enc_ids.get(name, None)
            if idx is None or idx >= len(encs):
                return None
            return encs[idx]

        # core encoders (required)
        self.enc_0 = _get_enc("enc_0")
        self.enc_1 = _get_enc("enc_1")
        if self.enc_0 is None or self.enc_1 is None:
            raise ValueError("FusionIBModel_SVAE_Phased requires enc_0 and enc_1.")

        # heads (expected provided by your config)
        self.enc_2 = _get_enc("enc_2")  # Fusion trunk
        self.enc_3 = _get_enc("enc_3")  # Fusion head
        self.enc_4 = _get_enc("enc_4")  # Unimodal head z1
        self.enc_5 = _get_enc("enc_5")  # Unimodal head z2
        if any(m is None for m in [self.enc_2, self.enc_3, self.enc_4, self.enc_5]):
            raise ValueError("Missing head blocks enc_2/enc_3/enc_4/enc_5 in encs. Provide them via config encoders.")

        # VAEs (optional, but required for VAE mode on that side)
        self.enc_6 = _get_enc("enc_6")  # posterior z1
        self.enc_7 = _get_enc("enc_7")  # decoder   z1
        self.enc_8 = _get_enc("enc_8")  # posterior z2
        self.enc_9 = _get_enc("enc_9")  # decoder   z2

        self.has_vae_z1 = (self.enc_6 is not None) and (self.enc_7 is not None)
        self.has_vae_z2 = (self.enc_8 is not None) and (self.enc_9 is not None)

        self.vae_train_side = str(_cfg(args, "vae_train_side", "z1")).lower()
        self.train_mode = str(_cfg(args, "train_mode", "phased")).lower()
        self.schedule = _cfg(args, "schedule", {}) or {}

        self._last_phase = None  # one-time freeze/unfreeze

    # -------------------------
    # freeze helpers
    # -------------------------
    @staticmethod
    def freeze_(module: nn.Module, flag: bool = True):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = (not flag)

    def _phase(self, epoch: int) -> str:
        # explicit override
        if self.train_mode in {"warmup", "vae", "after"}:
            return self.train_mode

        # phased schedule
        warmup_epochs = int(self.schedule.get("warmup_epochs", 0))
        vae_start = int(self.schedule.get("vae_start_epoch", warmup_epochs))
        vae_end = int(self.schedule.get("vae_end_epoch", vae_start))

        if epoch < warmup_epochs:
            return "warmup"
        if vae_start <= epoch < vae_end:
            return "vae"
        return "after"

    def _set_trainable_for_phase(self, phase: str):
        # Always freeze base encoders (your encoders already have freeze_encoder, but keep safe)
        self.freeze_(self.enc_0, True)
        self.freeze_(self.enc_1, True)

        heads = [self.enc_2, self.enc_3, self.enc_4, self.enc_5]
        vae_z1 = [self.enc_6, self.enc_7] if self.has_vae_z1 else []
        vae_z2 = [self.enc_8, self.enc_9] if self.has_vae_z2 else []

        if phase == "warmup":
            for m in heads:
                self.freeze_(m, False)
            for m in vae_z1 + vae_z2:
                self.freeze_(m, True)

        elif phase == "vae":
            for m in heads:
                self.freeze_(m, True)

            if self.vae_train_side == "z1":
                for m in vae_z1:
                    self.freeze_(m, False)
                for m in vae_z2:
                    self.freeze_(m, True)
            elif self.vae_train_side == "z2":
                for m in vae_z2:
                    self.freeze_(m, False)
                for m in vae_z1:
                    self.freeze_(m, True)
            else:
                raise ValueError("vae_train_side must be 'z1' or 'z2'.")

        else:  # after
            for m in heads + vae_z1 + vae_z2:
                self.freeze_(m, True)

    # -------------------------
    # feature extraction
    # -------------------------
    def _get_zs(self, x, **kwargs):
        out0 = self.enc_0(x, **kwargs)
        out1 = self.enc_1(x, **kwargs)
        z1, na_z1 = _as_tensor_features(out0)
        z2, na_z2 = _as_tensor_features(out1)
        z1 = F.layer_norm(z1, (z1.shape[-1],))
        z2 = F.layer_norm(z2, (z2.shape[-1],))
        na_z1 = F.layer_norm(na_z1, (na_z1.shape[-1],))
        na_z2 = F.layer_norm(na_z2, (na_z2.shape[-1],))
        return z1, z2, na_z1, na_z2

    def fusion_forward_from_z(self, z1, z2):
        # Your FusionTrunkLinear likely expects (z1,z2) directly
        try:
            feat = self.enc_2(z1, z2)
        except TypeError:
            feat = self.enc_2(torch.cat([z1, z2], dim=1))
        logits = self.enc_3(feat)
        return logits, feat

    def unimodal_forward_from_z(self, z1, z2):
        return self.enc_4(z1), self.enc_5(z2)

    # -------------------------
    # main forward
    # -------------------------
    def forward(self, x, **kwargs):
        epoch = int(kwargs.get("current_epoch", 0))
        phase = self._phase(epoch)

        if phase != self._last_phase:
            self._set_trainable_for_phase(phase)
            self._last_phase = phase

        z1, z2, na_z1, na_z2 = self._get_zs(x, **kwargs)
        fused_logits, feat = self.fusion_forward_from_z(z1, z2)
        p1, p2 = self.unimodal_forward_from_z(z1, z2)

        loss = fused_logits.new_tensor(0.0)

        # -------------------------
        # warmup: CE on heads
        # -------------------------
        losses, features = {}, {}
        if phase == "warmup":
            y = kwargs.get("label", None)
            if y is None:
                raise ValueError("Warmup requires kwargs['label'].")
            y = y.long()

            ce_combined = F.cross_entropy(fused_logits, y)
            ce_p1 = F.cross_entropy(p1, y)
            ce_p2 = F.cross_entropy(p2, y)

            # ml = _cfg(self.args, "multi_loss", {}) or {}
            # w = ml.get("multi_supervised_w", {}) if isinstance(ml, dict) else getattr(ml, "multi_supervised_w", {})
            w_comb = 1
            w_p1 = 1
            w_p2 = 1

            losses.update({
                "ce_combined": w_comb * ce_combined,
                "ce_p1": w_p1 * ce_p1,
                "ce_p2": w_p2 * ce_p2,
            })

        # -------------------------
        # vae: train ONE side only
        # -------------------------
        elif phase == "vae":
            vae_cfg = _cfg(self.args, "vae_loss", {}) or {}
            shp = _cfg(self.args, "svae_shaping", {}) or {}
            w_syn = float(shp.get("syn_kl_w", 0.0))
            T = float(shp.get("temp", 1.0))

            if self.vae_train_side == "z1":
                if not self.has_vae_z1:
                    raise RuntimeError("Selected vae_train_side='z1' but enc_6/enc_7 not provided.")

                mu, logvar = self.enc_6(z1)
                u = _reparam(mu, logvar)
                z1_tilde = self.enc_7(u)

                w_rec = float(vae_cfg.get("z1_rec_w", 1.0))
                w_kl = float(vae_cfg.get("z1_kl_w", 0.01))

                # base VAE losses
                losses["vae_z1_rec"] = w_rec * F.mse_loss(z1_tilde, z1, reduction="mean")
                losses["vae_z1_kl"] = w_kl * _kl_normal(mu, logvar)

                # optional syn shaping: KL(fuse(z1~,z2) || ensemble(z1~,z2))
                if w_syn > 0:
                    fused_tilde, _ = self.fusion_forward_from_z(z1_tilde, z2)
                    p_fuse_log = F.log_softmax(fused_tilde / T, dim=-1)
                    with torch.no_grad():
                        p1_tilde = self.enc_4(z1)
                        p2_orig = self.enc_5(z2)
                        p_ens = 0.5 * (F.softmax(p1_tilde / T, dim=-1) + F.softmax(p2_orig / T, dim=-1))

                    losses["svae_syn_kl_z1"] = w_syn * F.kl_div(p_fuse_log, p_ens, reduction="batchmean")

                # if you want to expose recon too
                features["z1_tilde"] = z1_tilde

            else:  # z2
                if not self.has_vae_z2:
                    raise RuntimeError("Selected vae_train_side='z2' but enc_8/enc_9 not provided.")

                mu, logvar = self.enc_8(z2)
                u = _reparam(mu, logvar)
                z2_tilde = self.enc_9(u)

                w_rec = float(vae_cfg.get("z2_rec_w", 1.0))
                w_kl = float(vae_cfg.get("z2_kl_w", 0.01))

                losses["vae_z2_rec"] = w_rec * F.mse_loss(z2_tilde, z2, reduction="mean")
                losses["vae_z2_kl"] = w_kl * _kl_normal(mu, logvar)

                if w_syn > 0:
                    fused_tilde, _ = self.fusion_forward_from_z(z1, z2_tilde)
                    p1_orig = self.enc_4(z1)
                    p2_tilde = self.enc_5(z2_tilde)

                    p_fuse_log = F.log_softmax(fused_tilde / T, dim=-1)
                    with torch.no_grad():
                        p_ens = 0.5 * (F.softmax(p1_orig / T, dim=-1) + F.softmax(p2_tilde / T, dim=-1))

                    losses["svae_syn_kl_z2"] = w_syn *  F.kl_div(p_fuse_log, p_ens, reduction="batchmean")

                features["z2_tilde"] = z2_tilde


        # -------------------------
        # after: do nothing
        # -------------------------
        else:
            loss = fused_logits.new_tensor(0.0)

        return {
            "preds": {"combined": fused_logits, "p1": p1, "p2": p2},
            "features": {"z1": z1, "z2": z2, "feat": feat},
            "losses": losses,   # <-- only optimized scalar
        }

class FusionIBModel_SVAEU_Phased(nn.Module):
    """
    Phase-controlled model to:
      - warmup: train heads with CE
      - vae:    train one VAE side (z1 or z2)
      - after:  do nothing

    IMPORTANT: your trainer optimizes whatever is in output["losses"].
    This model returns ONLY {"loss": loss_tensor} inside "losses".

    Expected encs (by default enc_ids):
      enc_0: modality-1 encoder (pretrained)
      enc_1: modality-2 encoder (pretrained)
      enc_2: fusion trunk  (FusionTrunkLinear or your impl)
      enc_3: fusion head   (LinearHead)
      enc_4: unimodal head z1 (MLPHead)
      enc_5: unimodal head z2 (MLPHead)
      enc_6: VAE posterior z1 (VAEPosterior)
      enc_7: VAE decoder   z1 (VAEDecoder)
      enc_8: VAE posterior z2 (VAEPosterior)
      enc_9: VAE decoder   z2 (VAEDecoder)

    Config compatibility:
      - args.train_mode: "phased" | "warmup" | "vae" | "after"
      - args.schedule: { warmup_epochs, vae_start_epoch, vae_end_epoch, after_mode }
      - kwargs["current_epoch"] is provided by your trainer
      - kwargs["label"] is used in warmup (CE)
      - args.vae_train_side: "z1" or "z2"
      - args.multi_loss.multi_supervised_w: {"combined":.., "c":.., "g":..}
      - args.vae_loss: {z1_rec_w,z1_kl_w,z2_rec_w,z2_kl_w}
      - args.svae_shaping: {syn_kl_w,temp}  (optional; used in VAE phase)
    """

    def __init__(self, args, encs, **kwargs):
        super().__init__()
        self.args = args

        # mapping from name -> index in encs
        enc_ids = _cfg(args, "enc_ids", None)
        if enc_ids is None:
            enc_ids = {
                "enc_0": 0, "enc_1": 1, "enc_2": 2, "enc_3": 3, "enc_4": 4,
                "enc_5": 5, "enc_6": 6, "enc_7": 7, "enc_8": 8, "enc_9": 9,
            }

        def _get_enc(name):
            idx = enc_ids.get(name, None)
            if idx is None or idx >= len(encs):
                return None
            return encs[idx]

        # core encoders (required)
        self.enc_0 = _get_enc("enc_0")
        self.enc_1 = _get_enc("enc_1")
        if self.enc_0 is None or self.enc_1 is None:
            raise ValueError("FusionIBModel_SVAE_Phased requires enc_0 and enc_1.")

        # heads (expected provided by your config)
        self.enc_2 = _get_enc("enc_2")  # Fusion trunk
        self.enc_3 = _get_enc("enc_3")  # Fusion head
        self.enc_4 = _get_enc("enc_4")  # Unimodal head z1
        self.enc_5 = _get_enc("enc_5")  # Unimodal head z2
        if any(m is None for m in [self.enc_2, self.enc_3, self.enc_4, self.enc_5]):
            raise ValueError("Missing head blocks enc_2/enc_3/enc_4/enc_5 in encs. Provide them via config encoders.")

        # VAEs (optional, but required for VAE mode on that side)
        self.enc_6 = _get_enc("enc_6")  # posterior z1
        self.enc_7 = _get_enc("enc_7")  # decoder   z1
        self.enc_8 = _get_enc("enc_8")  # posterior z2
        self.enc_9 = _get_enc("enc_9")  # decoder   z2

        self.has_vae_z1 = (self.enc_6 is not None) and (self.enc_7 is not None)
        self.has_vae_z2 = (self.enc_8 is not None) and (self.enc_9 is not None)

        self.vae_train_side = str(_cfg(args, "vae_train_side", "z1")).lower()
        self.train_mode = str(_cfg(args, "train_mode", "phased")).lower()
        self.schedule = _cfg(args, "schedule", {}) or {}

        self._last_phase = None  # one-time freeze/unfreeze

    # -------------------------
    # freeze helpers
    # -------------------------
    @staticmethod
    def freeze_(module: nn.Module, flag: bool = True):
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = (not flag)

    def _phase(self, epoch: int) -> str:
        # explicit override
        if self.train_mode in {"warmup", "vae", "after"}:
            return self.train_mode

        # phased schedule
        warmup_epochs = int(self.schedule.get("warmup_epochs", 0))
        vae_start = int(self.schedule.get("vae_start_epoch", warmup_epochs))
        vae_end = int(self.schedule.get("vae_end_epoch", vae_start))

        if epoch < warmup_epochs:
            return "warmup"
        if vae_start <= epoch < vae_end:
            return "vae"
        return "after"

    def _set_trainable_for_phase(self, phase: str):
        # Always freeze base encoders (your encoders already have freeze_encoder, but keep safe)
        self.freeze_(self.enc_0, True)
        self.freeze_(self.enc_1, True)

        heads = [self.enc_2, self.enc_3, self.enc_4, self.enc_5]
        vae_z1 = [self.enc_6, self.enc_7] if self.has_vae_z1 else []
        vae_z2 = [self.enc_8, self.enc_9] if self.has_vae_z2 else []

        if phase == "warmup":
            for m in heads:
                self.freeze_(m, False)
            for m in vae_z1 + vae_z2:
                self.freeze_(m, True)

        elif phase == "vae":
            for m in heads:
                self.freeze_(m, True)

            if self.vae_train_side == "z1":
                for m in vae_z1:
                    self.freeze_(m, False)
                for m in vae_z2:
                    self.freeze_(m, True)
            elif self.vae_train_side == "z2":
                for m in vae_z2:
                    self.freeze_(m, False)
                for m in vae_z1:
                    self.freeze_(m, True)
            else:
                raise ValueError("vae_train_side must be 'z1' or 'z2'.")

        else:  # after
            for m in heads + vae_z1 + vae_z2:
                self.freeze_(m, True)

    # -------------------------
    # feature extraction
    # -------------------------
    def _get_zs(self, x, **kwargs):
        out0 = self.enc_0(x, **kwargs)
        out1 = self.enc_1(x, **kwargs)
        z1, na_z1 = _as_tensor_features(out0)
        z2, na_z2 = _as_tensor_features(out1)
        z1 = F.layer_norm(z1, (z1.shape[-1],))
        z2 = F.layer_norm(z2, (z2.shape[-1],))
        na_z1 = F.layer_norm(na_z1, (na_z1.shape[-1],))
        na_z2 = F.layer_norm(na_z2, (na_z2.shape[-1],))
        return z1, z2, na_z1, na_z2

    def fusion_forward_from_z(self, z1, z2):
        # Your FusionTrunkLinear likely expects (z1,z2) directly
        try:
            feat = self.enc_2(z1, z2)
        except TypeError:
            feat = self.enc_2(torch.cat([z1, z2], dim=1))
        logits = self.enc_3(feat)
        return logits, feat

    def unimodal_forward_from_z(self, z1, z2):
        return self.enc_4(z1), self.enc_5(z2)

    def _oracle_grad_check(self, z1, z2, T=1.0, eps=1e-3):
        self.eval()

        z1_var = z1.detach().clone().requires_grad_(True)
        z2_det = z2.detach()

        with torch.no_grad():
            tgt = (self.enc_5(z2_det) / T).detach()

        with torch.enable_grad():
            mu, logvar = self.enc_6(z1)
            u = _reparam(mu, logvar)
            z1_tilde = self.enc_7(u)

            fused, _ = self.fusion_forward_from_z(z1_tilde, z2_det)
            loss = F.mse_loss(fused / T, tgt, reduction="mean")

            (g,) = torch.autograd.grad(loss, z1_tilde, retain_graph=False, create_graph=False)

            # basic stats
            info = {
                "loss": loss.detach(),
                "grad_norm": g.norm().detach(),
                "grad_absmax": g.abs().max().detach(),
                "grad_finite": torch.isfinite(g).all().detach(),
            }

            # take a very small step along -grad and re-evaluate
            z1_step = (z1_var - eps * g).detach().requires_grad_(False)
            fused2, _ = self.fusion_forward_from_z(z1_step, z2_det)
            loss2 = F.mse_loss(fused2 / T, tgt, reduction="mean")
            info["loss_after_step"] = loss2.detach()
            info["delta_loss"] = (loss2 - loss).detach()

        return info

    def _oracle_minimize_fuse_to_p2(
            self,
            z1, z2,
            T: float = 2.0,
            steps: int = 2,
            lr: float = 0.1,  # step length for normalized update
            lam_close: float = 0.0,  # optional closeness penalty to original z1
            init: str = "z1",  # "z1" | "zeros" | "randn"
            loss_mode: str = "prob_kl",  # "prob_kl" | "prob_mse" | "centered_logit_mse"
            rand_trials: int = 0,  # e.g. 32 to estimate floor
    ):
        """
        Feasibility test: optimize z1_var (per batch) to make fusion(z1_var,z2) match p2(z2).

        Returns dict with:
          oracle_start / oracle_final  (match loss)
          oracle_close (if lam_close>0)
          oracle_rand_min / oracle_rand_mean (if rand_trials>0)
        """

        def _match_loss(fuse_logits, teacher_logits):
            if loss_mode == "prob_kl":
                log_p = F.log_softmax(fuse_logits / T, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(teacher_logits / T, dim=-1)
                return F.kl_div(log_p, p_t, reduction="batchmean")

            if loss_mode == "prob_mse":
                p = F.softmax(fuse_logits / T, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(teacher_logits / T, dim=-1)
                return F.mse_loss(p, p_t, reduction="mean")

            if loss_mode == "centered_logit_mse":
                lf = fuse_logits / T
                lt = teacher_logits / T
                lf = lf - lf.mean(dim=-1, keepdim=True)
                lt = lt - lt.mean(dim=-1, keepdim=True)
                return F.mse_loss(lf, lt, reduction="mean")

            raise ValueError(f"Unknown loss_mode={loss_mode}")

        # Teacher (z2-only) — fixed target
        with torch.no_grad():
            teacher_logits = self.enc_5(z2).detach()

        # Initialize z1_var
        if init == "zeros":
            z1_var0 = torch.zeros_like(z1, device=z1.device)
        elif init == "randn":
            z1_var0 = torch.randn_like(z1, device=z1.device)
        else:  # "z1"
            z1_var0 = z1.detach().clone()

        z1_ref = z1.detach()
        z2_det = z2.detach()

        out = {}

        # Optional: random z1 sweep to detect a hard floor
        if rand_trials and rand_trials > 0:
            vals = []
            with torch.no_grad():
                for _ in range(int(rand_trials)):
                    z1_r = torch.randn_like(z1_ref)
                    fused_r, _ = self.fusion_forward_from_z(z1_r, z2_det)
                    vals.append(float(_match_loss(fused_r, teacher_logits).detach()))
            out["oracle_rand_min"] = torch.tensor(min(vals), device=z1.device)
            out["oracle_rand_mean"] = torch.tensor(sum(vals) / len(vals), device=z1.device)

        # Important: we may be under outer no_grad; force autograd locally
        with torch.enable_grad():
            z1_var = z1_var0.detach().requires_grad_(True)

            # initial
            fused0, _ = self.fusion_forward_from_z(z1_var, z2_det)
            loss0 = _match_loss(fused0, teacher_logits)

            for _ in range(int(steps)):
                #get z1_var from the vae


                fused, _ = self.fusion_forward_from_z(z1_var, z2_det)
                loss_match = _match_loss(fused, teacher_logits)

                if lam_close > 0:
                    loss_close = F.mse_loss(z1_var, z1_ref, reduction="mean")
                    loss = loss_match + lam_close * loss_close
                else:
                    loss = loss_match

                (grad,) = torch.autograd.grad(loss, z1_var, retain_graph=False, create_graph=False)

                # normalized step (per-sample): prevents tiny grads -> no movement
                gnorm = grad.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                step = lr * (grad / gnorm)

                z1_var = (z1_var - step).detach().requires_grad_(True)

            # final
            fusedF, _ = self.fusion_forward_from_z(z1_var, z2_det)
            lossF = _match_loss(fusedF, teacher_logits)

        out["oracle_start"] = loss0.detach()
        out["oracle_final"] = lossF.detach()
        if lam_close > 0:
            out["oracle_close"] = F.mse_loss(z1_var.detach(), z1_ref, reduction="mean").detach()

        return out

    def _vae_oracle_losses_z1(
            self,
            na_z1, z1, z2,
            T: float = 1.0,
            refine_steps: int = 0,  # 1–3 is enough
            refine_lr: float = 0.3,  # step length in latent space (normalized)
            loss_mode: str = "centered_logit_mse",  # "prob_kl" | "prob_mse" | "centered_logit_mse"
            w_start: float = 1.0,
            w_refined: float = 1.0,
            w_amort: float = 1.0,
            w_kl: float = 0.0,  # optional VAE KL
            lam_close: float = 1.0,  # optional: keep z_tilde close to z1 (in feature space)
    ):
        """
        Trains BOTH enc_6 (posterior) and enc_7 (decoder) using oracle-style refinement on latent u.
        Returns a dict of losses (no detach) + optional features.
        """

        def _match_loss(fuse_logits, teacher_logits):
            if loss_mode == "prob_kl":
                log_p = F.log_softmax(fuse_logits / T, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(teacher_logits / T, dim=-1)
                return F.kl_div(log_p, p_t, reduction="batchmean")

            if loss_mode == "prob_mse":
                p = F.softmax(fuse_logits / T, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(teacher_logits / T, dim=-1)
                return F.mse_loss(p, p_t, reduction="mean")

            if loss_mode == "centered_logit_mse":
                lf = fuse_logits / T
                lt = teacher_logits / T
                lf = lf - lf.mean(dim=-1, keepdim=True)
                lt = lt - lt.mean(dim=-1, keepdim=True)
                return F.mse_loss(lf, lt, reduction="mean")

            raise ValueError(f"Unknown loss_mode={loss_mode}")

        losses = {}

        na_z1_ref = na_z1.detach()
        z1_ref = z1.detach()
        z2_det = z2.detach()

        with torch.no_grad():
            teacher_logits = self.enc_5(z2_det).detach()

        mu, logvar = self.enc_6(z1_ref, na_z1_ref)
        u0 = _reparam(mu, logvar)  # depends on enc_6
        z1_tilde0, na_z1_tilde0 = self.enc_7(u0)  # depends on enc_7

        fused0, _ = self.fusion_forward_from_z(z1_tilde0, z2_det)

        losses["~fused_z2"] = w_start * _match_loss(fused0, teacher_logits)
        losses["~z1_z1"] = lam_close *_match_loss(z1_tilde0, z1_ref)
        if not self.training:
            return losses, {"z1_tilde": z1_tilde0.detach()}

        if w_kl > 0:
            losses["vae_kl_z1"] = w_kl * _kl_normal(mu, logvar)


        if refine_steps > 0:
            with torch.enable_grad():
                u_star = u0.detach()

                for _ in range(int(refine_steps)):
                    u_star = u_star.detach().requires_grad_(True)

                    z_tmp, na_z_tmp = self.enc_7(u_star)  # uses enc_7 params, grad only wrt u_star
                    fused_tmp, _ = self.fusion_forward_from_z(z_tmp, z2_det)
                    L_tmp = _match_loss(fused_tmp, teacher_logits)

                    (g_u,) = torch.autograd.grad(L_tmp, u_star, retain_graph=False, create_graph=False)
                    gnorm = g_u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    u_star = (u_star - refine_lr * (g_u / gnorm)).detach()
            z1_star, na_z1_star = self.enc_7(u_star)  # grad to enc_7
            losses["vae_oracle_amort_mu"] = w_amort * F.mse_loss(mu, u_star, reduction="mean")
        else:
            z1_star, na_z1_star = z1_tilde0, na_z1_tilde0  # no refinement
            u_star = u0

        fused_star, _ = self.fusion_forward_from_z(z1_star, z2_det)
        losses["oracle_~fused_z2"] = w_refined * _match_loss(fused_star, teacher_logits)
        losses["oracle_~z1_z1"] = lam_close * _match_loss(z1_star, z1_ref)

        return losses, {"z1_tilde": z1_tilde0, "z1_star": z1_star, "u_star": u_star}
        # return losses

    def _vae_oracle_losses_z2(
            self,
            na_z2, z2, z1,
            T: float = 1.0,
            refine_steps: int = 0,  # 1–3 is enough
            refine_lr: float = 0.3,  # step length in latent space (normalized)
            loss_mode: str = "centered_logit_mse",  # "prob_kl" | "prob_mse" | "centered_logit_mse"
            w_start: float = 1.0,
            w_refined: float = 1.0,
            w_amort: float = 1.0,
            w_kl: float = 0.0,  # optional VAE KL
            lam_close: float = 1.0,  # optional: keep z_tilde close to z2 (in feature space)
    ):

        def _match_loss(fuse_logits, teacher_logits):
            if loss_mode == "prob_kl":
                log_p = F.log_softmax(fuse_logits / T, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(teacher_logits / T, dim=-1)
                return F.kl_div(log_p, p_t, reduction="batchmean")

            if loss_mode == "prob_mse":
                p = F.softmax(fuse_logits / T, dim=-1)
                with torch.no_grad():
                    p_t = F.softmax(teacher_logits / T, dim=-1)
                return F.mse_loss(p, p_t, reduction="mean")

            if loss_mode == "centered_logit_mse":
                lf = fuse_logits / T
                lt = teacher_logits / T
                lf = lf - lf.mean(dim=-1, keepdim=True)
                lt = lt - lt.mean(dim=-1, keepdim=True)
                return F.mse_loss(lf, lt, reduction="mean")

            raise ValueError(f"Unknown loss_mode={loss_mode}")

        losses = {}

        na_z2_ref = na_z2.detach()
        z2_ref = z2.detach()
        z1_det = z1.detach()

        with torch.no_grad():
            teacher_logits = self.enc_5(z1_det).detach()

        mu, logvar = self.enc_8(z2_ref, na_z2_ref)
        u0 = _reparam(mu, logvar)  # depends on enc_6
        z2_tilde0, na_z2_tilde0 = self.enc_9(u0)  # depends on enc_7

        fused0, _ = self.fusion_forward_from_z(z2_tilde0, z1_det)

        losses["~fused_z1"] = w_start * _match_loss(fused0, teacher_logits)
        losses["~z2_z2"] = lam_close *_match_loss(z2_tilde0, z2_ref)
        if not self.training:
            return losses, {"z2_tilde": z2_tilde0.detach()}

        if w_kl > 0:
            losses["vae_kl_z2"] = w_kl * _kl_normal(mu, logvar)

        if refine_steps > 0:
            with torch.enable_grad():
                u_star = u0.detach()

                for _ in range(int(refine_steps)):
                    u_star = u_star.detach().requires_grad_(True)

                    z_tmp, na_z_tmp = self.enc_9(u_star)  # uses enc_7 params, grad only wrt u_star
                    fused_tmp, _ = self.fusion_forward_from_z(z_tmp, z1_det)
                    L_tmp = _match_loss(fused_tmp, teacher_logits)

                    (g_u,) = torch.autograd.grad(L_tmp, u_star, retain_graph=False, create_graph=False)
                    gnorm = g_u.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                    u_star = (u_star - refine_lr * (g_u / gnorm)).detach()
            z2_star, na_z2_star = self.enc_9(u_star)  # grad to enc_7
            losses["vae_oracle_amort_mu"] = w_amort * F.mse_loss(mu, u_star, reduction="mean")
        else:
            u_star = u0
            z2_star = z2_tilde0
            # na_z2_star = na_z2_tilde0

        fused_star, _ = self.fusion_forward_from_z(z2_star, z1_det)
        losses["oracle_~fused_z1"] = w_refined * _match_loss(fused_star, teacher_logits)
        losses["oracle_~z2_z2"] = lam_close * _match_loss(z2_star, z2_ref)

        return losses, {"z2_tilde": z2_tilde0, "z2_star": z2_star, "u_star": u_star}

    def forward(self, x, **kwargs):
        epoch = int(kwargs.get("current_epoch", 0))
        phase = self._phase(epoch)

        if phase != self._last_phase:
            self._set_trainable_for_phase(phase)
            self._last_phase = phase

        z1, z2, na_z1, na_z2 = self._get_zs(x, **kwargs)
        fused_logits, feat = self.fusion_forward_from_z(z1, z2)
        p1, p2 = self.unimodal_forward_from_z(z1, z2)
        losses, features = {}, {}
        if phase == "warmup":
            y = kwargs.get("label", None)
            if y is None:
                raise ValueError("Warmup requires kwargs['label'].")
            y = y.long()

            ce_combined = F.cross_entropy(fused_logits, y)
            ce_p1 = F.cross_entropy(p1, y)
            ce_p2 = F.cross_entropy(p2, y)

            w_comb, w_p1, w_p2 = 1, 1, 1
            losses.update({
                "ce_combined": w_comb * ce_combined,
                "ce_p1": w_p1 * ce_p1,
                "ce_p2": w_p2 * ce_p2,
            })

        elif phase == "vae":
            if self.vae_train_side == "z1":
                if not self.has_vae_z1:
                    raise RuntimeError("Selected vae_train_side='z1' but enc_6/enc_7 not provided.")
                oracle_out, vae_feat = self._vae_oracle_losses_z1(na_z1, z1, z2)
                losses.update({k: v for k, v in oracle_out.items()})
                features["z1_tilde"] = vae_feat["z1_tilde"]
            else:
                if not self.has_vae_z2:
                    raise RuntimeError("Selected vae_train_side='z2' but enc_6/enc_7 not provided.")
                oracle_out, vae_feat = self._vae_oracle_losses_z2(na_z2, z2, z1)
                losses.update({k: v for k, v in oracle_out.items()})
                features["z2_tilde"] = vae_feat["z2_tilde"]

        return {
            "preds": {"combined": fused_logits, "p1": p1, "p2": p2},
            "features": {"z1": z1, "z2": z2, "feat": feat},
            "losses": losses,   # <-- only optimized scalar
        }

#
# class FeatureStatsMasker(nn.Module):
#     def __init__(self, d1, ema_beta=0.99, eps=1e-6, device=None, dtype=None):
#         super().__init__()
#         factory_kwargs = dict(device=device, dtype=dtype)
#         self.d1 = d1
#         self.ema_beta = float(ema_beta)
#         self.eps = float(eps)
#         self.register_buffer("mu1", torch.zeros(d1, **factory_kwargs))
#         self.register_buffer("m21", torch.ones(d1, **factory_kwargs))
#         self.register_buffer("n1", torch.zeros((), **factory_kwargs))
#
#     @torch.no_grad()
#     def ema_update(self, z1):
#         x = z1.detach()
#         while x.dim() > 2:
#             x = x.flatten(0, -2)
#         if x.dim() == 1:
#             x = x[None, :]
#         mu = x.mean(0)
#         m2 = x.var(0, unbiased=False)
#         b = self.ema_beta if self.n1.item() > 0 else 0.0
#         self.mu1.lerp_(mu, 1.0 - b)
#         self.m21.lerp_(m2, 1.0 - b)
#         self.n1.add_(1)
#
#     def feature_stats(self):
#         return self.mu1, self.m21.clamp_min(self.eps)
#
#     def noise_like(self, z1, noise_scale=1.0):
#         mu, m2 = self.feature_stats()
#         std = (m2.sqrt() * float(noise_scale)).view(*([1] * (z1.dim() - 1)), -1)
#         mu = mu.view(*([1] * (z1.dim() - 1)), -1)
#         return mu + torch.randn_like(z1) * std
#
# class SynIB_Mask(nn.Module):
#     """
#     Same interface as SynIB, but uses pretrained VAEs as skew samplers.
#     IMPORTANT: VAEs are used under torch.no_grad(); no VAE losses are returned.
#
#     Expected encs layout:
#       encs[0] = modality-1 encoder (main model uses)
#       encs[1] = modality-2 encoder (main model uses)
#       encs[2] = VAE posterior for z1
#       encs[3] = VAE decoder   for z1
#       encs[4] = VAE posterior for z2
#       encs[5] = VAE decoder   for z2
#     """
#     def __init__(self, args, encs, main):
#         super().__init__()
#         object.__setattr__(self, "main", main)
#
#         self.perturb = _cfg(args, "perturb", {}) or {}
#         self.reestimate_features = bool(self.perturb.get("reestimate_features", False))
#
#         bias = _cfg(args, "bias_infusion", {}) or {}
#         self.synergy_weight = float(bias.get("l", 0.0))
#         self.contrastive_weight = float(bias.get("contrcoeff", 0.0) or 0.0)
#         self.synergy_type = getattr(args, "synergy_type", "gaussian")  # "gaussian" or "dirichlet"
#
#
#         fc_inner = int(_cfg(args, "fc_inner"))
#         num_classes = int(_cfg(args, "num_classes"))
#
#         if self.synergy_type == "gaussian":
#             self.logvar_head = nn.Linear(fc_inner, num_classes)
#             self.dirichlet_prior_conc = None
#         elif self.synergy_type == "dirichlet":
#             self.evidence_head = nn.Linear(fc_inner, num_classes)
#             self.dirichlet_prior_conc = float(_cfg(args, "dirichlet_prior_conc", 1.0))
#         else:
#             raise ValueError(f"Unknown synergy_type: {self.synergy_type}")
#
#         self.stats_z1 = FeatureStatsMasker(d1=512, ema_beta=0.99)
#         self.stats_z2 = FeatureStatsMasker(d1=512, ema_beta=0.99)
#
#
#     @staticmethod
#     def _gaussian_kl(mu, logvar):
#         return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, dim=1).mean()
#
#     @staticmethod
#     def _dirichlet_kl(alpha, prior_conc=1.0):
#         alpha0 = torch.full_like(alpha, prior_conc) if isinstance(prior_conc, float) else prior_conc
#         alpha0_sum = alpha0.sum(dim=1, keepdim=True)
#         alpha_sum = alpha.sum(dim=1, keepdim=True)
#
#         lgamma = torch.lgamma
#         digamma = torch.digamma
#
#         logB_alpha = torch.sum(lgamma(alpha), dim=1) - lgamma(alpha_sum.squeeze(1))
#         logB_alpha0 = torch.sum(lgamma(alpha0), dim=1) - lgamma(alpha0_sum.squeeze(1))
#
#         term1 = logB_alpha0 - logB_alpha
#         term2 = torch.sum((alpha - alpha0) * (digamma(alpha) - digamma(alpha_sum)), dim=1)
#         return (term1 + term2).mean()
#
#     @staticmethod
#     def _cat_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#         p = F.softmax(p, dim=-1)
#         q = F.softmax(q, dim=-1)
#         p = p.clamp_min(eps)
#         q = q.clamp_min(eps)
#         return (p * (p.log() - q.log())).sum(dim=-1).mean()
#
#     @staticmethod
#     def kl_to_uniform_multiclass_from_logits(logits: torch.Tensor) -> torch.Tensor:
#         logp = F.log_softmax(logits, dim=-1)
#         p = logp.exp()
#         K = logits.size(-1)
#         return (p * logp).sum(dim=-1).mean() + torch.log(torch.tensor(float(K), device=logits.device))
#
#     def _encode_and_perturb(self, x, features, **kwargs):
#         self.main.eval()
#         losses = {}
#         out = self._perturb_masked(features, **kwargs)
#
#         features.update({
#             "tz1": out["z1_tilde"],
#             "tz2": out["z2_tilde"],
#             "ez1": features["z1"],
#             "ez2": features["z2"],
#             "ey1": kwargs["label"]
#         })
#
#         self.main.train()
#         return features, losses
#
#     def _repeat_tensor(self, t, K):
#         if t is None: return None
#         return t.repeat_interleave(K, dim=0)
#
#     def _logit_kl(self, pred_logits, target_logits):
#         return F.kl_div(
#             F.log_softmax(pred_logits, dim=-1),
#             F.softmax(target_logits, dim=-1),
#             reduction="batchmean"
#         )
#
#     def _freeze_model(self, model):
#         snap = {}
#         for name, p in model.named_parameters():
#             snap[name] = bool(p.requires_grad)
#             p.requires_grad_(False)
#         return snap
#
#     def _unfreeze_model(self, model, snap=None, default=True):
#         for name, p in model.named_parameters():
#             if snap is None:
#                 p.requires_grad_(bool(default))
#             else:
#                 p.requires_grad_(bool(snap.get(name, default)))
#
#     @torch.no_grad()
#     def track_step(self, history, **kv):
#         def _add(k, x):
#             history.setdefault(k, []).append(x)
#
#         def _tensor_stats(t: torch.Tensor):
#             t = t.detach()
#             t = t.float()
#             return {
#                 "mean": t.mean().item(),
#                 "std": t.std(unbiased=False).item() if t.numel() > 1 else 0.0,
#                 "min": t.min().item(),
#                 "max": t.max().item(),
#             }
#
#         for k, v in kv.items():
#             if v is None:
#                 continue
#
#             if isinstance(v, dict):
#                 for kk, vv in v.items():
#                     self.track_step(history, **{f"{k}.{kk}": vv})
#                 continue
#
#             if torch.is_tensor(v):
#                 if v.numel() == 1:
#                     _add(k, float(v.detach().item()))
#                 else:
#                     stats = _tensor_stats(v)
#                     for sk, sv in stats.items():
#                         _add(f"{k}.{sk}", float(sv))
#                 continue
#
#             if isinstance(v, (float, int)):
#                 _add(k, float(v))
#                 continue
#
#             if isinstance(v, (list, tuple)):
#                 if len(v) == 0:
#                     continue
#                 if all(torch.is_tensor(x) for x in v):
#                     for j, x in enumerate(v):
#                         self.track_step(history, **{f"{k}[{j}]": x})
#                 elif all(isinstance(x, (float, int)) for x in v):
#                     _add(k, [float(x) for x in v])
#                 else:
#                     _add(k, v)
#                 continue
#
#             _add(k, v)
#
#     @torch.no_grad()
#     @staticmethod
#     def progress_report(self, i, steps, history):
#         def last(key, default=None):
#             xs = history.get(key, None)
#             if not xs:
#                 return default
#             return xs[-1]
#
#         keys = [
#             "obj", "score", "div_f", "div_u0", "div_u1", "sparsity", "g0m", "g1m", "e_f_t", "grad0", "grad1",
#             "m.f0.mean", "m.f0.std", "m.f0.min", "m.f0.max",
#             "m.f1.mean", "m.f1.std", "m.f1.min", "m.f1.max",
#         ]
#
#         parts = [f"[{i + 1:03d}/{steps}]"]
#         for k in keys:
#             v = last(k, None)
#             if v is None:
#                 continue
#             if isinstance(v, (float, int)):
#                 parts.append(f"{k}={v:.4f}")
#             else:
#                 parts.append(f"{k}={v}")
#         print(" ".join(parts))
#
#     @torch.no_grad()
#     @staticmethod
#     def _entropy_from_logits(self, logits: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
#         p = F.softmax(logits, dim=-1).clamp_min(eps)
#         return -(p * p.log()).sum(dim=-1).mean()
#
#     @torch.no_grad()
#     def check_marginals(self, x0, x1, x0_t, x1_t, g0, g1, eps=1e-6):
#         def _rel_change(a, b):
#             na = a.float().pow(2).mean().sqrt()
#             nb = (a.float() - b.float()).pow(2).mean().sqrt()
#             return nb / (na + eps)
#
#         def _zscore_mean(x, mu, m2):
#             std = (m2.float().clamp_min(eps)).sqrt()
#             z = (x.float() - mu.float()) / std
#             return z.abs().mean()
#
#         def _std_ratio(x, m2_ref):
#             std_x = x.float().std(dim=0, unbiased=False).mean()
#             std_r = (m2_ref.float().clamp_min(eps)).sqrt().mean()
#             return std_x / (std_r + eps)
#
#         g0 = g0.detach()
#         g1 = g1.detach()
#         g0m = g0.float().mean()
#         g1m = g1.float().mean()
#         g_sat0 = ((g0 < 0.05) | (g0 > 0.95)).float().mean()
#         g_sat1 = ((g1 < 0.05) | (g1 > 0.95)).float().mean()
#
#         rel0 = _rel_change(x0, x0_t)
#         rel1 = _rel_change(x1, x1_t)
#
#         mu0, m20 = self.stats_z1.feature_stats()
#         mu1, m21 = self.stats_z2.feature_stats()
#
#         mu0 = mu0.view(*([1] * (x0_t.dim() - 1)), -1)
#         m20 = m20.view(*([1] * (x0_t.dim() - 1)), -1)
#         mu1 = mu1.view(*([1] * (x1_t.dim() - 1)), -1)
#         m21 = m21.view(*([1] * (x1_t.dim() - 1)), -1)
#
#         z0 = _zscore_mean(x0_t, mu0, m20)
#         z1 = _zscore_mean(x1_t, mu1, m21)
#
#         sr0 = _std_ratio(x0_t, m20.squeeze(0) if x0_t.dim() == 2 else m20.view(-1))
#         sr1 = _std_ratio(x1_t, m21.squeeze(0) if x1_t.dim() == 2 else m21.view(-1))
#
#         return {
#             "m": {
#                 "rel0": float(rel0),
#                 "rel1": float(rel1),
#                 "z0": float(z0),
#                 "z1": float(z1),
#                 "std_ratio0": float(sr0),
#                 "std_ratio1": float(sr1),
#                 "g_sat0": float(g_sat0),
#                 "g_sat1": float(g_sat1),
#                 "g0m": float(g0m),
#                 "g1m": float(g1m),
#             }
#         }
#
#     # def get_learned_destroy_mask_multiclass(
#     #         self,
#     #         x0: torch.Tensor,
#     #         x1: torch.Tensor,
#     #         device: str,
#     #         *,
#     #         method="kl_uniform",
#     #         steps: int = 200,
#     #         lr: float = 5e-2,
#     #         tau: float = 1.0,
#     #         noise_std: float = 1.0,
#     #         lsparse: float = 5e-3,
#     #         alpha_unimodal: float = 5.0,
#     #         hard: bool = True,
#     #         hard_thresh: float = 0.5,
#     #         **kwargs
#     # ) -> Dict[str, torch.Tensor]:
#     #
#     #     self.main.eval()
#     #     snap = self._freeze_model(self.main)
#     #
#     #     x0, x1 = x0.to(device), x1.to(device)
#     #     B, d0 = x0.shape
#     #     _, d1 = x1.shape
#     #
#     #     self.stats_z1.ema_update(x0)
#     #     self.stats_z2.ema_update(x1)
#     #
#     #     # Mask parameters
#     #     # ell0 = torch.zeros(d0, device=device, requires_grad=True)
#     #     # ell1 = torch.zeros(d1, device=device, requires_grad=True)
#     #
#     #     with torch.no_grad():
#     #         # Clean targets must be detached from the graph
#     #         f_clean, _ = self.main._compute_logits(x0, x1)
#     #         u0_clean = self.main._compute_logits_unimodal(x0, x1, direction="z1")
#     #         u1_clean = self.main._compute_logits_unimodal(x0, x1, direction="z2")
#     #
#     #         f_target = f_clean.detach()
#     #         u0_target = u0_clean.detach()
#     #         u1_target = u1_clean.detach()
#     #
#     #     ell0 = torch.zeros(x0.shape[1], device=device).requires_grad_(True)
#     #     ell1 = torch.zeros(x1.shape[1], device=device).requires_grad_(True)
#     #
#     #     opt = torch.optim.Adam([ell0, ell1], lr=lr)
#     #     history = {}
#     #     with torch.enable_grad():
#     #         for i in range(steps):
#     #
#     #             beta = 5.0  # try 2, 5, 10, 20
#     #             g0 = torch.sigmoid(beta * ell0)
#     #             g1 = torch.sigmoid(beta * ell1)
#     #
#     #             x0_t = (1 - g0) * x0 + g0 * self.stats_z1.noise_like(x0, noise_scale=noise_std)
#     #             x1_t = (1 - g1) * x1 + g1 * self.stats_z2.noise_like(x1, noise_scale=noise_std)
#     #
#     #             f_t, _ = self.main._compute_logits(x0_t, x1_t)
#     #             u0_t = self.main._compute_logits_unimodal(x0_t, x1_t, direction="z1")
#     #             u1_t = self.main._compute_logits_unimodal(x0_t, x1_t, direction="z2")
#     #
#     #             sparsity = g0.mean() + g1.mean()
#     #
#     #             # obj = self.kl_to_uniform_multiclass_from_logits(f_t) + lsparse * sparsity
#     #
#     #             div_f = self._cat_kl(f_clean, f_t)
#     #             div_u0 = self._cat_kl(u0_clean, u0_t)
#     #             div_u1 = self._cat_kl(u1_clean, u1_t)
#     #             score = div_f - float(alpha_unimodal) * (div_u0 + div_u1)
#     #             obj = -score + float(lsparse) * sparsity
#     #
#     #             opt.zero_grad()
#     #
#     #             obj.backward()
#     #
#     #             def _normalize_grad_(p, eps=1e-12, max_norm=None):
#     #                 if p.grad is None:
#     #                     return
#     #                 g = p.grad
#     #                 n = g.norm().clamp_min(eps)
#     #                 g.div_(n)
#     #                 if max_norm is not None:
#     #                     g.mul_(float(max_norm))
#     #
#     #             # _normalize_grad_(ell0, max_norm=1.0)
#     #             # _normalize_grad_(ell1, max_norm=1.0)
#     #             opt.step()
#     #
#     #             grad0 = ell0.grad.norm()
#     #             grad1 = ell1.grad.norm()
#     #
#     #             m = self.check_marginals(x0, x1, x0_t, x1_t, g0, g1)
#     #
#     #             self.track_step(
#     #                 history,
#     #                 obj=obj,
#     #                 score=score,
#     #                 div_f=div_f,
#     #                 div_u0=div_u0,
#     #                 div_u1=div_u1,
#     #                 sparsity=sparsity,
#     #                 g0m=g0.mean(),
#     #                 g1m=g1.mean(),
#     #                 e_f_t=self._entropy_from_logits(f_t),
#     #                 grad0=grad0,
#     #                 grad1=grad1,
#     #                 m = m
#     #             )
#     #
#     #             if (i + 1) % steps == 0 or i == 0:
#     #                 self.progress_report(i, steps, history)
#     #     if "current_step" in kwargs:
#     #         latest_history = {k: v[-1] for k, v in history.items()}
#     #         wandb.log(latest_history, step=kwargs["current_step"]+1)
#     #
#     #     # Detach and threshold
#     #     final_g0 = torch.sigmoid(ell0 / tau).detach()
#     #     final_g1 = torch.sigmoid(ell1 / tau).detach()
#     #
#     #     if hard:
#     #         final_g0 = (final_g0 > hard_thresh).float()
#     #         final_g1 = (final_g1 > hard_thresh).float()
#     #
#     #     self._unfreeze_model(self.main, snap=snap)
#     #
#     #     return {"g0": final_g0, "g1": final_g1}
#
#     def get_learned_destroy_mask_multiclass(
#             self,
#             x0: torch.Tensor,
#             x1: torch.Tensor,
#             device: str,
#             method="kl_uniform",
#             steps: int = 200,
#             lr: float = 5e-2,
#             tau: float = 1.0,
#             noise_std: float = 1.0,
#             lsparse: float = 5e-3,
#             alpha_unimodal: float = 5.0,
#             hard: bool = True,
#             hard_thresh: float = 0.5,
#             **kwargs
#     ) -> Dict[str, torch.Tensor]:
#         """
#         Sequential, decoupled mask learning:
#
#           Phase A (learn g0): corrupt x0 only, keep x1 clean.
#             objective uses div_f - alpha * div_u0  + sparsity(g0)
#
#           Phase B (learn g1): corrupt x1 only, keep x0 clean.
#             objective uses div_f - alpha * div_u1  + sparsity(g1)
#
#         No "other" mask is used while learning a given mask.
#         """
#
#         # ---- setup ----
#         self.main.eval()
#         snap = self._freeze_model(self.main)
#
#         x0, x1 = x0.to(device), x1.to(device)
#         d0, d1 = x0.shape[1], x1.shape[1]
#
#         self.stats_z1.ema_update(x0)
#         self.stats_z2.ema_update(x1)
#
#         # ---- clean reference logits (detached) ----
#         with torch.no_grad():
#             f_clean, _ = self.main._compute_logits(x0, x1)
#             u0_clean = self.main._compute_logits_unimodal(x0, x1, direction="z1")
#             u1_clean = self.main._compute_logits_unimodal(x0, x1, direction="z2")
#
#         # ---- small helpers ----
#         beta = 5.0  # sigmoid sharpness during optimization
#
#         def gates_from_ell(ell: torch.Tensor) -> torch.Tensor:
#             return torch.sigmoid(beta * ell)
#
#         def destroy(x: torch.Tensor, g: torch.Tensor, noise_fn) -> torch.Tensor:
#             # g is (d,) and broadcasts over batch
#             return (1 - g) * x + g * noise_fn(x, noise_scale=noise_std)
#
#         history = {}
#
#         def learn_one_mask(
#                 *,
#                 phase: str,
#                 ell: torch.Tensor,
#                 opt: torch.optim.Optimizer,
#                 corrupt: str,  # "x0" or "x1"
#         ) -> torch.Tensor:
#             for i in range(steps):
#                 g = gates_from_ell(ell)
#
#                 if corrupt == "x0":
#                     x0_t = destroy(x0, g, self.stats_z1.noise_like)
#                     x1_t = x1
#                     f_t, _ = self.main._compute_logits(x0_t, x1_t)
#                     u_t = self.main._compute_logits_unimodal(x0_t, x1_t, direction="z1")
#                     div_u = self._cat_kl(u0_clean, u_t)  # only the corresponding unimodal term
#                 elif corrupt == "x1":
#                     x0_t = x0
#                     x1_t = destroy(x1, g, self.stats_z2.noise_like)
#                     f_t, _ = self.main._compute_logits(x0_t, x1_t)
#                     u_t = self.main._compute_logits_unimodal(x0_t, x1_t, direction="z2")
#                     div_u = self._cat_kl(u1_clean, u_t)
#                 else:
#                     raise ValueError("corrupt must be 'x0' or 'x1'")
#
#                 div_f = self._cat_kl(f_clean, f_t)
#                 sparsity = g.mean()
#
#                 score = div_f - float(alpha_unimodal) * div_u
#                 obj = -score + float(lsparse) * sparsity
#
#                 opt.zero_grad()
#                 obj.backward()
#                 opt.step()
#
#                 gradn = ell.grad.norm() if ell.grad is not None else torch.tensor(0.0, device=device)
#                 m = self.check_marginals(x0, x1, x0_t, x1_t,
#                                          g0=(g if corrupt == "x0" else torch.zeros(d0, device=device)),
#                                          g1=(g if corrupt == "x1" else torch.zeros(d1, device=device)))
#
#                 # keep logging keys consistent-ish with your previous code
#                 self.track_step(
#                     history,
#                     phase=phase,
#                     obj=obj,
#                     score=score,
#                     div_f=div_f,
#                     div_u=div_u,
#                     sparsity=sparsity,
#                     gm=g.mean(),
#                     e_f_t=self._entropy_from_logits(f_t),
#                     grad=gradn,
#                     m=m,
#                 )
#
#                 if i == 0 or (i + 1) == steps:
#                     self.progress_report(i, steps, history)
#
#             # return final soft gate for this phase (still continuous)
#             return ell.detach(), history
#
#         with torch.enable_grad():
#             # ---- Phase A: learn g0 (corrupt x0 only) ----
#             ell0 = torch.ones(d0, device=device).requires_grad_(True)
#             opt0 = torch.optim.Adam([ell0], lr=lr)
#             ell0, h0 = learn_one_mask(phase="learn_g0", ell=ell0, opt=opt0, corrupt="x0")
#
#             # ---- Phase B: learn g1 (corrupt x1 only) ----
#             ell1 = torch.ones(d1, device=device).requires_grad_(True)
#             opt1 = torch.optim.Adam([ell1], lr=lr)
#             ell1, h1 = learn_one_mask(phase="learn_g1", ell=ell1, opt=opt1, corrupt="x1")
#
#         if "current_step" in kwargs:
#             latest_h_0 = {k: v[-1] for k, v in h0.items()}
#             latest_h_1 = {k: v[-1] for k, v in h1.items()}
#             lh = {"g0":latest_h_0, "g1":latest_h_1}
#             wandb.log(lh, step=kwargs["current_step"] + 1)
#
#         # ---- final: detach + (optional) hard threshold ----
#         final_g0 = torch.sigmoid(ell0 / tau).detach()
#         final_g1 = torch.sigmoid(ell1 / tau).detach()
#
#         if hard:
#             final_g0 = (final_g0 > hard_thresh).float()
#             final_g1 = (final_g1 > hard_thresh).float()
#
#         self._unfreeze_model(self.main, snap=snap)
#         return {"g0": final_g0, "g1": final_g1}
#
#     def _kl_loss(self, mu, feat):
#         if self.synergy_type == "gaussian":
#             logvar = self.logvar_head(feat)
#             kl = self._gaussian_kl(mu, logvar)
#         else:
#             evidence = F.softplus(self.evidence_head(feat))
#             alpha = evidence + 1.0
#             kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
#         return kl * self.synergy_weight
#     def _kl_pass(self, base_output, **kwargs):
#
#         feat = base_output["features"]["mask0"]
#         mu = base_output["preds"]["mask0"]
#         kl_loss_1 = self._kl_loss(mu, feat)
#
#
#         feat = base_output["features"]["mask1"]
#         mu = base_output["preds"]["mask1"]
#         kl_loss_2 = self._kl_loss(mu, feat)
#
#
#         features, losses = self._encode_and_perturb(x, features, **kwargs)
#         # z1_tilde = features["tz1"]
#         # z2_tilde = features["tz2"]
#         # z1 = features["ez1"]
#         # z2 = features["ez2"]
#
#
#         # mu, feat = self.main._compute_logits(z1_tilde, z2, features["na_z1"], features["na_z2"])
#         # mu, feat = self.main._compute_logits(z1, z2_tilde, features["na_z1"], features["na_z2"])
#
#
#         losses.update({"kl_synergy_1": kl_loss_1, "kl_synergy_2": kl_loss_2})
#         return losses
#
#     def compute_training_losses(self, x, base_output, **kwargs):
#
#         losses = self._kl_pass(x, base_output["features"], **kwargs)
#
#         # z1, z2 = base_output["features"]["z1"], base_output["features"]["z2"]
#         # infonce = nt_xent_loss(z1, z2, temperature=1.0)
#         # losses["sl_sqdiff"] = (kl1 - kl2).pow(2.0).mean() * self.synergy_weight*10000
#         # losses["infonce"] = infonce * self.contrastive_weight
#
#         if self.training:
#             if "current_step" in kwargs:
#                 wandb.log(losses, step=kwargs.get("current_step", 0)+1)
#
#         return losses


# class TF_Fusion_Transformer(nn.Module):
#     """
#     PyTorch Transformer fusion (batch_first) with:
#       - modality tokens + CLS token
#       - optional per-modality padding masks (att_mask1/2/3) combined + CLS slot prepended
#       - optional attn_mask (S,S) expanded to include CLS -> (S+1,S+1)
#     Mask conventions:
#       - src_key_padding_mask / att_mask*: (B,S) bool where True = PAD/IGNORE
#       - attn_mask: (S,S) bool True=BLOCK or float with -inf=BLOCK, 0=ALLOW
#     """
#
#     def __init__(
#         self,
#         input_dim: int,
#         dim: int,
#         layers: int,
#         output_dim: int,
#         nhead: int = 8,
#         dropout: float = 0.1,
#         ff_mult: int = 4,
#         norm_first: bool = True,
#         activation: str = "gelu",
#     ):
#         super().__init__()
#
#         self.in_proj = nn.Identity() if input_dim == dim else nn.Linear(input_dim, dim)
#
#         enc_layer = nn.TransformerEncoderLayer(
#             d_model=dim,
#             nhead=nhead,
#             dim_feedforward=ff_mult * dim,
#             dropout=dropout,
#             batch_first=True,
#             norm_first=norm_first,
#             activation=activation,
#         )
#         self.common_net = nn.TransformerEncoder(enc_layer, num_layers=layers)
#
#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#
#         self.mod_0_token = nn.Parameter(torch.randn(1, 1, input_dim))
#         self.mod_1_token = nn.Parameter(torch.randn(1, 1, input_dim))
#         self.mod_2_token = nn.Parameter(torch.randn(1, 1, input_dim))
#
#         self.common_fc = nn.Linear(dim, output_dim)
#
#     @staticmethod
#     def _prepend_cls_padding(pad_mask: torch.Tensor, device) -> torch.Tensor:
#         if pad_mask is None:
#             return None
#         pad_mask = pad_mask.to(device=device, dtype=torch.bool)
#         cls_col = torch.zeros((pad_mask.shape[0], 1), device=device, dtype=torch.bool)
#         return torch.cat([cls_col, pad_mask], dim=1)  # (B, S+1)
#
#     @staticmethod
#     def _prepend_cls_attn(attn_mask: torch.Tensor, device) -> torch.Tensor:
#         if attn_mask is None:
#             return None
#         attn_mask = attn_mask.to(device=device)
#         S = attn_mask.shape[0]
#         if attn_mask.dtype == torch.bool:
#             out = torch.zeros((S + 1, S + 1), device=device, dtype=torch.bool)
#             out[1:, 1:] = attn_mask
#             return out
#         else:
#             out = torch.zeros((S + 1, S + 1), device=device, dtype=attn_mask.dtype)
#             out[1:, 1:] = attn_mask
#             return out
#
#     def _build_src_key_padding_mask(self, masks, lengths, B, device):
#         """
#         masks:   list like [m1, m2, m3], each either None or (B, Li) bool (True=pad/ignore)
#         lengths: list like [L1, L2, L3] matching the concatenated segments
#         Returns:
#           None if all masks are None,
#           else (B, 1 + sum(lengths)) bool with CLS prepended as False.
#         """
#         if all(m is None for m in masks):
#             return None
#
#         parts = []
#         for m, L in zip(masks, lengths):
#             if L == 0:
#                 continue
#             if m is None:
#                 parts.append(torch.zeros((B, L), device=device, dtype=torch.bool))  # keep all
#             else:
#                 parts.append(m.to(device=device, dtype=torch.bool))
#                 if parts[-1].shape != (B, L):
#                     raise ValueError(f"Mask shape {parts[-1].shape} does not match expected {(B, L)}")
#
#         combined = torch.cat(parts, dim=1) if parts else None
#         if combined is None:
#             return None
#
#         cls_pad = torch.zeros((B, 1), device=device, dtype=torch.bool)  # CLS never ignored
#         return torch.cat([cls_pad, combined], dim=1)  # (B, 1+S_total)
#
#     def _debug_print_ignored(self, masks, lengths, names=("att_mask1", "att_mask2", "att_mask3")):
#         """
#         Print ignored-token stats per modality.
#
#         masks:   list of per-modality padding masks, each either None or (B, Li)
#                  Convention: True = PAD/IGNORE
#         lengths: list of Li matching the concatenated segments
#         names:   modality names (same length as masks/lengths)
#
#         If at least one mask is provided, None masks are treated as all-False of the proper size.
#         Reports % ignored across the whole batch for each modality.
#         """
#         if all(m is None for m in masks):
#             print("[mask dbg] all modality masks are None -> no ignored tokens (no masking).")
#             return
#
#         # infer B/device
#         B, device = None, None
#         for m in masks:
#             if m is not None:
#                 B, device = m.shape[0], m.device
#                 break
#
#         if B is None:
#             print("[mask dbg] no usable masks found.")
#             return
#
#         for m, L, name in zip(masks, lengths, names):
#             if L == 0:
#                 print(f"[mask dbg] {name}: length=0 (skipped)")
#                 continue
#
#             if m is None:
#                 mm = torch.zeros((B, L), device=device, dtype=torch.bool)
#                 provided = False
#             else:
#                 mm = m.to(device=device, dtype=torch.bool)
#                 provided = True
#                 if mm.shape != (B, L):
#                     raise ValueError(f"[mask dbg] {name}: expected mask shape {(B, L)} but got {tuple(mm.shape)}")
#
#             ignored = mm.sum().item()
#             total = B * L
#             pct = 100.0 * ignored / max(total, 1)
#
#             # per-sample distribution (optional but useful)
#             ignored_per_row = mm.sum(dim=1).float()
#             pct_min = 100.0 * ignored_per_row.min().item() / max(L, 1)
#             pct_mean = 100.0 * ignored_per_row.mean().item() / max(L, 1)
#             pct_max = 100.0 * ignored_per_row.max().item() / max(L, 1)
#
#             print(
#                 f"[mask dbg] {name}: provided={provided} "
#                 f"ignored={int(ignored)}/{int(total)} ({pct:.2f}%) | "
#                 f"per-sample ignored% (min/mean/max)={pct_min:.2f}/{pct_mean:.2f}/{pct_max:.2f}"
#             )
#
#     def forward_unimodal(self, x, direction="x1", **kwargs):
#         if direction == "x0":
#             x0 = self.mod_0_token.repeat(x0.shape[0], x0.shape[1], 1) + x0
#             feat = self.in_proj(x0)  # (B, S, dim)
#         elif
#             feat = self.in_proj(x)               # (B, S_total, dim)
#
#         cls = self.cls_token.repeat(feat.shape[0], 1, 1)  # (B,1,dim)
#         feat = torch.cat([cls, feat], dim=1)              # (B,1+S_total,dim)
#
#         m = kwargs.get("att_mask", None)
#         feat = self.common_net(feat, src_key_padding_mask=m)
#
#         cls_feat = feat[:, 0]
#         pred = self.common_fc(cls_feat)
#
#         if kwargs.get("return_all", False):
#             return pred, cls_feat, feat
#         return pred
#
#     def forward(self, x, **kwargs):
#         # ---- inputs + modality tokens ----
#         x0, x1 = x[0], x[1]  # expected (B,S,input_dim)
#         x0 = self.mod_0_token.repeat(x0.shape[0], x0.shape[1], 1) + x0
#         x1 = self.mod_1_token.repeat(x1.shape[0], x1.shape[1], 1) + x1
#         xlist = [x0, x1]
#
#         if len(x) > 2:
#             x2 = x[2]
#             x2 = self.mod_2_token.repeat(x2.shape[0], x2.shape[1], 1) + x2
#             xlist.append(x2)
#
#         if kwargs.get("detach_a", False):
#             xlist[0] = xlist[0].detach()
#         if kwargs.get("detach_v", False):
#             xlist[1] = xlist[1].detach()
#
#         feat = torch.cat(xlist, dim=1)          # (B, S_total, input_dim)
#         feat = self.in_proj(feat)               # (B, S_total, dim)
#
#         cls = self.cls_token.repeat(feat.shape[0], 1, 1)  # (B,1,dim)
#         feat = torch.cat([cls, feat], dim=1)              # (B,1+S_total,dim)
#
#         B = feat.shape[0]
#         lengths = [xi.shape[1] for xi in xlist]  # [L1, L2, (L3)]
#         m1 = kwargs.get("att_mask1", None)
#         m2 = kwargs.get("att_mask2", None)
#         m3 = kwargs.get("att_mask3", None) if len(xlist) > 2 else None
#
#         src_key_padding_mask = kwargs.get("src_key_padding_mask", None)
#         if src_key_padding_mask is None:
#             src_key_padding_mask = self._build_src_key_padding_mask(
#                 masks=[m1, m2, m3],
#                 lengths=lengths,
#                 B=B,
#                 device=feat.device,
#             )
#         else:
#             src_key_padding_mask = src_key_padding_mask.to(device=feat.device, dtype=torch.bool)
#             # if user forgot CLS, prepend it
#             if src_key_padding_mask.shape[1] == feat.shape[1] - 1:
#                 cls_pad = torch.zeros((B, 1), device=feat.device, dtype=torch.bool)
#                 src_key_padding_mask = torch.cat([cls_pad, src_key_padding_mask], dim=1)
#
#         # self._debug_print_ignored([m1, m2, m3], lengths, names=("att_mask1", "att_mask2", "att_mask3"))
#
#         # ---- transformer ----
#         feat = self.common_net(feat, src_key_padding_mask=src_key_padding_mask)
#
#         cls_feat = feat[:, 0]
#         pred = self.common_fc(cls_feat)
#
#         if kwargs.get("return_all", False):
#             return pred, cls_feat, feat
#         return pred
#
# import torch
# import torch.nn as nn


from typing import Dict, Optional, Union, List, Tuple

class TF_Fusion_Transformer(nn.Module):
    """
    Accepts x as:
      - Tensor (B,S,D)
      - list/tuple of tensors
      - dict {mod_id: tensor}

    Also accepts masks either as:
      - att_masks: dict {mod_id: (B,Si)} OR list aligned with modalities
      - OR legacy att_mask1/att_mask2/att_mask3 kwargs (works with dict/list inputs)
      - OR src_key_padding_mask: (B,S_total) or (B,1+S_total)

    Adds:
      - modality tokens (per modality, in input_dim space)
      - CLS token
      - learnable positional embeddings (in dim space)
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        layers: int,
        output_dim: int,
        nhead: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        norm_first: bool = True,
        activation: str = "gelu",
        max_seq_len: int = 4096,
        max_modalities: int = 16,
        use_pos_emb: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.max_modalities = max_modalities
        self.use_pos_emb = use_pos_emb

        self.in_proj = nn.Identity() if input_dim == dim else nn.Linear(input_dim, dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=nhead,
            dim_feedforward=ff_mult * dim,
            dropout=dropout,
            batch_first=True,
            norm_first=norm_first,
            activation=activation,
        )
        self.common_net = nn.TransformerEncoder(enc_layer, num_layers=layers)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # index by modality id
        self.mod_tokens = nn.Parameter(torch.randn(max_modalities, 1, 1, input_dim))

        if use_pos_emb:
            self.pos_emb = nn.Parameter(torch.randn(1, 1 + max_seq_len, dim) * 0.02)
        else:
            self.register_parameter("pos_emb", None)

        self.common_fc = nn.Linear(dim, output_dim)

    @staticmethod
    def _prepend_cls_padding(pad_mask: torch.Tensor, device) -> torch.Tensor:
        if pad_mask is None:
            return None
        pad_mask = pad_mask.to(device=device, dtype=torch.bool)
        cls_col = torch.zeros((pad_mask.shape[0], 1), device=device, dtype=torch.bool)
        return torch.cat([cls_col, pad_mask], dim=1)  # (B,1+S)

    @staticmethod
    def _prepend_cls_attn(attn_mask: torch.Tensor, device) -> torch.Tensor:
        if attn_mask is None:
            return None
        attn_mask = attn_mask.to(device=device)
        if attn_mask.ndim != 2 or attn_mask.shape[0] != attn_mask.shape[1]:
            raise ValueError(f"attn_mask must be square (S,S); got {tuple(attn_mask.shape)}")
        S = attn_mask.shape[0]
        if attn_mask.dtype == torch.bool:
            out = torch.zeros((S + 1, S + 1), device=device, dtype=torch.bool)
            out[1:, 1:] = attn_mask
            return out
        out = torch.zeros((S + 1, S + 1), device=device, dtype=attn_mask.dtype)
        out[1:, 1:] = attn_mask
        return out

    def _add_positional_embeddings(self, feat: torch.Tensor) -> torch.Tensor:
        if not self.use_pos_emb:
            return feat
        S1 = feat.shape[1]  # includes CLS
        if S1 > self.pos_emb.shape[1]:
            raise ValueError(
                f"Sequence too long for pos_emb: got (1+S)={S1}, "
                f"but pos_emb supports up to {self.pos_emb.shape[1]}."
            )
        return feat + self.pos_emb[:, :S1, :].to(device=feat.device, dtype=feat.dtype)

    def _normalize_inputs(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[int, torch.Tensor]],
    ) -> Tuple[List[int], List[torch.Tensor]]:
        if torch.is_tensor(x):
            return [0], [x]

        if isinstance(x, dict):
            if len(x) == 0:
                raise ValueError("Empty multimodal dict input.")
            mod_ids = sorted(x.keys())
            xlist = [x[mid] for mid in mod_ids]
            return mod_ids, xlist

        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                raise ValueError("Empty multimodal list/tuple input.")
            mod_ids = list(range(len(x)))
            return mod_ids, list(x)

        raise TypeError("x must be a Tensor, a list/tuple of Tensors, or a dict {mod_id: Tensor}.")

    def _normalize_per_mod_masks_from_kwargs(
        self,
        kwargs: dict,
        mod_ids: List[int],
    ) -> Optional[Union[Dict[int, torch.Tensor], List[Optional[torch.Tensor]]]]:
        """
        Supports your legacy call:
          att_mask1, att_mask2, att_mask3

        We interpret these as "first/second/third modality in sorted(mod_ids)".
        So with x={0:...,1:...}, att_mask1 -> mod_id 0, att_mask2 -> mod_id 1.
        """
        m1 = kwargs.get("att_mask1", None)
        m2 = kwargs.get("att_mask2", None)
        m3 = kwargs.get("att_mask3", None)

        if m1 is None and m2 is None and m3 is None:
            return None

        masks = [m1, m2, m3]
        masks = masks[: len(mod_ids)]
        # return list aligned with modalities (ordered by mod_ids)
        return masks

    def _combine_att_masks(
        self,
        masks_ordered: List[Optional[torch.Tensor]],
        lengths: List[int],
        B: int,
        device,
    ) -> Optional[torch.Tensor]:
        if all(m is None for m in masks_ordered):
            return None

        parts = []
        for m, L in zip(masks_ordered, lengths):
            if L == 0:
                continue
            if m is None:
                parts.append(torch.zeros((B, L), device=device, dtype=torch.bool))
            else:
                mm = m.to(device=device, dtype=torch.bool)
                if mm.shape != (B, L):
                    raise ValueError(f"Mask shape {tuple(mm.shape)} does not match expected {(B, L)}")
                parts.append(mm)

        combined = torch.cat(parts, dim=1) if parts else None
        if combined is None:
            return None
        return self._prepend_cls_padding(combined, device=device)  # (B,1+S_total)

    def forward(
        self,
        x: Union[torch.Tensor, List[torch.Tensor], Tuple[torch.Tensor, ...], Dict[int, torch.Tensor]],
        **kwargs,
    ):
        """
        Works with your call:
          self.enc_2({0:na_z1, 1:na_z2}, att_mask1=..., att_mask2=...)

        Also supports:
          att_masks=... (dict or list)
          src_key_padding_mask=...
          attn_mask=...
          detach_mods=...
          return_all=True
        """
        mod_ids, xlist = self._normalize_inputs(x)
        if len(mod_ids) > self.max_modalities:
            raise ValueError(f"Got {len(mod_ids)} modalities but max_modalities={self.max_modalities}")

        # Validate shapes + batch consistency
        B = xlist[0].shape[0]
        for mid, xi in zip(mod_ids, xlist):
            if xi.ndim != 3:
                raise ValueError(f"Modality {mid} must be (B,S,D); got {tuple(xi.shape)}")
            if xi.shape[0] != B:
                raise ValueError(f"Batch mismatch: expected B={B} but modality {mid} has B={xi.shape[0]}")
            if xi.shape[2] != self.input_dim:
                raise ValueError(
                    f"Input dim mismatch: expected input_dim={self.input_dim} but modality {mid} has D={xi.shape[2]}"
                )

        device = xlist[0].device

        # Detach controls
        detach_mods = set(kwargs.get("detach_mods", []))
        if kwargs.get("detach_a", False):
            detach_mods.add(mod_ids[0] if len(mod_ids) > 0 else 0)
        if kwargs.get("detach_v", False) and len(mod_ids) > 1:
            detach_mods.add(mod_ids[1])

        # Add modality tokens
        xlist_tok = []
        for mid, xi in zip(mod_ids, xlist):
            if mid < 0 or mid >= self.max_modalities:
                raise ValueError(f"modality id {mid} out of range [0, {self.max_modalities-1}]")
            if mid in detach_mods:
                xi = xi.detach()
            tok = self.mod_tokens[mid].to(device=xi.device, dtype=xi.dtype).squeeze(0)  # (1,1,input_dim)
            xlist_tok.append(xi + tok)  # (B,S,input_dim)

        lengths = [xi.shape[1] for xi in xlist_tok]
        S_total = sum(lengths)

        # # --- masks ---
        src_key_padding_mask = kwargs.get("src_key_padding_mask", None)
        # if src_key_padding_mask is not None:
        #     src_key_padding_mask = src_key_padding_mask.to(device=device, dtype=torch.bool)
        #     if src_key_padding_mask.shape == (B, S_total):
        #         src_key_padding_mask = self._prepend_cls_padding(src_key_padding_mask, device=device)
        #     elif src_key_padding_mask.shape != (B, 1 + S_total):
        #         raise ValueError(
        #             f"src_key_padding_mask must be (B,S_total) or (B,1+S_total). "
        #             f"Got {tuple(src_key_padding_mask.shape)}; expected {(B, S_total)} or {(B, 1+S_total)}."
        #         )
        # else:
        #     # Prefer explicit att_masks=..., otherwise accept legacy att_mask1/2/3
        #     att_masks = kwargs.get("att_masks", None)
        #     if att_masks is None:
        #         masks_ordered = self._normalize_per_mod_masks_from_kwargs(kwargs, mod_ids)
        #     else:
        #         # att_masks can be dict or list/tuple
        #         if isinstance(att_masks, dict):
        #             masks_ordered = [att_masks.get(mid, None) for mid in mod_ids]
        #         elif isinstance(att_masks, (list, tuple)):
        #             if len(att_masks) != len(mod_ids):
        #                 raise ValueError(f"att_masks length {len(att_masks)} must match #modalities {len(mod_ids)}")
        #             masks_ordered = list(att_masks)
        #         else:
        #             raise TypeError("att_masks must be a dict {mod_id: mask} or a list/tuple aligned with modalities.")
        #
        #     if masks_ordered is None:
        #         src_key_padding_mask = None
        #     else:
        #         src_key_padding_mask = self._combine_att_masks(masks_ordered, lengths, B, device=device)

        attn_mask = kwargs.get("attn_mask", None)
        if attn_mask is not None:
            if attn_mask.shape != (S_total, S_total):
                raise ValueError(f"attn_mask must be (S_total,S_total)={(S_total,S_total)}; got {tuple(attn_mask.shape)}")
            attn_mask = self._prepend_cls_attn(attn_mask, device=device)

        # --- forward ---
        feat = torch.cat(xlist_tok, dim=1)      # (B,S_total,input_dim)
        feat = self.in_proj(feat)               # (B,S_total,dim)

        cls = self.cls_token.to(device=device, dtype=feat.dtype).repeat(B, 1, 1)  # (B,1,dim)
        feat = torch.cat([cls, feat], dim=1)                                      # (B,1+S_total,dim)
        feat = self._add_positional_embeddings(feat)

        feat = self.common_net(feat, mask=attn_mask, src_key_padding_mask=src_key_padding_mask)

        cls_feat = feat[:, 0]
        pred = self.common_fc(cls_feat)

        if kwargs.get("return_all", False):
            return pred, cls_feat, feat
        return pred



class UnimodalWrapper(nn.Module):
    """
    Renamed version of your FusionIBModel:
      enc_0, enc_1 : modality encoders (same as before)
      enc_2        : fusion trunk (was common_fc_1)
      enc_3        : fusion projector (was common_fc_2)
      enc_4        : fusion head (was mu_head)
      synergy      : SynIB_VAE (or SynIB) using main=self
    """
    # def __init__(self, args, encs):
    #     super().__init__()
    #     self.args = args
    #     self.cls_type = _cfg(args, "cls_type")
    #     self.norm_decision = _cfg(args, "norm_decision", False)
    #
    #     # main encoders
    #     self.enc_0 = encs[0]
    #     self.enc_1 = LinearHead(args)

    def __init__(self, args, encs):
        super().__init__()
        self.args = args
        self.cls_type = _cfg(args, "cls_type")
        self.norm_decision = _cfg(args, "norm_decision", False)

        self.enc_0 = encs[0]
        self.enc_1 = LinearHead(args)

    def _get_features(self, x, **kwargs):
        out0 = self.enc_0(x, **kwargs)
        z1, na_z1 = _as_tensor_features(out0)
        z1 = F.layer_norm(z1, (z1.shape[-1],))
        na_z1 = F.layer_norm(na_z1, (na_z1.shape[-1],))
        return z1, na_z1

    def _compute_logits_unimodal(self, z, na_z=None, direction="z1", detach_it=True,**kwargs):
        if detach_it:
            this_z = z.detach()
        else:
            this_z = z
        pred = self.enc_1(this_z, **kwargs)

        return pred
    def _base_forward(self, x, **kwargs):
        z1, na_z1 = self._get_features(x, **kwargs)

        uni_pred_1 = self._compute_logits_unimodal(z1, na_z1, direction="z1")

        return {
            "preds": {
                "combined": uni_pred_1,
            },
            "features": {
                "combined": z1,
                "z1": z1,
                "na_z1":na_z1,
            },
            "losses": {},
        }

    def forward(self, x, **kwargs):

        output =  self._base_forward(x, **kwargs)

        return output

class FeatureStatsMasker(nn.Module):
    def __init__(self, d1, ema_beta=0.99, eps=1e-6, device=None, dtype=None):
        super().__init__()
        factory_kwargs = dict(device=device, dtype=dtype)
        self.d1 = int(d1)
        self.ema_beta = float(ema_beta)
        self.eps = float(eps)

        # EMA of E[x] and E[x^2]
        self.register_buffer("ex",  torch.zeros(self.d1, **factory_kwargs))
        self.register_buffer("ex2", torch.zeros(self.d1, **factory_kwargs))
        self.register_buffer("n",   torch.zeros((), **factory_kwargs))  # number of updates

    @torch.no_grad()
    def ema_update(self, z: torch.Tensor):
        """
        z1: (..., F) where ... can be (B,) or (B,T) or (B,T,...) etc.
        Keeps EMA per feature over all leading dims.
        """
        x = z.detach()
        if x.numel() == 0:
            return

        # collapse all dims except feature dim
        if x.dim() == 1:
            x = x[None, :]  # (1, F)
        else:
            x = x.reshape(-1, x.shape[-1])  # (N, F)

        if x.shape[-1] != self.d1:
            raise ValueError(f"Expected feature dim {self.d1}, got {x.shape[-1]}")

        batch_ex  = x.mean(0)               # E[x]
        batch_ex2 = (x * x).mean(0)         # E[x^2]

        # standard EMA; first update copies batch stats (no lag)
        b = self.ema_beta if self.n.item() > 0 else 0.0
        a = 1.0 - b
        self.ex.lerp_(batch_ex,  a)
        self.ex2.lerp_(batch_ex2, a)
        self.n.add_(1)

    def feature_stats(self):
        """
        Returns (mean, var) per feature.
        """
        mu = self.ex
        var = (self.ex2 - mu * mu).clamp_min(self.eps)
        return mu, var

    def noise_like(self, z: torch.Tensor, noise_scale=1.0):
        mu, var = self.feature_stats()
        # broadcast to z1 shape
        shape = [1] * (z.dim() - 1) + [-1]
        mu = mu.view(*shape)
        std = (var.sqrt() * float(noise_scale)).view(*shape)
        return mu + torch.randn_like(z) * std
class SynIB(nn.Module):
    """
    Same interface as SynIB, but uses pretrained VAEs as skew samplers.
    IMPORTANT: VAEs are used under torch.no_grad(); no VAE losses are returned.

    Expected encs layout:
      encs[0] = modality-1 encoder (main model uses)
      encs[1] = modality-2 encoder (main model uses)
      encs[2] = VAE posterior for z1
      encs[3] = VAE decoder   for z1
      encs[4] = VAE posterior for z2
      encs[5] = VAE decoder   for z2
    """
    def __init__(self, args, encs, main):
        super().__init__()
        object.__setattr__(self, "main", main)

        self.args = args
        self.perturb = _cfg(args, "perturb", {}) or {}
        self.reestimate_features = bool(self.perturb.get("reestimate_features", False))

        bias = _cfg(args, "bias_infusion", {}) or {}
        self.synergy_weight = float(bias.get("l", 0.0))
        self.contrastive_weight = float(bias.get("contrcoeff", 0.0) or 0.0)
        self.synergy_type = getattr(args, "synergy_type", "gaussian")  # "gaussian" or "dirichlet"


        fc_inner = int(_cfg(args, "fc_inner"))
        num_classes = int(_cfg(args, "num_classes"))

        if self.synergy_type == "gaussian":
            self.logvar_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = None
        elif self.synergy_type == "dirichlet":
            self.evidence_head = nn.Linear(fc_inner, num_classes)
            self.dirichlet_prior_conc = float(_cfg(args, "dirichlet_prior_conc", 1.0))
        else:
            raise ValueError(f"Unknown synergy_type: {self.synergy_type}")

        self.cls_type = _cfg(args, "cls_type")

        self.p = float(self.perturb.get("p_min", 0.5))
        self.noise_std = float(self.perturb.get("noise_std", 1.0))
        self.K = int(self.perturb.get("num_samples", 1))
        self.fill = self.perturb.get("fill", "ema")
        self.p_type = self.perturb.get("type", "diff")

        self.p_min = float(self.perturb.get("p_min", 0.3))
        self.p_max = float(self.perturb.get("p_max", 0.9))
        self.cosine_s = 0.008

        if self.cls_type == "mlp":
            self.stats_z1 = FeatureStatsMasker(d1=512, ema_beta=0.99)
            self.stats_z2 = FeatureStatsMasker(d1=512, ema_beta=0.99)
        elif self.cls_type == "tf":
            self.stats_na_z1 = FeatureStatsMasker(d1=512, ema_beta=0.99)
            self.stats_na_z2 = FeatureStatsMasker(d1=512, ema_beta=0.99)


    @staticmethod
    def _gaussian_kl(mu, logvar):
        return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, dim=1).mean()

    @staticmethod
    def _dirichlet_kl(alpha, prior_conc=1.0):
        alpha0 = torch.full_like(alpha, prior_conc) if isinstance(prior_conc, float) else prior_conc
        alpha0_sum = alpha0.sum(dim=1, keepdim=True)
        alpha_sum = alpha.sum(dim=1, keepdim=True)

        lgamma = torch.lgamma
        digamma = torch.digamma

        logB_alpha = torch.sum(lgamma(alpha), dim=1) - lgamma(alpha_sum.squeeze(1))
        logB_alpha0 = torch.sum(lgamma(alpha0), dim=1) - lgamma(alpha0_sum.squeeze(1))

        term1 = logB_alpha0 - logB_alpha
        term2 = torch.sum((alpha - alpha0) * (digamma(alpha) - digamma(alpha_sum)), dim=1)
        return (term1 + term2).mean()

    # @staticmethod
    # def _cat_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    #     p = F.softmax(p, dim=-1)
    #     q = F.softmax(q, dim=-1)
    #     p = p.clamp_min(eps)
    #     q = q.clamp_min(eps)
    #     return (p * (p.log() - q.log())).sum(dim=-1).mean()

    # @staticmethod
    # def kl_to_uniform_multiclass_from_logits(logits: torch.Tensor) -> torch.Tensor:
    #     logp = F.log_softmax(logits, dim=-1)
    #     p = logp.exp()
    #     K = logits.size(-1)
    #     return (p * logp).sum(dim=-1).mean() + torch.log(torch.tensor(float(K), device=logits.device))

    # def _logit_kl(self, pred_logits, target_logits):
    #     return F.kl_div(
    #         F.log_softmax(pred_logits, dim=-1),
    #         F.softmax(target_logits, dim=-1),
    #         reduction="batchmean"
    #     )

    def _get_diff_p(self, t):
        """
        Returns p(t) shaped as a scalar tensor.
        p(t) ~= alpha_bar(t) with cosine schedule:
          alpha_bar(t) = cos^2( ((t/T)+s)/(1+s) * pi/2 )
        This decreases from ~1 to ~0, matching increasing noise over time.

        Scale to [p_min, p_max].
        """
        u = (torch.as_tensor(t) / (self.K - 1)).clamp(0.0, 1.0)

        s = self.cosine_s
        alpha_bar = torch.cos(((u + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2  # ~1 -> ~0
        p = self.p_min + (self.p_max - self.p_min) * alpha_bar
        return p

    def get_random_mask_multiclass( self, features, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Sequential, decoupled mask learning:

          Phase A (learn g0): corrupt x0 only, keep x1 clean.
            objective uses div_f - alpha * div_u0  + sparsity(g0)

          Phase B (learn g1): corrupt x1 only, keep x0 clean.
            objective uses div_f - alpha * div_u1  + sparsity(g1)

        No "other" mask is used while learning a given mask.
        """

        z1, z2, na_z1, na_z2 = features["z1"], features["z2"], features["na_z1"], features["na_z2"]

        def repeat_k(z):
            return z.unsqueeze(0).expand(self.K, *z.shape).reshape(self.K * z.shape[0], *z.shape[1:])

        def make_keep(z, p):
            return (torch.rand_like(z) < p).to(z.dtype)

        def make_keep_token(z, p):
            return (torch.rand_like(z[:,:,0]) < p).to(z.dtype)

        def noise_fn(z, ema):
            if self.fill=="zeros":
                return torch.zeros_like(z)
            elif self.fill == "noise":
                return torch.randn_like(z) * self.noise_std
            elif self.fill == "ema":
                return ema.noise_like(z, self.noise_std)

        def fill_func(z, keep, ema=None):
            eps = z[torch.randperm(z.size(0))]
            # return (1 - keep) * z + keep * noise_fn(z, ema=ema)
            return (1 - keep) * z + keep * eps

        def make_tilde_once(z, ema):
            zK = repeat_k(z)
            if self.fill == "token":
                token_mask = make_keep_token(zK, self.p)
                return zK, None, token_mask
            keep = make_keep(zK, self.p)
            tzK = fill_func(zK, keep, ema)
            return zK, tzK

        def make_tilde_diff(z, ema):

            zK = repeat_k(z)
            if self.fill == "token":
                token_mask = torch.cat([make_keep_token(z, self._get_diff_p(k)) for k in range(self.K)], dim=0)
                return zK, zK, token_mask
            keep = torch.cat([make_keep(z, self._get_diff_p(k)) for k in range(self.K)], dim=0)
            tzK = fill_func(zK, keep, ema)
            return zK, tzK, None

        make_tilde_fn = make_tilde_diff if self.p_type=="diff" else make_tilde_once

        if self.cls_type == "mlp":
            self.stats_z1.ema_update(z1)
            self.stats_z2.ema_update(z2)

            z1K, tz1K, token_mask1 = make_tilde_fn(z1,self.stats_z1)
            z2K, tz2K, token_mask2 = make_tilde_fn(z2,self.stats_z2)
            na_z1K, na_z2K, na_tz1K, na_tz2K = na_z1, na_z2, na_z1, na_z2

        elif self.cls_type == "tf":

            self.stats_na_z1.ema_update(na_z1)
            self.stats_na_z2.ema_update(na_z2)

            na_z1K, na_tz1K, token_mask1 = make_tilde_fn(na_z1,self.stats_na_z1)
            na_z2K, na_tz2K, token_mask2 = make_tilde_fn(na_z2,self.stats_na_z2)
            z1K, tz1K, z2K, tz2K = z1, z1, z2, z2

        return {
            "z1K": z1K,
            "tz1K": tz1K,
            "z2K": z2K,
            "tz2K": tz2K,
            "na_z1K": na_z1K,
            "na_tz1K": na_tz1K,
            "na_z2K": na_z2K,
            "na_tz2K": na_tz2K,
            "mask1": token_mask1,
            "mask2": token_mask2
        }

    def get_learnable_mask_multiclass(self, x, features, preds, **kwargs) -> Dict[str, torch.Tensor]:

        y = kwargs["label"]
        device = y.device
        pcfg = self.args.perturb
        steps = int(getattr(pcfg, "steps", 20))
        lr = float(getattr(pcfg, "lr", 1e-1))
        tau = float(getattr(pcfg, "tau", 1.0))
        noise_std = float(getattr(pcfg, "noise_std", 1.0))
        lsparse = float(getattr(pcfg, "lsparse", 1))
        alpha_unimodal = float(getattr(pcfg, "alpha_unimodal", 1.0))
        method = getattr(pcfg, "method", "adv_unimodal")
        gate_shape = getattr(pcfg, "gate_shape", "per_example")  # "global" or "per_example"
        hard = bool(getattr(pcfg, "hard", True))
        hard_thresh = float(getattr(pcfg, "hard_thresh", 0.5))
        fill_mode = getattr(pcfg, "fill", "ema")  # e.g. "noise" / "zeros" / "ema" / "token" (token optional)

        # -------------------------
        # 2) Helpers (small + local)
        # -------------------------
        def _init_gate_logits(B: int, d: int) -> torch.Tensor:
            return torch.ones((B, d), device=device)

        def _gate_probs_from_logits(ell: torch.Tensor, B: int, d: int) -> torch.Tensor:

            if gate_shape == "global":
                return torch.sigmoid(ell / tau).view(1, d)
            return torch.sigmoid(ell / tau).view(B, d)

        def _mask_objective(method, which: Literal["z1", "z2"], p_f_clean: torch.Tensor,
                           p_u0_clean: torch.Tensor, p_u1_clean: torch.Tensor, p_f_t: torch.Tensor, p_u_t: torch.Tensor,
                           y: torch.Tensor, sparsity: torch.Tensor, lsparse: float,
                           alpha_unimodal: float) -> torch.Tensor:
            if method == "adv_unimodal":
                return -F.cross_entropy(p_u_t, y) + float(
                lsparse) * sparsity
            raise ValueError(f"Unknown method {method}")
        def _apply_destroy(z: torch.Tensor, g: torch.Tensor, ema_stats=None, inv_mask=False) -> torch.Tensor:
            if g.dim() == 1:
                g_ = g.view(1, -1)
            else:
                g_ = g
            g_ = g_.to(z.device).type_as(z)

            if fill_mode == "zeros":
                eps = torch.zeros_like(z)
            elif fill_mode == "ema":
                eps = ema_stats.noise_like(z, noise_std) if ema_stats is not None else (torch.randn_like(z) * noise_std)
            elif fill_mode == "shuffle":
                eps = z[torch.randperm(z.size(0))]
            else:
                raise ValueError(f"Unknown fill mode {fill_mode}")

            if inv_mask:
                return g_ * z + (1 - g_) * eps
            return (1 - g_) * z + g_ * eps
        def acc_from_logits_multiclass(logits: torch.Tensor, y: torch.Tensor) -> float:

            preds = logits.argmax(dim=1)
            return float((preds == y).float().mean().item())
        def _forward_probs(feat_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            p_f, _ = self.main._compute_logits(feat_dict["z1"], feat_dict["z2"], feat_dict["na_z1"], feat_dict["na_z2"])  # <-- implement in your class
            p_u1 = self.main.enc_0.forward_uni(feat_dict["z1"], feat_dict["na_z1"], detach_pred=False)  # <-- implement in your class
            p_u2 = self.main.enc_1.forward_uni(feat_dict["z2"], feat_dict["na_z2"], detach_pred=False)  # <-- implement in your class
            return p_f, p_u1, p_u2
        def gate_to_hard_mask(g: torch.Tensor, hard_thresh: float, ref_shape: torch.Size,
                              inv_mask: bool = False) -> torch.Tensor:
            pred = (g >= float(hard_thresh)) if g.dtype.is_floating_point else g.bool()
            if pred.dim() == 1: pred = pred.view(1, -1)
            if pred.shape[0] == 1 and ref_shape[0] > 1: pred = pred.expand(ref_shape[0], ref_shape[1])
            if inv_mask: pred = ~pred
            return pred

        def hard_concrete(ell, tau=2 / 3, l=-0.1, r=1.1, training=True):
            if training:
                u = torch.rand_like(ell)
                s = torch.sigmoid((ell + torch.log(u) - torch.log(1 - u)) / tau)
            else:
                s = torch.sigmoid(ell)  # deterministic at test time
            s_bar = s * (r - l) + l
            z = s_bar.clamp(0, 1)
            return z

        def freeze_model_(m: nn.Module) -> None:
            for p in m.parameters(): p.requires_grad_(False)

        req = [p.requires_grad for p in self.main.parameters()]
        freeze_model_(self.main)
        try:

            p_u1_clean, p_u2_clean, p_f_clean = preds["c"], preds["g"], preds["combined"]
            z1, z2, na_z1, na_z2 = features["z1"], features["z2"], features["na_z1"], features["na_z2"]
            z1 = z1.detach(); z2 = z2.detach()
            p_u1_clean = p_u1_clean.detach(); p_u2_clean = p_u2_clean.detach()
            p_f_clean = p_f_clean.detach()

            #make a new forward pass
            p_f_clean, p_u1_clean, p_u2_clean = _forward_probs(features)

            # if self.cls_type == "mlp":
            #     self.stats_z1.ema_update(z1)
            #     self.stats_z2.ema_update(z2)
            # elif self.cls_type == "tf":
            #     self.stats_na_z1.ema_update(na_z1)
            #     self.stats_na_z2.ema_update(na_z2)
            #
            with torch.enable_grad():
                this_z1 = z1 if self.cls_type == "mlp" else na_z1
                this_key = 'z1' if self.cls_type == "mlp" else 'na_z1'
                this_stats1 = self.stats_z1 if self.cls_type == "mlp" else self.stats_na_z1
                this_stats1.ema_update(this_z1)
                ell1 = torch.nn.Parameter(torch.ones(this_z1.shape, device=device), requires_grad=True)
                opt1 = torch.optim.Adam([ell1], lr=lr)

                for i in range(steps):
                    g1 = torch.sigmoid(ell1 / tau)
                    tz1 = _apply_destroy(this_z1, g1, ema_stats=this_stats1, inv_mask=True)
                    feat_t = dict(features)
                    feat_t[this_key] = tz1

                    p_f_t, p_u_t, _ = _forward_probs(feat_t)
                    sparsity = (1-g1).mean()
                    obj1 = -F.cross_entropy(p_u_t, y) + float(lsparse) * sparsity
                    opt1.zero_grad(set_to_none=True)
                    obj1.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_([ell1], 1.0)
                    opt1.step()
                    # if i==0 or i==steps-1:
                    #     with torch.no_grad():
                    #         pred = _gate_probs_from_logits(ell1, B, d1)
                    #         sp = float((1-pred).float().mean().item())
                    #         tz1 = _apply_destroy(z1, pred, ema_stats=getattr(self, "stats_z1", None), inv_mask=True)
                    #         feat_t = dict(this_features)
                    #         feat_t["z1"] = tz1
                    #         p_f_t, p_u1_t, p_u2_t = _forward_probs(feat_t)
                    #
                    #         acc_u1_t = acc_from_logits_multiclass(p_u1_t, y)
                    #         acc_u_clean = acc_from_logits_multiclass(p_u1_clean, y)
                    #
                    #     print(f"Z1 -{i}- Obj: {obj}, Sparsity:{sparsity}, uni_perf_clean:{acc_u_clean}, uni_perf_t: {acc_u1_t}, hard_sparsity: {sp}")

                g1_final = torch.sigmoid(ell1 / tau)


                this_z2 = z2 if self.cls_type == "mlp" else na_z2
                this_key = 'z2' if self.cls_type == "mlp" else 'na_z2'
                this_stats2 = self.stats_z2 if self.cls_type == "mlp" else self.stats_na_z2
                this_stats2.ema_update(this_z2)
                ell2 = torch.nn.Parameter(torch.ones(this_z2.shape, device=device), requires_grad=True)
                opt2 = torch.optim.Adam([ell2], lr=lr)

                for i in range(steps):
                    g2 = torch.sigmoid(ell2 / tau)
                    tz2 = _apply_destroy(this_z2, g2, ema_stats=this_stats2, inv_mask=True)
                    feat_t = dict(features)
                    feat_t[this_key] = tz2

                    p_f_t, _, p_u_t = _forward_probs(feat_t)
                    sparsity = (1-g2).mean()

                    obj2 = -F.cross_entropy(p_u_t, y) + float(lsparse) * sparsity
                    opt2.zero_grad(set_to_none=True)
                    obj2.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_([ell2], 1.0)
                    opt2.step()
                    # if i==0 or i==steps-1:
                    #
                    #     with torch.no_grad():
                    #         pred = _gate_probs_from_logits(ell2, B, d2)
                    #         sp = float((1-pred).float().mean().item())
                    #         tz2 = _apply_destroy(z2, pred, ema_stats=getattr(self, "stats_z2", None), inv_mask=True)
                    #         feat_t = dict(this_features)
                    #         feat_t["z2"] = tz2
                    #         p_f_t, p_u1_t, p_u2_t = _forward_probs(feat_t)
                    #
                    #         acc_u2_t = acc_from_logits_multiclass(p_u2_t, y)
                    #         acc_u_clean = acc_from_logits_multiclass(p_u1_clean, y)
                    #
                    #     print(f"Z2 -{i}- Obj: {obj}, Sparsity:{sparsity}, uni_perf_clean:{acc_u_clean}, uni_perf_t: {acc_u2_t}, hard_sparsity: {sp}")

                g2_final = torch.sigmoid(ell2 / tau)

                # -------------------------
                # 6) Build outputs (keep same keys as your old function)
                # -------------------------
                # Keep K behavior: if you rely on K>1, we can expand later.

        finally:
            self.main.train()
            for p, r in zip(self.main.parameters(), req):
                p.requires_grad_(r)

        if self.cls_type == "mlp":
            z1K, z2K = z1, z2
            tz1K = _apply_destroy(this_z1, g1_final, ema_stats=this_stats1, inv_mask=True)
            tz2K = _apply_destroy(this_z2, g2_final, ema_stats=this_stats2, inv_mask=True)
            na_z1K, na_z2K = na_z1, na_z2
            na_tz1K, na_tz2K = na_z1, na_z2
        elif self.cls_type == "tf":
            na_z1K, na_z2K = na_z1, na_z2
            na_tz1K = _apply_destroy(this_z1, g1_final, ema_stats=this_stats1, inv_mask=True)
            na_tz2K = _apply_destroy(this_z2, g2_final, ema_stats=this_stats2, inv_mask=True)
            z1K, z2K = z1, z2
            tz1K, tz2K = z1, z2

        # For compatibility: if your old code returned token masks, you can interpret these as "corrupt mask"
        # mask1/mask2 shapes should match your expectations:
        # - if fill=="token" you previously returned [K*B,T]; here we return [B,d]
        mask1 = g1_final.detach()
        mask2 = g2_final.detach()

        return {
            "z1K": z1K,
            "tz1K": tz1K,
            "z2K": z2K,
            "tz2K": tz2K,
            "na_z1K": na_z1K,
            "na_tz1K": na_tz1K,
            "na_z2K": na_z2K,
            "na_tz2K": na_tz2K,
            "mask1": mask1,
            "mask2": mask2,
        }


    def _perturb_masked(self, features, **kwargs):
        mask_dict = self.get_random_mask_multiclass(features, **kwargs)
        z1 = features["z1"].detach()
        z2 = features["z2"].detach()
        na_z1 = features.get("na_z1", None)
        na_z2 = features.get("na_z2", None)
        g1 = mask_dict["g0"]
        g2 = mask_dict["g1"]

        baseline1 = torch.zeros_like(z1)
        baseline2 = torch.zeros_like(z2)
        g1 = g1.view(1, -1)
        g2 = g2.view(1, -1)

        z1_tilde = (1 - g1) * z1 + g1 * baseline1
        z2_tilde = (1 - g2) * z2 + g2 * baseline2

        return {
            "z1_tilde": z1_tilde,
            "z2_tilde": z2_tilde,
            "losses": {},
            "masks": mask_dict
        }
    def _kl_loss(self, mu, feat):
        if self.synergy_type == "gaussian":
            logvar = self.logvar_head(feat)
            kl = self._gaussian_kl(mu, logvar)
        else:
            evidence = F.softplus(self.evidence_head(feat))
            alpha = evidence + 1.0
            kl = self._dirichlet_kl(alpha, prior_conc=self.dirichlet_prior_conc)
        return kl * self.synergy_weight
    def _kl_pass(self, feat, mu, name, **kwargs):
        return {name: self._kl_loss(mu, feat)}

    def ce_losses(self, base_output, **kwargs):
        loss = {}
        for k, pred in base_output["preds"].items():
            this_label = kwargs["label"].repeat(self.K) if pred.shape[0]!=kwargs["label"].shape[0] else kwargs["label"]
            loss.update({k:F.cross_entropy(pred, this_label)})
        return loss

    def compute_training_losses(self, base_output, **kwargs):
        losses = {}

        #draw uniformly 0-1 random number
        # r = torch.bernoulli(torch.tensor(0.5)).item()
        # if r > 0.5:
        #     ce_losses = self.ce_losses(base_output, **kwargs)
        #     losses.update({"ce_mask0":ce_losses["mask0"]})
        #     losses.update({"ce_mask1":ce_losses["mask1"]})
        #     losses.update({"kl_synergy_1":torch.tensor(0.0)})
        #     losses.update({"kl_synergy_2":torch.tensor(0.0)})
        # else:
        #     losses.update({"ce_mask0":torch.tensor(0.0)})
        #     losses.update({"ce_mask1":torch.tensor(0.0)})
        ce_losses = self.ce_losses(base_output, **kwargs)
        losses.update({"ce_mask0": ce_losses["mask0"]})
        losses.update({"ce_mask1": ce_losses["mask1"]})
        losses.update(self._kl_pass(base_output["features"]["mask0"], base_output["preds"]["mask0"], name="kl_synergy_1", **kwargs))
        losses.update(self._kl_pass(base_output["features"]["mask1"], base_output["preds"]["mask1"], name="kl_synergy_2", **kwargs))
        # losses.update(self._kl_pass(base_output["features"]["mask01"], base_output["preds"]["mask01"], name="kl_synergy_12", **kwargs))



        # if self.perturb.get("type", None) == "competition":
        #     losses.update(self._kl_pass(base_output["features"]["combined"], base_output["features"]["combined"], name="kl_synergy_combined",**kwargs))
        #     losses.update(self.ce_losses(base_output, **kwargs))

        # z1, z2 = base_output["features"]["z1"], base_output["features"]["z2"]
        # infonce = nt_xent_loss(z1, z2, temperature=1.0)
        # losses["sl_sqdiff"] = (kl1 - kl2).pow(2.0).mean() * self.synergy_weight*10000
        # losses["infonce"] = infonce * self.contrastive_weight


        if self.training:
            if "current_step" in kwargs:
                wandb.log(losses, step=kwargs.get("current_step", 0)+1)

        return losses

class FusionIBModel_Mask(nn.Module):
    """
    Renamed version of your FusionIBModel:
      enc_0, enc_1 : modality encoders (same as before)
      enc_2        : fusion trunk (was common_fc_1)
      enc_3        : fusion projector (was common_fc_2)
      enc_4        : fusion head (was mu_head)
      synergy      : SynIB_VAE (or SynIB) using main=self
    """
    def __init__(self, args, encs):
        super().__init__()
        self.args = args
        self.cls_type = _cfg(args, "cls_type")
        self.norm_decision = _cfg(args, "norm_decision", False)

        self.num_classes = int(_cfg(args, "num_classes"))
        d_model = int(args.get("d_model",512))
        fc_inner = int(args.get("fc_inner",512))
        dropout = float(_cfg(args, "dropout", 0.1))

        self.synergy_weight = float(_cfg(args, "bias_infusion", {}).get("l", 0.0))
        self.ending_epoch = int(_cfg(args, "perturb", {}).get("ending_epoch", 1000.0))

        # main encoders
        self.enc_0 = encs[0]
        self.enc_1 = encs[1]

        if self.cls_type == "mlp":
            if len(encs)>2:
                self.enc_2 = encs[2]
                self.enc_3 = encs[3]
            else:
                self.enc_2 = FusionTrunkLinear(args)
                self.enc_3 = LinearHead(args)
        elif self.cls_type == "tf":
            self.enc_2 = TF_Fusion_Transformer(input_dim=d_model, dim=d_model, layers=2, output_dim=fc_inner)
            self.enc_3 = LinearHead(args)

        if len(encs) > 4:
            self.enc_4 = encs[4]
            self.enc_5 = encs[5]
        else:
            self.enc_4 = MLPHead(args)
            self.enc_5 = MLPHead(args)


        self.synib = SynIB(args, [], main=self)

    # -------------------------
    # original interfaces kept
    # -------------------------
    def _get_features(self, x, **kwargs):
        out0 = self.enc_0(x, detach_pred=True, **kwargs)
        out1 = self.enc_1(x, detach_pred=True, **kwargs)
        z1, na_z1 = _as_tensor_features(out0)
        z2, na_z2 = _as_tensor_features(out1)
        preds1 = _as_tensor_preds(out0)
        preds2 = _as_tensor_preds(out1)
        z1 = F.layer_norm(z1, (z1.shape[-1],))
        z2 = F.layer_norm(z2, (z2.shape[-1],))
        na_z1 = F.layer_norm(na_z1, (na_z1.shape[-1],))
        na_z2 = F.layer_norm(na_z2, (na_z2.shape[-1],))
        return preds1, preds2, z1, z2, na_z1, na_z2

    def _compute_logits(self, z1, z2, na_z1=None, na_z2=None, att_mask1=None, att_mask2=None,**kwargs):
        if self.cls_type == "tf":
            feat = self.enc_2({0:na_z1, 1:na_z2}, att_mask1=att_mask1, att_mask2=att_mask2)
        else:
            feat = self.enc_2(z1, z2)
        logits = self.enc_3(feat)
        return logits, feat

    def _base_forward(self, x, **kwargs):
        uni_pred_1, uni_pred_2, z1, z2, na_z1, na_z2 = self._get_features(x, **kwargs)
        pred, feat = self._compute_logits(z1, z2, na_z1, na_z2)

        return {
            "preds": {
                "combined": pred,
                "c":uni_pred_1,
                "g":uni_pred_2
            },
            "features": {
                "combined": feat,
                "z1": z1,
                "z2": z2,
                "na_z1":na_z1,
                "na_z2":na_z2
            },
            "losses": {},
        }

    def _base_forward_synib(self, x, **kwargs):
        uni_pred_1, uni_pred_2, z1, z2, na_z1, na_z2 = self._get_features(x, **kwargs)
        pred, feat = self._compute_logits(z1, z2, na_z1, na_z2)

        # uni_pred_1 = self._compute_logits_unimodal(z1, na_z1, direction="z1")
        # uni_pred_2 = self._compute_logits_unimodal(z2, na_z2, direction="z2")

        features = {"combined": feat,
                    "z1": z1,
                    "z2": z2,
                    "na_z1":na_z1,
                    "na_z2":na_z2}
        preds = {"combined": pred,
                 "c":uni_pred_1,
                 "g":uni_pred_2}
        feat_tilde_random = self.synib.get_random_mask_multiclass(features)
        feat_tilde = self.synib.get_learnable_mask_multiclass(x, features, preds, **kwargs)

        pred_mask0, feat_mask0 = self._compute_logits(feat_tilde["z1K"], feat_tilde["tz2K"], feat_tilde["na_z1K"], feat_tilde["na_tz2K"], att_mask1=feat_tilde["mask1"], att_mask2=None)
        pred_mask1, feat_mask1 = self._compute_logits(feat_tilde["tz1K"], feat_tilde["z2K"], feat_tilde["na_tz1K"], feat_tilde["na_z2K"], att_mask1=None, att_mask2=feat_tilde["mask2"])
        pred_mask01, feat_mask01 = self._compute_logits(feat_tilde["tz1K"], feat_tilde["tz2K"], feat_tilde["na_tz1K"], feat_tilde["na_tz2K"], att_mask1=feat_tilde["mask1"], att_mask2=feat_tilde["mask2"])


        pred_randmask0, feat_randmask0 = self._compute_logits(feat_tilde_random["z1K"], feat_tilde_random["tz2K"], feat_tilde_random["na_z1K"], feat_tilde_random["na_tz2K"], att_mask1=feat_tilde_random["mask1"], att_mask2=None)
        pred_randmask1, feat_randmask1 = self._compute_logits(feat_tilde_random["tz1K"], feat_tilde_random["z2K"], feat_tilde_random["na_tz1K"], feat_tilde_random["na_z2K"], att_mask1=None, att_mask2=feat_tilde_random["mask2"])
        pred_randmask01, feat_randmask01 = self._compute_logits(feat_tilde_random["tz1K"], feat_tilde_random["tz2K"], feat_tilde_random["na_tz1K"], feat_tilde_random["na_tz2K"], att_mask1=feat_tilde_random["mask1"], att_mask2=feat_tilde_random["mask2"])

        preds.update({"mask0":pred_mask0,
                     "mask1":pred_mask1,
                     "mask01":pred_mask01,
                      "randmask0": pred_randmask0,
                      "randmask1": pred_randmask1,
                      "randmask01": pred_randmask01,
                      })
        features.update({
                        "mask0":feat_mask0,
                         "mask1":feat_mask1,
                         "mask01":feat_mask01,
                        "randmask0":feat_randmask0,
                         "randmask1":feat_randmask1,
                         "randmask01":feat_randmask01
                         })
        losses = {}
        ce_losses = self.synib.ce_losses({"preds": preds, "features": features}, **kwargs)
        losses.update({"ce_mask0": ce_losses["randmask0"]})
        losses.update({"ce_mask1": ce_losses["randmask1"]})
        losses.update(self.synib._kl_pass(feat_mask0, pred_mask0, name="kl_synergy_1", **kwargs))
        losses.update(self.synib._kl_pass(feat_mask1, pred_mask1, name="kl_synergy_2", **kwargs))


        return {
            "preds": preds,
            "features": features,
            "losses": losses,
        }

    def forward(self, x, **kwargs):

        okay_epoch = False if "current_epoch" in kwargs and self.ending_epoch<kwargs["current_epoch"] else True
        if self.synergy_weight > 0 and okay_epoch:
            output = self._base_forward_synib(x, **kwargs)
        else:
            output =  self._base_forward(x, **kwargs)


        return output


