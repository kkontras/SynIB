import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from synib.models.vlm.qwen_base_models import *
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
        elif self.synergy_type == "unimodal_anchor":
            self.dirichlet_prior_conc = None  # no logvar/evidence heads needed
        else:
            raise ValueError(f"Unknown synergy_type: {self.synergy_type}")

        self.anchor_to_unimodal = (self.synergy_type == "unimodal_anchor") or bool(_cfg(args, "anchor_unimodal", False))

        self.cls_type = _cfg(args, "cls_type")

        self.p = float(self.perturb.get("p_min", 0.5))
        self.noise_std = float(self.perturb.get("noise_std", 1.0))
        self.K = int(self.perturb.get("num_samples", 1))
        self.fill = self.perturb.get("fill", "ema")
        self.p_type = self.perturb.get("type", "diff")

        self.p_min = float(self.perturb.get("p_min", 0.3))
        self.p_max = float(self.perturb.get("p_max", 0.9))
        self.cosine_s = 0.008

        _feat_dim = int(_cfg(args, "d_model", 512))
        if self.cls_type == "mlp":
            self.stats_z1 = FeatureStatsMasker(d1=_feat_dim, ema_beta=0.99)
            self.stats_z2 = FeatureStatsMasker(d1=_feat_dim, ema_beta=0.99)
        elif self.cls_type == "tf":
            self.stats_na_z1 = FeatureStatsMasker(d1=_feat_dim, ema_beta=0.99)
            self.stats_na_z2 = FeatureStatsMasker(d1=_feat_dim, ema_beta=0.99)


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
            return zK, tzK, None

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

    def _kl_unimodal_anchor(self, pred_masked, pred_unimodal_target, name):
        """KL from a masked-modality prediction toward the complementary unimodal prediction."""
        p_masked = F.log_softmax(pred_masked, dim=-1)
        p_target = F.softmax(pred_unimodal_target.detach(), dim=-1)
        kl = F.kl_div(p_masked, p_target, reduction="batchmean") * self.synergy_weight
        return {name: kl}

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
        if self.synib.anchor_to_unimodal:
            losses.update(self.synib._kl_unimodal_anchor(pred_mask0, uni_pred_2, name="kl_synergy_1"))
            losses.update(self.synib._kl_unimodal_anchor(pred_mask1, uni_pred_1, name="kl_synergy_2"))
        else:
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


class FusionIBModel_Mask_U(FusionIBModel_Mask):
    """SynIB Unimodal-anchored (SynIBU) for multibench.
    KL drives masked-modality predictions toward the complementary unimodal baseline.
    Set synergy_type='unimodal_anchor' and bias_infusion.l > 0 in model args.
    """
    pass
