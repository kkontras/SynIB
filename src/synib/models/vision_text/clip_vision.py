"""
CLIP vision encoder wrapper for the SynIB HM pipeline.

Contract (aligned with FusionIBModel_Mask._get_features / _as_tensor_features):
    forward(batch_data, detach_pred=True, **kwargs) -> {
        "features":         {"combined": (B, d_model)},
        "nonaggr_features": {"combined": (B, N, d_model)},
        "preds":            {"combined": (B, num_classes)},
    }

Supports both paths:
    - cached: batch_data[modality_key] is a dict {"tokens","mask","pool"}.
    - live:   batch_data[modality_key] is a stacked image tensor (B,3,H,W).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


class CLIPVisionEncoder(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        self.args = args
        self.modality_key = int(_cfg(args, "modality_key", 1))
        self.native_dim = int(_cfg(args, "native_dim"))
        self.d_model = int(_cfg(args, "d_model"))
        self.num_classes = int(_cfg(args, "num_classes", 2))
        self.live_encode = bool(_cfg(args, "live_encode", False))
        self.ckpt = str(_cfg(args, "ckpt", "openai/clip-vit-base-patch16"))
        self.image_size = int(_cfg(args, "image_size", 224))

        self.proj = nn.Linear(self.native_dim, self.d_model)
        self.head = nn.Linear(self.d_model, self.num_classes)

        self.backbone: Optional[nn.Module] = None
        self.processor = None
        if self.live_encode:
            self._build_backbone()
            self._apply_lora()

    def _build_backbone(self):
        from transformers import CLIPImageProcessor, CLIPVisionModel

        self.backbone = CLIPVisionModel.from_pretrained(self.ckpt)
        self.processor = CLIPImageProcessor.from_pretrained(self.ckpt)
        if bool(_cfg(self.args, "freeze_backbone", True)):
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def _apply_lora(self):
        cfg = _cfg(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return
        from peft import LoraConfig, get_peft_model

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _encode_live(self, pixel_values: torch.Tensor):
        assert self.backbone is not None, "live_encode=True requires backbone"
        dtype = next(self.backbone.parameters()).dtype
        pv = pixel_values.to(device=self.backbone.device, dtype=dtype)
        out = self.backbone(pixel_values=pv, return_dict=True)
        tokens = out.last_hidden_state
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else tokens[:, 0, :]
        mask = torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)
        return pooled, tokens, mask

    def _pick(self, batch_data):
        if isinstance(batch_data, dict) and self.modality_key in batch_data:
            return batch_data[self.modality_key]
        return batch_data

    def forward(self, batch_data, *, detach_pred: bool = False, **kwargs):
        x = self._pick(batch_data)

        if isinstance(x, dict) and "pool" in x:
            dtype = self.proj.weight.dtype
            pooled = x["pool"].to(device=self.proj.weight.device, dtype=dtype)
            tokens = x["tokens"].to(device=self.proj.weight.device, dtype=dtype)
        else:
            if not torch.is_tensor(x):
                raise TypeError(f"CLIPVisionEncoder expects dict (cached) or tensor (live), got {type(x)}")
            pooled, tokens, _ = self._encode_live(x)
            pooled = pooled.to(self.proj.weight.dtype)
            tokens = tokens.to(self.proj.weight.dtype)

        pooled_p = self.proj(pooled)
        tokens_p = self.proj(tokens)

        z = F.layer_norm(pooled_p, (pooled_p.shape[-1],))
        na_z = F.layer_norm(tokens_p, (tokens_p.shape[-1],))

        head_in = z.detach() if detach_pred else z
        logits = self.head(head_in)

        return {
            "features": {"combined": z},
            "nonaggr_features": {"combined": na_z},
            "preds": {"combined": logits},
        }

    def forward_uni(self, z, na_z=None, *, detach_pred: bool = False, **kwargs):
        """Unimodal-head forward on already-extracted features (used by SynIB's learnable-mask path)."""
        h = z.detach() if detach_pred else z
        return self.head(h)
