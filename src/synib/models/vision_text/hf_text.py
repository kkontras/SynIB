"""
Generic HuggingFace text encoder wrapper for the SynIB HM pipeline.

Supports:
    - encoder models (DeBERTa-v3, BERT, RoBERTa) — pool via CLS token.
    - decoder-only LMs (Qwen2.5, Llama-3.2) — pool via last non-pad token.

Contract matches CLIPVisionEncoder. Both cached and live paths.
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


class HFTextEncoder(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        self.args = args
        self.modality_key = int(_cfg(args, "modality_key", 0))
        self.native_dim = int(_cfg(args, "native_dim"))
        self.d_model = int(_cfg(args, "d_model"))
        self.num_classes = int(_cfg(args, "num_classes", 2))
        self.live_encode = bool(_cfg(args, "live_encode", False))
        self.ckpt = str(_cfg(args, "ckpt", "microsoft/deberta-v3-base"))
        self.text_kind = str(_cfg(args, "text_kind", "encoder"))
        self.max_length = int(_cfg(args, "max_length", 64))

        if self.text_kind not in ("encoder", "decoder"):
            raise ValueError(f"text_kind must be 'encoder' or 'decoder', got {self.text_kind!r}")

        self.proj = nn.Linear(self.native_dim, self.d_model)
        self.head = nn.Linear(self.d_model, self.num_classes)

        self.backbone: Optional[nn.Module] = None
        self.tokenizer = None
        if self.live_encode:
            self._build_backbone()
            self._apply_lora()

    def _build_backbone(self):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.ckpt)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.backbone = AutoModel.from_pretrained(self.ckpt)
        if bool(_cfg(self.args, "freeze_backbone", True)):
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def _apply_lora(self):
        cfg = _cfg(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return
        from peft import LoraConfig, get_peft_model

        defaults = ["query", "value"] if self.text_kind == "encoder" else ["q_proj", "v_proj"]
        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", defaults)),
            bias=str(cfg.get("lora_bias", "none")),
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _encode_live(self, texts):
        assert self.backbone is not None, "live_encode=True requires backbone"
        device = next(self.backbone.parameters()).device
        enc = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        hidden = out.last_hidden_state
        mask = attention_mask.bool()
        if self.text_kind == "encoder":
            pooled = hidden[:, 0, :]
        else:
            lens = attention_mask.sum(dim=1)
            idx = (lens - 1).clamp(min=0)
            pooled = hidden[torch.arange(hidden.size(0), device=device), idx]
        return pooled, hidden, mask

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
            if not isinstance(x, (list, tuple)):
                raise TypeError(f"HFTextEncoder expects dict (cached) or list-of-str (live), got {type(x)}")
            pooled, tokens, _ = self._encode_live(list(x))
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
