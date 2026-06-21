"""
Adapters that plug existing fusion modules into FusionIBModel_Mask's
`enc_2(z1, z2)` interface without rewriting the fusion module itself.
"""

from __future__ import annotations

import torch.nn as nn

from synib.models.model_utils.fusion_gates import FiLM, GatedFusion


def _cfg(args, key, default=None):
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


class GatedFusionTrunk(nn.Module):
    """Wraps GatedFusion so it can drop into `self.enc_2` with `cls_type=mlp`."""

    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_model = int(_cfg(args, "d_model"))
        fc_inner = int(_cfg(args, "fc_inner"))
        x_gate = bool(_cfg(args, "x_gate", True))
        self.net = GatedFusion(input_dim=d_model, dim=d_model, output_dim=fc_inner, x_gate=x_gate)

    def forward(self, z1, z2, **kwargs):
        return self.net([z1, z2], **kwargs)


class FiLMTrunk(nn.Module):
    """FiLM as a fusion trunk (experimental, not in the primary plan)."""

    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        d_model = int(_cfg(args, "d_model"))
        fc_inner = int(_cfg(args, "fc_inner"))
        x_film = bool(_cfg(args, "x_film", True))
        self.net = FiLM(input_dim=d_model, dim=d_model, output_dim=fc_inner, x_film=x_film)

    def forward(self, z1, z2, **kwargs):
        return self.net([z1, z2], **kwargs)
