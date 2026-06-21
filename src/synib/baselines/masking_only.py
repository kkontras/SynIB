"""Masking-only ablation: SynIB's mask construction without the KL term.

The masking-only baseline uses the SAME mask construction as SynIB
(`SynIB.get_random_mask_multiclass` for M_Random, `SynIB.get_learnable_mask_multiclass`
for M_Learned) but trains with cross-entropy on (intact ∪ masked), with no KL
penalty between intact and masked predictions. This isolates the data-augmentation
effect of masked inputs from the KL objective itself.

The actual gating lives in `FusionIBModel_Mask._base_forward_synib` in
`src/synib/models/vlm/synib_mask_model.py`: when `synib_masking_only=True`, the
KL emissions in the random and learned branches are skipped while the masked
forward passes still execute, leaving CE on the masked predictions to be driven
by `multi_loss.multi_supervised_w` in the trainer.

This module provides a small factory for tests and programmatic config building
to make the shared-mask-construction property mechanically obvious.
"""

from __future__ import annotations

import copy
from typing import Any, Mapping


def make_masking_only_args(base_args: Mapping[str, Any], mask_kind: str) -> dict:
    """Return SynIB args configured for the masking-only ablation.

    The same `SynIB.get_random_mask_multiclass` / `get_learnable_mask_multiclass`
    methods are reused — this function only flips configuration flags.

    Args:
        base_args: starting args dict (typically a SynIB method config). Not mutated.
        mask_kind: one of {"random", "learned"}. Selects which mask branch is active.

    Returns:
        A new args dict with `synib_masking_only=True`, the appropriate branch
        gates, and `bias_infusion.l = 1.0` as a documented entry-condition value.
    """
    if mask_kind not in ("random", "learned"):
        raise ValueError(f"mask_kind must be 'random' or 'learned', got {mask_kind!r}")

    args = copy.deepcopy(dict(base_args))
    args["synib_masking_only"] = True

    bi = dict(args.get("bias_infusion", {}) or {})
    if float(bi.get("l", 0.0) or 0.0) <= 0.0:
        bi["l"] = 1.0
    args["bias_infusion"] = bi

    if mask_kind == "random":
        args["synib_use_random_ce"] = True
        args["synib_use_learnable_kl"] = False
    else:
        args["synib_use_random_ce"] = False
        args["synib_use_learnable_kl"] = True

    return args
