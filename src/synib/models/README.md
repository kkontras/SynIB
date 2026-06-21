# Models layout

Grouped by role. For a task-oriented map see [`docs/WHERE_IS_WHAT.md`](../../../docs/WHERE_IS_WHAT.md).

## `vlm/synib_mask_model.py` — the SynIB method
- `SynIB` — encoders, mask construction, intact + counterfactual forward passes.
- `FusionIBModel_Mask` — CE on intact inputs + KL confidence penalty under masking.
- `FusionIBModel_Mask_U` — asymmetric variant (per-modality KL weights, targeted dropout).
- Fusion components (`TF_Fusion_Transformer`, `FusionTrunkLinear`, `FusionConformer`) and the
  feature masker used by the objective.

## `vision_text/` — Hateful Memes encoders
- `clip_vision.py` (CLIP-ViT), `hf_text.py` (DeBERTa / HF text), `fusion_adapters.py`.

## `crema_d/crema_backbone_fusion_models.py` — CREMA-D + baselines
- CREMA-D audio/visual backbones and the competing objectives used as baselines
  (`MCR`, `AGM`, `MLA`, vanilla/ensemble fusion).

## Shared utilities
- `model_utils/` — backbones, fusion gates, helpers.

The SynIB objective is the entry point — start with `vlm/synib_mask_model.py`.
