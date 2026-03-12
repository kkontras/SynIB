# Models Layout

Models are grouped by dataset/task for easier onboarding.

## `src/synib/models/crema_d/`
- `crema_backbone_fusion_models.py`
- CREMA-D-focused backbones and fusion methods (`MCR_Model`, `AGM_Model`, `MLA_Model`, etc.)

## `src/synib/models/xor/`
- `synprom_models.py`
- `synprom_perf_models.py`
- XOR and SynProm family models (`Fusion_Synprom*`, `TriModalFusionClassifier`, etc.)

## `src/synib/models/vlm/`
- `qwen_base_models.py`
- `qwen_prompt_models.py`
- `qwen_cached.py`
- `synib_mask_model.py`
- Qwen VLM families split by pipeline: `qwen_prompt_models.py` for prompt-driven variants and `qwen_cached.py` for cached variants.
- `synib_mask_model.py` holds the reusable SynIB / transformer components still used outside the Qwen cached stack.

## Shared utilities
- `src/synib/models/model_utils/`
- `src/synib/models/conformer/` (Conformer encoder, Apache 2.0, Soohwan Kim)
