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
- `qwen_synergy_decoder_models.py`
- `qwen_synergy_svae_models.py`
- ScienceQA/ESNLI VLM and SynIB models (`QwenVL_*`, `FusionIBModel*`, etc.)

## Shared utilities
- `src/synib/models/model_utils/`
- `src/synib/models/VAVL_git/VAVL/conformer/` (kept minimal to support current imports)
