# Where is what — code map

A guide to the SynIB codebase, oriented around the paper.

## The method (start here)

| What | File | Notes |
|---|---|---|
| **SynIB objective** | `src/synib/models/vlm/synib_mask_model.py` | The whole method lives here. |
| `SynIB` | same file | Encoders + mask construction + the counterfactual/intact forward passes. |
| `FusionIBModel_Mask` | same file | Wrapper with classification head: CE on intact `(Z₁,Z₂)` + KL penalty under masking. |
| `FusionIBModel_Mask_U` | same file | **Asymmetric** variant (per-modality KL weights `l_z1_masked`/`l_z2_masked`, targeted modality dropout). Project default since the asymmetric study. |
| Random vs learned masks | `get_random_mask_multiclass()`, `get_learnable_mask_multiclass()` | `M_random` (Bernoulli) and `M_learned` (adversarial). |
| Masking-without-KL ablation | `src/synib/baselines/masking_only.py` | Isolates the mask augmentation from the KL term (`make_masking_only_args`). |

## Encoders & fusion

| What | File |
|---|---|
| CLIP-ViT image encoder (HM) | `src/synib/models/vision_text/clip_vision.py` |
| DeBERTa / HF text encoder (HM) | `src/synib/models/vision_text/hf_text.py` |
| Fusion adapters (gated / FiLM / TF) | `src/synib/models/vision_text/fusion_adapters.py` |
| CREMA-D audio/visual backbones | `src/synib/models/crema_d/crema_backbone_fusion_models.py` |

## Baselines

The competing objectives (Vanilla, D&R, MMPareto, ReconBoost, MCR, Ensemble) are
implemented in `src/synib/models/crema_d/crema_backbone_fusion_models.py` (e.g. `MCR`,
`AGM`, `MLA`) and selected through the per-dataset **method configs**, e.g.
`run/configs/hateful_memes/methods/{vanilla,dnr,mmpareto,reconboost,mcr,synib,synib_u}.json`.

## Datasets

| Dataset | Loader |
|---|---|
| MOSI / UR-Funny / MUStARD | `src/synib/mydatasets/Factor_CL_Datasets/` (frozen MultiBench features) |
| Hateful Memes | `src/synib/mydatasets/HatefulMemes/` (CLIP + DeBERTa cache) |
| CREMA-D / CREMA-D-Irony | `src/synib/mydatasets/Irony_Cremad/` |

Dataloader and model classes are resolved by name from config
(`Agent._resolve_dataloader_class`, `src/synib/training/pipeline/agent.py`).

## Synthetic experiments (paper Figs 1, 6, 7, 8)

| What | File |
|---|---|
| Spurious-shortcut XOR | `scripts/analysis/xor_spurious_pub.py` |
| PID-controlled XOR + mask search | `scripts/analysis/Xor_PID3Main_MaskSynIB_Search.py` |
| NTK / per-source dynamics | `scripts/analysis/Xor_PIDExamineNTK.py`, `scripts/analysis/figs_pid_ntk_dynamics.py` |
| Training-dynamics figure | `scripts/analysis/figs_training_dynamics.py` |

## Running things

| What | Where |
|---|---|
| CLI entrypoints | `src/synib/entrypoints/{train,show,get_ceu_cli,get_ceu_hm_cli,eval_ceu_only}.py` |
| Per-dataset launch wrappers | `run/multibench/`, `run/hateful_memes/`, `run/cremad/` |
| All configs | `run/configs/{multibench,hateful_memes,cremad}/` |
| Synergy-subset CEU eval | `scripts/compute_synergy_subset.py` |

## Results & analysis

| What | Where |
|---|---|
| Reported numbers for every benchmark | `docs/paper.pdf` (Fig. 5 and result tables) |
| Every training command for every paper cell | `docs/REPRODUCE.md` |
| Paper figures (PNG) | `docs/figures/` |
