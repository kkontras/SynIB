# SynIB: Informational Bottleneck for Maximizing Synergy in Multimodal Learning

SynIB is a multimodal learning framework that explicitly promotes synergistic cross-modal behavior: task-relevant information that is only available from combining modalities and not from either modality alone.

## Paper Summary

> A central objective in multimodal learning is to capture synergistic information:
> task-relevant cues that arise only from the joint use of multiple modalities and
> are not available from any single modality alone. In practice, however, standard
> training procedures often emphasize unimodal or redundant signals, allowing
> models to achieve strong overall performance while falling short on examples
> that require cross-modal reasoning. We argue that this behavior does not reflect
> an inherent conflict between unimodal and synergistic learning, but rather an
> imbalance in learning dynamics: synergistic cues are typically rarer, harder to
> learn, and more susceptible to overfitting. To address this imbalance, we formalize
> multimodal synergy via the lens of information theory and derive scalable training
> objectives that explicitly promote synergetic cross-modal behavior. The key idea is
> to penalize overconfident predictions when task-relevant information is removed
> from a counterfactual modality. We first validate our approach through controlled
> Gaussian-based XOR synthetic experiments that isolate the training dynamics of
> multimodal synergy. We then introduce a new irony class for emotion recognition,
> augmenting an existing dataset with a controllable degree of multimodal synergy
> to serve as a semi-synthetic benchmark. We further evaluate on several affective
> computing benchmarks from MultiBench. To demonstrate scalability, we fine-tune
> Qwen3-VL for visual instruction tuning, obtaining consistent improvements on
> synergy-dependent subsets of E-SNLI-VE, MultiQA, and LongVALE.

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** A CUDA-capable GPU is required. The code has been tested with PyTorch 2.x and CUDA 11.8/12.1.

## Quick Navigation

- `CREMA-D / Irony_Cremad`: [run/crema_d/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/crema_d/README.md)
- `e-SNLI-VE`: [run/esnli/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/esnli/README.md)
- `ScienceQA`: [run/scienceqa/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/scienceqa/README.md)
- generic wrapper usage: [run/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/README.md)

## Pipeline Concepts

### Caches

For ESNLI and ScienceQA, this repo can preprocess the raw dataset into Qwen-ready cache shards. These caches avoid recomputing expensive VLM tokenization and visual embeddings during training.

### CEUs

`CEU` files are pickled validation/test predictions from trained unimodal models. They are generated with [src/synib/entrypoints/get_ceu_cli.py](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/src/synib/entrypoints/get_ceu_cli.py) and are used by downstream bias-infusion and synergy-aware methods.

In practice:
- train unimodal models first
- export their predictions into CEU pickles
- point downstream configs to those CEU files when the method expects `model.ceu.val` / `model.ceu.test`

## By Dataset

### CREMA-D / Irony_Cremad

Data layout:

```text
CREMA-D/
  AudioWAV/
  VideoFlash/
  VideoMP4/
```

Entry point:
- [run/crema_d/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/crema_d/README.md)

Typical flow:

```bash
./run/crema_d/train.sh --scenarios
./run/crema_d/train.sh default --fold 0
./run/crema_d/show.sh default --fold 0
```

### ScienceQA

Data root:

```text
/data/ScienceQA/
```

Build the cache:

```bash
python src/synib/mydatasets/ScienceQA/ScienceQA_Codebook_v2.py \
  --data_root /data/ScienceQA \
  --out_dir /data/ScienceQA/cache_tokens8B \
  --model_name Qwen/Qwen3-VL-8B-Instruct
```

Then use:
- [run/scienceqa/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/scienceqa/README.md)

### e-SNLI-VE

Expected raw data layout:

```text
/data/ESNLI/
  flickr30k-images/
```

The split metadata is downloaded automatically by the cache builder. The Flickr30k image archive is not.

Download Flickr30k:

```bash
mkdir -p /data/ESNLI/_downloads

huggingface-cli download nlphuji/flickr30k flickr30k-images.zip \
  --repo-type dataset \
  --local-dir /data/ESNLI/_downloads/flickr30k

unzip -q /data/ESNLI/_downloads/flickr30k/flickr30k-images.zip -d /data/ESNLI
```

Build the cache:

```bash
CUDA_VISIBLE_DEVICES=0 python src/synib/mydatasets/ESNLI/ESNLI_CodeBook_v3.py \
  --data_root /data/ESNLI \
  --out_dir /data/ESNLI/cache_qwen3_vl_2b_nocls_vis \
  --model_name Qwen/Qwen3-VL-2B-Instruct \
  --split train

CUDA_VISIBLE_DEVICES=0 python src/synib/mydatasets/ESNLI/ESNLI_CodeBook_v3.py \
  --data_root /data/ESNLI \
  --out_dir /data/ESNLI/cache_qwen3_vl_2b_nocls_vis \
  --model_name Qwen/Qwen3-VL-2B-Instruct \
  --split validation

CUDA_VISIBLE_DEVICES=0 python src/synib/mydatasets/ESNLI/ESNLI_CodeBook_v3.py \
  --data_root /data/ESNLI \
  --out_dir /data/ESNLI/cache_qwen3_vl_2b_nocls_vis \
  --model_name Qwen/Qwen3-VL-2B-Instruct \
  --split test
```

Minimal training flow:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_image_lora.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_text_lora.json --fold 0

PYTHONPATH=src python -m synib.entrypoints.get_ceu_cli \
  --dataset esnli \
  --default_config run/configs/ESNLI/default_config_esnli_cache.json \
  --unimodal_configs run/configs/ESNLI/cache_image_lora.json run/configs/ESNLI/cache_text_lora.json \
  --folds 0 1 2 \
  --output_root ./artifacts/ceus

./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --fold 0
./run/esnli/show.sh run/configs/ESNLI/cache_synib_lora.json --fold 0
```

For the full staged ESNLI walkthrough, use:
- [run/esnli/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/esnli/README.md)

## Available Bias-Infusion Methods

| Method | Key | Description |
|--------|-----|-------------|
| Baseline | `main` | No bias infusion |
| OGM-GE | `ogm` | On-the-fly Gradient Modulation |
| MLB | `mlb` | Multi-modal Lazy Bias |
| AGM | `agm` | Adaptive Gradient Modulation |
| MMPareto | `mmpareto` | Pareto-optimal multimodal learning |
| DnR | `dnr` | Dominant and Residual learning |
| ReconBoost | `recon` | Reconstruction-based boosting |
| MCR | `mcr` | Modality Competition Regularization |
| SynIB (random mask) | `synib_rand` | SynIB with random masking |
| SynIB (learned mask) | `synib_star` | SynIB with learned masking |
| ... | ... | See `run/configs/` for full list |

## Config System

Configs are JSON files in `run/configs/`. Each config inherits from a `default_config` and overrides specific fields. Key fields:

- `model.model_class`: which model architecture to use
- `dataset.data_roots`: path to dataset (must be set per installation)
- `bias_infusion.l`: synergy loss weight (λ)
- `perturb.p_min`, `perturb.p_max`: masking probability range
- `training_params.fold`: cross-validation fold index

## Repository Map

```
SynIB/
├── src/synib/
│   ├── entrypoints/        # CLI entry points (train, show, get_ceu)
│   ├── models/
│   │   ├── vlm/            # Vision-language model components (Qwen-based)
│   │   ├── crema_d/        # CREMA-D backbone and fusion models
│   │   ├── conformer/      # Conformer encoder (Apache 2.0, Soohwan Kim)
│   │   └── model_utils/    # Shared utilities (backbone, losses, gates)
│   ├── mydatasets/
│   │   ├── Irony_Cremad/   # CREMA-D dataloaders and assets
│   │   ├── ESNLI/          # e-SNLI-VE dataloaders
│   │   └── ScienceQA/      # ScienceQA dataloaders
│   └── trainers/           # Training loops
└── run/
    ├── configs/            # Experiment configs (JSON)
    ├── crema_d/            # Shell scripts for CREMA-D
    ├── esnli/              # Shell scripts for e-SNLI-VE
    └── scienceqa/          # Shell scripts for ScienceQA
```

## License

This project is released under the MIT License. The Conformer implementation in `src/synib/models/conformer/` is licensed under the Apache License 2.0 (Copyright 2021 Soohwan Kim).
