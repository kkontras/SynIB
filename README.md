# SynIB — Synergy-based Information Bottleneck for Multimodal Learning

SynIB is a framework for multimodal bias infusion that leverages the information bottleneck principle to improve synergy between modalities. 
<!-- ## Citation

```bibtex
@article{synib2025,
  title   = {TODO: Fill in paper title},
  author  = {TODO: Fill in authors},
  journal = {TODO: Fill in venue},
  year    = {2025}
}
``` -->

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** A CUDA-capable GPU is required. The code has been tested with PyTorch 2.x and CUDA 11.8/12.1.

## Data Setup

### CREMA-D

1. Download CREMA-D from [GitHub](https://github.com/CheyneyComputerScience/CREMA-D)
2. Place the dataset at your chosen root directory (e.g., `/data/CREMA-D/`)
3. Set `data_roots` in your config to that path

Expected layout:
```
CREMA-D/
  AudioWAV/
  VideoFlash/
  VideoMP4/
```

### ScienceQA

1. Download ScienceQA from [HuggingFace](https://huggingface.co/datasets/derek-thomas/ScienceQA) or the [official repo](https://github.com/lupantech/ScienceQA)
2. Place at your chosen root (e.g., `/data/ScienceQA/`)
3. Build the token cache (see End-to-End Pipeline below)

### e-SNLI-VE

1. Create a data root (example: `/data/ESNLI`)
2. Download Flickr30k images into that root
3. Build the token cache

Example commands:

```bash
mkdir -p /data/ESNLI/_downloads

huggingface-cli download nlphuji/flickr30k flickr30k-images.zip \
  --repo-type dataset \
  --local-dir /data/ESNLI/_downloads/flickr30k

unzip -q /data/ESNLI/_downloads/flickr30k/flickr30k-images.zip -d /data/ESNLI
```

Expected layout before cache building:

```text
/data/ESNLI/
  flickr30k-images/
```

Notes:
- The cache builder will automatically download the e-SNLI-VE split metadata from the `e-ViL` repo into `data_root`.
- The raw ESNLI dataloader can also auto-download Flickr30k, but the cached ESNLI pipeline documented below assumes `flickr30k-images/` already exists under `data_root`.

## End-to-End Pipeline

### Step 1: Build Caches (ScienceQA / e-SNLI-VE)

For ScienceQA:
```bash
python src/synib/mydatasets/ScienceQA/ScienceQA_Codebook_v2.py \
    --data_root /data/ScienceQA \
    --out_dir /data/ScienceQA/cache_tokens8B \
    --model_name Qwen/Qwen3-VL-8B-Instruct
```

For e-SNLI-VE:
```bash
python src/synib/mydatasets/ESNLI/ESNLI_CodeBook_v3.py \
    --data_root /data/ESNLI \
    --out_dir /data/ESNLI/cache_tokens8B \
    --model_name Qwen/Qwen3-VL-2B-Instruct
```

## ESNLI Pipeline

The ESNLI cached pipeline is:

1. Download Flickr30k images under the ESNLI data root
2. Build the Qwen cache shards
3. Train the unimodal cached models
4. Generate CEU pickles from the trained unimodals
5. Train the combined / SynIB / ensemble / full-method models

### 1. Download ESNLI data

```bash
mkdir -p /data/ESNLI/_downloads

huggingface-cli download nlphuji/flickr30k flickr30k-images.zip \
  --repo-type dataset \
  --local-dir /data/ESNLI/_downloads/flickr30k

unzip -q /data/ESNLI/_downloads/flickr30k/flickr30k-images.zip -d /data/ESNLI
```

### 2. Build the cached codebook

Run once per split:

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

After that, point `dataset.cache_root` in the ESNLI cached configs to the cache directory you created.

### 3. Train the unimodals

Image-only:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_image_lora.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_image_lora.json --fold 1
./run/esnli/train.sh run/configs/ESNLI/cache_image_lora.json --fold 2
```

Text-only:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_text_lora.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_text_lora.json --fold 1
./run/esnli/train.sh run/configs/ESNLI/cache_text_lora.json --fold 2
```

Optional combined cached baseline:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_lora.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_lora.json --fold 1
./run/esnli/train.sh run/configs/ESNLI/cache_lora.json --fold 2
```

### 4. Generate CEU pickles

Generate CEU files from the two trained unimodals:

```bash
PYTHONPATH=src python -m synib.entrypoints.get_ceu_cli \
  --dataset esnli \
  --default_config run/configs/ESNLI/default_config_esnli_cache.json \
  --unimodal_configs run/configs/ESNLI/cache_image_lora.json run/configs/ESNLI/cache_text_lora.json \
  --folds 0 1 2 \
  --output_root ./artifacts/ceus
```

This writes:

```text
./artifacts/ceus/esnli/esnli_ceu_val.pkl
./artifacts/ceus/esnli/esnli_ceu_test.pkl
```

If a downstream config expects CEU files through `model.ceu.val` / `model.ceu.test`, point those fields at the generated pickles.

### 5. Train the downstream methods

Cached SynIB:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --fold 1
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --fold 2
```

Cached ensemble:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_ens.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_ens.json --fold 1
./run/esnli/train.sh run/configs/ESNLI/cache_ens.json --fold 2
```

Full-method experiments:

```bash
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_synib.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_mcr.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_dnr.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_mmpareto.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_reconboost.json --fold 0
```

Evaluation:

```bash
./run/esnli/show.sh run/configs/ESNLI/cache_image_lora.json --fold 0
./run/esnli/show.sh run/configs/ESNLI/cache_text_lora.json --fold 0
./run/esnli/show.sh run/configs/ESNLI/cache_synib_lora.json --fold 0
./run/esnli/show.sh run/configs/ESNLI/full/esnli_full_synib.json --fold 0
```

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
