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

1. Download e-SNLI-VE from the [official repo](https://github.com/maximek3/e-ViL)
2. Place at your chosen root (e.g., `/data/ESNLI/`)
3. Build the token cache (see End-to-End Pipeline below)

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
python src/synib/mydatasets/ESNLI/ESNLI_CodeBook_v2.py \
    --data_root /data/ESNLI \
    --out_dir /data/ESNLI/cache_tokens8B \
    --model_name Qwen/Qwen3-VL-8B-Instruct
```

### Step 2: Generate Unimodal CEU Predictions

```bash
python -m synib.entrypoints.get_ceu_cli \
    --dataset crema_d \
    --config run/configs/CREMA_D/synergy/jan/synprom_RMask.json \
    --fold 0
```

### Step 3: Train

```bash
./run/crema_d/train.sh rmask-random-l1.0-pmin0.20 --fold 0
```

### Step 4: Evaluate

```bash
./run/crema_d/show.sh rmask-random-l1.0-pmin0.20 --fold 0
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
│   │   ├── xor/            # XOR synthetic dataset models
│   │   ├── conformer/      # Conformer encoder (Apache 2.0, Soohwan Kim)
│   │   └── model_utils/    # Shared utilities (backbone, losses, gates)
│   ├── mydatasets/
│   │   ├── CREMAD/         # CREMA-D dataloaders
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
