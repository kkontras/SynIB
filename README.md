# SynIB: Informational Bottleneck for Maximizing Synergy in Multimodal Learning

SynIB is a multimodal learning framework for training models to use synergistic information: task-relevant cues that only emerge when modalities are combined, and are not available from either modality alone.

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

> A CUDA-capable GPU is required. The code has been tested with PyTorch 2.x and CUDA 11.8/12.1.

## What To Read

- ESNLI pipeline: [run/esnli/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/esnli/README.md)
- CREMA-D / Irony_Cremad wrappers: [run/crema_d/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/crema_d/README.md)
- Generic wrappers: [run/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/README.md)

## Core Concepts

### Cached pipeline

For ESNLI, the raw dataset is first converted into cached Qwen-ready artifacts. This avoids recomputing expensive visual-language preprocessing during every training run.

### CEUs

`CEU` files are pickled validation/test predictions from unimodal models. They are used by downstream multimodal methods that need unimodal counterfactual signals.

In practice, the ESNLI workflow is:

1. download the raw data
2. build the cache
3. train the unimodal image/text models
4. export CEUs from those unimodals
5. train SynIB or the other downstream methods

## ESNLI Quickstart

### 1. Download Flickr30k

Expected layout:

```text
/data/ESNLI/
  flickr30k-images/
```

Download example:

```bash
mkdir -p /data/ESNLI/_downloads

huggingface-cli download nlphuji/flickr30k flickr30k-images.zip \
  --repo-type dataset \
  --local-dir /data/ESNLI/_downloads/flickr30k

unzip -q /data/ESNLI/_downloads/flickr30k/flickr30k-images.zip -d /data/ESNLI
```

The ESNLI split metadata is fetched automatically by the codebook builder.

### 2. Build the codebook / cache

```bash
export ESNLI_ROOT=/data/ESNLI
export ESNLI_CACHE=/data/ESNLI/cache_qwen3_vl_2b_nocls_vis
export QWEN_MODEL=Qwen/Qwen3-VL-2B-Instruct

for SET in train validation test; do
  CUDA_VISIBLE_DEVICES=0 python src/synib/mydatasets/ESNLI/ESNLI_CodeBook_v3.py \
    --data_root "$ESNLI_ROOT" \
    --out_dir "$ESNLI_CACHE" \
    --model_name "$QWEN_MODEL" \
    --split "$SET"
done
```

Make sure the ESNLI cached configs point `dataset.cache_root` to `"$ESNLI_CACHE"`.

### 3. Train the unimodals

Run each config for folds `0`, `1`, and `2`:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_image_lora.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_text_lora.json --fold 0
```

### 4. Export CEUs

```bash
PYTHONPATH=src python -m synib.entrypoints.get_ceu_cli \
  --dataset esnli \
  --default_config run/configs/ESNLI/default_config_esnli_cache.json \
  --unimodal_configs run/configs/ESNLI/cache_image_lora.json run/configs/ESNLI/cache_text_lora.json \
  --folds 0 1 2 \
  --output_root ./artifacts/ceus
```

Expected outputs:

```text
./artifacts/ceus/esnli/esnli_ceu_val.pkl
./artifacts/ceus/esnli/esnli_ceu_test.pkl
```

### 5. Train the downstream methods

Cached baseline:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_lora.json --fold 0
```

Cached SynIB:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --fold 0
```

Other supported ESNLI methods:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_ens.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_synib.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_mcr.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_dnr.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_mmpareto.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_reconboost.json --fold 0
```

### 6. Evaluate

```bash
./run/esnli/show.sh run/configs/ESNLI/cache_synib_lora.json --fold 0
```

For the longer ESNLI walkthrough and more config examples, see [run/esnli/README.md](/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB/run/esnli/README.md).

## Methods

Main method families currently exposed in configs:

- Baseline
- Ensemble
- SynIB
- MCR
- DnR
- MMPareto
- ReconBoost

The config root is:

```text
run/configs/ESNLI/
```

## Repository Layout

```text
src/synib/
  entrypoints/   train, show, get_ceu_cli
  models/        multimodal and VLM models
  mydatasets/    ESNLI and Irony_Cremad data code
  training/      training and evaluation pipeline
run/
  esnli/         ESNLI shell wrappers
  crema_d/       CREMA-D shell wrappers
  configs/       experiment configs
```

## License

This project is released under the MIT License.
