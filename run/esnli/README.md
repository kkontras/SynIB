# ESNLI

Use these wrappers for e-SNLI training and evaluation.

- Train: `./run/esnli/train.sh [config_path] [extra args]`
- Show/evaluate: `./run/esnli/show.sh [config_path] [extra args]`
- Config root: `run/configs/ESNLI/`

## End-to-End ESNLI Pipeline

### 1. Download Flickr30k images

The cached ESNLI pipeline expects:

```text
/data/ESNLI/
  flickr30k-images/
```

Example commands:

```bash
mkdir -p /data/ESNLI/_downloads

huggingface-cli download nlphuji/flickr30k flickr30k-images.zip \
  --repo-type dataset \
  --local-dir /data/ESNLI/_downloads/flickr30k

unzip -q /data/ESNLI/_downloads/flickr30k/flickr30k-images.zip -d /data/ESNLI
```

The codebook builder will fetch the e-SNLI-VE split metadata automatically from the `e-ViL` repository.

### 2. Build the cache

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

Make sure `dataset.cache_root` in the ESNLI cached configs points to that output directory.

### 3. Train the unimodals

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_image_lora.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/cache_text_lora.json --fold 0
```

Run the same commands for folds `1` and `2`.

### 4. Generate CEU pickles

```bash
PYTHONPATH=src python -m synib.entrypoints.get_ceu_cli \
  --dataset esnli \
  --default_config run/configs/ESNLI/default_config_esnli_cache.json \
  --unimodal_configs run/configs/ESNLI/cache_image_lora.json run/configs/ESNLI/cache_text_lora.json \
  --folds 0 1 2 \
  --output_root ./artifacts/ceus
```

Outputs:

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

Cached ensemble:

```bash
./run/esnli/train.sh run/configs/ESNLI/cache_ens.json --fold 0
```

Full experiments:

```bash
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_synib.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_mcr.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_dnr.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_mmpareto.json --fold 0
./run/esnli/train.sh run/configs/ESNLI/full/esnli_full_reconboost.json --fold 0
```

### 6. Evaluate

```bash
./run/esnli/show.sh run/configs/ESNLI/cache_image_lora.json --fold 0
./run/esnli/show.sh run/configs/ESNLI/cache_text_lora.json --fold 0
./run/esnli/show.sh run/configs/ESNLI/cache_synib_lora.json --fold 0
./run/esnli/show.sh run/configs/ESNLI/full/esnli_full_synib.json --fold 0
```

## Typical workflow

```bash
# 1) train a cached multimodal baseline
./run/esnli/train.sh run/configs/ESNLI/cache_lora.json --fold 0

# 2) evaluate the same config
./run/esnli/show.sh run/configs/ESNLI/cache_lora.json --fold 0
```

## More examples

```bash
./run/esnli/train.sh
./run/esnli/train.sh run/configs/ESNLI/cache_synib_lora.json --lr 1e-4
./run/esnli/show.sh run/configs/ESNLI/cache_ens.json
```

## Tip

To view all available e-SNLI configs:

```bash
./run/list_configs.sh esnli
```
