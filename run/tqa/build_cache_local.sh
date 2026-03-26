#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOTS="${DATA_ROOTS:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/TQA}"
OUT_DIR="${OUT_DIR:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/TQA/cache_qwen3_vl_2b}"
GPU="${1:-0}"

echo "[build_cache_local] DATA_ROOTS=${DATA_ROOTS}"
echo "[build_cache_local] OUT_DIR=${OUT_DIR}"
echo "[build_cache_local] GPU=${GPU}"

for SPLIT in train val test; do
  echo "[build_cache_local] Building split=${SPLIT} ..."
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" python \
    src/synib/mydatasets/TQA/TQA_Codebook.py \
    --data_root "${DATA_ROOTS}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --batch_size 16 \
    --cache_image_embeds \
    --device "cuda:${GPU}"
  echo "[build_cache_local] Done split=${SPLIT}."
done

echo "[build_cache_local] All splits done. Cache at: ${OUT_DIR}"
