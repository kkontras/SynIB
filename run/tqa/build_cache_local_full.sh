#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOTS="${DATA_ROOTS:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/TQA}"
OUT_DIR="${OUT_DIR:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/TQA/cache_qwen3_vl_2b}"
GPU="${1:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SHARD_SIZE="${SHARD_SIZE:-4096}"

echo "[build_cache_local_full] DATA_ROOTS=${DATA_ROOTS}"
echo "[build_cache_local_full] OUT_DIR=${OUT_DIR}"
echo "[build_cache_local_full] GPU=${GPU}"
echo "[build_cache_local_full] BATCH_SIZE=${BATCH_SIZE}"
echo "[build_cache_local_full] SHARD_SIZE=${SHARD_SIZE}"

for SPLIT in train val test; do
  echo "[build_cache_local_full] Building split=${SPLIT} ..."
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" python \
    src/synib/mydatasets/TQA/TQA_Codebook.py \
    --data_root "${DATA_ROOTS}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --shard_size "${SHARD_SIZE}" \
    --cache_image_embeds \
    --device "cuda:${GPU}"
  echo "[build_cache_local_full] Done split=${SPLIT}."
done

echo "[build_cache_local_full] All splits done. Cache at: ${OUT_DIR}"
