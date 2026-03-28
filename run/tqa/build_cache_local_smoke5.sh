#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOTS="${DATA_ROOTS:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/TQA}"
OUT_DIR="${OUT_DIR:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/TQA/cache_qwen3_vl_2b_smoke5}"
GPU="${1:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-5}"
BATCH_SIZE="${BATCH_SIZE:-5}"

echo "[build_cache_local_smoke5] DATA_ROOTS=${DATA_ROOTS}"
echo "[build_cache_local_smoke5] OUT_DIR=${OUT_DIR}"
echo "[build_cache_local_smoke5] GPU=${GPU}"
echo "[build_cache_local_smoke5] MAX_SAMPLES=${MAX_SAMPLES}"
echo "[build_cache_local_smoke5] BATCH_SIZE=${BATCH_SIZE}"

rm -rf "${OUT_DIR}"

for SPLIT in train val test; do
  echo "[build_cache_local_smoke5] Building split=${SPLIT} ..."
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" python \
    src/synib/mydatasets/TQA/TQA_Codebook.py \
    --data_root "${DATA_ROOTS}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --cache_image_embeds \
    --device "cuda:${GPU}"
  echo "[build_cache_local_smoke5] Done split=${SPLIT}."
done

echo "[build_cache_local_smoke5] All splits done. Cache at: ${OUT_DIR}"
