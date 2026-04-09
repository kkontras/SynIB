#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOTS="${DATA_ROOTS:-/scratch/leuven/350/vsc35057/data/TQA}"
OUT_DIR="${OUT_DIR:-/scratch/leuven/350/vsc35057/data/TQA/cache_qwen3_vl_2b}"
GPU="${1:-0}"

echo "[build_cache_wice] DATA_ROOTS=${DATA_ROOTS}"
echo "[build_cache_wice] OUT_DIR=${OUT_DIR}"
echo "[build_cache_wice] GPU=${GPU}"

for SPLIT in train val test; do
  echo "[build_cache_wice] Building split=${SPLIT} ..."
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" python \
    src/synib/mydatasets/TQA/TQA_Codebook.py \
    --data_root "${DATA_ROOTS}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --batch_size 4 \
    --cache_image_embeds \
    --device "cuda:${GPU}"
  echo "[build_cache_wice] Done split=${SPLIT}."
done

echo "[build_cache_wice] All splits done. Cache at: ${OUT_DIR}"
