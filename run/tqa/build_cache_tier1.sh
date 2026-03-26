#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOTS="${DATA_ROOTS:-/dodrio/scratch/projects/2026_029/kkontras/data/TQA}"
OUT_DIR="${OUT_DIR:-/dodrio/scratch/projects/2026_029/kkontras/data/TQA/cache_qwen3_vl_2b}"
GPU="${1:-0}"

echo "[build_cache_tier1] DATA_ROOTS=${DATA_ROOTS}"
echo "[build_cache_tier1] OUT_DIR=${OUT_DIR}"
echo "[build_cache_tier1] GPU=${GPU}"

for SPLIT in train val test; do
  echo "[build_cache_tier1] Building split=${SPLIT} ..."
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" python \
    src/synib/mydatasets/TQA/TQA_Codebook.py \
    --data_root "${DATA_ROOTS}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --batch_size 2 \
    --cache_image_embeds \
    --device "cuda:${GPU}"
  echo "[build_cache_tier1] Done split=${SPLIT}."
done

echo "[build_cache_tier1] All splits done. Cache at: ${OUT_DIR}"
