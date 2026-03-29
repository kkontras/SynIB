#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATA_ROOTS="${DATA_ROOTS:-/dodrio/scratch/projects/2026_029/kkontras/data/TQA}"
OUT_DIR="${OUT_DIR:-/dodrio/scratch/projects/2026_029/kkontras/data/TQA/cache_qwen3_vl_2b}"
GPU="${1:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SHARD_SIZE="${SHARD_SIZE:-4096}"
NUM_WORKERS="${NUM_WORKERS:-1}"

echo "[build_cache_tier1_full] DATA_ROOTS=${DATA_ROOTS}"
echo "[build_cache_tier1_full] OUT_DIR=${OUT_DIR}"
echo "[build_cache_tier1_full] GPU=${GPU}"
echo "[build_cache_tier1_full] BATCH_SIZE=${BATCH_SIZE}"
echo "[build_cache_tier1_full] SHARD_SIZE=${SHARD_SIZE}"
echo "[build_cache_tier1_full] NUM_WORKERS=${NUM_WORKERS}"

for SPLIT in train val test; do
  echo "[build_cache_tier1_full] Building split=${SPLIT} ..."
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" python \
    src/synib/mydatasets/TQA/TQA_Codebook.py \
    --data_root "${DATA_ROOTS}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --shard_size "${SHARD_SIZE}" \
    --cache_image_embeds \
    --device "cuda:${GPU}"
  echo "[build_cache_tier1_full] Done split=${SPLIT}."
done

echo "[build_cache_tier1_full] All splits done. Cache at: ${OUT_DIR}"
