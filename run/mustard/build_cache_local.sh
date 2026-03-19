#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${CONDA_ENV_PATH}" || true
fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

DATA_ROOTS="${DATA_ROOTS:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MUSTARD/mustard_raw}"
OUT_DIR="${OUT_DIR:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/MUSTARD/cache_qwen3_vl_2b}"
GPU="${1:-0}"

echo "[build_cache_local] DATA_ROOTS=${DATA_ROOTS}"
echo "[build_cache_local] OUT_DIR=${OUT_DIR}"
echo "[build_cache_local] GPU=${GPU}"

for SPLIT in train val test; do
  echo "[build_cache_local] Building split=${SPLIT} ..."
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" \
    src/synib/mydatasets/MUStARD/MUStARD_CodeBook.py \
    --data_roots "${DATA_ROOTS}" \
    --out_dir "${OUT_DIR}" \
    --split "${SPLIT}" \
    --batch_size 1 \
    --device "cuda:${GPU}"
  echo "[build_cache_local] Done split=${SPLIT}."
done

echo "[build_cache_local] All splits done. Cache at: ${OUT_DIR}"
