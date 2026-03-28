#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-all}"   # all | base | nopre | learned | random
GPU="${GPU:-0}"

SYNERGY_REPO="${SYNERGY_REPO:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/Synergy}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy}"

if [[ ! -d "${SYNERGY_REPO}" ]]; then
  echo "Missing Synergy repo at ${SYNERGY_REPO}" >&2
  exit 1
fi

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV_PATH}"
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
DEFAULT_CONFIG="configs/FactorCL/URFunny/default_config_ur_funny_VT.json"

cd "${SYNERGY_REPO}"

run_case() {
  local label="$1"
  local config="$2"
  shift 2

  for fold in 0 1 2; do
    echo
    echo "[${label}] fold=${fold}"
    echo "CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=. ${PYTHON_BIN} scripts/entrypoints/train.py --config ${config} --default_config ${DEFAULT_CONFIG} --fold ${fold} --validate_with accuracy $*"
    CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=. "${PYTHON_BIN}" scripts/entrypoints/train.py \
      --config "${config}" \
      --default_config "${DEFAULT_CONFIG}" \
      --fold "${fold}" \
      --validate_with accuracy \
      "$@"
  done
}

if [[ "${MODE}" == "all" || "${MODE}" == "base" ]]; then
    run_case \
      "synib_base" \
      "configs/FactorCL/URFunny/release/VT/synprom_RMask.json" \
      --lr 0.001 --wd 0.001 --l 0
fi

if [[ "${MODE}" == "all" || "${MODE}" == "nopre" ]]; then
    run_case \
      "synib_nopre" \
      "configs/FactorCL/URFunny/release/VT/synprom_RMask_nopre.json" \
      --lr 0.001 --wd 0.001
fi

if [[ "${MODE}" == "all" || "${MODE}" == "learned" ]]; then
    run_case \
      "synib_learned" \
      "configs/FactorCL/URFunny/release/VT/synprom_RMask.json" \
      --lr 0.001 --wd 0.001 --l 1 \
      --perturb learned --perturb_fill ema --perturb_lsparse 0.1
fi

if [[ "${MODE}" == "all" || "${MODE}" == "random" ]]; then
    run_case \
      "synib_random" \
      "configs/FactorCL/URFunny/release/VT/synprom_RMask.json" \
      --lr 0.001 --wd 0.001 --l 0.1 \
      --perturb random --perturb_fill ema --perturb_pmin 0.5
fi

if [[ ! " all base nopre learned random " =~ [[:space:]]${MODE}[[:space:]] ]]; then
  echo "Usage: $0 [all|base|nopre|learned|random]" >&2
  exit 1
fi
