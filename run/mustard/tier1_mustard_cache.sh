#!/usr/bin/env bash
# tier1_mustard_cache.sh
#
# Orchestrates: (1) cache build, (2) full cached experiment sweep for MUStARD.
# Usage:
#   ./run/mustard/tier1_mustard_cache.sh [GPU] [MODE]
#
# MODE: all | unimodal | ceu | ens | methods
#   all      – build cache (unless SKIP_CACHE_BUILD=1), then run full sweep
#   unimodal – unimodal sweep only (video + text, LR/WD grid)
#   ceu      – CEU stage only (requires BEST_VIDEO_* and BEST_TEXT_* env vars)
#   ens      – ensemble stage only
#   methods  – methods stage only (mcr, mmpareto, synib)
#
# Key env vars:
#   SKIP_CACHE_BUILD=1     – skip cache building (cache already exists)
#   FOLDS_CSV=0,1,2        – comma-separated fold indices (default: 0)
#   UNIMODAL_LRS_CSV       – comma-separated LR values for unimodal sweep
#   UNIMODAL_WDS_CSV       – comma-separated WD values for unimodal sweep
#   BEST_VIDEO_LR / BEST_VIDEO_WD   – best video unimodal hyperparams
#   BEST_TEXT_LR  / BEST_TEXT_WD    – best text unimodal hyperparams
#   METHOD_FIXED_LR / METHOD_FIXED_WD

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

CONDA_ENV_PATH="${CONDA_ENV_PATH:-/scratch/kkontras/miniconda3/envs/synergy_new}"
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)" || true
  conda activate "${CONDA_ENV_PATH}" || true
fi
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV_PATH}/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python"
fi

GPU="${1:-0}"
MODE="${2:-all}"  # all | unimodal | ceu | ens | methods

SKIP_CACHE_BUILD="${SKIP_CACHE_BUILD:-0}"

DEFAULT_CONFIG="run/configs/MUStARD/default_config_mustard_cache_mib.json"
UNIMODAL_VIDEO="run/configs/MUStARD/cache_video_lora.json"
UNIMODAL_TEXT="run/configs/MUStARD/cache_text_lora.json"
ENS_CFG="run/configs/MUStARD/cache_ens.json"
METHODS=(
  "run/configs/MUStARD/cache_synib_lora.json"
  "run/configs/MUStARD/cache_mcr.json"
  "run/configs/MUStARD/cache_mmpareto.json"
)

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.01,0.001}"

BEST_VIDEO_LR="${BEST_VIDEO_LR:-}"
BEST_VIDEO_WD="${BEST_VIDEO_WD:-}"
BEST_TEXT_LR="${BEST_TEXT_LR:-}"
BEST_TEXT_WD="${BEST_TEXT_WD:-}"

METHOD_FIXED_LR="${METHOD_FIXED_LR:-0.0005}"
METHOD_FIXED_WD="${METHOD_FIXED_WD:-0.01}"

run_train() {
  CUDA_VISIBLE_DEVICES="${GPU}" ./run/mustard/train.sh "$@"
}

run_ceu() {
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" \
    -m synib.entrypoints.get_ceu_cli "$@"
}

run_train_safe() {
  if ! run_train "$@"; then
    echo "Method run failed (continuing): $*"
    return 0
  fi
}

need_best_unimodals() {
  [[ -z "${BEST_VIDEO_LR}" || -z "${BEST_VIDEO_WD}" || -z "${BEST_TEXT_LR}" || -z "${BEST_TEXT_WD}" ]]
}

# ---------------------------------------------------------------------------
# Stage: build cache
# ---------------------------------------------------------------------------
run_cache_build() {
  echo "[tier1_mustard_cache] Building embedding cache (all splits) ..."
  bash "${ROOT_DIR}/run/mustard/build_cache_mib.sh" "${GPU}"
  echo "[tier1_mustard_cache] Cache build complete."
}

# ---------------------------------------------------------------------------
# Stage: unimodal sweep
# ---------------------------------------------------------------------------
run_unimodal_sweep() {
  local fold lr wd
  for fold in "${FOLDS[@]}"; do
    for lr in "${UNIMODAL_LRS[@]}"; do
      for wd in "${UNIMODAL_WDS[@]}"; do
        run_train "${UNIMODAL_VIDEO}" --fold "${fold}" --lr "${lr}" --wd "${wd}" \
          --default_config "${DEFAULT_CONFIG}" --validate_with accuracy
        run_train "${UNIMODAL_TEXT}" --fold "${fold}" --lr "${lr}" --wd "${wd}" \
          --default_config "${DEFAULT_CONFIG}" --validate_with accuracy
      done
    done
  done
}

# ---------------------------------------------------------------------------
# Stage: CEU
# ---------------------------------------------------------------------------
run_ceu_stage() {
  if need_best_unimodals; then
    echo "Missing BEST_* vars for CEU stage."
    echo "Required: BEST_VIDEO_LR BEST_VIDEO_WD BEST_TEXT_LR BEST_TEXT_WD"
    exit 1
  fi
  run_ceu \
    --dataset mustard \
    --default_config "${DEFAULT_CONFIG}" \
    --unimodal_configs "${UNIMODAL_VIDEO}" "${UNIMODAL_TEXT}" \
    --folds "${FOLDS[@]}" \
    --lr "${BEST_VIDEO_LR}" \
    --wd "${BEST_VIDEO_WD}" \
    --validate_with accuracy
}

# ---------------------------------------------------------------------------
# Stage: ensemble
# ---------------------------------------------------------------------------
run_ensemble_stage() {
  local fold
  for fold in "${FOLDS[@]}"; do
    run_train_safe "${ENS_CFG}" --fold "${fold}" \
      --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
      --default_config "${DEFAULT_CONFIG}" --validate_with accuracy
  done
}

# ---------------------------------------------------------------------------
# Stage: methods
# ---------------------------------------------------------------------------
run_methods_stage() {
  local fold cfg
  for fold in "${FOLDS[@]}"; do
    for cfg in "${METHODS[@]}"; do
      local cfg_name
      cfg_name="$(basename "${cfg}")"
      case "${cfg_name}" in
        cache_synib_lora.json)
          local l lsparse
          for l in 0.001 0.01 0.1 1; do
            for lsparse in 0.001 0.01 0.1 1 3 5 10; do
              run_train_safe "${cfg}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --default_config "${DEFAULT_CONFIG}" \
                --l "${l}" --perturb learned --perturb_fill ema --perturb_lsparse "${lsparse}" \
                --validate_with accuracy
            done
          done
          ;;
        cache_mcr.json)
          local l multil
          for l in 0.001 0.01 0.1 1; do
            for multil in 0.01 0.1 1; do
              run_train_safe "${cfg}" --fold "${fold}" \
                --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --default_config "${DEFAULT_CONFIG}" \
                --l "${l}" --multil "${multil}" --validate_with accuracy
            done
          done
          ;;
        cache_mmpareto.json)
          local alpha
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            run_train_safe "${cfg}" --fold "${fold}" \
              --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
              --default_config "${DEFAULT_CONFIG}" \
              --alpha "${alpha}" --validate_with accuracy
          done
          ;;
        *)
          run_train_safe "${cfg}" --fold "${fold}" \
            --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
            --default_config "${DEFAULT_CONFIG}" --validate_with accuracy
          ;;
      esac
    done
  done
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "${MODE}" in
  unimodal)
    run_unimodal_sweep
    ;;
  ceu)
    run_ceu_stage
    ;;
  ens)
    run_ensemble_stage
    ;;
  methods)
    run_methods_stage
    ;;
  all)
    if [[ "${SKIP_CACHE_BUILD}" != "1" ]]; then
      run_cache_build
    else
      echo "[tier1_mustard_cache] Skipping cache build (SKIP_CACHE_BUILD=1)."
    fi

    if need_best_unimodals; then
      run_unimodal_sweep
      echo ""
      echo "[tier1_mustard_cache] Unimodal sweep finished."
      echo "Set BEST_VIDEO_LR / BEST_VIDEO_WD and BEST_TEXT_LR / BEST_TEXT_WD,"
      echo "then rerun with MODE=ceu or MODE=all (with SKIP_CACHE_BUILD=1)."
      exit 0
    fi
    run_ceu_stage
    run_ensemble_stage
    run_methods_stage
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    echo "Valid modes: all unimodal ceu ens methods"
    exit 1
    ;;
esac
