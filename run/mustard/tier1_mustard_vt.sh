#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
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

GPU="${1:-0}"
MODE="${2:-all}"  # all | unimodal | ceu | ens | methods

DEFAULT_CONFIG="run/configs/MUStARD/default_config_mustard_tier1.json"
UNIMODAL_VIDEO="run/configs/MUStARD/prompt_video_lora.json"
UNIMODAL_TEXT="run/configs/MUStARD/prompt_text_lora.json"
ENS_CFG="run/configs/MUStARD/prompt_ens.json"
METHODS=(
  "run/configs/MUStARD/prompt_ens.json"
  # "run/configs/MUStARD/prompt_mmpareto.json"
  # "run/configs/MUStARD/prompt_dnr.json"
  # "run/configs/MUStARD/prompt_mcr.json"
  # "run/configs/MUStARD/prompt_reconboost.json"
  "run/configs/MUStARD/prompt_synib_random.json"
  "run/configs/MUStARD/prompt_synib_learned.json"
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
METHOD_RUN_MODE="${METHOD_RUN_MODE:-all}"  # all | single
METHOD_TARGET="${METHOD_TARGET:-prompt_mmpareto.json}"

run_train() {
  CUDA_VISIBLE_DEVICES="${GPU}" ./run/mustard/train.sh "$@"
}

run_ceu() {
  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -m synib.entrypoints.get_ceu_cli "$@"
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

run_unimodal_sweep() {
  local fold lr wd
  for fold in "${FOLDS[@]}"; do
    for lr in "${UNIMODAL_LRS[@]}"; do
      for wd in "${UNIMODAL_WDS[@]}"; do
        run_train "${UNIMODAL_VIDEO}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
        run_train "${UNIMODAL_TEXT}" --fold "${fold}" --lr "${lr}" --wd "${wd}" --validate_with accuracy
      done
    done
  done
}

run_ceu_stage() {
  if need_best_unimodals; then
    echo "Missing BEST_* vars for CEU stage."
    echo "Required: BEST_VIDEO_LR BEST_VIDEO_WD BEST_TEXT_LR BEST_TEXT_WD"
    exit 1
  fi
  if [[ "${BEST_VIDEO_LR}" != "${BEST_TEXT_LR}" || "${BEST_VIDEO_WD}" != "${BEST_TEXT_WD}" ]]; then
    echo "Warning: get_ceu_cli uses one lr/wd suffix for both unimodals."
    echo "Using video settings for CEU: lr=${BEST_VIDEO_LR} wd=${BEST_VIDEO_WD}"
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

run_ensemble_stage() {
  local fold
  for fold in "${FOLDS[@]}"; do
    run_train_safe "${ENS_CFG}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --validate_with accuracy
  done
}

run_methods_stage() {
  local fold cfg cfg_name
  for fold in "${FOLDS[@]}"; do
    for cfg in "${METHODS[@]}"; do
      cfg_name="$(basename "${cfg}")"
      if [[ "${METHOD_RUN_MODE}" == "single" ]]; then
        local target="${METHOD_TARGET}"
        [[ "${target}" == *.json ]] || target="${target}.json"
        if [[ "${cfg_name}" != "${target}" ]]; then
          continue
        fi
      fi
      case "${cfg_name}" in
        prompt_ens.json)
          run_train_safe "${cfg}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" --validate_with accuracy
          ;;
        prompt_dnr.json)
          local alpha kmepoch
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            for kmepoch in 1 3 5 10; do
              run_train_safe "${cfg}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --alpha "${alpha}" --kmepoch "${kmepoch}" --validate_with accuracy
            done
          done
          ;;
        prompt_mcr.json)
          local l multil
          for l in 0.001 0.01 0.1 1; do
            for multil in 0.01 0.1 1; do
              run_train_safe "${cfg}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --multil "${multil}" --validate_with accuracy
            done
          done
          ;;
        prompt_mmpareto.json)
          local alpha
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            run_train_safe "${cfg}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
              --alpha "${alpha}" --validate_with accuracy
          done
          ;;
        prompt_reconboost.json)
          local alpha recon_stages recon_weight1
          for alpha in 0.5 1.0 1.5 2.0 3.0 5.0; do
            for recon_stages in 1 4 10; do
              for recon_weight1 in 1 3 5 10; do
                run_train_safe "${cfg}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                  --alpha "${alpha}" \
                  --recon_weight1 "${recon_weight1}" --recon_weight2 1 \
                  --recon_epochstages "${recon_stages}" --recon_ensemblestages "${recon_stages}" \
                  --validate_with accuracy
              done
            done
          done
          ;;
        prompt_synib_learned.json)
          local l lsparse
          for l in 0.001 0.01 0.1 1; do
            for lsparse in 0.001 0.01 0.1 1 3 5 10; do
              run_train_safe "${cfg}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --perturb learned --perturb_fill ema --perturb_lsparse "${lsparse}" \
                --validate_with accuracy
            done
          done
          ;;
        prompt_synib_random.json)
          local l pmin
          for l in 0.001 0.01 0.1 1; do
            for pmin in 0.1 0.3 0.5 0.7 0.9; do
              run_train_safe "${cfg}" --fold "${fold}" --lr "${METHOD_FIXED_LR}" --wd "${METHOD_FIXED_WD}" \
                --l "${l}" --perturb random --perturb_fill ema --perturb_pmin "${pmin}" \
                --validate_with accuracy
            done
          done
          ;;
      esac
    done
  done
}

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
    if need_best_unimodals; then
      run_unimodal_sweep
      echo "Unimodal sweep finished."
      echo "Set BEST_VIDEO_LR/BEST_VIDEO_WD and BEST_TEXT_LR/BEST_TEXT_WD, then rerun with MODE=ceu or MODE=all."
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
