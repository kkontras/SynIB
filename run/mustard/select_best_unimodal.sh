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

DEFAULT_CONFIG="${DEFAULT_CONFIG:-run/configs/MUStARD/default_config_mustard.json}"
UNIMODAL_VIDEO="${UNIMODAL_VIDEO:-run/configs/MUStARD/prompt_video_lora.json}"
UNIMODAL_TEXT="${UNIMODAL_TEXT:-run/configs/MUStARD/prompt_text_lora.json}"
VALIDATE_WITH="${VALIDATE_WITH:-accuracy}"
SELECT_WITH="${SELECT_WITH:-val}"  # val | test
RUN_SHOW="${RUN_SHOW:-0}"
SHOW_FOLD="${SHOW_FOLD:-0}"

usage() {
  cat <<'EOF'
Usage:
  ./run/mustard/select_best_unimodal.sh [--select_with val|test] [--run_show 0|1] [--show_fold N] [--validate_with METRIC]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --select_with)
      SELECT_WITH="${2:-}"
      shift 2
      ;;
    --run_show)
      RUN_SHOW="${2:-}"
      shift 2
      ;;
    --show_fold)
      SHOW_FOLD="${2:-}"
      shift 2
      ;;
    --validate_with)
      VALIDATE_WITH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

IFS=',' read -r -a FOLDS <<< "${FOLDS_CSV:-0,1,2}"
IFS=',' read -r -a UNIMODAL_LRS <<< "${UNIMODAL_LRS_CSV:-0.001,0.0005,0.0001}"
IFS=',' read -r -a UNIMODAL_WDS <<< "${UNIMODAL_WDS_CSV:-0.01,0.001}"

select_best() {
  local prefix="$1"
  "${PYTHON_BIN}" - "${DEFAULT_CONFIG}" "${prefix}" "${VALIDATE_WITH}" "${SELECT_WITH}" "${FOLDS[*]}" "${UNIMODAL_LRS[*]}" "${UNIMODAL_WDS[*]}" <<'PY'
import json
import math
import os
import sys

default_config = sys.argv[1]
prefix = sys.argv[2]
validate_with = sys.argv[3]
select_with = sys.argv[4]
folds = sys.argv[5].split()
lrs = sys.argv[6].split()
wds = sys.argv[7].split()

with open(default_config, "r", encoding="utf-8") as handle:
    save_base = json.load(handle)["model"]["save_base_dir"]

def load_ckpt(path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def get_score(path):
    ckpt = load_ckpt(path)
    if select_with == "test":
        score = ckpt["post_test_results"]["acc"]["combined"]
    else:
        score = ckpt["logs"]["best_logs"][f"best_v{validate_with}"]["acc"]["combined"]
    if hasattr(score, "item"):
        score = score.item()
    return float(score)

rows = []
best = None
for lr in lrs:
    for wd in wds:
        per_fold = []
        ok = True
        for fold in folds:
            name = f"{prefix}_fold{fold}_vld{validate_with}_lr{lr}_wd{wd}.pth.tar"
            path = os.path.join(save_base, name)
            if not os.path.exists(path):
                ok = False
                break
            try:
                val = get_score(path)
            except Exception:
                ok = False
                break
            if math.isnan(val):
                ok = False
                break
            per_fold.append(val)
        if not ok or not per_fold:
            continue
        mean_val = sum(per_fold) / len(per_fold)
        rows.append((mean_val, lr, wd, per_fold))
        if best is None or mean_val > best[0]:
            best = (mean_val, lr, wd, per_fold)

if best is None:
    print("BEST_LR=NONE")
    print("BEST_WD=NONE")
    print("BEST_MEAN=nan")
    print("BEST_FOLDS=none")
    sys.exit(0)

print(f"BEST_LR={best[1]}")
print(f"BEST_WD={best[2]}")
print(f"BEST_MEAN={best[0]:.6f}")
print("BEST_FOLDS=" + ",".join(f"{x:.6f}" for x in best[3]))
PY
}

eval "$(select_best "MUStARD_video_lora")"
BEST_VIDEO_LR="${BEST_LR}"
BEST_VIDEO_WD="${BEST_WD}"
BEST_VIDEO_MEAN="${BEST_MEAN}"
BEST_VIDEO_FOLDS="${BEST_FOLDS}"

eval "$(select_best "MUStARD_text_lora")"
BEST_TEXT_LR="${BEST_LR}"
BEST_TEXT_WD="${BEST_WD}"
BEST_TEXT_MEAN="${BEST_MEAN}"
BEST_TEXT_FOLDS="${BEST_FOLDS}"

echo "Video best: lr=${BEST_VIDEO_LR} wd=${BEST_VIDEO_WD} mean_${SELECT_WITH}_acc=${BEST_VIDEO_MEAN} per_fold=[${BEST_VIDEO_FOLDS}]"
echo "Text  best: lr=${BEST_TEXT_LR} wd=${BEST_TEXT_WD} mean_${SELECT_WITH}_acc=${BEST_TEXT_MEAN} per_fold=[${BEST_TEXT_FOLDS}]"
echo
echo "export BEST_VIDEO_LR=${BEST_VIDEO_LR}"
echo "export BEST_VIDEO_WD=${BEST_VIDEO_WD}"
echo "export BEST_TEXT_LR=${BEST_TEXT_LR}"
echo "export BEST_TEXT_WD=${BEST_TEXT_WD}"

if [[ "${RUN_SHOW}" == "1" && "${BEST_VIDEO_LR}" != "NONE" ]]; then
  ./run/mustard/show.sh "${UNIMODAL_VIDEO}" --fold "${SHOW_FOLD}" --lr "${BEST_VIDEO_LR}" --wd "${BEST_VIDEO_WD}" --validate_with "${VALIDATE_WITH}"
fi
if [[ "${RUN_SHOW}" == "1" && "${BEST_TEXT_LR}" != "NONE" ]]; then
  ./run/mustard/show.sh "${UNIMODAL_TEXT}" --fold "${SHOW_FOLD}" --lr "${BEST_TEXT_LR}" --wd "${BEST_TEXT_WD}" --validate_with "${VALIDATE_WITH}"
fi
