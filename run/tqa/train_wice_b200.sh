#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_CONFIG="run/configs/TQA/default_config_tqa_cache_wice_b200_train.json"

usage() {
  cat <<EOF
Usage: $0 <target> [method-config.json] [extra train args]

Targets:
  image  -> run/configs/TQA/cache_image_lora_tier1.json
  text   -> run/configs/TQA/cache_text_lora_tier1.json
  joint  -> run/configs/TQA/cache_lora_tier1.json
  synib  -> run/configs/TQA/cache_synib_lora_tier1.json
  iha    -> run/configs/TQA/cache_iha_lora_tier1.json

Effective batch size = batch_size (2) x gradient_accumulation_steps (2) = 4

Examples:
  $0 iha --fold 0 --lr 0.0001 --wd 0.0 --iha_init identity --iha_layers all --iha_lr 0.005
  $0 synib --fold 0 --lr 0.0001 --wd 0.0 --perturb random --perturb_fill ema --perturb_pmin 0.5 --l 0.1
  $0 synib --fold 0 --lr 0.0001 --wd 0.0 --perturb learned --perturb_fill ema --perturb_lsparse 0.1 --l 0.1
EOF
  exit 1
}

[[ $# -eq 0 ]] && usage

TARGET="$1"
shift

case "$TARGET" in
  image)
    CONFIG="run/configs/TQA/cache_image_lora_tier1.json"
    ;;
  text)
    CONFIG="run/configs/TQA/cache_text_lora_tier1.json"
    ;;
  joint)
    CONFIG="run/configs/TQA/cache_lora_tier1.json"
    ;;
  synib)
    CONFIG="run/configs/TQA/cache_synib_lora_tier1.json"
    ;;
  iha)
    CONFIG="run/configs/TQA/cache_iha_lora_tier1.json"
    ;;
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown TQA b200 target: $TARGET"
    usage
    ;;
esac

if [[ $# -gt 0 ]]; then
  case "$1" in
    *.json|run/configs/*)
      CONFIG="$1"
      shift
      ;;
  esac
fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train \
  --config "$CONFIG" \
  --default_config "$DEFAULT_CONFIG" \
  "$@"
