#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_CONFIG="run/configs/TQA/default_config_tqa_cache_tier1_train.json"

usage() {
  cat <<EOF
Usage: $0 <target> [method-config.json] [extra train args]

Targets:
  image  -> run/configs/TQA/cache_image_lora_tier1.json
  text   -> run/configs/TQA/cache_text_lora_tier1.json
  joint  -> run/configs/TQA/cache_lora_tier1.json

Examples:
  $0 image --fold 0 --lr 0.0001 --wd 0.0001 --validate_with accuracy
  $0 text --fold 0 --lr 0.0001 --wd 0.0001 --validate_with accuracy
  $0 joint --fold 0 --lr 0.0005 --wd 0.0001 --validate_with accuracy
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
  -h|--help)
    usage
    ;;
  *)
    echo "Unknown TQA tier1 target: $TARGET"
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

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train   --config "$CONFIG"   --default_config "$DEFAULT_CONFIG"   "$@"
