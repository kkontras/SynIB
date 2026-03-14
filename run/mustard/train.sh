#!/usr/bin/env bash
# run/mustard/train.sh — MUStARD training wrapper
#
# Usage:
#   ./run/mustard/train.sh [config] [options]
#
# Examples:
#   ./run/mustard/train.sh run/configs/MUStARD/prompt_lora.json
#   ./run/mustard/train.sh run/configs/MUStARD/prompt_text_lora.json --fold 1
#   ./run/mustard/train.sh run/configs/MUStARD/prompt_mcr.json --fold 0
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-run/configs/MUStARD/prompt_lora.json}"
if [[ $# -gt 0 ]]; then shift; fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train \
  --config "$CONFIG" \
  --default_config "run/configs/MUStARD/default_config_mustard.json" \
  "$@"
