#!/usr/bin/env bash
# run/mustard/show.sh — MUStARD evaluation / result display wrapper
#
# Usage:
#   ./run/mustard/show.sh [config] [options]
#
# Examples:
#   ./run/mustard/show.sh run/configs/MUStARD/prompt_lora.json
#   ./run/mustard/show.sh run/configs/MUStARD/prompt_lora.json --fold 1
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-run/configs/MUStARD/prompt_lora.json}"
if [[ $# -gt 0 ]]; then shift; fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.show \
  --config "$CONFIG" \
  --default_config "run/configs/MUStARD/default_config_mustard.json" \
  "$@"
