#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-run/configs/MUStARD/prompt_lora.json}"
if [[ $# -gt 0 ]]; then shift; fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train \
  --config "$CONFIG" \
  --default_config "run/configs/MUStARD/default_config_mustard_tier1.json" \
  "$@"
