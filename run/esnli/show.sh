#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-run/configs/ESNLI/cache_lora.json}"
if [[ $# -gt 0 ]]; then shift; fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.show \
  --config "$CONFIG" \
  --default_config "run/configs/ESNLI/default_config_esnli_cache.json" \
  "$@"
