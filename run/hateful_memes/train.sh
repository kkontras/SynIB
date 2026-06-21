#!/usr/bin/env bash
# Launch an HM training run with a tier config and (optionally) a method overlay.
#
# Usage:
#   ./run/hateful_memes/train.sh <tier.json>                 # tier only (vanilla by default)
#   ./run/hateful_memes/train.sh <tier.json> <method.json>   # tier merged with method overlay
#   ./run/hateful_memes/train.sh <tier.json> <method.json> --fold 0 --lr 1e-4 ...
#
# When a method JSON is supplied, it is deep-merged on top of the tier JSON into
# a temp file that is then passed to train.py as --config. The default_config
# (optimizer/scheduler/dataset paths) is always supplied separately.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_CONFIG="run/configs/hateful_memes/default_config_hm.json"
TIER="${1:-run/configs/hateful_memes/tiers/small_tf_deberta.json}"
shift || true

METHOD=""
if [[ $# -gt 0 ]] && [[ "$1" != --* ]]; then
  METHOD="$1"; shift
fi

if [[ -n "$METHOD" ]]; then
  MERGED="$(mktemp -t hm_merged_XXXXXX.json)"
  trap 'rm -f "$MERGED"' EXIT
  python - <<PY
import json, sys
def deep_merge(base, over):
    for k, v in over.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base
base = json.load(open("${TIER}"))
over = json.load(open("${METHOD}"))
merged = deep_merge(base, over)
json.dump(merged, open("${MERGED}", "w"), indent=2)
PY
  CONFIG="$MERGED"
  echo "[hm-train] merged tier=${TIER} method=${METHOD} -> ${MERGED}"
else
  CONFIG="$TIER"
  echo "[hm-train] tier-only config=${CONFIG}"
fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train \
  --config "$CONFIG" \
  --default_config "$DEFAULT_CONFIG" \
  "$@"
