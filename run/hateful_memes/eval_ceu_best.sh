#!/usr/bin/env bash
# Evaluate a trained best-model checkpoint on val+test and print CEU.
# Uses src/synib/entrypoints/eval_ceu_only.py which patches Agent.run to a
# no-op; finalize() still runs load_best_model + validate(test_set=True).
#
# Usage:
#   ./run/hateful_memes/eval_ceu_best.sh <tier.json> <method.json> <seed> <save_dir_basename>

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_CONFIG="run/configs/hateful_memes/default_config_hm.json"
TIER="$1"; METHOD="$2"; SEED="$3"; SAVE_DIR_BASENAME="$4"

MERGED="$(mktemp -t hm_eval_merged_XXXXXX.json)"
trap 'rm -f "$MERGED"' EXIT

python - <<PY
import json
def deep_merge(b, o):
    for k, v in o.items():
        if k in b and isinstance(b[k], dict) and isinstance(v, dict): deep_merge(b[k], v)
        else: b[k] = v
    return b
default = json.load(open("${DEFAULT_CONFIG}"))
tier    = json.load(open("${TIER}"))
method  = json.load(open("${METHOD}"))
cfg = deep_merge(deep_merge(default, tier), method)
cfg["training_params"]["seed"] = ${SEED}
cfg["training_params"]["wandb_disable"] = True
cfg["model"]["save_dir"] = "${SAVE_DIR_BASENAME}"
cfg["model"]["start_over"] = True
cfg["model"]["no_model_save"] = True
json.dump(cfg, open("${MERGED}", "w"), indent=2)
PY

echo "[eval_ceu] merged=$MERGED save_dir=${SAVE_DIR_BASENAME}"

PYTHONPATH="${ROOT_DIR}/src" python \
  -m synib.entrypoints.eval_ceu_only \
  --config "$MERGED" \
  --default_config "$DEFAULT_CONFIG"
