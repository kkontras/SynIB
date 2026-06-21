#!/usr/bin/env bash
# Phase-A first slice: on the Small-TF DeBERTa tier, build the full
# clip_b16_deberta_base cache for the 3 labelled splits actually used
# (train / dev_unseen / test_seen) and run the 3 core methods.
#
# Methods: vanilla, synib, synib_u. Skips the unimodal baselines (they use
# the same encoder stack; add them by reusing this script with another method).

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYBIN=python
DATA_ROOT=./data/HatefulMemes
CACHE_ROOT="$DATA_ROOT/cache_clip_b16_deberta_base"
SAVE_BASE=./data/data/2025_data/synergy/HatefulMemes

export PYTHONPATH="${ROOT_DIR}/src"

# ---------- 1) cache build (skip splits that already have a shard) ----------
for SPLIT in train dev_unseen test_seen; do
  if [[ -f "$CACHE_ROOT/$SPLIT/manifest.jsonl" ]]; then
    echo "[cache] $SPLIT already built at $CACHE_ROOT/$SPLIT — skipping"
    continue
  fi
  echo "[cache] building $SPLIT ..."
  "$PYBIN" -m synib.mydatasets.HatefulMemes.HM_CodeBook \
    --data_root "$DATA_ROOT" \
    --out_dir "$CACHE_ROOT" \
    --split "$SPLIT" \
    --vision_ckpt openai/clip-vit-base-patch16 \
    --text_ckpt microsoft/deberta-v3-base \
    --text_kind encoder \
    --image_subdir img \
    --batch_size 32 \
    --shard_size 4000
done

# ---------- 2) training: vanilla + synib + synib_u ----------
mkdir -p "$SAVE_BASE"

run_method () {
  local METHOD_NAME="$1"
  local METHOD_CFG="run/configs/hateful_memes/methods/${METHOD_NAME}.json"
  local MERGED; MERGED="$(mktemp -t hm_real_${METHOD_NAME}_XXXXXX.json)"

  "$PYBIN" - <<PY
import json, copy
def dm(a,b):
    for k,v in b.items():
        if k in a and isinstance(a[k],dict) and isinstance(v,dict): dm(a[k],v)
        else: a[k]=v
    return a
default = json.load(open('run/configs/hateful_memes/default_config_hm.json'))
tier    = json.load(open('run/configs/hateful_memes/tiers/small_tf_deberta.json'))
method  = json.load(open('${METHOD_CFG}'))
cfg = dm(dm(copy.deepcopy(default), tier), method)
cfg['training_params']['batch_size'] = 32
cfg['training_params']['test_batch_size'] = 64
cfg['training_params']['wandb_disable'] = True
cfg['early_stopping']['max_epoch'] = 20
cfg['early_stopping']['validate_every'] = 50
cfg['early_stopping']['save_every_valstep'] = 5
cfg['early_stopping']['n_steps_stop'] = 10
cfg['model']['start_over'] = True
# methodology-specific suffix for save_dir to avoid clobbering between runs
cfg['model']['save_dir'] = cfg['model']['save_dir'].replace('.pth.tar', f'_${METHOD_NAME}.pth.tar')
json.dump(cfg, open('${MERGED}','w'), indent=2)
PY
  echo "[${METHOD_NAME}] merged config -> ${MERGED}"
  "$PYBIN" -m synib.entrypoints.train \
    --config "$MERGED" \
    --default_config run/configs/hateful_memes/default_config_hm.json \
    --fold 0 2>&1 | tee "/tmp/hm_${METHOD_NAME}.log"
  rm -f "$MERGED"
  echo "[${METHOD_NAME}] DONE"
}

for M in vanilla synib synib_u; do
  echo
  echo "================ ${M} ================"
  run_method "$M"
done

echo
echo "PHASE_A_SMALL_DEBERTA_DONE"
