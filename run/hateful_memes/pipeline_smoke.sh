#!/usr/bin/env bash
# End-to-end HM pipeline smoke: waits for download, verifies layout, builds
# cache on dev_seen only, then runs a tiny vanilla training smoke.
#
# Run after `download` window has started. Intended to be launched in another
# tmux window and left to chain.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT=./data/HatefulMemes
DOWNLOAD_LOG=/tmp/hm_download.log
PYBIN=python

echo "[1/4] waiting for download to finish ..."
until grep -q "DOWNLOAD_DONE\|DOWNLOAD_EXIT" "$DOWNLOAD_LOG" 2>/dev/null; do
  sleep 10
done
echo "[1/4] download log shows completion:"
tail -3 "$DOWNLOAD_LOG"

echo
echo "[2/4] verifying layout ..."
for split in train dev_seen dev_unseen test_seen test_unseen; do
  test -f "$DATA_ROOT/${split}.jsonl" || { echo "missing $DATA_ROOT/${split}.jsonl"; exit 1; }
done
test -d "$DATA_ROOT/img" || { echo "missing $DATA_ROOT/img"; exit 1; }
N_IMG=$(find "$DATA_ROOT/img" -name '*.png' | wc -l)
echo "  OK: 5 jsonls + $N_IMG png files under img/"

echo
echo "[3/4] building cache on dev_seen (clip_b16_deberta_base) ..."
PYTHONPATH="${ROOT_DIR}/src" "$PYBIN" -m synib.mydatasets.HatefulMemes.HM_CodeBook \
  --data_root "$DATA_ROOT" \
  --out_dir "$DATA_ROOT/cache_clip_b16_deberta_base" \
  --split dev_seen \
  --vision_ckpt openai/clip-vit-base-patch16 \
  --text_ckpt microsoft/deberta-v3-base \
  --text_kind encoder \
  --image_subdir img \
  --batch_size 16 \
  --shard_size 2000

echo
echo "[4/4] running vanilla training smoke (dev_seen only, 3 epochs) ..."
MERGED="$(mktemp -t hm_smoke_XXXXXX.json)"
"$PYBIN" - <<PY
import json, copy
def dm(a,b):
    for k,v in b.items():
        if k in a and isinstance(a[k],dict) and isinstance(v,dict): dm(a[k],v)
        else: a[k]=v
    return a
default = json.load(open('run/configs/hateful_memes/default_config_hm.json'))
tier    = json.load(open('run/configs/hateful_memes/tiers/small_tf_deberta.json'))
method  = json.load(open('run/configs/hateful_memes/methods/vanilla.json'))
cfg = dm(dm(copy.deepcopy(default), tier), method)
# smoke overrides: tiny, everything from dev_seen
cfg['dataset']['train_split'] = 'dev_seen'
cfg['dataset']['valid_split'] = 'dev_seen'
cfg['dataset']['test_split']  = 'dev_seen'
cfg.setdefault('dataset', {}); cfg['dataset']['train_max_items'] = 64
cfg['training_params']['batch_size'] = 8
cfg['training_params']['test_batch_size'] = 8
cfg['training_params']['wandb_disable'] = True
cfg['training_params']['data_loader_workers'] = 0
cfg['early_stopping']['max_epoch'] = 3
cfg['early_stopping']['validate_every'] = 5
cfg['early_stopping']['save_every_valstep'] = 1
cfg['early_stopping']['n_steps_stop'] = 1000
cfg['model']['no_model_save'] = True
cfg['model']['start_over'] = True
json.dump(cfg, open('${MERGED}','w'), indent=2)
PY
echo "  merged smoke config at $MERGED"

PYTHONPATH="${ROOT_DIR}/src" "$PYBIN" -m synib.entrypoints.train \
  --config "$MERGED" \
  --default_config run/configs/hateful_memes/default_config_hm.json \
  --fold 0

rm -f "$MERGED"
echo
echo "PIPELINE_SMOKE_DONE"
