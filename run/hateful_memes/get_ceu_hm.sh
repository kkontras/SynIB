#!/bin/bash
# Convenience wrapper: extract HM val/test CEU pickles from the 6 unimodal
# checkpoints currently on local disk (3 seeds × {text, image}).
#
# Usage:
#   bash run/hateful_memes/get_ceu_hm.sh [output_tag]
#
# Outputs two pickles under artifacts/ceus/hatefulmemes/ that can be fed back
# into a SynIB HM run via config.model.ceu.{val,test}.
set -euo pipefail

REPO=.
cd "$REPO"

PYBIN=python
SAVE_BASE_DIR=${SAVE_BASE_DIR:-./data/data/2025_data/synergy/HatefulMemes}
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO/artifacts/ceus}
OUTPUT_TAG=${1:-small_tf_deberta}

PYTHONPATH=src "$PYBIN" -m synib.entrypoints.get_ceu_hm_cli \
  --default_config run/configs/hateful_memes/default_config_hm.json \
  --tier_config    run/configs/hateful_memes/tiers/small_tf_deberta.json \
  --unimodal_configs \
        run/configs/hateful_memes/methods/uni_text.json \
        run/configs/hateful_memes/methods/uni_image.json \
  --seeds 27 109 3407 \
  --save_base_dir  "$SAVE_BASE_DIR" \
  --output_root    "$OUTPUT_ROOT" \
  --output_tag     "$OUTPUT_TAG" \
  --test_batch_size 64 \
  --allow_missing
