#!/usr/bin/env bash
# Build Hateful Memes feature caches for one encoder stack across all splits.
#
# Usage:
#   ./run/hateful_memes/build_cache.sh <encoder_stack_name> [extra args passed to HM_CodeBook]
#
# Encoder stacks:
#   clip_b16_deberta_base
#   clip_b16_qwen2p5_0p5b
#   clip_l14_deberta_large
#   clip_l14_qwen2p5_1p5b
#   blip2

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

STACK="${1:-clip_b16_deberta_base}"
shift || true

DATA_ROOT="${SYNIB_HM_DATA_ROOT:-./data/HatefulMemes}"
OUT_ROOT="${SYNIB_HM_CACHE_ROOT:-$DATA_ROOT}"
IMAGE_SUBDIR="${SYNIB_HM_IMAGE_SUBDIR:-img_clean}"

case "$STACK" in
  clip_b16_deberta_base)
    VISION_CKPT="openai/clip-vit-base-patch16"
    TEXT_CKPT="microsoft/deberta-v3-base"
    TEXT_KIND="encoder"
    ;;
  clip_b16_qwen2p5_0p5b)
    VISION_CKPT="openai/clip-vit-base-patch16"
    TEXT_CKPT="Qwen/Qwen2.5-0.5B"
    TEXT_KIND="decoder"
    ;;
  clip_l14_deberta_large)
    VISION_CKPT="openai/clip-vit-large-patch14"
    TEXT_CKPT="microsoft/deberta-v3-large"
    TEXT_KIND="encoder"
    ;;
  clip_l14_qwen2p5_1p5b)
    VISION_CKPT="openai/clip-vit-large-patch14"
    TEXT_CKPT="Qwen/Qwen2.5-1.5B"
    TEXT_KIND="decoder"
    ;;
  blip2)
    VISION_CKPT="Salesforce/blip2-opt-2.7b"
    TEXT_CKPT="Salesforce/blip2-opt-2.7b"
    TEXT_KIND="encoder"
    ;;
  *)
    echo "Unknown encoder stack: $STACK" >&2
    echo "Valid: clip_b16_deberta_base | clip_b16_qwen2p5_0p5b | clip_l14_deberta_large | clip_l14_qwen2p5_1p5b | blip2" >&2
    exit 1
    ;;
esac

OUT_DIR="$OUT_ROOT/cache_${STACK}"

for SPLIT in train dev_seen dev_unseen test_seen test_unseen; do
  echo "[hm-cache] stack=$STACK split=$SPLIT -> $OUT_DIR"
  PYTHONPATH="${ROOT_DIR}/src" python -m synib.mydatasets.HatefulMemes.HM_CodeBook \
    --data_root "$DATA_ROOT" \
    --out_dir "$OUT_DIR" \
    --split "$SPLIT" \
    --vision_ckpt "$VISION_CKPT" \
    --text_ckpt "$TEXT_CKPT" \
    --text_kind "$TEXT_KIND" \
    --image_subdir "$IMAGE_SUBDIR" \
    "$@"
done
