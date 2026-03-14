#!/usr/bin/env bash
# run/mustard/verify.sh — MUStARD generation-based verification
#
# Usage:
#   ./run/mustard/verify.sh [options]
#
# Options (all forwarded to mustard_verify.py):
#   --data_root PATH       Path to mustard_raw/ directory
#   --model_name NAME      HuggingFace model ID (default: Qwen/Qwen3-VL-2B-Instruct)
#   --save_base_dir PATH   HuggingFace cache / checkpoint base dir
#   --ckpt PATH            Optional fine-tuned checkpoint (.pt)
#   --num_samples N        Number of samples to verify (default: 5)
#   --split SPLIT          Dataset split: train / val / test (default: test)
#   --max_new_tokens N     Max tokens to generate per prompt (default: 512)
#   --output PATH          Save JSON results to this path
#   --bf16                 Load backbone in bfloat16
#   -v / --verbose         Debug logging
#
# Examples:
#   # Zero-shot verification (pretrained, no LoRA):
#   ./run/mustard/verify.sh --num_samples 10
#
#   # Post-training verification with a fine-tuned checkpoint:
#   ./run/mustard/verify.sh --ckpt path/to/checkpoint.pt --split test --output out/verify.json
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PYTHONPATH="${ROOT_DIR}/src" python -m synib.mydatasets.MUStARD.mustard_verify "$@"
