#!/bin/bash
# Condor executable for the rebuttal reference-distribution ablation (Reviewer RpxH, W1).
# Runs from the paper-release SynIB repo (branch rebuttal-ref-ablation) with the pinned env.
set -euo pipefail

echo "Starting SynIB reference-ablation job"

# Pinned runtime env — activate by FULL path (never rely on ambient $PATH on exec nodes).
PYTHON_BIN=/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy/bin/python
"$PYTHON_BIN" -V

# wandb must never try to go online on exec nodes (default_config_hm has wandb_disable: false).
export WANDB_MODE=disabled

cd /esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB
echo "$PWD  (git $(git rev-parse --short HEAD 2>/dev/null || echo unknown))"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus || true
fi

echo "Executing: PYTHONPATH=src $PYTHON_BIN -m synib.entrypoints.train $@"
PYTHONPATH=src "$PYTHON_BIN" -m synib.entrypoints.train "$@"

echo "Job finished"
