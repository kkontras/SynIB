#!/bin/bash
set -euo pipefail

echo "Starting TQA Tier1 training job"

if [ -z "${CONDA_PREFIX:-}" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
    conda activate /esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy
fi

PYTHON_BIN="${PYTHON:-$(command -v python)}"
echo "$PYTHON_BIN"
"$PYTHON_BIN" -V

cd /esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB
echo "$PWD"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus || true
fi

echo "Executing: bash run/tqa/train_tier1_local.sh $@"
bash run/tqa/train_tier1_local.sh "$@"

echo "Job finished"
