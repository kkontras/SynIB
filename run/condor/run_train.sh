#!/bin/bash
set -euo pipefail

echo "Starting SynIB training job"

export PATH="/users/sista/kkontras/anaconda3/bin:$PATH"
source ~/anaconda3/etc/profile.d/conda.sh
conda activate /esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new

which python
python -V

cd /esat/smcdata/users/kkontras/Image_Dataset/no_backup/git/SynIB
echo "$PWD"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --list-gpus || true
fi

echo "Executing: PYTHONPATH=src python -m synib.entrypoints.train $@"
PYTHONPATH=src python -m synib.entrypoints.train "$@"

echo "Job finished"
