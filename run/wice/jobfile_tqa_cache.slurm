#!/usr/bin/env bash
set -euo pipefail

#SBATCH --time=12:00:00
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -G 1
#SBATCH --mem=64G
#SBATCH --partition=dedicated_big_gpu
#SBATCH --open-mode=append
#SBATCH --output=/scratch/leuven/350/vsc35057/projects/SynIB/condor_logs/logs_wice/slurmout_%J.txt
#SBATCH --error=/scratch/leuven/350/vsc35057/projects/SynIB/condor_logs/logs_wice/slurmerr_%J.txt

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/leuven/350/vsc35057/projects/SynIB}"
ENV_PATH="${ENV_PATH:-/data/leuven/350/vsc35057/synib_env}"
DATA_ROOTS="${DATA_ROOTS:-/scratch/leuven/350/vsc35057/data/TQA}"
OUT_DIR="${OUT_DIR:-/scratch/leuven/350/vsc35057/data/TQA/cache_qwen3_vl_2b}"
GPU="${GPU:-0}"

cd "${PROJECT_ROOT}"
mkdir -p ./condor_logs/logs_wice

module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

source /data/leuven/350/vsc35057/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"

which python
python -V
echo "PWD=${PWD}"
echo "DATA_ROOTS=${DATA_ROOTS}"
echo "OUT_DIR=${OUT_DIR}"
echo "GPU=${GPU}"

DATA_ROOTS="${DATA_ROOTS}" OUT_DIR="${OUT_DIR}" \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  bash run/tqa/build_cache_wice.sh "${GPU}"
