#!/usr/bin/env bash
# NOTE: update --partition if the B200 partition name differs from dedicated_big_b200
#SBATCH --time=72:00:00
#SBATCH --cluster=wice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH -G 1
#SBATCH --mem=100G
#SBATCH --partition=gpu_b200
#SBATCH --open-mode=append
#SBATCH --output=/scratch/leuven/350/vsc35057/projects/SynIB/condor_logs/logs_mindwell/slurmout_%J.txt
#SBATCH --error=/scratch/leuven/350/vsc35057/projects/SynIB/condor_logs/logs_mindwell/slurmerr_%J.txt
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/scratch/leuven/350/vsc35057/projects/SynIB}"
ENV_PATH="${ENV_PATH:-/data/leuven/350/vsc35057/synergy_new}"
RUN_SCRIPT="${RUN_SCRIPT:-./run/tqa/train_wice_b200.sh}"
JOB_ARGS="${JOB_ARGS:-}"
GPU="${GPU:-0}"

cd "${PROJECT_ROOT}"
mkdir -p ./condor_logs/logs_mindwell

module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1

source /data/leuven/350/vsc35057/anaconda3/etc/profile.d/conda.sh
conda activate "${ENV_PATH}"

which python
python -V
echo "PWD=${PWD}"
echo "RUN_SCRIPT=${RUN_SCRIPT}"
echo "JOB_ARGS=${JOB_ARGS}"
echo "GPU=${GPU}"

if [[ -n "${JOB_ARGS}" ]]; then
  ARGS=( ${JOB_ARGS} )
else
  ARGS=()
fi

CUDA_VISIBLE_DEVICES="${GPU}" bash "${RUN_SCRIPT}" "${ARGS[@]}"
