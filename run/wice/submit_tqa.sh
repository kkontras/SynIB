#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/scratch/leuven/350/vsc35057/projects/SynIB}"
ENV_PATH="${ENV_PATH:-/scratch/leuven/350/vsc35057/synergy_new}"
WICE_ACCOUNT="${WICE_ACCOUNT:-lp_big_wice_gpu}"
RUN_SCRIPT_DEFAULT="./run/tqa/train_wice.sh"

usage() {
  cat <<'EOF'
Usage:
  ./run/wice/submit_tqa.sh --cache
  ./run/wice/submit_tqa.sh --jobs-file run/wice/tqa_wice_jobs.txt
  ./run/wice/submit_tqa.sh --run-script ./run/tqa/train_wice.sh -- iha --fold 0 --lr 0.0001 --wd 0.0 --iha_init identity --iha_layers all --iha_lr 0.005

Environment variables:
  GPU            GPU index to use (default: 0)
  PROJECT_ROOT   Path to SynIB on WICE (default: /scratch/leuven/350/vsc35057/projects/SynIB)
  ENV_PATH       Conda env path (default: /data/leuven/350/vsc35057/synib_env)
  WICE_ACCOUNT   SLURM billing account (default: lp_big_wice_gpu)
EOF
}

submit_cache() {
  sbatch -A "${WICE_ACCOUNT}" \
    -M wice \
    --export=GPU="${GPU}",PROJECT_ROOT="${PROJECT_ROOT}",ENV_PATH="${ENV_PATH}" \
    ./run/wice/jobfile_tqa_cache.slurm
}

submit_job() {
  local run_script="$1"
  local job_args="$2"
  echo "Submitting: ${run_script} ${job_args}"
  sbatch -A "${WICE_ACCOUNT}" \
    -M wice \
    --export=GPU="${GPU}",PROJECT_ROOT="${PROJECT_ROOT}",ENV_PATH="${ENV_PATH}",RUN_SCRIPT="${run_script}",JOB_ARGS="${job_args}" \
    ./run/wice/jobfile_tqa_train.slurm
}

submit_jobs_file() {
  local jobs_file="$1"
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line// }" ]] && continue
    [[ "${line}" =~ ^# ]] && continue
    if [[ "${line}" != *"::"* ]]; then
      echo "Invalid jobs-file line (missing '::'): ${line}" >&2
      exit 1
    fi
    local run_script="${line%%::*}"
    local job_args="${line#*::}"
    run_script="$(echo "${run_script}" | xargs)"
    job_args="$(echo "${job_args}" | sed 's/^ *//')"
    submit_job "${run_script}" "${job_args}"
  done < "${jobs_file}"
}

[[ $# -eq 0 ]] && { usage; exit 1; }

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--cache" ]]; then
  submit_cache
  exit 0
fi

if [[ "${1:-}" == "--jobs-file" ]]; then
  [[ $# -ge 2 ]] || { echo "--jobs-file requires a path" >&2; exit 1; }
  submit_jobs_file "$2"
  exit 0
fi

current_run_script="${RUN_SCRIPT_DEFAULT}"
current_args=()
seen_job=0

flush_job() {
  if [[ ${#current_args[@]} -eq 0 && "${seen_job}" -eq 0 ]]; then
    return 0
  fi
  local joined=""
  local arg
  for arg in "${current_args[@]}"; do
    if [[ -n "${joined}" ]]; then
      joined+=" "
    fi
    joined+="${arg}"
  done
  submit_job "${current_run_script}" "${joined}"
  current_run_script="${RUN_SCRIPT_DEFAULT}"
  current_args=()
  seen_job=1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-script)
      [[ $# -ge 2 ]] || { echo "--run-script requires a path" >&2; exit 1; }
      current_run_script="$2"
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 && "$1" != ":::s" ]]; do
        current_args+=("$1")
        shift
      done
      ;;
    :::)
      flush_job
      shift
      ;;
    *)
      current_args+=("$1")
      shift
      ;;
  esac
done

flush_job
