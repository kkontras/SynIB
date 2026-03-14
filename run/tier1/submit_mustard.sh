#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

GPU="${GPU:-0}"
PROJECT_ROOT="${PROJECT_ROOT:-/dodrio/scratch/projects/2026_029/kkontras/projects/SynIB}"
ENV_PATH="${ENV_PATH:-/dodrio/scratch/projects/2026_029/kkontras/envs/synergy_new}"
RUN_SCRIPT_DEFAULT="./run/mustard/train.sh"

usage() {
  cat <<'EOF'
Usage:
  ./run/tier1/submit_mustard.sh [job_spec ...]

Job spec formats:
  1. Pass one direct job:
     ./run/tier1/submit_mustard.sh --run-script ./run/mustard/train.sh -- run/configs/MUStARD/prompt_lora.json --fold 0 --lr 0.0005 --wd 0.01

  2. Pass multiple jobs separated by ::: 
     ./run/tier1/submit_mustard.sh \
       --run-script ./run/mustard/train.sh -- run/configs/MUStARD/prompt_video_lora.json --fold 0 --lr 0.0005 --wd 0.01 ::: \
       --run-script ./run/mustard/train.sh -- run/configs/MUStARD/prompt_text_lora.json --fold 0 --lr 0.0005 --wd 0.01

  3. Submit from a jobs file, one job per line:
     ./run/tier1/submit_mustard.sh --jobs-file run/tier1/mustard_jobs.txt

Jobs file format:
  ./run/mustard/train.sh :: run/configs/MUStARD/prompt_lora.json --fold 0 --lr 0.0005 --wd 0.01
  ./run/mustard/show.sh  :: run/configs/MUStARD/prompt_ens.json --fold 0 --lr 0.0005 --wd 0.01
EOF
}

submit_job() {
  local run_script="$1"
  local job_args="$2"
  echo "Submitting: ${run_script} ${job_args}"
  qsub ./run/tier1/jobfile_mustard_train.pbs \
    -v GPU="${GPU}",PROJECT_ROOT="${PROJECT_ROOT}",ENV_PATH="${ENV_PATH}",RUN_SCRIPT="${run_script}",JOB_ARGS="${job_args}"
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

if [[ $# -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--jobs-file" ]]; then
  [[ $# -ge 2 ]] || { echo "--jobs-file requires a path" >&2; exit 1; }
  module swap cluster/dodrio/gpu_rome_a100_80
  submit_jobs_file "$2"
  exit 0
fi

module swap cluster/dodrio/gpu_rome_a100_80

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
      while [[ $# -gt 0 && "$1" != ":::" ]]; do
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
