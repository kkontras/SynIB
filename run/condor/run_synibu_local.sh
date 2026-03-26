#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<EOF
Usage: $0 <dataset> [--gpu GPU_ID] [--job JOB_IDX]

Datasets: mustard  urfunny  mosi  mosei

Options:
  --gpu  GPU_ID    CUDA device to use (default: 0)
  --job  JOB_IDX  Run only this job index from the args file (0-based)
                  If omitted, runs all 162 jobs sequentially

Examples:
  $0 mustard
  $0 mustard --gpu 1
  $0 mosei --gpu 0 --job 42
EOF
  exit 1
}

[[ $# -eq 0 ]] && usage

DATASET="$1"; shift
GPU=0
JOB_IDX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)  GPU="$2";     shift 2 ;;
    --job)  JOB_IDX="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; usage ;;
  esac
done

case "$DATASET" in
  mustard) ARGS_FILE="${ROOT_DIR}/run/condor/mustard_synibu.args" ;;
  urfunny) ARGS_FILE="${ROOT_DIR}/run/condor/urfunny_synibu.args" ;;
  mosi)    ARGS_FILE="${ROOT_DIR}/run/condor/mosi_synibu.args"    ;;
  mosei)   ARGS_FILE="${ROOT_DIR}/run/condor/mosei_synibu.args"   ;;
  *) echo "Unknown dataset: $DATASET"; usage ;;
esac

export CUDA_VISIBLE_DEVICES="$GPU"

run_job() {
  local args="$1"
  local idx="$2"
  echo "--------------------------------------"
  echo "Job $idx | GPU $GPU"
  echo "Args: $args"
  echo "--------------------------------------"
  PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train $args
}

if [[ -n "$JOB_IDX" ]]; then
  LINE=$(sed -n "$((JOB_IDX + 1))p" "$ARGS_FILE")
  [[ -z "$LINE" ]] && { echo "Job index $JOB_IDX out of range."; exit 1; }
  run_job "$LINE" "$JOB_IDX"
else
  idx=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && { idx=$((idx + 1)); continue; }
    run_job "$line" "$idx"
    idx=$((idx + 1))
  done < "$ARGS_FILE"
  echo "All jobs done."
fi
