#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${CONDA_PREFIX:-}" ]; then
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate /esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy
fi

PYTHON_BIN="${PYTHON:-$(command -v python)}"

usage() {
  cat <<EOF
Usage: $0 <dataset|all> [--job JOB_IDX]

Datasets:
  mustard
  urfunny
  mosi
  mosei
  all

Options:
  --job  JOB_IDX  Run only this job index from the dataset args file (0-based)

Examples:
  $0 mosei
  $0 mustard --job 17
  $0 all
EOF
  exit 1
}

[[ $# -eq 0 ]] && usage

TARGET="$1"
shift
JOB_IDX=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)
      JOB_IDX="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      ;;
  esac
done

dataset_args_file() {
  case "$1" in
    mustard) echo "${ROOT_DIR}/run/condor/mustard_synibu.args" ;;
    urfunny) echo "${ROOT_DIR}/run/condor/urfunny_synibu.args" ;;
    mosi) echo "${ROOT_DIR}/run/condor/mosi_synibu.args" ;;
    mosei) echo "${ROOT_DIR}/run/condor/mosei_synibu.args" ;;
    *)
      return 1
      ;;
  esac
}

run_show() {
  local dataset="$1"
  local args_file="$2"
  local line="$3"
  local idx="$4"

  [[ -z "$line" ]] && return 0

  local show_args
  show_args="$(printf '%s\n' "$line" | sed 's/ --no_model_save//g')"

  echo "--------------------------------------"
  echo "Dataset: $dataset | Job $idx"
  echo "Args: $show_args"
  echo "--------------------------------------"
  PYTHONPATH="${ROOT_DIR}/src" "$PYTHON_BIN" -m synib.entrypoints.show $show_args
}

run_dataset() {
  local dataset="$1"
  local args_file

  args_file="$(dataset_args_file "$dataset")" || {
    echo "Unknown dataset: $dataset"
    usage
  }

  if [[ -n "$JOB_IDX" ]]; then
    local line
    line="$(sed -n "$((JOB_IDX + 1))p" "$args_file")"
    [[ -z "$line" ]] && {
      echo "Job index $JOB_IDX out of range for $dataset."
      exit 1
    }
    run_show "$dataset" "$args_file" "$line" "$JOB_IDX"
    return 0
  fi

  local idx=0
  while IFS= read -r line; do
    run_show "$dataset" "$args_file" "$line" "$idx"
    idx=$((idx + 1))
  done < "$args_file"
}

case "$TARGET" in
  all)
    [[ -n "$JOB_IDX" ]] && {
      echo "--job can only be used with a single dataset."
      exit 1
    }
    for dataset in mustard urfunny mosi mosei; do
      run_dataset "$dataset"
    done
    ;;
  mustard|urfunny|mosi|mosei)
    run_dataset "$TARGET"
    ;;
  *)
    echo "Unknown dataset: $TARGET"
    usage
    ;;
esac
