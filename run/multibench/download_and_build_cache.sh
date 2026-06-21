#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  TARGET="all"
else
  TARGET="${1:-all}"
  if [[ $# -gt 0 ]]; then
    shift
  fi
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/src/synib/mydatasets/Factor_CL_Datasets/raw}"
CACHE_ROOT="${CACHE_ROOT:-${ROOT_DIR}/artifacts/cache/multibench}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-VL-2B-Instruct}"
GPU="${GPU:-0}"
FOLDS="${FOLDS:-0 1 2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
FORCE_DOWNLOAD=0
FORCE_CACHE=0
SKIP_DOWNLOAD=0
SKIP_CACHE=0
VERBOSE=0
LOCAL_FILES_ONLY=0

usage() {
  cat <<'EOF'
Usage:
  ./run/multibench/download_and_build_cache.sh [all|mosi|urfunny|mustard] [options]

Options:
  --output-root PATH      Raw dataset root (default: src/synib/mydatasets/Factor_CL_Datasets/raw)
  --cache-root PATH       Cache root (default: artifacts/cache/multibench)
  --model-name NAME       VLM model name (default: Qwen/Qwen3-VL-2B-Instruct)
  --gpu ID                CUDA device id (default: 0)
  --folds "0 1 2"         Space-separated fold list for fold manifests
  --batch-size N          Samples per processor/model batch (default: 4)
  --force-download        Re-materialize raw roots
  --force-cache           Rebuild cache artifacts
  --skip-download         Skip raw download/materialization
  --skip-cache            Skip cache building
  --local-files-only      Pass local_files_only to the cache builder
  --verbose               Echo commands
  -h, --help              Show this help

Local-source env vars:
  MOSI_SOURCE_URL, MOSI_LOCAL_ROOT, MOSI_LOCAL_ARCHIVE, MOSI_METADATA_JSONL, MOSI_FOLDS_JSON, MOSI_MEDIA_ROOT
  UR_FUNNY_SOURCE_URL, UR_FUNNY_LOCAL_ROOT, UR_FUNNY_LOCAL_ARCHIVE, UR_FUNNY_METADATA_JSONL, UR_FUNNY_FOLDS_JSON, UR_FUNNY_MEDIA_ROOT
  MUSTARD_LOCAL_REPO, MUSTARD_VIDEO_ZIP
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --folds) FOLDS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --force-download) FORCE_DOWNLOAD=1; shift ;;
    --force-cache) FORCE_CACHE=1; shift ;;
    --skip-download) SKIP_DOWNLOAD=1; shift ;;
    --skip-cache) SKIP_CACHE=1; shift ;;
    --local-files-only) LOCAL_FILES_ONLY=1; shift ;;
    --verbose) VERBOSE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ "${VERBOSE}" == "1" ]]; then
  set -x
fi

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}"

case "${TARGET}" in
  all) DATASETS=(mosi ur_funny mustard) ;;
  mosi) DATASETS=(mosi) ;;
  urfunny) DATASETS=(ur_funny) ;;
  mustard) DATASETS=(mustard) ;;
  *) echo "Unknown target: ${TARGET}" >&2; usage; exit 1 ;;
esac

download_raw() {
  local dataset="$1"
  if [[ "${SKIP_DOWNLOAD}" == "1" ]]; then
    return
  fi

  case "${dataset}" in
    mosi)
      local args=()
      [[ "${FORCE_DOWNLOAD}" == "1" ]] && args+=(--force)
      [[ -n "${MOSI_SOURCE_URL:-}" ]] && args+=(--source-url "${MOSI_SOURCE_URL}")
      [[ -n "${MOSI_LOCAL_ROOT:-}" ]] && args+=(--local-root "${MOSI_LOCAL_ROOT}")
      [[ -n "${MOSI_LOCAL_ARCHIVE:-}" ]] && args+=(--local-archive "${MOSI_LOCAL_ARCHIVE}")
      [[ -n "${MOSI_METADATA_JSONL:-}" ]] && args+=(--metadata-jsonl "${MOSI_METADATA_JSONL}")
      [[ -n "${MOSI_FOLDS_JSON:-}" ]] && args+=(--folds-json "${MOSI_FOLDS_JSON}")
      [[ -n "${MOSI_MEDIA_ROOT:-}" ]] && args+=(--media-root "${MOSI_MEDIA_ROOT}")
      PYTHONPATH="${ROOT_DIR}/src" python -m synib.mydatasets.Factor_CL_Datasets.downloaders.download_mosi_raw \
        --output-root "${OUTPUT_ROOT}" "${args[@]}"
      ;;
    ur_funny)
      local args=()
      [[ "${FORCE_DOWNLOAD}" == "1" ]] && args+=(--force)
      [[ -n "${UR_FUNNY_SOURCE_URL:-}" ]] && args+=(--source-url "${UR_FUNNY_SOURCE_URL}")
      [[ -n "${UR_FUNNY_LOCAL_ROOT:-}" ]] && args+=(--local-root "${UR_FUNNY_LOCAL_ROOT}")
      [[ -n "${UR_FUNNY_LOCAL_ARCHIVE:-}" ]] && args+=(--local-archive "${UR_FUNNY_LOCAL_ARCHIVE}")
      [[ -n "${UR_FUNNY_METADATA_JSONL:-}" ]] && args+=(--metadata-jsonl "${UR_FUNNY_METADATA_JSONL}")
      [[ -n "${UR_FUNNY_FOLDS_JSON:-}" ]] && args+=(--folds-json "${UR_FUNNY_FOLDS_JSON}")
      [[ -n "${UR_FUNNY_MEDIA_ROOT:-}" ]] && args+=(--media-root "${UR_FUNNY_MEDIA_ROOT}")
      PYTHONPATH="${ROOT_DIR}/src" python -m synib.mydatasets.Factor_CL_Datasets.downloaders.download_ur_funny_raw \
        --output-root "${OUTPUT_ROOT}" "${args[@]}"
      ;;
    mustard)
      local args=()
      [[ "${FORCE_DOWNLOAD}" == "1" ]] && args+=(--force)
      [[ -n "${MUSTARD_LOCAL_REPO:-}" ]] && args+=(--local-repo "${MUSTARD_LOCAL_REPO}")
      [[ -n "${MUSTARD_VIDEO_ZIP:-}" ]] && args+=(--video-zip "${MUSTARD_VIDEO_ZIP}")
      ./run/multibench/download_mustard.sh --output-root "${OUTPUT_ROOT}" "${args[@]}"
      ;;
  esac
}

build_cache() {
  local dataset="$1"
  if [[ "${SKIP_CACHE}" == "1" ]]; then
    return
  fi

  local raw_root="${OUTPUT_ROOT}/${dataset}_raw"
  local cache_dir="${CACHE_ROOT}/${dataset}"
  mkdir -p "${cache_dir}"

  if [[ "${FORCE_CACHE}" == "1" ]]; then
    rm -rf "${cache_dir}"
    mkdir -p "${cache_dir}"
  fi

  local args=(
    --dataset "${dataset}"
    --raw_root "${raw_root}"
    --out_dir "${cache_dir}"
    --model_name "${MODEL_NAME}"
    --folds ${FOLDS}
    --batch_size "${BATCH_SIZE}"
    --device "cuda:${GPU}"
  )
  [[ "${LOCAL_FILES_ONLY}" == "1" ]] && args+=(--local_files_only)

  CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH="${ROOT_DIR}/src" python -m synib.mydatasets.Factor_CL_Datasets.FactorCL_Raw_CodeBook "${args[@]}"
}

for dataset in "${DATASETS[@]}"; do
  echo "[multibench-cache] dataset=${dataset} output_root=${OUTPUT_ROOT} cache_root=${CACHE_ROOT}"
  download_raw "${dataset}"
  build_cache "${dataset}"
done

echo "[multibench-cache] done"
