#!/usr/bin/env bash
# Download TQA v2 dataset from AllenAI
# Usage: ./run/tqa/download.sh [--output-root PATH] [--force]
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${SYNIB_TQA_DATA_ROOT:-${ROOT_DIR}/data/TQA}"
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"; shift 2 ;;
    --force)
      FORCE=1; shift ;;
    -h|--help)
      cat <<'EOF'
Usage:
  ./run/tqa/download.sh [options]

Options:
  --output-root PATH   Root directory for downloaded data
                       (default: data/TQA or $SYNIB_TQA_DATA_ROOT)
  --force              Re-download and overwrite existing files

Notes:
  Downloads TQA v2 from Allen AI (https://prior.allenai.org/projects/tqa).
  The archive is extracted as: <output-root>/{train,val,test}/

  If the automatic download fails (e.g. URL changes), manually download
  tqa_v2.zip from the AllenAI project page and place it at:
    <output-root>/tqa_v2.zip
  Then re-run this script.

Example:
  ./run/tqa/download.sh --output-root /data/TQA
EOF
      exit 0 ;;
    *)
      echo "[download] Unknown argument: $1" >&2; exit 1 ;;
  esac
done

TQA_URL="https://s3.amazonaws.com/ai2-vision-textbook-dataset/dataset_releases/tqa/tqa_train_val_test.zip"
ZIP_PATH="${OUTPUT_ROOT}/tqa_train_val_test.zip"

mkdir -p "${OUTPUT_ROOT}"

if [[ -d "${OUTPUT_ROOT}/train" ]] && [[ "${FORCE}" -eq 0 ]]; then
  echo "[download] TQA already present at ${OUTPUT_ROOT}. Use --force to re-download."
  exit 0
fi

if [[ ! -f "${ZIP_PATH}" ]] || [[ "${FORCE}" -eq 1 ]]; then
  echo "[download] Downloading TQA (~1.6 GB) from ${TQA_URL} ..."
  wget -c -O "${ZIP_PATH}" "${TQA_URL}" || {
    echo "[download] ERROR: wget failed. Please manually download tqa_train_val_test.zip from:"
    echo "  https://prior.allenai.org/projects/tqa"
    echo "and place it at: ${ZIP_PATH}"
    exit 1
  }
fi

echo "[download] Extracting ${ZIP_PATH} -> ${OUTPUT_ROOT} ..."
unzip -q "${ZIP_PATH}" -d "${OUTPUT_ROOT}"

# The zip extracts into tqa_train_val_test/; move contents up
if [[ -d "${OUTPUT_ROOT}/tqa_train_val_test" ]]; then
  mv "${OUTPUT_ROOT}/tqa_train_val_test/"* "${OUTPUT_ROOT}/"
  rmdir "${OUTPUT_ROOT}/tqa_train_val_test"
fi

echo "[download] Done. TQA data at: ${OUTPUT_ROOT}"
ls "${OUTPUT_ROOT}"
