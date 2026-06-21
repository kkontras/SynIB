#!/usr/bin/env bash
set -euo pipefail

ROOT="${RAW_SOURCES_ROOT:-./data/FactorCL_Raw/raw_sources}"
MOSI_DIR="${ROOT}/mosi"
PYTHON_BIN="${PYTHON_BIN:-python}"
MOSI_RAW_URL="${MOSI_RAW_URL:-http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip}"

mkdir -p "${MOSI_DIR}"

fetch_zip() {
  local url="$1"
  local out="$2"
  echo "[download] trying ${url}"
  wget --tries=2 --timeout=30 -O "${out}" "${url}"
}

have_valid_zip() {
  local path="$1"
  [[ -f "${path}" && -s "${path}" ]]
}

echo "[paths] MOSI -> ${MOSI_DIR}"
MOSI_ZIP="${MOSI_DIR}/CMU_MOSI.zip"
MOSI_OK=0

if have_valid_zip "${MOSI_ZIP}"; then
  echo "[download] reusing ${MOSI_ZIP}"; MOSI_OK=1
elif fetch_zip "${MOSI_RAW_URL}" "${MOSI_ZIP}"; then
  MOSI_OK=1
else
  rm -f "${MOSI_ZIP}"
fi

if [[ "${MOSI_OK}" == "1" ]]; then
  echo "[extract] ${MOSI_ZIP}"
  unzip -o "${MOSI_ZIP}" -d "${MOSI_DIR}"
else
  echo "[fallback] installing audb and downloading the audEERING release"
  "${PYTHON_BIN}" -m pip install -U audb
  ROOT_FOR_AUDB="${ROOT}" "${PYTHON_BIN}" - <<'PY'
import audb, os
root = os.environ["ROOT_FOR_AUDB"]
audb.load("cmu-mosi", version="1.1.1", full_path=True,
          cache_root=f"{root}/mosi/audb_cache", verbose=True)
PY
fi

echo "[done] MOSI raw acquisition finished"
