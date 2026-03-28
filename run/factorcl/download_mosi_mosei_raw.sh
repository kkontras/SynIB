#!/usr/bin/env bash
set -euo pipefail

ROOT="${RAW_SOURCES_ROOT:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/FactorCL_Raw/raw_sources}"
MOSI_DIR="${ROOT}/mosi"
MOSEI_DIR="${ROOT}/mosei"
PYTHON_BIN="${PYTHON_BIN:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new/bin/python}"
MOSI_RAW_URL="${MOSI_RAW_URL:-http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip}"
MOSEI_RAW_URL="${MOSEI_RAW_URL:-http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip}"

mkdir -p "${MOSI_DIR}" "${MOSEI_DIR}"

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

try_cmu_raw() {
  local dataset="$1"
  local url="$2"
  local out="$3"
  if have_valid_zip "${out}"; then
    echo "[download] reusing ${out}"
    return 0
  fi
  if fetch_zip "${url}" "${out}"; then
    return 0
  fi
  rm -f "${out}"
  return 1
}

echo "[paths] MOSI  -> ${MOSI_DIR}"
echo "[paths] MOSEI -> ${MOSEI_DIR}"

MOSI_ZIP="${MOSI_DIR}/CMU_MOSI.zip"
MOSEI_ZIP="${MOSEI_DIR}/CMU_MOSEI.zip"

MOSI_OK=0
MOSEI_OK=0

if try_cmu_raw "mosi" "${MOSI_RAW_URL}" "${MOSI_ZIP}"; then
  MOSI_OK=1
fi

if try_cmu_raw "mosei" "${MOSEI_RAW_URL}" "${MOSEI_ZIP}"; then
  MOSEI_OK=1
fi

if [[ "${MOSI_OK}" == "1" ]]; then
  echo "[extract] ${MOSI_ZIP}"
  unzip -o "${MOSI_ZIP}" -d "${MOSI_DIR}"
fi

if [[ "${MOSEI_OK}" == "1" ]]; then
  echo "[extract] ${MOSEI_ZIP}"
  unzip -o "${MOSEI_ZIP}" -d "${MOSEI_DIR}"
fi

if [[ "${MOSI_OK}" == "0" || "${MOSEI_OK}" == "0" ]]; then
  echo "[fallback] installing audb and downloading audEERING releases"
  "${PYTHON_BIN}" -m pip install -U audb
  ROOT_FOR_AUDB="${ROOT}" "${PYTHON_BIN}" - <<'PY'
import audb
import os

root = os.environ["ROOT_FOR_AUDB"]

audb.load(
    "cmu-mosi",
    version="1.1.1",
    full_path=True,
    cache_root=f"{root}/mosi/audb_cache",
    verbose=True,
)

audb.load(
    "cmu-mosei",
    version="1.2.4",
    full_path=True,
    cache_root=f"{root}/mosei/audb_cache",
    verbose=True,
)
PY
fi

echo "[done] raw acquisition attempt finished"
echo "[note] audEERING fallback provides audio + transcripts + labels + splits."
echo "[note] If the legacy CMU zips were reachable, they were extracted above."
