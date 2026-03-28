#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

RAW_ROOT="${RAW_ROOT:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/FactorCL_Raw}"
RAW_SOURCES_ROOT="${RAW_SOURCES_ROOT:-${RAW_ROOT}/raw_sources}"
PYTHON_BIN="${PYTHON_BIN:-/esat/smcdata/users/kkontras/Image_Dataset/no_backup/envs/synergy_new/bin/python}"

MOSI_RAW_URL="${MOSI_RAW_URL:-http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSI.zip}"
MOSEI_RAW_URL="${MOSEI_RAW_URL:-http://immortal.multicomp.cs.cmu.edu/raw_datasets/CMU_MOSEI.zip}"
UR_FUNNY_RAW_URL="${UR_FUNNY_RAW_URL:-https://www.dropbox.com/s/lg7kjx0kul3ansq/urfunny2_videos.zip?dl=1}"

usage() {
  cat <<EOF
Usage:
  ./run/factorcl/download_all_raw.sh

Downloads raw-source data for:
  - MOSI
  - MOSEI
  - UR-FUNNY
  - MUSTARD

Paths:
  RAW_ROOT=${RAW_ROOT}
  RAW_SOURCES_ROOT=${RAW_SOURCES_ROOT}

Env overrides:
  RAW_ROOT=/path/to/FactorCL_Raw
  RAW_SOURCES_ROOT=/path/to/raw_sources
  PYTHON_BIN=/path/to/python
  MOSI_RAW_URL=<alternate raw zip or mirror>
  MOSEI_RAW_URL=<alternate raw zip or mirror>
  UR_FUNNY_RAW_URL=<alternate raw zip or mirror>
  MUSTARD_LOCAL_REPO=/path/to/MUStARD
  MUSTARD_VIDEO_ZIP=/path/to/mmsd_raw_data.zip

Notes:
  - MOSI/MOSEI first try the legacy CMU raw zip URLs.
  - If those fail, the script falls back to audEERING via audb (audio + transcripts + labels + splits).
  - UR-FUNNY uses the public raw-video zip from the official repo README.
  - MUSTARD uses the existing repo downloader.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

mkdir -p "${RAW_ROOT}" "${RAW_SOURCES_ROOT}"

echo "[raw-download] RAW_ROOT=${RAW_ROOT}"
echo "[raw-download] RAW_SOURCES_ROOT=${RAW_SOURCES_ROOT}"

echo "[raw-download] step=mosi_mosei"
MOSI_RAW_URL="${MOSI_RAW_URL}" \
MOSEI_RAW_URL="${MOSEI_RAW_URL}" \
PYTHON_BIN="${PYTHON_BIN}" \
RAW_SOURCES_ROOT="${RAW_SOURCES_ROOT}" \
./run/factorcl/download_mosi_mosei_raw.sh

echo "[raw-download] step=ur_funny"
UR_FUNNY_ARCHIVE="${RAW_SOURCES_ROOT}/ur_funny/urfunny2_videos.zip"
mkdir -p "${RAW_SOURCES_ROOT}/ur_funny"
if [[ ! -f "${UR_FUNNY_ARCHIVE}" || ! -s "${UR_FUNNY_ARCHIVE}" ]]; then
  wget --tries=2 --timeout=30 -O "${UR_FUNNY_ARCHIVE}" "${UR_FUNNY_RAW_URL}"
fi
PYTHONPATH="${ROOT_DIR}/src" "${PYTHON_BIN}" -m synib.mydatasets.Factor_CL_Datasets.downloaders.download_ur_funny_raw \
  --output-root "${RAW_ROOT}" \
  --local-archive "${UR_FUNNY_ARCHIVE}"

echo "[raw-download] step=mustard"
MUSTARD_ARGS=(--output-root "${RAW_ROOT}")
if [[ -n "${MUSTARD_LOCAL_REPO:-}" ]]; then
  MUSTARD_ARGS+=(--local-repo "${MUSTARD_LOCAL_REPO}")
fi
if [[ -n "${MUSTARD_VIDEO_ZIP:-}" ]]; then
  MUSTARD_ARGS+=(--video-zip "${MUSTARD_VIDEO_ZIP}")
fi
./run/mustard/download.sh "${MUSTARD_ARGS[@]}"

echo "[raw-download] done"
echo "[raw-download] MOSI source root:   ${RAW_SOURCES_ROOT}/mosi"
echo "[raw-download] MOSEI source root:  ${RAW_SOURCES_ROOT}/mosei"
echo "[raw-download] UR-FUNNY raw root:  ${RAW_ROOT}/ur_funny_raw"
echo "[raw-download] MUSTARD raw root:   ${RAW_ROOT}/mustard_raw"
