#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TARGET="${1:-all}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$TARGET" in
  -h|--help)
    cat <<'EOF'
Usage:
  ./run/factorcl/download.sh [all|mosi|mosei|urfunny|mustard] [extra downloader args]

Legacy:
  This wrapper downloads the old prepared FactorCL pickles.
  For the raw-data + cache workflow use:
    ./run/factorcl/download_and_build_cache.sh [target]

Examples:
  ./run/factorcl/download.sh all
  ./run/factorcl/download.sh mosi --local-file /path/to/mosi_data.pkl --symlink
  ./run/factorcl/download.sh mustard --output-root /data/factorcl
EOF
    exit 0
    ;;
  all)
    MODULE="synib.mydatasets.Factor_CL_Datasets.downloaders.download_all_factorcl_affect"
    ;;
  mosi)
    MODULE="synib.mydatasets.Factor_CL_Datasets.downloaders.download_mosi"
    ;;
  mosei)
    MODULE="synib.mydatasets.Factor_CL_Datasets.downloaders.download_mosei"
    ;;
  urfunny)
    MODULE="synib.mydatasets.Factor_CL_Datasets.downloaders.download_ur_funny"
    ;;
  mustard)
    MODULE="synib.mydatasets.Factor_CL_Datasets.downloaders.download_mustard"
    ;;
  *)
    echo "Unknown FactorCL download target: $TARGET"
    exit 1
    ;;
esac

PYTHONPATH="${ROOT_DIR}/src" python -m "$MODULE" "$@"
