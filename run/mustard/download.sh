#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage:
  ./run/mustard/download.sh [options]

Options:
  --output-root PATH   Root directory for downloaded data
                       (default: src/synib/mydatasets/MUStARD/prepared or $SYNIB_MUSTARD_DATA_ROOT)
  --force              Re-download and overwrite existing files
  --local-repo PATH    Path to an already-cloned MUStARD repo (skips git clone)
  --video-zip PATH     Path to an already-downloaded mmsd_raw_data.zip (skips HuggingFace download)
  --symlink            Symlink video files instead of copying
  --verbose            Enable debug logging

Note:
  The git repo contains only metadata (data/sarcasm_data.json).
  Video clips (~1.2 GB) are downloaded automatically from HuggingFace unless
  --video-zip is supplied with a pre-downloaded copy.

Examples:
  # Download everything automatically (clones repo + downloads videos):
  ./run/mustard/download.sh

  # Use a pre-downloaded zip (saves bandwidth if you already have it):
  ./run/mustard/download.sh --video-zip /path/to/mmsd_raw_data.zip

  # Use an already-cloned local repo + pre-downloaded zip:
  ./run/mustard/download.sh --local-repo /path/to/MUStARD --video-zip /path/to/mmsd_raw_data.zip

  # Use a local copy with utterances_final/ already present, symlink videos:
  ./run/mustard/download.sh --local-repo /path/to/MUStARD --symlink --output-root /data/mustard
EOF
    exit 0
    ;;
esac

PYTHONPATH="${ROOT_DIR}/src" python -m synib.mydatasets.MUStARD.download_mustard_raw "$@"
