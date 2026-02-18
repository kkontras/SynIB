#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${1:-all}"

list_one() {
  local d="$1"
  echo "== $d =="
  find "$ROOT_DIR/run/configs/$d" -type f | sort
  echo
}

case "$DATASET" in
  crema_d) list_one "CREMA_D" ;;
  scienceqa) list_one "ScienceQA" ;;
  esnli) list_one "ESNLI" ;;
  xor) list_one "xor" ;;
  all)
    list_one "CREMA_D"
    list_one "ScienceQA"
    list_one "ESNLI"
    list_one "xor"
    ;;
  *)
    echo "Unknown dataset: $DATASET"
    echo "Use one of: crema_d, scienceqa, esnli, xor, all"
    exit 1
    ;;
esac
