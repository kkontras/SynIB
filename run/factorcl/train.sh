#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "${ROOT_DIR}/run/factorcl/targets.sh"

if [[ $# -eq 0 ]]; then
  print_factorcl_targets
  exit 1
fi

case "$1" in
  -h|--help|--targets)
    print_factorcl_targets
    exit 0
    ;;
esac

TARGET="$1"
shift

if ! resolve_factorcl_target "$TARGET"; then
  echo "Unknown FactorCL target: $TARGET"
  echo
  print_factorcl_targets
  exit 1
fi

CONFIG="$FACTORCL_DEFAULT_METHOD_CONFIG"
if [[ $# -gt 0 ]]; then
  case "$1" in
    *.json|run/configs/*)
      CONFIG="$1"
      shift
      ;;
  esac
fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train \
  --config "$CONFIG" \
  --default_config "$FACTORCL_DEFAULT_CONFIG" \
  "$@"

