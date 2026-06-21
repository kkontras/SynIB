#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "${ROOT_DIR}/run/multibench/targets.sh"

if [[ $# -eq 0 ]]; then
  print_multibench_targets
  exit 1
fi

case "$1" in
  -h|--help|--targets)
    print_multibench_targets
    exit 0
    ;;
esac

TARGET="$1"
shift

if ! resolve_multibench_target "$TARGET"; then
  echo "Unknown MultiBench target: $TARGET"
  echo
  print_multibench_targets
  exit 1
fi

CONFIG="$MULTIBENCH_DEFAULT_METHOD_CONFIG"
if [[ $# -gt 0 ]]; then
  case "$1" in
    *.json|run/configs/*)
      CONFIG="$1"
      shift
      ;;
  esac
fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.show \
  --config "$CONFIG" \
  --default_config "$MULTIBENCH_DEFAULT_CONFIG" \
  "$@"

