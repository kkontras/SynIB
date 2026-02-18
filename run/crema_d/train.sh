#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

source "${ROOT_DIR}/run/crema_d/scenarios.sh"

TARGET="default"
if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help|--scenarios)
      print_cremad_scenarios
      echo
      echo "Usage: ./run/crema_d/train.sh [scenario|config.json] [extra train args]"
      exit 0
      ;;
    --*)
      TARGET="default"
      ;;
    *)
      TARGET="$1"
      shift
      ;;
  esac
fi

if ! resolve_cremad_target "$TARGET"; then
  echo "Unknown scenario: $TARGET"
  echo
  print_cremad_scenarios
  exit 1
fi

PYTHONPATH="${ROOT_DIR}/src" python -m synib.entrypoints.train \
  --config "$CREMAD_CONFIG" \
  --default_config "$CREMAD_DEFAULT_CONFIG" \
  "${CREMAD_SCENARIO_ARGS[@]}" \
  "$@"
