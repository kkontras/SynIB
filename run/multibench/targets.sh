#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

print_multibench_targets() {
  cat <<'EOF'
Available MultiBench targets:
  mosi-vt
    - default config: run/configs/multibench/mosi/default.json
  urfunny-vt
    - default config: run/configs/multibench/urfunny/default.json
  mustard-vt
    - default config: run/configs/multibench/mustard/default.json

Usage:
  ./run/multibench/train.sh <target> [method-config.json] [extra train args]
  ./run/multibench/show.sh <target> [method-config.json] [extra show args]

Examples:
  ./run/multibench/train.sh mosi-vt    run/configs/multibench/mosi/synib_u.json --fold 0 --rmask random --l 0.1
  ./run/multibench/train.sh urfunny-vt run/configs/multibench/urfunny/synib.json --fold 0 --rmask learned --l 1
  ./run/multibench/show.sh  mustard-vt run/configs/multibench/mustard/reconboost.json --fold 0
EOF
}

resolve_multibench_target() {
  local target="${1:-}"

  MULTIBENCH_DEFAULT_CONFIG=""
  MULTIBENCH_DEFAULT_METHOD_CONFIG=""

  case "$target" in
    mosi-vt)
      MULTIBENCH_DEFAULT_CONFIG="run/configs/multibench/mosi/default.json"
      MULTIBENCH_DEFAULT_METHOD_CONFIG="$MULTIBENCH_DEFAULT_CONFIG"
      ;;
    urfunny-vt)
      MULTIBENCH_DEFAULT_CONFIG="run/configs/multibench/urfunny/default.json"
      MULTIBENCH_DEFAULT_METHOD_CONFIG="$MULTIBENCH_DEFAULT_CONFIG"
      ;;
    mustard-vt)
      MULTIBENCH_DEFAULT_CONFIG="run/configs/multibench/mustard/default.json"
      MULTIBENCH_DEFAULT_METHOD_CONFIG="$MULTIBENCH_DEFAULT_CONFIG"
      ;;
    *)
      return 1
      ;;
  esac

  return 0
}
