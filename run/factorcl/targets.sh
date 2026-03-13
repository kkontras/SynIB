#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

print_factorcl_targets() {
  cat <<'EOF'
Available FactorCL targets:
  mosi-vt
    - default config: run/configs/FactorCL/Mosi/default_config_mosi_VT.json
  mosi-vta
    - default config: run/configs/FactorCL/Mosi/default_config_mosi_VTA.json
  mosei-vt
    - default config: run/configs/FactorCL/Mosei/default_config_mosei_VT_syn.json
  urfunny-vt
    - default config: run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json
  mustard-vt
    - default config: run/configs/FactorCL/Mustard/default_config_mustard_VT.json

Usage:
  ./run/factorcl/train.sh <target> [method-config.json] [extra train args]
  ./run/factorcl/show.sh <target> [method-config.json] [extra show args]

Examples:
  ./run/factorcl/train.sh mosi-vt run/configs/FactorCL/Mosi/release/VT/unimodal_text.json --fold 0
  ./run/factorcl/train.sh mosei-vt run/configs/FactorCL/Mosei/syn/VT/synprom_RMask.json --fold 0 --rmask random --l 1.0 --pmin 0.2
  ./run/factorcl/show.sh urfunny-vt run/configs/FactorCL/URFunny/release/VT/ReconBoost.json --fold 0
EOF
}

resolve_factorcl_target() {
  local target="${1:-}"

  FACTORCL_DEFAULT_CONFIG=""
  FACTORCL_DEFAULT_METHOD_CONFIG=""

  case "$target" in
    mosi-vt)
      FACTORCL_DEFAULT_CONFIG="run/configs/FactorCL/Mosi/default_config_mosi_VT.json"
      FACTORCL_DEFAULT_METHOD_CONFIG="$FACTORCL_DEFAULT_CONFIG"
      ;;
    mosi-vta)
      FACTORCL_DEFAULT_CONFIG="run/configs/FactorCL/Mosi/default_config_mosi_VTA.json"
      FACTORCL_DEFAULT_METHOD_CONFIG="$FACTORCL_DEFAULT_CONFIG"
      ;;
    mosei-vt)
      FACTORCL_DEFAULT_CONFIG="run/configs/FactorCL/Mosei/default_config_mosei_VT_syn.json"
      FACTORCL_DEFAULT_METHOD_CONFIG="$FACTORCL_DEFAULT_CONFIG"
      ;;
    urfunny-vt)
      FACTORCL_DEFAULT_CONFIG="run/configs/FactorCL/URFunny/default_config_ur_funny_VT.json"
      FACTORCL_DEFAULT_METHOD_CONFIG="$FACTORCL_DEFAULT_CONFIG"
      ;;
    mustard-vt)
      FACTORCL_DEFAULT_CONFIG="run/configs/FactorCL/Mustard/default_config_mustard_VT.json"
      FACTORCL_DEFAULT_METHOD_CONFIG="$FACTORCL_DEFAULT_CONFIG"
      ;;
    *)
      return 1
      ;;
  esac

  return 0
}

