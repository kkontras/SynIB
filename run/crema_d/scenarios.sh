#!/usr/bin/env bash

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_CONFIG_PATH="run/configs/CREMA_D/default_config_cremad_res.json"
DEFAULT_TRAIN_SHOW_CONFIG="run/configs/CREMA_D/release/res/joint_training.json"
RMASK_CONFIG_PATH="run/configs/CREMA_D/synergy/dec/synprom_RMask.json"

print_cremad_scenarios() {
  cat <<'EOF'
Available CREMA-D scenarios:
  default
    - run/configs/CREMA_D/release/res/joint_training.json
  rmask-random-l0.5-pmin0.10
    - --perturb random --perturb_fill random --l 0.5 --perturb_pmin 0.10
  rmask-random-l1.0-pmin0.20
    - --perturb random --perturb_fill random --l 1.0 --perturb_pmin 0.20
  rmask-random-l2.0-pmin0.30
    - --perturb random --perturb_fill random --l 2.0 --perturb_pmin 0.30
  rmask-learned-l0.5-lsparse0.001
    - --perturb learned --perturb_fill learned --l 0.5 --perturb_lsparse 0.001
  rmask-learned-l1.0-lsparse0.010
    - --perturb learned --perturb_fill learned --l 1.0 --perturb_lsparse 0.010
  rmask-learned-l2.0-lsparse0.050
    - --perturb learned --perturb_fill learned --l 2.0 --perturb_lsparse 0.050
EOF
}

resolve_cremad_target() {
  local target="${1:-default}"

  CREMAD_CONFIG=""
  CREMAD_DEFAULT_CONFIG="$DEFAULT_CONFIG_PATH"
  CREMAD_SCENARIO_ARGS=()
  CREMAD_SCENARIO_NAME="$target"

  case "$target" in
    default)
      CREMAD_CONFIG="$DEFAULT_TRAIN_SHOW_CONFIG"
      ;;
    rmask-random-l0.5-pmin0.10)
      CREMAD_CONFIG="$RMASK_CONFIG_PATH"
      CREMAD_SCENARIO_ARGS=(--perturb random --perturb_fill random --l 0.5 --perturb_pmin 0.10)
      ;;
    rmask-random-l1.0-pmin0.20)
      CREMAD_CONFIG="$RMASK_CONFIG_PATH"
      CREMAD_SCENARIO_ARGS=(--perturb random --perturb_fill random --l 1.0 --perturb_pmin 0.20)
      ;;
    rmask-random-l2.0-pmin0.30)
      CREMAD_CONFIG="$RMASK_CONFIG_PATH"
      CREMAD_SCENARIO_ARGS=(--perturb random --perturb_fill random --l 2.0 --perturb_pmin 0.30)
      ;;
    rmask-learned-l0.5-lsparse0.001)
      CREMAD_CONFIG="$RMASK_CONFIG_PATH"
      CREMAD_SCENARIO_ARGS=(--perturb learned --perturb_fill learned --l 0.5 --perturb_lsparse 0.001)
      ;;
    rmask-learned-l1.0-lsparse0.010)
      CREMAD_CONFIG="$RMASK_CONFIG_PATH"
      CREMAD_SCENARIO_ARGS=(--perturb learned --perturb_fill learned --l 1.0 --perturb_lsparse 0.010)
      ;;
    rmask-learned-l2.0-lsparse0.050)
      CREMAD_CONFIG="$RMASK_CONFIG_PATH"
      CREMAD_SCENARIO_ARGS=(--perturb learned --perturb_fill learned --l 2.0 --perturb_lsparse 0.050)
      ;;
    *.json|run/configs/*)
      CREMAD_CONFIG="$target"
      ;;
    *)
      return 1
      ;;
  esac

  return 0
}
