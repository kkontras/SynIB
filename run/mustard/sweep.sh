#!/usr/bin/env bash
# run/mustard/sweep.sh — MUStARD full hyperparameter sweep
#
# Runs all methods × hyperparameter combinations × folds sequentially.
# Skips any run whose checkpoint already exists in SAVE_BASE_DIR.
#
# Usage:
#   ./run/mustard/sweep.sh [PHASE]
#
#   PHASE (optional): run only one phase
#     video       – unimodal video LoRA
#     text        – unimodal text LoRA
#     combined    – combined (video+text) LoRA baseline
#     mcr         – MCR
#     dnr         – DnR
#     mmpareto    – MMPareto
#     reconboost  – ReconBoost
#     all         – everything (default)
#
# Examples:
#   ./run/mustard/sweep.sh              # run everything
#   ./run/mustard/sweep.sh video        # unimodal video only
#   ./run/mustard/sweep.sh dnr          # DnR sweep only
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

PHASE="${1:-all}"

TRAIN="./run/mustard/train.sh"
SAVE_BASE_DIR="/esat/smcdata/users/kkontras/Image_Dataset/no_backup/data/2025_data/synergy/MUSTARD"

# ── Shared sweep axes ────────────────────────────────────────────────────────
FOLDS=(0 1 2)
LRS=(0.0001 0.0005 0.001)
WDS=(0.001 0.01)

# ── Helper: skip if checkpoint exists ───────────────────────────────────────
# The training script names the checkpoint as save_dir (with the suffix built
# from CLI flags), e.g. MUStARD_video_lora_fold0_lr0.0001_wd0.001.pth.tar
run_if_missing() {
    local ckpt_path="$1"; shift
    if [[ -f "${SAVE_BASE_DIR}/${ckpt_path}" ]]; then
        echo "[SKIP] ${ckpt_path} already exists"
        return 0
    fi
    echo "[RUN ] $*"
    "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Unimodal Video LoRA
# ─────────────────────────────────────────────────────────────────────────────
phase_video() {
    echo "=== Phase: Unimodal Video LoRA ==="
    local CFG="run/configs/MUStARD/prompt_video_lora.json"
    for fold in "${FOLDS[@]}"; do
        for lr in "${LRS[@]}"; do
            for wd in "${WDS[@]}"; do
                local ckpt="MUStARD_video_lora_fold${fold}_lr${lr}_wd${wd}.pth.tar"
                run_if_missing "$ckpt" \
                    "$TRAIN" "$CFG" \
                    --fold "$fold" --lr "$lr" --wd "$wd" \
                    --validate_with accuracy
            done
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Unimodal Text LoRA
# ─────────────────────────────────────────────────────────────────────────────
phase_text() {
    echo "=== Phase: Unimodal Text LoRA ==="
    local CFG="run/configs/MUStARD/prompt_text_lora.json"
    for fold in "${FOLDS[@]}"; do
        for lr in "${LRS[@]}"; do
            for wd in "${WDS[@]}"; do
                local ckpt="MUStARD_text_lora_fold${fold}_lr${lr}_wd${wd}.pth.tar"
                run_if_missing "$ckpt" \
                    "$TRAIN" "$CFG" \
                    --fold "$fold" --lr "$lr" --wd "$wd" \
                    --validate_with accuracy
            done
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Combined LoRA baseline
# ─────────────────────────────────────────────────────────────────────────────
phase_combined() {
    echo "=== Phase: Combined LoRA baseline ==="
    local CFG="run/configs/MUStARD/prompt_lora.json"
    for fold in "${FOLDS[@]}"; do
        for lr in "${LRS[@]}"; do
            for wd in "${WDS[@]}"; do
                local ckpt="MUStARD_combined_lora_fold${fold}_lr${lr}_wd${wd}.pth.tar"
                run_if_missing "$ckpt" \
                    "$TRAIN" "$CFG" \
                    --fold "$fold" --lr "$lr" --wd "$wd" \
                    --validate_with accuracy
            done
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — MCR
# ─────────────────────────────────────────────────────────────────────────────
phase_mcr() {
    echo "=== Phase: MCR ==="
    local CFG="run/configs/MUStARD/prompt_mcr.json"
    local NUM_SAMPLES=(16 32 64)
    local LS=(0.1 0.5 1.0)
    # Fix best lr/wd from combined sweep (adjust once you know the best)
    local lr=0.0005
    local wd=0.001
    for fold in "${FOLDS[@]}"; do
        for ns in "${NUM_SAMPLES[@]}"; do
            for l in "${LS[@]}"; do
                local ckpt="MUStARD_mcr_fold${fold}_l${l}_numsamples${ns}_vldaccuracy_lr${lr}_wd${wd}.pth.tar"
                run_if_missing "$ckpt" \
                    "$TRAIN" "$CFG" \
                    --fold "$fold" --lr "$lr" --wd "$wd" \
                    --num_samples "$ns" --l "$l" \
                    --validate_with accuracy
            done
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — DnR
# ─────────────────────────────────────────────────────────────────────────────
phase_dnr() {
    echo "=== Phase: DnR ==="
    local CFG="run/configs/MUStARD/prompt_dnr.json"
    local ALPHAS=(0.5 1.0 2.0 3.0)
    local KMEPOCHS=(1 3 5 10)
    local lr=0.0005
    local wd=0.001
    for fold in "${FOLDS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            for kmepoch in "${KMEPOCHS[@]}"; do
                local ckpt="MUStARD_dnr_fold${fold}_alpha${alpha}_kmepoch${kmepoch}_vldaccuracy_lr${lr}_wd${wd}.pth.tar"
                run_if_missing "$ckpt" \
                    "$TRAIN" "$CFG" \
                    --fold "$fold" --lr "$lr" --wd "$wd" \
                    --alpha "$alpha" --kmepoch "$kmepoch" \
                    --validate_with accuracy
            done
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 6 — MMPareto
# ─────────────────────────────────────────────────────────────────────────────
phase_mmpareto() {
    echo "=== Phase: MMPareto ==="
    local CFG="run/configs/MUStARD/prompt_mmpareto.json"
    local ALPHAS=(0.5 0.8 1.0 1.5)
    local lr=0.0005
    local wd=0.001
    for fold in "${FOLDS[@]}"; do
        for alpha in "${ALPHAS[@]}"; do
            local ckpt="MUStARD_mmpareto_fold${fold}_alpha${alpha}_vldaccuracy_lr${lr}_wd${wd}.pth.tar"
            run_if_missing "$ckpt" \
                "$TRAIN" "$CFG" \
                --fold "$fold" --lr "$lr" --wd "$wd" \
                --alpha "$alpha" \
                --validate_with accuracy --tdqm_disable
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 7 — ReconBoost
# ─────────────────────────────────────────────────────────────────────────────
phase_reconboost() {
    echo "=== Phase: ReconBoost ==="
    local CFG="run/configs/MUStARD/prompt_reconboost.json"
    local ENS_STAGES=(1 2 4)
    local WEIGHTS1=(1.0 5.0 10.0)
    local lr=0.0005
    local wd=0.001
    for fold in "${FOLDS[@]}"; do
        for ens in "${ENS_STAGES[@]}"; do
            for w1 in "${WEIGHTS1[@]}"; do
                local ckpt="MUStARD_reconboost_fold${fold}_w1${w1}_ensstage${ens}_vldaccuracy_lr${lr}_wd${wd}.pth.tar"
                run_if_missing "$ckpt" \
                    "$TRAIN" "$CFG" \
                    --fold "$fold" --lr "$lr" --wd "$wd" \
                    --recon_ensemblestages "$ens" --recon_weight1 "$w1" \
                    --validate_with accuracy
            done
        done
    done
}

# ─────────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────────
case "$PHASE" in
    video)      phase_video ;;
    text)       phase_text ;;
    combined)   phase_combined ;;
    mcr)        phase_mcr ;;
    dnr)        phase_dnr ;;
    mmpareto)   phase_mmpareto ;;
    reconboost) phase_reconboost ;;
    all)
        phase_video
        phase_text
        phase_combined
        phase_mcr
        phase_dnr
        phase_mmpareto
        phase_reconboost
        ;;
    *)
        echo "Unknown phase: $PHASE"
        echo "Valid phases: video text combined mcr dnr mmpareto reconboost all"
        exit 1
        ;;
esac

echo "=== Sweep complete ==="
