#!/bin/bash
# Unified training script for all phases
# Usage: ./train_phase.sh <phase> [epochs] [batch_size] [extra_args...]
#   phase: 0, A, or B
#   extra_args: Additional arguments to pass to main.py (e.g., --use_discriminative_lr)

set -e

# Change to project root directory
cd "$(dirname "$0")/.."

PHASE="${1:-0}"
EPOCHS="${2:-30}"
BATCH_SIZE="${3:-64}"
EXTRA_ARGS="${@:4}"  # All remaining arguments

# Phase configurations
declare -A PHASE_MODELS
declare -A PHASE_NAMES
declare -A PHASE_TITLES
declare -A PHASE_DESCRIPTIONS

# Phase 0: Basic models
PHASE_MODELS[0]="basic_r18 basic_effb0 basic_effl0 basic_mv3l basic_mv3s basic_sv2"
PHASE_NAMES[0]="P0-A: Basic-R18 P0-B: Basic-EFFB0 P0-C: Basic-EFFL0 P0-D: Basic-MV3-L P0-E: Basic-MV3-S P0-F: Basic-SV2"
PHASE_TITLES[0]="PHASE 0 - BASIC MODELS (Single-Frame Control)"
PHASE_DESCRIPTIONS[0]="Training all 6 basic models with different backbones"

# Phase A: Sequence baselines
PHASE_MODELS[A]="seq_brnn_mp seq_urnn_fs seq_bgru_mp seq_ugru_fs seq_blstm_mp seq_ulstm_fs"
PHASE_NAMES[A]="P1-A: SEQ-BRNN-R18-H128-L1-MP P1-B: SEQ-URNN-R18-H128-L1-FS P1-C: SEQ-BGRU-R18-H128-L1-MP P1-D: SEQ-UGRU-R18-H128-L1-FS P1-E: SEQ-BLSTM-R18-H128-L1-MP P1-F: SEQ-ULSTM-R18-H128-L1-FS"
PHASE_TITLES[A]="PHASE A - SEQUENCE BASELINES (CNN + RNN Variants)"
PHASE_DESCRIPTIONS[A]="Training all 6 sequence baseline models"

# Phase B: Attention models
PHASE_MODELS[B]="attn_bgru_bahdanau attn_bgru_luong attn_bgru_gate attn_bgru_hc attn_ugru_gate attn_ugru_hc"
PHASE_NAMES[B]="P2-A: ATTN-BGRU + Bahdanau P2-B: ATTN-BGRU + Luong P2-C: ATTN-BGRU + Gate P2-D: ATTN-BGRU + HC P2-E: ATTN-UGRU + Gate P2-F: ATTN-UGRU + HC"
PHASE_TITLES[B]="PHASE B - LIGHTWEIGHT ATTENTION / FRAME SELECTION"
PHASE_DESCRIPTIONS[B]="Training all 6 attention models"

# Validate phase
if [[ ! -v PHASE_MODELS[$PHASE] ]]; then
    echo "Error: Invalid phase '$PHASE'. Use: 0, A, or B"
    exit 1
fi

# Get phase configuration
MODELS=(${PHASE_MODELS[$PHASE]})
MODEL_NAMES=(${PHASE_NAMES[$PHASE]})
TITLE="${PHASE_TITLES[$PHASE]}"
DESCRIPTION="${PHASE_DESCRIPTIONS[$PHASE]}"

# Display header
echo "=================================================================================="
echo "$TITLE"
echo "=================================================================================="
echo ""
echo "$DESCRIPTION:"
for i in "${!MODEL_NAMES[@]}"; do
    echo "  ${MODEL_NAMES[$i]}"
done
echo ""
echo "Configuration:"
echo "  - Epochs: $EPOCHS"
echo "  - Batch size: $BATCH_SIZE"
if [[ -n "$EXTRA_ARGS" ]]; then
    echo "  - Extra args: $EXTRA_ARGS"
fi
echo ""

# Train each model
for i in "${!MODELS[@]}"; do
    model_type="${MODELS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    
    echo ""
    echo "=================================================================================="
    echo "Training: $model_name ($model_type)"
    echo "=================================================================================="
    
    python main.py --model_type "$model_type" --epochs "$EPOCHS" --batch_size "$BATCH_SIZE" $EXTRA_ARGS
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_name completed successfully"
    else
        echo "❌ $model_name failed"
        exit 1
    fi
done

echo ""
echo "=================================================================================="
echo "PHASE $PHASE COMPLETE"
echo "=================================================================================="
echo ""
echo "All models trained. Check outputs/logs/ for training logs."
echo "Check outputs/checkpoints/ for saved models."
echo "Run 'experiments/compare_phase.sh $PHASE' to compare results."
echo "=================================================================================="

