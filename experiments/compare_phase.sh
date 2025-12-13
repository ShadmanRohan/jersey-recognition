#!/bin/bash
# Unified comparison script for all phases
# Usage: ./compare_phase.sh <phase>
#   phase: 0, A, or B

# Change to project root directory
cd "$(dirname "$0")/.."

PHASE="${1:-0}"

# Phase configurations
declare -A PHASE_MODELS
declare -A PHASE_NAMES
declare -A PHASE_TITLES

# Phase 0: Basic models
PHASE_MODELS[0]="basic_r18 basic_effb0 basic_effl0 basic_mv3l basic_mv3s basic_sv2"
PHASE_NAMES[0]="P0-A: Basic-R18 P0-B: Basic-EFFB0 P0-C: Basic-EFFL0 P0-D: Basic-MV3-L P0-E: Basic-MV3-S P0-F: Basic-SV2"
PHASE_TITLES[0]="PHASE 0 - BASIC MODELS COMPARISON"

# Phase A: Sequence baselines
PHASE_MODELS[A]="seq_brnn_mp seq_urnn_fs seq_bgru_mp seq_ugru_fs seq_blstm_mp seq_ulstm_fs"
PHASE_NAMES[A]="P1-A: SEQ-BRNN-R18-H128-L1-MP P1-B: SEQ-URNN-R18-H128-L1-FS P1-C: SEQ-BGRU-R18-H128-L1-MP P1-D: SEQ-UGRU-R18-H128-L1-FS P1-E: SEQ-BLSTM-R18-H128-L1-MP P1-F: SEQ-ULSTM-R18-H128-L1-FS"
PHASE_TITLES[A]="PHASE A - SEQUENCE BASELINES COMPARISON"

# Phase B: Attention models
PHASE_MODELS[B]="attn_bgru_bahdanau attn_bgru_luong attn_bgru_gate attn_bgru_hc attn_ugru_gate attn_ugru_hc"
PHASE_NAMES[B]="P2-A: ATTN-BGRU + Bahdanau P2-B: ATTN-BGRU + Luong P2-C: ATTN-BGRU + Gate P2-D: ATTN-BGRU + HC P2-E: ATTN-UGRU + Gate P2-F: ATTN-UGRU + HC"
PHASE_TITLES[B]="PHASE B - ATTENTION MODELS COMPARISON"

# Validate phase
if [[ ! -v PHASE_MODELS[$PHASE] ]]; then
    echo "Error: Invalid phase '$PHASE'. Use: 0, A, or B"
    exit 1
fi

# Get phase configuration
MODELS=(${PHASE_MODELS[$PHASE]})
MODEL_NAMES=(${PHASE_NAMES[$PHASE]})
TITLE="${PHASE_TITLES[$PHASE]}"

# Display header
echo "=================================================================================="
echo "$TITLE"
echo "=================================================================================="
echo ""
echo "Comparing all models:"
echo ""

echo "Model results summary:"
echo "----------------------"
echo ""

for i in "${!MODELS[@]}"; do
    model_type="${MODELS[$i]}"
    model_name="${MODEL_NAMES[$i]}"
    
    log_file="outputs/logs/${model_type}_training.log"
    checkpoint="outputs/checkpoints/${model_type}_best.pth"
    
    if [ -f "$log_file" ]; then
        echo "$model_name:"
        echo "  Log: $log_file"
        if [ -f "$checkpoint" ]; then
            echo "  Checkpoint: $checkpoint"
        else
            echo "  Checkpoint: Not found"
        fi
        echo ""
    else
        echo "$model_name: Not trained yet"
        echo ""
    fi
done

echo "=================================================================================="
echo "For detailed comparison, check the training logs or run analysis scripts."
echo "Run 'python analysis/print_all_results.py' for comprehensive results table."
echo "=================================================================================="

