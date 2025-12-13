#!/bin/bash
# Test script: Run all phase experiments for 1 epoch
# Purpose: Quick testing to verify all models work correctly

# Change to project root directory
cd "$(dirname "$0")/.."

echo "=================================================================================="
echo "TESTING ALL PHASES - 1 EPOCH EACH"
echo "=================================================================================="
echo ""
echo "This will train all models from all phases for 1 epoch each:"
echo "  Phase 0: 6 basic models"
echo "  Phase A: 6 sequence baseline models"
echo "  Phase B: 6 attention models"
echo "  Total: 18 models"
echo ""

# Phase 0: Basic models
echo "=================================================================================="
echo "PHASE 0 - BASIC MODELS (Single-Frame)"
echo "=================================================================================="
echo ""

phase0_models=(
    "basic_r18"
    "basic_effb0"
    "basic_effl0"
    "basic_mv3l"
    "basic_mv3s"
    "basic_sv2"
)

phase0_names=(
    "P0-A: Basic-R18"
    "P0-B: Basic-EFFB0"
    "P0-C: Basic-EFFL0"
    "P0-D: Basic-MV3-L"
    "P0-E: Basic-MV3-S"
    "P0-F: Basic-SV2"
)

for i in "${!phase0_models[@]}"; do
    model_type="${phase0_models[$i]}"
    model_name="${phase0_names[$i]}"
    
    echo "Training: $model_name ($model_type)"
    python main.py --model_type "$model_type" --epochs 1 --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_name completed"
    else
        echo "❌ $model_name failed"
        exit 1
    fi
    echo ""
done

# Phase A: Sequence baselines
echo "=================================================================================="
echo "PHASE A - SEQUENCE BASELINES"
echo "=================================================================================="
echo ""

phaseA_models=(
    "seq_brnn_mp"
    "seq_urnn_fs"
    "seq_bgru_mp"
    "seq_ugru_fs"
    "seq_blstm_mp"
    "seq_ulstm_fs"
)

phaseA_names=(
    "P1-A: SEQ-BRNN-MP"
    "P1-B: SEQ-URNN-FS"
    "P1-C: SEQ-BGRU-MP"
    "P1-D: SEQ-UGRU-FS"
    "P1-E: SEQ-BLSTM-MP"
    "P1-F: SEQ-ULSTM-FS"
)

for i in "${!phaseA_models[@]}"; do
    model_type="${phaseA_models[$i]}"
    model_name="${phaseA_names[$i]}"
    
    echo "Training: $model_name ($model_type)"
    python main.py --model_type "$model_type" --epochs 1 --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_name completed"
    else
        echo "❌ $model_name failed"
        exit 1
    fi
    echo ""
done

# Phase B: Attention models
echo "=================================================================================="
echo "PHASE B - ATTENTION MODELS"
echo "=================================================================================="
echo ""

phaseB_models=(
    "attn_bgru_bahdanau"
    "attn_bgru_luong"
    "attn_bgru_gate"
    "attn_bgru_hc"
    "attn_ugru_gate"
    "attn_ugru_hc"
)

phaseB_names=(
    "P2-A: ATTN-BGRU + Bahdanau"
    "P2-B: ATTN-BGRU + Luong"
    "P2-C: ATTN-BGRU + Gate"
    "P2-D: ATTN-BGRU + HC"
    "P2-E: ATTN-UGRU + Gate"
    "P2-F: ATTN-UGRU + HC"
)

for i in "${!phaseB_models[@]}"; do
    model_type="${phaseB_models[$i]}"
    model_name="${phaseB_names[$i]}"
    
    echo "Training: $model_name ($model_type)"
    python main.py --model_type "$model_type" --epochs 1 --batch_size 64
    
    if [ $? -eq 0 ]; then
        echo "✅ $model_name completed"
    else
        echo "❌ $model_name failed"
        exit 1
    fi
    echo ""
done

echo "=================================================================================="
echo "ALL PHASES COMPLETE - TEST SUCCESSFUL"
echo "=================================================================================="
echo ""
echo "All 18 models trained successfully for 1 epoch each."
echo "Check outputs/logs/ for training logs."
echo "Check outputs/checkpoints/ for saved models."
echo "=================================================================================="

