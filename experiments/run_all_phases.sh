#!/bin/bash
# Master script: Run all three phases sequentially
# This will train all 18 models (6 Phase 0 + 6 Phase A + 6 Phase B)

cd "$(dirname "$0")"

echo "=================================================================================="
echo "MASTER EXPERIMENT: RUN ALL THREE PHASES"
echo "=================================================================================="
echo ""
echo "This will train all 18 models across three phases:"
echo "  Phase 0: 6 basic models (single-frame control)"
echo "  Phase A: 6 sequence baseline models (RNN variants) - with discriminative LR"
echo "  Phase B: 6 attention models (lightweight attention) - with discriminative LR"
echo ""
echo "This will take a very long time. Press Ctrl+C to cancel, or wait 10 seconds to continue..."
sleep 10

for phase in 0 A B; do
    echo ""
    echo "=================================================================================="
    echo "STARTING PHASE $phase"
    echo "=================================================================================="
    
    if [ "$phase" = "0" ]; then
        ./phase0_train_all.sh
    elif [ "$phase" = "A" ]; then
        ./phaseA_train_all.sh
    else
        ./phaseB_train_all.sh
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Phase $phase failed. Stopping."
        exit 1
    fi
done

echo ""
echo "=================================================================================="
echo "ALL PHASES COMPLETE"
echo "=================================================================================="
echo ""
echo "All 18 models have been trained across three phases."
echo "Run 'experiments/compare_all_phases.sh' to see comprehensive comparison."
echo "=================================================================================="

