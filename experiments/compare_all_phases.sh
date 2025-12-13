#!/bin/bash
# Master script: Compare results across all three phases
# Simplified to use unified compare_phase.sh

cd "$(dirname "$0")"

echo "=================================================================================="
echo "COMPREHENSIVE COMPARISON: ALL THREE PHASES"
echo "=================================================================================="
echo ""

for phase in 0 A B; do
    ./compare_phase.sh "$phase"
    echo ""
done

echo "=================================================================================="
echo "COMPARISON COMPLETE"
echo "=================================================================================="
echo ""
echo "For detailed analysis, check individual training logs in outputs/logs/"
echo "Check saved checkpoints in outputs/checkpoints/"
echo "Run 'python analysis/print_all_results.py' for comprehensive results table."
echo "=================================================================================="

