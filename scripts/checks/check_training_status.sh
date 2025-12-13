#!/bin/bash
# Check training status and show which models are ready

# Change to project root directory
cd "$(dirname "$0")/../.."

echo "Checking training status..."
echo ""

backbones=("resnet18" "efficientnet_b0" "mobilenet_v3_large" "mobilenet_v3_small" "shufflenet_v2_x1_0")

for backbone in "${backbones[@]}"; do
    checkpoint="outputs/checkpoints/anchor_${backbone}_best.pth"
    if [ -f "$checkpoint" ]; then
        echo "✅ $backbone - Ready"
    else
        echo "⏳ $backbone - Training..."
    fi
done

echo ""
echo "To run final comparison after all models are trained:"
echo "  python analysis/print_all_results.py"



