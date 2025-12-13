#!/bin/bash
# Check training status for EfficientNet-B0 seq and frame models

# Change to project root directory
cd "$(dirname "$0")/../.."

echo "Checking EfficientNet-B0 training status..."
echo ""

checkpoints=(
    "outputs/checkpoints/seq_efficientnet_b0_best.pth"
    "outputs/checkpoints/frame_efficientnet_b0_best.pth"
)

models=("seq" "frame")

for i in "${!models[@]}"; do
    model="${models[$i]}"
    checkpoint="${checkpoints[$i]}"
    
    if [ -f "$checkpoint" ]; then
        echo "✅ ${model} model - Training complete"
    else
        echo "⏳ ${model} model - Training in progress..."
    fi
done

echo ""
echo "To check logs:"
echo "  tail -f outputs/logs/seq_efficientnet_b0_*.log"
echo "  tail -f outputs/logs/frame_efficientnet_b0_*.log"

