"""
Compare old vs new augmentation strategies for Frame model.
"""

import json
from pathlib import Path
from tabulate import tabulate

from analysis.utils import extract_test_metrics_from_log

def main():
    """Compare augmentation strategies."""
    base_dir = Path(__file__).parent.parent
    log_dir = base_dir / "outputs" / "logs"
    
    print("="*80)
    print("AUGMENTATION STRATEGY COMPARISON")
    print("="*80)
    print()
    
    # Old augmentation (horizontal flip, ±10° rotation, color jitter)
    old_log = log_dir / "frame_resnet18_training.log"
    old_metrics = extract_test_metrics_from_log(old_log)
    
    # New augmentation (±5° rotation, blur, noise, NO flip, color jitter)
    new_log = log_dir / "frame_training.log"
    new_metrics = extract_test_metrics_from_log(new_log)
    
    if not old_metrics:
        print("⚠️  Could not read old augmentation results")
        print(f"   Looking for: {old_log}")
        if old_log.exists():
            print("   File exists but format may be different")
        return
    
    if not new_metrics:
        print("⚠️  Could not read new augmentation results")
        print(f"   Looking for: {new_log}")
        print("   Training may still be in progress...")
        return
    
    print("OLD AUGMENTATION STRATEGY:")
    print("  - Horizontal flip (50% probability)")
    print("  - Random rotation: ±10 degrees")
    print("  - Color jitter: brightness, contrast, saturation")
    print()
    
    print("NEW AUGMENTATION STRATEGY:")
    print("  - ❌ Horizontal flip REMOVED")
    print("  - Random rotation: ±5 degrees (reduced)")
    print("  - Color jitter: brightness, contrast, saturation")
    print("  - 🆕 Motion blur simulation (30% chance)")
    print("  - 🆕 Gaussian noise (20% chance)")
    print()
    
    print("="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    print()
    
    table_data = []
    headers = ["Metric", "Old Augmentation", "New Augmentation", "Difference"]
    
    metrics_list = [
        ("Test Loss", "loss"),
        ("Acc Number", "acc_number"),
        ("Acc Tens", "acc_tens"),
        ("Acc Ones", "acc_ones"),
        ("Acc Full", "acc_full"),
    ]
    
    for metric_name, metric_key in metrics_list:
        # Map display names to actual metric keys
        actual_key = f"test_{metric_key}" if metric_key != "loss" else "test_loss"
        old_val = old_metrics.get(actual_key, old_metrics.get(metric_key, 0.0))
        new_val = new_metrics.get(actual_key, new_metrics.get(metric_key, 0.0))
        diff = new_val - old_val
        sign = "+" if diff >= 0 else ""
        table_data.append([
            metric_name,
            f"{old_val:.4f}",
            f"{new_val:.4f}",
            f"{sign}{diff:.4f} ({sign}{diff*100:.2f}%)"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()
    
    # Analysis
    old_acc_number = old_metrics.get('test_acc_number', old_metrics.get('acc_number', 0.0))
    new_acc_number = new_metrics.get('test_acc_number', new_metrics.get('acc_number', 0.0))
    main_diff = new_acc_number - old_acc_number
    print("="*80)
    print("ANALYSIS")
    print("="*80)
    print()
    
    if main_diff > 0.001:
        print(f"✅ IMPROVED: New augmentation strategy performed better by {main_diff:.4f} ({main_diff*100:.2f}%)")
        print("   The improvements (removed flip, reduced rotation, added blur/noise) helped!")
    elif main_diff < -0.001:
        print(f"⚠️  REGRESSION: New augmentation strategy performed worse by {abs(main_diff):.4f} ({abs(main_diff)*100:.2f}%)")
        print("   The old strategy may have been better, or more training is needed.")
    else:
        print("➡️  SIMILAR: Performance is essentially the same.")
        print("   Both strategies are comparable.")
    
    print()
    print("Key Changes:")
    print("  • Removed horizontal flip (prevents invalid number samples)")
    print("  • Reduced rotation from ±10° to ±5° (better digit preservation)")
    print("  • Added motion blur (simulates real video conditions)")
    print("  • Added noise (simulates compression artifacts)")

if __name__ == "__main__":
    main()

