"""
View and analyze loss ablation study results.
"""

import json
from pathlib import Path
from tabulate import tabulate


def main():
    """Display loss ablation study results."""
    base_dir = Path(__file__).parent
    results_file = base_dir / "outputs" / "ablation_studies" / "loss_ablation_results.json"
    
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("   Run loss_ablation_study.py first to generate results.")
        return
    
    with open(results_file) as f:
        results = json.load(f)
    
    # Filter successful results
    successful = [r for r in results if r.get("status") != "failed" and "test_acc_number" in r]
    
    if not successful:
        print("❌ No successful experiments found in results!")
        if results:
            print("\nFailed experiments:")
            for r in results:
                if r.get("status") == "failed":
                    print(f"  - {r.get('experiment_name', 'Unknown')}: {r.get('error', 'Unknown error')}")
        return
    
    # Sort by Acc Number (descending)
    successful.sort(key=lambda x: x["test_acc_number"], reverse=True)
    
    # Find baseline
    baseline = next((r for r in successful if r["experiment_name"] == "Baseline (all equal)"), None)
    
    print("="*80)
    print("LOSS COMPONENT ABLATION STUDY RESULTS")
    print("="*80)
    print()
    
    # Create table
    table_data = []
    headers = [
        "Experiment",
        "Loss Weights\n(Tens, Ones, Full)",
        "Acc Number",
        "Acc Tens",
        "Acc Ones",
        "Acc Full",
        "Test Loss"
    ]
    
    for r in successful:
        weights_str = f"({r['loss_weights']['tens']}, {r['loss_weights']['ones']}, {r['loss_weights']['full']})"
        
        # Calculate difference from baseline
        if baseline:
            diff_number = r["test_acc_number"] - baseline["test_acc_number"]
            diff_tens = r["test_acc_tens"] - baseline["test_acc_tens"]
            diff_ones = r["test_acc_ones"] - baseline["test_acc_ones"]
            
            acc_number_str = f"{r['test_acc_number']:.4f} ({diff_number:+.4f})"
            acc_tens_str = f"{r['test_acc_tens']:.4f} ({diff_tens:+.4f})"
            acc_ones_str = f"{r['test_acc_ones']:.4f} ({diff_ones:+.4f})"
        else:
            acc_number_str = f"{r['test_acc_number']:.4f}"
            acc_tens_str = f"{r['test_acc_tens']:.4f}"
            acc_ones_str = f"{r['test_acc_ones']:.4f}"
        
        table_data.append([
            r["experiment_name"],
            weights_str,
            acc_number_str,
            acc_tens_str,
            acc_ones_str,
            f"{r['test_acc_full']:.4f}",
            f"{r['test_loss']:.4f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Analysis
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    if baseline:
        print(f"\n📊 Baseline (all equal): Acc Number = {baseline['test_acc_number']:.4f}")
        
        # Find best configuration
        best = successful[0]
        if best["experiment_name"] != "Baseline (all equal)":
            diff = best["test_acc_number"] - baseline["test_acc_number"]
            print(f"\n🏆 Best Configuration: {best['experiment_name']}")
            print(f"   Acc Number: {best['test_acc_number']:.4f} ({diff:+.4f} vs baseline)")
            print(f"   Loss Weights: ({best['loss_weights']['tens']}, {best['loss_weights']['ones']}, {best['loss_weights']['full']})")
        
        # Compare no full loss vs baseline
        no_full = next((r for r in successful if r["experiment_name"] == "No Full Loss (tens + ones only)"), None)
        if no_full:
            diff = no_full["test_acc_number"] - baseline["test_acc_number"]
            print(f"\n🔍 No Full Loss vs Baseline:")
            print(f"   Acc Number: {no_full['test_acc_number']:.4f} ({diff:+.4f}, {diff*100:+.2f}%)")
            print(f"   Acc Tens: {no_full['test_acc_tens']:.4f} ({no_full['test_acc_tens'] - baseline['test_acc_tens']:+.4f})")
            print(f"   Acc Ones: {no_full['test_acc_ones']:.4f} ({no_full['test_acc_ones'] - baseline['test_acc_ones']:+.4f})")
            if abs(diff) < 0.001:
                print("   ➡️  Removing full loss has MINIMAL effect on Acc Number")
            elif diff < 0:
                print("   ⚠️  Removing full loss HURTS Acc Number")
            else:
                print("   ✅ Removing full loss IMPROVES Acc Number")
        
        # Compare only full vs baseline
        only_full = next((r for r in successful if r["experiment_name"] == "Only Full Loss"), None)
        if only_full:
            diff = only_full["test_acc_number"] - baseline["test_acc_number"]
            print(f"\n🔍 Only Full Loss vs Baseline:")
            print(f"   Acc Number: {only_full['test_acc_number']:.4f} ({diff:+.4f}, {diff*100:+.2f}%)")
            if diff < -0.01:
                print("   ⚠️  Using only full loss SIGNIFICANTLY HURTS performance")
                print("   ℹ️  Full loss alone cannot learn digit-level features effectively")
        
        # Compare only tens vs baseline
        only_tens = next((r for r in successful if r["experiment_name"] == "Only Tens Loss"), None)
        if only_tens:
            diff = only_tens["test_acc_number"] - baseline["test_acc_number"]
            print(f"\n🔍 Only Tens Loss vs Baseline:")
            print(f"   Acc Number: {only_tens['test_acc_number']:.4f} ({diff:+.4f})")
            print(f"   Acc Tens: {only_tens['test_acc_tens']:.4f}")
            print(f"   Acc Ones: {only_tens['test_acc_ones']:.4f}")
            if only_tens['test_acc_ones'] < 0.5:
                print("   ⚠️  Without ones loss, model cannot learn ones digit")
        
        # Compare only ones vs baseline
        only_ones = next((r for r in successful if r["experiment_name"] == "Only Ones Loss"), None)
        if only_ones:
            diff = only_ones["test_acc_number"] - baseline["test_acc_number"]
            print(f"\n🔍 Only Ones Loss vs Baseline:")
            print(f"   Acc Number: {only_ones['test_acc_number']:.4f} ({diff:+.4f})")
            print(f"   Acc Tens: {only_ones['test_acc_tens']:.4f}")
            print(f"   Acc Ones: {only_ones['test_acc_ones']:.4f}")
            if only_ones['test_acc_tens'] < 0.5:
                print("   ⚠️  Without tens loss, model cannot learn tens digit")
    
    print(f"\n📁 Full results: {results_file}")
    print("="*80)


if __name__ == "__main__":
    main()

