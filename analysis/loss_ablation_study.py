"""
Ablation study on loss components to understand their contribution to learning.

Tests different loss weight combinations:
- Baseline: (1.0, 1.0, 1.0) - all components equal
- No full loss: (1.0, 1.0, 0.0) - only tens + ones
- Only tens: (1.0, 0.0, 0.0)
- Only ones: (0.0, 1.0, 0.0)
- Only full: (0.0, 0.0, 1.0)
- Reduced full: (1.0, 1.0, 0.5)
- Increased full: (1.0, 1.0, 2.0)
"""

import subprocess
import json
from pathlib import Path
from typing import Dict, Tuple, List
from tabulate import tabulate
import torch

from config import Config
from data import build_dataloaders
from models import build_model
from trainer import evaluate, run_training
from utils import get_device, set_seed


def train_with_loss_weights(
    loss_weights: Tuple[float, float, float],
    config: Config,
    model_type: str = "anchor",
    backbone: str = "resnet18",
    epochs: int = 30,
    experiment_name: str = None
) -> Dict:
    """
    Train a model with specific loss weights and return test results.
    
    Args:
        loss_weights: (w_tens, w_ones, w_full)
        config: Config object
        model_type: Model type to train
        backbone: Backbone name
        epochs: Number of epochs
        experiment_name: Name for logging
    
    Returns:
        Dictionary with test metrics and loss weights
    """
    print(f"\n{'='*80}")
    print(f"Training with loss weights: Tens={loss_weights[0]}, Ones={loss_weights[1]}, Full={loss_weights[2]}")
    if experiment_name:
        print(f"Experiment: {experiment_name}")
    print(f"{'='*80}\n")
    
    # Create a copy of config with modified loss weights
    config_copy = Config()
    config_copy.loss_weights = loss_weights
    config_copy.max_epochs = epochs
    config_copy.batch_size = config.batch_size
    config_copy.backbone = backbone
    
    # Use unique checkpoint name to avoid conflicts
    exp_safe_name = experiment_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(",", "")[:50]
    config_copy.checkpoint_dir = str(Path(config.checkpoint_dir) / "ablation" / exp_safe_name)
    Path(config_copy.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Build dataloaders
    train_loader, val_loader, test_loader = build_dataloaders(config_copy, model_type=model_type)
    
    # Build model
    device = get_device()
    model = build_model(model_type, config_copy, backbone_name=backbone).to(device)
    
    # Train (this will save checkpoints to unique directory)
    history = run_training(model_type, config_copy, backbone_name=backbone)
    
    # Load best model and evaluate on test set
    print("\nEvaluating on test set with best model...")
    
    # Find best checkpoint
    checkpoint_dir = Path(config_copy.checkpoint_dir)
    if model_type == "anchor":
        checkpoint_path = checkpoint_dir / f"anchor_{backbone}_best.pth"
    else:
        checkpoint_path = checkpoint_dir / f"best_{model_type}.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Check both possible key names
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
        elif "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)  # Fallback: checkpoint is state dict itself
        print(f"Loaded best checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        print("⚠️  No checkpoint found, using current model state")
    
    test_loss, test_metrics = evaluate(model, test_loader, device, model_type, config_copy, phase="Testing")
    
    result = {
        "loss_weights": {
            "tens": loss_weights[0],
            "ones": loss_weights[1],
            "full": loss_weights[2]
        },
        "experiment_name": experiment_name or f"tens_{loss_weights[0]}_ones_{loss_weights[1]}_full_{loss_weights[2]}",
        "test_loss": float(test_loss),
        "test_acc_number": float(test_metrics["acc_number"]),
        "test_acc_tens": float(test_metrics["acc_tens"]),
        "test_acc_ones": float(test_metrics["acc_ones"]),
        "test_acc_full": float(test_metrics.get("acc_full", 0.0)),
        "model_type": model_type,
        "backbone": backbone,
        "epochs": epochs
    }
    
    return result


def main():
    """Run ablation study on loss components."""
    print("="*80)
    print("LOSS COMPONENT ABLATION STUDY")
    print("="*80)
    print("\nThis study tests how different loss weight combinations affect:")
    print("  - Acc Tens (tens digit accuracy)")
    print("  - Acc Ones (ones digit accuracy)")
    print("  - Acc Number (both digits correct)")
    print("  - Acc Full (full number classification)")
    print()
    
    config = Config()
    base_dir = Path(__file__).parent
    results_dir = base_dir / "outputs" / "ablation_studies"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Define loss weight combinations to test
    loss_configs = [
        # (tens_weight, ones_weight, full_weight, experiment_name)
        (1.0, 1.0, 1.0, "Baseline (all equal)"),
        (1.0, 1.0, 0.0, "No Full Loss (tens + ones only)"),
        (1.0, 0.0, 0.0, "Only Tens Loss"),
        (0.0, 1.0, 0.0, "Only Ones Loss"),
        (0.0, 0.0, 1.0, "Only Full Loss"),
        (1.0, 1.0, 0.5, "Reduced Full (0.5x)"),
        (1.0, 1.0, 2.0, "Increased Full (2.0x)"),
        (2.0, 1.0, 1.0, "Increased Tens (2.0x)"),
        (1.0, 2.0, 1.0, "Increased Ones (2.0x)"),
    ]
    
    # Use anchor model with ResNet18 for faster training
    model_type = "anchor"
    backbone = "resnet18"
    epochs = 30  # Full training for meaningful results
    
    results = []
    
    for tens_w, ones_w, full_w, exp_name in loss_configs:
        loss_weights = (tens_w, ones_w, full_w)
        
        try:
            result = train_with_loss_weights(
                loss_weights=loss_weights,
                config=config,
                model_type=model_type,
                backbone=backbone,
                epochs=epochs,
                experiment_name=exp_name
            )
            results.append(result)
            
            # Save intermediate results
            results_file = results_dir / "loss_ablation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\n✅ Completed: {exp_name}")
            print(f"   Acc Number: {result['test_acc_number']:.4f}")
            print(f"   Acc Tens: {result['test_acc_tens']:.4f}")
            print(f"   Acc Ones: {result['test_acc_ones']:.4f}")
            print(f"   Acc Full: {result['test_acc_full']:.4f}")
            
        except Exception as e:
            print(f"\n❌ Failed: {exp_name}")
            print(f"   Error: {e}")
            results.append({
                "loss_weights": {"tens": tens_w, "ones": ones_w, "full": full_w},
                "experiment_name": exp_name,
                "status": "failed",
                "error": str(e)
            })
    
    # Save final results
    results_file = results_dir / "loss_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create comparison table
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS")
    print("="*80)
    
    # Filter successful results
    successful = [r for r in results if r.get("status") != "failed" and "test_acc_number" in r]
    
    if not successful:
        print("❌ No successful experiments!")
        return
    
    # Sort by Acc Number (descending)
    successful.sort(key=lambda x: x["test_acc_number"], reverse=True)
    
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
    
    baseline = None
    for r in successful:
        if r["experiment_name"] == "Baseline (all equal)":
            baseline = r
            break
    
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
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if baseline:
        print(f"\n📊 Baseline (all equal): Acc Number = {baseline['test_acc_number']:.4f}")
        
        # Find best configuration
        best = successful[0]
        print(f"\n🏆 Best Configuration: {best['experiment_name']}")
        print(f"   Acc Number: {best['test_acc_number']:.4f}")
        print(f"   Loss Weights: ({best['loss_weights']['tens']}, {best['loss_weights']['ones']}, {best['loss_weights']['full']})")
        
        # Compare no full loss vs baseline
        no_full = next((r for r in successful if r["experiment_name"] == "No Full Loss (tens + ones only)"), None)
        if no_full:
            diff = no_full["test_acc_number"] - baseline["test_acc_number"]
            print(f"\n🔍 No Full Loss vs Baseline:")
            print(f"   Acc Number difference: {diff:+.4f} ({diff*100:+.2f}%)")
            if diff < -0.001:
                print("   ⚠️  Removing full loss HURTS performance")
            elif diff > 0.001:
                print("   ✅ Removing full loss IMPROVES performance")
            else:
                print("   ➡️  Removing full loss has minimal effect")
        
        # Compare only full vs baseline
        only_full = next((r for r in successful if r["experiment_name"] == "Only Full Loss"), None)
        if only_full:
            diff = only_full["test_acc_number"] - baseline["test_acc_number"]
            print(f"\n🔍 Only Full Loss vs Baseline:")
            print(f"   Acc Number difference: {diff:+.4f} ({diff*100:+.2f}%)")
            if diff < -0.001:
                print("   ⚠️  Using only full loss HURTS performance significantly")
            else:
                print("   ℹ️  Full loss alone is insufficient")
    
    print(f"\n📁 Results saved to: {results_file}")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

