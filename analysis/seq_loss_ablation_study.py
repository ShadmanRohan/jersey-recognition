"""
Loss ablation study for SEQUENCE model: With vs Without Loss Full
Tests only two configurations to see if loss_full contributes to learning.
"""

import json
from pathlib import Path
from typing import Dict
from tabulate import tabulate

from config import Config
from data import build_dataloaders
from models import build_model
from trainer import run_training
from utils import set_seed
from analysis.utils import extract_test_metrics_from_log


def main():
    """Run loss ablation study for sequence model."""
    print("="*80)
    print("LOSS COMPONENT ABLATION STUDY - SEQUENCE MODEL")
    print("="*80)
    print("This study tests how loss_full affects sequence model performance:")
    print("  - Acc Tens (tens digit accuracy)")
    print("  - Acc Ones (ones digit accuracy)")
    print("  - Acc Number (both digits correct)")
    print("  - Acc Full (full number classification)")
    print("="*80)
    
    config = Config()
    results_dir = Path("outputs/ablation_studies")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Only two experiments: with and without loss_full
    experiments = [
        {
            "name": "With Loss Full (baseline)",
            "weights": (1.0, 1.0, 1.0),
            "log_suffix": "with_full"
        },
        {
            "name": "Without Loss Full",
            "weights": (1.0, 1.0, 0.0),
            "log_suffix": "without_full"
        }
    ]
    
    model_type = "seq"
    backbone = "resnet18"
    epochs = 30
    batch_size = 64
    
    results = []
    base_dir = Path(__file__).parent
    
    for exp in experiments:
        tens_w, ones_w, full_w = exp["weights"]
        
        print(f"\n{'='*80}")
        print(f"Training SEQUENCE model with loss weights: Tens={tens_w}, Ones={ones_w}, Full={full_w}")
        print(f"Experiment: {exp['name']}")
        print(f"{'='*80}")
        
        # Set seed for reproducibility
        set_seed(config.seed)
        
        # Modify config loss weights
        original_weights = config.loss_weights
        config.loss_weights = exp["weights"]
        config.max_epochs = epochs
        config.batch_size = batch_size
        config.backbone = backbone
        
        try:
            # Run training - this will save to seq_training.log
            # We'll need to backup/rename the log file after each run
            log_file = base_dir / "outputs" / "logs" / f"seq_loss_ablation_{exp['log_suffix']}_training.log"
            
            # Temporarily redirect log file
            original_log_dir = config.log_dir
            config.log_dir = str(base_dir / "outputs" / "logs")
            
            # Run training
            history = run_training(
                model_type=model_type,
                config=config,
                backbone_name=backbone
            )
            
            # The actual log file will be seq_training.log, let's copy it
            actual_log_file = base_dir / "outputs" / "logs" / "seq_training.log"
            if actual_log_file.exists():
                import shutil
                shutil.copy(actual_log_file, log_file)
            
            # Extract metrics from log file
            test_metrics = extract_test_metrics_from_log(log_file)
            
            if test_metrics:
                result_dict = {
                    "loss_weights": {
                        "tens": tens_w,
                        "ones": ones_w,
                        "full": full_w
                    },
                    "experiment_name": exp["name"],
                    "test_loss": test_metrics['test_loss'],
                    "test_acc_number": test_metrics['test_acc_number'],
                    "test_acc_tens": test_metrics['test_acc_tens'],
                    "test_acc_ones": test_metrics['test_acc_ones'],
                    "test_acc_full": test_metrics['test_acc_full'],
                    "model_type": model_type,
                    "backbone": backbone,
                    "epochs": epochs
                }
                results.append(result_dict)
                
                print(f"\n✅ Completed: {exp['name']}")
                print(f"   Test Acc Number: {result_dict['test_acc_number']:.4f}")
            else:
                print(f"⚠️  Could not extract test metrics from log file")
                results.append({
                    "loss_weights": {"tens": tens_w, "ones": ones_w, "full": full_w},
                    "experiment_name": exp["name"],
                    "status": "incomplete",
                    "error": "Could not extract metrics from log"
                })
            
            # Save intermediate results
            results_file = results_dir / "seq_loss_ablation_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"\n❌ Failed: {exp['name']}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append({
                "loss_weights": {"tens": tens_w, "ones": ones_w, "full": full_w},
                "experiment_name": exp["name"],
                "status": "failed",
                "error": str(e)
            })
        finally:
            # Restore original weights
            config.loss_weights = original_weights
    
    # Final results summary
    print("\n" + "="*80)
    print("ABLATION STUDY RESULTS - SEQUENCE MODEL")
    print("="*80)
    
    successful = [r for r in results if r.get("status") != "failed" and "test_acc_number" in r]
    
    if successful:
        table_data = []
        for r in successful:
            weights = r["loss_weights"]
            table_data.append([
                r["experiment_name"],
                f"({weights['tens']}, {weights['ones']}, {weights['full']})",
                f"{r['test_acc_number']:.4f}",
                f"{r['test_acc_tens']:.4f}",
                f"{r['test_acc_ones']:.4f}",
                f"{r['test_acc_full']:.4f}",
                f"{r['test_loss']:.4f}"
            ])
        
        print(tabulate(
            table_data,
            headers=["Experiment", "Loss Weights\n(Tens, Ones, Full)", "Acc Number", "Acc Tens", "Acc Ones", "Acc Full", "Test Loss"],
            tablefmt="grid"
        ))
        
        # Calculate differences
        if len(successful) == 2:
            with_full = successful[0] if successful[0]["loss_weights"]["full"] > 0 else successful[1]
            without_full = successful[1] if successful[1]["loss_weights"]["full"] == 0 else successful[0]
            
            print("\n" + "="*80)
            print("COMPARISON")
            print("="*80)
            print(f"Acc Number: {with_full['test_acc_number']:.4f} (with) vs {without_full['test_acc_number']:.4f} (without)")
            diff_num = with_full['test_acc_number'] - without_full['test_acc_number']
            pct_num = (diff_num / without_full['test_acc_number'] * 100) if without_full['test_acc_number'] > 0 else 0
            print(f"  Difference: {diff_num:+.4f} ({pct_num:+.2f}%)")
            
            print(f"Acc Tens: {with_full['test_acc_tens']:.4f} (with) vs {without_full['test_acc_tens']:.4f} (without)")
            diff_tens = with_full['test_acc_tens'] - without_full['test_acc_tens']
            pct_tens = (diff_tens / without_full['test_acc_tens'] * 100) if without_full['test_acc_tens'] > 0 else 0
            print(f"  Difference: {diff_tens:+.4f} ({pct_tens:+.2f}%)")
            
            print(f"Acc Ones: {with_full['test_acc_ones']:.4f} (with) vs {without_full['test_acc_ones']:.4f} (without)")
            diff_ones = with_full['test_acc_ones'] - without_full['test_acc_ones']
            pct_ones = (diff_ones / without_full['test_acc_ones'] * 100) if without_full['test_acc_ones'] > 0 else 0
            print(f"  Difference: {diff_ones:+.4f} ({pct_ones:+.2f}%)")
    
    # Save final results
    results_file = results_dir / "seq_loss_ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")


if __name__ == "__main__":
    main()

