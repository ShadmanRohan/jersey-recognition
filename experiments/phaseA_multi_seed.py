#!/usr/bin/env python3
"""
Phase A: Run all sequence baseline models with multiple seeds and compute statistics.
Runs each model for 30 epochs with different seeds, then computes mean and std dev.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
import numpy as np
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from utils import set_seed, ensure_dirs
from trainer import run_training
from analysis.utils import extract_test_metrics_from_log


def run_single_experiment(model_type: str, seed: int, epochs: int = 30) -> Dict[str, float]:
    """Run a single training experiment and return test metrics."""
    config = Config()
    config.max_epochs = epochs
    config.seed = seed
    config.use_discriminative_lr = True  # Phase A uses discriminative LR
    config.scheduler_type = "cosine"  # Phase A uses cosine scheduler
    
    set_seed(seed)
    ensure_dirs(config)
    
    print(f"\n{'='*60}")
    print(f"Running: {model_type} with seed {seed}")
    print(f"{'='*60}")
    
    try:
        # Run training
        history = run_training(model_type, config, backbone_name=None)
        
        # Extract test metrics from log file
        log_file = Path(config.log_dir) / f"{model_type}_training.log"
        metrics = extract_test_metrics_from_log(log_file)
        
        if metrics:
            print(f"✅ Completed: {model_type} (seed {seed})")
            print(f"   Test Acc Number: {metrics['test_acc_number']:.4f}")
            return metrics
        else:
            print(f"⚠️  Could not extract metrics from log for {model_type} (seed {seed})")
            return None
            
    except Exception as e:
        print(f"❌ Failed: {model_type} (seed {seed}) - {e}")
        import traceback
        traceback.print_exc()
        return None


def compute_statistics(results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std dev for each metric."""
    if not results:
        return {}
    
    # Collect all metric values
    metrics_dict = defaultdict(list)
    for result in results:
        for key, value in result.items():
            if key != 'seed':  # Exclude seed from statistics
                metrics_dict[key].append(value)
    
    # Compute statistics
    stats = {}
    for key, values in metrics_dict.items():
        stats[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "n_runs": len(values)
        }
    
    return stats


def main():
    """Run Phase A experiments with multiple seeds."""
    # Phase A models
    models = [
        "seq_brnn_mp",
        "seq_urnn_fs",
        "seq_bgru_mp",
        "seq_ugru_fs",
        "seq_blstm_mp",
        "seq_ulstm_fs",
    ]
    
    model_names = [
        "P1-A: SEQ-BRNN-R18-H128-L1-MP",
        "P1-B: SEQ-URNN-R18-H128-L1-FS",
        "P1-C: SEQ-BGRU-R18-H128-L1-MP",
        "P1-D: SEQ-UGRU-R18-H128-L1-FS",
        "P1-E: SEQ-BLSTM-R18-H128-L1-MP",
        "P1-F: SEQ-ULSTM-R18-H128-L1-FS",
    ]
    
    # Seeds to use
    seeds = [42, 123, 456, 789, 2024]
    epochs = 30
    
    print("="*80)
    print("PHASE A - MULTI-SEED EXPERIMENTS")
    print("="*80)
    print(f"Models: {len(models)}")
    print(f"Seeds per model: {len(seeds)}")
    print(f"Epochs per run: {epochs}")
    print(f"Total runs: {len(models) * len(seeds)}")
    print(f"\nConfiguration:")
    print(f"  - Discriminative LR: Enabled")
    print(f"    * Backbone: 1e-4")
    print(f"    * Temporal: 3e-4")
    print(f"    * Heads: 3e-4")
    print(f"  - Scheduler: Cosine annealing with 1-epoch warmup")
    print(f"  - Mixed precision: Enabled")
    print(f"  - Gradient clipping: 1.0")
    print("="*80)
    
    # Results storage
    all_results = {}
    
    # Run experiments for each model
    for model_type, model_name in zip(models, model_names):
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name} ({model_type})")
        print(f"{'='*80}")
        
        model_results = []
        
        for seed in seeds:
            metrics = run_single_experiment(model_type, seed, epochs=epochs)
            if metrics:
                metrics['seed'] = seed
                model_results.append(metrics)
        
        if model_results:
            # Compute statistics
            stats = compute_statistics(model_results)
            all_results[model_type] = {
                "model_name": model_name,
                "individual_runs": model_results,
                "statistics": stats
            }
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"SUMMARY: {model_name}")
            print(f"{'='*60}")
            print(f"Successful runs: {len(model_results)}/{len(seeds)}")
            if stats:
                print(f"\nTest Accuracy (Number):")
                print(f"  Mean: {stats['test_acc_number']['mean']:.4f} ± {stats['test_acc_number']['std']:.4f}")
                print(f"  Range: [{stats['test_acc_number']['min']:.4f}, {stats['test_acc_number']['max']:.4f}]")
                print(f"\nTest Accuracy (Tens):")
                print(f"  Mean: {stats['test_acc_tens']['mean']:.4f} ± {stats['test_acc_tens']['std']:.4f}")
                print(f"\nTest Accuracy (Ones):")
                print(f"  Mean: {stats['test_acc_ones']['mean']:.4f} ± {stats['test_acc_ones']['std']:.4f}")
                print(f"\nTest Loss:")
                print(f"  Mean: {stats['test_loss']['mean']:.4f} ± {stats['test_loss']['std']:.4f}")
        else:
            print(f"\n❌ No successful runs for {model_name}")
            all_results[model_type] = {
                "model_name": model_name,
                "status": "failed",
                "individual_runs": []
            }
    
    # Save results
    results_dir = Path(__file__).parent.parent / "outputs"
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / "phaseA_multi_seed_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults saved to: {results_file}")
    
    # Print final summary table
    print(f"\n{'='*80}")
    print("FINAL SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Model':<30} {'Acc Number (Mean±Std)':<25} {'Acc Tens':<15} {'Acc Ones':<15} {'Loss':<15}")
    print(f"{'-'*80}")
    
    for model_type, result in all_results.items():
        if "statistics" in result and result["statistics"]:
            stats = result["statistics"]
            model_name = result["model_name"].split(":")[1].strip() if ":" in result["model_name"] else model_type
            acc_num = f"{stats['test_acc_number']['mean']:.4f}±{stats['test_acc_number']['std']:.4f}"
            acc_tens = f"{stats['test_acc_tens']['mean']:.4f}"
            acc_ones = f"{stats['test_acc_ones']['mean']:.4f}"
            loss = f"{stats['test_loss']['mean']:.4f}"
            print(f"{model_name:<30} {acc_num:<25} {acc_tens:<15} {acc_ones:<15} {loss:<15}")
    
    print(f"{'='*80}")


if __name__ == "__main__":
    main()




