"""
Production Benchmarking for attn_bgru_luong

Main script that orchestrates the complete production benchmarking pipeline:
FP32 baseline, FP16 inference, TorchScript optimization, batching experiments,
and accuracy validation.
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data import build_dataloaders
from models import build_model
from utils import get_device, set_seed

from analysis.benchmark_steps import (
    step1_fp32_baseline,
    step2_fp16_inference,
    step3_torchscript,
    step4_batching_experiment,
    step5_accuracy_validation,
)
from analysis.benchmark_report import generate_report
from analysis.benchmark_utils import ACCURACY_TOLERANCE


def main() -> None:
    """Main benchmarking function."""
    # Setup
    config = Config()
    device = get_device()
    set_seed(config.seed)
    
    print("="*80)
    print("PRODUCTION BENCHMARKING: attn_bgru_luong")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model: attn_bgru_luong")
    
    # Load best model
    best_model_type = "attn_bgru_luong"
    checkpoint_path = Path(config.checkpoint_dir) / f"{best_model_type}_best.pth"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(best_model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Build test loader for accuracy validation
    print("\n📦 Building test dataloader for accuracy validation...")
    _, _, test_loader = build_dataloaders(config, model_type="seq")
    
    # Store all results
    all_results = {}
    
    # Step 1: FP32 Baseline
    fp32_latency = step1_fp32_baseline(model, config, device)
    fp32_accuracy = step5_accuracy_validation(
        model, "FP32", test_loader, device, best_model_type, config
    )
    all_results["fp32"] = {
        "latency": fp32_latency,
        "accuracy": fp32_accuracy,
    }
    
    # Step 2: FP16 Inference
    model_fp16, fp16_latency = step2_fp16_inference(model, config, device)
    fp16_accuracy = step5_accuracy_validation(
        model_fp16, "FP16", test_loader, device, best_model_type, config, use_fp16=True
    )
    all_results["fp16"] = {
        "latency": fp16_latency,
        "accuracy": fp16_accuracy,
    }
    
    # Step 3: FP16 + TorchScript
    model_scripted, ts_latency = step3_torchscript(model_fp16, config, device)
    if model_scripted is not None:
        ts_accuracy = step5_accuracy_validation(
            model_scripted, "FP16+TorchScript", test_loader, device, best_model_type, config, use_fp16=True
        )
        all_results["fp16_torchscript"] = {
            "latency": ts_latency,
            "accuracy": ts_accuracy,
        }
    
    # Step 4: Batching Experiment
    batching_results = step4_batching_experiment(model_fp16, config, device)
    all_results["batching"] = batching_results
    
    # Generate report
    generate_report(all_results, Path(config.output_dir))
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    fp32_ms = fp32_latency["mean_ms"]
    fp16_ms = fp16_latency["mean_ms"]
    speedup = fp32_ms / fp16_ms
    print(f"FP32 Baseline: {fp32_ms:.2f} ms")
    print(f"FP16 Speedup: {speedup:.2f}x ({fp16_ms:.2f} ms)")
    
    if model_scripted is not None:
        ts_ms = ts_latency["mean_ms"]
        ts_speedup = fp32_ms / ts_ms
        print(f"FP16+TorchScript Speedup: {ts_speedup:.2f}x ({ts_ms:.2f} ms)")
    
    accuracy_preserved = abs(fp32_accuracy['test_acc_number'] - fp16_accuracy['test_acc_number']) < ACCURACY_TOLERANCE
    print(f"Accuracy preserved: {accuracy_preserved}")


if __name__ == "__main__":
    main()
