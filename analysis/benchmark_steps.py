"""
Benchmarking step functions for production benchmarking.

Implements the individual benchmarking steps: FP32 baseline, FP16 inference,
TorchScript optimization, batching experiments, and accuracy validation.
"""

from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from trainer import multitask_loss
from analysis.benchmark_utils import (
    create_input_tensors,
    benchmark_forward_pass,
    DEFAULT_BATCH_SIZES,
)


def step1_fp32_baseline(
    model: torch.nn.Module,
    config: Config,
    device: torch.device
) -> Dict[str, float]:
    """
    Step 1: Clean FP32 baseline benchmark.
    
    Args:
        model: FP32 model
        config: Configuration object
        device: Target device
    
    Returns:
        Latency statistics dictionary
    """
    print("\n" + "="*60)
    print("STEP 1: FP32 BASELINE")
    print("="*60)
    
    inputs, lengths = create_input_tensors(
        batch_size=1,
        seq_len=config.max_seq_len,
        img_h=config.img_height,
        img_w=config.img_width,
        device=device,
        dtype=torch.float32
    )
    
    results = benchmark_forward_pass(model, inputs, lengths)
    
    print(f"  Mean latency: {results['mean_ms']:.2f} ms")
    print(f"  Median latency: {results['median_ms']:.2f} ms")
    print(f"  P95 latency: {results['p95_ms']:.2f} ms")
    
    return results


def step2_fp16_inference(
    model: torch.nn.Module,
    config: Config,
    device: torch.device
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """
    Step 2: FP16 inference benchmark.
    
    Args:
        model: FP32 model (will be converted to FP16)
        config: Configuration object
        device: Target device
    
    Returns:
        Tuple of (FP16 model, latency statistics)
    """
    print("\n" + "="*60)
    print("STEP 2: FP16 INFERENCE")
    print("="*60)
    
    # Convert model to FP16
    model_fp16 = model.half()
    
    inputs, lengths = create_input_tensors(
        batch_size=1,
        seq_len=config.max_seq_len,
        img_h=config.img_height,
        img_w=config.img_width,
        device=device,
        dtype=torch.float16
    )
    
    results = benchmark_forward_pass(model_fp16, inputs, lengths, use_autocast=True)
    
    print(f"  Mean latency: {results['mean_ms']:.2f} ms")
    print(f"  Median latency: {results['median_ms']:.2f} ms")
    print(f"  P95 latency: {results['p95_ms']:.2f} ms")
    
    return model_fp16, results


def step3_torchscript(
    model_fp16: torch.nn.Module,
    config: Config,
    device: torch.device
) -> Tuple[Optional[torch.jit.ScriptModule], Optional[Dict[str, float]]]:
    """
    Step 3: FP16 + TorchScript optimization benchmark.
    
    Args:
        model_fp16: FP16 model
        config: Configuration object
        device: Target device
    
    Returns:
        Tuple of (scripted model or None, latency statistics or None)
    """
    print("\n" + "="*60)
    print("STEP 3: FP16 + TORCHSCRIPT")
    print("="*60)
    
    try:
        print("  Scripting model...")
        model_scripted = torch.jit.script(model_fp16)
        model_scripted = torch.jit.optimize_for_inference(model_scripted)
        
        inputs, lengths = create_input_tensors(
            batch_size=1,
            seq_len=config.max_seq_len,
            img_h=config.img_height,
            img_w=config.img_width,
            device=device,
            dtype=torch.float16
        )
        
        results = benchmark_forward_pass(model_scripted, inputs, lengths, use_autocast=True)
        
        print(f"  Mean latency: {results['mean_ms']:.2f} ms")
        print(f"  Median latency: {results['median_ms']:.2f} ms")
        print(f"  P95 latency: {results['p95_ms']:.2f} ms")
        
        return model_scripted, results
    except Exception as e:
        print(f"  ⚠️  TorchScript failed: {e}")
        return None, None


def step4_batching_experiment(
    model_fp16: torch.nn.Module,
    config: Config,
    device: torch.device,
    batch_sizes: List[int] = DEFAULT_BATCH_SIZES
) -> Dict[int, Dict[str, float]]:
    """
    Step 4: Batching experiment to measure throughput scaling.
    
    Args:
        model_fp16: FP16 model
        config: Configuration object
        device: Target device
        batch_sizes: List of batch sizes to test
    
    Returns:
        Dictionary mapping batch_size to latency and throughput metrics
    """
    print("\n" + "="*60)
    print("STEP 4: BATCHING EXPERIMENT")
    print("="*60)
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        try:
            inputs, lengths = create_input_tensors(
                batch_size=batch_size,
                seq_len=config.max_seq_len,
                img_h=config.img_height,
                img_w=config.img_width,
                device=device,
                dtype=torch.float16
            )
            
            batch_results = benchmark_forward_pass(model_fp16, inputs, lengths, use_autocast=True)
            
            # Calculate per-sequence metrics
            latency_per_seq = batch_results['mean_ms'] / batch_size
            throughput = (batch_size / batch_results['mean_ms']) * 1000  # sequences per second
            
            results[batch_size] = {
                **batch_results,
                "latency_per_seq_ms": latency_per_seq,
                "throughput_seq_per_sec": throughput,
            }
            
            print(f"    Total latency: {batch_results['mean_ms']:.2f} ms")
            print(f"    Latency per seq: {latency_per_seq:.2f} ms")
            print(f"    Throughput: {throughput:.2f} seq/s")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    ⚠️  OOM at batch_size={batch_size}, skipping larger batches")
                break
            else:
                raise
    
    return results


def step5_accuracy_validation(
    model: torch.nn.Module,
    model_variant: str,
    test_loader,
    device: torch.device,
    model_type: str,
    config: Config,
    use_fp16: bool = False
) -> Dict[str, float]:
    """
    Step 5: Accuracy validation on test set.
    
    Args:
        model: Model to evaluate
        model_variant: String identifier for the model variant
        test_loader: Test data loader
        device: Target device
        model_type: Model type string
        config: Configuration object
        use_fp16: Whether model uses FP16 (for input conversion)
    
    Returns:
        Dictionary with test loss and accuracy metrics
    """
    print(f"\n  Evaluating {model_variant} accuracy on test set...")
    
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_counts = {"acc_tens": 0, "acc_ones": 0, "acc_number": 0}
    total_counts = {"acc_tens": 0, "acc_ones": 0, "acc_number": 0}
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", unit="batch"):
            frames = batch["frames"].to(device)
            if use_fp16:
                frames = frames.half()  # Convert to FP16 for FP16 models
            
            lengths = batch["lengths"].to(device)
            tens_label = batch["tens_label"].to(device)
            ones_label = batch["ones_label"].to(device)
            
            outputs = model(frames, lengths)
            loss, _ = multitask_loss(outputs, tens_label, ones_label, weights=config.loss_weights)
            
            tens_pred = outputs["tens_logits"].argmax(dim=-1)
            ones_pred = outputs["ones_logits"].argmax(dim=-1)
            
            # Accumulate correct predictions
            correct_counts["acc_tens"] += (tens_pred == tens_label).sum().item()
            correct_counts["acc_ones"] += (ones_pred == ones_label).sum().item()
            correct_counts["acc_number"] += ((tens_pred == tens_label) & (ones_pred == ones_label)).sum().item()
            
            # Accumulate total counts
            total_counts["acc_tens"] += tens_label.numel()
            total_counts["acc_ones"] += ones_label.numel()
            total_counts["acc_number"] += tens_label.numel()
            
            total_loss += loss.item() * frames.shape[0]
            total_samples += frames.shape[0]
    
    # Calculate final metrics
    avg_loss = total_loss / max(total_samples, 1)
    avg_metrics = {k: correct_counts[k] / max(total_counts[k], 1) for k in correct_counts.keys()}
    
    return {
        "test_loss": avg_loss,
        "test_acc_number": avg_metrics["acc_number"],
        "test_acc_tens": avg_metrics["acc_tens"],
        "test_acc_ones": avg_metrics["acc_ones"],
    }
