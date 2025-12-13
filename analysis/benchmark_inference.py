"""
Benchmark inference speed for anchor, seq, and frame models.
Measures real-world production performance on test set.
"""

import torch
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

from config import Config
from data import build_dataloaders
from models import build_model
from utils import get_device, set_seed


def load_checkpoint(model_path: Path, model_type: str, config: Config, device: torch.device):
    """Load model from checkpoint."""
    if not model_path.exists():
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    model = build_model(model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def warmup_model(model: torch.nn.Module, model_type: str, device: torch.device, config: Config, num_warmup: int = 10):
    """Warmup the model with dummy data."""
    model.eval()
    
    with torch.no_grad():
        if model_type == "anchor":
            # Single image batch
            dummy_input = torch.randn(1, 3, config.img_height, config.img_width).to(device)
            for _ in range(num_warmup):
                _ = model(dummy_input)
        elif model_type in ["seq", "seq_attn", "seq_uni"]:
            # Sequence batch
            dummy_frames = torch.randn(1, 8, 3, config.img_height, config.img_width).to(device)
            dummy_lengths = torch.tensor([8], dtype=torch.long).to(device)
            for _ in range(num_warmup):
                _ = model(dummy_frames, dummy_lengths)
    
    # Synchronize GPU
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_model(
    model: torch.nn.Module,
    model_type: str,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    config: Config,
    num_batches: int = None
) -> Dict[str, float]:
    """
    Benchmark model inference speed.
    
    Returns:
        Dictionary with timing metrics
    """
    model.eval()
    
    # Warmup
    print(f"  Warming up {model_type} model...")
    warmup_model(model, model_type, device, config, num_warmup=20)
    
    # Synchronize before timing
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    total_samples = 0
    total_forward_time = 0.0
    total_end_to_end_time = 0.0
    batch_times = []
    forward_times = []
    
    num_batches_to_test = num_batches if num_batches else len(test_loader)
    num_batches_to_test = min(num_batches_to_test, len(test_loader))
    
    print(f"  Running inference on {num_batches_to_test} batches...")
    
    with torch.no_grad():
        batch_idx = 0
        for batch in test_loader:
            if batch_idx >= num_batches_to_test:
                break
            
            # End-to-end timing (includes data transfer)
            start_e2e = time.perf_counter()
            
            if model_type == "anchor":
                images = batch["image"].to(device)
                batch_size = images.shape[0]
                
                # Forward pass timing
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_forward = time.perf_counter()
                
                outputs = model(images)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_forward = time.perf_counter()
                
            elif model_type in ["seq", "seq_attn", "seq_uni"]:
                frames = batch["frames"].to(device)
                lengths = batch["lengths"].to(device)
                batch_size = frames.shape[0]
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                start_forward = time.perf_counter()
                
                outputs = model(frames, lengths)
                
                if device.type == "cuda":
                    torch.cuda.synchronize()
                end_forward = time.perf_counter()
                
            
            end_e2e = time.perf_counter()
            
            forward_time = end_forward - start_forward
            e2e_time = end_e2e - start_e2e
            
            total_forward_time += forward_time
            total_end_to_end_time += e2e_time
            total_samples += batch_size
            batch_times.append(e2e_time)
            forward_times.append(forward_time)
            
            batch_idx += 1
    
    # Calculate statistics
    avg_forward_time = total_forward_time / num_batches_to_test
    avg_e2e_time = total_end_to_end_time / num_batches_to_test
    avg_per_sample_forward = total_forward_time / total_samples
    avg_per_sample_e2e = total_end_to_end_time / total_samples
    
    throughput_forward = total_samples / total_forward_time  # samples/second
    throughput_e2e = total_samples / total_end_to_end_time  # samples/second
    
    # Percentiles for latency
    batch_times_sorted = sorted(batch_times)
    p50_e2e = np.percentile(batch_times_sorted, 50) / (total_samples / num_batches_to_test)  # per sample
    p95_e2e = np.percentile(batch_times_sorted, 95) / (total_samples / num_batches_to_test)
    p99_e2e = np.percentile(batch_times_sorted, 99) / (total_samples / num_batches_to_test)
    
    return {
        "total_samples": total_samples,
        "num_batches": num_batches_to_test,
        "avg_batch_time_e2e_ms": avg_e2e_time * 1000,
        "avg_batch_time_forward_ms": avg_forward_time * 1000,
        "avg_per_sample_e2e_ms": avg_per_sample_e2e * 1000,
        "avg_per_sample_forward_ms": avg_per_sample_forward * 1000,
        "throughput_e2e_samples_per_sec": throughput_e2e,
        "throughput_forward_samples_per_sec": throughput_forward,
        "p50_latency_ms": p50_e2e * 1000,
        "p95_latency_ms": p95_e2e * 1000,
        "p99_latency_ms": p99_e2e * 1000,
    }


def run_benchmark(config: Config):
    """Run comprehensive inference benchmark."""
    device = get_device()
    set_seed(config.seed)
    
    print("="*80)
    print("INFERENCE SPEED BENCHMARK")
    print("="*80)
    print(f"Device: {device}")
    print(f"Batch size: {config.batch_size}")
    print()
    
    # Build test dataloader
    print("Building test dataloader...")
    _, _, test_loader = build_dataloaders(config, model_type="seq")  # Use seq for test loader structure
    print(f"Test set size: {len(test_loader.dataset)} samples")
    print(f"Number of batches: {len(test_loader)}")
    print()
    
    checkpoint_dir = Path(config.checkpoint_dir)
    results = {}
    
    # Models to benchmark
    models_to_test = [
        ("anchor", "anchor_model_best.pth"),
        ("seq", "best_seq.pt"),
        ("frame", "best_frame.pt"),
    ]
    
    # Test each model
    for model_type, checkpoint_name in models_to_test:
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            print(f"⚠ Skipping {model_type}: checkpoint not found at {checkpoint_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Benchmarking {model_type.upper()} Model")
        print(f"{'='*80}")
        
        # Load model
        print(f"Loading {model_type} model from {checkpoint_path}...")
        model, checkpoint = load_checkpoint(checkpoint_path, model_type, config, device)
        
        if model is None:
            print(f"⚠ Failed to load {model_type} model")
            continue
        
        # Build appropriate dataloader for this model type
        if model_type == "anchor":
            _, _, test_loader_model = build_dataloaders(config, model_type="anchor")
        else:
            test_loader_model = test_loader
        
        # Benchmark
        benchmark_results = benchmark_model(
            model,
            model_type,
            test_loader_model,
            device,
            config,
            num_batches=None  # Use all batches
        )
        
        results[model_type] = benchmark_results
        
        # Print summary
        print(f"\n{model_type.upper()} Model Results:")
        print(f"  Total samples processed: {benchmark_results['total_samples']}")
        print(f"  Batches processed: {benchmark_results['num_batches']}")
        print(f"  Avg per-sample latency (E2E): {benchmark_results['avg_per_sample_e2e_ms']:.3f} ms")
        print(f"  Avg per-sample latency (forward): {benchmark_results['avg_per_sample_forward_ms']:.3f} ms")
        print(f"  Throughput (E2E): {benchmark_results['throughput_e2e_samples_per_sec']:.2f} samples/sec")
        print(f"  Throughput (forward): {benchmark_results['throughput_forward_samples_per_sec']:.2f} samples/sec")
        print(f"  P50 latency: {benchmark_results['p50_latency_ms']:.3f} ms")
        print(f"  P95 latency: {benchmark_results['p95_latency_ms']:.3f} ms")
        print(f"  P99 latency: {benchmark_results['p99_latency_ms']:.3f} ms")
    
    # Create comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    if results:
        # Prepare table data
        table_data = []
        headers = [
            "Model",
            "Latency (ms)",
            "Throughput (samples/sec)",
            "P50 (ms)",
            "P95 (ms)",
            "P99 (ms)",
            "Batch Time (ms)"
        ]
        
        for model_type in ["anchor", "seq"]:
            if model_type in results:
                r = results[model_type]
                table_data.append([
                    model_type.upper(),
                    f"{r['avg_per_sample_e2e_ms']:.3f}",
                    f"{r['throughput_e2e_samples_per_sec']:.2f}",
                    f"{r['p50_latency_ms']:.3f}",
                    f"{r['p95_latency_ms']:.3f}",
                    f"{r['p99_latency_ms']:.3f}",
                    f"{r['avg_batch_time_e2e_ms']:.2f}"
                ])
        
        if HAS_TABULATE:
            print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".3f"))
        else:
            # Fallback: simple table formatting
            col_widths = [max(len(str(h)), max(len(str(row[i])) for row in table_data)) for i, h in enumerate(headers)]
            # Print header
            header_row = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
            print(header_row)
            print("-" * len(header_row))
            # Print data
            for row in table_data:
                print(" | ".join(str(cell).ljust(col_widths[i]) for i, cell in enumerate(row)))
        
        # Save results to JSON
        output_file = Path(config.output_dir) / "inference_benchmark.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to native Python types for JSON
        results_json = {}
        for model_type, metrics in results.items():
            results_json[model_type] = {
                k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                for k, v in metrics.items()
            }
        
        with open(output_file, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        
        # Find fastest model
        if len(results) > 1:
            fastest = min(results.items(), key=lambda x: x[1]['avg_per_sample_e2e_ms'])
            print(f"\n🏆 Fastest model: {fastest[0].upper()} ({fastest[1]['avg_per_sample_e2e_ms']:.3f} ms per sample)")
            print(f"   Throughput: {fastest[1]['throughput_e2e_samples_per_sec']:.2f} samples/sec")
    else:
        print("No results to display. Check that model checkpoints exist.")
    
    return results


if __name__ == "__main__":
    config = Config()
    results = run_benchmark(config)

