"""
Benchmarking utilities for production benchmarking.

Provides constants and utility functions for creating inputs and benchmarking
forward passes without dataloader overhead.
"""

import time
from typing import Dict, Tuple

import numpy as np
import torch


# ============================================================================
# Constants
# ============================================================================

BENCHMARK_WARMUP_ITERATIONS = 50
BENCHMARK_NUM_RUNS = 100
DEFAULT_BATCH_SIZES = [1, 4, 8, 16]
ACCURACY_TOLERANCE = 0.0001  # For accuracy preservation check


# ============================================================================
# Utility Functions
# ============================================================================

def create_input_tensors(
    batch_size: int,
    seq_len: int,
    img_h: int,
    img_w: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create input tensors for benchmarking (no dataloader overhead).
    
    Args:
        batch_size: Batch size for the input
        seq_len: Sequence length (number of frames)
        img_h: Image height
        img_w: Image width
        device: Target device (CPU/GPU)
        dtype: Data type for inputs (float32 or float16)
    
    Returns:
        Tuple of (input_tensors, length_tensors)
    """
    inputs = torch.randn(batch_size, seq_len, 3, img_h, img_w, dtype=dtype, device=device)
    lengths = torch.tensor([seq_len] * batch_size, dtype=torch.long, device=device)
    return inputs, lengths


def benchmark_forward_pass(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    lengths: torch.Tensor,
    num_warmup: int = BENCHMARK_WARMUP_ITERATIONS,
    num_runs: int = BENCHMARK_NUM_RUNS,
    use_autocast: bool = False
) -> Dict[str, float]:
    """
    Clean forward pass benchmark without dataloader overhead.
    
    Args:
        model: Model to benchmark
        inputs: Input tensors (batch_size, seq_len, 3, H, W)
        lengths: Sequence length tensors
        num_warmup: Number of warmup iterations
        num_runs: Number of benchmark runs
        use_autocast: Whether to use automatic mixed precision
    
    Returns:
        Dictionary with latency statistics (mean, median, min, max, p95, p99, std)
    """
    model.eval()
    latencies = []
    
    # Warmup phase
    with torch.inference_mode():
        for _ in range(num_warmup):
            if use_autocast:
                with torch.amp.autocast(device_type='cuda'):
                    _ = model(inputs, lengths)
            else:
                _ = model(inputs, lengths)
    
    if inputs.device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark phase
    with torch.inference_mode():
        for _ in range(num_runs):
            if inputs.device.type == 'cuda':
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            if use_autocast:
                with torch.amp.autocast(device_type='cuda'):
                    _ = model(inputs, lengths)
            else:
                _ = model(inputs, lengths)
            
            if inputs.device.type == 'cuda':
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms
    
    # Calculate statistics
    latencies = np.array(latencies)
    return {
        "mean_ms": float(np.mean(latencies)),
        "median_ms": float(np.median(latencies)),
        "min_ms": float(np.min(latencies)),
        "max_ms": float(np.max(latencies)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "std_ms": float(np.std(latencies)),
    }
