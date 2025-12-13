"""
Report generation for production benchmarking.

Generates formatted tables and saves results to JSON and text files.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.benchmark_utils import ACCURACY_TOLERANCE


def _create_table1_gpu_comparison(all_results: Dict, log_lines: List[str]) -> None:
    """Create Table 1: GPU Inference Comparison."""
    fp32_latency = all_results["fp32"]["latency"]["mean_ms"]
    
    log_lines.append("Table 1: GPU Inference Comparison (RTX 3090)")
    log_lines.append("-" * 100)
    header = f"{'Model Variant':<20} {'Precision':<12} {'TorchScript':<12} {'Latency (ms)':<15} {'Speedup':<12} {'Accuracy':<12}"
    log_lines.append(header)
    log_lines.append("-" * 100)
    
    # FP32 baseline
    row = f"{'Base Model':<20} {'FP32':<12} {'No':<12} {fp32_latency:<15.2f} {'1.00×':<12} {all_results['fp32']['accuracy']['test_acc_number']*100:.2f}%"
    log_lines.append(row)
    print("\n" + "Table 1: GPU Inference Comparison (RTX 3090)")
    print("-" * 100)
    print(header)
    print("-" * 100)
    print(row)
    
    # FP16
    if "fp16" in all_results:
        fp16_latency = all_results["fp16"]["latency"]["mean_ms"]
        fp16_speedup = fp32_latency / fp16_latency
        row = f"{'Base Model':<20} {'FP16':<12} {'No':<12} {fp16_latency:<15.2f} {fp16_speedup:.2f}×{'':<9} {all_results['fp16']['accuracy']['test_acc_number']*100:.2f}%"
        log_lines.append(row)
        print(row)
    
    # FP16 + TorchScript
    if "fp16_torchscript" in all_results and all_results["fp16_torchscript"]["latency"]:
        ts_latency = all_results["fp16_torchscript"]["latency"]["mean_ms"]
        ts_speedup = fp32_latency / ts_latency
        row = f"{'Base Model':<20} {'FP16':<12} {'Yes':<12} {ts_latency:<15.2f} {ts_speedup:.2f}×{'':<9} {all_results['fp16_torchscript']['accuracy']['test_acc_number']*100:.2f}%"
        log_lines.append(row)
        print(row)
    
    log_lines.append("")


def _create_table2_accuracy_preservation(all_results: Dict, log_lines: List[str]) -> None:
    """Create Table 2: Accuracy Preservation."""
    log_lines.append("Table 2: Accuracy Preservation")
    log_lines.append("-" * 100)
    header = f"{'Precision':<15} {'Acc Number':<15} {'Acc Tens':<15} {'Acc Ones':<15} {'Status':<15}"
    log_lines.append(header)
    log_lines.append("-" * 100)
    print("\n" + "Table 2: Accuracy Preservation")
    print("-" * 100)
    print(header)
    print("-" * 100)
    
    fp32_acc = all_results["fp32"]["accuracy"]["test_acc_number"]
    
    for variant in ["fp32", "fp16", "fp16_torchscript"]:
        if variant in all_results and "accuracy" in all_results[variant]:
            acc = all_results[variant]["accuracy"]
            is_preserved = abs(acc["test_acc_number"] - fp32_acc) < ACCURACY_TOLERANCE
            status = "Preserved" if is_preserved else "Changed"
            
            precision = "FP32" if variant == "fp32" else "FP16" if variant == "fp16" else "FP16+JIT"
            row = f"{precision:<15} {acc['test_acc_number']*100:<15.2f} {acc['test_acc_tens']*100:<15.2f} {acc['test_acc_ones']*100:<15.2f} {status:<15}"
            log_lines.append(row)
            print(row)
    
    log_lines.append("")


def _create_table3_throughput_scaling(all_results: Dict, log_lines: List[str]) -> None:
    """Create Table 3: Throughput Scaling."""
    if "batching" not in all_results:
        return
    
    log_lines.append("Table 3: Throughput Scaling")
    log_lines.append("-" * 100)
    header = f"{'Batch Size':<12} {'Total Latency (ms)':<20} {'Latency/Seq (ms)':<20} {'Throughput (seq/s)':<20}"
    log_lines.append(header)
    log_lines.append("-" * 100)
    print("\n" + "Table 3: Throughput Scaling")
    print("-" * 100)
    print(header)
    print("-" * 100)
    
    for batch_size in sorted(all_results["batching"].keys(), key=int):
        batch_data = all_results["batching"][batch_size]
        row = f"{batch_size:<12} {batch_data['mean_ms']:<20.2f} {batch_data['latency_per_seq_ms']:<20.2f} {batch_data['throughput_seq_per_sec']:<20.2f}"
        log_lines.append(row)
        print(row)
    
    log_lines.append("")


def generate_report(all_results: Dict, output_dir: Path) -> None:
    """
    Generate comparison tables, log to file, and display.
    
    Args:
        all_results: Dictionary containing all benchmark results
        output_dir: Output directory for saving results
    """
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    # Prepare log content
    log_lines = []
    log_lines.append("="*80)
    log_lines.append("PRODUCTION BENCHMARK RESULTS: attn_bgru_luong")
    log_lines.append("="*80)
    log_lines.append("")
    
    # Create all tables
    _create_table1_gpu_comparison(all_results, log_lines)
    _create_table2_accuracy_preservation(all_results, log_lines)
    _create_table3_throughput_scaling(all_results, log_lines)
    
    log_lines.append("="*80)
    
    # Save JSON results
    results_file = output_dir / "production_benchmark_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save log file
    log_file = output_dir / "production_benchmark_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines))
    
    print(f"\n✅ Results saved to: {results_file}")
    print(f"✅ Log saved to: {log_file}")
