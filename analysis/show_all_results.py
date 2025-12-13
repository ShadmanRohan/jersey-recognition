#!/usr/bin/env python3
"""
Extract and display comprehensive results from all phase training logs.
Shows three separate tables with mean ± std dev for each phase.
"""

import numpy as np
from pathlib import Path
from tabulate import tabulate

from analysis.utils import extract_test_metrics_from_log

def main():
    base_dir = Path(__file__).parent.parent
    log_dir = base_dir / "outputs" / "logs"
    
    # Define all models by phase
    phase_models = {
        "Phase 0": [
            ("basic_r18", "Basic ResNet18"),
            ("basic_effb0", "Basic EfficientNet-B0"),
            ("basic_effl0", "Basic EfficientNet-L0"),
            ("basic_mv3l", "Basic MobileNetV3-Large"),
            ("basic_mv3s", "Basic MobileNetV3-Small"),
            ("basic_sv2", "Basic ShuffleNetV2"),
        ],
        "Phase A": [
            ("seq_brnn_mp", "SEQ-BRNN (Max Pool)"),
            ("seq_urnn_fs", "SEQ-URNN (Frame Selection)"),
            ("seq_bgru_mp", "SEQ-BGRU (Max Pool)"),
            ("seq_ugru_fs", "SEQ-UGRU (Frame Selection)"),
            ("seq_blstm_mp", "SEQ-BLSTM (Max Pool)"),
            ("seq_ulstm_fs", "SEQ-ULSTM (Frame Selection)"),
        ],
        "Phase B": [
            ("attn_bgru_bahdanau", "ATTN-BGRU + Bahdanau"),
            ("attn_bgru_luong", "ATTN-BGRU + Luong"),
            ("attn_bgru_gate", "ATTN-BGRU + Gate"),
            ("attn_bgru_hc", "ATTN-BGRU + HC"),
            ("attn_ugru_gate", "ATTN-UGRU + Gate"),
            ("attn_ugru_hc", "ATTN-UGRU + HC"),
        ],
    }
    
    # Collect results organized by phase
    phase_results = {}
    
    for phase_key, models in phase_models.items():
        phase_results[phase_key] = []
        
        for model_id, model_name in models:
            log_file = log_dir / f"{model_id}_training.log"
            metrics = extract_test_metrics_from_log(log_file)
            
            if metrics:
                phase_results[phase_key].append({
                    "model_name": model_name,
                    "acc_number": metrics.get('test_acc_number', 0.0),
                    "acc_tens": metrics.get('test_acc_tens', 0.0),
                    "acc_ones": metrics.get('test_acc_ones', 0.0),
                })
    
    # Print three separate tables, one per phase
    for phase_key in ["Phase 0", "Phase A", "Phase B"]:
        if phase_key not in phase_results:
            continue
            
        results = phase_results[phase_key]
        
        print("="*100)
        print(f"{phase_key.upper()}")
        print("="*100)
        print()
        
        # Create table data
        table_data = []
        for result in results:
            table_data.append([
                result["model_name"],
                f"{result['acc_number']:.4f}",
                f"{result['acc_tens']:.4f}",
                f"{result['acc_ones']:.4f}",
            ])
        
        headers = ["Model", "Acc Number", "Acc Tens", "Acc Ones"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
        
        # Calculate and display statistics
        if results:
            acc_numbers = [r['acc_number'] for r in results]
            acc_tens = [r['acc_tens'] for r in results]
            acc_ones = [r['acc_ones'] for r in results]
            
            print("Statistics:")
            print(f"  Acc Number: {np.mean(acc_numbers):.4f} ± {np.std(acc_numbers):.4f}")
            print(f"  Acc Tens:   {np.mean(acc_tens):.4f} ± {np.std(acc_tens):.4f}")
            print(f"  Acc Ones:   {np.mean(acc_ones):.4f} ± {np.std(acc_ones):.4f}")
            print()
            print()

if __name__ == "__main__":
    main()

