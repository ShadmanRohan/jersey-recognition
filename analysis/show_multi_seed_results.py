#!/usr/bin/env python3
"""
Display results tables with mean ± std dev over 5 runs for all phases.
"""

import json
import sys
from pathlib import Path
from tabulate import tabulate

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from models import build_model
from utils import count_parameters

def load_multi_seed_results(results_file: Path):
    """Load multi-seed results from JSON file."""
    if not results_file.exists():
        return None
    
    try:
        with open(results_file) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading {results_file}: {e}")
        return None

def main():
    base_dir = Path(__file__).parent.parent
    outputs_dir = base_dir / "outputs"
    
    # Define phase configurations
    phases = {
        "Phase 0": {
            "file": outputs_dir / "phase0_multi_seed_results.json",
            "model_order": [
                "basic_r18",
                "basic_effb0",
                "basic_effl0",
                "basic_mv3l",
                "basic_mv3s",
                "basic_sv2",
            ],
        },
        "Phase A": {
            "file": outputs_dir / "phaseA_multi_seed_results.json",
            "model_order": [
                "seq_brnn_mp",
                "seq_urnn_fs",
                "seq_bgru_mp",
                "seq_ugru_fs",
                "seq_blstm_mp",
                "seq_ulstm_fs",
            ],
        },
        "Phase B": {
            "file": outputs_dir / "phaseB_multi_seed_results.json",
            "model_order": [
                "attn_bgru_bahdanau",
                "attn_bgru_luong",
                "attn_bgru_gate",
                "attn_bgru_hc",
                "attn_ugru_gate",
                "attn_ugru_hc",
            ],
        },
    }
    
    # Model name mappings (removed "Basic" prefix for Phase 0)
    model_names = {
        # Phase 0
        "basic_r18": "ResNet18",
        "basic_effb0": "EfficientNet-B0",
        "basic_effl0": "EfficientNet-L0",
        "basic_mv3l": "MobileNetV3-Large",
        "basic_mv3s": "MobileNetV3-Small",
        "basic_sv2": "ShuffleNetV2",
        # Phase A
        "seq_brnn_mp": "SEQ-BRNN (Max Pool)",
        "seq_urnn_fs": "SEQ-URNN (Frame Selection)",
        "seq_bgru_mp": "SEQ-BGRU (Max Pool)",
        "seq_ugru_fs": "SEQ-UGRU (Frame Selection)",
        "seq_blstm_mp": "SEQ-BLSTM (Max Pool)",
        "seq_ulstm_fs": "SEQ-ULSTM (Frame Selection)",
        # Phase B
        "attn_bgru_bahdanau": "ATTN-BGRU + Bahdanau",
        "attn_bgru_luong": "ATTN-BGRU + Luong",
        "attn_bgru_gate": "ATTN-BGRU + Gate",
        "attn_bgru_hc": "ATTN-BGRU + HC",
        "attn_ugru_gate": "ATTN-UGRU + Gate",
        "attn_ugru_hc": "ATTN-UGRU + HC",
    }
    
    # Get parameter counts for all models
    config = Config()
    all_param_counts = {}
    
    print("Computing parameter counts for all models...")
    for phase_name, phase_config in phases.items():
        for model_id in phase_config["model_order"]:
            try:
                model = build_model(model_id, config)
                param_count = count_parameters(model)
                all_param_counts[model_id] = param_count
                print(f"  {model_id}: {param_count:,} parameters")
            except Exception as e:
                print(f"Warning: Could not get parameter count for {model_id}: {e}")
                all_param_counts[model_id] = None
    print()
    
    # Process each phase
    for phase_name, phase_config in phases.items():
        results_file = phase_config["file"]
        model_order = phase_config["model_order"]
        
        results_data = load_multi_seed_results(results_file)
        
        if not results_data:
            print(f"⚠️  No results file found for {phase_name}: {results_file}")
            print()
            continue
        
        print("="*100)
        print(f"{phase_name.upper()} - Mean ± Std Dev (5 runs)")
        print("="*100)
        print()
        
        table_data = []
        
        for model_id in model_order:
            if model_id not in results_data:
                continue
            
            model_data = results_data[model_id]
            
            if "statistics" not in model_data or not model_data["statistics"]:
                continue
            
            stats = model_data["statistics"]
            model_name = model_names.get(model_id, model_id)
            
            # Extract statistics
            acc_num_mean = stats.get("test_acc_number", {}).get("mean", 0.0)
            acc_num_std = stats.get("test_acc_number", {}).get("std", 0.0)
            acc_tens_mean = stats.get("test_acc_tens", {}).get("mean", 0.0)
            acc_tens_std = stats.get("test_acc_tens", {}).get("std", 0.0)
            acc_ones_mean = stats.get("test_acc_ones", {}).get("mean", 0.0)
            acc_ones_std = stats.get("test_acc_ones", {}).get("std", 0.0)
            
            # Build table row
            row = [
                model_name,
                f"{acc_num_mean:.4f} ± {acc_num_std:.4f}",
                f"{acc_tens_mean:.4f} ± {acc_tens_std:.4f}",
                f"{acc_ones_mean:.4f} ± {acc_ones_std:.4f}",
            ]
            
            # Add parameter counts for all phases
            if model_id in all_param_counts:
                param_count = all_param_counts[model_id]
                if param_count is not None:
                    row.insert(1, f"{param_count:,}")  # Insert after model name
                else:
                    row.insert(1, "N/A")
            
            table_data.append(row)
        
        # Set headers (all phases now have Parameters column)
        headers = ["Model", "Parameters", "Acc Number (Mean ± Std)", "Acc Tens (Mean ± Std)", "Acc Ones (Mean ± Std)"]
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print()
        print()

if __name__ == "__main__":
    main()

