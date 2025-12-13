"""
Extract sequence datapoints where the best model makes mistakes.
Saves 10 mistake sequences with their frames, predictions, and ground truth.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import shutil
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, get_class_mapping
from data import build_dataloaders, SequenceRecord
from models import build_model
from utils import get_device, set_seed


def find_best_model_from_results(results_file: Path) -> Tuple[str, float, int]:
    """Find the best performing model from multi-seed results JSON file.
    Returns: (model_type, best_acc, best_seed)
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    best_model_type = None
    best_acc = 0.0
    best_seed = None
    
    for model_type, data in results.items():
        if "individual_runs" in data:
            for run in data["individual_runs"]:
                if run["test_acc_number"] > best_acc:
                    best_acc = run["test_acc_number"]
                    best_model_type = model_type
                    best_seed = run.get("seed", None)
    
    return best_model_type, best_acc, best_seed


def extract_mistakes(config: Config, output_dir: Path = None, num_mistakes: int = 10):
    """Extract sequence datapoints where the model makes mistakes."""
    device = get_device()
    set_seed(config.seed)
    
    if output_dir is None:
        output_dir = Path(config.output_dir) / "mistake_sequences"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find best model
    results_file = Path(config.output_dir) / "phaseB_multi_seed_results.json"
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run phaseB_multi_seed.py first.")
        return
    
    best_model_type, best_acc, best_seed = find_best_model_from_results(results_file)
    print(f"📊 Best model: {best_model_type}")
    print(f"   Test Accuracy: {best_acc:.4f}")
    print(f"   Seed: {best_seed}")
    
    # Load checkpoint
    checkpoint_path = Path(config.checkpoint_dir) / f"{best_model_type}_best.pth"
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"📂 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Build model
    print(f"🔨 Building model: {best_model_type}")
    model = build_model(best_model_type, config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    # Build test dataloader
    print("📦 Building test dataloader...")
    is_basic_model = best_model_type in ["basic", "anchor"] or best_model_type.startswith("basic_") or best_model_type.startswith("anchor_")
    model_type_for_dataloader = "anchor" if is_basic_model else "seq"
    _, _, test_loader = build_dataloaders(config, model_type=model_type_for_dataloader)
    
    # Get dataset and records
    test_dataset = test_loader.dataset
    test_records = test_dataset.records
    
    print(f"🔍 Running inference on {len(test_records)} test sequences...")
    
    # Collect mistakes
    mistakes = []
    _, idx2str = get_class_mapping(config)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Collecting mistakes")):
            batch_start_idx = batch_idx * config.batch_size
            
            # Get ground truth
            tens_label = batch["tens_label"]
            ones_label = batch["ones_label"]
            
            # Run inference
            if is_basic_model:
                images = batch["image"].to(device)
                outputs = model(images)
            else:
                frames = batch["frames"].to(device)
                lengths = batch["lengths"].to(device)
                outputs = model(frames, lengths)
            
            # Get predictions
            tens_pred = outputs["tens_logits"].argmax(dim=-1).cpu()
            ones_pred = outputs["ones_logits"].argmax(dim=-1).cpu()
            
            # Get probabilities for analysis
            tens_probs = torch.softmax(outputs["tens_logits"], dim=-1).cpu()
            ones_probs = torch.softmax(outputs["ones_logits"], dim=-1).cpu()
            
            # Check each sample in batch
            for i in range(len(tens_label)):
                idx = batch_start_idx + i
                if idx >= len(test_records):
                    break
                
                gt_tens = tens_label[i].item()
                gt_ones = ones_label[i].item()
                pred_tens = tens_pred[i].item()
                pred_ones = ones_pred[i].item()
                
                # Check if prediction is wrong (both tens and ones must match)
                is_correct = (pred_tens == gt_tens) and (pred_ones == gt_ones)
                
                if not is_correct:
                    record = test_records[idx]
                    
                    # Get prediction probabilities
                    tens_prob_values = tens_probs[i].numpy()
                    ones_prob_values = ones_probs[i].numpy()
                    
                    mistake_info = {
                        "index": idx,
                        "ground_truth": {
                            "tens": int(gt_tens),
                            "ones": int(gt_ones),
                            "jersey_str": record.jersey_str,
                            "full_number": int(gt_tens * 10 + gt_ones) if gt_tens < 10 else int(gt_ones)
                        },
                        "prediction": {
                            "tens": int(pred_tens),
                            "ones": int(pred_ones),
                            "jersey_str": f"{pred_tens}{pred_ones}" if pred_tens < 10 else str(pred_ones),
                            "full_number": int(pred_tens * 10 + pred_ones) if pred_tens < 10 else int(pred_ones)
                        },
                        "probabilities": {
                            "tens": {
                                "predicted_class": float(tens_prob_values[pred_tens]),
                                "ground_truth_class": float(tens_prob_values[gt_tens]),
                                "all_classes": tens_prob_values.tolist()
                            },
                            "ones": {
                                "predicted_class": float(ones_prob_values[pred_ones]),
                                "ground_truth_class": float(ones_prob_values[gt_ones]),
                                "all_classes": ones_prob_values.tolist()
                            }
                        },
                        "sequence_path": str(record.sequence_path),
                        "track_id": record.track_id,
                        "frame_paths": [str(fp) for fp in record.frame_paths],
                        "num_frames": len(record.frame_paths),
                        "record": record  # Keep for copying files, but don't serialize
                    }
                    
                    mistakes.append(mistake_info)
                    
                    # Stop if we have enough mistakes
                    if len(mistakes) >= num_mistakes:
                        break
            
            if len(mistakes) >= num_mistakes:
                break
    
    print(f"\n✅ Found {len(mistakes)} mistakes (requested {num_mistakes})")
    
    # Save mistakes
    mistakes_dir = output_dir / "mistakes"
    mistakes_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary (without SequenceRecord objects)
    summary = {
        "model_type": best_model_type,
        "model_accuracy": float(best_acc),
        "seed": best_seed,
        "num_mistakes_extracted": len(mistakes),
        "mistakes": []
    }
    
    # Convert mistakes to JSON-serializable format
    for mistake in mistakes:
        mistake_serializable = {k: v for k, v in mistake.items() if k != "record"}
        summary["mistakes"].append(mistake_serializable)
    
    summary_file = output_dir / "mistakes_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"💾 Saved summary to: {summary_file}")
    
    # Copy sequence frames for each mistake
    print(f"\n📁 Copying sequence frames...")
    for mistake_idx, mistake in enumerate(tqdm(mistakes, desc="Copying sequences")):
        record = mistake["record"]
        gt = mistake["ground_truth"]
        pred = mistake["prediction"]
        
        # Create descriptive folder name
        seq_name = record.sequence_path.name
        folder_name = f"mistake_{mistake_idx+1:02d}_{seq_name}_gt{gt['jersey_str']}_pred{pred['jersey_str']}"
        mistake_dir = mistakes_dir / folder_name
        mistake_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all frames
        for frame_idx, frame_path in enumerate(record.frame_paths):
            if frame_path.exists():
                dest_path = mistake_dir / f"frame_{frame_idx:03d}_{frame_path.name}"
                shutil.copy2(frame_path, dest_path)
        
        # Create info file
        info_file = mistake_dir / "info.txt"
        with open(info_file, 'w') as f:
            f.write(f"Mistake #{mistake_idx + 1}\n")
            f.write(f"{'='*60}\n\n")
            f.write(f"Model: {best_model_type}\n")
            f.write(f"Model Accuracy: {best_acc:.4f}\n")
            f.write(f"Seed: {best_seed}\n\n")
            f.write(f"Ground Truth:\n")
            f.write(f"  Jersey Number: {gt['jersey_str']}\n")
            f.write(f"  Tens Digit: {gt['tens']} ({'blank' if gt['tens'] == 10 else str(gt['tens'])})\n")
            f.write(f"  Ones Digit: {gt['ones']}\n")
            f.write(f"  Full Number: {gt['full_number']}\n\n")
            f.write(f"Prediction:\n")
            f.write(f"  Jersey Number: {pred['jersey_str']}\n")
            f.write(f"  Tens Digit: {pred['tens']} ({'blank' if pred['tens'] == 10 else str(pred['tens'])})\n")
            f.write(f"  Ones Digit: {pred['ones']}\n")
            f.write(f"  Full Number: {pred['full_number']}\n\n")
            f.write(f"Probabilities:\n")
            f.write(f"  Tens - Predicted ({pred['tens']}): {mistake['probabilities']['tens']['predicted_class']:.4f}\n")
            f.write(f"  Tens - Ground Truth ({gt['tens']}): {mistake['probabilities']['tens']['ground_truth_class']:.4f}\n")
            f.write(f"  Ones - Predicted ({pred['ones']}): {mistake['probabilities']['ones']['predicted_class']:.4f}\n")
            f.write(f"  Ones - Ground Truth ({gt['ones']}): {mistake['probabilities']['ones']['ground_truth_class']:.4f}\n\n")
            f.write(f"Sequence Info:\n")
            f.write(f"  Path: {mistake['sequence_path']}\n")
            f.write(f"  Track ID: {mistake['track_id']}\n")
            f.write(f"  Number of Frames: {mistake['num_frames']}\n")
    
    print(f"\n{'='*60}")
    print("EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Extracted {len(mistakes)} mistake sequences")
    print(f"Output directory: {output_dir}")
    print(f"Summary file: {summary_file}")
    print(f"Sequences copied to: {mistakes_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    config = Config()
    extract_mistakes(config, num_mistakes=10)
