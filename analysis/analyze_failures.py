"""
Post-training analysis: Find sequences where models fail and copy examples to a folder.

Analyzes validation set predictions from frame and seq models to find:
1. Sequences where both models are wrong
2. Sequences where only frame model is wrong
3. Sequences where only seq model is wrong

Copies 5-10 random examples from each category to an output folder.
"""

import torch
import random
from pathlib import Path
import shutil
from typing import Dict, List, Tuple
from collections import defaultdict

from config import Config, get_class_mapping
from data import build_dataloaders, SequenceRecord, JerseySequenceDataset
from models import build_model
from utils import get_device, set_seed


def load_checkpoint(model_path: Path, model_type: str, config: Config):
    """Load model from checkpoint."""
    checkpoint = torch.load(model_path, map_location="cpu")
    model = build_model(model_type, config)
    model.load_state_dict(checkpoint["model_state"])
    return model


def predict_batch(model, batch, device, model_type: str, config: Config):
    """Run inference on a batch and return predictions."""
    model.eval()
    with torch.no_grad():
        frames = batch["frames"].to(device)
        lengths = batch["lengths"].to(device)
        
        if model_type == "seq":
            outputs = model(frames, lengths)
        elif model_type == "anchor":
            images = batch["image"].to(device)
            outputs = model(images)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Get predictions
        tens_pred = outputs["tens_logits"].argmax(dim=-1).cpu()
        ones_pred = outputs["ones_logits"].argmax(dim=-1).cpu()
        
        return tens_pred, ones_pred


def analyze_failures(config: Config, output_dir: Path = None, num_examples: int = 10):
    """Analyze validation set failures and copy examples to folder."""
    device = get_device()
    set_seed(config.seed)
    
    if output_dir is None:
        output_dir = Path(config.output_dir) / "failure_analysis"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoints
    checkpoint_dir = Path(config.checkpoint_dir)
    frame_path = checkpoint_dir / "best_frame.pt"
    seq_path = checkpoint_dir / "best_seq.pt"
    
    if not frame_path.exists():
        print(f"⚠ Frame model checkpoint not found: {frame_path}")
        return
    
    if not seq_path.exists():
        print(f"⚠ Seq model checkpoint not found: {seq_path}")
        return
    
    print(f"Loading models...")
    model_frame = load_checkpoint(frame_path, "frame", config).to(device)
    model_seq = load_checkpoint(seq_path, "seq", config).to(device)
    
    # Build validation dataloader
    print(f"Building validation dataloader...")
    _, val_loader, _ = build_dataloaders(config)
    
    # Get dataset and records for accessing sequence paths
    val_dataset = val_loader.dataset
    val_records = val_dataset.records
    
    print(f"Running inference on {len(val_records)} validation sequences...")
    
    # Track predictions and ground truth
    predictions_frame = []
    predictions_seq = []
    ground_truths = []
    record_indices = []
    
    # Run inference
    for batch_idx, batch in enumerate(val_loader):
        batch_start_idx = batch_idx * config.batch_size
        
        # Get ground truth
        tens_label = batch["tens_label"]
        ones_label = batch["ones_label"]
        
        # Predictions
        tens_pred_frame, ones_pred_frame = predict_batch(model_frame, batch, device, "frame", config)
        tens_pred_seq, ones_pred_seq = predict_batch(model_seq, batch, device, "seq", config)
        
        # Check correctness for each sample in batch
        for i in range(len(tens_label)):
            idx = batch_start_idx + i
            if idx >= len(val_records):
                break
            
            gt_tens = tens_label[i].item()
            gt_ones = ones_label[i].item()
            pred_tens_frame = tens_pred_frame[i].item()
            pred_ones_frame = ones_pred_frame[i].item()
            pred_tens_seq = tens_pred_seq[i].item()
            pred_ones_seq = ones_pred_seq[i].item()
            
            # Check if prediction is correct (both tens and ones must match)
            correct_frame = (pred_tens_frame == gt_tens) and (pred_ones_frame == gt_ones)
            correct_seq = (pred_tens_seq == gt_tens) and (pred_ones_seq == gt_ones)
            
            predictions_frame.append(correct_frame)
            predictions_seq.append(correct_seq)
            ground_truths.append((gt_tens, gt_ones))
            record_indices.append(idx)
    
    # Categorize failures
    both_wrong = []
    only_frame_wrong = []
    only_seq_wrong = []
    
    for idx, (correct_frame, correct_seq) in enumerate(zip(predictions_frame, predictions_seq)):
        record = val_records[record_indices[idx]]
        gt_tens, gt_ones = ground_truths[idx]
        
        # Get predicted jersey numbers for display
        tens_pred_frame = predictions_frame[idx]
        # We need to track the actual predictions, let me fix this
        # Actually, we need to recompute or store predictions properly
        pass
    
    # Let me fix this by running inference again but storing more info
    both_wrong = []
    only_frame_wrong = []
    only_seq_wrong = []
    
    batch_idx_global = 0
    for batch_idx, batch in enumerate(val_loader):
        batch_start_idx = batch_idx * config.batch_size
        
        tens_label = batch["tens_label"]
        ones_label = batch["ones_label"]
        tens_pred_frame, ones_pred_frame = predict_batch(model_frame, batch, device, "frame", config)
        tens_pred_seq, ones_pred_seq = predict_batch(model_seq, batch, device, "seq", config)
        
        for i in range(len(tens_label)):
            idx = batch_start_idx + i
            if idx >= len(val_records):
                break
            
            record = val_records[idx]
            gt_tens = tens_label[i].item()
            gt_ones = ones_label[i].item()
            pred_tens_frame = tens_pred_frame[i].item()
            pred_ones_frame = ones_pred_frame[i].item()
            pred_tens_seq = tens_pred_seq[i].item()
            pred_ones_seq = ones_pred_seq[i].item()
            
            correct_frame = (pred_tens_frame == gt_tens) and (pred_ones_frame == gt_ones)
            correct_seq = (pred_tens_seq == gt_tens) and (pred_ones_seq == gt_ones)
            
            info = {
                "record": record,
                "idx": idx,
                "gt_tens": gt_tens,
                "gt_ones": gt_ones,
                "pred_tens_frame": pred_tens_frame,
                "pred_ones_frame": pred_ones_frame,
                "pred_tens_seq": pred_tens_seq,
                "pred_ones_seq": pred_ones_seq,
            }
            
            if not correct_frame and not correct_seq:
                both_wrong.append(info)
            elif not correct_frame and correct_seq:
                only_frame_wrong.append(info)
            elif correct_frame and not correct_seq:
                only_seq_wrong.append(info)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Failure Analysis Results")
    print(f"{'='*60}")
    print(f"Total validation sequences: {len(val_records)}")
    print(f"Both models wrong: {len(both_wrong)}")
    print(f"Only frame model wrong: {len(only_frame_wrong)}")
    print(f"Only seq model wrong: {len(only_seq_wrong)}")
    
    # Sample and copy examples
    def copy_sequence(record: SequenceRecord, dest_dir: Path, suffix: str = ""):
        """Copy all frames from a sequence to destination directory."""
        seq_name = record.sequence_path.name
        dest_seq_dir = dest_dir / f"{seq_name}{suffix}"
        dest_seq_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all frames
        for frame_path in record.frame_paths:
            if frame_path.exists():
                dest_path = dest_seq_dir / frame_path.name
                shutil.copy2(frame_path, dest_path)
        
        return dest_seq_dir
    
    # Copy examples from each category
    categories = [
        ("both_wrong", both_wrong, num_examples),
        ("only_frame_wrong", only_frame_wrong, num_examples),
        ("only_seq_wrong", only_seq_wrong, num_examples),
    ]
    
    _, idx2str = get_class_mapping(config)
    
    for cat_name, examples, max_count in categories:
        if len(examples) == 0:
            print(f"\n⚠ No examples found for category: {cat_name}")
            continue
        
        sampled = random.sample(examples, min(max_count, len(examples)))
        cat_dir = output_dir / cat_name
        cat_dir.mkdir(exist_ok=True)
        
        print(f"\nCopying {len(sampled)} examples to {cat_dir}...")
        
        for ex in sampled:
            record = ex["record"]
            
            # Create a descriptive name with predictions
            gt_str = f"gt_{ex['gt_tens']}_{ex['gt_ones']}"
            pred_frame_str = f"frame_{ex['pred_tens_frame']}_{ex['pred_ones_frame']}"
            pred_seq_str = f"seq_{ex['pred_tens_seq']}_{ex['pred_ones_seq']}"
            
            seq_name = record.sequence_path.name
            dest_seq_dir = cat_dir / f"{seq_name}_{gt_str}_{pred_frame_str}_{pred_seq_str}"
            dest_seq_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy all frames
            for frame_path in record.frame_paths:
                if frame_path.exists():
                    dest_path = dest_seq_dir / frame_path.name
                    shutil.copy2(frame_path, dest_path)
            
            # Create a text file with info
            info_file = dest_seq_dir / "info.txt"
            with open(info_file, "w") as f:
                f.write(f"Ground Truth: {ex['gt_tens']} {ex['gt_ones']}\n")
                f.write(f"Frame Model Prediction: {ex['pred_tens_frame']} {ex['pred_ones_frame']}\n")
                f.write(f"Seq Model Prediction: {ex['pred_tens_seq']} {ex['pred_ones_seq']}\n")
                f.write(f"Sequence Path: {record.sequence_path}\n")
                f.write(f"Track ID: {record.track_id}\n")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Examples copied to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    config = Config()
    analyze_failures(config, num_examples=10)

