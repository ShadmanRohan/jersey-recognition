"""
Deep dive analysis for a specific sequence to understand why models fail.

Analyzes:
- Frame-level predictions and confidence scores
- Attention weights (for attention models)
- Feature visualizations
- Prediction probabilities for each class
- Comparison between frame and seq model predictions
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F

from config import Config, get_class_mapping
from data import build_dataloaders, SequenceRecord, load_and_preprocess_image
from models import build_model
from utils import get_device, set_seed


def find_sequence_in_dataset(sequence_path: str, val_loader) -> Tuple[int, SequenceRecord]:
    """Find a sequence in the validation dataset by its path."""
    val_dataset = val_loader.dataset
    sequence_path_obj = Path(sequence_path)
    
    for idx, record in enumerate(val_dataset.records):
        if record.sequence_path == sequence_path_obj:
            return idx, record
    
    # Try to find by partial match
    seq_name = sequence_path_obj.name
    for idx, record in enumerate(val_dataset.records):
        if record.sequence_path.name == seq_name:
            return idx, record
    
    raise ValueError(f"Sequence not found: {sequence_path}")


def analyze_frame_predictions(model, frames_tensor, length_value, model_type: str, 
                              device: torch.device, config: Config) -> Dict:
    """Analyze predictions at frame level for a sequence."""
    model.eval()
    
    with torch.no_grad():
        frames = frames_tensor.unsqueeze(0).to(device)  # Add batch dimension (B=1, T, C, H, W)
        # Lengths must be 1D tensor on CPU for pack_padded_sequence
        if isinstance(length_value, int):
            lengths = torch.tensor([length_value], dtype=torch.long)
        else:
            lengths = torch.tensor([length_value], dtype=torch.long) if not isinstance(length_value, torch.Tensor) else length_value
        
        if model_type == "seq":
            outputs = model(frames, lengths)
            
            # Get sequence-level predictions
            seq_tens_logits = outputs["tens_logits"]
            seq_ones_logits = outputs["ones_logits"]
            seq_full_logits = outputs.get("full_logits", None)
            
            seq_tens_probs = F.softmax(seq_tens_logits, dim=-1)[0]
            seq_ones_probs = F.softmax(seq_ones_logits, dim=-1)[0]
            
            result = {
                "type": "sequence",
                "tens_logits": seq_tens_logits[0].cpu().numpy(),
                "ones_logits": seq_ones_logits[0].cpu().numpy(),
                "tens_probs": seq_tens_probs.cpu().numpy(),
                "ones_probs": seq_ones_probs.cpu().numpy(),
                "tens_pred": seq_tens_logits[0].argmax().item(),
                "ones_pred": seq_ones_logits[0].argmax().item(),
                "frame_level": None,  # Seq model doesn't give frame-level
            }
            
            if seq_full_logits is not None:
                result["full_logits"] = seq_full_logits[0].cpu().numpy()
                result["full_probs"] = F.softmax(seq_full_logits, dim=-1)[0].cpu().numpy()
                result["full_pred"] = seq_full_logits[0].argmax().item()
            
            return result
            
                "seq_tens_probs": F.softmax(seq_tens_logits, dim=-1)[0].cpu().numpy(),
                "seq_ones_probs": F.softmax(seq_ones_logits, dim=-1)[0].cpu().numpy(),
                "tens_pred": seq_tens_logits[0].argmax().item(),
                "ones_pred": seq_ones_logits[0].argmax().item(),
                "attention_weights": attention_weights[0].cpu().numpy() if attention_weights is not None else None,
                "frame_level": {
                    "tens": frame_tens_preds.cpu().numpy().tolist(),
                    "ones": frame_ones_preds.cpu().numpy().tolist(),
                }
            }
            
            return result
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


def print_prediction_analysis(analysis: Dict, model_name: str, gt_tens: int, gt_ones: int, 
                             idx2str: dict, record: SequenceRecord):
    """Print detailed analysis of predictions."""
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} Model Analysis")
    print(f"{'='*70}")
    print(f"Ground Truth: {gt_tens} {gt_ones} (Jersey: {gt_tens if gt_tens != 10 else ''}{gt_ones})")
    print(f"Sequence Path: {record.sequence_path}")
    print(f"Number of frames: {len(record.frame_paths)}")
    
    # Sequence-level prediction
    pred_tens = analysis["tens_pred"]
    pred_ones = analysis["ones_pred"]
    print(f"\nSequence-Level Prediction: {pred_tens} {pred_ones} (Jersey: {pred_tens if pred_tens != 10 else ''}{pred_ones})")
    
    # Confidence scores for top predictions
    if "seq_tens_probs" in analysis:
        tens_probs = analysis["seq_tens_probs"]
        ones_probs = analysis["seq_ones_probs"]
        
        print(f"\nTens Digit Probabilities (Top 3):")
        top_tens = np.argsort(tens_probs)[-3:][::-1]
        for idx in top_tens:
            label = "blank" if idx == 10 else str(idx)
            print(f"  {label}: {tens_probs[idx]:.4f}")
        
        print(f"\nOnes Digit Probabilities (Top 3):")
        top_ones = np.argsort(ones_probs)[-3:][::-1]
        for idx in top_ones:
            print(f"  {idx}: {ones_probs[idx]:.4f}")
    
    # Frame-level analysis (for frame models)
    if analysis["frame_level"] is not None:
        frame_tens = analysis["frame_level"]["tens"]
        frame_ones = analysis["frame_level"]["ones"]
        
        print(f"\nFrame-Level Predictions:")
        print(f"  Frame Tens: {frame_tens}")
        print(f"  Frame Ones: {frame_ones}")
        
        # Show which frames predict correctly
        correct_tens_frames = sum(1 for f in frame_tens if f == gt_tens)
        correct_ones_frames = sum(1 for f in frame_ones if f == gt_ones)
        print(f"\n  Frames with correct tens prediction: {correct_tens_frames}/{len(frame_tens)}")
        print(f"  Frames with correct ones prediction: {correct_ones_frames}/{len(frame_ones)}")
        
        # Attention weights (if available)
        if "attention_weights" in analysis and analysis["attention_weights"] is not None:
            attn_weights = analysis["attention_weights"]
            print(f"\nAttention Weights (importance per frame):")
            for i, weight in enumerate(attn_weights[:len(frame_tens)]):
                frame_pred = f"{frame_tens[i]}_{frame_ones[i]}"
                correct = "✓" if (frame_tens[i] == gt_tens and frame_ones[i] == gt_ones) else "✗"
                print(f"  Frame {i:2d}: {weight:.4f} - Pred: {frame_pred} {correct}")
    
    # Full number prediction if available
    if "full_pred" in analysis:
        full_pred = analysis["full_pred"]
        full_jersey = idx2str.get(full_pred, f"idx_{full_pred}")
        print(f"\nFull Jersey Number Prediction: {full_jersey} (index {full_pred})")


def deep_dive_sequence(sequence_path: str, config: Config = None, output_dir: Path = None):
    """Deep dive analysis for a specific sequence."""
    if config is None:
        config = Config()
    
    device = get_device()
    set_seed(config.seed)
    
    if output_dir is None:
        output_dir = Path(config.output_dir) / "deep_dive"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    checkpoint_dir = Path(config.checkpoint_dir)
    seq_path = checkpoint_dir / "best_seq.pt"
    
    if not seq_path.exists():
        print(f"⚠ Checkpoint not found. Looking for:")
        print(f"  Seq: {seq_path}")
        return
    
    print("Loading model...")
    model_seq = build_model("seq", config).to(device)
    
    checkpoint_seq = torch.load(seq_path, map_location=device)
    model_seq.load_state_dict(checkpoint_seq["model_state"])
    
    # Build validation dataloader to find the sequence
    print("Building validation dataloader...")
    _, val_loader, _ = build_dataloaders(config)
    
    # Find sequence
    try:
        idx, record = find_sequence_in_dataset(sequence_path, val_loader)
        print(f"Found sequence at index {idx}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Load the sequence data
    val_dataset = val_loader.dataset
    sample = val_dataset[idx]
    
    frames_tensor = sample["frames"]  # (T, 3, H, W)
    length = sample["length"]
    gt_tens = sample["tens_label"]
    gt_ones = sample["ones_label"]
    
    _, idx2str = get_class_mapping(config)
    
    # Analyze both models
    print("\n" + "="*70)
    print("ANALYZING SEQUENCE")
    print("="*70)
    
    analysis_seq = analyze_frame_predictions(model_seq, frames_tensor, length, 
                                             "seq", device, config)
    
    # Print analysis
    print_prediction_analysis(analysis_seq, "Seq", gt_tens, gt_ones, idx2str, record)
    
    # Save detailed results to JSON
    results = {
        "sequence_path": str(record.sequence_path),
        "ground_truth": {"tens": gt_tens, "ones": gt_ones},
        "seq_model": analysis_seq,
        "frame_paths": [str(p) for p in record.frame_paths],
    }
    
    output_file = output_dir / f"analysis_{record.sequence_path.name}.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x))
    
    print(f"\n{'='*70}")
    print(f"Detailed analysis saved to: {output_file}")
    print(f"{'='*70}")
    
    # Additional insights
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")
    
    print(f"\n  Recommendation: Check if frames show both digits clearly.")
    print(f"  If only one digit is visible, consider adjusting crop or preprocessing.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deep dive analysis for a specific sequence")
    parser.add_argument("--sequence_path", type=str, required=True,
                       help="Path to sequence folder or sequence name")
    args = parser.parse_args()
    
    config = Config()
    deep_dive_sequence(args.sequence_path, config)

