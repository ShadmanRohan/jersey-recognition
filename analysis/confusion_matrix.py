"""
Generate confusion matrices for the best model.
Computes confusion matrices for ones digit, tens digit, and full number.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import json
import sys

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from data import build_dataloaders
from models import build_model
from utils import get_device, set_seed


def find_best_model_from_results(results_file: Path) -> tuple:
    """Find the best performing model from multi-seed results JSON file."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    best_model_type = None
    best_acc = 0.0
    
    for model_type, data in results.items():
        if "statistics" in data and "test_acc_number" in data["statistics"]:
            mean_acc = data["statistics"]["test_acc_number"]["mean"]
            # Also check individual runs for max
            if "individual_runs" in data:
                for run in data["individual_runs"]:
                    if run["test_acc_number"] > best_acc:
                        best_acc = run["test_acc_number"]
                        best_model_type = model_type
    
    # If no individual runs, use mean
    if best_model_type is None:
        for model_type, data in results.items():
            if "statistics" in data and "test_acc_number" in data["statistics"]:
                mean_acc = data["statistics"]["test_acc_number"]["mean"]
                if mean_acc > best_acc:
                    best_acc = mean_acc
                    best_model_type = model_type
    
    return best_model_type, best_acc


def collect_test_predictions(model, test_loader, device, model_type: str, config: Config):
    """Collect all predictions and ground truth labels from test set."""
    model.eval()
    
    all_tens_pred = []
    all_ones_pred = []
    all_tens_true = []
    all_ones_true = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting predictions"):
            is_basic_model = model_type in ["basic", "anchor"] or model_type.startswith("basic_") or model_type.startswith("anchor_")
            
            if is_basic_model:
                images = batch["image"].to(device)
                tens_label = batch["tens_label"].to(device)
                ones_label = batch["ones_label"].to(device)
                outputs = model(images)
            else:
                frames = batch["frames"].to(device)
                lengths = batch["lengths"].to(device)
                tens_label = batch["tens_label"].to(device)
                ones_label = batch["ones_label"].to(device)
                outputs = model(frames, lengths)
            
            tens_pred = outputs["tens_logits"].argmax(dim=-1).cpu().numpy()
            ones_pred = outputs["ones_logits"].argmax(dim=-1).cpu().numpy()
            
            all_tens_pred.extend(tens_pred)
            all_ones_pred.extend(ones_pred)
            all_tens_true.extend(tens_label.cpu().numpy())
            all_ones_true.extend(ones_label.cpu().numpy())
    
    return (
        np.array(all_tens_true),
        np.array(all_ones_true),
        np.array(all_tens_pred),
        np.array(all_ones_pred)
    )


def compute_full_number_labels(tens_true, ones_true, tens_pred, ones_pred):
    """
    Compute full number labels for confusion matrix.
    Full number = tens * 10 + ones, but if tens == 10 (blank), then it's just ones.
    Actually, for jersey numbers, if tens is blank, the number is just the ones digit.
    But for confusion matrix, we can represent 0-99.
    """
    # For ground truth: if tens == 10 (blank), full number is just ones (0-9)
    # Otherwise, full number is tens * 10 + ones
    full_true = np.where(tens_true == 10, ones_true, tens_true * 10 + ones_true)
    full_pred = np.where(tens_pred == 10, ones_pred, tens_pred * 10 + ones_pred)
    
    return full_true, full_pred


def plot_confusion_matrix(cm, labels, title, save_path, figsize=(10, 8)):
    """Plot and save confusion matrix."""
    plt.figure(figsize=figsize)
    
    # Normalize confusion matrix to percentages
    row_sums = cm.sum(axis=1)[:, np.newaxis]
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    cm_normalized = cm.astype('float') / row_sums * 100
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
    
    # Create heatmap
    if HAS_SEABORN:
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.1f',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Percentage (%)'}
        )
    else:
        # Use matplotlib imshow as fallback
        im = plt.imshow(cm_normalized, cmap='Blues', aspect='auto')
        plt.colorbar(im, label='Percentage (%)')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        
        # Add text annotations
        thresh = cm_normalized.max() / 2.
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = plt.text(j, i, f'{cm_normalized[i, j]:.1f}',
                              ha="center", va="center",
                              color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Also save raw counts version
    plt.figure(figsize=figsize)
    if HAS_SEABORN:
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Count'}
        )
    else:
        # Use matplotlib imshow as fallback
        im = plt.imshow(cm, cmap='Blues', aspect='auto')
        plt.colorbar(im, label='Count')
        plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
        plt.yticks(range(len(labels)), labels)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = plt.text(j, i, f'{cm[i, j]}',
                              ha="center", va="center",
                              color="white" if cm[i, j] > thresh else "black")
    
    plt.title(f"{title} (Raw Counts)", fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    save_path_counts = save_path.parent / f"{save_path.stem}_counts{save_path.suffix}"
    plt.savefig(save_path_counts, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function to generate confusion matrices."""
    # Setup
    config = Config()
    device = get_device()
    set_seed(config.seed)
    
    # Find best model from results
    results_file = Path(config.output_dir) / "phaseB_multi_seed_results.json"
    if not results_file.exists():
        print(f"❌ Results file not found: {results_file}")
        print("Please run phaseB_multi_seed.py first or specify model manually.")
        return
    
    best_model_type, best_acc = find_best_model_from_results(results_file)
    print(f"📊 Best model: {best_model_type}")
    print(f"   Test Accuracy: {best_acc:.4f}")
    
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
    
    # Collect predictions
    print("🔍 Collecting predictions from test set...")
    tens_true, ones_true, tens_pred, ones_pred = collect_test_predictions(
        model, test_loader, device, best_model_type, config
    )
    
    print(f"✅ Collected {len(tens_true)} predictions")
    
    # Compute confusion matrices
    print("📈 Computing confusion matrices...")
    
    # 1. Ones digit confusion matrix (0-9)
    ones_labels = [str(i) for i in range(10)]
    cm_ones = confusion_matrix(ones_true, ones_pred, labels=list(range(10)))
    
    # 2. Tens digit confusion matrix (0-10, where 10 is blank)
    tens_labels = [str(i) if i < 10 else "blank" for i in range(11)]
    cm_tens = confusion_matrix(tens_true, tens_pred, labels=list(range(11)))
    
    # 3. Full number confusion matrix
    full_true, full_pred = compute_full_number_labels(tens_true, ones_true, tens_pred, ones_pred)
    
    # Get unique labels that appear in the data (0-99, but only include those that appear)
    unique_labels = sorted(set(np.concatenate([full_true, full_pred])))
    full_labels = [str(label) for label in unique_labels]
    
    cm_full = confusion_matrix(full_true, full_pred, labels=unique_labels)
    
    # Save outputs
    output_dir = Path(config.output_dir) / "confusion_matrices"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving confusion matrices to: {output_dir}")
    
    # Plot and save
    plot_confusion_matrix(
        cm_ones,
        ones_labels,
        f"Ones Digit Confusion Matrix\n{best_model_type} (Acc: {best_acc:.4f})",
        output_dir / f"{best_model_type}_ones_cm.png",
        figsize=(10, 8)
    )
    
    plot_confusion_matrix(
        cm_tens,
        tens_labels,
        f"Tens Digit Confusion Matrix\n{best_model_type} (Acc: {best_acc:.4f})",
        output_dir / f"{best_model_type}_tens_cm.png",
        figsize=(12, 10)
    )
    
    # For full number, if there are too many classes, use a smaller figure or subset
    if len(unique_labels) > 50:
        print(f"⚠️  Full number has {len(unique_labels)} unique classes, creating subset visualization")
        # Only show classes that have at least some predictions
        row_sums = cm_full.sum(axis=1)
        col_sums = cm_full.sum(axis=0)
        active_indices = np.where((row_sums > 0) | (col_sums > 0))[0]
        active_labels = [unique_labels[i] for i in active_indices]
        cm_full_subset = cm_full[np.ix_(active_indices, active_indices)]
        plot_confusion_matrix(
            cm_full_subset,
            [str(unique_labels[i]) for i in active_indices],
            f"Full Number Confusion Matrix (Active Classes)\n{best_model_type} (Acc: {best_acc:.4f})",
            output_dir / f"{best_model_type}_full_cm.png",
            figsize=(14, 12)
        )
    else:
        plot_confusion_matrix(
            cm_full,
            full_labels,
            f"Full Number Confusion Matrix\n{best_model_type} (Acc: {best_acc:.4f})",
            output_dir / f"{best_model_type}_full_cm.png",
            figsize=(14, 12)
        )
    
    # Save raw confusion matrices as numpy arrays and JSON summary
    np.save(output_dir / f"{best_model_type}_ones_cm.npy", cm_ones)
    np.save(output_dir / f"{best_model_type}_tens_cm.npy", cm_tens)
    np.save(output_dir / f"{best_model_type}_full_cm.npy", cm_full)
    
    # Compute and save metrics
    from sklearn.metrics import accuracy_score, classification_report
    
    ones_acc = accuracy_score(ones_true, ones_pred)
    tens_acc = accuracy_score(tens_true, tens_pred)
    full_acc = accuracy_score(full_true, full_pred)
    
    # Convert numpy types to native Python types for JSON serialization
    def convert_to_native(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: convert_to_native(value) for key, value in obj.items()}
        return obj
    
    summary = {
        "model": best_model_type,
        "test_acc_from_results": float(best_acc),
        "confusion_matrix_accuracies": {
            "ones_digit": float(ones_acc),
            "tens_digit": float(tens_acc),
            "full_number": float(full_acc)
        },
        "num_samples": int(len(tens_true)),
        "ones_classes": list(range(10)),
        "tens_classes": list(range(11)),
        "full_number_classes": convert_to_native(unique_labels)
    }
    
    with open(output_dir / f"{best_model_type}_cm_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("CONFUSION MATRIX SUMMARY")
    print("="*60)
    print(f"Model: {best_model_type}")
    print(f"Test Accuracy (from results): {best_acc:.4f}")
    print(f"\nConfusion Matrix Accuracies:")
    print(f"  Ones Digit:   {ones_acc:.4f}")
    print(f"  Tens Digit:   {tens_acc:.4f}")
    print(f"  Full Number:  {full_acc:.4f}")
    print(f"\nOutput directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
