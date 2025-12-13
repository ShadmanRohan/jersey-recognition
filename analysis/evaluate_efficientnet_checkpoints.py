"""
Evaluate EfficientNet-B0 checkpoints to get correct test metrics.
"""

import torch
from pathlib import Path
from config import Config
from data import build_dataloaders
from models import build_model
from trainer import evaluate
from utils import get_device, set_seed

def main():
    """Evaluate EfficientNet-B0 checkpoints."""
    config = Config()
    device = get_device()
    set_seed(config.seed)
    
    base_dir = Path(__file__).parent
    checkpoint_dir = base_dir / "outputs" / "checkpoints"
    
    results = {}
    
    # Check EfficientNet-B0 seq model
    seq_checkpoint = checkpoint_dir / "seq_efficientnet_b0_best.pth"
    if seq_checkpoint.exists():
        print("Evaluating Seq (EfficientNet-B0)...")
        config.backbone = "efficientnet_b0"
        train_loader, val_loader, test_loader = build_dataloaders(config, model_type="seq")
        model = build_model("seq", config, backbone_name=None).to(device)
        
        checkpoint = torch.load(seq_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        
        test_loss, test_metrics = evaluate(model, test_loader, device, "seq", config, phase="Testing")
        results["seq_efficientnet_b0"] = {
            "loss": test_loss,
            "acc_number": test_metrics["acc_number"],
            "acc_tens": test_metrics["acc_tens"],
            "acc_ones": test_metrics["acc_ones"],
            "acc_full": test_metrics["acc_full"],
        }
        print(f"Seq (EfficientNet-B0): Acc Number = {test_metrics['acc_number']:.4f}\n")
    
    
    # Save results
    import json
    output_file = base_dir / "outputs" / "efficientnet_b0_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    main()

