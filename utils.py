"""
Utility functions for device management, seeding, directory creation, and plotting.
"""

import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
from config import Config


def get_device():
    """Returns the device (cuda if available, else cpu)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_dirs(config: Config) -> None:
    """Create output/checkpoint/log/plot directories if they don't exist."""
    for d in [config.output_dir, config.checkpoint_dir, config.log_dir, config.plot_dir]:
        Path(d).mkdir(parents=True, exist_ok=True)


def plot_curves(history: Dict[str, List[float]], out_prefix: str, config: Config) -> None:
    """
    Plot training/validation loss and accuracy curves.

    Expected keys in history:
        'train_loss', 'val_loss'
        'train_acc_number', 'val_acc_number'  # or similar naming

    Saves:
        f"{config.plot_dir}/{out_prefix}_loss.png"
        f"{config.plot_dir}/{out_prefix}_acc_number.png"
    """
    ensure_dirs(config)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    
    # Plot loss curves
    if "train_loss" in history and "val_loss" in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_loss"], label="Train Loss", marker="o")
        plt.plot(epochs, history["val_loss"], label="Val Loss", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{config.plot_dir}/{out_prefix}_loss.png", dpi=150)
        plt.close()
    
    # Plot accuracy curves
    if "train_acc_number" in history and "val_acc_number" in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_acc_number"], label="Train Accuracy", marker="o")
        plt.plot(epochs, history["val_acc_number"], label="Val Accuracy", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy (Jersey Number)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{config.plot_dir}/{out_prefix}_acc_number.png", dpi=150)
        plt.close()
    
    # Plot individual digit accuracies if available
    if "val_acc_tens" in history and "val_acc_ones" in history:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history.get("train_acc_tens", []), label="Train Tens", marker="o", alpha=0.7)
        plt.plot(epochs, history.get("val_acc_tens", []), label="Val Tens", marker="s", alpha=0.7)
        plt.plot(epochs, history.get("train_acc_ones", []), label="Train Ones", marker="o", alpha=0.7)
        plt.plot(epochs, history.get("val_acc_ones", []), label="Val Ones", marker="s", alpha=0.7)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Tens and Ones Digit Accuracy")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{config.plot_dir}/{out_prefix}_digits.png", dpi=150)
        plt.close()


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Logger:
    """Simple logger that prints and writes to a file."""
    
    def __init__(self, log_file=None):
        self.log_file = log_file
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            # Clear existing log file
            with open(log_file, "w") as f:
                f.write("")
    
    def log(self, message):
        """Log a message to console and file."""
        print(message)
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

