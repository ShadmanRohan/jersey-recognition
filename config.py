"""
Configuration file for jersey number recognition project.
Centralizes all paths, hyperparameters, and model settings.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass
class Config:
    # paths
    data_root: str = "/home/rohan/Desktop/Acme2/temporal_jersey_nr_recognition_dataset_subset"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
    plot_dir: str = "outputs/plots"

    # image / sequence
    img_height: int = 192
    img_width: int = 96
    max_seq_len: int = 16  # T_MAX
    use_anchor_always: bool = True

    # training
    batch_size: int = 64  # Optimized for RTX 3090 (24GB VRAM) - adjust if OOM occurs
    num_workers: int = 15  # Data loading workers
    max_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0  # Gradient clipping value
    warmup_epochs: int = 1  # Warmup epochs for learning rate
    use_mixed_precision: bool = True  # Enable mixed precision training (AMP)
    early_stopping_patience: int = 30  # Stop training if val loss doesn't improve for N epochs (set to max_epochs to disable effectively)
    seed: int = 42
    
    # Discriminative learning rates (for future experiments)
    use_discriminative_lr: bool = False  # Enable different LR for backbone/temporal/heads
    lr_backbone: float = 1e-4  # LR for CNN backbone (gentle fine-tuning)
    lr_temporal: float = 3e-4  # LR for temporal layers (GRU/LSTM)
    lr_heads: float = 3e-4  # LR for classification heads
    scheduler_type: str = "cosine"  # "cosine" or "onecycle" - scheduler type

    # model settings
    backbone: str = "resnet18"
    seq_hidden_dim: int = 128

    # jersey classes present in dataset (full numbers)
    classes: List[str] = None

    # Loss weights: (tens, ones)
    loss_weights: Tuple[float, float] = (1.0, 1.0)

    # Train/Val/Test split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    def __post_init__(self):
        if self.classes is None:
            self.classes = ["4", "6", "8", "9", "48", "49", "64", "66", "88", "89"]


def get_class_mapping(config: Config) -> Tuple[dict, dict]:
    """
    Returns:
        str2idx: dict[str, int] mapping jersey string -> class index (0..len-1)
        idx2str: dict[int, str] inverse mapping
    """
    str2idx = {s: i for i, s in enumerate(config.classes)}
    idx2str = {i: s for s, i in str2idx.items()}
    return str2idx, idx2str

