"""
Evaluation metrics for jersey number recognition.
Separates metric computation from training logic.
"""

import torch
from typing import Dict, Tuple


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int]:
    """
    Compute accuracy from logits and targets.
    
    Args:
        logits: (N, num_classes) tensor of logits
        targets: (N,) tensor of target class indices
        
    Returns:
        correct: number of correct predictions
        total: total number of samples
    """
    preds = logits.argmax(dim=-1)
    correct = (preds == targets).float().sum().item()
    total = targets.numel()
    return int(correct), int(total)


def compute_multitask_metrics(outputs: Dict, labels: Dict) -> Dict[str, float]:
    """
    Compute multi-task metrics for tens/ones/full predictions.
    
    Args:
        outputs: dict with keys:
            - "tens_logits": (N, 11) tensor
            - "ones_logits": (N, 10) tensor
            - "full_logits": (N, 10) tensor (optional)
        labels: dict with keys:
            - "tens": (N,) tensor
            - "ones": (N,) tensor
            - "full": (N,) tensor (optional)
    
    Returns:
        dict with metrics:
            - "acc_tens": accuracy for tens digit
            - "acc_ones": accuracy for ones digit
            - "acc_full": accuracy for full jersey (if available)
            - "acc_number": combined accuracy (both digits correct)
    """
    tens_logits = outputs["tens_logits"]
    ones_logits = outputs["ones_logits"]
    full_logits = outputs.get("full_logits", None)
    
    tens_labels = labels["tens"]
    ones_labels = labels["ones"]
    full_labels = labels.get("full", None)
    
    # Compute accuracies
    tens_correct, tens_total = accuracy_from_logits(tens_logits, tens_labels)
    ones_correct, ones_total = accuracy_from_logits(ones_logits, ones_labels)
    
    metrics = {
        "acc_tens": tens_correct / tens_total if tens_total > 0 else 0.0,
        "acc_ones": ones_correct / ones_total if ones_total > 0 else 0.0,
    }
    
    if full_logits is not None and full_labels is not None:
        full_correct, full_total = accuracy_from_logits(full_logits, full_labels)
        metrics["acc_full"] = full_correct / full_total if full_total > 0 else 0.0
    
    # Combined accuracy: both digits correct (i.e., the final jersey number is correct)
    tens_pred = tens_logits.argmax(dim=-1)
    ones_pred = ones_logits.argmax(dim=-1)
    combined_correct = ((tens_pred == tens_labels) & (ones_pred == ones_labels)).float().sum().item()
    metrics["acc_number"] = combined_correct / tens_total if tens_total > 0 else 0.0
    
    return metrics

