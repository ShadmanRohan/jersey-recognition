"""
Common utilities and helper functions for sequence models.
Extracted to reduce code duplication.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def encode_frames(encoder: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """
    Encode a batch of frame sequences.
    
    Args:
        encoder: CNNEncoder instance
        frames: (B, T, 3, H, W) batch of frame sequences
    Returns:
        feats: (B, T, F) encoded features
    """
    B, T, C, H, W = frames.shape
    x = frames.view(B * T, C, H, W)  # (B*T, C, H, W)
    feats = encoder(x)                # (B*T, F)
    feats = feats.view(B, T, -1)      # (B, T, F)
    return feats


def create_length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    Create a boolean mask for valid (non-padded) positions.
    
    Args:
        lengths: (B,) actual sequence lengths
        max_len: Maximum sequence length
    Returns:
        mask: (B, max_len) boolean mask, True for valid positions
    """
    B = lengths.shape[0]
    mask = torch.arange(max_len, device=lengths.device).expand(B, max_len) < lengths.unsqueeze(1)
    return mask


def mean_pool(output: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool over time dimension, masking out padding.
    
    Args:
        output: (B, T_max, D) RNN output
        lengths: (B,) actual sequence lengths
    Returns:
        pooled: (B, D) mean-pooled features
    """
    B, T_max, D = output.shape
    mask = create_length_mask(lengths, T_max)
    mask = mask.unsqueeze(-1).float()  # (B, T_max, 1)
    masked_output = output * mask      # (B, T_max, D)
    pooled = masked_output.sum(dim=1) / lengths.unsqueeze(1).float()  # (B, D)
    return pooled


def final_state(hidden: torch.Tensor) -> torch.Tensor:
    """
    Extract final hidden state from RNN output.
    
    Args:
        hidden: (num_layers, B, H) or (num_layers*num_directions, B, H)
    Returns:
        final: (B, H) final hidden state
    """
    return hidden[-1]


def apply_attention_mask(scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """
    Mask attention scores for padding positions.
    
    Args:
        scores: (B, T_max) attention scores
        lengths: (B,) actual sequence lengths
    Returns:
        masked_scores: (B, T_max) masked scores (padding = -inf)
    """
    max_len = scores.size(1)
    mask = create_length_mask(lengths, max_len)
    return scores.masked_fill(~mask, float('-inf'))


def weighted_sum(output: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Compute weighted sum over time dimension.
    
    Args:
        output: (B, T_max, D) sequence features
        weights: (B, T_max) attention weights
    Returns:
        weighted: (B, D) weighted sum
    """
    weights = weights.unsqueeze(-1)  # (B, T_max, 1)
    return (output * weights).sum(dim=1)  # (B, D)

