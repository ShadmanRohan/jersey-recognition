"""
Model architectures for jersey number recognition.
Shared utilities: multitask_loss and build_model.
Model implementations are in separate files:
- basic.py: Phase 0 basic single-frame models
- sequence.py: Phase A sequence baseline models (contains CNNEncoder)
- attention.py: Phase B attention models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from config import Config


def multitask_loss(
    outputs: Dict[str, torch.Tensor],
    tens_labels: torch.Tensor,
    ones_labels: torch.Tensor,
    weights: Tuple[float, float] = (1.0, 1.0),
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Returns:
        total_loss: scalar tensor
        loss_dict: {
            "loss_tens": float,
            "loss_ones": float,
        }
    """
    w_tens, w_ones = weights
    
    loss_tens = F.cross_entropy(outputs["tens_logits"], tens_labels)
    loss_ones = F.cross_entropy(outputs["ones_logits"], ones_labels)
    
    total = w_tens * loss_tens + w_ones * loss_ones
    
    loss_dict = {
        "loss_tens": loss_tens.detach().item(),
        "loss_ones": loss_ones.detach().item(),
    }
    return total, loss_dict


def build_model(model_type: str, config: Config, backbone_name: str = None) -> nn.Module:
    """
    Build model based on model_type.
    
    Phase 0 (Basic models):
        - basic_r18, basic_effb0, basic_effl0, basic_mv3l, basic_mv3s, basic_sv2
    
    Phase A (Sequence baselines):
        - seq_brnn_mp, seq_urnn_fs, seq_bgru_mp, seq_ugru_fs, seq_blstm_mp, seq_ulstm_fs
    
    Phase B (Attention models):
        - attn_bgru_bahdanau, attn_bgru_luong, attn_bgru_gate, attn_bgru_hc
        - attn_ugru_gate, attn_ugru_hc
    
    Legacy model types (for backward compatibility):
        - seq -> seq_bgru_mp
        - seq_attn -> attn_bgru_bahdanau
        - seq_bgru_bahdanau -> attn_bgru_bahdanau (old Phase B naming)
        - seq_bgru_luong -> attn_bgru_luong
        - seq_bgru_gate -> attn_bgru_gate
        - seq_bgru_hc -> attn_bgru_hc
        - seq_ugru_gate -> attn_ugru_gate
        - seq_ugru_hc -> attn_ugru_hc
        - seq_uni -> seq_ugru_fs
        - seq_bilstm -> seq_blstm_mp
        - anchor -> basic_r18 (if no backbone specified) [legacy]
        - basic -> basic_r18 (if no backbone specified)
    
    Args:
        model_type: Model identifier (see above)
        config: Config object
        backbone_name: For basic models, specify backbone (e.g., "resnet18", "efficientnet_b0")
    """
    # Legacy model types (backward compatibility) - check first
    if model_type == "seq":
        from models.sequence import SeqModelBGRU
        return SeqModelBGRU(config)
    elif model_type == "seq_attn":
        from models.attention import SeqModelBGRU_Bahdanau
        return SeqModelBGRU_Bahdanau(config)
    # Legacy Phase B model names (backward compatibility)
    elif model_type in ["seq_bgru_bahdanau", "seq_bgru_luong", "seq_bgru_gate", "seq_bgru_hc", 
                        "seq_ugru_gate", "seq_ugru_hc"]:
        # Map old names to new names
        old_to_new = {
            "seq_bgru_bahdanau": "attn_bgru_bahdanau",
            "seq_bgru_luong": "attn_bgru_luong",
            "seq_bgru_gate": "attn_bgru_gate",
            "seq_bgru_hc": "attn_bgru_hc",
            "seq_ugru_gate": "attn_ugru_gate",
            "seq_ugru_hc": "attn_ugru_hc",
        }
        return build_model(old_to_new[model_type], config, backbone_name)
    elif model_type == "seq_uni":
        from models.sequence import SeqModelUGRU
        return SeqModelUGRU(config)
    elif model_type == "seq_bilstm":
        from models.sequence import SeqModelBLSTM
        return SeqModelBLSTM(config)
    elif model_type in ["anchor", "basic"]:  # Support both legacy and new names
        from models.basic import build_basic_model, BACKBONE_ALIASES
        if backbone_name:
            backbone = BACKBONE_ALIASES.get(backbone_name.lower(), backbone_name.lower())
            return build_basic_model(backbone, config)
        else:
            # Default to resnet18 for backward compatibility
            return build_basic_model("resnet18", config)
    
    # Phase 0: Basic models (new naming)
    elif model_type.startswith("basic_"):
        from models.basic import build_basic_model, BACKBONE_ALIASES
        
        # Map CLI names to backbone names
        basic_map = {
            "basic_r18": "resnet18",
            "basic_effb0": "efficientnet_b0",
            "basic_effl0": "efficientnet_lite0",
            "basic_mv3l": "mobilenet_v3_large",
            "basic_mv3s": "mobilenet_v3_small",
            "basic_sv2": "shufflenet_v2_x1_0",
        }
        
        backbone = basic_map.get(model_type)
        if backbone:
            return build_basic_model(backbone, config)
        elif backbone_name:
            backbone = BACKBONE_ALIASES.get(backbone_name.lower(), backbone_name.lower())
            return build_basic_model(backbone, config)
        else:
            raise ValueError(f"Unknown basic model: {model_type}. Use one of: {list(basic_map.keys())}")
    
    # Phase 0: Legacy anchor models (backward compatibility)
    elif model_type.startswith("anchor_"):
        from models.basic import build_basic_model, BACKBONE_ALIASES
        
        # Map legacy CLI names to backbone names
        anchor_map = {
            "anchor_r18": "resnet18",
            "anchor_effb0": "efficientnet_b0",
            "anchor_effl0": "efficientnet_lite0",
            "anchor_mv3l": "mobilenet_v3_large",
            "anchor_mv3s": "mobilenet_v3_small",
            "anchor_sv2": "shufflenet_v2_x1_0",
        }
        
        backbone = anchor_map.get(model_type)
        if backbone:
            return build_basic_model(backbone, config)
        elif backbone_name:
            backbone = BACKBONE_ALIASES.get(backbone_name.lower(), backbone_name.lower())
            return build_basic_model(backbone, config)
        else:
            raise ValueError(f"Unknown anchor model: {model_type}. Use one of: {list(anchor_map.keys())}")
    
    # Phase A: Sequence baseline models
    elif model_type.startswith("seq_") and not any(x in model_type for x in ["bahdanau", "luong", "gate", "hc"]):
        from models.sequence import (
            SeqModelBRNN, SeqModelURNN, SeqModelBGRU, SeqModelUGRU,
            SeqModelBLSTM, SeqModelULSTM
        )
        
        seq_map = {
            "seq_brnn_mp": SeqModelBRNN,
            "seq_urnn_fs": SeqModelURNN,
            "seq_bgru_mp": SeqModelBGRU,
            "seq_ugru_fs": SeqModelUGRU,
            "seq_blstm_mp": SeqModelBLSTM,
            "seq_ulstm_fs": SeqModelULSTM,
        }
        
        model_class = seq_map.get(model_type)
        if model_class:
            return model_class(config)
        else:
            raise ValueError(f"Unknown sequence model: {model_type}. Use one of: {list(seq_map.keys())}")
    
    # Phase B: Attention models (new naming: attn_*)
    elif model_type.startswith("attn_"):
        from models.attention import (
            SeqModelBGRU_Bahdanau, SeqModelBGRU_Luong, SeqModelBGRU_Gate, SeqModelBGRU_HC,
            SeqModelUGRU_Gate, SeqModelUGRU_HC
        )
        
        attn_map = {
            "attn_bgru_bahdanau": SeqModelBGRU_Bahdanau,
            "attn_bgru_luong": SeqModelBGRU_Luong,
            "attn_bgru_gate": SeqModelBGRU_Gate,
            "attn_bgru_hc": SeqModelBGRU_HC,
            "attn_ugru_gate": SeqModelUGRU_Gate,
            "attn_ugru_hc": SeqModelUGRU_HC,
        }
        
        model_class = attn_map.get(model_type)
        if model_class:
            return model_class(config)
        else:
            raise ValueError(f"Unknown attention model: {model_type}. Use one of: {list(attn_map.keys())}")
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Phase 0: basic_r18, basic_effb0, basic_effl0, basic_mv3l, basic_mv3s, basic_sv2 (or legacy anchor_*). "
            f"Phase A: seq_brnn_mp, seq_urnn_fs, seq_bgru_mp, seq_ugru_fs, seq_blstm_mp, seq_ulstm_fs. "
            f"Phase B: attn_bgru_bahdanau, attn_bgru_luong, attn_bgru_gate, attn_bgru_hc, attn_ugru_gate, attn_ugru_hc."
        )

