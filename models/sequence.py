"""
Phase A: Sequence baseline models with RNN variants.
All models use ResNet18 backbone, hidden_dim=128, single layer.
Bidirectional models use Mean-Pool, unidirectional models use Final State.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict

from config import Config
from models.common import encode_frames, mean_pool, final_state


class CNNEncoder(nn.Module):
    """
    CNN backbone encoder that takes (B, 3, H, W) and returns (B, F) features.
    Used by all sequence and attention models.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        if config.backbone == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif config.backbone == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        else:
            raise NotImplementedError(f"Backbone {config.backbone} not implemented")
        
        self.backbone = backbone
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feats: (B, F)
        """
        return self.backbone(x)


class BaseSequenceModel(nn.Module):
    """
    Base class for sequence models with common forward logic.
    Subclasses specify RNN type and pooling strategy.
    """
    
    def __init__(self, config: Config, rnn_factory, bidirectional: bool):
        """
        Args:
            config: Config object
            rnn_factory: Function that takes (input_size, hidden_dim) and returns RNN module
            bidirectional: Whether RNN is bidirectional
        """
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.hidden_dim = config.seq_hidden_dim
        self.rnn = rnn_factory(self.encoder.feature_dim, self.hidden_dim)
        self.seq_feat_dim = self.hidden_dim * (2 if bidirectional else 1)
        
        # Classification heads
        self.fc_tens = nn.Linear(self.seq_feat_dim, 11)  # 0..9 + blank
        self.fc_ones = nn.Linear(self.seq_feat_dim, 10)  # 0..9
    
    def forward(self, frames: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            frames: (B, T, 3, H, W)
            lengths: (B,) actual sequence lengths before padding
        Returns:
            {
              "tens_logits": (B, 11),
              "ones_logits": (B, 10),
            }
        """
        # Encode frames
        feats = encode_frames(self.encoder, frames)  # (B, T, F)
        
        # Pack → RNN → unpack
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )
        
        # Aggregate: mean-pool for bidirectional, final state for unidirectional
        if self.seq_feat_dim == self.hidden_dim * 2:  # Bidirectional
            seq_feat = mean_pool(output, lengths)
        else:  # Unidirectional
            seq_feat = final_state(h_n)
        
        # Classification
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }


class SeqModelBRNN(BaseSequenceModel):
    """
    Phase A - P1-A: SEQ-BRNN-R18-H128-L1-MP
    Bidirectional Vanilla RNN with Mean-Pool aggregation.
    """
    
    def __init__(self, config: Config):
        def make_rnn(input_size, hidden_dim):
            return nn.RNN(input_size, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        super().__init__(config, make_rnn, bidirectional=True)


class SeqModelURNN(BaseSequenceModel):
    """
    Phase A - P1-B: SEQ-URNN-R18-H128-L1-FS
    Unidirectional Vanilla RNN with Final State aggregation.
    """
    
    def __init__(self, config: Config):
        def make_rnn(input_size, hidden_dim):
            return nn.RNN(input_size, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        super().__init__(config, make_rnn, bidirectional=False)


class SeqModelBGRU(BaseSequenceModel):
    """
    Phase A - P1-C: SEQ-BGRU-R18-H128-L1-MP
    Bidirectional GRU with Mean-Pool aggregation.
    """
    
    def __init__(self, config: Config):
        def make_rnn(input_size, hidden_dim):
            return nn.GRU(input_size, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        super().__init__(config, make_rnn, bidirectional=True)


class SeqModelUGRU(BaseSequenceModel):
    """
    Phase A - P1-D: SEQ-UGRU-R18-H128-L1-FS
    Unidirectional GRU with Final State aggregation.
    """
    
    def __init__(self, config: Config):
        def make_rnn(input_size, hidden_dim):
            return nn.GRU(input_size, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        super().__init__(config, make_rnn, bidirectional=False)


class SeqModelBLSTM(BaseSequenceModel):
    """
    Phase A - P1-E: SEQ-BLSTM-R18-H128-L1-MP
    Bidirectional LSTM with Mean-Pool aggregation.
    """
    
    def __init__(self, config: Config):
        def make_rnn(input_size, hidden_dim):
            return nn.LSTM(input_size, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        super().__init__(config, make_rnn, bidirectional=True)
    
    def forward(self, frames: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Override to handle LSTM's (h_n, c_n) tuple."""
        feats = encode_frames(self.encoder, frames)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, (h_n, c_n) = self.rnn(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )
        
        seq_feat = mean_pool(output, lengths)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }


class SeqModelULSTM(BaseSequenceModel):
    """
    Phase A - P1-F: SEQ-ULSTM-R18-H128-L1-FS
    Unidirectional LSTM with Final State aggregation.
    """
    
    def __init__(self, config: Config):
        def make_rnn(input_size, hidden_dim):
            return nn.LSTM(input_size, hidden_dim, num_layers=1, bidirectional=False, batch_first=True)
        super().__init__(config, make_rnn, bidirectional=False)
    
    def forward(self, frames: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Override to handle LSTM's (h_n, c_n) tuple."""
        feats = encode_frames(self.encoder, frames)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, c_n) = self.rnn(packed)
        seq_feat = final_state(h_n)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }
