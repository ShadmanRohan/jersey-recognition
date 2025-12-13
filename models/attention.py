"""
Phase B: Lightweight attention mechanisms on top of Phase A sequence models.
All models use ResNet18 backbone, hidden_dim=128, single layer.
Base models: BiGRU (for bidirectional) and UniGRU (for unidirectional).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import Config
from models.sequence import CNNEncoder


class SeqModelBGRU_Bahdanau(nn.Module):
    """
    Phase B - P2-A: SEQ-BGRU-R18-H128-L1-MP + Bahdanau
    Bidirectional GRU with Bahdanau (additive) attention.
    Refactored from original SeqModelAttn.
    """
    
    def __init__(self, config: Config, attention_dim: int = 128):
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.hidden_dim = config.seq_hidden_dim
        self.attention_dim = attention_dim
        self.gru = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.seq_feat_dim = self.hidden_dim * 2
        
        # Bahdanau attention: additive attention mechanism
        # Query: concatenated last forward and backward hidden states
        # Keys: all hidden states from GRU
        self.attn_query = nn.Linear(self.seq_feat_dim, attention_dim)  # Query projection
        self.attn_key = nn.Linear(self.seq_feat_dim, attention_dim)    # Key projection
        self.attn_score = nn.Linear(attention_dim, 1, bias=False)      # Score computation (v^T in Bahdanau)
        
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
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)               # (B*T, C, H, W)
        feats = self.encoder(x)                       # (B*T, F)
        feats = feats.view(B, T, -1)                  # (B, T, F)
        
        # Pack and pass through GRU to get all hidden states
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.gru(packed)         # h_n: (2, B, H) for bidirectional
        
        # Unpack to get all hidden states: (B, T, 2*H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )  # output: (B, T_max, 2*H)
        
        # Get query: concatenated last forward and backward hidden states
        h_fwd = h_n[0]  # (B, H) - forward direction
        h_bwd = h_n[1]  # (B, H) - backward direction
        query = torch.cat([h_fwd, h_bwd], dim=1)      # (B, 2*H)
        
        # Bahdanau attention: attention_score(query, key) = v^T * tanh(W_q * query + W_k * key)
        # Project query and all hidden states (keys)
        query_proj = self.attn_query(query)           # (B, attention_dim)
        query_proj = query_proj.unsqueeze(1)          # (B, 1, attention_dim)
        keys = self.attn_key(output)                  # (B, T, attention_dim)
        
        # Compute attention scores using additive attention
        # score = v^T * tanh(query + key)
        scores = self.attn_score(torch.tanh(query_proj + keys))  # (B, T, 1)
        scores = scores.squeeze(-1)                   # (B, T)
        
        # Mask out padding positions
        max_len = output.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(B, max_len) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (B, T)
        
        # Compute context vector as attention-weighted sum of hidden states
        attention_weights = attention_weights.unsqueeze(-1)  # (B, T, 1)
        seq_feat = (output * attention_weights).sum(dim=1)    # (B, 2*H)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }


class SeqModelBGRU_Luong(nn.Module):
    """
    Phase B - P2-B: SEQ-BGRU-R18-H128-L1-MP + Luong
    Bidirectional GRU with Luong (dot-product) attention.
    """
    
    def __init__(self, config: Config, attention_dim: int = 128):
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.hidden_dim = config.seq_hidden_dim
        self.attention_dim = attention_dim
        self.gru = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.seq_feat_dim = self.hidden_dim * 2
        
        # Luong attention: dot-product attention
        # Query: concatenated last forward and backward hidden states
        # Keys: all hidden states from GRU
        self.attn_query = nn.Linear(self.seq_feat_dim, attention_dim)  # Query projection
        self.attn_key = nn.Linear(self.seq_feat_dim, attention_dim)    # Key projection
        
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
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)               # (B*T, C, H, W)
        feats = self.encoder(x)                       # (B*T, F)
        feats = feats.view(B, T, -1)                  # (B, T, F)
        
        # Pack and pass through GRU to get all hidden states
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, h_n = self.gru(packed)         # h_n: (2, B, H) for bidirectional
        
        # Unpack to get all hidden states: (B, T, 2*H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )  # output: (B, T_max, 2*H)
        
        # Get query: concatenated last forward and backward hidden states
        h_fwd = h_n[0]  # (B, H) - forward direction
        h_bwd = h_n[1]  # (B, H) - backward direction
        query = torch.cat([h_fwd, h_bwd], dim=1)      # (B, 2*H)
        
        # Luong attention: dot-product attention
        # Project query and all hidden states (keys)
        query_proj = self.attn_query(query)           # (B, attention_dim)
        query_proj = query_proj.unsqueeze(1)          # (B, 1, attention_dim)
        keys = self.attn_key(output)                  # (B, T, attention_dim)
        
        # Compute attention scores using dot-product
        # score = query · key^T
        scores = torch.bmm(query_proj, keys.transpose(1, 2))  # (B, 1, T)
        scores = scores.squeeze(1)                    # (B, T)
        
        # Scale by sqrt of attention_dim (standard practice)
        scores = scores / (self.attention_dim ** 0.5)
        
        # Mask out padding positions
        max_len = output.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(B, max_len) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=1)  # (B, T)
        
        # Compute context vector as attention-weighted sum of hidden states
        attention_weights = attention_weights.unsqueeze(-1)  # (B, T, 1)
        seq_feat = (output * attention_weights).sum(dim=1)    # (B, 2*H)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }


class SeqModelBGRU_Gate(nn.Module):
    """
    Phase B - P2-C: SEQ-BGRU-R18-H128-L1-MP + Gate
    Bidirectional GRU with Gated temporal pooling.
    Small MLP outputs per-frame weights, then weighted sum.
    """
    
    def __init__(self, config: Config, gate_dim: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.hidden_dim = config.seq_hidden_dim
        self.gate_dim = gate_dim
        self.gru = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.seq_feat_dim = self.hidden_dim * 2
        
        # Gated pooling: MLP that outputs per-frame weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.seq_feat_dim, self.gate_dim),
            nn.ReLU(),
            nn.Linear(self.gate_dim, 1)
        )
        
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
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)               # (B*T, C, H, W)
        feats = self.encoder(x)                       # (B*T, F)
        feats = feats.view(B, T, -1)                  # (B, T, F)
        
        # Pack and pass through GRU to get all hidden states
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        
        # Unpack to get all hidden states: (B, T, 2*H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )  # output: (B, T_max, 2*H)
        
        # Compute gated weights for each frame
        gate_weights = self.gate_mlp(output)           # (B, T_max, 1)
        gate_weights = gate_weights.squeeze(-1)       # (B, T_max)
        
        # Mask out padding positions
        max_len = output.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(B, max_len) < lengths.unsqueeze(1)
        gate_weights = gate_weights.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get normalized weights
        gate_weights = F.softmax(gate_weights, dim=1)  # (B, T_max)
        
        # Compute weighted sum
        gate_weights = gate_weights.unsqueeze(-1)     # (B, T_max, 1)
        seq_feat = (output * gate_weights).sum(dim=1)  # (B, 2*H)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }


class SeqModelBGRU_HC(nn.Module):
    """
    Phase B - P2-D: SEQ-BGRU-R18-H128-L1-MP + HC
    Bidirectional GRU with Hard-Concrete (top-k) attention.
    Sparse attention mechanism that selects top-k frames.
    """
    
    def __init__(self, config: Config, top_k: int = 3, temperature: float = 0.1):
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.hidden_dim = config.seq_hidden_dim
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.gru = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )
        self.seq_feat_dim = self.hidden_dim * 2
        
        # Hard-Concrete: learnable scoring for top-k selection
        self.score_mlp = nn.Sequential(
            nn.Linear(self.seq_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Classification heads
        self.fc_tens = nn.Linear(self.seq_feat_dim, 11)  # 0..9 + blank
        self.fc_ones = nn.Linear(self.seq_feat_dim, 10)  # 0..9
    
    def hard_concrete_attention(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Hard-Concrete attention: selects top-k frames with learnable temperature.
        
        Args:
            scores: (B, T_max) raw scores
            lengths: (B,) actual sequence lengths
        Returns:
            attention_weights: (B, T_max) sparse attention weights
        """
        B, T_max = scores.shape
        
        # Mask out padding positions
        mask = torch.arange(T_max, device=lengths.device).expand(B, T_max) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Check for all -inf (shouldn't happen, but handle gracefully)
        # Replace any NaN or inf with -inf for masked positions
        scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, float('-inf')))
        
        # Apply temperature-scaled softmax
        # Clamp temperature to prevent numerical instability (must be positive)
        temp = torch.clamp(self.temperature, min=1e-3, max=10.0)
        logits = scores / temp
        
        # Check for NaN/Inf in logits before softmax
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Fallback: use uniform attention over valid positions
            attention_weights = mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            return attention_weights
        
        probs = F.softmax(logits, dim=1)  # (B, T_max)
        
        # Check for NaN in probs
        if torch.isnan(probs).any():
            # Fallback: use uniform attention over valid positions
            attention_weights = mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            return attention_weights
        
        # Hard-Concrete: select top-k and renormalize
        # Get top-k indices for each batch
        top_k_actual = torch.clamp(lengths, max=self.top_k)  # Don't select more than available
        k_to_use = min(self.top_k, T_max)
        top_k_values, top_k_indices = torch.topk(probs, k=k_to_use, dim=1)
        
        # Create sparse attention weights
        attention_weights = torch.zeros_like(probs)
        for b in range(B):
            k = int(top_k_actual[b].item())
            if k > 0:
                indices = top_k_indices[b, :k]
                values = top_k_values[b, :k]
                # Renormalize top-k values
                value_sum = values.sum()
                if value_sum > 1e-8:
                    values = values / value_sum
                    attention_weights[b, indices] = values
                else:
                    # Fallback: uniform over valid positions for this batch
                    valid_mask = mask[b].float()
                    if valid_mask.sum() > 0:
                        attention_weights[b] = valid_mask / valid_mask.sum()
        
        # Final check for NaN
        if torch.isnan(attention_weights).any():
            # Ultimate fallback: uniform attention
            attention_weights = mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return attention_weights
    
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
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)               # (B*T, C, H, W)
        feats = self.encoder(x)                       # (B*T, F)
        feats = feats.view(B, T, -1)                  # (B, T, F)
        
        # Pack and pass through GRU to get all hidden states
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        
        # Unpack to get all hidden states: (B, T, 2*H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )  # output: (B, T_max, 2*H)
        
        # Compute scores for Hard-Concrete attention
        scores = self.score_mlp(output).squeeze(-1)   # (B, T_max)
        
        # Apply Hard-Concrete attention
        attention_weights = self.hard_concrete_attention(scores, lengths)  # (B, T_max)
        
        # Compute weighted sum
        attention_weights = attention_weights.unsqueeze(-1)  # (B, T_max, 1)
        seq_feat = (output * attention_weights).sum(dim=1)    # (B, 2*H)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }


class SeqModelUGRU_Gate(nn.Module):
    """
    Phase B - P2-E: SEQ-UGRU-R18-H128-L1-FS + Gate
    Unidirectional GRU with Gated temporal pooling.
    Efficient causal variant.
    """
    
    def __init__(self, config: Config, gate_dim: int = 64):
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.hidden_dim = config.seq_hidden_dim
        self.gate_dim = gate_dim
        self.gru = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        self.seq_feat_dim = self.hidden_dim  # Unidirectional
        
        # Gated pooling: MLP that outputs per-frame weights
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.seq_feat_dim, self.gate_dim),
            nn.ReLU(),
            nn.Linear(self.gate_dim, 1)
        )
        
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
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)               # (B*T, C, H, W)
        feats = self.encoder(x)                       # (B*T, F)
        feats = feats.view(B, T, -1)                  # (B, T, F)
        
        # Pack and pass through GRU to get all hidden states
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        
        # Unpack to get all hidden states: (B, T, H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )  # output: (B, T_max, H)
        
        # Compute gated weights for each frame
        gate_weights = self.gate_mlp(output)           # (B, T_max, 1)
        gate_weights = gate_weights.squeeze(-1)       # (B, T_max)
        
        # Mask out padding positions
        max_len = output.size(1)
        mask = torch.arange(max_len, device=lengths.device).expand(B, max_len) < lengths.unsqueeze(1)
        gate_weights = gate_weights.masked_fill(~mask, float('-inf'))
        
        # Apply softmax to get normalized weights
        gate_weights = F.softmax(gate_weights, dim=1)  # (B, T_max)
        
        # Compute weighted sum
        gate_weights = gate_weights.unsqueeze(-1)     # (B, T_max, 1)
        seq_feat = (output * gate_weights).sum(dim=1)  # (B, H)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }


class SeqModelUGRU_HC(nn.Module):
    """
    Phase B - P2-F: SEQ-UGRU-R18-H128-L1-FS + HC
    Unidirectional GRU with Hard-Concrete (top-k) attention.
    Efficient causal variant with sparse attention.
    """
    
    def __init__(self, config: Config, top_k: int = 3, temperature: float = 0.1):
        super().__init__()
        self.encoder = CNNEncoder(config)
        self.hidden_dim = config.seq_hidden_dim
        self.top_k = top_k
        self.temperature = nn.Parameter(torch.tensor(temperature))
        self.gru = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        self.seq_feat_dim = self.hidden_dim  # Unidirectional
        
        # Hard-Concrete: learnable scoring for top-k selection
        self.score_mlp = nn.Sequential(
            nn.Linear(self.seq_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Classification heads
        self.fc_tens = nn.Linear(self.seq_feat_dim, 11)  # 0..9 + blank
        self.fc_ones = nn.Linear(self.seq_feat_dim, 10)  # 0..9
    
    def hard_concrete_attention(self, scores: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Hard-Concrete attention: selects top-k frames with learnable temperature.
        
        Args:
            scores: (B, T_max) raw scores
            lengths: (B,) actual sequence lengths
        Returns:
            attention_weights: (B, T_max) sparse attention weights
        """
        B, T_max = scores.shape
        
        # Mask out padding positions
        mask = torch.arange(T_max, device=lengths.device).expand(B, T_max) < lengths.unsqueeze(1)
        scores = scores.masked_fill(~mask, float('-inf'))
        
        # Check for all -inf (shouldn't happen, but handle gracefully)
        # Replace any NaN or inf with -inf for masked positions
        scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, float('-inf')))
        
        # Apply temperature-scaled softmax
        # Clamp temperature to prevent numerical instability (must be positive)
        temp = torch.clamp(self.temperature, min=1e-3, max=10.0)
        logits = scores / temp
        
        # Check for NaN/Inf in logits before softmax
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            # Fallback: use uniform attention over valid positions
            attention_weights = mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            return attention_weights
        
        probs = F.softmax(logits, dim=1)  # (B, T_max)
        
        # Check for NaN in probs
        if torch.isnan(probs).any():
            # Fallback: use uniform attention over valid positions
            attention_weights = mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
            return attention_weights
        
        # Hard-Concrete: select top-k and renormalize
        # Get top-k indices for each batch
        top_k_actual = torch.clamp(lengths, max=self.top_k)  # Don't select more than available
        k_to_use = min(self.top_k, T_max)
        top_k_values, top_k_indices = torch.topk(probs, k=k_to_use, dim=1)
        
        # Create sparse attention weights
        attention_weights = torch.zeros_like(probs)
        for b in range(B):
            k = int(top_k_actual[b].item())
            if k > 0:
                indices = top_k_indices[b, :k]
                values = top_k_values[b, :k]
                # Renormalize top-k values
                value_sum = values.sum()
                if value_sum > 1e-8:
                    values = values / value_sum
                    attention_weights[b, indices] = values
                else:
                    # Fallback: uniform over valid positions for this batch
                    valid_mask = mask[b].float()
                    if valid_mask.sum() > 0:
                        attention_weights[b] = valid_mask / valid_mask.sum()
        
        # Final check for NaN
        if torch.isnan(attention_weights).any():
            # Ultimate fallback: uniform attention
            attention_weights = mask.float()
            attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)
        
        return attention_weights
    
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
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)               # (B*T, C, H, W)
        feats = self.encoder(x)                       # (B*T, F)
        feats = feats.view(B, T, -1)                  # (B, T, F)
        
        # Pack and pass through GRU to get all hidden states
        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.gru(packed)
        
        # Unpack to get all hidden states: (B, T, H)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, padding_value=0.0
        )  # output: (B, T_max, H)
        
        # Compute scores for Hard-Concrete attention
        scores = self.score_mlp(output).squeeze(-1)   # (B, T_max)
        
        # Apply Hard-Concrete attention
        attention_weights = self.hard_concrete_attention(scores, lengths)  # (B, T_max)
        
        # Compute weighted sum
        attention_weights = attention_weights.unsqueeze(-1)  # (B, T_max, 1)
        seq_feat = (output * attention_weights).sum(dim=1)    # (B, H)
        
        tens_logits = self.fc_tens(seq_feat)
        ones_logits = self.fc_ones(seq_feat)
        
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }

