"""
Basic single-frame models with different backbones for comparison.
All models use the same architecture: backbone → feature vector → 2 FC heads.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict

from config import Config


class BasicModelWithBackbone(nn.Module):
    """
    Basic single-frame model with configurable backbone.
    Architecture: Backbone → Feature Vector → 2 FC Heads (tens, ones)
    """
    
    def __init__(self, config: Config, backbone_name: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone_name
        self.config = config
        
        # Build backbone
        backbone, feature_dim = self._build_backbone(backbone_name)
        self.backbone = backbone
        self.feature_dim = feature_dim
        
        # FC heads (same for all backbones)
        self.fc_tens = nn.Linear(self.feature_dim, 11)  # 0..9 + blank
        self.fc_ones = nn.Linear(self.feature_dim, 10)  # 0..9
    
    def _build_backbone(self, backbone_name: str):
        """Build backbone and return (backbone, feature_dim)."""
        
        if backbone_name == "resnet18":
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        elif backbone_name == "efficientnet_b0":
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
        elif backbone_name == "efficientnet_lite0":
            # EfficientNet-Lite0 is not available in torchvision
            # Using EfficientNet-B0 as the closest alternative (both are ~4M params, similar architecture)
            # Lite0 is edge-optimized but B0 provides similar performance characteristics
            print("Note: EfficientNet-Lite0 not in torchvision, using EfficientNet-B0 as equivalent alternative")
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = backbone.classifier[1].in_features
            backbone.classifier = nn.Identity()
            
        elif backbone_name == "mobilenet_v3_large":
            backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            # MobileNetV3 classifier: Linear(960->1280) -> Hardswish -> Dropout -> Linear(1280->1000)
            # We want the output of first Linear layer (1280-dim)
            feature_dim = backbone.classifier[0].out_features  # 1280
            # Replace classifier with: AdaptivePool -> Flatten -> First Linear -> Hardswish
            # This gives us 1280-dim features
            backbone.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                backbone.classifier[0],  # Linear(960->1280)
                backbone.classifier[1],  # Hardswish
            )
            
        elif backbone_name == "mobilenet_v3_small":
            backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
            # MobileNetV3 classifier: Linear(576->1024) -> Hardswish -> Dropout -> Linear(1024->1000)
            # We want the output of first Linear layer (1024-dim)
            feature_dim = backbone.classifier[0].out_features  # 1024
            # Replace classifier with: AdaptivePool -> Flatten -> First Linear -> Hardswish
            # This gives us 1024-dim features
            backbone.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                backbone.classifier[0],  # Linear(576->1024)
                backbone.classifier[1],  # Hardswish
            )
            
        elif backbone_name == "shufflenet_v2_x1_0":
            backbone = models.shufflenet_v2_x1_0(weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
            feature_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        return backbone, feature_dim
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 3, H, W) - batch of single-frame images
        Returns:
            {
              "tens_logits": (B, 11),
              "ones_logits": (B, 10),
            }
        """
        # For MobileNetV3, we need to extract features from .features first (spatial),
        # then pass through modified classifier (adaptive pooling + linear)
        # For other models, backbone directly gives feature vector
        if self.backbone_name in ["mobilenet_v3_large", "mobilenet_v3_small"]:
            spatial_feats = self.backbone.features(x)  # (B, C, H', W')
            feats = self.backbone.classifier(spatial_feats)  # (B, feature_dim)
        else:
            feats = self.backbone(x)  # (B, feature_dim)
        
        tens_logits = self.fc_tens(feats)
        ones_logits = self.fc_ones(feats)
        return {
            "tens_logits": tens_logits,
            "ones_logits": ones_logits,
        }
    
    def get_param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_basic_model(backbone_name: str, config: Config) -> BasicModelWithBackbone:
    """
    Build basic single-frame model with specified backbone.
    
    Args:
        backbone_name: One of:
            - "resnet18"
            - "efficientnet_b0"
            - "mobilenet_v3_large"
            - "mobilenet_v3_small"
            - "shufflenet_v2_x1_0"
        config: Config object
    """
    return BasicModelWithBackbone(config, backbone_name=backbone_name)


# Backbone name mapping for easier CLI usage
BACKBONE_ALIASES = {
    "resnet18": "resnet18",
    "efficientnet-b": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet-lite0": "efficientnet_lite0",
    "efficientnet_lite0": "efficientnet_lite0",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "mobilenet_v3_small": "mobilenet_v3_small",
    "shufflenet_v2": "shufflenet_v2_x1_0",
    "shufflenet_v2_x1_0": "shufflenet_v2_x1_0",
}

