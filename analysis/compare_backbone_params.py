"""
Quick comparison of basic model backbones - parameter counts only.
No training required.
"""

import torch
from config import Config
from models.basic import build_basic_model
from utils import count_parameters

def compare_backbone_parameters():
    """Compare parameter counts for all backbone architectures."""
    config = Config()
    
    backbones = [
        ("ResNet18", "resnet18"),
        ("EfficientNet-B0", "efficientnet_b0"),
        ("MobileNetV3-Large", "mobilenet_v3_large"),
        ("MobileNetV3-Small", "mobilenet_v3_small"),
        ("ShuffleNetV2", "shufflenet_v2_x1_0"),
    ]
    
    print("="*80)
    print("BASIC MODEL BACKBONE PARAMETER COMPARISON")
    print("="*80)
    print()
    
    results = []
    
    for name, backbone_name in backbones:
        try:
            model = build_basic_model(backbone_name, config)
            param_count = count_parameters(model)
            feature_dim = model.feature_dim
            
            # Count backbone params
            backbone_params = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
            fc_params = param_count - backbone_params
            
            results.append({
                "name": name,
                "backbone": backbone_name,
                "total_params": param_count,
                "backbone_params": backbone_params,
                "fc_params": fc_params,
                "feature_dim": feature_dim
            })
            
            print(f"✅ {name:20s} | Total: {param_count:>10,} | Backbone: {backbone_params:>10,} | FC: {fc_params:>6,} | FeatDim: {feature_dim}")
        except Exception as e:
            print(f"❌ {name:20s} | Error: {e}")
            results.append({
                "name": name,
                "backbone": backbone_name,
                "error": str(e)
            })
    
    print()
    print("="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print()
    
    # Sort by parameter count
    successful = [r for r in results if "error" not in r]
    successful.sort(key=lambda x: x["total_params"])
    
    print(f"{'Backbone':<25} {'Total Params':<15} {'Backbone Params':<18} {'FC Params':<12} {'Feature Dim':<12}")
    print("-" * 80)
    for r in successful:
        print(f"{r['name']:<25} {r['total_params']:>14,} {r['backbone_params']:>17,} {r['fc_params']:>11,} {r['feature_dim']:>11}")
    
    if successful:
        smallest = successful[0]
        largest = successful[-1]
        print()
        print(f"Smallest: {smallest['name']} ({smallest['total_params']:,} parameters)")
        print(f"Largest:  {largest['name']} ({largest['total_params']:,} parameters)")
        print(f"Ratio:    {largest['total_params'] / smallest['total_params']:.2f}x difference")
    
    return results

if __name__ == "__main__":
    results = compare_backbone_parameters()



