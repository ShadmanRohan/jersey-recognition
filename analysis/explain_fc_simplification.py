"""
Explanation of T, FC, and potential simplifications.
"""

import torch
import torch.nn as nn

print("="*80)
print("EXPLANATION: T, FC, and Simplification")
print("="*80)

print("\n1. WHAT IS 'T'?")
print("-" * 80)
print("""
T = Sequence Length (number of frames per sequence)

Example:
  - Batch size B = 64 sequences
  - Sequence length T = 8 frames per sequence
  - Total frames = B × T = 64 × 8 = 512 frames

In the models:
  - Input shape: (B, T, 3, H, W) = (64, 8, 3, 192, 96)
  - This means: 64 sequences, each with 8 frames
""")

print("\n2. WHAT IS 'FC'?")
print("-" * 80)
print("""
FC = Fully Connected Layer (also called Linear Layer or Dense Layer)

In PyTorch: nn.Linear(input_dim, output_dim)

Current FC layers in the models:
  - fc_tens: Linear(512, 11)  → 512 inputs → 11 outputs (0-9 digits + blank)
  - fc_ones: Linear(512, 10)  → 512 inputs → 10 outputs (0-9 digits)
  - fc_full: Linear(512, 10)  → 512 inputs → 10 outputs (10 jersey classes)

The 512 comes from ResNet18's feature dimension.
""")

print("\n3. CURRENT FC COMPLEXITY")
print("-" * 80)

# Current setup
feature_dim = 512
num_tens_classes = 11
num_ones_classes = 10
num_full_classes = 10

current_params_tens = feature_dim * num_tens_classes + num_tens_classes  # weights + bias
current_params_ones = feature_dim * num_ones_classes + num_ones_classes
current_params_full = feature_dim * num_full_classes + num_full_classes
total_current = current_params_tens + current_params_ones + current_params_full

print(f"""
Current FC layers:
  - fc_tens:  Linear({feature_dim}, {num_tens_classes})  = {current_params_tens:,} parameters
  - fc_ones:  Linear({feature_dim}, {num_ones_classes})  = {current_params_ones:,} parameters
  - fc_full:  Linear({feature_dim}, {num_full_classes}) = {current_params_full:,} parameters
  - Total: {total_current:,} parameters

These are already VERY simple - just one matrix multiplication each!
""")

print("\n4. CAN FC BE SIMPLIFIED?")
print("-" * 80)
print("""
The FC layers are already extremely simple (single linear layer).
However, we could simplify further by:

Option 1: Reduce feature dimension first (bottleneck)
  - Add a bottleneck: Linear(512, 128) → then Linear(128, 11/10)
  - This would reduce parameters but add an extra layer
  - Might hurt performance

Option 2: Remove bias terms
  - Save ~31 parameters (11+10+10)
  - Negligible impact

Option 3: Use smaller feature dimension
  - Use ResNet with smaller feature dim (e.g., ResNet18 → 512 is already small)
  - Not much room here

Option 4: Share weights between heads
  - Share some layers between tens/ones/full
  - Could reduce parameters but might hurt task-specific learning

**Current FC layers are already optimal for simplicity!**
They're just: feature_vector × weight_matrix + bias = logits
""")

print("\n5. WHY FRAME MODEL IS SLOWER (REVISITED)")
print("-" * 80)
print(f"""
The issue isn't FC complexity - it's the NUMBER of FC operations:

SEQ Model:
  - Processes {feature_dim}-dim features through BiGRU → 256-dim
  - Runs FC heads on 64 sequences: 64 × 3 = 192 FC operations
  - Each FC: Linear(256, 11/10) = small matrices

FRAME Model:
  - Processes {feature_dim}-dim features directly
  - Runs FC heads on 512 frames: 512 × 3 = 1,536 FC operations
  - Each FC: Linear(512, 11/10) = larger matrices

The FC layers themselves are simple, but FRAME runs them 8x more times!
""")

print("\n6. COMPUTATIONAL COMPARISON")
print("-" * 80)

# Calculate operations
B = 64
T = 8
seq_feat_dim = 256  # After BiGRU
frame_feat_dim = 512  # Direct from ResNet

# SEQ: B sequences, each with 256-dim features
seq_fc_ops = B * (seq_feat_dim * num_tens_classes + seq_feat_dim * num_ones_classes + seq_feat_dim * num_full_classes)

# FRAME: B*T frames, each with 512-dim features  
frame_fc_ops = (B * T) * (frame_feat_dim * num_tens_classes + frame_feat_dim * num_ones_classes + frame_feat_dim * num_full_classes)

print(f"""
FC Operations (multiplications) per batch:

SEQ Model:
  - {B} sequences × 3 heads × {seq_feat_dim} features = {seq_fc_ops:,} operations
  - Feature dim: {seq_feat_dim} (after BiGRU reduction)

FRAME Model:
  - {B*T} frames × 3 heads × {frame_feat_dim} features = {frame_fc_ops:,} operations
  - Feature dim: {frame_feat_dim} (direct from ResNet)

Ratio: FRAME does {frame_fc_ops / seq_fc_ops:.1f}x more FC operations!

The FC layers are simple, but FRAME runs them on MORE data.
""")

print("\n7. CONCLUSION")
print("-" * 80)
print("""
✅ T = Sequence length (number of frames)
✅ FC = Fully Connected (Linear) layer
✅ Current FC layers are already very simple (single linear layer)
✅ The slowdown comes from running FC on MORE frames, not FC complexity
✅ Simplifying FC further would likely hurt accuracy more than help speed

The real optimization would be to reduce the number of frames processed,
not to simplify the FC layers themselves.
""")



