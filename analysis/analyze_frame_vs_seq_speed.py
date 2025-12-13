"""
Analysis: Why is FRAME model slower than SEQ model?

This script breaks down the computational differences between FRAME and SEQ models.
"""

import torch
import torch.nn as nn
from config import Config

def analyze_frame_vs_seq_computation():
    """Analyze computational differences between FRAME and SEQ models."""
    
    config = Config()
    B = 64  # batch size
    T = 8   # average sequence length
    F = 512 # feature dimension (ResNet18)
    
    print("="*80)
    print("COMPUTATIONAL ANALYSIS: FRAME vs SEQ Models")
    print("="*80)
    print(f"\nAssumptions: Batch size={B}, Sequence length={T}, Feature dim={F}\n")
    
    print("="*80)
    print("SEQ MODEL COMPUTATION")
    print("="*80)
    print("\n1. Encoder (ResNet18):")
    print(f"   Input: (B*T, 3, H, W) = ({B*T}, 3, 192, 96)")
    print(f"   Output: (B*T, F) = ({B*T}, {F})")
    print(f"   Operations: {B*T} forward passes through ResNet18")
    
    print("\n2. Reshape:")
    print(f"   (B*T, F) → (B, T, F) = ({B*T}, {F}) → ({B}, {T}, {F})")
    
    print("\n3. BiGRU:")
    print(f"   Input: (B, T, F) = ({B}, {T}, {F})")
    print(f"   Hidden dim: 128 per direction = 256 total")
    print(f"   Operations: Recurrent processing of {T} timesteps")
    print(f"   Output: (B, 256) = ({B}, 256)")
    print(f"   Note: GRU operations are efficient - just matrix multiplications on features")
    
    print("\n4. FC Heads (3 heads):")
    print(f"   Input: (B, 256) = ({B}, 256)")
    print(f"   - FC tens: ({B}, 256) → ({B}, 11)")
    print(f"   - FC ones: ({B}, 256) → ({B}, 10)")
    print(f"   - FC full: ({B}, 256) → ({B}, 10)")
    print(f"   Total FC operations: {B} samples × 3 heads = {B*3} operations")
    
    seq_total_fc = B * 3
    print(f"\n✅ SEQ Total FC head operations: {seq_total_fc}")
    
    print("\n" + "="*80)
    print("FRAME MODEL COMPUTATION")
    print("="*80)
    print("\n1. Encoder (ResNet18):")
    print(f"   Input: (B*T, 3, H, W) = ({B*T}, 3, 192, 96)")
    print(f"   Output: (B*T, F) = ({B*T}, {F})")
    print(f"   Operations: {B*T} forward passes through ResNet18")
    print(f"   Note: Same as SEQ model")
    
    print("\n2. FC Heads (3 heads) - PER FRAME:")
    print(f"   Input: (B*T, F) = ({B*T}, {F})")
    print(f"   - FC tens: ({B*T}, {F}) → ({B*T}, 11)")
    print(f"   - FC ones: ({B*T}, {F}) → ({B*T}, 10)")
    print(f"   - FC full: ({B*T}, {F}) → ({B*T}, 10)")
    print(f"   Total FC operations: {B*T} frames × 3 heads = {B*T*3} operations")
    
    print("\n3. Aggregation:")
    print(f"   a) Reshape: ({B*T}, C) → ({B}, {T}, C) for each head")
    print(f"   b) Log-softmax: ({B}, {T}, C) → ({B}, {T}, C) for each head (3 times)")
    print(f"   c) Mean: ({B}, {T}, C) → ({B}, C) for each head (3 times)")
    print(f"   Note: These operations add overhead")
    
    frame_total_fc = B * T * 3
    print(f"\n✅ FRAME Total FC head operations: {frame_total_fc}")
    
    print("\n" + "="*80)
    print("KEY DIFFERENCES")
    print("="*80)
    
    fc_ratio = frame_total_fc / seq_total_fc
    print(f"\n1. FC Head Operations:")
    print(f"   FRAME: {frame_total_fc} operations ({B*T} frames × 3 heads)")
    print(f"   SEQ:   {seq_total_fc} operations ({B} sequences × 3 heads)")
    print(f"   Ratio: FRAME does {fc_ratio:.1f}x MORE FC operations!")
    
    print(f"\n2. Additional Operations in FRAME:")
    print(f"   - Reshape operations: 3 times (one per head)")
    print(f"   - Log-softmax: 3 times (one per head)")
    print(f"   - Mean aggregation: 3 times (one per head)")
    print(f"   These add computational overhead")
    
    print(f"\n3. BiGRU in SEQ:")
    print(f"   - Processes {T} timesteps sequentially")
    print(f"   - But only operates on {F}-dim features (not full images)")
    print(f"   - GRU operations are highly optimized in PyTorch")
    print(f"   - Output dimension: 256 (concatenated forward+backward)")
    print(f"   - Much faster than running FC heads on {B*T} frames!")
    
    print("\n" + "="*80)
    print("WHY FRAME IS SLOWER")
    print("="*80)
    print(f"""
The FRAME model is slower because:

1. **More FC Head Operations**: 
   - FRAME runs FC heads on ALL {B*T} frames
   - SEQ runs FC heads on only {B} aggregated sequences
   - That's {fc_ratio:.1f}x more FC operations!

2. **Aggregation Overhead**:
   - FRAME must reshape, compute log-softmax, and average across frames
   - This adds extra computation that SEQ doesn't need
   - SEQ gets aggregation 'for free' from the GRU

3. **BiGRU Efficiency**:
   - The BiGRU in SEQ is actually quite efficient
   - It processes {T} timesteps of {F}-dim features (not full images)
   - PyTorch's GRU implementation is highly optimized
   - The GRU output (256-dim) is much smaller than processing {B*T} frames

4. **Memory Access Patterns**:
   - FRAME processes {B*T} independent frames → more memory access
   - SEQ processes sequences sequentially → better cache locality
   - Sequential processing can be more efficient on modern hardware

**Conclusion**: Even though FRAME doesn't have a GRU, it does MORE computation
because it processes every frame through FC heads, while SEQ only processes
aggregated features. The GRU aggregation is actually faster than frame-wise
processing + explicit aggregation.
""")
    
    # Estimate relative costs
    print("\n" + "="*80)
    print("ESTIMATED COMPUTATIONAL COST BREAKDOWN")
    print("="*80)
    print(f"""
For a batch of {B} sequences with {T} frames each:

SEQ Model:
  - Encoder: {B*T} ResNet18 passes
  - BiGRU: ~{T} × {F} × 256 operations (recurrent)
  - FC heads: {B} × 3 operations
  - Total: Encoder + Small GRU + {B} FC operations

FRAME Model:
  - Encoder: {B*T} ResNet18 passes (same as SEQ)
  - FC heads: {B*T} × 3 operations ({fc_ratio:.1f}x more than SEQ!)
  - Aggregation: Reshape + 3×log_softmax + 3×mean
  - Total: Encoder + {B*T} FC operations + Aggregation overhead

The {fc_ratio:.1f}x more FC operations in FRAME outweigh the GRU cost in SEQ!
""")

if __name__ == "__main__":
    analyze_frame_vs_seq_computation()



