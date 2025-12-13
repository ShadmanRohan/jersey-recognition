"""
Analyze FC layer parallelization and computational complexity.
"""

import torch
import torch.nn as nn
import time

print("="*80)
print("FC PARALLELIZATION ANALYSIS")
print("="*80)

# Simulate the operations
B = 64  # batch size
T = 8   # sequence length
F_seq = 256  # feature dim after BiGRU (SEQ)
F_frame = 512  # feature dim from ResNet (FRAME)
num_classes = 11  # for tens head

print(f"\nSetup: B={B}, T={T}, F_seq={F_seq}, F_frame={F_frame}\n")

# Create FC layers
fc_seq = nn.Linear(F_seq, num_classes)
fc_frame = nn.Linear(F_frame, num_classes)

# Create dummy inputs
seq_features = torch.randn(B, F_seq)  # (64, 256)
frame_features = torch.randn(B * T, F_frame)  # (512, 512)

print("1. ARE FC OPERATIONS PARALLELIZED?")
print("-" * 80)
print("""
YES! PyTorch's nn.Linear uses batched matrix multiplication which is:
  - Automatically parallelized on CPU (using BLAS libraries)
  - Automatically parallelized on GPU (using CUDA kernels)
  - Both models run FC operations in parallel
""")

print("\n2. MATRIX MULTIPLICATION BREAKDOWN")
print("-" * 80)

print(f"""
SEQ Model FC operation:
  Input:  (B, F_seq) = ({B}, {F_seq})
  Weight: (F_seq, num_classes) = ({F_seq}, {num_classes})
  Output: (B, num_classes) = ({B}, {num_classes})
  
  Operation: (64, 256) @ (256, 11) = (64, 11)
  This is ONE batched matrix multiplication - fully parallelized!
  
  Computational cost: B × F_seq × num_classes = {B} × {F_seq} × {num_classes} = {B * F_seq * num_classes:,} operations

FRAME Model FC operation:
  Input:  (B*T, F_frame) = ({B*T}, {F_frame})
  Weight: (F_frame, num_classes) = ({F_frame}, {num_classes})
  Output: (B*T, num_classes) = ({B*T}, {num_classes})
  
  Operation: (512, 512) @ (512, 11) = (512, 11)
  This is ONE batched matrix multiplication - fully parallelized!
  
  Computational cost: B*T × F_frame × num_classes = {B*T} × {F_frame} × {num_classes} = {B*T * F_frame * num_classes:,} operations
""")

seq_ops = B * F_seq * num_classes
frame_ops = B * T * F_frame * num_classes
ratio = frame_ops / seq_ops

print(f"\n3. COMPUTATIONAL COMPARISON")
print("-" * 80)
print(f"""
Both are parallelized, but FRAME still does more work:

SEQ:  {seq_ops:,} operations (parallelized across {B} samples)
FRAME: {frame_ops:,} operations (parallelized across {B*T} samples)

Ratio: FRAME does {ratio:.1f}x MORE operations!

Even though both are parallelized, FRAME's matrix is:
  - {ratio:.1f}x larger (more rows: {B*T} vs {B})
  - 2x larger input dimension ({F_frame} vs {F_seq})
  - Total: {ratio:.1f}x more computation
""")

print("\n4. WHY PARALLELIZATION DOESN'T HELP ENOUGH")
print("-" * 80)
print("""
Parallelization helps, but doesn't eliminate the cost difference:

1. **Matrix Size Matters**:
   - Larger matrices take more time even when parallelized
   - (512, 512) @ (512, 11) is inherently more work than (64, 256) @ (256, 11)
   - More memory access, more cache misses

2. **Input Dimension**:
   - FRAME: 512-dim features (direct from ResNet)
   - SEQ: 256-dim features (after BiGRU reduction)
   - 2x larger input dimension = 2x more computation per sample

3. **Batch Size Effect**:
   - Larger batches (512 vs 64) can help with parallelization
   - But the total work is still {ratio:.1f}x more
   - GPU utilization might be better, but wall-clock time is still longer

4. **Memory Bandwidth**:
   - FRAME processes 512 × 512 = 262,144 values
   - SEQ processes 64 × 256 = 16,384 values
   - More data to move = more time
""")

print("\n5. BENCHMARK TEST")
print("-" * 80)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Warmup
for _ in range(10):
    _ = fc_seq(seq_features.to(device))
    _ = fc_frame(frame_features.to(device))

if device.type == "cuda":
    torch.cuda.synchronize()

# Time SEQ
start = time.perf_counter()
for _ in range(100):
    _ = fc_seq(seq_features.to(device))
if device.type == "cuda":
    torch.cuda.synchronize()
seq_time = (time.perf_counter() - start) / 100

# Time FRAME
start = time.perf_counter()
for _ in range(100):
    _ = fc_frame(frame_features.to(device))
if device.type == "cuda":
    torch.cuda.synchronize()
frame_time = (time.perf_counter() - start) / 100

print(f"""
Measured FC forward pass time (averaged over 100 runs):

SEQ FC:   {seq_time*1000:.4f} ms  (processing {B} samples)
FRAME FC: {frame_time*1000:.4f} ms  (processing {B*T} samples)

Ratio: FRAME takes {frame_time/seq_time:.2f}x longer

Even with full parallelization, FRAME is slower because:
  - {ratio:.1f}x more operations
  - 2x larger input dimension
  - More memory access
""")

print("\n6. CONCLUSION")
print("-" * 80)
print(f"""
✅ YES, FC operations ARE fully parallelized (both CPU and GPU)
✅ But parallelization doesn't eliminate the cost difference
✅ FRAME still does {ratio:.1f}x more computation:
   - {B*T} samples vs {B} samples ({T}x more)
   - {F_frame}-dim vs {F_seq}-dim features (2x larger)
   - Combined: {ratio:.1f}x total operations

The parallelization helps both models equally, but FRAME's larger
matrix multiplication is inherently more expensive.

**Key Insight**: Parallelization makes both fast, but doesn't change
the relative cost - FRAME still processes {ratio:.1f}x more data!
""")



