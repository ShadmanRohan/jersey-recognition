"""
Check GPU setup and provide instructions for CUDA PyTorch installation.
"""

import subprocess
import sys

def check_nvidia_smi():
    """Check if nvidia-smi is available."""
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            return True, result.stdout
        return False, None
    except FileNotFoundError:
        return False, None

def check_pytorch_cuda():
    """Check PyTorch CUDA availability."""
    try:
        import torch
        return {
            "version": torch.__version__,
            "cuda_compiled": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "is_cpu_version": "+cpu" in torch.__version__
        }
    except ImportError:
        return None

def main():
    print("="*80)
    print("GPU SETUP CHECK")
    print("="*80)
    print()
    
    # Check nvidia-smi
    nvidia_available, nvidia_output = check_nvidia_smi()
    if nvidia_available:
        print("✅ NVIDIA GPU detected:")
        # Extract GPU name from nvidia-smi output
        lines = nvidia_output.split('\n')
        for line in lines:
            if 'GeForce' in line or 'RTX' in line or 'GTX' in line or 'Tesla' in line:
                print(f"   {line.strip()}")
        print()
    else:
        print("❌ nvidia-smi not found. GPU may not be available.")
        print()
    
    # Check PyTorch
    pytorch_info = check_pytorch_cuda()
    if pytorch_info:
        print(f"PyTorch version: {pytorch_info['version']}")
        print(f"CUDA compiled: {pytorch_info['cuda_compiled'] or 'None (CPU-only build)'}")
        print(f"CUDA available: {pytorch_info['cuda_available']}")
        print()
        
        if pytorch_info['is_cpu_version']:
            print("⚠️  PROBLEM DETECTED:")
            print("   PyTorch is installed with CPU-only version!")
            print("   Your GPU is available but PyTorch cannot use it.")
            print()
            print("="*80)
            print("SOLUTION: Install PyTorch with CUDA support")
            print("="*80)
            print()
            print("Option 1: Install via pip (recommended)")
            print("   pip uninstall torch torchvision")
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
            print()
            print("Option 2: Check PyTorch website for your CUDA version")
            print("   Visit: https://pytorch.org/get-started/locally/")
            print("   Select: CUDA 12.1 (or your CUDA version)")
            print()
            print("Option 3: If using conda")
            print("   conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia")
            print()
            print("After installation, verify with:")
            print("   python -c \"import torch; print(torch.cuda.is_available())\"")
            print("   Should print: True")
        elif not pytorch_info['cuda_available']:
            print("⚠️  CUDA not available in PyTorch")
            print("   This might be a driver or CUDA toolkit issue.")
        else:
            print("✅ PyTorch CUDA support is working!")
            import torch
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("❌ PyTorch not installed")

if __name__ == "__main__":
    main()



