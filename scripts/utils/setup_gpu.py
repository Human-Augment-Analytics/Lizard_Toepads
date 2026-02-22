#!/usr/bin/env python3
"""
Setup GPU-enabled PyTorch for the project.
This script ensures PyTorch with CUDA support is installed.
"""

import subprocess
import sys
from pathlib import Path


def check_cuda_available():
    """Check if CUDA PyTorch is installed"""
    try:
        import torch
        return torch.cuda.is_available(), torch.__version__
    except ImportError:
        return False, None


def install_cuda_pytorch():
    """Install PyTorch with CUDA 12.4 support"""
    print("Installing PyTorch with CUDA 12.4 support...")

    # Use the project's virtual environment
    cmd = [
        "uv", "pip", "install",
        "--python", ".venv",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu124"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error installing PyTorch: {result.stderr}")
        return False

    print("PyTorch with CUDA installed successfully!")
    return True


def main():
    print("GPU Setup for Lizard Toepads Project")
    print("=" * 50)

    # Check current status
    cuda_available, torch_version = check_cuda_available()

    if cuda_available:
        print(f"✓ PyTorch {torch_version} with CUDA support is already installed!")
        import torch
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
        return

    if torch_version:
        print(f"✗ PyTorch {torch_version} is installed but CUDA is not available")
    else:
        print("✗ PyTorch is not installed")

    print("\nInstalling CUDA-enabled PyTorch...")

    if install_cuda_pytorch():
        # Verify installation
        cuda_available, torch_version = check_cuda_available()
        if cuda_available:
            import torch
            print(f"\n✓ Success! PyTorch {torch_version} with CUDA is now installed")
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("\n✗ Installation completed but CUDA is still not available")
            print("Please check your NVIDIA drivers and CUDA installation")
    else:
        print("\n✗ Failed to install PyTorch with CUDA support")
        sys.exit(1)


if __name__ == "__main__":
    main()