"""
Device management for GPU/CPU handling.
Centralizes device detection and fallback logic.
"""

import torch
from typing import Literal


def get_device(device: Literal["auto", "cuda", "cpu"] = "auto") -> str:
    """
    Get the device to use for training.
    
    Args:
        device: "auto" (detect GPU, fallback to CPU), "cuda", or "cpu"
    
    Returns:
        Device string: "cuda:0" or "cpu"
    
    Raises:
        ValueError: If cuda requested but not available
    """
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA requested but not available. Install PyTorch with CUDA support.")
        return "cuda"
    
    if device == "cpu":
        return "cpu"
    
    # Auto mode: use GPU if available, else CPU
    if torch.cuda.is_available():
        print(f"✓ CUDA available - using GPU: {torch.cuda.get_device_name(0)}")
        return "cuda"
    else:
        print("⊘ CUDA not available - using CPU")
        return "cpu"


def print_device_info(device: str) -> None:
    """Print detailed device information for debugging."""
    if device == "cpu":
        print("Device: CPU")
    elif device.startswith("cuda"):
        print(f"Device: CUDA GPU")
        if torch.cuda.is_available():
            print(f"  GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
