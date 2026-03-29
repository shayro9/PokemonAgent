"""
Simple tests for GPU device integration.
Verifies that the policy works on both CPU and GPU.
"""

import torch
import numpy as np
from gymnasium import spaces

from env.states.gen1.battle_state_gen_1 import BattleStateGen1
from policy.policy import AttentionPointerPolicy
from policy.device_manager import get_device


def test_device_manager_auto():
    """Test auto device detection."""
    device = get_device(device="auto")
    assert device in ("cpu", "cuda"), f"Invalid device: {device}"
    print(f"✓ Auto-detect device: {device}")


def test_device_manager_cpu():
    """Test explicit CPU selection."""
    device = get_device(device="cpu")
    assert device == "cpu"
    print(f"✓ CPU device: {device}")


def test_device_manager_cuda_raises():
    """Test that explicit CUDA raises if unavailable."""
    if not torch.cuda.is_available():
        try:
            device = get_device(device="cuda")
            assert False, "Should have raised ValueError"
        except ValueError:
            print("✓ CUDA error handling works (CUDA not available)")
    else:
        device = get_device(device="cuda")
        assert device == "cuda"
        print(f"✓ CUDA device available: {device}")


def test_policy_cpu_forward():
    """Test policy forward pass on CPU."""
    device = "cpu"
    
    obs_space = spaces.Box(low=-1, high=1, shape=(BattleStateGen1.array_len(),), dtype=np.float32)
    act_space = spaces.Discrete(26)
    
    policy = AttentionPointerPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda progress: 3e-4,
    )
    policy.to(device)
    
    # Simulate observation batch
    obs_batch = torch.randn(4, BattleStateGen1.array_len(), device=device, dtype=torch.float32)
    
    # Forward pass should work
    with torch.no_grad():
        actions, values, log_probs = policy.forward(obs_batch, deterministic=True)
    
    assert actions.shape == (4,)
    assert values.shape == (4, 1)
    assert log_probs.shape == (4,)
    print(f"✓ Policy CPU forward pass works: actions={actions.shape}, values={values.shape}")


def test_policy_cuda_forward():
    """Test policy forward pass on GPU if available."""
    if not torch.cuda.is_available():
        print("⊘ Skipping GPU test (CUDA not available)")
        return
    
    device = "cuda"
    
    obs_space = spaces.Box(low=-1, high=1, shape=(BattleStateGen1.array_len(),), dtype=np.float32)
    act_space = spaces.Discrete(26)
    
    policy = AttentionPointerPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda progress: 3e-4,
    )
    policy.to(device)
    
    # Simulate observation batch
    obs_batch = torch.randn(4, BattleStateGen1.array_len(), device=device, dtype=torch.float32)
    
    # Forward pass should work
    with torch.no_grad():
        actions, values, log_probs = policy.forward(obs_batch, deterministic=True)
    
    assert actions.shape == (4,)
    assert values.shape == (4, 1)
    assert log_probs.shape == (4,)
    print(f"✓ Policy GPU forward pass works: actions={actions.shape}, values={values.shape}")


if __name__ == "__main__":
    print("\n=== Testing Device Integration ===\n")
    test_device_manager_auto()
    test_device_manager_cpu()
    test_device_manager_cuda_raises()
    test_policy_cpu_forward()
    test_policy_cuda_forward()
    print("\n✓ All device tests passed!\n")
