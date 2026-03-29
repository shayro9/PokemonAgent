"""
Validation that observations work with GPU policy.
Confirms NumPy arrays from environment work with device handling.
"""

import torch
import numpy as np
from gymnasium import spaces

from env.states.gen1.battle_state_gen_1 import BattleStateGen1
from policy.policy import AttentionPointerPolicy


def test_numpy_to_gpu_conversion():
    """
    Test that NumPy observations from environment convert correctly to GPU tensors.
    SB3 automatically converts NumPy to torch tensors using the device from the model.
    Policy always expects batched observations: (B, 383)
    """
    obs_space = spaces.Box(low=-1, high=1, shape=(BattleStateGen1.array_len(),), dtype=np.float32)
    act_space = spaces.Discrete(26)
    
    # Environment returns single obs (383,)
    single_obs = np.random.randn(BattleStateGen1.array_len()).astype(np.float32)
    # Add batch dimension for policy: (1, 383)
    single_obs_batched = np.expand_dims(single_obs, axis=0)
    # Batch observations: (4, 383)
    batch_obs = np.random.randn(4, BattleStateGen1.array_len()).astype(np.float32)
    
    assert isinstance(single_obs, np.ndarray), "Environment should return NumPy array"
    print(f"✓ Single obs: shape={single_obs.shape}")
    print(f"✓ Single obs (batched): shape={single_obs_batched.shape}")
    print(f"✓ Batch obs: shape={batch_obs.shape}")
    
    # Policy on CPU
    policy_cpu = AttentionPointerPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda progress: 3e-4,
    )
    
    # Test single observation with batch dimension
    with torch.no_grad():
        action_cpu, _, _ = policy_cpu.forward(single_obs_batched, deterministic=True)
    
    assert isinstance(action_cpu, torch.Tensor), "Policy should return torch tensor"
    assert action_cpu.shape == (1,), f"Expected 1 action, got {action_cpu.shape}"
    print(f"✓ Policy CPU single obs (batched) works: action shape={action_cpu.shape}")
    
    # Test batch observations
    with torch.no_grad():
        actions_batch, values_batch, logprobs_batch = policy_cpu.forward(batch_obs, deterministic=True)
    
    assert actions_batch.shape == (4,), f"Expected batch of 4 actions, got {actions_batch.shape}"
    assert values_batch.shape == (4, 1), f"Expected batch of 4 values, got {values_batch.shape}"
    print(f"✓ Policy CPU batch inference works: actions={actions_batch.shape}, values={values_batch.shape}")
    
    # Policy on GPU if available
    if torch.cuda.is_available():
        policy_gpu = AttentionPointerPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda progress: 3e-4,
        )
        policy_gpu.to("cuda")
        
        # Test single observation on GPU
        with torch.no_grad():
            action_gpu, _, _ = policy_gpu.forward(single_obs_batched, deterministic=True)
        
        assert isinstance(action_gpu, torch.Tensor), "Policy should return torch tensor"
        assert action_gpu.shape == (1,), f"Expected 1 action, got {action_gpu.shape}"
        print(f"✓ Policy GPU single obs (batched) works: action shape={action_gpu.shape}")
        
        # Test batch observations on GPU
        with torch.no_grad():
            actions_batch_gpu, values_batch_gpu, _ = policy_gpu.forward(batch_obs, deterministic=True)
        
        assert actions_batch_gpu.shape == (4,), f"Expected batch of 4 actions, got {actions_batch_gpu.shape}"
        assert values_batch_gpu.shape == (4, 1), f"Expected batch of 4 values, got {values_batch_gpu.shape}"
        print(f"✓ Policy GPU batch inference works: actions={actions_batch_gpu.shape}, values={values_batch_gpu.shape}")
    else:
        print("⊘ GPU not available - skipping GPU test")


if __name__ == "__main__":
    print("\n=== Testing NumPy to GPU Conversion ===\n")
    test_numpy_to_gpu_conversion()
    print("\n✓ Observation validation passed!\n")
