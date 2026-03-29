"""
Batch Inference Utilities for AttentionPointerPolicy

Provides utilities for efficient batch inference on multiple observations.
Useful for evaluation and parallel environment scenarios.

Example:
    batch_inferrer = BatchInferrer(policy)
    observations = [obs1, obs2, obs3]
    actions, values = batch_inferrer.infer(observations, deterministic=True)
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


class BatchInferrer:
    """Efficient batch inference wrapper for AttentionPointerPolicy."""
    
    def __init__(self, policy, device: str = "cpu"):
        """
        Args:
            policy: AttentionPointerPolicy instance
            device: "cpu" or "cuda"
        """
        self.policy = policy
        self.device = device
        self.policy.to(device)
    
    def infer_single(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        action_mask: Optional[np.ndarray] = None,
    ) -> Tuple[int, float]:
        """
        Inference on single observation.
        
        Args:
            obs: Single observation (383,) as NumPy array
            deterministic: Whether to use deterministic action selection
            action_mask: Optional action mask
        
        Returns:
            (action, value) - Selected action and state value
        """
        # Add batch dimension
        obs_batched = np.expand_dims(obs, axis=0)
        
        with torch.no_grad():
            actions, values, _ = self.policy.forward(
                obs_batched, deterministic=deterministic, action_masks=action_mask
            )
        
        action = actions[0].item()
        value = values[0, 0].item()
        
        return action, value
    
    def infer_batch(
        self,
        observations: np.ndarray,
        deterministic: bool = True,
        action_masks: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference on batch of observations.
        
        Args:
            observations: Batch observations (B, 383) as NumPy array
            deterministic: Whether to use deterministic action selection
            action_masks: Optional action masks (B, 26)
        
        Returns:
            (actions, values) - Arrays of actions and state values
        """
        with torch.no_grad():
            actions, values, _ = self.policy.forward(
                observations, deterministic=deterministic, action_masks=action_masks
            )
        
        actions_np = actions.cpu().numpy()
        values_np = values.squeeze(-1).cpu().numpy()
        
        return actions_np, values_np
    
    def infer_list(
        self,
        observations: List[np.ndarray],
        deterministic: bool = True,
        action_masks: Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Inference on list of observations (converts to batch internally).
        
        Args:
            observations: List of observations, each (383,)
            deterministic: Whether to use deterministic action selection
            action_masks: Optional list of action masks, each (26,)
        
        Returns:
            (actions_list, values_list) - Lists of actions and values
        """
        # Stack into batch
        obs_batch = np.stack(observations, axis=0)
        
        # Stack masks if provided
        mask_batch = None
        if action_masks is not None:
            mask_batch = np.stack(action_masks, axis=0)
        
        # Run batch inference
        actions, values = self.infer_batch(obs_batch, deterministic, mask_batch)
        
        # Convert to lists
        actions_list = actions.tolist()
        values_list = values.tolist()
        
        return actions_list, values_list


class BatchInferenceBenchmark:
    """Benchmark batch inference performance."""
    
    def __init__(self, policy, device: str = "cpu"):
        self.inferrer = BatchInferrer(policy, device)
        self.policy = policy
        self.device = device
    
    def benchmark(self):
        """Run batch inference benchmarks."""
        print(f"\n📊 Batch Inference Benchmark ({self.device})")
        print("-" * 50)
        
        import time
        
        batch_sizes = [1, 8, 32, 64, 128]
        results = {}
        
        for batch_size in batch_sizes:
            # Create test observations
            obs_batch = np.random.randn(batch_size, 383).astype(np.float32)
            
            # Warmup
            for _ in range(3):
                with torch.no_grad():
                    self.policy.forward(obs_batch, deterministic=True)
            
            # Benchmark
            torch.cuda.synchronize() if self.device == "cuda" else None
            start = time.perf_counter()
            for _ in range(10):
                with torch.no_grad():
                    self.policy.forward(obs_batch, deterministic=True)
            torch.cuda.synchronize() if self.device == "cuda" else None
            elapsed = time.perf_counter() - start
            
            avg_ms = (elapsed / 10) * 1000
            throughput = batch_size * 10 / elapsed
            
            results[batch_size] = {
                "latency_ms": round(avg_ms, 3),
                "throughput_samples_per_sec": round(throughput, 0)
            }
            
            print(f"  Batch {batch_size:3d}: {avg_ms:6.2f}ms, {throughput:7.0f} samples/sec")
        
        # Calculate efficiency
        print(f"\n💡 Batch Efficiency (vs Batch 1):")
        if 1 in results and batch_sizes[-1] in results:
            single_latency = results[1]["latency_ms"]
            batch_latency = results[batch_sizes[-1]]["latency_ms"]
            ratio = single_latency / batch_latency
            print(f"  Batch {batch_sizes[-1]} is {ratio:.1f}x more efficient per sample")
        
        return results


def example_usage():
    """Example of using batch inferrer."""
    from policy.policy import AttentionPointerPolicy
    from gymnasium import spaces
    
    print("\n" + "="*60)
    print("BATCH INFERENCE EXAMPLE")
    print("="*60)
    
    # Create policy
    obs_space = spaces.Box(low=-1, high=1, shape=(383,), dtype=np.float32)
    act_space = spaces.Discrete(26)
    
    policy = AttentionPointerPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda progress: 3e-4,
    )
    
    # Create inferrer
    inferrer = BatchInferrer(policy, device="cpu")
    
    # Example 1: Single inference
    print("\n📌 Single Inference:")
    single_obs = np.random.randn(383).astype(np.float32)
    action, value = inferrer.infer_single(single_obs, deterministic=True)
    print(f"  Observation shape: {single_obs.shape}")
    print(f"  Action: {action}, Value: {value:.4f}")
    
    # Example 2: Batch inference
    print("\n📌 Batch Inference:")
    batch_obs = np.random.randn(32, 383).astype(np.float32)
    actions, values = inferrer.infer_batch(batch_obs, deterministic=True)
    print(f"  Observation shape: {batch_obs.shape}")
    print(f"  Actions shape: {actions.shape}, Values shape: {values.shape}")
    print(f"  Actions: {actions[:5]}..., Values: {values[:5]}...")
    
    # Example 3: List inference
    print("\n📌 List Inference:")
    obs_list = [np.random.randn(383).astype(np.float32) for _ in range(8)]
    actions_list, values_list = inferrer.infer_list(obs_list, deterministic=True)
    print(f"  Input: {len(obs_list)} observations")
    print(f"  Output: {len(actions_list)} actions, {len(values_list)} values")
    print(f"  Actions: {actions_list[:3]}..., Values: {[f'{v:.4f}' for v in values_list[:3]]}...")
    
    # Benchmark
    print("\n📊 Benchmarking:")
    benchmark = BatchInferenceBenchmark(policy, device="cpu")
    benchmark.benchmark()


if __name__ == "__main__":
    example_usage()
