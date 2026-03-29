"""
GPU Benchmarking Suite for PokemonAgent Policy

Measures and compares performance on CPU vs GPU:
- Policy forward pass latency
- Gradient computation time
- Memory usage
- Batch size scaling

Run with: python -m training.gpu_benchmark
"""

import time
import torch
import numpy as np
import json
from datetime import datetime
from pathlib import Path
from gymnasium import spaces

from env.states.gen1.battle_state_gen_1 import BattleStateGen1
from policy.policy import AttentionPointerPolicy
from policy.device_manager import get_device


class BenchmarkResults:
    """Store and format benchmark results."""
    
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "cpu": {},
            "cuda": {},
        }
    
    def add_result(self, device, metric_name, value):
        """Add a result."""
        if device == "cpu":
            self.results["cpu"][metric_name] = value
        elif device == "cuda":
            self.results["cuda"][metric_name] = value
    
    def save_json(self, path="training/benchmark_results.json"):
        """Save results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {path}")
    
    def print_summary(self):
        """Print human-readable summary."""
        print("\n" + "="*60)
        print("BENCHMARK RESULTS")
        print("="*60)
        
        if self.results["cpu"]:
            print("\n📊 CPU Results:")
            for key, value in self.results["cpu"].items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
        
        if self.results["cuda"]:
            print("\n📊 CUDA Results:")
            for key, value in self.results["cuda"].items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")
            
            # Calculate speedups
            print("\n⚡ Speedup (CUDA vs CPU):")
            for key in self.results["cpu"]:
                if key in self.results["cuda"] and isinstance(self.results["cpu"][key], (int, float)):
                    cpu_val = self.results["cpu"][key]
                    cuda_val = self.results["cuda"][key]
                    if cpu_val > 0:
                        speedup = cpu_val / cuda_val
                        print(f"  {key}: {speedup:.2f}x")


def benchmark_forward_pass(policy, obs_space, device, batch_sizes=[1, 32, 256], warmup_runs=3, benchmark_runs=10):
    """Benchmark policy forward pass latency."""
    print(f"\n🔄 Forward Pass Benchmark ({device})...")
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create random observations
        obs = np.random.randn(batch_size, obs_space.shape[0]).astype(np.float32)
        
        # Warmup
        for _ in range(warmup_runs):
            with torch.no_grad():
                policy.forward(obs, deterministic=True)
        
        # Synchronize GPU if needed
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(benchmark_runs):
            with torch.no_grad():
                policy.forward(obs, deterministic=True)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        avg_latency_ms = (elapsed / benchmark_runs) * 1000
        throughput = batch_size * benchmark_runs / elapsed  # samples/sec
        
        results[f"batch_{batch_size}_latency_ms"] = round(avg_latency_ms, 3)
        results[f"batch_{batch_size}_throughput_samples_per_sec"] = round(throughput, 0)
        
        print(f"  Batch {batch_size:3d}: {avg_latency_ms:6.2f}ms, {throughput:7.0f} samples/sec")
    
    return results


def benchmark_gradient_computation(policy, obs_space, device, batch_size=256, warmup_runs=2, benchmark_runs=5):
    """Benchmark gradient computation (forward + backward)."""
    print(f"\n🔄 Gradient Computation Benchmark ({device})...")
    
    # Create observations and dummy actions
    obs = np.random.randn(batch_size, obs_space.shape[0]).astype(np.float32)
    actions = torch.randint(0, 26, (batch_size,))
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    
    # Warmup
    for _ in range(warmup_runs):
        optimizer.zero_grad()
        values, log_probs, _ = policy.evaluate_actions(obs, actions)
        loss = -log_probs.mean()
        loss.backward()
        optimizer.step()
    
    # Synchronize GPU if needed
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(benchmark_runs):
        optimizer.zero_grad()
        values, log_probs, _ = policy.evaluate_actions(obs, actions)
        loss = -log_probs.mean()
        loss.backward()
        optimizer.step()
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time_ms = (elapsed / benchmark_runs) * 1000
    
    print(f"  Batch {batch_size}: {avg_time_ms:.2f}ms per step")
    
    return {
        f"gradient_step_ms": round(avg_time_ms, 3),
        f"gradient_steps_per_sec": round(1000 / avg_time_ms, 1),
    }


def benchmark_memory(policy, device, batch_size=256):
    """Benchmark memory usage."""
    print(f"\n💾 Memory Benchmark ({device})...")
    
    if device == "cpu":
        # Rough estimate from model parameters
        param_count = sum(p.numel() for p in policy.parameters())
        model_size_mb = (param_count * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"  Model size: ~{model_size_mb:.2f} MB (estimate)")
        return {"model_size_mb": round(model_size_mb, 2)}
    
    elif device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Forward pass
        obs = np.random.randn(batch_size, 383).astype(np.float32)
        with torch.no_grad():
            policy.forward(obs)
        
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        
        print(f"  Peak GPU memory: {peak_memory_mb:.2f} MB")
        return {"peak_memory_mb": round(peak_memory_mb, 2)}


def run_benchmarks():
    """Run complete benchmark suite."""
    print("\n" + "="*60)
    print("GPU BENCHMARK SUITE - PokemonAgent Policy")
    print("="*60)
    
    obs_space = spaces.Box(low=-1, high=1, shape=(BattleStateGen1.array_len(),), dtype=np.float32)
    act_space = spaces.Discrete(26)
    
    results = BenchmarkResults()
    
    # ========== CPU Benchmarks ==========
    print("\n🖥️  CPU Benchmarks")
    print("-" * 60)
    
    policy_cpu = AttentionPointerPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda progress: 3e-4,
    )
    
    forward_cpu = benchmark_forward_pass(policy_cpu, obs_space, "cpu", batch_sizes=[1, 32, 256])
    for key, val in forward_cpu.items():
        results.add_result("cpu", key, val)
    
    gradient_cpu = benchmark_gradient_computation(policy_cpu, obs_space, "cpu")
    for key, val in gradient_cpu.items():
        results.add_result("cpu", key, val)
    
    memory_cpu = benchmark_memory(policy_cpu, "cpu")
    for key, val in memory_cpu.items():
        results.add_result("cpu", key, val)
    
    # ========== GPU Benchmarks ==========
    if torch.cuda.is_available():
        print("\n🚀 GPU Benchmarks")
        print("-" * 60)
        
        policy_gpu = AttentionPointerPolicy(
            observation_space=obs_space,
            action_space=act_space,
            lr_schedule=lambda progress: 3e-4,
        )
        policy_gpu.to("cuda")
        
        forward_gpu = benchmark_forward_pass(policy_gpu, obs_space, "cuda", batch_sizes=[1, 32, 256])
        for key, val in forward_gpu.items():
            results.add_result("cuda", key, val)
        
        gradient_gpu = benchmark_gradient_computation(policy_gpu, obs_space, "cuda")
        for key, val in gradient_gpu.items():
            results.add_result("cuda", key, val)
        
        memory_gpu = benchmark_memory(policy_gpu, "cuda")
        for key, val in memory_gpu.items():
            results.add_result("cuda", key, val)
    else:
        print("\n⊘ GPU not available - skipping GPU benchmarks")
    
    # ========== Save Results ==========
    results.print_summary()
    results.save_json()
    
    return results


if __name__ == "__main__":
    run_benchmarks()
