# GPU Benchmarking Guide

## Overview

The GPU benchmark suite measures policy network performance on CPU vs GPU across:
- **Forward pass latency** - Inference speed at different batch sizes (1, 32, 256)
- **Gradient computation** - Training step time (forward + backward)
- **Memory usage** - Model and peak GPU memory

## Running Benchmarks

```bash
# Run benchmarks (will test CPU, and GPU if available)
python -m training.gpu_benchmark

# Output:
# - Console summary with timing and speedups
# - JSON file: training/benchmark_results.json
```

## Expected Results

### CPU Baseline
- Forward pass (batch 256): ~5-10ms
- Gradient step (batch 256): ~20-40ms
- Model size: ~3-5 MB

### GPU Expected (NVIDIA RTX 4090/A100)
- Forward pass (batch 256): ~1-2ms (5-10x speedup)
- Gradient step (batch 256): ~5-10ms (3-5x speedup)
- GPU memory: ~500-1000 MB

### GPU Expected (NVIDIA RTX 3060/RTX 4070)
- Forward pass (batch 256): ~2-4ms (2-5x speedup)
- Gradient step (batch 256): ~10-20ms (2-3x speedup)
- GPU memory: ~300-600 MB

## Interpreting Results

**Speedup = CPU Time / GPU Time**

- **Speedup 1.0-1.5x**: GPU not efficient for this workload (I/O bound)
- **Speedup 1.5-3.0x**: GPU helps, but modest gains
- **Speedup 3.0-10x**: Excellent GPU efficiency

## Key Metrics

| Metric | What It Means | Target |
|--------|---------------|--------|
| Forward latency | Time to compute action from observation | < 10ms CPU, < 2ms GPU |
| Gradient step | Time for one training update | < 50ms CPU, < 15ms GPU |
| Throughput | Samples processed per second | Higher is better |
| Memory | GPU RAM used | < 2GB for training |

## Example Output

```
============================================================
GPU BENCHMARK SUITE - PokemonAgent Policy
============================================================

🖥️  CPU Benchmarks
------------------------------------------------------------

🔄 Forward Pass Benchmark (cpu)...
  Batch   1: 1.23ms,      813 samples/sec
  Batch  32:  5.67ms,    5642 samples/sec
  Batch 256: 45.23ms,    5658 samples/sec

🔄 Gradient Computation Benchmark (cpu)...
  Batch 256: 42.15ms per step

💾 Memory Benchmark (cpu)...
  Model size: ~4.23 MB (estimate)

🚀 GPU Benchmarks
------------------------------------------------------------

🔄 Forward Pass Benchmark (cuda)...
  Batch   1: 0.45ms,    2222 samples/sec
  Batch  32:  1.23ms,   26016 samples/sec
  Batch 256:  8.90ms,   28764 samples/sec

🔄 Gradient Computation Benchmark (cuda)...
  Batch 256: 12.34ms per step

💾 Memory Benchmark (cuda)...
  Peak GPU memory: 512.34 MB

⚡ Speedup (CUDA vs CPU):
  batch_1_latency_ms: 2.73x
  batch_32_latency_ms: 4.61x
  batch_256_latency_ms: 5.08x
  gradient_step_ms: 3.42x
```

## Next Steps

After benchmarking:

1. **Phase 2B**: Profile attention operations for optimization opportunities
2. **Phase 2C**: Create batch inference utilities
3. **Phase 4**: Run full training with GPU to validate end-to-end speedup
