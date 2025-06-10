# Benchmark and Utility Scripts

This directory contains benchmark scripts and utilities for testing Poker Knight's performance.

## Benchmark Scripts

- **benchmark_gpu.py** - Basic GPU vs CPU performance comparison
- **benchmark_gpu_detailed.py** - Detailed GPU performance analysis with various scenarios
- **analyze_kernel_perf.py** - CUDA kernel performance analysis
- **monitor_gpu.py** - Real-time GPU utilization monitoring

## Utility Scripts

- **check_gpu_default.py** - Check if GPU is used by default
- **clear_cuda_cache.py** - Clear CUDA compilation cache

## Usage Examples

```bash
# Run basic GPU benchmark
python examples/benchmarks/benchmark_gpu.py

# Monitor GPU utilization during analysis
python examples/benchmarks/monitor_gpu.py

# Clear CUDA cache if needed
python examples/clear_cuda_cache.py
```