# CUDA Usage Guide for Poker Knight

This guide covers how to use GPU acceleration in Poker Knight for massive performance improvements.

## Table of Contents

1. [Requirements](#requirements)
2. [Installation](#installation)
3. [Basic Usage](#basic-usage)
4. [Configuration](#configuration)
5. [Performance Guidelines](#performance-guidelines)
6. [Troubleshooting](#troubleshooting)
7. [API Reference](#api-reference)

## Requirements

### Hardware
- NVIDIA GPU with CUDA Compute Capability 3.5 or higher
- Recommended: GTX 1060 / RTX 2060 or better
- Minimum 2GB VRAM (4GB+ recommended)

### Software
- CUDA Toolkit 11.0 or higher
- Python 3.8+
- CuPy (GPU array library)

## Installation

### 1. Install CUDA Toolkit

Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

Verify installation:
```bash
nvidia-smi
nvcc --version
```

### 2. Install CuPy

For CUDA 11.x:
```bash
pip install cupy-cuda11x
```

For CUDA 12.x:
```bash
pip install cupy-cuda12x
```

### 3. Install Poker Knight

```bash
pip install poker-knight[cuda]
```

Or from source:
```bash
git clone https://github.com/yourusername/poker_knight.git
cd poker_knight
pip install -e ".[cuda]"
```

## Basic Usage

### Automatic GPU Acceleration

Poker Knight automatically uses GPU when beneficial:

```python
from poker_knight import solve_poker_hand

# Automatically uses GPU for large simulations
result = solve_poker_hand(
    ['A♠', 'K♠'],  # Your hand
    3,              # Number of opponents
    simulation_mode='default'  # 100k simulations
)

print(f"Win probability: {result.win_probability:.1%}")
```

### Check GPU Availability

```python
from poker_knight.cuda import CUDA_AVAILABLE, get_device_info

if CUDA_AVAILABLE:
    print("GPU acceleration available!")
    info = get_device_info()
    print(f"Device: {info['name']}")
    print(f"Memory: {info['total_memory'] / 1e9:.1f} GB")
else:
    print("GPU not available - using CPU")
```

### Force GPU Usage

```python
from poker_knight import MonteCarloSolver

solver = MonteCarloSolver()

# Ensure GPU is used (if available)
if solver.gpu_solver:
    result = solver.analyze_hand(
        ['A♠', 'A♥'],
        4,
        ['K♠', 'Q♥', 'J♦'],
        simulation_mode='precision'  # 500k simulations
    )
```

## Configuration

### Config File Settings

Edit `poker_knight/config.json`:

```json
{
  "cuda_settings": {
    "enable_cuda": true,              // Enable/disable GPU
    "always_use_gpu": false,          // Always use GPU when enabled (ignores thresholds)
    "min_simulations_for_gpu": 1000,  // Minimum sims to use GPU (if always_use_gpu=false)
    "gpu_batch_size": 1000,           // Batch size for GPU processing
    "texture_memory_for_lookups": true, // Use texture memory
    "persistent_kernel_cache": true,   // Cache compiled kernels
    "multi_gpu_threshold": 1000000,   // When to use multiple GPUs
    "fallback_to_cpu": true           // Fallback if GPU fails
  }
}
```

### Always Use GPU Mode

To always use GPU regardless of simulation count:

```json
{
  "cuda_settings": {
    "enable_cuda": true,
    "always_use_gpu": true  // Forces GPU usage for all simulations
  }
}
```

This ensures consistency but may be slightly slower for very small simulations (<1000).

### Environment Variables

```bash
# Disable GPU acceleration
export POKER_KNIGHT_DISABLE_CUDA=1

# Select specific GPU
export CUDA_VISIBLE_DEVICES=0

# Set GPU memory fraction
export POKER_KNIGHT_GPU_MEMORY_FRACTION=0.8
```

## Performance Guidelines

### When GPU Acceleration Helps

GPU provides significant speedups for:

1. **Large Simulation Counts** (>10,000)
   - 10-30x speedup typical
   - Up to 50x for very large simulations

2. **Multi-Opponent Scenarios** (3+ opponents)
   - More parallel work per simulation
   - Better GPU utilization

3. **Batch Processing**
   - Process multiple hands simultaneously
   - Ideal for AI training/analysis

4. **Tournament Simulations**
   - ICM calculations benefit from parallelization
   - Complex multi-way scenarios

### When to Use CPU

CPU is preferred for:

1. **Small Simulations** (<10,000)
   - GPU overhead exceeds benefits
   - CPU cache efficiency

2. **Single Opponent** pre-flop
   - Simple calculations
   - Low parallelism opportunity

3. **Real-time Decisions**
   - Sub-millisecond latency requirements
   - GPU initialization overhead

### Performance Examples

| Scenario | Simulations | CPU Time | GPU Time | Speedup |
|----------|-------------|----------|----------|---------|
| Pre-flop, 2 opponents | 100k | 1.2s | 0.08s | 15x |
| Flop, 4 opponents | 100k | 2.5s | 0.12s | 21x |
| River, 6 opponents | 500k | 15.0s | 0.45s | 33x |
| Batch (10 hands) | 50k each | 8.0s | 0.6s | 13x |

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```
Error: CUDA not available
```

**Solution:**
- Verify NVIDIA GPU: `nvidia-smi`
- Check CUDA installation: `nvcc --version`
- Reinstall CuPy: `pip install --upgrade cupy-cuda11x`

#### 2. Out of Memory
```
Error: CUDA out of memory
```

**Solution:**
- Reduce batch size in config
- Close other GPU applications
- Use memory pooling:

```python
import cupy as cp
mempool = cp.get_default_memory_pool()
mempool.free_all_blocks()
```

#### 3. Kernel Compilation Error
```
Error: Failed to compile CUDA kernel
```

**Solution:**
- Clear kernel cache: `rm -rf poker_knight/cuda/.cuda_cache`
- Update GPU drivers
- Check CUDA/CuPy version compatibility

#### 4. Performance Not Improved

**Checklist:**
- Simulation count >10,000?
- GPU not throttling? Check: `nvidia-smi`
- Other GPU processes? Check utilization
- Try benchmark: `python examples/cuda_demo.py`

### Debug Mode

Enable detailed GPU logging:

```python
import logging
logging.getLogger('poker_knight.cuda').setLevel(logging.DEBUG)

# Now run analysis
result = solve_poker_hand(['A♠', 'A♥'], 3)
```

## API Reference

### CUDA Module Functions

```python
from poker_knight.cuda import (
    CUDA_AVAILABLE,      # bool: Is CUDA available?
    should_use_gpu,      # function: Should GPU be used?
    get_device_info,     # function: Get GPU information
)

# Check if GPU should be used
if should_use_gpu(num_simulations=100000, num_opponents=3):
    print("GPU will be used")

# Get device information
info = get_device_info()
# Returns: {
#     'name': 'NVIDIA GeForce RTX 3080',
#     'compute_capability': '8.6',
#     'total_memory': 10737418240,
#     'multiprocessors': 68,
#     ...
# }
```

### GPU Solver Class

```python
from poker_knight.cuda.gpu_solver import GPUSolver

# Create GPU solver directly
gpu_solver = GPUSolver()

# Analyze hand
result = gpu_solver.analyze_hand(
    hero_hand=['A♠', 'K♠'],
    num_opponents=3,
    board_cards=['Q♠', 'J♠', '10♥'],
    num_simulations=100000
)

# Clean up
gpu_solver.close()
```

### Benchmarking

```python
from poker_knight.cuda.benchmark import BenchmarkSuite

# Run comprehensive benchmarks
suite = BenchmarkSuite()
results = suite.run_all_benchmarks()

# Generate report
report = suite.generate_report(results)
print(report)
```

### Memory Management

```python
from poker_knight.cuda.memory import (
    GPUMemoryPool,
    estimate_memory_requirements,
    optimize_batch_size
)

# Estimate memory needs
mem_req = estimate_memory_requirements(
    num_simulations=100000,
    num_opponents=3,
    batch_size=10
)
print(f"Required GPU memory: {mem_req['total'] / 1e6:.1f} MB")

# Optimize batch size for available memory
import cupy as cp
free_mem = cp.cuda.Device().mem_info[0]
optimal_batch = optimize_batch_size(free_mem, {
    'num_simulations': 100000,
    'num_opponents': 3
})
```

## Advanced Topics

### Custom CUDA Kernels

For advanced users who want to modify CUDA kernels:

1. Edit kernel source: `poker_knight/cuda/kernels/monte_carlo.cu`
2. Clear cache: `rm -rf poker_knight/cuda/.cuda_cache`
3. Kernels recompile automatically on next use

### Multi-GPU Support

For systems with multiple GPUs:

```python
# Use specific GPU
import cupy as cp
with cp.cuda.Device(1):  # Use GPU 1
    result = solve_poker_hand(['A♠', 'A♥'], 3)

# Distribute across GPUs (future feature)
# Currently uses single GPU selected by CUDA_VISIBLE_DEVICES
```

### Performance Profiling

Profile GPU performance:

```python
# Using CuPy's profiler
from cupy import prof

with prof.time_range('poker_analysis', color_id=0):
    result = solve_poker_hand(['A♠', 'K♠'], 3)

# Or use NVIDIA Nsight for detailed profiling
```

## Conclusion

GPU acceleration in Poker Knight provides massive performance improvements for:
- Large-scale simulations
- AI training data generation  
- Real-time analysis with high accuracy
- Tournament strategy optimization

For most use cases, the automatic GPU detection and usage will provide optimal performance without any configuration needed.