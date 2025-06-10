# CUDA Implementation Plan for Poker Knight

## Overview

This document outlines the plan for integrating CUDA acceleration into Poker Knight v1.8.0, targeting 10-30x performance improvements for Monte Carlo simulations.

## Architecture Design

### 1. Hybrid CPU-GPU System

```
┌─────────────────────┐
│   Python API        │  (No changes to public interface)
├─────────────────────┤
│  Dispatcher Logic   │  (Decides CPU vs GPU based on problem size)
├──────────┬──────────┤
│   CPU    │   GPU    │
│ Backend  │ Backend  │
└──────────┴──────────┘
```

### 2. GPU Memory Layout

#### Card Representation (8-bit packed)
```cuda
// Bits 0-3: Rank (2=0, 3=1, ..., A=12)
// Bits 4-5: Suit (0=♠, 1=♥, 2=♦, 3=♣)
// Bit 7: Valid flag
typedef uint8_t Card;
```

#### Hand Evaluation Lookup Tables (Constant Memory)
- Flush lookup: 8,192 entries (13-bit index)
- Straight lookup: 8,192 entries
- Rank lookup: 52C5 = 2,598,960 entries (pre-computed)

### 3. CUDA Kernel Design

#### Main Simulation Kernel
```cuda
__global__ void monte_carlo_kernel(
    Card* hero_hand,           // 2 cards
    Card* board_cards,         // 0-5 cards
    int num_opponents,
    int simulations_per_thread,
    uint32_t* results,         // Win/tie/loss counts
    curandState* rng_states
);
```

#### Optimizations
- Warp-level primitives for reduction
- Shared memory for frequently accessed data
- Texture memory for lookup tables
- Coalesced memory access patterns

## Implementation Phases

### Phase 1: Foundation (Week 1-2)
- [ ] Set up CUDA development environment
- [ ] Create PyCUDA/CuPy integration layer
- [ ] Implement basic card representation conversion
- [ ] Write unit tests for CPU-GPU data transfer

### Phase 2: Core Algorithms (Week 3-4)
- [ ] Port hand evaluation to CUDA
- [ ] Implement GPU-based RNG (cuRAND)
- [ ] Create basic Monte Carlo kernel
- [ ] Benchmark against CPU implementation

### Phase 3: Optimization (Week 5-6)
- [ ] Generate and optimize lookup tables
- [ ] Implement shared memory optimizations
- [ ] Add multi-GPU support
- [ ] Profile and tune kernel parameters

### Phase 4: Integration (Week 7-8)
- [ ] Create seamless CPU/GPU dispatcher
- [ ] Add configuration options
- [ ] Implement fallback mechanisms
- [ ] Comprehensive testing suite

## Dependencies

### Required
- CUDA Toolkit 11.0+ (for modern GPU support)
- PyCUDA or CuPy (Python-CUDA interface)
- NumPy (for array operations)

### Optional
- NVIDIA Nsight for profiling
- cuDNN for potential ML extensions

## Performance Targets

| Metric | Current (CPU) | Target (GPU) | Speedup |
|--------|--------------|--------------|---------|
| Hand evaluation | ~1M/sec | 50-100M/sec | 50-100x |
| Full simulation | ~100K/sec | 2-5M/sec | 20-50x |
| API latency | 10-100ms | 5-20ms | 2-10x |

## Compatibility Considerations

### GPU Requirements
- Minimum: CUDA Compute Capability 3.5 (Kepler)
- Recommended: CC 7.0+ (Volta or newer)
- Memory: 2GB+ VRAM

### Fallback Strategy
```python
def solve_poker_hand(hero_hand, num_opponents, **kwargs):
    if cuda_available() and should_use_gpu(num_simulations):
        return gpu_solver.analyze(hero_hand, num_opponents, **kwargs)
    else:
        return cpu_solver.analyze(hero_hand, num_opponents, **kwargs)
```

## Code Organization

```
poker_knight/
├── cuda/
│   ├── __init__.py
│   ├── kernels/
│   │   ├── monte_carlo.cu
│   │   ├── hand_eval.cu
│   │   └── utilities.cu
│   ├── memory.py          # GPU memory management
│   ├── dispatcher.py      # CPU/GPU routing logic
│   └── bindings.py        # PyCUDA/CuPy interface
├── core/                  # Existing CPU implementation
└── solver.py              # Modified to support GPU backend
```

## Testing Strategy

1. **Correctness Tests**: Ensure GPU results match CPU exactly
2. **Performance Tests**: Benchmark across various scenarios
3. **Stress Tests**: Large-scale simulations, memory limits
4. **Compatibility Tests**: Different GPU architectures

## Example Usage (Post-Implementation)

```python
from poker_knight import solve_poker_hand

# Automatically uses GPU if available
result = solve_poker_hand(
    ['A♠', 'K♠'], 
    3,
    simulation_mode='precision',
    backend='cuda'  # Optional: force GPU usage
)

# Check which backend was used
print(f"Backend: {result.backend}")  # 'cuda' or 'cpu'
print(f"Device: {result.device}")     # 'GeForce RTX 3080' etc
```

## Risk Mitigation

1. **Platform Lock-in**: Maintain complete CPU implementation
2. **Debugging Complexity**: Comprehensive logging and validation
3. **Memory Limitations**: Batch processing for large simulations
4. **Driver Issues**: Clear documentation and version requirements

## Next Steps

1. Create feature branch: `feature/cuda-acceleration`
2. Set up CUDA development environment
3. Implement Phase 1 foundation
4. Create benchmarking framework
5. Begin iterative development