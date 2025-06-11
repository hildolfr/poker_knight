# kernelPokerSimNG Design Document

## Overview
A unified next-generation CUDA kernel that consolidates all Monte Carlo poker simulation functionality into a single, highly configurable kernel.

## Design Goals
1. **Feature Parity** - Match CPU solver outputs where GPU-feasible
2. **Single Kernel** - Replace all 7 existing kernels
3. **Configurable** - Runtime parameters control behavior
4. **Efficient** - Optimize for modern GPUs
5. **Extensible** - Easy to add new features

## Core Features

### 1. Basic Monte Carlo Simulation
- Win/tie/loss counting
- Configurable opponent count (1-6)
- Board completion (0-5 cards)
- High-performance RNG (cuRAND)

### 2. Hand Category Tracking
- Track frequency of each hand type (high card through royal flush)
- Optional via configuration flag
- Per-thread accumulation with atomic reduction

### 3. Statistical Data Collection
- Running sum and sum-of-squares for variance calculation
- Enables confidence interval computation
- Optional histogram of hand strengths

### 4. Batch Processing
- Process multiple hands in single kernel launch
- Efficient for AI training workloads
- Coalesced memory access patterns

### 5. Advanced Metrics (GPU-feasible subset)
- Simple board texture analysis (flush/straight potential)
- Position-based equity adjustments (pre-computed factors)
- Multi-way statistics (win rates vs different opponent counts)

## Kernel Interface

```cuda
__global__ void kernelPokerSimNG(
    // Input arrays
    const Card* hero_hands,           // Batch of hero hands
    const Card* board_cards,          // Board cards per hand
    const uint8_t* board_sizes,       // Number of board cards per hand
    const uint8_t* num_opponents,     // Opponents per hand
    
    // Configuration
    const SimConfig config,           // Bit flags for features
    const int num_hands,              // Batch size
    const int total_simulations,      // Total sims per hand
    
    // Output arrays
    uint32_t* wins,                   // Win counts per hand
    uint32_t* ties,                   // Tie counts per hand
    uint32_t* losses,                 // Loss counts per hand
    
    // Optional outputs (if enabled in config)
    uint32_t* hand_categories,        // 11 categories x num_hands
    float* variances,                 // For confidence intervals
    uint16_t* board_textures,         // Encoded board analysis
    
    // RNG state
    curandState* rng_states,
    uint64_t seed
);
```

## Configuration Flags

```cpp
struct SimConfig {
    uint32_t flags;
    
    // Feature flags
    static constexpr uint32_t TRACK_CATEGORIES   = 1 << 0;
    static constexpr uint32_t COMPUTE_VARIANCE    = 1 << 1;
    static constexpr uint32_t ANALYZE_BOARD       = 1 << 2;
    static constexpr uint32_t MULTIWAY_STATS      = 1 << 3;
    static constexpr uint32_t USE_SHARED_MEMORY   = 1 << 4;
    static constexpr uint32_t WARP_REDUCTION      = 1 << 5;
    
    // Simulation parameters packed into upper bits
    uint16_t threads_per_block;  // bits 16-31
};
```

## Memory Layout

### Input Format
- Hero hands: 2 cards per hand, packed array
- Board cards: 5 cards max per hand, packed array
- Metadata: board sizes and opponent counts

### Output Format
- Primary results: wins/ties/losses as uint32_t
- Hand categories: 11-element array per hand
- Statistical data: variance for confidence intervals
- Board analysis: bit-packed texture information

## Optimization Strategies

1. **Memory Coalescence**
   - Structure-of-arrays layout for batch processing
   - Aligned memory access patterns
   - Texture memory for lookup tables

2. **Thread Utilization**
   - Dynamic parallelism for large batches
   - Persistent threads for small batches
   - Adaptive work distribution

3. **Register Pressure**
   - Template specialization for common configurations
   - Compile-time optimization of unused features
   - Careful register allocation

4. **Shared Memory**
   - Optional reduction trees
   - Cached lookup tables
   - Inter-thread communication

## Implementation Phases

### Phase 1: Core Functionality
- Basic Monte Carlo with win/tie/loss
- Single hand processing
- cuRAND integration

### Phase 2: Hand Categories
- Category tracking during evaluation
- Efficient accumulation
- Atomic-free reduction where possible

### Phase 3: Batch Processing
- Multi-hand kernel launch
- Optimized memory patterns
- Load balancing

### Phase 4: Advanced Features
- Variance calculation
- Board texture analysis
- Position-aware adjustments

## Testing Strategy

1. **Correctness Testing**
   - Compare against CPU solver outputs
   - Statistical validation of distributions
   - Edge case coverage

2. **Performance Testing**
   - Benchmark against existing kernels
   - Profile with nsight
   - Measure speedup vs CPU

3. **Compatibility Testing**
   - Ensure drop-in replacement
   - Validate all configuration modes
   - Test on different GPU architectures

## Migration Plan

1. Implement kernelPokerSimNG alongside existing kernels
2. Comprehensive testing and validation
3. Update Python bindings to use new kernel
4. Archive old kernels after verification
5. Update documentation

## Success Criteria

- ✅ Single kernel replaces all 7 existing kernels
- ✅ Matches or exceeds current performance
- ✅ Provides all GPU-feasible CPU solver outputs
- ✅ Clean, maintainable code
- ✅ Comprehensive test coverage