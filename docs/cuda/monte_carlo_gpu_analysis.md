# Monte Carlo Simulation GPU Acceleration Analysis for Poker Knight

## Executive Summary

This analysis identifies the main computational bottlenecks in the Poker Knight Monte Carlo simulation engine and evaluates opportunities for GPU parallelization. The codebase shows sophisticated CPU-based parallelization but contains several hot paths that would benefit significantly from GPU acceleration.

## 1. Main Computational Bottlenecks

### 1.1 Core Simulation Loop (`simulation/runner.py`)

The primary bottleneck is the Monte Carlo simulation loop that runs 10,000 to 500,000+ iterations:

**Key Hot Path Functions:**
- `simulate_hand()` (lines 45-93): Core single-hand simulation
- `run_sequential_simulations()` (lines 102-249): Sequential simulation execution
- `run_parallel_simulations()` (lines 251-325): Thread-based parallel execution

**Computational Profile:**
```python
# Per-simulation operations:
1. Deck shuffling and card dealing (random number generation)
2. Hand evaluation for hero + N opponents (combinatorial logic)
3. Rank comparison and winner determination
4. Result aggregation and statistics update
```

**Time Complexity:** O(num_simulations × num_opponents × hand_evaluation_cost)

### 1.2 Hand Evaluation (`core/evaluation.py`)

The `HandEvaluator.evaluate_hand()` function is called multiple times per simulation:

**Critical Operations:**
- Pattern matching for straights/flushes (lines 69-119)
- Rank counting and sorting (lines 72-73)
- Combinatorial evaluation for 6-7 card hands (lines 47-58)

**Per-evaluation Cost:**
- 5-card hands: Single evaluation pass
- 6-7 card hands: C(n,5) combinations evaluated (21 for 7 cards)

### 1.3 Current Parallel Processing Approach

The system uses three levels of parallelization:

1. **ThreadPoolExecutor** (basic parallel mode)
   - Up to 4-16 threads by default
   - Shared memory, GIL-limited
   - Good for I/O-bound operations

2. **ProcessPoolExecutor** (advanced parallel mode)
   - True multiprocessing for CPU-bound work
   - Process spawn overhead
   - Limited by CPU core count

3. **NUMA-aware Processing** (`core/parallel.py`)
   - Sophisticated work distribution
   - CPU affinity management
   - Memory locality optimization

## 2. Data Structures for GPU Parallelization

### 2.1 Card Representation

Current representation is object-oriented but can be optimized for GPU:

```python
# Current (OOP)
@dataclass
class Card:
    rank: str  # '2'-'A'
    suit: str  # '♠','♥','♦','♣'
    
# GPU-Optimized (bit-packed)
# Single 8-bit integer: RRRRSSXX
# - 4 bits for rank (0-12)
# - 2 bits for suit (0-3)  
# - 2 bits padding
```

### 2.2 Deck State

Transform from object arrays to GPU-friendly structures:

```python
# Current
available_cards: List[Card]
removed_cards: Set[Card]

# GPU-Optimized
deck_bitmap: uint64  # 52-bit mask for available cards
dealt_mask: uint32   # Track dealt cards per simulation
```

### 2.3 Hand Evaluation Tables

Pre-computed lookup tables for GPU constant memory:

```python
# Straight patterns (can be bit patterns)
STRAIGHT_MASKS = [0x1F00, 0x0F80, ...]  # 5-card straight patterns

# Flush detection (suit masks)
SUIT_MASKS = [0x1111111111111, ...]     # Every 4th bit for each suit

# Rank lookup tables
RANK_PRIME_PRODUCTS = [...]  # Prime number products for unique hand IDs
```

## 3. GPU Parallelization Opportunities

### 3.1 Embarrassingly Parallel Simulations

Each Monte Carlo simulation is independent, making it ideal for GPU:

```cuda
__global__ void simulate_hands_kernel(
    uint8_t* hero_cards,        // 2 cards per hero
    uint8_t* board_cards,       // 0-5 community cards
    int num_opponents,
    int num_simulations,
    float* win_counts,          // Output: wins per thread
    float* tie_counts,          // Output: ties per thread
    uint32_t* rng_states        // Per-thread RNG state
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_simulations) return;
    
    // Each thread simulates one complete hand
    // 1. Initialize local RNG
    // 2. Deal opponent cards
    // 3. Complete board if needed
    // 4. Evaluate all hands
    // 5. Determine winner
    // 6. Update counts
}
```

**Parallelization Strategy:**
- 1 thread = 1 complete simulation
- Block size: 256-512 threads (tunable)
- Grid size: (num_simulations + block_size - 1) / block_size

### 3.2 Hand Evaluation Acceleration

Implement bit-manipulation based hand evaluation:

```cuda
__device__ int evaluate_7_cards_gpu(uint8_t* cards) {
    // Bit-parallel evaluation
    uint64_t rank_mask = 0;
    uint16_t suit_counts[4] = {0};
    
    // Build masks in parallel
    for (int i = 0; i < 7; i++) {
        int rank = cards[i] >> 4;
        int suit = (cards[i] >> 2) & 0x3;
        rank_mask |= (1ULL << rank);
        suit_counts[suit]++;
    }
    
    // Check flush (5+ of same suit)
    bool is_flush = false;
    int flush_suit = -1;
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] >= 5) {
            is_flush = true;
            flush_suit = s;
            break;
        }
    }
    
    // Fast straight detection using bit patterns
    // Fast pair/trips/quads detection using lookup tables
    // Return hand rank + tiebreakers
}
```

### 3.3 RNG Optimization

Replace CPU random generation with GPU-optimized PRNGs:

```cuda
// Xorshift RNG for GPU
__device__ uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Deal random cards efficiently
__device__ void deal_cards_gpu(
    uint32_t* rng_state,
    uint64_t available_mask,
    uint8_t* output_cards,
    int num_cards
) {
    // Fisher-Yates shuffle using bit operations
}
```

### 3.4 Memory Access Patterns

Optimize for GPU memory hierarchy:

```cuda
// Constant memory for lookup tables
__constant__ uint64_t STRAIGHT_PATTERNS[10];
__constant__ uint32_t RANK_PRODUCTS[13];

// Shared memory for intermediate results
__shared__ float block_win_counts[BLOCK_SIZE];
__shared__ float block_tie_counts[BLOCK_SIZE];

// Coalesced global memory access
// Structure arrays for better access patterns
```

## 4. Specific Algorithms for GPU Acceleration

### 4.1 Parallel Reduction for Results

```cuda
__global__ void reduce_results_kernel(
    float* win_counts,
    float* tie_counts,
    float* loss_counts,
    int num_simulations,
    float* total_wins,
    float* total_ties,
    float* total_losses
) {
    // Efficient parallel reduction using shared memory
    extern __shared__ float sdata[];
    // ... reduction logic
}
```

### 4.2 Stratified Sampling on GPU

Implement importance sampling strategies:

```cuda
__device__ void stratified_sample_gpu(
    uint32_t* rng_state,
    int stratum_id,
    uint8_t* hero_cards,
    uint8_t* board_cards,
    StratumConfig* strata
) {
    // Generate samples biased toward specific hand strengths
    // Adjust weights for unbiased estimates
}
```

### 4.3 Batched Hand Evaluation

Process multiple hand combinations in parallel:

```cuda
__global__ void evaluate_hand_combinations_kernel(
    uint8_t* all_cards,      // 7 cards per hand
    int num_hands,
    int* hand_ranks,         // Output: best rank per hand
    int* tiebreakers         // Output: tiebreaker values
) {
    // Each thread evaluates C(7,5) = 21 combinations
    // Find best 5-card combination
}
```

## 5. Implementation Recommendations

### 5.1 Hybrid CPU-GPU Approach

1. **Keep on CPU:**
   - High-level game logic
   - Complex multi-way analysis
   - ICM calculations
   - Result aggregation and reporting

2. **Move to GPU:**
   - Monte Carlo simulation loop
   - Hand evaluation
   - Card dealing/shuffling
   - Win/loss counting

### 5.2 GPU Framework Options

1. **CUDA** (NVIDIA only)
   - Best performance
   - Mature ecosystem
   - CuPy for Python integration

2. **OpenCL** (Cross-platform)
   - Works on AMD, Intel, NVIDIA
   - PyOpenCL for Python
   - Slightly more complex

3. **PyTorch/JAX** (High-level)
   - Easier integration
   - Automatic differentiation (not needed here)
   - Good for prototyping

### 5.3 Expected Performance Improvements

Based on the analysis:

- **Hand Evaluation**: 50-100x speedup (bit operations + parallelism)
- **Simulation Loop**: 20-50x speedup (massive parallelism)
- **Overall System**: 10-30x speedup (including CPU-GPU transfer overhead)

**Estimated GPU Performance:**
- RTX 3080: ~10-20 million simulations/second
- RTX 4090: ~20-40 million simulations/second
- A100: ~30-60 million simulations/second

### 5.4 Memory Requirements

- **Per simulation**: ~64 bytes (cards, RNG state, results)
- **100K simulations**: ~6.4 MB (fits in GPU shared memory)
- **1M simulations**: ~64 MB (requires streaming)

## 6. Bottleneck Priority Ranking

1. **Hand Evaluation** (40% of runtime)
   - Highest impact
   - Well-suited for GPU
   - Bit manipulation benefits

2. **Simulation Loop** (35% of runtime)
   - Embarrassingly parallel
   - Simple memory patterns
   - Easy to implement

3. **Random Number Generation** (15% of runtime)
   - GPU-optimized RNGs available
   - Can generate in parallel

4. **Result Aggregation** (10% of runtime)
   - Parallel reduction
   - Can overlap with computation

## Conclusion

The Poker Knight Monte Carlo engine is an excellent candidate for GPU acceleration. The core algorithms are embarrassingly parallel, the data structures can be optimized for GPU memory patterns, and the potential speedup is significant (10-30x overall). The main implementation effort would be in:

1. Converting card representations to bit-packed formats
2. Implementing GPU-optimized hand evaluation
3. Managing CPU-GPU memory transfers efficiently
4. Integrating GPU kernels with the existing Python codebase

The hybrid approach maintaining high-level logic on CPU while offloading compute-intensive simulations to GPU would provide the best balance of performance and maintainability.