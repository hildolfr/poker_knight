# CUDA Kernel Specifications for Poker Knight

## Core Data Structures

### 1. Card Representation

```cuda
// 8-bit card representation
typedef uint8_t Card;

// Card encoding/decoding
__device__ inline Card make_card(int rank, int suit) {
    return (rank & 0xF) | ((suit & 0x3) << 4) | 0x80;  // Set valid bit
}

__device__ inline int get_rank(Card c) { return c & 0xF; }
__device__ inline int get_suit(Card c) { return (c >> 4) & 0x3; }
__device__ inline bool is_valid(Card c) { return c & 0x80; }
```

### 2. Deck as Bitmask

```cuda
// 64-bit deck representation (52 cards + padding)
typedef uint64_t DeckMask;

__device__ inline void remove_card(DeckMask& deck, Card c) {
    int pos = get_rank(c) * 4 + get_suit(c);
    deck &= ~(1ULL << pos);
}

__device__ inline Card draw_random_card(DeckMask& deck, curandState* state) {
    int count = __popcll(deck);  // Count remaining cards
    int nth = curand(state) % count;
    
    // Find nth set bit
    int pos = 0;
    for (int i = 0; i <= nth; i++) {
        pos = __ffsll(deck) - 1;
        deck &= ~(1ULL << pos);
    }
    
    return make_card(pos / 4, pos % 4);
}
```

## Main Kernels

### 1. Monte Carlo Simulation Kernel

```cuda
__global__ void simulate_hands_kernel(
    const Card* hero_hand,      // 2 cards
    const Card* board_cards,    // 0-5 cards
    int board_size,
    int num_opponents,
    int sims_per_thread,
    uint32_t* win_counts,       // Output: wins per thread
    uint32_t* tie_counts,       // Output: ties per thread
    curandState* rng_states
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState local_state = rng_states[tid];
    
    // Thread-local counters
    uint32_t wins = 0, ties = 0;
    
    // Shared memory for hero hand evaluation (computed once)
    __shared__ uint32_t hero_rank_cache[CACHE_SIZE];
    
    for (int sim = 0; sim < sims_per_thread; sim++) {
        // Initialize deck excluding known cards
        DeckMask deck = FULL_DECK_MASK;
        remove_card(deck, hero_hand[0]);
        remove_card(deck, hero_hand[1]);
        for (int i = 0; i < board_size; i++) {
            remove_card(deck, board_cards[i]);
        }
        
        // Complete the board
        Card final_board[5];
        for (int i = 0; i < board_size; i++) {
            final_board[i] = board_cards[i];
        }
        for (int i = board_size; i < 5; i++) {
            final_board[i] = draw_random_card(deck, &local_state);
        }
        
        // Evaluate hero hand
        uint32_t hero_rank = evaluate_7_cards(
            hero_hand[0], hero_hand[1],
            final_board[0], final_board[1], final_board[2],
            final_board[3], final_board[4]
        );
        
        // Simulate opponents
        uint32_t best_opponent_rank = 0;
        for (int opp = 0; opp < num_opponents; opp++) {
            Card opp_cards[2];
            opp_cards[0] = draw_random_card(deck, &local_state);
            opp_cards[1] = draw_random_card(deck, &local_state);
            
            uint32_t opp_rank = evaluate_7_cards(
                opp_cards[0], opp_cards[1],
                final_board[0], final_board[1], final_board[2],
                final_board[3], final_board[4]
            );
            
            best_opponent_rank = max(best_opponent_rank, opp_rank);
        }
        
        // Update counters
        if (hero_rank > best_opponent_rank) wins++;
        else if (hero_rank == best_opponent_rank) ties++;
    }
    
    // Write results
    win_counts[tid] = wins;
    tie_counts[tid] = ties;
    rng_states[tid] = local_state;
}
```

### 2. Hand Evaluation Kernel

```cuda
// Constant memory for lookup tables
__constant__ uint32_t flush_lookup[8192];      // 13-bit index
__constant__ uint32_t straight_lookup[8192];   // 13-bit index
__constant__ uint16_t rank_counts[52][52];     // Precomputed combinations

__device__ uint32_t evaluate_7_cards(
    Card c0, Card c1, Card c2, Card c3, Card c4, Card c5, Card c6
) {
    // Group cards by suit
    uint16_t suit_masks[4] = {0, 0, 0, 0};
    Card cards[7] = {c0, c1, c2, c3, c4, c5, c6};
    
    for (int i = 0; i < 7; i++) {
        int suit = get_suit(cards[i]);
        int rank = get_rank(cards[i]);
        suit_masks[suit] |= (1 << rank);
    }
    
    // Check for flush
    uint32_t best_rank = 0;
    for (int suit = 0; suit < 4; suit++) {
        int count = __popc(suit_masks[suit]);
        if (count >= 5) {
            // Use lookup table for flush evaluation
            uint32_t flush_rank = flush_lookup[suit_masks[suit]];
            best_rank = max(best_rank, flush_rank | (FLUSH_FLAG << 20));
        }
    }
    
    // Combined rank mask for straights
    uint16_t all_ranks = suit_masks[0] | suit_masks[1] | 
                         suit_masks[2] | suit_masks[3];
    
    // Check for straight
    uint32_t straight_rank = straight_lookup[all_ranks];
    if (straight_rank > 0) {
        best_rank = max(best_rank, straight_rank | (STRAIGHT_FLAG << 20));
    }
    
    // Count rank frequencies for pairs, trips, etc.
    uint8_t rank_freq[13] = {0};
    for (int i = 0; i < 7; i++) {
        rank_freq[get_rank(cards[i])]++;
    }
    
    // Evaluate high card combinations
    // ... (implementation continues with pair/trip/quad detection)
    
    return best_rank;
}
```

### 3. Batch Processing Kernel

```cuda
__global__ void batch_solve_kernel(
    const SimulationRequest* requests,  // Array of different hands to solve
    SimulationResult* results,          // Output array
    int num_requests,
    curandState* rng_states
) {
    int request_id = blockIdx.y;
    if (request_id >= num_requests) return;
    
    const SimulationRequest& req = requests[request_id];
    
    // Each block handles one request
    // Threads within block collaborate on simulations
    extern __shared__ uint32_t shared_results[];
    
    // ... (similar to single simulation but batched)
}
```

## Memory Management

### Texture Memory Usage

```cuda
// Bind lookup tables to texture memory for better cache behavior
texture<uint32_t, 1, cudaReadModeElementType> tex_flush_lookup;
texture<uint32_t, 1, cudaReadModeElementType> tex_straight_lookup;

__device__ uint32_t get_flush_rank(uint16_t mask) {
    return tex1Dfetch(tex_flush_lookup, mask);
}
```

### Shared Memory Optimization

```cuda
#define SHARED_MEMORY_SIZE 48KB
#define THREADS_PER_BLOCK 256

// Use shared memory for frequently accessed data
__shared__ uint32_t shared_eval_cache[EVAL_CACHE_SIZE];
__shared__ Card shared_board[5];

// Cooperative loading
if (threadIdx.x < 5 && threadIdx.x < board_size) {
    shared_board[threadIdx.x] = board_cards[threadIdx.x];
}
__syncthreads();
```

## Performance Tuning

### Grid Configuration

```cuda
// Optimal configuration based on GPU architecture
int get_optimal_blocks(int total_simulations, int device_id) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    
    int blocks_per_sm = prop.maxThreadsPerMultiProcessor / THREADS_PER_BLOCK;
    int total_blocks = prop.multiProcessorCount * blocks_per_sm;
    
    // Ensure enough work per thread
    int sims_per_thread = max(100, total_simulations / (total_blocks * THREADS_PER_BLOCK));
    
    return min(total_blocks, (total_simulations + sims_per_thread - 1) / sims_per_thread);
}
```

### Warp-Level Primitives

```cuda
// Fast reduction using warp shuffle
__device__ uint32_t warp_reduce_sum(uint32_t val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction
__device__ void block_reduce_results(uint32_t* wins, uint32_t* ties) {
    uint32_t thread_wins = *wins;
    uint32_t thread_ties = *ties;
    
    thread_wins = warp_reduce_sum(thread_wins);
    thread_ties = warp_reduce_sum(thread_ties);
    
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&shared_results[0], thread_wins);
        atomicAdd(&shared_results[1], thread_ties);
    }
}
```

## Error Handling

```cuda
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(error)); \
        return error; \
    } \
} while(0)

// Kernel error checking
__global__ void validate_kernel_params(/* params */) {
    if (num_opponents < 1 || num_opponents > 9) {
        // Set error flag in global memory
        atomicExch(&g_error_flag, ERROR_INVALID_OPPONENTS);
        return;
    }
    // ... additional validation
}
```