// Optimized CUDA kernel with better GPU utilization

#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef unsigned char Card;
typedef unsigned int uint32_t;

#define NUM_RANKS 13
#define NUM_SUITS 4
#define DECK_SIZE 52
#define WARP_SIZE 32

// Shared memory for deck - one per warp to reduce conflicts
__shared__ Card shared_decks[8][DECK_SIZE];  // 8 warps per block max

__device__ inline int get_rank(Card c) { return c & 0xF; }
__device__ inline int get_suit(Card c) { return (c >> 4) & 0x3; }
__device__ inline Card make_card(int rank, int suit) {
    return ((rank & 0xF) | ((suit & 0x3) << 4) | 0x80);
}
__device__ inline int card_to_index(Card c) {
    return get_rank(c) * 4 + get_suit(c);
}

// Better random number generator using cuRAND
__device__ uint32_t lcg_rand(curandState_t* state) {
    return curand(state);
}

// Optimized shuffle using shared memory
__device__ void shuffle_deck_optimized(Card* deck, int deck_size, curandState_t* state) {
    for (int i = deck_size - 1; i > 0; i--) {
        int j = curand(state) % (i + 1);
        Card temp = deck[i];
        deck[i] = deck[j];
        deck[j] = temp;
    }
}

// More complex hand evaluation with better bit manipulation
__device__ uint32_t evaluate_hand_optimized(Card* hand, int num_cards) {
    uint32_t rank_mask = 0;
    uint32_t suit_masks[4] = {0, 0, 0, 0};
    int rank_counts[NUM_RANKS] = {0};
    
    // Build masks and counts in single pass
    #pragma unroll
    for (int i = 0; i < num_cards; i++) {
        if (hand[i] & 0x80) {
            int rank = get_rank(hand[i]);
            int suit = get_suit(hand[i]);
            rank_mask |= (1 << rank);
            suit_masks[suit] |= (1 << rank);
            rank_counts[rank]++;
        }
    }
    
    // Check for flush using bit manipulation
    bool has_flush = false;
    int flush_mask = 0;
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        if (__popc(suit_masks[s]) >= 5) {
            has_flush = true;
            flush_mask = suit_masks[s];
            break;
        }
    }
    
    // Check for straights using bit patterns
    const uint32_t straight_patterns[10] = {
        0x100F, // A-5 (0001 0000 0000 1111)
        0x001F, // 2-6
        0x003E, // 3-7
        0x007C, // 4-8
        0x00F8, // 5-9
        0x01F0, // 6-10
        0x03E0, // 7-J
        0x07C0, // 8-Q
        0x0F80, // 9-K
        0x1F00  // 10-A
    };
    
    int straight_high = -1;
    #pragma unroll
    for (int i = 9; i >= 0; i--) {
        if ((rank_mask & straight_patterns[i]) == straight_patterns[i]) {
            straight_high = i + 4; // Adjust for actual high card
            break;
        }
    }
    
    // Count pairs, trips, quads with bit manipulation
    uint32_t pairs = 0, trips = 0, quads = 0;
    int pair_ranks[2] = {-1, -1}, trip_rank = -1, quad_rank = -1;
    int num_pairs = 0;
    
    #pragma unroll
    for (int r = 12; r >= 0; r--) {
        if (rank_counts[r] == 4) {
            quad_rank = r;
            quads++;
        } else if (rank_counts[r] == 3) {
            trip_rank = r;
            trips++;
        } else if (rank_counts[r] == 2) {
            if (num_pairs < 2) pair_ranks[num_pairs++] = r;
            pairs++;
        }
    }
    
    // Build hand value with optimized encoding
    uint32_t hand_value = 0;
    
    if (has_flush && straight_high >= 0) {
        // Check if straight flush
        uint32_t straight_flush_mask = 0;
        for (int i = 0; i < 5; i++) {
            straight_flush_mask |= (1 << ((straight_high - 4 + i) % 13));
        }
        if ((flush_mask & straight_flush_mask) == straight_flush_mask) {
            hand_value = (8 << 28) | (straight_high << 24);
        } else {
            // Find highest bit in flush_mask
            int highest_bit = 31 - __clz(flush_mask);
            hand_value = (5 << 28) | (highest_bit << 24);
        }
    } else if (quads > 0) {
        hand_value = (7 << 28) | (quad_rank << 24);
    } else if (trips > 0 && pairs > 0) {
        hand_value = (6 << 28) | (trip_rank << 24) | (pair_ranks[0] << 20);
    } else if (has_flush) {
        hand_value = (5 << 28);
        int shift = 24;
        for (int r = 12; r >= 0 && shift >= 4; r--) {
            if (flush_mask & (1 << r)) {
                hand_value |= (r << shift);
                shift -= 4;
            }
        }
    } else if (straight_high >= 0) {
        hand_value = (4 << 28) | (straight_high << 24);
    } else if (trips > 0) {
        hand_value = (3 << 28) | (trip_rank << 24);
    } else if (pairs >= 2) {
        hand_value = (2 << 28) | (pair_ranks[0] << 24) | (pair_ranks[1] << 20);
    } else if (pairs == 1) {
        hand_value = (1 << 28) | (pair_ranks[0] << 24);
        int shift = 20;
        for (int r = 12; r >= 0 && shift >= 4; r--) {
            if (rank_counts[r] == 1) {
                hand_value |= (r << shift);
                shift -= 4;
            }
        }
    } else {
        int shift = 24;
        for (int r = 12; r >= 0 && shift >= 4; r--) {
            if (rank_counts[r] > 0) {
                hand_value |= (r << shift);
                shift -= 4;
            }
        }
    }
    
    return hand_value;
}

extern "C" __global__ void monte_carlo_optimized(
    const Card* hero_hand,
    const Card* board_cards,
    int board_size,
    int num_opponents,
    int simulations_per_thread,
    uint32_t* wins,
    uint32_t* ties,
    uint64_t seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;
    
    // Initialize cuRAND
    curandState_t rng_state;
    curand_init(seed + tid, 0, 0, &rng_state);
    
    // Get shared deck for this warp
    Card* warp_deck = shared_decks[warp_id];
    
    // Initialize deck in shared memory (coalesced)
    if (lane_id < 26) {
        int idx1 = lane_id;
        int idx2 = lane_id + 26;
        warp_deck[idx1] = make_card(idx1 / 4, idx1 % 4);
        if (idx2 < DECK_SIZE) {
            warp_deck[idx2] = make_card(idx2 / 4, idx2 % 4);
        }
    }
    __syncwarp();
    
    // Local counters
    uint32_t thread_wins = 0;
    uint32_t thread_ties = 0;
    
    // Run simulations
    for (int sim = 0; sim < simulations_per_thread; sim++) {
        // Build used card mask
        uint64_t used_mask = 0;
        used_mask |= (1ULL << card_to_index(hero_hand[0]));
        used_mask |= (1ULL << card_to_index(hero_hand[1]));
        
        for (int i = 0; i < board_size; i++) {
            if (board_cards[i] & 0x80) {
                used_mask |= (1ULL << card_to_index(board_cards[i]));
            }
        }
        
        // Build remaining deck efficiently
        Card local_deck[DECK_SIZE];
        int deck_size = 0;
        
        #pragma unroll 8
        for (int i = 0; i < DECK_SIZE; i++) {
            if (!(used_mask & (1ULL << i))) {
                local_deck[deck_size++] = warp_deck[i];
            }
        }
        
        // Shuffle
        shuffle_deck_optimized(local_deck, deck_size, &rng_state);
        
        // Deal remaining board
        Card full_board[5];
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            if (i < board_size) {
                full_board[i] = board_cards[i];
            } else {
                full_board[i] = local_deck[--deck_size];
            }
        }
        
        // Build hero hand
        Card hero_full[7];
        hero_full[0] = hero_hand[0];
        hero_full[1] = hero_hand[1];
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            hero_full[2 + i] = full_board[i];
        }
        
        // Evaluate hero
        uint32_t hero_value = evaluate_hand_optimized(hero_full, 7);
        
        // Evaluate opponents
        uint32_t best_opponent = 0;
        
        for (int opp = 0; opp < num_opponents; opp++) {
            Card opp_hand[7];
            opp_hand[0] = local_deck[--deck_size];
            opp_hand[1] = local_deck[--deck_size];
            #pragma unroll
            for (int i = 0; i < 5; i++) {
                opp_hand[2 + i] = full_board[i];
            }
            
            uint32_t opp_value = evaluate_hand_optimized(opp_hand, 7);
            best_opponent = max(best_opponent, opp_value);
        }
        
        // Compare
        if (hero_value > best_opponent) {
            thread_wins++;
        } else if (hero_value == best_opponent) {
            thread_ties++;
        }
    }
    
    // Accumulate results
    atomicAdd(&wins[0], thread_wins);
    atomicAdd(&ties[0], thread_ties);
}