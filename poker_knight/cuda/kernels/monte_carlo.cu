/**
 * Monte Carlo simulation kernels for Poker Knight
 * 
 * High-performance CUDA kernels for parallel poker hand evaluation
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdint.h>

#define THREADS_PER_BLOCK 256
#define WARP_SIZE 32
#define MAX_OPPONENTS 9
#define DECK_SIZE 52

// Card representation
typedef uint8_t Card;
typedef uint64_t DeckMask;

// Hand ranking constants
#define STRAIGHT_FLUSH  8
#define FOUR_OF_A_KIND  7
#define FULL_HOUSE      6
#define FLUSH           5
#define STRAIGHT        4
#define THREE_OF_A_KIND 3
#define TWO_PAIR        2
#define ONE_PAIR        1
#define HIGH_CARD       0

// Constant memory for lookup tables
__constant__ uint32_t d_flush_lookup[8192];
__constant__ uint32_t d_straight_lookup[8192];
__constant__ uint16_t d_prime_products[13];

// Device functions
__device__ inline Card make_card(int rank, int suit) {
    return (rank & 0xF) | ((suit & 0x3) << 4) | 0x80;
}

__device__ inline int get_rank(Card c) { return c & 0xF; }
__device__ inline int get_suit(Card c) { return (c >> 4) & 0x3; }
__device__ inline bool is_valid(Card c) { return c & 0x80; }

__device__ inline int card_to_deck_position(Card c) {
    return get_rank(c) * 4 + get_suit(c);
}

__device__ inline void remove_card_from_deck(DeckMask& deck, Card c) {
    int pos = card_to_deck_position(c);
    deck &= ~(1ULL << pos);
}

__device__ Card draw_random_card(DeckMask& deck, curandState* state) {
    int remaining = __popcll(deck);
    if (remaining == 0) return 0;  // Invalid card
    
    int nth = curand(state) % remaining;
    int count = 0;
    
    // Find nth set bit
    for (int pos = 0; pos < 52; pos++) {
        if (deck & (1ULL << pos)) {
            if (count == nth) {
                deck &= ~(1ULL << pos);  // Remove from deck
                return make_card(pos / 4, pos % 4);
            }
            count++;
        }
    }
    
    return 0;  // Should never reach here
}

// Fast 7-card hand evaluation
__device__ uint32_t evaluate_7_cards(const Card cards[7]) {
    // Separate cards by suit
    uint16_t suit_masks[4] = {0, 0, 0, 0};
    uint8_t rank_counts[13] = {0};
    
    for (int i = 0; i < 7; i++) {
        if (!is_valid(cards[i])) continue;
        
        int rank = get_rank(cards[i]);
        int suit = get_suit(cards[i]);
        
        suit_masks[suit] |= (1 << rank);
        rank_counts[rank]++;
    }
    
    // Check for flush
    uint32_t best_hand = 0;
    for (int suit = 0; suit < 4; suit++) {
        int bits = __popc(suit_masks[suit]);
        if (bits >= 5) {
            uint32_t flush_rank = d_flush_lookup[suit_masks[suit]];
            best_hand = max(best_hand, (FLUSH << 20) | flush_rank);
            
            // Check for straight flush
            uint32_t sf_rank = d_straight_lookup[suit_masks[suit]];
            if (sf_rank > 0) {
                best_hand = max(best_hand, (STRAIGHT_FLUSH << 20) | sf_rank);
            }
        }
    }
    
    // Combined ranks for straight check
    uint16_t all_ranks = suit_masks[0] | suit_masks[1] | suit_masks[2] | suit_masks[3];
    uint32_t straight_rank = d_straight_lookup[all_ranks];
    if (straight_rank > 0) {
        best_hand = max(best_hand, (STRAIGHT << 20) | straight_rank);
    }
    
    // Count pairs, trips, quads
    int pairs = 0, trips = 0, quads = 0;
    uint32_t pair_ranks = 0, trip_ranks = 0, quad_ranks = 0;
    
    for (int rank = 12; rank >= 0; rank--) {
        switch (rank_counts[rank]) {
            case 4:
                quads++;
                quad_ranks = (quad_ranks << 4) | rank;
                break;
            case 3:
                trips++;
                trip_ranks = (trip_ranks << 4) | rank;
                break;
            case 2:
                pairs++;
                pair_ranks = (pair_ranks << 4) | rank;
                break;
        }
    }
    
    // Determine hand type
    if (quads > 0) {
        // Four of a kind
        uint32_t kicker = 0;
        for (int rank = 12; rank >= 0; rank--) {
            if (rank_counts[rank] > 0 && rank != (quad_ranks & 0xF)) {
                kicker = rank;
                break;
            }
        }
        best_hand = max(best_hand, (FOUR_OF_A_KIND << 20) | (quad_ranks << 4) | kicker);
    }
    else if (trips > 0 && pairs > 0) {
        // Full house
        uint32_t trip_rank = trip_ranks & 0xF;  // Highest trip
        uint32_t pair_rank = (pairs > 1) ? (pair_ranks >> 4) & 0xF : pair_ranks & 0xF;
        best_hand = max(best_hand, (FULL_HOUSE << 20) | (trip_rank << 4) | pair_rank);
    }
    else if (trips > 0) {
        // Three of a kind
        uint32_t kickers = 0;
        int kicker_count = 0;
        for (int rank = 12; rank >= 0 && kicker_count < 2; rank--) {
            if (rank_counts[rank] > 0 && rank != (trip_ranks & 0xF)) {
                kickers = (kickers << 4) | rank;
                kicker_count++;
            }
        }
        best_hand = max(best_hand, (THREE_OF_A_KIND << 20) | (trip_ranks << 8) | kickers);
    }
    else if (pairs >= 2) {
        // Two pair
        uint32_t top_pairs = pair_ranks >> (max(0, (pairs - 2)) * 4);
        uint32_t kicker = 0;
        for (int rank = 12; rank >= 0; rank--) {
            if (rank_counts[rank] > 0 && 
                rank != ((top_pairs >> 4) & 0xF) && 
                rank != (top_pairs & 0xF)) {
                kicker = rank;
                break;
            }
        }
        best_hand = max(best_hand, (TWO_PAIR << 20) | (top_pairs << 4) | kicker);
    }
    else if (pairs == 1) {
        // One pair
        uint32_t kickers = 0;
        int kicker_count = 0;
        for (int rank = 12; rank >= 0 && kicker_count < 3; rank--) {
            if (rank_counts[rank] > 0 && rank != (pair_ranks & 0xF)) {
                kickers = (kickers << 4) | rank;
                kicker_count++;
            }
        }
        best_hand = max(best_hand, (ONE_PAIR << 20) | (pair_ranks << 12) | kickers);
    }
    else {
        // High card
        uint32_t high_cards = 0;
        int count = 0;
        for (int rank = 12; rank >= 0 && count < 5; rank--) {
            if (rank_counts[rank] > 0) {
                high_cards = (high_cards << 4) | rank;
                count++;
            }
        }
        best_hand = max(best_hand, (HIGH_CARD << 20) | high_cards);
    }
    
    return best_hand;
}

// Warp-level reduction for win/tie counts
__device__ void warp_reduce(volatile uint32_t* sdata, int tid) {
    if (tid < 32) {
        sdata[tid] += sdata[tid + 32];
        sdata[tid] += sdata[tid + 16];
        sdata[tid] += sdata[tid + 8];
        sdata[tid] += sdata[tid + 4];
        sdata[tid] += sdata[tid + 2];
        sdata[tid] += sdata[tid + 1];
    }
}

// Main Monte Carlo kernel
__global__ void monte_carlo_kernel(
    const Card* hero_hand,
    const Card* board_cards,
    int board_size,
    int num_opponents,
    int simulations_per_thread,
    uint32_t* global_wins,
    uint32_t* global_ties,
    curandState* rng_states
) {
    extern __shared__ uint32_t shared_mem[];
    uint32_t* shared_wins = shared_mem;
    uint32_t* shared_ties = &shared_mem[blockDim.x];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    shared_wins[tid] = 0;
    shared_ties[tid] = 0;
    
    // Load RNG state
    curandState local_state = rng_states[gid];
    
    // Thread-local counters
    uint32_t wins = 0, ties = 0;
    
    // Run simulations
    for (int sim = 0; sim < simulations_per_thread; sim++) {
        // Initialize deck
        DeckMask deck = 0xFFFFFFFFFFFFF;  // All 52 cards available
        
        // Remove hero cards
        remove_card_from_deck(deck, hero_hand[0]);
        remove_card_from_deck(deck, hero_hand[1]);
        
        // Remove known board cards
        for (int i = 0; i < board_size; i++) {
            remove_card_from_deck(deck, board_cards[i]);
        }
        
        // Complete the board
        Card full_board[5];
        for (int i = 0; i < board_size; i++) {
            full_board[i] = board_cards[i];
        }
        for (int i = board_size; i < 5; i++) {
            full_board[i] = draw_random_card(deck, &local_state);
        }
        
        // Evaluate hero hand
        Card hero_7cards[7] = {
            hero_hand[0], hero_hand[1],
            full_board[0], full_board[1], full_board[2],
            full_board[3], full_board[4]
        };
        uint32_t hero_rank = evaluate_7_cards(hero_7cards);
        
        // Evaluate opponents
        uint32_t best_opponent_rank = 0;
        for (int opp = 0; opp < num_opponents; opp++) {
            Card opp_cards[2];
            opp_cards[0] = draw_random_card(deck, &local_state);
            opp_cards[1] = draw_random_card(deck, &local_state);
            
            Card opp_7cards[7] = {
                opp_cards[0], opp_cards[1],
                full_board[0], full_board[1], full_board[2],
                full_board[3], full_board[4]
            };
            uint32_t opp_rank = evaluate_7_cards(opp_7cards);
            
            best_opponent_rank = max(best_opponent_rank, opp_rank);
        }
        
        // Update counters
        if (hero_rank > best_opponent_rank) {
            wins++;
        } else if (hero_rank == best_opponent_rank) {
            ties++;
        }
    }
    
    // Store RNG state
    rng_states[gid] = local_state;
    
    // Add to shared memory
    shared_wins[tid] = wins;
    shared_ties[tid] = ties;
    __syncthreads();
    
    // Reduce within block
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            shared_wins[tid] += shared_wins[tid + s];
            shared_ties[tid] += shared_ties[tid + s];
        }
        __syncthreads();
    }
    
    // Warp reduction
    if (tid < 32) {
        warp_reduce(shared_wins, tid);
        warp_reduce(shared_ties, tid);
    }
    
    // Write block result
    if (tid == 0) {
        atomicAdd(&global_wins[0], shared_wins[0]);
        atomicAdd(&global_ties[0], shared_ties[0]);
    }
}

// RNG initialization kernel
__global__ void init_rng_kernel(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

// Batch processing kernel for multiple hands
__global__ void batch_monte_carlo_kernel(
    const Card* all_hero_hands,      // 2 * num_hands cards
    const Card* all_board_cards,     // 5 * num_hands cards (padded)
    const int* board_sizes,          // num_hands sizes
    const int* num_opponents,        // num_hands opponent counts
    int num_hands,
    int simulations_per_thread,
    uint32_t* results,               // 3 * num_hands (wins, ties, total)
    curandState* rng_states
) {
    int hand_idx = blockIdx.y;
    if (hand_idx >= num_hands) return;
    
    // Similar to single hand kernel but processes specific hand
    // Implementation follows same pattern...
}