// Improved CUDA kernel for Monte Carlo poker simulation

#include <cuda_runtime.h>

typedef unsigned char Card;
typedef unsigned int uint32_t;

// Constants
#define NUM_RANKS 13
#define NUM_SUITS 4
#define DECK_SIZE 52
#define MAX_PLAYERS 10

// Extract rank from card (0-12)
__device__ inline int get_rank(Card c) {
    return c & 0xF;
}

// Extract suit from card (0-3)
__device__ inline int get_suit(Card c) {
    return (c >> 4) & 0x3;
}

// Create a card from rank and suit
__device__ inline Card make_card(int rank, int suit) {
    return ((rank & 0xF) | ((suit & 0x3) << 4) | 0x80);
}

// Convert card to deck index (0-51)
__device__ inline int card_to_index(Card c) {
    return get_rank(c) * 4 + get_suit(c);
}

// Simple XorShift RNG
__device__ uint32_t xorshift32(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

// Fisher-Yates shuffle for remaining deck
__device__ void shuffle_deck(Card* deck, int deck_size, uint32_t* rng_state) {
    for (int i = deck_size - 1; i > 0; i--) {
        int j = xorshift32(rng_state) % (i + 1);
        Card temp = deck[i];
        deck[i] = deck[j];
        deck[j] = temp;
    }
}

// Evaluate hand strength (simplified but more accurate)
__device__ uint32_t evaluate_hand(Card* hand, int num_cards) {
    int rank_counts[NUM_RANKS] = {0};
    int suit_counts[NUM_SUITS] = {0};
    
    // Count ranks and suits
    for (int i = 0; i < num_cards; i++) {
        if (hand[i] & 0x80) {  // Valid card
            rank_counts[get_rank(hand[i])]++;
            suit_counts[get_suit(hand[i])]++;
        }
    }
    
    // Check for flush
    bool has_flush = false;
    for (int s = 0; s < NUM_SUITS; s++) {
        if (suit_counts[s] >= 5) {
            has_flush = true;
            break;
        }
    }
    
    // Count pairs, trips, quads
    int pairs = 0, trips = 0, quads = 0;
    int pair_ranks[2] = {-1, -1};
    int trip_rank = -1, quad_rank = -1;
    
    for (int r = NUM_RANKS - 1; r >= 0; r--) {
        if (rank_counts[r] == 4) {
            quad_rank = r;
            quads++;
        } else if (rank_counts[r] == 3) {
            trip_rank = r;
            trips++;
        } else if (rank_counts[r] == 2) {
            if (pairs < 2) pair_ranks[pairs] = r;
            pairs++;
        }
    }
    
    // Check for straight
    bool has_straight = false;
    int straight_high = -1;
    
    // Check A-5 straight first
    if (rank_counts[12] > 0 && rank_counts[0] > 0 && 
        rank_counts[1] > 0 && rank_counts[2] > 0 && rank_counts[3] > 0) {
        has_straight = true;
        straight_high = 3;  // 5-high straight
    }
    
    // Check other straights
    for (int r = NUM_RANKS - 1; r >= 4; r--) {
        bool is_straight = true;
        for (int i = 0; i < 5; i++) {
            if (rank_counts[r - i] == 0) {
                is_straight = false;
                break;
            }
        }
        if (is_straight) {
            has_straight = true;
            straight_high = r;
            break;
        }
    }
    
    // Calculate hand value
    // Format: [hand_type(4 bits)][primary(4 bits)][secondary(4 bits)][kickers(20 bits)]
    uint32_t hand_value = 0;
    
    if (has_straight && has_flush) {
        // Straight flush (need to verify same suit)
        hand_value = (8 << 28) | (straight_high << 24);
    } else if (quads > 0) {
        // Four of a kind
        hand_value = (7 << 28) | (quad_rank << 24);
    } else if (trips > 0 && pairs > 0) {
        // Full house
        hand_value = (6 << 28) | (trip_rank << 24) | (pair_ranks[0] << 20);
    } else if (has_flush) {
        // Flush
        hand_value = (5 << 28);
        // Add high cards
        int kicker_shift = 24;
        for (int r = NUM_RANKS - 1; r >= 0 && kicker_shift >= 4; r--) {
            if (rank_counts[r] > 0) {
                hand_value |= (r << kicker_shift);
                kicker_shift -= 4;
            }
        }
    } else if (has_straight) {
        // Straight
        hand_value = (4 << 28) | (straight_high << 24);
    } else if (trips > 0) {
        // Three of a kind
        hand_value = (3 << 28) | (trip_rank << 24);
    } else if (pairs >= 2) {
        // Two pair
        hand_value = (2 << 28) | (pair_ranks[0] << 24) | (pair_ranks[1] << 20);
    } else if (pairs == 1) {
        // One pair
        hand_value = (1 << 28) | (pair_ranks[0] << 24);
        // Add kickers
        int kicker_shift = 20;
        for (int r = NUM_RANKS - 1; r >= 0 && kicker_shift >= 4; r--) {
            if (rank_counts[r] == 1) {
                hand_value |= (r << kicker_shift);
                kicker_shift -= 4;
            }
        }
    } else {
        // High card
        hand_value = 0;
        int kicker_shift = 24;
        for (int r = NUM_RANKS - 1; r >= 0 && kicker_shift >= 4; r--) {
            if (rank_counts[r] > 0) {
                hand_value |= (r << kicker_shift);
                kicker_shift -= 4;
            }
        }
    }
    
    return hand_value;
}

// Monte Carlo simulation kernel
extern "C" __global__ void monte_carlo_improved(
    const Card* hero_hand,
    const Card* board_cards,
    int board_size,
    int num_opponents,
    int simulations_per_thread,
    uint32_t* wins,
    uint32_t* ties,
    uint32_t seed
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize RNG
    uint32_t rng_state = seed + tid * 1103515245;
    
    // Local counters
    uint32_t thread_wins = 0;
    uint32_t thread_ties = 0;
    
    // Shared memory for deck
    __shared__ Card shared_deck[DECK_SIZE];
    
    // Initialize deck in shared memory (once per block)
    if (threadIdx.x < DECK_SIZE) {
        int rank = threadIdx.x / 4;
        int suit = threadIdx.x % 4;
        shared_deck[threadIdx.x] = make_card(rank, suit);
    }
    __syncthreads();
    
    // Run simulations
    for (int sim = 0; sim < simulations_per_thread; sim++) {
        // Create deck copy in local memory
        Card deck[DECK_SIZE];
        bool used[DECK_SIZE] = {false};
        
        // Mark hero cards as used
        used[card_to_index(hero_hand[0])] = true;
        used[card_to_index(hero_hand[1])] = true;
        
        // Mark board cards as used
        for (int i = 0; i < board_size; i++) {
            if (board_cards[i] & 0x80) {
                used[card_to_index(board_cards[i])] = true;
            }
        }
        
        // Build remaining deck
        int deck_size = 0;
        for (int i = 0; i < DECK_SIZE; i++) {
            if (!used[i]) {
                deck[deck_size++] = shared_deck[i];
            }
        }
        
        // Shuffle remaining deck
        shuffle_deck(deck, deck_size, &rng_state);
        
        // Deal remaining board cards
        Card full_board[5];
        for (int i = 0; i < 5; i++) {
            if (i < board_size) {
                full_board[i] = board_cards[i];
            } else {
                full_board[i] = deck[--deck_size];
            }
        }
        
        // Build hero's 7-card hand
        Card hero_full[7];
        hero_full[0] = hero_hand[0];
        hero_full[1] = hero_hand[1];
        for (int i = 0; i < 5; i++) {
            hero_full[2 + i] = full_board[i];
        }
        
        // Evaluate hero hand
        uint32_t hero_value = evaluate_hand(hero_full, 7);
        
        // Evaluate opponents
        uint32_t best_opponent_value = 0;
        
        for (int opp = 0; opp < num_opponents; opp++) {
            // Deal opponent cards
            Card opp_hand[7];
            opp_hand[0] = deck[--deck_size];
            opp_hand[1] = deck[--deck_size];
            for (int i = 0; i < 5; i++) {
                opp_hand[2 + i] = full_board[i];
            }
            
            uint32_t opp_value = evaluate_hand(opp_hand, 7);
            if (opp_value > best_opponent_value) {
                best_opponent_value = opp_value;
            }
        }
        
        // Compare results
        if (hero_value > best_opponent_value) {
            thread_wins++;
        } else if (hero_value == best_opponent_value) {
            thread_ties++;
        }
    }
    
    // Accumulate results
    atomicAdd(&wins[0], thread_wins);
    atomicAdd(&ties[0], thread_ties);
}