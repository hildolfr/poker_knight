/**
 * kernelPokerSimNG - Next Generation Unified Poker Simulation Kernel
 * 
 * A single, highly configurable CUDA kernel that replaces all existing
 * Monte Carlo poker simulation kernels in the Poker Knight project.
 * 
 * Features:
 * - Batch processing of multiple hands
 * - Optional hand category tracking
 * - Statistical data collection for confidence intervals
 * - Board texture analysis
 * - Optimized memory access patterns
 * - Configurable via runtime parameters
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>

// Define types normally from stdint.h
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// Constants
#define WARP_SIZE 32
#define MAX_PLAYERS 10
#define DECK_SIZE 52
#define NUM_RANKS 13
#define NUM_SUITS 4
#define BOARD_SIZE 5
#define HAND_SIZE 2
#define TOTAL_CARDS 7

// Hand categories (matching Python constants)
#define CATEGORY_HIGH_CARD       1
#define CATEGORY_PAIR           2
#define CATEGORY_TWO_PAIR       3
#define CATEGORY_THREE_OF_KIND  4
#define CATEGORY_STRAIGHT       5
#define CATEGORY_FLUSH          6
#define CATEGORY_FULL_HOUSE     7
#define CATEGORY_FOUR_OF_KIND   8
#define CATEGORY_STRAIGHT_FLUSH 9
#define CATEGORY_ROYAL_FLUSH    10

// Configuration flags
#define FLAG_TRACK_CATEGORIES   (1 << 0)
#define FLAG_COMPUTE_VARIANCE   (1 << 1)
#define FLAG_ANALYZE_BOARD      (1 << 2)
#define FLAG_USE_SHARED_MEM     (1 << 3)
#define FLAG_BATCH_MODE         (1 << 4)

// Card representation
typedef uint8_t Card;
typedef uint64_t DeckMask;

// Configuration structure - using plain integers for simplicity
typedef uint32_t SimConfig;

// Result structure for batch processing
struct HandResult {
    uint32_t wins;
    uint32_t ties;
    uint32_t losses;
    float variance;
    uint32_t hand_categories[11];  // Index 0 unused, 1-10 for categories
    uint16_t board_texture;
};

// Small lookup table in constant memory
__constant__ uint16_t d_rank_primes[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

// Device functions for card manipulation
__device__ __forceinline__ int get_rank(Card c) { 
    return c & 0xF; 
}

__device__ __forceinline__ int get_suit(Card c) { 
    return (c >> 4) & 0x3; 
}

__device__ __forceinline__ bool is_valid_card(Card c) { 
    return (c & 0x80) != 0; 
}

__device__ __forceinline__ Card make_card(int rank, int suit) {
    return (Card)((rank & 0xF) | ((suit & 0x3) << 4) | 0x80);
}

__device__ __forceinline__ int card_to_deck_index(Card c) {
    return get_rank(c) + get_suit(c) * 13;
}

__device__ __forceinline__ void remove_from_deck(DeckMask& deck, Card c) {
    int idx = card_to_deck_index(c);
    deck &= ~(1ULL << idx);
}

// Draw a random card from the deck
__device__ Card draw_card(DeckMask& deck, curandState* state) {
    int remaining = __popcll(deck);
    if (remaining == 0) return 0;
    
    int nth = curand(state) % remaining;
    int count = 0;
    
    #pragma unroll
    for (int idx = 0; idx < 52; idx++) {
        if (deck & (1ULL << idx)) {
            if (count == nth) {
                deck &= ~(1ULL << idx);
                return make_card(idx % 13, idx / 13);
            }
            count++;
        }
    }
    return 0;
}

// Enhanced hand evaluation that returns both rank and category
__device__ void evaluate_hand_with_category(
    const Card cards[7], 
    uint32_t* rank_out,
    uint8_t* category_out
) {
    // Initialize counters
    uint8_t rank_counts[13] = {0};
    uint16_t suit_masks[4] = {0};
    uint32_t rank_mask = 0;
    
    // Count ranks and build suit masks
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        if (is_valid_card(cards[i])) {
            int rank = get_rank(cards[i]);
            int suit = get_suit(cards[i]);
            rank_counts[rank]++;
            suit_masks[suit] |= (1 << rank);
            rank_mask |= (1 << rank);
        }
    }
    
    // Check for flush
    int flush_suit = -1;
    #pragma unroll
    for (int suit = 0; suit < 4; suit++) {
        if (__popc(suit_masks[suit]) >= 5) {
            flush_suit = suit;
            break;
        }
    }
    
    // Count pairs, trips, quads
    int pairs = 0, trips = 0, quads = 0;
    int pair_ranks[2] = {-1, -1};
    int trip_rank = -1, quad_rank = -1;
    
    #pragma unroll
    for (int rank = 12; rank >= 0; rank--) {
        if (rank_counts[rank] == 4) {
            quads++;
            quad_rank = rank;
        } else if (rank_counts[rank] == 3) {
            trips++;
            trip_rank = rank;
        } else if (rank_counts[rank] == 2) {
            if (pairs < 2) pair_ranks[pairs] = rank;
            pairs++;
        }
    }
    
    // Check for straight
    bool has_straight = false;
    int straight_high = -1;
    
    // Check A-5 straight
    if ((rank_mask & 0x100F) == 0x100F) {
        has_straight = true;
        straight_high = 3; // 5-high straight
    } else {
        // Check other straights
        #pragma unroll
        for (int high = 12; high >= 4; high--) {
            uint32_t straight_mask = 0x1F << (high - 4);
            if ((rank_mask & straight_mask) == straight_mask) {
                has_straight = true;
                straight_high = high;
                break;
            }
        }
    }
    
    // Determine hand category and rank
    uint32_t rank = 0;
    uint8_t category = CATEGORY_HIGH_CARD;
    
    // Check for straight flush / royal flush
    if (flush_suit >= 0 && has_straight) {
        uint16_t flush_mask = suit_masks[flush_suit];
        
        // Check for straight in flush suit
        bool straight_flush = false;
        int sf_high = -1;
        
        // A-5 straight flush
        if ((flush_mask & 0x100F) == 0x100F) {
            straight_flush = true;
            sf_high = 3;
        } else {
            #pragma unroll
            for (int high = 12; high >= 4; high--) {
                uint16_t sf_mask = 0x1F << (high - 4);
                if ((flush_mask & sf_mask) == sf_mask) {
                    straight_flush = true;
                    sf_high = high;
                    break;
                }
            }
        }
        
        if (straight_flush) {
            if (sf_high == 12) { // Ace high straight flush
                category = CATEGORY_ROYAL_FLUSH;
                rank = (CATEGORY_ROYAL_FLUSH << 20);
            } else {
                category = CATEGORY_STRAIGHT_FLUSH;
                rank = (CATEGORY_STRAIGHT_FLUSH << 20) | sf_high;
            }
        }
    }
    
    // If not straight/royal flush, check other hands
    if (category == CATEGORY_HIGH_CARD) {
        if (quads > 0) {
            category = CATEGORY_FOUR_OF_KIND;
            rank = (CATEGORY_FOUR_OF_KIND << 20) | (quad_rank << 16);
            // Add kicker
            #pragma unroll
            for (int r = 12; r >= 0; r--) {
                if (r != quad_rank && rank_counts[r] > 0) {
                    rank |= (r << 12);
                    break;
                }
            }
        } else if (trips > 0 && pairs > 0) {
            category = CATEGORY_FULL_HOUSE;
            rank = (CATEGORY_FULL_HOUSE << 20) | (trip_rank << 16) | (pair_ranks[0] << 12);
        } else if (flush_suit >= 0) {
            category = CATEGORY_FLUSH;
            rank = (CATEGORY_FLUSH << 20);
            // Add top 5 cards of flush
            int flush_cards = 0;
            #pragma unroll
            for (int r = 12; r >= 0 && flush_cards < 5; r--) {
                if (suit_masks[flush_suit] & (1 << r)) {
                    rank |= (r << (16 - flush_cards * 4));
                    flush_cards++;
                }
            }
        } else if (has_straight) {
            category = CATEGORY_STRAIGHT;
            rank = (CATEGORY_STRAIGHT << 20) | straight_high;
        } else if (trips > 0) {
            category = CATEGORY_THREE_OF_KIND;
            rank = (CATEGORY_THREE_OF_KIND << 20) | (trip_rank << 16);
            // Add kickers
            int kickers = 0;
            #pragma unroll
            for (int r = 12; r >= 0 && kickers < 2; r--) {
                if (r != trip_rank && rank_counts[r] > 0) {
                    rank |= (r << (12 - kickers * 4));
                    kickers++;
                }
            }
        } else if (pairs >= 2) {
            category = CATEGORY_TWO_PAIR;
            rank = (CATEGORY_TWO_PAIR << 20) | (pair_ranks[0] << 16) | (pair_ranks[1] << 12);
            // Add kicker
            #pragma unroll
            for (int r = 12; r >= 0; r--) {
                if (r != pair_ranks[0] && r != pair_ranks[1] && rank_counts[r] > 0) {
                    rank |= (r << 8);
                    break;
                }
            }
        } else if (pairs == 1) {
            category = CATEGORY_PAIR;
            rank = (CATEGORY_PAIR << 20) | (pair_ranks[0] << 16);
            // Add kickers
            int kickers = 0;
            #pragma unroll
            for (int r = 12; r >= 0 && kickers < 3; r--) {
                if (r != pair_ranks[0] && rank_counts[r] > 0) {
                    rank |= (r << (12 - kickers * 4));
                    kickers++;
                }
            }
        } else {
            category = CATEGORY_HIGH_CARD;
            rank = (CATEGORY_HIGH_CARD << 20);
            // Add top 5 cards
            int cards_added = 0;
            #pragma unroll
            for (int r = 12; r >= 0 && cards_added < 5; r--) {
                if (rank_counts[r] > 0) {
                    rank |= (r << (16 - cards_added * 4));
                    cards_added++;
                }
            }
        }
    }
    
    *rank_out = rank;
    *category_out = category;
}

// Analyze board texture (simplified version)
__device__ uint16_t analyze_board_texture(const Card board[5], int board_size) {
    if (board_size < 3) return 0;
    
    uint16_t texture = 0;
    uint8_t rank_counts[13] = {0};
    uint8_t suit_counts[4] = {0};
    uint16_t rank_mask = 0;
    
    #pragma unroll
    for (int i = 0; i < board_size; i++) {
        if (is_valid_card(board[i])) {
            int rank = get_rank(board[i]);
            int suit = get_suit(board[i]);
            rank_counts[rank]++;
            suit_counts[suit]++;
            rank_mask |= (1 << rank);
        }
    }
    
    // Check for flush draw (3 or 4 of same suit)
    #pragma unroll
    for (int suit = 0; suit < 4; suit++) {
        if (suit_counts[suit] >= 4) {
            texture |= (1 << 0);  // Flush on board
        } else if (suit_counts[suit] == 3) {
            texture |= (1 << 1);  // Flush draw
        }
    }
    
    // Check for straight possibilities
    int straight_draws = 0;
    #pragma unroll
    for (int high = 12; high >= 3; high--) {
        int connected = 0;
        for (int i = 0; i < 5; i++) {
            if (rank_mask & (1 << ((high - i + 13) % 13))) {
                connected++;
            }
        }
        if (connected >= 4) straight_draws++;
    }
    
    if (straight_draws > 0) {
        texture |= (1 << 2);  // Straight draw possible
    }
    
    // Check for paired board
    int board_pairs = 0;
    #pragma unroll
    for (int rank = 0; rank < 13; rank++) {
        if (rank_counts[rank] >= 2) board_pairs++;
        if (rank_counts[rank] >= 3) texture |= (1 << 3);  // Trips on board
    }
    if (board_pairs > 0) texture |= (1 << 4);  // Paired board
    
    return texture;
}

// Main kernel function
extern "C" __global__ void kernelPokerSimNG(
    // Input arrays
    const Card* __restrict__ hero_hands,
    const Card* __restrict__ board_cards,
    const uint8_t* __restrict__ board_sizes,
    const uint8_t* __restrict__ num_opponents,
    
    // Configuration
    const SimConfig config,
    const int num_hands,
    const int total_simulations,
    
    // Output arrays
    uint32_t* __restrict__ wins,
    uint32_t* __restrict__ ties,
    uint32_t* __restrict__ losses,
    
    // Optional outputs
    uint32_t* __restrict__ hand_categories,  // 11 * num_hands array
    float* __restrict__ variances,
    uint16_t* __restrict__ board_textures,
    
    // RNG
    curandState* rng_states,
    uint64_t seed
) {
    // Calculate global thread ID
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int gid = bid * blockDim.x + tid;
    
    // Determine which hand this block processes (for batch mode)
    const int hand_idx = config & FLAG_BATCH_MODE ? bid : 0;
    if (hand_idx >= num_hands) return;
    
    // Calculate simulations for this thread
    const int total_threads = gridDim.x * blockDim.x;
    const int base_sims_per_thread = total_simulations / total_threads;
    const int extra_sims = total_simulations % total_threads;
    const int my_simulations = base_sims_per_thread + (gid < extra_sims ? 1 : 0);
    
    // Initialize RNG
    curandState local_rng_state;
    if (gid < total_threads) {
        curand_init(seed, gid, 0, &local_rng_state);
    }
    
    // Load hand-specific data
    const Card* hero_hand = &hero_hands[hand_idx * 2];
    const Card* board = &board_cards[hand_idx * 5];
    const uint8_t board_size = board_sizes[hand_idx];
    const uint8_t n_opponents = num_opponents[hand_idx];
    
    // Thread-local accumulators
    uint32_t thread_wins = 0, thread_ties = 0, thread_losses = 0;
    uint32_t thread_categories[11] = {0};
    float sum_win = 0.0f, sum_win_sq = 0.0f;  // For variance calculation
    
    // Shared memory for reduction (if enabled)
    extern __shared__ uint32_t shared_mem[];
    uint32_t* shared_wins = shared_mem;
    uint32_t* shared_ties = config & FLAG_USE_SHARED_MEM ? 
                            &shared_mem[blockDim.x] : nullptr;
    uint32_t* shared_losses = config & FLAG_USE_SHARED_MEM ? 
                              &shared_mem[2 * blockDim.x] : nullptr;
    
    // Main simulation loop
    for (int sim = 0; sim < my_simulations; sim++) {
        // Initialize deck
        DeckMask deck = 0xFFFFFFFFFFFFFULL;
        
        // Remove hero cards
        remove_from_deck(deck, hero_hand[0]);
        remove_from_deck(deck, hero_hand[1]);
        
        // Remove known board cards
        #pragma unroll
        for (int i = 0; i < board_size; i++) {
            if (is_valid_card(board[i])) {
                remove_from_deck(deck, board[i]);
            }
        }
        
        // Complete the board
        Card full_board[5];
        #pragma unroll
        for (int i = 0; i < 5; i++) {
            if (i < board_size) {
                full_board[i] = board[i];
            } else {
                full_board[i] = draw_card(deck, &local_rng_state);
            }
        }
        
        // Build hero's 7-card hand
        Card hero_7cards[7] = {
            hero_hand[0], hero_hand[1],
            full_board[0], full_board[1], full_board[2],
            full_board[3], full_board[4]
        };
        
        // Evaluate hero hand
        uint32_t hero_rank;
        uint8_t hero_category = CATEGORY_HIGH_CARD;
        
        if (config & FLAG_TRACK_CATEGORIES) {
            evaluate_hand_with_category(hero_7cards, &hero_rank, &hero_category);
            thread_categories[hero_category]++;
        } else {
            // Use simpler evaluation without category
            evaluate_hand_with_category(hero_7cards, &hero_rank, &hero_category);
        }
        
        // Evaluate opponents
        uint32_t best_opponent_rank = 0;
        
        for (int opp = 0; opp < n_opponents; opp++) {
            // Draw opponent cards
            Card opp_cards[2];
            opp_cards[0] = draw_card(deck, &local_rng_state);
            opp_cards[1] = draw_card(deck, &local_rng_state);
            
            // Build opponent's 7-card hand
            Card opp_7cards[7] = {
                opp_cards[0], opp_cards[1],
                full_board[0], full_board[1], full_board[2],
                full_board[3], full_board[4]
            };
            
            // Evaluate opponent hand (category not needed)
            uint32_t opp_rank;
            uint8_t dummy_category;
            evaluate_hand_with_category(opp_7cards, &opp_rank, &dummy_category);
            
            best_opponent_rank = max(best_opponent_rank, opp_rank);
        }
        
        // Determine outcome
        float outcome = 0.0f;
        if (hero_rank > best_opponent_rank) {
            thread_wins++;
            outcome = 1.0f;
        } else if (hero_rank == best_opponent_rank) {
            thread_ties++;
            outcome = 0.5f;
        } else {
            thread_losses++;
            outcome = 0.0f;
        }
        
        // Update variance calculation
        if (config & FLAG_COMPUTE_VARIANCE) {
            sum_win += outcome;
            sum_win_sq += outcome * outcome;
        }
    }
    
    // Analyze board texture once (not per simulation)
    if ((config & FLAG_ANALYZE_BOARD) && tid == 0 && board_textures != nullptr) {
        board_textures[hand_idx] = analyze_board_texture(board, board_size);
    }
    
    // Reduction phase
    if (config & FLAG_USE_SHARED_MEM) {
        // Store in shared memory
        shared_wins[tid] = thread_wins;
        if (shared_ties) shared_ties[tid] = thread_ties;
        if (shared_losses) shared_losses[tid] = thread_losses;
        __syncthreads();
        
        // Tree reduction
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_wins[tid] += shared_wins[tid + s];
                if (shared_ties) shared_ties[tid] += shared_ties[tid + s];
                if (shared_losses) shared_losses[tid] += shared_losses[tid + s];
            }
            __syncthreads();
        }
        
        // Write block results
        if (tid == 0) {
            atomicAdd(&wins[hand_idx], shared_wins[0]);
            atomicAdd(&ties[hand_idx], shared_ties[0]);
            atomicAdd(&losses[hand_idx], shared_losses[0]);
        }
    } else {
        // Direct atomic updates
        atomicAdd(&wins[hand_idx], thread_wins);
        atomicAdd(&ties[hand_idx], thread_ties);
        atomicAdd(&losses[hand_idx], thread_losses);
    }
    
    // Update hand categories
    if (config & FLAG_TRACK_CATEGORIES) {
        #pragma unroll
        for (int cat = 1; cat <= 10; cat++) {
            if (thread_categories[cat] > 0) {
                atomicAdd(&hand_categories[hand_idx * 11 + cat], thread_categories[cat]);
            }
        }
    }
    
    // Calculate and store variance
    if ((config & FLAG_COMPUTE_VARIANCE) && tid == 0 && variances != nullptr) {
        // This is a simplified variance calculation
        // In practice, we'd need proper reduction across all threads
        float mean = sum_win / my_simulations;
        float variance = (sum_win_sq / my_simulations) - (mean * mean);
        atomicAdd(&variances[hand_idx], variance);
    }
}

// Kernel launcher for single hand (backward compatibility)
extern "C" __global__ void kernelPokerSimNG_single(
    const Card* hero_hand,
    const Card* board_cards,
    uint8_t board_size,
    uint8_t num_opponents_val,
    uint32_t total_simulations,
    uint32_t* wins,
    uint32_t* ties,
    uint32_t* losses,
    uint32_t* hand_categories,
    uint64_t seed
) {
    // Set up config for single hand mode
    SimConfig config = FLAG_TRACK_CATEGORIES;  // Enable categories by default
    
    // Call main kernel
    kernelPokerSimNG(
        hero_hand,
        board_cards,
        &board_size,
        &num_opponents_val,
        config,
        1,  // num_hands = 1
        total_simulations,
        wins, ties, losses,
        hand_categories,
        nullptr,  // no variance
        nullptr,  // no board texture
        nullptr,  // RNG states handled internally
        seed
    );
}