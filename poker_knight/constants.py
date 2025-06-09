"""
Constants and configuration for Poker Knight.

This module contains all constants used throughout the poker solver,
including card representations, hand rankings, and suit mappings.
"""

# Card suits using Unicode symbols
SUITS = ['♠', '♥', '♦', '♣']

# Suit mapping for backward compatibility and flexibility
# Supports various input formats including emoji variants
SUIT_MAPPING = {
    # Spades
    'S': '♠', 's': '♠', '♠': '♠', '♠️': '♠',
    'spades': '♠', 'SPADES': '♠', 'Spades': '♠',
    
    # Hearts  
    'H': '♥', 'h': '♥', '♥': '♥', '♥️': '♥',
    'hearts': '♥', 'HEARTS': '♥', 'Hearts': '♥',
    
    # Diamonds
    'D': '♦', 'd': '♦', '♦': '♦', '♦️': '♦',
    'diamonds': '♦', 'DIAMONDS': '♦', 'Diamonds': '♦',
    
    # Clubs
    'C': '♣', 'c': '♣', '♣': '♣', '♣️': '♣',
    'clubs': '♣', 'CLUBS': '♣', 'Clubs': '♣'
}

# Card ranks in ascending order
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']

# Rank values for quick comparison (0-indexed)
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}

# Hand rankings (higher number = better hand)
HAND_RANKINGS = {
    'high_card': 1,
    'pair': 2,
    'two_pair': 3,
    'three_of_a_kind': 4,
    'straight': 5,
    'flush': 6,
    'full_house': 7,
    'four_of_a_kind': 8,
    'straight_flush': 9,
    'royal_flush': 10
}

# Precomputed straight patterns for optimization
# Each pattern represents rank indices that form a straight
STRAIGHT_PATTERNS = [
    [12, 3, 2, 1, 0],   # A-5 wheel straight (A,2,3,4,5)
    [4, 3, 2, 1, 0],    # 2-6 straight
    [5, 4, 3, 2, 1],    # 3-7 straight
    [6, 5, 4, 3, 2],    # 4-8 straight
    [7, 6, 5, 4, 3],    # 5-9 straight
    [8, 7, 6, 5, 4],    # 6-10 straight
    [9, 8, 7, 6, 5],    # 7-J straight
    [10, 9, 8, 7, 6],   # 8-Q straight
    [11, 10, 9, 8, 7],  # 9-K straight
    [12, 11, 10, 9, 8]  # 10-A straight
]

# High cards for each straight (used for comparison)
STRAIGHT_HIGHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Default number of simulations for different modes
DEFAULT_SIMULATIONS = {
    'fast': 10_000,
    'default': 100_000,
    'precision': 500_000
}

# Performance constants
DEFAULT_BATCH_SIZE = 1000
MIN_BATCH_SIZE = 100
MAX_BATCH_SIZE = 10000


# Convergence thresholds
DEFAULT_CONVERGENCE_THRESHOLD = 0.001
CONVERGENCE_CHECK_INTERVAL = 1000

# Thread pool settings
MAX_WORKERS = 16  # Maximum number of worker threads
CHUNK_SIZE = 100  # Size of work chunks for parallel processing