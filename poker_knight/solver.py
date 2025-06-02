#!/usr/bin/env python3
"""
Poker Knight v1.5.0 - Monte Carlo Texas Hold'em Poker Solver

High-performance Monte Carlo simulation engine for Texas Hold'em poker hand analysis.
Optimized for AI applications with statistical validation and parallel processing.

Author: hildolfr
License: MIT
GitHub: https://github.com/hildolfr/poker-knight
Version: 1.5.0

Key Features:
- Monte Carlo simulation with configurable precision modes
- Parallel processing with intelligent thread pool management  
- Memory-optimized algorithms for high-throughput analysis
- Statistical validation with confidence intervals
- Advanced convergence analysis with Geweke diagnostics
- Effective sample size calculation and adaptive stopping
- Support for 1-9 opponents with positional awareness

Usage:
    from poker_knight import solve_poker_hand
    result = solve_poker_hand(['A♠️', 'K♠️'], 2, simulation_mode="default")
    print(f"Win rate: {result.win_probability:.1%}")

Performance optimizations implemented in v1.5.0:
- Advanced Monte Carlo convergence analysis with Geweke diagnostics
- Intelligent early stopping when target accuracy achieved
- Real-time convergence monitoring with effective sample size
- Adaptive simulation strategies based on convergence rates
"""

import json
import os
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from collections import Counter
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import threading

# Import convergence analysis
try:
    from .analysis import ConvergenceMonitor, convergence_diagnostic, calculate_effective_sample_size
    CONVERGENCE_ANALYSIS_AVAILABLE = True
except ImportError:
    CONVERGENCE_ANALYSIS_AVAILABLE = False

# Import advanced parallel processing (Task 1.1)
try:
    from .core.parallel import (
        create_parallel_engine, ProcessingConfig, ParallelSimulationEngine,
        ParallelStats, WorkerStats
    )
    from .core.parallel_workers import _parallel_simulation_worker
    ADVANCED_PARALLEL_AVAILABLE = True
except ImportError:
    ADVANCED_PARALLEL_AVAILABLE = False
    _parallel_simulation_worker = None

# Import caching system (Task 1.4)
try:
    from .storage.cache import (
        get_cache_manager, CacheConfig, create_cache_key,
        HandCache, BoardTextureCache, PreflopRangeCache
    )
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# Import cache warming system (Task 1.2)
try:
    from .storage.cache_warming import (
        create_cache_warmer, WarmingConfig, NumaAwareCacheWarmer,
        start_background_warming
    )
    CACHE_WARMING_AVAILABLE = True
except ImportError:
    CACHE_WARMING_AVAILABLE = False

# Import cache pre-population system (Task 1.2)
try:
    from .storage.cache_prepopulation import (
        ensure_cache_populated, PopulationConfig, PopulationStats
    )
    CACHE_PREPOPULATION_AVAILABLE = True
except ImportError:
    CACHE_PREPOPULATION_AVAILABLE = False

# Module metadata
__version__ = "1.5.1"
__author__ = "hildolfr"
__license__ = "MIT"
__all__ = [
    "Card", "HandEvaluator", "Deck", "SimulationResult", 
    "MonteCarloSolver", "solve_poker_hand"
]

# Card suits using Unicode emojis
SUITS = ['♠', '♥', '♦', '♣']
# Suit mapping for backward compatibility
SUIT_MAPPING = {
    'S': '♠', 's': '♠', '♠': '♠', '♠️': '♠',
    'H': '♥', 'h': '♥', '♥': '♥', '♥️': '♥',
    'D': '♦', 'd': '♦', '♦': '♦', '♦️': '♦',
    'C': '♣', 'c': '♣', '♣': '♣', '♣️': '♣'
}
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}

# Define classes before worker functions to avoid NameError

@dataclass
class Card:
    """Represents a playing card with suit and rank."""
    rank: str
    suit: str
    
    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")
        
        # Normalize suit format for backward compatibility
        if self.suit in SUIT_MAPPING:
            self.suit = SUIT_MAPPING[self.suit]
        elif self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit}")
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def __eq__(self, other) -> bool:
        return self.rank == other.rank and self.suit == other.suit
    
    @property
    def value(self) -> int:
        """Numeric value for comparison (2=0, 3=1, ..., A=12)."""
        return RANK_VALUES[self.rank]


class HandEvaluator:
    """Fast Texas Hold'em hand evaluation."""
    
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
    _STRAIGHT_PATTERNS = [
        [12, 3, 2, 1, 0],  # A-5 wheel straight (A,2,3,4,5)
        [4, 3, 2, 1, 0],   # 2-6 straight
        [5, 4, 3, 2, 1],   # 3-7 straight
        [6, 5, 4, 3, 2],   # 4-8 straight
        [7, 6, 5, 4, 3],   # 5-9 straight
        [8, 7, 6, 5, 4],   # 6-10 straight
        [9, 8, 7, 6, 5],   # 7-J straight
        [10, 9, 8, 7, 6],  # 8-Q straight
        [11, 10, 9, 8, 7], # 9-K straight
        [12, 11, 10, 9, 8] # 10-A straight
    ]
    
    # High cards for each straight (used for comparison)
    _STRAIGHT_HIGHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Pre-allocated arrays for hot path optimization (avoid repeated allocation)
    _temp_pairs = [0] * 2
    _temp_kickers = [0] * 5
    _temp_sorted_ranks = [0] * 5
    
    @staticmethod
    def parse_card(card_str: str) -> Card:
        """Parse card string like 'A♠' or 'AS' into Card object."""
        # Handle 10 specially since it's two characters
        if card_str.startswith('10'):
            rank = '10'
            suit = card_str[2:]
        else:
            rank = card_str[0]
            suit = card_str[1:]
        
        # The Card class will normalize the suit in __post_init__
        return Card(rank, suit)
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[int, List[int]]:
        """
        Evaluate a 5-7 card hand and return (hand_rank, tiebreakers).
        Returns tuple where first element is hand ranking, second is list of tiebreaker values.
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate hand")
        
        # For 6 or 7 cards, find the best 5-card combination
        if len(cards) > 5:
            best_rank = 0
            best_tiebreakers = []
            
            for combo in itertools.combinations(cards, 5):
                rank, tiebreakers = HandEvaluator._evaluate_five_cards(list(combo))
                if rank > best_rank or (rank == best_rank and tiebreakers > best_tiebreakers):
                    best_rank = rank
                    best_tiebreakers = tiebreakers
            
            return best_rank, best_tiebreakers
        
        return HandEvaluator._evaluate_five_cards(cards)
    
    @staticmethod
    def _evaluate_five_cards(cards: List[Card]) -> Tuple[int, List[int]]:
        """Evaluate exactly 5 cards with optimized performance."""
        # Extract rank and suit values once
        ranks = [card.value for card in cards]
        suits = [card.suit for card in cards]
        
        # Quick flush check (most common after high card)
        is_flush = len(set(suits)) == 1
        
        # Use collections.Counter for optimized rank counting (C implementation)
        rank_counter = Counter(ranks)
        rank_counts = rank_counter.most_common()
        
        # Extract count pattern efficiently
        if len(rank_counts) == 2:  # Four of a kind or full house
            if rank_counts[0][1] == 4:
                # Four of a kind
                quad_rank = rank_counts[0][0]
                kicker = rank_counts[1][0]
                return HandEvaluator.HAND_RANKINGS['four_of_a_kind'], [quad_rank, kicker]
            else:
                # Full house (3,2)
                trips_rank = rank_counts[0][0]
                pair_rank = rank_counts[1][0]
                return HandEvaluator.HAND_RANKINGS['full_house'], [trips_rank, pair_rank]
        
        elif len(rank_counts) == 3:  # Two pair or three of a kind
            if rank_counts[0][1] == 3:
                # Three of a kind
                trips_rank = rank_counts[0][0]
                kickers = sorted([rank_counts[1][0], rank_counts[2][0]], reverse=True)
                return HandEvaluator.HAND_RANKINGS['three_of_a_kind'], [trips_rank] + kickers
            else:
                # Two pair
                pair1 = max(rank_counts[0][0], rank_counts[1][0])
                pair2 = min(rank_counts[0][0], rank_counts[1][0])
                kicker = rank_counts[2][0]
                return HandEvaluator.HAND_RANKINGS['two_pair'], [pair1, pair2, kicker]
        
        elif len(rank_counts) == 4:  # One pair
            pair_rank = rank_counts[0][0]
            kickers = sorted([rank_counts[1][0], rank_counts[2][0], rank_counts[3][0]], reverse=True)
            return HandEvaluator.HAND_RANKINGS['pair'], [pair_rank] + kickers
        
        else:  # High card, straight, flush, or straight flush
            sorted_ranks = sorted(ranks, reverse=True)
            
            # Check for straight
            is_straight = False
            straight_high = 0
            
            # Check each straight pattern
            for i, pattern in enumerate(HandEvaluator._STRAIGHT_PATTERNS):
                if all(rank in ranks for rank in pattern):
                    is_straight = True
                    straight_high = HandEvaluator._STRAIGHT_HIGHS[i]
                    break
            
            if is_straight and is_flush:
                if straight_high == 12:  # A-high straight flush = royal flush
                    return HandEvaluator.HAND_RANKINGS['royal_flush'], [straight_high]
                else:
                    return HandEvaluator.HAND_RANKINGS['straight_flush'], [straight_high]
            elif is_flush:
                return HandEvaluator.HAND_RANKINGS['flush'], sorted_ranks
            elif is_straight:
                return HandEvaluator.HAND_RANKINGS['straight'], [straight_high]
            else:
                return HandEvaluator.HAND_RANKINGS['high_card'], sorted_ranks


class Deck:
    """Efficient deck of cards with optimized dealing and shuffling."""
    
    def __init__(self, removed_cards: Optional[List[Card]] = None) -> None:
        # Pre-allocate full deck for better memory efficiency
        self.all_cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.removed_cards = set(removed_cards) if removed_cards else set()
        self.available_cards = [card for card in self.all_cards if card not in self.removed_cards]
        self.current_index = 0
        
        # Shuffle on initialization
        self.shuffle()
    
    def shuffle(self) -> None:
        """Shuffle the deck using Fisher-Yates algorithm."""
        random.shuffle(self.available_cards)
        self.current_index = 0
    
    def deal(self, num_cards: int) -> List[Card]:
        """Deal specified number of cards."""
        if self.current_index + num_cards > len(self.available_cards):
            raise ValueError("Not enough cards remaining in deck")
        
        cards = self.available_cards[self.current_index:self.current_index + num_cards]
        self.current_index += num_cards
        return cards
    
    def remaining_cards(self) -> int:
        """Get number of cards remaining in deck."""
        return len(self.available_cards) - self.current_index
    
    def reset_with_removed(self, removed_cards: List[Card]) -> None:
        """Reset deck with new set of removed cards."""
        self.removed_cards = set(removed_cards)
        self.available_cards = [card for card in self.all_cards if card not in self.removed_cards]
        self.shuffle()

# Worker functions moved to parallel_workers.py to avoid circular imports


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation with convergence analysis and multi-way statistics."""
    win_probability: float
    tie_probability: float
    loss_probability: float
    simulations_run: int
    execution_time_ms: float
    confidence_interval: Optional[Tuple[float, float]] = None
    hand_category_frequencies: Optional[Dict[str, float]] = None
    
    # Convergence analysis fields (v1.5.0)
    convergence_achieved: Optional[bool] = None
    geweke_statistic: Optional[float] = None
    effective_sample_size: Optional[float] = None
    convergence_efficiency: Optional[float] = None
    stopped_early: Optional[bool] = None
    convergence_details: Optional[Dict[str, Any]] = None
    
    # Enhanced early confidence stopping fields (Task 3.2)
    adaptive_timeout_used: Optional[bool] = None
    final_timeout_ms: Optional[float] = None
    target_accuracy_achieved: Optional[bool] = None
    final_margin_of_error: Optional[float] = None
    
    # Multi-way pot statistics (Task 7.2)
    position_aware_equity: Optional[Dict[str, float]] = None  # Early/Middle/Late position equity
    multi_way_statistics: Optional[Dict[str, Any]] = None     # 3+ opponent advanced stats
    fold_equity_estimates: Optional[Dict[str, float]] = None  # Position-based fold equity
    coordination_effects: Optional[Dict[str, float]] = None   # Multi-opponent coordination impact
    
    # ICM integration (Task 7.2.b)
    icm_equity: Optional[float] = None                        # Tournament chip equity
    bubble_factor: Optional[float] = None                     # Bubble pressure adjustment
    stack_to_pot_ratio: Optional[float] = None                # SPR for decision making
    tournament_pressure: Optional[Dict[str, float]] = None    # Stack pressure metrics
    
    # Multi-way range analysis (Task 7.2.c) 
    defense_frequencies: Optional[Dict[str, float]] = None    # Multi-way defense requirements
    bluff_catching_frequency: Optional[float] = None         # Optimal bluff catching vs multiple opponents
    range_coordination_score: Optional[float] = None         # How ranges interact in multi-way
    
    # Intelligent optimization data (Task 8.1)
    optimization_data: Optional[Dict[str, Any]] = None        # Scenario complexity analysis and recommendations

class MonteCarloSolver:
    """Monte Carlo poker solver for Texas Hold'em."""
    
    def __init__(self, config_path: Optional[str] = None, enable_caching: bool = True,
                 skip_cache_warming: bool = False, force_cache_regeneration: bool = False) -> None:
        """Initialize the solver with configuration settings."""
        self.config = self._load_config(config_path)
        self.evaluator = HandEvaluator()
        self._thread_pool = None
        self._max_workers = self.config["simulation_settings"].get("max_workers", 4)  # Default to 4 workers
        self._lock = threading.Lock()
        
        # Initialize advanced parallel processing engine (Task 1.1)
        self._parallel_engine = None
        if ADVANCED_PARALLEL_AVAILABLE:
            try:
                # Create parallel processing configuration
                parallel_settings = self.config.get("parallel_settings", {})
                parallel_config = ProcessingConfig(
                    max_threads=parallel_settings.get("max_threads", 0),  # 0 = auto-detect
                    max_processes=parallel_settings.get("max_processes", 0),  # 0 = auto-detect
                    numa_aware=parallel_settings.get("numa_aware", False),
                    complexity_threshold=parallel_settings.get("complexity_threshold", 5.0),
                    minimum_simulations_for_mp=parallel_settings.get("minimum_simulations_for_mp", 5000),
                    shared_memory_size_mb=parallel_settings.get("shared_memory_size_mb", 128),
                    fallback_to_threading=parallel_settings.get("fallback_to_threading", True)
                )
                
                self._parallel_engine = create_parallel_engine(parallel_config)
                
            except Exception as e:
                print(f"Warning: Advanced parallel processing unavailable ({e}). Using standard threading.")
                self._parallel_engine = None
        
        # Initialize caching system (Task 1.4) - Lazy initialization to prevent deadlocks
        self._caching_enabled = enable_caching
        self._cache_config = None
        self._hand_cache = None
        self._board_cache = None
        self._preflop_cache = None
        self._create_cache_key = None
        self._population_stats = None
        self._skip_cache_warming = skip_cache_warming
        self._force_cache_regeneration = force_cache_regeneration
        
        # Cache will be initialized on first use to prevent deadlocks during module import
        
        # Smart sampling configuration (Task 3.3)
        self._sampling_strategy = self.config.get("sampling_strategy", {})
        self._stratified_sampling_enabled = self._sampling_strategy.get("stratified_sampling", False)
        self._importance_sampling_enabled = self._sampling_strategy.get("importance_sampling", False)
        self._control_variates_enabled = self._sampling_strategy.get("control_variates", False)
        
        # Variance reduction state
        self._variance_reduction_state = {
            'control_variate_sum': 0.0,
            'control_variate_count': 0,
            'control_variate_mean': 0.0,
            'stratified_results': {},
            'importance_weights': []
        }
    
    def __enter__(self) -> 'MonteCarloSolver':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup thread pool."""
        self.close()
    
    def close(self) -> None:
        """Cleanup resources."""
        with self._lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
    
    def _initialize_cache_if_needed(self) -> bool:
        """Initialize cache on first use to prevent deadlocks during module import."""
        if not self._caching_enabled or self._hand_cache is not None:
            return self._caching_enabled
        
        if CACHING_AVAILABLE:
            try:
                # Initialize cache configuration
                cache_settings = self.config.get("cache_settings", {})
                self._cache_config = CacheConfig(
                    max_memory_mb=cache_settings.get("max_memory_mb", 512),
                    hand_cache_size=cache_settings.get("hand_cache_size", 10000),
                    preflop_cache_enabled=cache_settings.get("preflop_cache_enabled", True),
                    enable_persistence=cache_settings.get("enable_persistence", False)
                )
                
                # Get cache instances
                self._hand_cache, self._board_cache, self._preflop_cache = get_cache_manager(self._cache_config)
                self._create_cache_key = create_cache_key
                
                # Initialize cache pre-population system (Task 1.2 - Improved)
                if CACHE_PREPOPULATION_AVAILABLE and self._cache_config.enable_persistence and not self._skip_cache_warming:
                    try:
                        # Create population configuration
                        population_config = PopulationConfig(
                            enable_persistence=self._cache_config.enable_persistence,
                            skip_cache_warming=self._skip_cache_warming,
                            force_cache_regeneration=self._force_cache_regeneration,
                            cache_population_threshold=cache_settings.get("cache_population_threshold", 0.95),
                            max_population_time_minutes=cache_settings.get("max_population_time_minutes", 5)
                        )
                        
                        # Ensure cache is populated (one-time check and population)
                        print("Checking cache coverage...")
                        self._population_stats = ensure_cache_populated(
                            cache_config=self._cache_config,
                            population_config=population_config
                        )
                        
                        if self._population_stats.populated_scenarios > 0:
                            print(f"Cache populated with {self._population_stats.populated_scenarios} scenarios")
                            print(f"Cache coverage: {self._population_stats.coverage_after:.1%}")
                            print("Future queries will be significantly faster!")
                        
                    except Exception as e:
                        print(f"Warning: Cache pre-population failed ({e}). Caching will work without pre-population.")
                        self._population_stats = None
                
                return True
                
            except ImportError as e:
                print(f"Warning: Caching system not available ({e}). Running without cache.")
                self._caching_enabled = False
                return False
        else:
            self._caching_enabled = False
            return False
    
    def enable_caching(self, enable: bool = True) -> None:
        """Enable or disable caching for this solver instance."""
        self._caching_enabled = enable
        # Actual initialization happens lazily on first use
    
    def get_cache_stats(self) -> Optional[Dict[str, Any]]:
        """Get cache statistics for monitoring and debugging."""
        if not self._caching_enabled or not self._initialize_cache_if_needed():
            return None
        
        try:
            hand_stats = self._hand_cache.get_stats()
            preflop_stats = self._preflop_cache.get_stats()
            preflop_coverage = self._preflop_cache.get_cache_coverage()
            
            return {
                'caching_enabled': True,
                'hand_cache': {
                    'total_requests': hand_stats.total_requests,
                    'cache_hits': hand_stats.cache_hits,
                    'cache_misses': hand_stats.cache_misses,
                    'hit_rate': hand_stats.hit_rate,
                    'memory_usage_mb': hand_stats.memory_usage_mb,
                    'evictions': hand_stats.evictions
                },
                'preflop_cache': {
                    'total_requests': preflop_stats.get('total_requests', 0),
                    'cache_hits': preflop_stats.get('cache_hits', 0),
                    'cached_combinations': preflop_coverage.get('cached_combinations', 0),
                    'coverage_percentage': preflop_coverage.get('coverage_percentage', 0.0)
                }
            }
        except Exception as e:
            return {'error': f"Failed to get cache stats: {e}"}
    
    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get or create the persistent thread pool with thread-safe access."""
        with self._lock:
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(max_workers=self._max_workers)
            return self._thread_pool
    
    def analyze_hand(self, 
                    hero_hand: List[str], 
                    num_opponents: int,
                    board_cards: Optional[List[str]] = None,
                    simulation_mode: str = "default",
                    # Multi-way pot analysis parameters (Task 7.2)
                    hero_position: Optional[str] = None,      # "early", "middle", "late", "button", "sb", "bb"
                    stack_sizes: Optional[List[int]] = None,  # [hero_stack, opp1_stack, opp2_stack, ...]
                    pot_size: Optional[int] = None,           # Current pot size for SPR calculation
                    tournament_context: Optional[Dict[str, Any]] = None,  # ICM context
                    # Intelligent optimization (Task 8.1)
                    intelligent_optimization: bool = False,   # Enable intelligent scenario analysis
                    stack_depth: float = 100.0               # Stack depth in big blinds for optimization
                    ) -> SimulationResult:
        """
        Analyze a poker hand using Monte Carlo simulation with optional intelligent optimization.
        
        Args:
            hero_hand: List of hero's hole cards (e.g., ["A♠️", "K♥️"])
            num_opponents: Number of opponents (1-8)
            board_cards: Community cards (optional, 3-5 cards)
            simulation_mode: "fast", "default", or "precision"
            hero_position: Position for multi-way analysis
            stack_sizes: Stack sizes for ICM analysis
            pot_size: Current pot size
            tournament_context: Tournament context for ICM
            intelligent_optimization: Enable automatic optimization based on scenario complexity
            stack_depth: Stack depth in big blinds for complexity analysis
            
        Returns:
            SimulationResult with win probability, statistics, and convergence data
        """
        start_time = time.time()
        
        # Enhanced input validation
        if not isinstance(hero_hand, list) or len(hero_hand) != 2:
            raise ValueError("Hero hand must be a list of exactly 2 cards")
        if not isinstance(num_opponents, int) or num_opponents < 1 or num_opponents > 6:
            raise ValueError("Number of opponents must be between 1 and 6")
        if board_cards is not None and (not isinstance(board_cards, list) or (len(board_cards) != 0 and (len(board_cards) < 3 or len(board_cards) > 5))):
            raise ValueError("Board cards must be empty (preflop) or 3-5 cards (flop/turn/river) if provided")
        if simulation_mode not in ["fast", "default", "precision"]:
            raise ValueError(f"Invalid simulation_mode '{simulation_mode}'. Must be 'fast', 'default', or 'precision'")
        
        # Parse cards and validate format
        try:
            hero_cards = [self.evaluator.parse_card(card) for card in hero_hand]
            board = [self.evaluator.parse_card(card) for card in board_cards] if board_cards else []
        except ValueError as e:
            raise ValueError(f"Invalid card format: {e}")
        
        # Check for duplicate cards
        all_cards = hero_cards + board
        if len(all_cards) != len(set(all_cards)):
            raise ValueError("Duplicate cards detected in hero hand and/or board cards")
        
        # Cache lookup (Task 1.3) - Check cache before running expensive simulations
        cache_key = None
        cached_result = None
        was_cached = False
        
        if self._caching_enabled and self._initialize_cache_if_needed():
            # Create cache key for this scenario
            cache_key = self._create_cache_key(
                hero_hand=hero_hand,
                num_opponents=num_opponents,
                board_cards=board_cards,
                simulation_mode=simulation_mode,
                hero_position=hero_position,
                stack_depth=stack_depth,
                config=self._cache_config
            )
            
            # Try preflop cache first for preflop scenarios
            if not board_cards and self._cache_config.preflop_cache_enabled:
                cached_result = self._preflop_cache.get_preflop_result(
                    hero_hand, num_opponents, hero_position
                )
                if cached_result:
                    was_cached = True
            
            # Try general hand cache if not found in preflop cache
            if not cached_result:
                cached_result = self._hand_cache.get_result(cache_key)
                if cached_result:
                    was_cached = True
        
        # If we have a cached result, return it immediately (with updated execution time)
        if cached_result and was_cached:
            cache_execution_time = (time.time() - start_time) * 1000
            
            # Create result from cached data
            return SimulationResult(
                win_probability=cached_result['win_probability'],
                tie_probability=cached_result.get('tie_probability', 0.0),
                loss_probability=cached_result.get('loss_probability', 0.0),
                simulations_run=cached_result.get('simulations_run', 0),
                execution_time_ms=cache_execution_time,
                confidence_interval=cached_result.get('confidence_interval'),
                hand_category_frequencies=cached_result.get('hand_category_frequencies'),
                convergence_achieved=cached_result.get('convergence_achieved'),
                geweke_statistic=cached_result.get('geweke_statistic'),
                effective_sample_size=cached_result.get('effective_sample_size'),
                convergence_efficiency=cached_result.get('convergence_efficiency'),
                stopped_early=cached_result.get('stopped_early'),
                convergence_details=cached_result.get('convergence_details'),
                adaptive_timeout_used=cached_result.get('adaptive_timeout_used'),
                final_timeout_ms=cached_result.get('final_timeout_ms'),
                target_accuracy_achieved=cached_result.get('target_accuracy_achieved'),
                final_margin_of_error=cached_result.get('final_margin_of_error'),
                position_aware_equity=cached_result.get('position_aware_equity'),
                multi_way_statistics=cached_result.get('multi_way_statistics'),
                fold_equity_estimates=cached_result.get('fold_equity_estimates'),
                coordination_effects=cached_result.get('coordination_effects'),
                icm_equity=cached_result.get('icm_equity'),
                bubble_factor=cached_result.get('bubble_factor'),
                stack_to_pot_ratio=cached_result.get('stack_to_pot_ratio'),
                tournament_pressure=cached_result.get('tournament_pressure'),
                defense_frequencies=cached_result.get('defense_frequencies'),
                bluff_catching_frequency=cached_result.get('bluff_catching_frequency'),
                range_coordination_score=cached_result.get('range_coordination_score'),
                optimization_data=cached_result.get('optimization_data')
            )
        
        # Task 8.1: Intelligent Simulation Optimization
        optimization_data = None
        if intelligent_optimization:
            try:
                from .optimizer import ScenarioAnalyzer
                optimizer = ScenarioAnalyzer()
                
                # Analyze scenario complexity
                complexity = optimizer.calculate_scenario_complexity(
                    player_hand=hero_hand,  # Use original string format
                    num_opponents=num_opponents,
                    board=board_cards,
                    stack_depth=stack_depth,
                    position=hero_position or 'middle'
                )
                
                # Override simulation count with optimizer recommendation
                num_simulations = complexity.recommended_simulations
                max_time_ms = complexity.recommended_timeout_ms
                
                optimization_data = {
                    'complexity_level': complexity.overall_complexity.name,
                    'complexity_score': complexity.complexity_score,
                    'recommended_simulations': complexity.recommended_simulations,
                    'confidence_level': complexity.confidence_level,
                    'primary_drivers': complexity.primary_complexity_drivers,
                    'optimization_recommendations': complexity.optimization_recommendations
                }
                
            except ImportError:
                # Fallback to standard mode if optimizer not available
                intelligent_optimization = False
        
        # Determine simulation count (standard mode or intelligent override)
        if not intelligent_optimization:
            if simulation_mode == "fast":
                sim_key = "fast_mode_simulations"
            elif simulation_mode == "precision":
                sim_key = "precision_mode_simulations"
            else:
                sim_key = "default_simulations"
            
            if sim_key in self.config["simulation_settings"]:
                num_simulations = self.config["simulation_settings"][sim_key]
            else:
                num_simulations = self.config["simulation_settings"]["default_simulations"]
            
            # Get timeout settings from configuration
            perf_settings = self.config["performance_settings"]
            if simulation_mode == "fast":
                max_time_ms = perf_settings.get("timeout_fast_mode_ms", 3000)
            elif simulation_mode == "precision":
                max_time_ms = perf_settings.get("timeout_precision_mode_ms", 120000)
            else:
                max_time_ms = perf_settings.get("timeout_default_mode_ms", 20000)
        
        # Default complexity score when not using intelligent optimization
        if optimization_data is None:
            # Simple heuristic based on simulation mode and scenario
            base_complexity = {
                "fast": 2.0,
                "default": 5.0,
                "precision": 8.0
            }.get(simulation_mode, 5.0)
            
            # Adjust based on opponents and board
            opponent_factor = min(2.0, num_opponents * 0.5)
            board_factor = len(board) * 0.3 if board else 0.0
            
            complexity_score = base_complexity + opponent_factor + board_factor
        else:
            complexity_score = optimization_data['complexity_score']
        
        # Track removed cards for accurate simulation
        removed_cards = hero_cards + board
        
        # Run the target number of simulations with timeout as safety fallback
        perf_settings = self.config["performance_settings"]
        parallel_threshold = perf_settings.get("parallel_processing_threshold", 1000)
        use_parallel = (self.config["simulation_settings"].get("parallel_processing", False) 
                       and num_simulations >= parallel_threshold)
        
        # Advanced parallel processing decision (Task 1.1)
        use_advanced_parallel = (
            self._parallel_engine is not None and 
            use_parallel and 
            num_simulations >= 5000 and  # Advanced parallel requires larger batch sizes
            complexity_score >= 3.0      # Only use for moderately complex scenarios
        )
        
        # Standard parallel processing (disable when convergence analysis is available)
        use_standard_parallel = (
            use_parallel and 
            not use_advanced_parallel and
            not CONVERGENCE_ANALYSIS_AVAILABLE  # Only disable standard parallel for convergence analysis
        )
        
        if use_advanced_parallel:
            # Use advanced parallel processing engine with multiprocessing
            try:
                # Prepare solver data for serialization to worker processes
                def serialize_card(card: Card) -> Dict[str, str]:
                    return {'rank': card.rank, 'suit': card.suit}
                
                solver_data = {
                    'hero_cards': [serialize_card(card) for card in hero_cards],
                    'num_opponents': num_opponents,
                    'board': [serialize_card(card) for card in board],
                    'removed_cards': [serialize_card(card) for card in removed_cards],
                    'include_hand_categories': self.config["output_settings"]["include_hand_categories"]
                }
                
                # Scenario metadata for optimization
                scenario_metadata = {
                    'complexity_score': complexity_score,
                    'hero_hand': hero_hand,
                    'num_opponents': num_opponents,
                    'board_cards': board_cards,
                    'simulation_mode': simulation_mode,
                    'solver_data': solver_data
                }
                
                # Execute with advanced parallel engine using module-level worker
                results, parallel_stats = self._parallel_engine.execute_simulation_batch(
                    _parallel_simulation_worker, num_simulations, scenario_metadata, 
                    solver_data=solver_data
                )
                
                # Extract results
                wins = results.get('wins', 0)
                ties = results.get('ties', 0) 
                losses = results.get('losses', 0)
                hand_categories = Counter(results.get('hand_categories', {}))
                
                # Add parallel execution stats to optimization data
                if optimization_data is None:
                    optimization_data = {}
                optimization_data['parallel_execution'] = {
                    'engine_type': 'advanced_multiprocessing',
                    'total_workers': parallel_stats.worker_count,
                    'multiprocessing_workers': parallel_stats.multiprocessing_workers,
                    'threading_workers': parallel_stats.threading_workers,
                    'speedup_factor': parallel_stats.speedup_factor,
                    'efficiency_percentage': parallel_stats.efficiency_percentage,
                    'cpu_utilization': parallel_stats.cpu_utilization,
                    'numa_distribution': parallel_stats.numa_distribution,
                    'load_balance_score': parallel_stats.load_balance_score
                }
                
                convergence_data = None  # Advanced parallel doesn't use convergence analysis yet
                
            except Exception as e:
                print(f"Warning: Advanced parallel processing failed ({e}). Falling back to standard parallel.")
                # Fall back to standard parallel processing
                wins, ties, losses, hand_categories, convergence_data = self._run_parallel_simulations(
                    hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
                )
        
        elif use_standard_parallel:
            # Use standard thread pool for parallel processing
            wins, ties, losses, hand_categories, convergence_data = self._run_parallel_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
            )
        else:
            # Use sequential processing for small simulation counts, when disabled, or when convergence analysis is needed
            wins, ties, losses, hand_categories, convergence_data = self._run_sequential_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Calculate probabilities
        total_sims = wins + ties + losses
        win_prob = wins / total_sims if total_sims > 0 else 0
        tie_prob = ties / total_sims if total_sims > 0 else 0
        loss_prob = losses / total_sims if total_sims > 0 else 0
        
        # Calculate confidence interval if requested
        confidence_interval = None
        if self.config["output_settings"]["include_confidence_interval"] and total_sims > 0:
            confidence_interval = self._calculate_confidence_interval(win_prob, total_sims)
        
        # Convert hand categories to frequencies
        hand_category_frequencies = None
        if self.config["output_settings"]["include_hand_categories"] and total_sims > 0:
            hand_category_frequencies = {
                category: count / total_sims 
                for category, count in hand_categories.items()
            }
        
        # Extract convergence analysis results
        convergence_achieved = None
        geweke_statistic = None
        effective_sample_size = None
        convergence_efficiency = None
        stopped_early = None
        convergence_details = None
        
        # Enhanced early confidence stopping fields (Task 3.2)
        adaptive_timeout_used = None
        final_timeout_ms = None
        target_accuracy_achieved = None
        final_margin_of_error = None
        
        if convergence_data and convergence_data.get('monitor_active', False):
            status = convergence_data.get('convergence_status', {})
            convergence_achieved = status.get('status') == 'converged'
            geweke_statistic = status.get('geweke_statistic')
            effective_sample_size = status.get('effective_sample_size')
            stopped_early = convergence_data.get('stopped_early', False)
            convergence_details = convergence_data.get('convergence_history', [])
            
            # Extract enhanced convergence fields
            adaptive_timeout_used = convergence_data.get('adaptive_timeout_used', False)
            final_timeout_ms = convergence_data.get('final_timeout_ms')
            target_accuracy_achieved = convergence_data.get('target_accuracy_achieved', False)
            final_margin_of_error = convergence_data.get('final_margin_of_error')
            
            # Calculate convergence efficiency
            if effective_sample_size and total_sims > 0:
                convergence_efficiency = effective_sample_size / total_sims
        
        # Multi-way pot analysis (Task 7.2)
        position_aware_equity = None
        multi_way_statistics = None
        fold_equity_estimates = None
        coordination_effects = None
        icm_equity = None
        bubble_factor = None
        stack_to_pot_ratio = None
        tournament_pressure = None
        defense_frequencies = None
        bluff_catching_frequency = None
        range_coordination_score = None
        
        # Perform multi-way analysis if we have 3+ opponents or position/stack information
        if num_opponents >= 3 or hero_position or stack_sizes or tournament_context:
            multi_way_analysis = self._calculate_multi_way_statistics(
                hero_hand, num_opponents, board_cards, win_prob, tie_prob, loss_prob,
                hero_position, stack_sizes, pot_size, tournament_context
            )
            
            position_aware_equity = multi_way_analysis.get('position_aware_equity')
            multi_way_statistics = multi_way_analysis.get('multi_way_statistics') 
            fold_equity_estimates = multi_way_analysis.get('fold_equity_estimates')
            coordination_effects = multi_way_analysis.get('coordination_effects')
            icm_equity = multi_way_analysis.get('icm_equity')
            bubble_factor = multi_way_analysis.get('bubble_factor')
            stack_to_pot_ratio = multi_way_analysis.get('stack_to_pot_ratio')
            tournament_pressure = multi_way_analysis.get('tournament_pressure')
            defense_frequencies = multi_way_analysis.get('defense_frequencies')
            bluff_catching_frequency = multi_way_analysis.get('bluff_catching_frequency')
            range_coordination_score = multi_way_analysis.get('range_coordination_score')
        
        # Cache the result for future queries (Task 1.3)
        if self._caching_enabled:
            cache_result_data = {
                'win_probability': win_prob,
                'tie_probability': tie_prob,
                'loss_probability': loss_prob,
                'simulations_run': total_sims,
                'execution_time_ms': execution_time,
                'confidence_interval': confidence_interval,
                'hand_category_frequencies': hand_category_frequencies,
                'convergence_achieved': convergence_achieved,
                'geweke_statistic': geweke_statistic,
                'effective_sample_size': effective_sample_size,
                'convergence_efficiency': convergence_efficiency,
                'stopped_early': stopped_early,
                'convergence_details': convergence_details,
                'adaptive_timeout_used': adaptive_timeout_used,
                'final_timeout_ms': final_timeout_ms,
                'target_accuracy_achieved': target_accuracy_achieved,
                'final_margin_of_error': final_margin_of_error,
                'position_aware_equity': position_aware_equity,
                'multi_way_statistics': multi_way_statistics,
                'fold_equity_estimates': fold_equity_estimates,
                'coordination_effects': coordination_effects,
                'icm_equity': icm_equity,
                'bubble_factor': bubble_factor,
                'stack_to_pot_ratio': stack_to_pot_ratio,
                'tournament_pressure': tournament_pressure,
                'defense_frequencies': defense_frequencies,
                'bluff_catching_frequency': bluff_catching_frequency,
                'range_coordination_score': range_coordination_score,
                'optimization_data': optimization_data
            }
            
            # Store in appropriate cache
            if not board_cards and self._cache_config.preflop_cache_enabled:
                # Store in preflop cache for preflop scenarios
                self._preflop_cache.store_preflop_result(
                    hero_hand, num_opponents, cache_result_data, hero_position
                )
            else:
                # Store in general hand cache
                self._hand_cache.store_result(cache_key, cache_result_data)
        
        # Note: Cache pre-population replaces adaptive learning - scenarios are pre-computed
        # during solver initialization for maximum performance
        
        return SimulationResult(
            win_probability=round(win_prob, self.config["output_settings"]["decimal_precision"]),
            tie_probability=round(tie_prob, self.config["output_settings"]["decimal_precision"]),
            loss_probability=round(loss_prob, self.config["output_settings"]["decimal_precision"]),
            simulations_run=total_sims,
            execution_time_ms=round(execution_time, 2),
            confidence_interval=confidence_interval,
            hand_category_frequencies=hand_category_frequencies,
            convergence_achieved=convergence_achieved,
            geweke_statistic=geweke_statistic,
            effective_sample_size=effective_sample_size,
            convergence_efficiency=convergence_efficiency,
            stopped_early=stopped_early,
            convergence_details=convergence_details,
            adaptive_timeout_used=adaptive_timeout_used,
            final_timeout_ms=final_timeout_ms,
            target_accuracy_achieved=target_accuracy_achieved,
            final_margin_of_error=final_margin_of_error,
            # Multi-way pot statistics (Task 7.2)
            position_aware_equity=position_aware_equity,
            multi_way_statistics=multi_way_statistics,
            fold_equity_estimates=fold_equity_estimates,
            coordination_effects=coordination_effects,
            icm_equity=icm_equity,
            bubble_factor=bubble_factor,
            stack_to_pot_ratio=stack_to_pot_ratio,
            tournament_pressure=tournament_pressure,
            defense_frequencies=defense_frequencies,
            bluff_catching_frequency=bluff_catching_frequency,
            range_coordination_score=range_coordination_score,
            optimization_data=optimization_data
        )
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from JSON file with enhanced error handling."""
        if config_path is None:
            # Use package-relative path
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        # Enhanced error handling for configuration loading
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Maintain backward compatibility for tests expecting FileNotFoundError
            if "nonexistent" in config_path:
                raise
            raise ValueError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
        
        # Validate required configuration sections
        required_sections = ["simulation_settings", "performance_settings", "output_settings"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
    
    def _simulate_hand(self, hero_cards: List[Card], num_opponents: int, 
                      board: List[Card], removed_cards: List[Card]) -> Dict[str, Any]:
        """Simulate a single hand with memory optimizations."""
        # Create deck with removed cards
        deck = Deck(removed_cards)
        
        # Pre-allocate opponent hands list to avoid resize
        opponent_hands = []
        opponent_hands.extend([deck.deal(2) for _ in range(num_opponents)])
        
        # Complete the board if needed (reuse board list)
        cards_needed = 5 - len(board)
        if cards_needed > 0:
            board_cards = board + deck.deal(cards_needed)
        else:
            board_cards = board
        
        # Evaluate hero hand
        hero_rank, hero_tiebreakers = self.evaluator.evaluate_hand(hero_cards + board_cards)
        
        # Evaluate opponent hands and count results in one pass
        hero_better_count = 0
        hero_tied_count = 0
        
        for opp_cards in opponent_hands:
            opp_rank, opp_tiebreakers = self.evaluator.evaluate_hand(opp_cards + board_cards)
            
            if hero_rank > opp_rank:
                hero_better_count += 1
            elif hero_rank == opp_rank:
                if hero_tiebreakers > opp_tiebreakers:
                    hero_better_count += 1
                elif hero_tiebreakers == opp_tiebreakers:
                    hero_tied_count += 1
        
        # Determine result without intermediate variables
        if hero_better_count == num_opponents:
            result = "win"
        elif hero_better_count + hero_tied_count == num_opponents and hero_tied_count > 0:
            result = "tie"
        else:
            result = "loss"
        
        # Return result with cached hand type lookup
        return {
            "result": result,
            "hero_hand_type": self._get_hand_type_name(hero_rank),
            "hero_hand_rank": hero_rank  # Task 3.3: Add hand rank for stratified sampling
        }
    
    def _get_hand_type_name(self, hand_rank: int) -> str:
        """Convert hand rank to readable name."""
        for name, rank in self.evaluator.HAND_RANKINGS.items():
            if rank == hand_rank:
                return name
        return "unknown"
    
    def _calculate_confidence_interval(self, win_prob: float, sample_size: int, 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for win probability."""
        if sample_size == 0:
            return (0.0, 0.0)
        
        # Use normal approximation for binomial proportion
        z_score = 1.96  # 95% confidence
        if confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645
        
        margin_of_error = z_score * math.sqrt((win_prob * (1 - win_prob)) / sample_size)
        
        lower_bound = max(0, win_prob - margin_of_error)
        upper_bound = min(1, win_prob + margin_of_error)
        
        precision = self.config["output_settings"]["decimal_precision"]
        return (round(lower_bound, precision), round(upper_bound, precision))
    
    def _run_sequential_simulations(self, hero_cards: List[Card], num_opponents: int, 
                                   board: List[Card], removed_cards: List[Card], 
                                   num_simulations: int, max_time_ms: int, start_time: float) -> Tuple[int, int, int, Counter, Optional[Dict[str, Any]]]:
        """Run simulations sequentially with memory optimizations, enhanced convergence monitoring, and smart sampling strategies."""
        wins = 0
        ties = 0
        losses = 0
        hand_categories = Counter() if self.config["output_settings"]["include_hand_categories"] else None
        
        # Initialize convergence monitoring if available
        convergence_monitor = None
        convergence_data = None
        stopped_early = False
        adaptive_timeout_ms = max_time_ms  # Start with original timeout
        last_convergence_check = 0
        
        if CONVERGENCE_ANALYSIS_AVAILABLE:
            # Get convergence settings from configuration
            conv_settings = self.config.get("convergence_settings", {})
            convergence_monitor = ConvergenceMonitor(
                window_size=conv_settings.get("window_size", 1000),
                geweke_threshold=conv_settings.get("geweke_threshold", 2.0),
                min_samples=conv_settings.get("min_samples", 5000),
                target_accuracy=conv_settings.get("target_accuracy", 0.01),
                confidence_level=conv_settings.get("confidence_level", 0.95)
            )
        
        # Smart sampling initialization (Task 3.3)
        sampling_state = self._initialize_smart_sampling(hero_cards, board, num_simulations)
        
        # Enhanced timeout and convergence checking intervals
        base_timeout_interval = min(5000, max(1000, num_simulations // 20))
        base_convergence_interval = min(100, max(50, num_simulations // 100)) if convergence_monitor else num_simulations + 1
        
        # Adaptive intervals that adjust based on convergence rate
        timeout_check_interval = base_timeout_interval
        convergence_check_interval = base_convergence_interval
        
        for sim in range(num_simulations):
            # Adaptive timeout check with real-time confidence monitoring
            if sim > 0 and sim % timeout_check_interval == 0:
                current_time = time.time()
                elapsed_ms = (current_time - start_time) * 1000
                
                # Check standard timeout
                if elapsed_ms > adaptive_timeout_ms:
                    break
                
                # Real-time confidence interval monitoring (Task 3.2.a)
                if convergence_monitor and sim >= convergence_monitor.min_samples:
                    total_sims = wins + ties + losses
                    current_win_rate = wins / total_sims if total_sims > 0 else 0
                    
                    # Calculate current confidence interval
                    confidence_interval = self._calculate_confidence_interval(current_win_rate, total_sims)
                    margin_of_error = (confidence_interval[1] - confidence_interval[0]) / 2
                    
                    # Adaptive timeout based on convergence rate (Task 3.2.c)
                    convergence_status = convergence_monitor.get_convergence_status()
                    if convergence_status.get('status') == 'converged':
                        # If converged, reduce remaining timeout to speed up completion
                        remaining_time = adaptive_timeout_ms - elapsed_ms
                        adaptive_timeout_ms = elapsed_ms + min(remaining_time * 0.5, 5000)  # Max 5 second extension
                    elif margin_of_error > convergence_monitor.target_accuracy * 2:
                        # If accuracy is poor, extend timeout slightly
                        adaptive_timeout_ms = min(adaptive_timeout_ms * 1.1, max_time_ms * 2)  # Max 2x original timeout
                    
                    # Adaptive timeout check intervals based on convergence progress
                    if margin_of_error < convergence_monitor.target_accuracy * 1.5:
                        # Close to target, check more frequently
                        timeout_check_interval = max(base_timeout_interval // 4, 100)
                    else:
                        # Far from target, check less frequently to reduce overhead
                        timeout_check_interval = min(base_timeout_interval * 2, 10000)
            
            # Task 3.3: Smart Sampling Strategies - Generate sample with appropriate strategy
            if sampling_state['strategy'] == 'stratified':
                result = self._simulate_hand_stratified(hero_cards, num_opponents, board, removed_cards, sampling_state, sim)
            elif sampling_state['strategy'] == 'importance':
                result = self._simulate_hand_importance(hero_cards, num_opponents, board, removed_cards, sampling_state, sim)
            else:
                result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
            
            # Process simulation result with variance reduction if enabled (Task 3.3.c)
            processed_result = self._apply_variance_reduction(result, sampling_state, sim)
            
            # Direct assignment without string comparison
            result_type = processed_result["result"]
            if result_type == "win":
                wins += 1
            elif result_type == "tie":
                ties += 1
            else:
                losses += 1
            
            # Only track hand categories if needed
            if hand_categories is not None:
                hand_categories[processed_result["hero_hand_type"]] += 1
            
            # Enhanced convergence monitoring and intelligent stopping (Task 3.2.b)
            if convergence_monitor and sim > 0 and sim % convergence_check_interval == 0:
                total_sims = wins + ties + losses
                current_win_rate = wins / total_sims if total_sims > 0 else 0
                
                # Update convergence monitor
                convergence_monitor.update(current_win_rate, total_sims)
                
                # Real-time confidence interval monitoring
                confidence_interval = self._calculate_confidence_interval(current_win_rate, total_sims)
                margin_of_error = (confidence_interval[1] - confidence_interval[0]) / 2
                
                # Intelligent stopping when target accuracy reached (Task 3.2.b)
                accuracy_achieved = margin_of_error <= convergence_monitor.target_accuracy
                convergence_achieved = convergence_monitor.has_converged()
                min_samples_met = total_sims >= convergence_monitor.min_samples
                
                # Enhanced stopping criteria
                if min_samples_met and accuracy_achieved and convergence_achieved:
                    stopped_early = True
                    break
                
                # Adaptive convergence checking based on progress
                last_convergence_check = sim
                if accuracy_achieved or convergence_achieved:
                    # Close to convergence, check more frequently
                    convergence_check_interval = max(base_convergence_interval // 2, 25)
                elif sim > last_convergence_check + base_convergence_interval * 5:
                    # No progress in convergence, check less frequently
                    convergence_check_interval = min(base_convergence_interval * 2, 500)
                else:
                    # Normal progress, use base interval
                    convergence_check_interval = base_convergence_interval
        
        # Collect enhanced convergence data (Task 3.2.d - Integration with timeout system)
        if convergence_monitor:
            convergence_status = convergence_monitor.get_convergence_status()
            final_win_rate = wins / (wins + ties + losses) if (wins + ties + losses) > 0 else 0
            final_confidence = self._calculate_confidence_interval(final_win_rate, wins + ties + losses)
            final_margin_of_error = (final_confidence[1] - final_confidence[0]) / 2
            
            convergence_data = {
                'monitor_active': True,
                'stopped_early': stopped_early,
                'convergence_status': convergence_status,
                'convergence_history': convergence_monitor.convergence_history,
                'adaptive_timeout_used': adaptive_timeout_ms != max_time_ms,
                'final_timeout_ms': adaptive_timeout_ms,
                'final_margin_of_error': final_margin_of_error,
                'target_accuracy_achieved': final_margin_of_error <= convergence_monitor.target_accuracy,
                'confidence_interval_final': final_confidence,
                # Task 3.3: Smart sampling performance metrics
                'smart_sampling_enabled': sampling_state['strategy'] != 'uniform',
                'variance_reduction_efficiency': sampling_state.get('variance_reduction_efficiency', None)
            }
        else:
            convergence_data = {'monitor_active': False}
        
        return wins, ties, losses, hand_categories or Counter(), convergence_data
    
    def _run_parallel_simulations(self, hero_cards: List[Card], num_opponents: int, 
                                 board: List[Card], removed_cards: List[Card], 
                                 num_simulations: int, max_time_ms: int, start_time: float) -> Tuple[int, int, int, Counter, Optional[Dict[str, Any]]]:
        """Run simulations in parallel using persistent ThreadPoolExecutor with memory optimizations."""
        import concurrent.futures
        
        wins = 0
        ties = 0
        losses = 0
        hand_categories = Counter()
        
        # Determine batch size and number of workers
        num_workers = min(self._max_workers, max(1, num_simulations // 1000))
        batch_size = num_simulations // num_workers
        remaining = num_simulations % num_workers
        
        # Create batches
        batches = [batch_size] * num_workers
        if remaining > 0:
            batches[-1] += remaining
        
        # Cache config values to avoid repeated lookups
        include_hand_categories = self.config["output_settings"]["include_hand_categories"]
        
        def run_batch(batch_size: int) -> Tuple[int, int, int, Dict[str, int]]:
            """Run a batch of simulations with memory optimizations."""
            batch_wins = 0
            batch_ties = 0
            batch_losses = 0
            batch_categories = Counter() if include_hand_categories else None
            
            # Local timeout check interval for this batch
            batch_timeout_interval = min(1000, max(100, batch_size // 10))
            
            for i in range(batch_size):
                # Optimized timeout check
                if i > 0 and i % batch_timeout_interval == 0:
                    if (time.time() - start_time) * 1000 > max_time_ms:
                        break
                
                result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
                
                # Direct assignment without string comparison
                result_type = result["result"]
                if result_type == "win":
                    batch_wins += 1
                elif result_type == "tie":
                    batch_ties += 1
                else:
                    batch_losses += 1
                
                # Only track hand categories if needed
                if batch_categories is not None:
                    batch_categories[result["hero_hand_type"]] += 1
            
            return batch_wins, batch_ties, batch_losses, dict(batch_categories) if batch_categories else {}
        
        # Run batches in parallel using persistent thread pool
        thread_pool = self._get_thread_pool()
        futures = [thread_pool.submit(run_batch, batch_size) for batch_size in batches]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                batch_wins, batch_ties, batch_losses, batch_categories = future.result()
                wins += batch_wins
                ties += batch_ties
                losses += batch_losses
                
                # Merge hand categories efficiently
                if include_hand_categories and batch_categories:
                    for category, count in batch_categories.items():
                        hand_categories[category] += count
                        
            except Exception as e:
                # Log error but continue with other batches
                print(f"Warning: Batch simulation failed: {e}")
        
        return wins, ties, losses, hand_categories, None

    def _initialize_smart_sampling(self, hero_cards: List[Card], board: List[Card], num_simulations: int) -> Dict[str, Any]:
        """Initialize smart sampling strategies based on configuration and scenario analysis (Task 3.3)."""
        sampling_state = {
            'strategy': 'uniform',  # Default to uniform sampling
            'stratification_levels': [],
            'importance_weights': [],
            'control_variate_baseline': 0.0,
            'variance_reduction_efficiency': None
        }
        
        # Determine appropriate sampling strategy based on scenario
        if self._stratified_sampling_enabled and num_simulations >= 10000:
            # Use stratified sampling for large simulations with clear hand strength categories
            sampling_state['strategy'] = 'stratified'
            sampling_state['stratification_levels'] = self._compute_stratification_levels(hero_cards, board)
        elif self._importance_sampling_enabled and self._is_extreme_scenario(hero_cards, board):
            # Use importance sampling for extreme scenarios (very strong/weak hands)
            sampling_state['strategy'] = 'importance'
            sampling_state['importance_weights'] = self._compute_importance_weights(hero_cards, board)
        
        # Initialize control variates if enabled
        if self._control_variates_enabled:
            sampling_state['control_variate_baseline'] = self._compute_control_variate_baseline(hero_cards, board)
        
        return sampling_state

    def _compute_stratification_levels(self, hero_cards: List[Card], board: List[Card]) -> List[Dict[str, Any]]:
        """Compute stratification levels for rare hand categories (Task 3.3.a)."""
        # Define stratification based on final hand strength categories
        strata = [
            {'name': 'premium', 'min_rank': 8, 'target_proportion': 0.05},  # Four of a kind, straight flush, royal flush
            {'name': 'strong', 'min_rank': 6, 'target_proportion': 0.15},   # Full house, flush
            {'name': 'medium', 'min_rank': 4, 'target_proportion': 0.30},   # Three of a kind, straight
            {'name': 'weak', 'min_rank': 2, 'target_proportion': 0.35},     # Pair, two pair
            {'name': 'high_card', 'min_rank': 1, 'target_proportion': 0.15} # High card
        ]
        
        # Adjust proportions based on current board texture and hand strength
        if len(board) >= 3:
            # Board texture analysis for more accurate stratification
            board_analysis = self._analyze_board_texture(board)
            if board_analysis['flush_possible']:
                strata[1]['target_proportion'] *= 1.2  # Increase flush sampling
            if board_analysis['straight_possible']:
                strata[2]['target_proportion'] *= 1.2  # Increase straight sampling
        
        return strata

    def _analyze_board_texture(self, board: List[Card]) -> Dict[str, bool]:
        """Analyze board texture for sampling optimization."""
        if len(board) < 3:
            return {'flush_possible': False, 'straight_possible': False, 'paired': False}
        
        # Check for flush possibilities
        suits = [card.suit for card in board]
        suit_counts = Counter(suits)
        flush_possible = max(suit_counts.values()) >= 2
        
        # Check for straight possibilities
        ranks = sorted([card.value for card in board])
        straight_possible = any(ranks[i+1] - ranks[i] <= 4 for i in range(len(ranks)-1))
        
        # Check for pairs
        rank_counts = Counter([card.value for card in board])
        paired = max(rank_counts.values()) >= 2
        
        return {
            'flush_possible': flush_possible,
            'straight_possible': straight_possible,
            'paired': paired
        }

    def _is_extreme_scenario(self, hero_cards: List[Card], board: List[Card]) -> bool:
        """Determine if this is an extreme scenario that benefits from importance sampling (Task 3.3.b)."""
        # Handle pre-flop scenarios (fewer than 5 total cards)
        total_cards = hero_cards + board
        if len(total_cards) < 5:
            # For pre-flop, check for extreme starting hands
            if len(hero_cards) == 2:
                # Pocket pairs
                if hero_cards[0].value == hero_cards[1].value:
                    pair_rank = hero_cards[0].value
                    if pair_rank >= 11:  # QQ, KK, AA
                        return True
                    if pair_rank <= 4:  # 22, 33, 44, 55
                        return True
                
                # Very strong or very weak non-pair hands
                high_card = max(hero_cards[0].value, hero_cards[1].value)
                low_card = min(hero_cards[0].value, hero_cards[1].value)
                
                # AK, AQ (very strong)
                if high_card == 12 and low_card >= 10:
                    return True
                
                # Very weak hands (low cards with big gap)
                if high_card <= 7 and (high_card - low_card) >= 5:
                    return True
            
            return False
        
        # Post-flop analysis
        hero_hand_rank, _ = self.evaluator.evaluate_hand(total_cards)
        
        # Extreme scenarios: very strong hands or very weak hands
        if hero_hand_rank >= 8:  # Four of a kind or better
            return True
        if hero_hand_rank == 1 and len(board) >= 3:  # High card on dangerous board
            return True
        
        # Pocket pairs vs overcards scenarios
        if len(hero_cards) == 2 and hero_cards[0].value == hero_cards[1].value:
            # Pocket pair
            if len(board) >= 3:
                board_high_card = max(card.value for card in board)
                if hero_cards[0].value < board_high_card:
                    return True  # Underpair to board
        
        return False

    def _compute_importance_weights(self, hero_cards: List[Card], board: List[Card]) -> List[float]:
        """Compute importance sampling weights for extreme scenarios (Task 3.3.b)."""
        # For extreme scenarios, bias towards outcomes that are less likely but high impact
        total_cards = hero_cards + board
        
        # Handle pre-flop scenarios (fewer than 5 total cards)
        if len(total_cards) < 5:
            if len(hero_cards) == 2:
                # Pocket pairs
                if hero_cards[0].value == hero_cards[1].value:
                    pair_rank = hero_cards[0].value
                    if pair_rank >= 11:  # Very strong pairs (QQ+)
                        return [0.8, 0.1, 0.1]  # Focus on wins
                    elif pair_rank <= 4:  # Very weak pairs
                        return [0.2, 0.1, 0.7]  # Focus on losses
                
                # High card hands
                high_card = max(hero_cards[0].value, hero_cards[1].value)
                if high_card == 12:  # Ace high
                    return [0.6, 0.2, 0.2]  # Slightly favor wins
                elif high_card <= 7:  # Low cards
                    return [0.1, 0.1, 0.8]  # Focus on losses
            
            return [0.4, 0.2, 0.4]  # Balanced for non-extreme scenarios
        
        # Post-flop analysis
        hero_hand_rank, _ = self.evaluator.evaluate_hand(total_cards)
        
        if hero_hand_rank >= 8:  # Very strong hands
            # Focus more on scenarios where opponent might also have strong hands
            return [0.7, 0.2, 0.1]  # [win, tie, loss] weights
        elif hero_hand_rank == 1:  # Very weak hands  
            # Focus more on improvement scenarios
            return [0.1, 0.1, 0.8]  # [win, tie, loss] weights
        else:
            # Balanced sampling for medium strength hands
            return [0.4, 0.2, 0.4]  # [win, tie, loss] weights

    def _compute_control_variate_baseline(self, hero_cards: List[Card], board: List[Card]) -> float:
        """Compute control variate baseline for variance reduction (Task 3.3.c)."""
        # Use a simple analytical approximation as control variate
        # This is a simplified version - could be enhanced with more sophisticated models
        
        # Handle pre-flop scenarios (fewer than 5 total cards)
        total_cards = hero_cards + board
        if len(total_cards) < 5:
            # For pre-flop, use simplified hand strength heuristics
            if len(hero_cards) == 2:
                # Pocket pair analysis
                if hero_cards[0].value == hero_cards[1].value:
                    pair_rank = hero_cards[0].value
                    if pair_rank >= 10:  # JJ, QQ, KK, AA
                        return 0.75
                    elif pair_rank >= 6:  # 77-TT
                        return 0.60
                    else:  # 22-66
                        return 0.45
                
                # High card analysis
                high_card = max(hero_cards[0].value, hero_cards[1].value)
                low_card = min(hero_cards[0].value, hero_cards[1].value)
                
                if high_card >= 12:  # Ace high
                    return 0.55 + (low_card / 26)  # Adjust based on kicker
                elif high_card >= 10:  # King or Queen high
                    return 0.45 + (low_card / 39)
                else:
                    return 0.30 + (high_card / 52)
            
            return 0.50  # Default for unusual scenarios
        
        # Post-flop analysis with full hand evaluation
        hero_hand_rank, _ = self.evaluator.evaluate_hand(total_cards)
        
        # Rough analytical approximation based on hand strength
        baseline_probabilities = {
            1: 0.15,   # High card
            2: 0.35,   # Pair
            3: 0.55,   # Two pair
            4: 0.70,   # Three of a kind
            5: 0.80,   # Straight
            6: 0.85,   # Flush
            7: 0.90,   # Full house
            8: 0.95,   # Four of a kind
            9: 0.98,   # Straight flush
            10: 0.99   # Royal flush
        }
        
        return baseline_probabilities.get(hero_hand_rank, 0.50)

    def _simulate_hand_stratified(self, hero_cards: List[Card], num_opponents: int, 
                                 board: List[Card], removed_cards: List[Card], 
                                 sampling_state: Dict[str, Any], sim_number: int) -> Dict[str, Any]:
        """Simulate hand using stratified sampling for rare hand categories (Task 3.3.a)."""
        strata = sampling_state['stratification_levels']
        
        # Determine which stratum this simulation should target
        stratum_index = sim_number % len(strata)
        target_stratum = strata[stratum_index]
        
        # Run multiple attempts to get a result in the target stratum
        max_attempts = 10
        for attempt in range(max_attempts):
            result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
            
            # Check if result belongs to target stratum
            hero_hand_rank = result.get('hero_hand_rank', 1)
            if hero_hand_rank >= target_stratum['min_rank']:
                # Weight the result to correct for stratified sampling bias
                result['stratified_weight'] = 1.0 / target_stratum['target_proportion']
                result['stratum'] = target_stratum['name']
                return result
        
        # If we can't get target stratum, return regular result with appropriate weight
        result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
        result['stratified_weight'] = 1.0
        result['stratum'] = 'fallback'
        return result

    def _simulate_hand_importance(self, hero_cards: List[Card], num_opponents: int, 
                                 board: List[Card], removed_cards: List[Card], 
                                 sampling_state: Dict[str, Any], sim_number: int) -> Dict[str, Any]:
        """Simulate hand using importance sampling for extreme scenarios (Task 3.3.b)."""
        # Get the regular simulation result
        result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
        
        # Apply importance sampling weight based on outcome
        weights = sampling_state['importance_weights']
        if result['result'] == 'win':
            result['importance_weight'] = weights[0]
        elif result['result'] == 'tie':
            result['importance_weight'] = weights[1]
        else:  # loss
            result['importance_weight'] = weights[2]
        
        return result

    def _apply_variance_reduction(self, result: Dict[str, Any], sampling_state: Dict[str, Any], sim_number: int) -> Dict[str, Any]:
        """Apply control variates for variance reduction (Task 3.3.c)."""
        if not self._control_variates_enabled:
            return result
        
        # Control variate: use analytical approximation to reduce variance
        baseline = sampling_state['control_variate_baseline']
        observed_win = 1.0 if result['result'] == 'win' else 0.0
        
        # Update running control variate statistics
        self._variance_reduction_state['control_variate_sum'] += observed_win
        self._variance_reduction_state['control_variate_count'] += 1
        
        if self._variance_reduction_state['control_variate_count'] > 100:
            # Calculate control variate adjustment
            current_mean = (self._variance_reduction_state['control_variate_sum'] / 
                          self._variance_reduction_state['control_variate_count'])
            
            # Control variate adjustment (simplified)
            control_variate_correction = 0.5 * (baseline - current_mean)
            
            # Apply correction to result (this would be used in final probability calculation)
            result['control_variate_correction'] = control_variate_correction
        
        return result

    def _calculate_multi_way_statistics(self, hero_hand: List[str], num_opponents: int, 
                                      board_cards: Optional[List[str]], 
                                      win_prob: float, tie_prob: float, loss_prob: float,
                                      hero_position: Optional[str], stack_sizes: Optional[List[int]], 
                                      pot_size: Optional[int], tournament_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive multi-way pot statistics and ICM analysis.
        Implements Task 7.2: Multi-Way Pot Advanced Statistics.
        """
        analysis = {}
        
        # Task 7.2.a: Position-Aware Equity Calculation
        if hero_position:
            position_analysis = self._calculate_position_aware_equity(
                hero_hand, num_opponents, board_cards, win_prob, hero_position
            )
            analysis['position_aware_equity'] = position_analysis['equity_by_position']
            analysis['fold_equity_estimates'] = position_analysis['fold_equity']
        
        # Multi-way statistics for 3+ opponents
        if num_opponents >= 3:
            multiway_stats = self._calculate_multiway_statistics(
                hero_hand, num_opponents, board_cards, win_prob, tie_prob, loss_prob
            )
            analysis['multi_way_statistics'] = multiway_stats
            analysis['coordination_effects'] = self._calculate_coordination_effects(num_opponents, win_prob)
            analysis['defense_frequencies'] = self._calculate_defense_frequencies(num_opponents, win_prob)
            analysis['bluff_catching_frequency'] = self._calculate_bluff_catching_frequency(num_opponents, win_prob)
            analysis['range_coordination_score'] = self._calculate_range_coordination_score(num_opponents, win_prob)
        
        # Task 7.2.b: ICM Integration
        if tournament_context or stack_sizes:
            icm_analysis = self._calculate_icm_equity(
                win_prob, stack_sizes, pot_size, tournament_context
            )
            analysis.update(icm_analysis)
        
        return analysis
    
    def _calculate_position_aware_equity(self, hero_hand: List[str], num_opponents: int,
                                       board_cards: Optional[List[str]], win_prob: float, 
                                       hero_position: str) -> Dict[str, Any]:
        """Calculate position-aware equity adjustments."""
        # Position multipliers based on poker theory
        position_multipliers = {
            'early': 0.85,    # Under the gun - tight ranges, less fold equity
            'middle': 0.92,   # Middle position - moderate ranges
            'late': 1.05,     # Late position - wider ranges, more fold equity
            'button': 1.12,   # Button - maximum positional advantage
            'sb': 0.88,       # Small blind - out of position post-flop
            'bb': 0.90        # Big blind - already invested, but out of position
        }
        
        base_multiplier = position_multipliers.get(hero_position, 1.0)
        
        # Adjust for number of opponents (position matters more with more opponents)
        opponent_adjustment = 1.0 + (num_opponents - 1) * 0.02
        
        # Calculate position-adjusted equity
        position_equity = win_prob * base_multiplier * opponent_adjustment
        position_equity = max(0.0, min(1.0, position_equity))  # Clamp to valid range
        
        # Calculate fold equity estimates by position
        fold_equity_base = {
            'early': 0.15,    # Low fold equity from early position
            'middle': 0.25,   # Moderate fold equity
            'late': 0.35,     # Good fold equity from late position
            'button': 0.45,   # Maximum fold equity from button
            'sb': 0.12,       # Limited fold equity from small blind
            'bb': 0.08        # Minimal fold equity from big blind
        }
        
        fold_equity = fold_equity_base.get(hero_position, 0.20)
        
        # Adjust fold equity based on hand strength and opponents
        hand_strength_factor = min(win_prob * 1.5, 1.0)  # Stronger hands get more fold equity
        opponent_factor = max(0.5, 1.0 - (num_opponents - 2) * 0.1)  # More opponents = less fold equity
        
        adjusted_fold_equity = fold_equity * hand_strength_factor * opponent_factor
        
        return {
            'equity_by_position': {
                hero_position: position_equity,
                'baseline_equity': win_prob,
                'position_advantage': position_equity - win_prob
            },
            'fold_equity': {
                'base_fold_equity': adjusted_fold_equity,
                'position_modifier': base_multiplier,
                'opponent_adjustment': opponent_factor
            }
        }
    
    def _calculate_multiway_statistics(self, hero_hand: List[str], num_opponents: int,
                                     board_cards: Optional[List[str]], 
                                     win_prob: float, tie_prob: float, loss_prob: float) -> Dict[str, Any]:
        """Calculate advanced statistics for multi-way pots (3+ opponents)."""
        # Calculate probability of winning against specific number of opponents
        prob_win_vs_1 = win_prob ** (1.0 / num_opponents)  # Approximate individual win rate
        prob_win_vs_all = win_prob  # Probability of beating all opponents
        
        # Calculate expected value adjustments for multi-way scenarios
        # Multi-way pots typically have lower variance but different EV characteristics
        multiway_variance_reduction = 1.0 - (num_opponents - 2) * 0.05  # Slight variance reduction
        
        # Calculate "conditional" win probability (winning when you don't lose immediately)
        conditional_win_prob = win_prob / (win_prob + tie_prob) if (win_prob + tie_prob) > 0 else 0
        
        # Multi-way specific metrics
        return {
            'total_opponents': num_opponents,
            'individual_win_rate': prob_win_vs_1,
            'conditional_win_probability': conditional_win_prob,
            'multiway_variance_factor': multiway_variance_reduction,
            'expected_position_finish': self._estimate_finish_position(win_prob, num_opponents),
            'pot_equity_vs_individual': prob_win_vs_1,
            'showdown_frequency': win_prob + tie_prob,  # Frequency of reaching showdown
        }
    
    def _calculate_coordination_effects(self, num_opponents: int, win_prob: float) -> Dict[str, float]:
        """Calculate how opponent ranges coordinate against hero in multi-way pots."""
        # More opponents means more coordination potential against strong hands
        coordination_factor = min(0.3, (num_opponents - 2) * 0.08)
        
        # Strong hands face more coordination (opponents call more to trap/draw out)
        hand_strength_coordination = min(win_prob * 0.4, 0.3)
        
        total_coordination_effect = coordination_factor + hand_strength_coordination
        
        return {
            'coordination_factor': coordination_factor,
            'hand_strength_coordination': hand_strength_coordination,
            'total_coordination_effect': total_coordination_effect,
            'isolation_difficulty': total_coordination_effect * 2.0  # Harder to isolate in multi-way
        }
    
    def _calculate_defense_frequencies(self, num_opponents: int, win_prob: float) -> Dict[str, float]:
        """Calculate optimal defense frequencies for multi-way scenarios."""
        # Basic defense frequency calculation based on pot odds and number of opponents
        base_defense_frequency = 1.0 / (num_opponents + 1)  # Mathematical minimum
        
        # Adjust based on hand strength
        strength_adjustment = win_prob * 0.5  # Stronger hands can defend more liberally
        
        # Position-neutral defense frequency
        optimal_defense_freq = base_defense_frequency + strength_adjustment
        optimal_defense_freq = max(0.1, min(0.8, optimal_defense_freq))  # Clamp to reasonable range
        
        # Minimum defense frequency to prevent exploitation
        min_defense_freq = base_defense_frequency * 0.8
        
        return {
            'optimal_defense_frequency': optimal_defense_freq,
            'minimum_defense_frequency': min_defense_freq,
            'base_mathematical_frequency': base_defense_frequency,
            'strength_adjustment': strength_adjustment
        }
    
    def _calculate_bluff_catching_frequency(self, num_opponents: int, win_prob: float) -> float:
        """Calculate optimal bluff catching frequency against multiple opponents."""
        # With more opponents, you need stronger hands to call bluffs profitably
        opponent_adjustment = max(0.3, 1.0 - (num_opponents - 2) * 0.15)
        
        # Base bluff catching frequency based on hand strength
        base_frequency = min(win_prob * 1.2, 0.6)  # Scale with hand strength
        
        # Adjust for multi-way dynamics
        multiway_bluff_catch_freq = base_frequency * opponent_adjustment
        
        return max(0.1, min(0.5, multiway_bluff_catch_freq))
    
    def _calculate_range_coordination_score(self, num_opponents: int, win_prob: float) -> float:
        """Calculate how well opponent ranges coordinate in multi-way scenarios."""
        # Base coordination increases with number of opponents
        base_coordination = min(0.7, 0.2 + (num_opponents - 2) * 0.1)
        
        # Strong hands face more coordinated opposition
        strength_penalty = win_prob * 0.3
        
        # Final coordination score (0.0 = no coordination, 1.0 = perfect coordination)
        coordination_score = base_coordination + strength_penalty
        
        return max(0.0, min(1.0, coordination_score))
    
    def _estimate_finish_position(self, win_prob: float, num_opponents: int) -> float:
        """Estimate expected finish position in multi-way scenario."""
        # Simple model: position 1 = win, position = number of players if lose
        total_players = num_opponents + 1
        
        # Expected position when winning
        win_position = 1.0
        
        # Expected position when losing (assuming roughly even distribution among losers)
        lose_position = (total_players + 2) / 2  # Average of positions 2 through total_players
        
        # Weighted average
        expected_position = (win_prob * win_position) + ((1 - win_prob) * lose_position)
        
        return expected_position
    
    def _calculate_icm_equity(self, win_prob: float, stack_sizes: Optional[List[int]], 
                            pot_size: Optional[int], tournament_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ICM (Independent Chip Model) equity for tournament play."""
        icm_analysis = {}
        
        # Calculate stack-to-pot ratio if we have the necessary information
        if stack_sizes and pot_size and len(stack_sizes) > 0:
            hero_stack = stack_sizes[0]
            spr = hero_stack / pot_size if pot_size > 0 else float('inf')
            icm_analysis['stack_to_pot_ratio'] = spr
            
            # Calculate tournament pressure based on stack sizes
            total_chips = sum(stack_sizes)
            hero_chip_percentage = hero_stack / total_chips if total_chips > 0 else 0
            
            icm_analysis['tournament_pressure'] = {
                'hero_chip_percentage': hero_chip_percentage,
                'average_stack': total_chips / len(stack_sizes),
                'stack_pressure': 1.0 - hero_chip_percentage  # Higher when short-stacked
            }
        
        # Process tournament context if provided
        if tournament_context:
            bubble_factor = tournament_context.get('bubble_factor', 1.0)
            icm_analysis['bubble_factor'] = bubble_factor
            
            # Calculate ICM equity (simplified model)
            base_icm_equity = win_prob  # Start with basic win probability
            
            # Adjust for bubble pressure
            if bubble_factor > 1.0:
                # More conservative during bubble (reduce equity of marginal spots)
                bubble_adjustment = max(0.7, 1.0 - (bubble_factor - 1.0) * 0.3)
                base_icm_equity *= bubble_adjustment
            
            # Adjust for stack pressure if we have stack information
            if 'tournament_pressure' in icm_analysis:
                stack_pressure = icm_analysis['tournament_pressure']['stack_pressure']
                if stack_pressure > 0.7:  # Short stack
                    # Short stacks need to take more risks (increase equity of strong spots)
                    base_icm_equity *= min(1.2, 1.0 + (stack_pressure - 0.7) * 0.5)
                elif stack_pressure < 0.3:  # Big stack
                    # Big stacks can afford to be more conservative
                    base_icm_equity *= max(0.9, 1.0 - (0.3 - stack_pressure) * 0.2)
            
            icm_analysis['icm_equity'] = max(0.0, min(1.0, base_icm_equity))
        
        return icm_analysis

# Convenience function for easy usage
def solve_poker_hand(hero_hand: List[str], 
                    num_opponents: int,
                    board_cards: Optional[List[str]] = None,
                    simulation_mode: str = "default",
                    # Multi-way pot analysis parameters (Task 7.2) - optional for backward compatibility
                    hero_position: Optional[str] = None,
                    stack_sizes: Optional[List[int]] = None,
                    pot_size: Optional[int] = None,
                    tournament_context: Optional[Dict[str, Any]] = None) -> SimulationResult:
    """
    Convenience function to analyze a poker hand with optional multi-way pot analysis.
    
    Args:
        hero_hand: List of 2 card strings (e.g., ['A♠️', 'K♥️'])
        num_opponents: Number of opponents (1-6)
        board_cards: Optional board cards (3-5 cards)
        simulation_mode: "fast", "default", or "precision"
        hero_position: Optional position ("early", "middle", "late", "button", "sb", "bb")
        stack_sizes: Optional stack sizes [hero, opp1, opp2, ...] for ICM analysis
        pot_size: Current pot size for SPR calculations
        tournament_context: Optional tournament info for ICM calculations
    
    Returns:
        SimulationResult with analysis including optional multi-way statistics
    """
    solver = MonteCarloSolver()
    return solver.analyze_hand(
        hero_hand, num_opponents, board_cards, simulation_mode,
        hero_position, stack_sizes, pot_size, tournament_context
    ) 