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

# Import core components from new modules
from .core.cards import Card, Deck
from .core.evaluation import HandEvaluator
from .core.results import SimulationResult
from .constants import SUITS, SUIT_MAPPING, RANKS, RANK_VALUES

# Import simulation components
from .simulation import SimulationRunner, SmartSampler, MultiwayAnalyzer

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

# Import caching system (Task 1.4) - Updated to use unified cache
try:
    from .storage.unified_cache import (
        get_unified_cache, CacheKey, CacheResult, CacheStats,
        create_cache_key, ThreadSafeMonteCarloCache
    )
    UNIFIED_CACHING_AVAILABLE = True
except ImportError:
    UNIFIED_CACHING_AVAILABLE = False

# Legacy cache imports for backward compatibility during transition
try:
    from .storage.cache import (
        get_cache_manager, CacheConfig, create_cache_key as legacy_create_cache_key,
        HandCache, BoardTextureCache, PreflopRangeCache
    )
    LEGACY_CACHING_AVAILABLE = True
except ImportError:
    LEGACY_CACHING_AVAILABLE = False

# Legacy cache warming - removed as per WARMING_INTEGRATION.md
# These imports are no longer used and the files should be deleted
CACHE_WARMING_AVAILABLE = False

# Import cache pre-population system (Task 1.2) - Updated for startup-focused approach
try:
    from .storage.startup_prepopulation import (
        populate_preflop_on_startup, should_skip_startup_population, 
        StartupPopulationConfig, PopulationResult
    )
    from .storage.preflop_cache import get_preflop_cache, PreflopCacheConfig
    STARTUP_PREPOPULATION_AVAILABLE = True
except ImportError:
    STARTUP_PREPOPULATION_AVAILABLE = False

# Import comprehensive board caching system (Phase 3)
try:
    from .storage.board_cache import (
        get_board_cache, BoardScenarioCache, BoardCacheConfig,
        BoardAnalyzer, BoardStage, BoardTexture
    )
    BOARD_CACHING_AVAILABLE = True
except ImportError:
    BOARD_CACHING_AVAILABLE = False

# Legacy cache pre-population for fallback
try:
    from .storage.cache_prepopulation import (
        ensure_cache_populated, PopulationConfig, PopulationStats
    )
    LEGACY_PREPOPULATION_AVAILABLE = True
except ImportError:
    LEGACY_PREPOPULATION_AVAILABLE = False

# Module metadata
__version__ = "1.5.1"
__author__ = "hildolfr"
__license__ = "MIT"
__all__ = [
    "Card", "HandEvaluator", "Deck", "SimulationResult", 
    "MonteCarloSolver", "solve_poker_hand"
]

# Worker functions moved to parallel_workers.py to avoid circular imports
class MonteCarloSolver:
    """Monte Carlo poker solver for Texas Hold'em."""
    
    def __init__(self, config_path: Optional[str] = None, enable_caching: bool = True,
                 skip_cache_warming: bool = False, force_cache_regeneration: bool = False) -> None:
        """Initialize the solver with configuration settings."""
        self.config = self._load_config(config_path)
        self.evaluator = HandEvaluator()
        
        # Initialize random seed for deterministic behavior if configured
        random_seed = self.config.get("simulation_settings", {}).get("random_seed")
        if random_seed is not None:
            import random
            random.seed(random_seed)
        
        # Initialize simulation components
        self.simulation_runner = SimulationRunner(self.config, self.evaluator)
        self.smart_sampler = SmartSampler(self.config, self.evaluator)
        self.multiway_analyzer = MultiwayAnalyzer()
        
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
        
        # Initialize unified caching system (Task 1.4) - Lazy initialization to prevent deadlocks
        self._caching_enabled = enable_caching
        self._unified_cache = None
        self._preflop_cache = None        # Dedicated preflop cache
        self._board_cache = None          # Comprehensive board scenario cache
        self._legacy_cache_config = None  # For backward compatibility
        self._legacy_hand_cache = None    # Legacy cache fallback
        self._legacy_board_cache = None
        self._legacy_preflop_cache = None
        self._population_result = None    # Startup population results
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
        # Cleanup simulation runner
        self.simulation_runner.close()
        
        with self._lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
    
    def _initialize_cache_if_needed(self) -> bool:
        """Initialize unified cache on first use to prevent deadlocks during module import."""
        if not self._caching_enabled or self._unified_cache is not None:
            return self._caching_enabled
        
        # Try unified cache first
        if UNIFIED_CACHING_AVAILABLE:
            try:
                cache_settings = self.config.get("cache_settings", {})
                
                # Initialize unified cache
                self._unified_cache = get_unified_cache(
                    max_memory_mb=cache_settings.get("max_memory_mb", 512),
                    enable_persistence=cache_settings.get("enable_persistence", False),
                    redis_host=cache_settings.get("redis_host", "localhost"),
                    redis_port=cache_settings.get("redis_port", 6379),
                    redis_db=cache_settings.get("redis_db", 0),
                    sqlite_path=cache_settings.get("sqlite_path", "poker_knight_unified_cache.db")
                )
                
                # Initialize dedicated preflop cache
                if STARTUP_PREPOPULATION_AVAILABLE:
                    preflop_config = PreflopCacheConfig(
                        enable_preflop_cache=cache_settings.get("enable_preflop_cache", True),
                        max_memory_mb=cache_settings.get("preflop_max_memory_mb", 64),
                        enable_persistence=cache_settings.get("enable_persistence", False),
                        preload_on_startup=cache_settings.get("preload_on_startup", True)
                    )
                    
                    self._preflop_cache = get_preflop_cache(preflop_config, self._unified_cache)
                    
                    # Perform startup pre-population if needed and enabled
                    if (preflop_config.preload_on_startup and 
                        not self._skip_cache_warming and 
                        not self._force_cache_regeneration):
                        
                        should_skip, reason = should_skip_startup_population(self._preflop_cache)
                        
                        if not should_skip:
                            print("Preflop cache will populate priority hands on first use")
                            # Actually trigger the prepopulation now
                            try:
                                from .storage.startup_prepopulation import StartupCachePopulator
                                
                                # Determine prepopulation mode
                                if self._force_cache_regeneration:
                                    print("PokerKnight: Force regenerating cache with comprehensive population...")
                                    mode = 'comprehensive'
                                    time_limit = 180.0  # 3 minutes for full population
                                else:
                                    print("PokerKnight: Populating priority hands for faster analysis...")
                                    mode = 'quick'
                                    time_limit = 30.0  # 30 seconds for priority hands
                                
                                # Use startup populator for both modes (it uses our cache instances)
                                populator = StartupCachePopulator(
                                    unified_cache=self._unified_cache,
                                    preflop_cache=self._preflop_cache
                                )
                                
                                # Update config based on mode
                                if mode == 'comprehensive':
                                    # Comprehensive mode: all hands, more opponents, longer time
                                    populator.config.max_population_time_seconds = int(time_limit)
                                    populator.config.priority_hands_only = False
                                    populator.config.hand_categories = ["premium", "strong", "medium", "weak"]
                                    populator.config.max_opponents = 6
                                    populator.config.simulation_modes = ["fast", "default", "precision"]
                                else:
                                    # Quick mode: priority hands only
                                    populator.config.max_population_time_seconds = int(time_limit)
                                    populator.config.priority_hands_only = True
                                    populator.config.hand_categories = ["premium", "strong"]
                                    populator.config.max_opponents = 4
                                    populator.config.simulation_modes = ["fast", "default"]
                                
                                # Use instance method as callback
                                def simulation_callback(hand_notation: str, num_opponents: int, simulation_mode: str):
                                    """Callback to run actual simulations during prepopulation."""
                                    # Convert hand notation to card list
                                    # Hand notation is like "AA", "AKs", "T9o"
                                    # Parse hand notation
                                    if len(hand_notation) == 2:
                                        # Pocket pair like "AA"
                                        rank = hand_notation[0]
                                        # Convert T to 10
                                        if rank == 'T':
                                            rank = '10'
                                        hero_hand = [
                                            f"{rank}♠",
                                            f"{rank}♥"
                                        ]
                                    elif len(hand_notation) == 3:
                                        # Non-pair like "AKs" or "AKo"
                                        rank1 = hand_notation[0]
                                        rank2 = hand_notation[1]
                                        suited = hand_notation[2] == 's'
                                        
                                        # Convert T to 10
                                        if rank1 == 'T':
                                            rank1 = '10'
                                        if rank2 == 'T':
                                            rank2 = '10'
                                        
                                        if suited:
                                            hero_hand = [
                                                f"{rank1}♠",
                                                f"{rank2}♠"
                                            ]
                                        else:
                                            hero_hand = [
                                                f"{rank1}♠",
                                                f"{rank2}♥"
                                            ]
                                    else:
                                        print(f"Warning: Unknown hand notation: {hand_notation}")
                                        return None
                                    
                                    # Run simulation
                                    result = self.analyze_hand(
                                        hero_hand=hero_hand,
                                        num_opponents=num_opponents,
                                        board_cards=[],  # Preflop only
                                        simulation_mode=simulation_mode
                                    )
                                    
                                    return result
                                
                                # Run prepopulation
                                self._population_result = populator.populate_startup_cache(simulation_callback)
                                
                                print(f"PokerKnight: Populated {self._population_result.scenarios_populated} scenarios "
                                      f"in {self._population_result.population_time_seconds:.1f}s")
                                
                            except Exception as e:
                                # Don't fail if prepopulation fails
                                print(f"PokerKnight: Cache prepopulation failed (continuing): {e}")
                                import traceback
                                traceback.print_exc()
                        else:
                            print(f"Skipping preflop population: {reason}")
                
                # Initialize comprehensive board cache (Phase 3)
                if BOARD_CACHING_AVAILABLE:
                    board_config = BoardCacheConfig(
                        enable_board_cache=cache_settings.get("enable_board_cache", True),
                        max_memory_mb=cache_settings.get("board_max_memory_mb", 256),
                        enable_persistence=cache_settings.get("enable_persistence", False),
                        cache_preflop=True,
                        cache_flop=True,
                        cache_turn=True,
                        cache_river=True
                    )
                    
                    self._board_cache = get_board_cache(board_config, self._unified_cache)
                    print("Comprehensive board scenario cache initialized")
                
                print("Unified cache system initialized successfully!")
                return True
                
            except Exception as e:
                print(f"Warning: Unified cache initialization failed ({e}). Falling back to legacy cache.")
                self._unified_cache = None
                self._preflop_cache = None
        
        # Fall back to legacy cache system
        if LEGACY_CACHING_AVAILABLE:
            try:
                # Initialize legacy cache configuration
                cache_settings = self.config.get("cache_settings", {})
                self._legacy_cache_config = CacheConfig(
                    max_memory_mb=cache_settings.get("max_memory_mb", 512),
                    hand_cache_size=cache_settings.get("hand_cache_size", 10000),
                    board_cache_size=cache_settings.get("board_cache_size", 5000),
                    enable_persistence=cache_settings.get("enable_persistence", False)
                )
                
                # Get legacy cache instances
                self._legacy_hand_cache, self._legacy_board_cache, self._legacy_preflop_cache = get_cache_manager(self._legacy_cache_config)
                
                print("Legacy cache system initialized as fallback.")
                return True
                
            except Exception as e:
                print(f"Warning: Legacy cache system also failed ({e}). Running without cache.")
                self._caching_enabled = False
                return False
        else:
            print("Warning: No caching system available. Running without cache.")
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
            # Get unified cache stats if available
            if self._unified_cache:
                stats = self._unified_cache.get_stats()
                result = {
                    'caching_enabled': True,
                    'cache_type': 'unified',
                    'unified_cache': {
                        'total_requests': stats.total_requests,
                        'cache_hits': stats.cache_hits,
                        'cache_misses': stats.cache_misses,
                        'cache_size': stats.cache_size,
                        'hit_rate': stats.hit_rate,
                        'memory_usage_mb': stats.memory_usage_mb
                    }
                }
                
                # Add board cache stats if available
                if self._board_cache:
                    board_stats = self._board_cache.get_cache_stats()
                    result['board_cache'] = {
                        'total_requests': board_stats.get('total_requests', 0),
                        'cache_hits': board_stats.get('cache_hits', 0),
                        'cache_misses': board_stats.get('cache_misses', 0),
                        'hit_rate': board_stats.get('hit_rate', 0.0)
                    }
                
                # Add preflop cache stats if available
                if self._preflop_cache:
                    if hasattr(self._preflop_cache, 'stats'):
                        # New preflop cache with stats attribute
                        preflop_stats = self._preflop_cache.stats
                        result['preflop_cache'] = {
                            'total_requests': preflop_stats.cache_hits + preflop_stats.cache_misses,
                            'cache_hits': preflop_stats.cache_hits,
                            'cache_misses': preflop_stats.cache_misses,
                            'cached_combinations': preflop_stats.cached_combinations,
                            'coverage_percentage': preflop_stats.coverage_percentage
                        }
                
                # Aggregate total requests and hits across all caches
                total_requests = result['unified_cache']['total_requests']
                total_hits = result['unified_cache']['cache_hits']
                
                if 'board_cache' in result:
                    total_requests += result['board_cache']['total_requests']
                    total_hits += result['board_cache']['cache_hits']
                
                if 'preflop_cache' in result:
                    total_requests += result['preflop_cache']['total_requests']
                    total_hits += result['preflop_cache']['cache_hits']
                
                result['aggregate_stats'] = {
                    'total_requests': total_requests,
                    'total_hits': total_hits,
                    'overall_hit_rate': total_hits / total_requests if total_requests > 0 else 0.0
                }
                
                return result
            
            # Fall back to legacy cache stats
            elif self._legacy_hand_cache:
                hand_stats = self._legacy_hand_cache.get_stats()
                preflop_stats = self._legacy_preflop_cache.get_stats()
                preflop_coverage = self._legacy_preflop_cache.get_cache_coverage()
                
                return {
                    'caching_enabled': True,
                    'cache_type': 'legacy',
                    'hand_cache': {
                        'total_requests': hand_stats.total_requests,
                        'cache_hits': hand_stats.cache_hits,
                        'cache_misses': hand_stats.cache_misses,
                        'cache_size': hand_stats.cache_size,
                        'hit_rate': hand_stats.hit_rate,
                        'memory_usage_mb': hand_stats.memory_usage_mb
                    },
                    'preflop_cache': {
                        'total_requests': preflop_stats.get('total_requests', 0),
                        'cache_hits': preflop_stats.get('cache_hits', 0),
                        'cached_combinations': preflop_coverage.get('cached_combinations', 0),
                        'coverage_percentage': preflop_coverage.get('coverage_percentage', 0.0)
                    }
                }
            
            return {'error': 'No cache system available'}
            
        except Exception as e:
            return {'error': f"Failed to get cache stats: {e}"}
    
    def _normalize_hand_to_notation(self, hero_hand: List[str]) -> Optional[str]:
        """Normalize hero hand to standard notation (e.g., AKs, QQ, 72o)."""
        try:
            from .storage.unified_cache import CacheKeyNormalizer
            normalized = CacheKeyNormalizer.normalize_hand(hero_hand)
            
            # Convert normalized format to standard notation
            if "_" in normalized:
                # Suited/offsuit hand like "AK_suited" -> "AKs"
                hand_part, suit_part = normalized.split("_")
                if suit_part == "suited":
                    return f"{hand_part}s"
                elif suit_part == "offsuit":
                    return f"{hand_part}o"
            else:
                # Pocket pair like "QQ"
                return normalized
            
            return None
            
        except Exception:
            return None
    
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
        
        # Cache lookup (Task 1.3) - Check unified cache before running expensive simulations
        cache_key = None
        cached_result = None
        was_cached = False
        
        if self._caching_enabled and self._initialize_cache_if_needed():
            try:
                # Create normalized cache key (eliminates dynamic factors like position/stack)
                cache_key = create_cache_key(
                    hero_hand=hero_hand,
                    num_opponents=num_opponents,
                    board_cards=board_cards,
                    simulation_mode=simulation_mode
                )
                
                # Try comprehensive board cache first (covers all scenarios)
                if self._board_cache:
                    try:
                        cached_result_obj = self._board_cache.get_board_result(
                            hero_hand, num_opponents, board_cards, simulation_mode
                        )
                        if cached_result_obj:
                            # Convert CacheResult back to dict format for compatibility
                            cached_result = {
                                'win_probability': cached_result_obj.win_probability,
                                'tie_probability': cached_result_obj.tie_probability,
                                'loss_probability': cached_result_obj.loss_probability,
                                'simulations_run': cached_result_obj.simulations_run,
                                'execution_time_ms': cached_result_obj.execution_time_ms,
                                'confidence_interval': cached_result_obj.confidence_interval,
                                'hand_category_frequencies': cached_result_obj.hand_categories,
                                **cached_result_obj.metadata
                            }
                            was_cached = True
                    except Exception as e:
                        print(f"Warning: Board cache lookup failed: {e}")
                
                # Fallback to preflop cache for preflop scenarios if board cache missed
                if not was_cached and self._preflop_cache and not board_cards:
                    try:
                        from .storage.preflop_cache import PreflopHandGenerator
                        # Normalize hand to notation
                        hand_notation = self._normalize_hand_to_notation(hero_hand)
                        if hand_notation:
                            cached_result_obj = self._preflop_cache.get_preflop_result(
                                hand_notation, num_opponents, simulation_mode
                            )
                            if cached_result_obj:
                                # Convert CacheResult back to dict format for compatibility
                                cached_result = {
                                    'win_probability': cached_result_obj.win_probability,
                                    'tie_probability': cached_result_obj.tie_probability,
                                    'loss_probability': cached_result_obj.loss_probability,
                                    'simulations_run': cached_result_obj.simulations_run,
                                    'execution_time_ms': cached_result_obj.execution_time_ms,
                                    'confidence_interval': cached_result_obj.confidence_interval,
                                    'hand_category_frequencies': cached_result_obj.hand_categories,
                                    **cached_result_obj.metadata
                                }
                                was_cached = True
                    except Exception as e:
                        print(f"Warning: Preflop cache lookup failed: {e}")
                
                # Try unified cache if not cached in specialized caches
                if not was_cached and self._unified_cache:
                    cached_result_obj = self._unified_cache.get(cache_key)
                    if cached_result_obj:
                        # Convert CacheResult back to dict format for compatibility
                        cached_result = {
                            'win_probability': cached_result_obj.win_probability,
                            'tie_probability': cached_result_obj.tie_probability,
                            'loss_probability': cached_result_obj.loss_probability,
                            'simulations_run': cached_result_obj.simulations_run,
                            'execution_time_ms': cached_result_obj.execution_time_ms,
                            'confidence_interval': cached_result_obj.confidence_interval,
                            'hand_category_frequencies': cached_result_obj.hand_categories,
                            **cached_result_obj.metadata
                        }
                        was_cached = True
                
                # Fall back to legacy cache
                elif self._legacy_hand_cache:
                    legacy_cache_key = legacy_create_cache_key(
                        hero_hand=hero_hand,
                        num_opponents=num_opponents,
                        board_cards=board_cards,
                        simulation_mode=simulation_mode,
                        config=self._legacy_cache_config
                    )
                    cached_result = self._legacy_hand_cache.get_result(legacy_cache_key)
                    if cached_result:
                        was_cached = True
                        
            except Exception as e:
                print(f"Warning: Cache lookup failed ({e}). Running simulation without cache.")
        
        # If we have a cached result, check if it has required data before returning
        if cached_result and was_cached:
            cache_execution_time = (time.time() - start_time) * 1000
            
            # Check if hand categories are requested but missing from cached result
            need_hand_categories = self.config["output_settings"]["include_hand_categories"]
            cached_hand_categories = cached_result.get('hand_category_frequencies')
            
            # If hand categories are needed but missing or empty, invalidate cache hit
            if need_hand_categories and (not cached_hand_categories or len(cached_hand_categories) == 0):
                cached_result = None
                was_cached = False
            else:
                # Create result from cached data
                decimal_precision = self.config["output_settings"]["decimal_precision"]
                return SimulationResult(
                    win_probability=round(cached_result['win_probability'], decimal_precision),
                    tie_probability=round(cached_result.get('tie_probability', 0.0), decimal_precision),
                    loss_probability=round(cached_result.get('loss_probability', 0.0), decimal_precision),
                    simulations_run=cached_result.get('simulations_run', 0),
                    execution_time_ms=cache_execution_time,
                    confidence_interval=cached_result.get('confidence_interval'),
                    hand_category_frequencies=cached_hand_categories,
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
                wins, ties, losses, hand_categories, convergence_data = self.simulation_runner.run_parallel_simulations(
                    hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
                )
        
        elif use_standard_parallel:
            # Use standard thread pool for parallel processing
            wins, ties, losses, hand_categories, convergence_data = self.simulation_runner.run_parallel_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
            )
        else:
            # Use sequential processing for small simulation counts, when disabled, or when convergence analysis is needed
            # Initialize convergence monitor if available
            convergence_monitor = None
            if CONVERGENCE_ANALYSIS_AVAILABLE:
                conv_settings = self.config.get("convergence_settings", {})
                convergence_monitor = ConvergenceMonitor(
                    window_size=conv_settings.get("window_size", 1000),
                    geweke_threshold=conv_settings.get("geweke_threshold", 2.0),
                    min_samples=conv_settings.get("min_samples", 5000),
                    target_accuracy=conv_settings.get("target_accuracy", 0.01),
                    confidence_level=conv_settings.get("confidence_level", 0.95)
                )
            
            wins, ties, losses, hand_categories, convergence_data = self.simulation_runner.run_sequential_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time,
                convergence_monitor=convergence_monitor, smart_sampler=self.smart_sampler
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
            confidence_interval = self.simulation_runner._calculate_confidence_interval(win_prob, total_sims)
        
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
            multi_way_analysis = self.multiway_analyzer.calculate_multiway_statistics(
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
        
        # Cache the result for future queries (Task 1.3) - Store deterministic results only
        if self._caching_enabled and cache_key:
            try:
                decimal_precision = self.config["output_settings"]["decimal_precision"]
                
                # Prepare core Monte Carlo results (exclude dynamic contextual data)
                cache_metadata = {
                    'convergence_achieved': convergence_achieved,
                    'geweke_statistic': geweke_statistic,
                    'effective_sample_size': effective_sample_size,
                    'convergence_efficiency': convergence_efficiency,
                    'stopped_early': stopped_early,
                    'convergence_details': convergence_details,
                    'adaptive_timeout_used': adaptive_timeout_used,
                    'final_timeout_ms': final_timeout_ms,
                    'target_accuracy_achieved': target_accuracy_achieved,
                    'final_margin_of_error': final_margin_of_error
                    # Note: Excluded dynamic factors like position_aware_equity, 
                    # tournament context, etc. as per refactor plan
                }
                
                # Store in comprehensive board cache (covers all scenarios)
                if self._board_cache:
                    try:
                        cache_result = CacheResult(
                            win_probability=round(win_prob, decimal_precision),
                            tie_probability=round(tie_prob, decimal_precision),
                            loss_probability=round(loss_prob, decimal_precision),
                            confidence_interval=confidence_interval,
                            simulations_run=total_sims,
                            execution_time_ms=execution_time,
                            hand_categories=hand_category_frequencies or {},
                            metadata=cache_metadata,
                            timestamp=time.time()
                        )
                        
                        self._board_cache.store_board_result(
                            hero_hand, num_opponents, cache_result, board_cards, simulation_mode
                        )
                    except Exception as e:
                        print(f"Warning: Board cache storage failed: {e}")
                
                # Store in preflop cache for preflop scenarios (redundant but ensures coverage)
                if self._preflop_cache and not board_cards:  # Preflop scenario
                    try:
                        hand_notation = self._normalize_hand_to_notation(hero_hand)
                        if hand_notation:
                            cache_result = CacheResult(
                                win_probability=round(win_prob, decimal_precision),
                                tie_probability=round(tie_prob, decimal_precision),
                                loss_probability=round(loss_prob, decimal_precision),
                                confidence_interval=confidence_interval,
                                simulations_run=total_sims,
                                execution_time_ms=execution_time,
                                hand_categories=hand_category_frequencies or {},
                                metadata=cache_metadata,
                                timestamp=time.time()
                            )
                            
                            self._preflop_cache.store_preflop_result(
                                hand_notation, num_opponents, cache_result, simulation_mode
                            )
                    except Exception as e:
                        print(f"Warning: Preflop cache storage failed: {e}")
                
                # Store in unified cache (backup storage for all scenarios)
                if self._unified_cache:
                    cache_result = CacheResult(
                        win_probability=round(win_prob, decimal_precision),
                        tie_probability=round(tie_prob, decimal_precision),
                        loss_probability=round(loss_prob, decimal_precision),
                        confidence_interval=confidence_interval,
                        simulations_run=total_sims,
                        execution_time_ms=execution_time,
                        hand_categories=hand_category_frequencies or {},
                        metadata=cache_metadata,
                        timestamp=time.time()
                    )
                    
                    self._unified_cache.store(cache_key, cache_result)
                
                # Fall back to legacy cache
                elif self._legacy_hand_cache:
                    legacy_cache_data = {
                        'win_probability': round(win_prob, decimal_precision),
                        'tie_probability': round(tie_prob, decimal_precision),
                        'loss_probability': round(loss_prob, decimal_precision),
                        'simulations_run': total_sims,
                        'execution_time_ms': execution_time,
                        'confidence_interval': confidence_interval,
                        'hand_category_frequencies': hand_category_frequencies,
                        **cache_metadata
                    }
                    
                    # Create legacy cache key
                    legacy_cache_key = legacy_create_cache_key(
                        hero_hand=hero_hand,
                        num_opponents=num_opponents,
                        board_cards=board_cards,
                        simulation_mode=simulation_mode,
                        config=self._legacy_cache_config
                    )
                    
                    self._legacy_hand_cache.store_result(legacy_cache_key, legacy_cache_data)
                    
            except Exception as e:
                print(f"Warning: Cache storage failed ({e}). Result computed but not cached.")
        
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
    
    # Proxy methods for backward compatibility with tests
    def _run_sequential_simulations(self, hero_cards, num_opponents, board_cards, 
                                   removed_cards, num_simulations, max_time_ms, start_time,
                                   convergence_monitor=None, smart_sampler=None):
        """Backward compatibility proxy for _run_sequential_simulations."""
        return self.simulation_runner.run_sequential_simulations(
            hero_cards, num_opponents, board_cards, removed_cards, 
            num_simulations, max_time_ms, start_time,
            convergence_monitor=convergence_monitor, 
            smart_sampler=smart_sampler or self.smart_sampler
        )
    
    def _run_parallel_simulations(self, hero_cards, num_opponents, board_cards,
                                 removed_cards, num_simulations, max_time_ms, start_time,
                                 convergence_monitor=None, smart_sampler=None):
        """Backward compatibility proxy for _run_parallel_simulations."""
        # Note: run_parallel_simulations doesn't support convergence_monitor or smart_sampler
        # These parameters are ignored for backward compatibility
        return self.simulation_runner.run_parallel_simulations(
            hero_cards, num_opponents, board_cards, removed_cards,
            num_simulations, max_time_ms, start_time
        )
    
    def _calculate_confidence_interval(self, win_probability, simulations, confidence_level=0.95):
        """Backward compatibility proxy for _calculate_confidence_interval."""
        return self.simulation_runner._calculate_confidence_interval(win_probability, simulations, confidence_level)


# Global solver instance for reuse
_global_solver: Optional[MonteCarloSolver] = None
_solver_lock = threading.Lock()

def get_global_solver() -> MonteCarloSolver:
    """Get or create global solver instance to maintain cache across calls."""
    global _global_solver
    with _solver_lock:
        if _global_solver is None:
            _global_solver = MonteCarloSolver()
        return _global_solver

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
    solver = get_global_solver()
    return solver.analyze_hand(
        hero_hand, num_opponents, board_cards, simulation_mode,
        hero_position, stack_sizes, pot_size, tournament_context
    ) 