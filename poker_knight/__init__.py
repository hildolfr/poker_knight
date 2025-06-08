"""
♞ Poker Knight - Monte Carlo Texas Hold'em Poker Solver

A high-performance, statistically validated Monte Carlo poker analysis library
optimized for AI applications and poker strategy research.

Key Features:
- Monte Carlo simulation engine for precise hand analysis
- Configurable simulation modes (fast/default/precision)
- Parallel processing with intelligent thread management
- Comprehensive statistical validation and confidence intervals
- Memory-optimized algorithms for high-throughput analysis
- Intelligent caching system for near-instant repeated queries (v1.6)
"""

from .core import Card, HandEvaluator, Deck, SimulationResult
from .solver import MonteCarloSolver, solve_poker_hand

# Import caching functionality if available (Task 1.3)
try:
    from .storage import (
        HandCache, BoardTextureCache, PreflopRangeCache,
        CacheConfig, CacheStats, create_cache_key
    )
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# Import cache prepopulation functionality
try:
    from .storage.startup_prepopulation import StartupCachePopulator, StartupPopulationConfig
    PREPOPULATION_AVAILABLE = True
except ImportError:
    PREPOPULATION_AVAILABLE = False

# Build __all__ based on available components
__all__ = [
    "Card", "HandEvaluator", "Deck", "SimulationResult", 
    "MonteCarloSolver", "solve_poker_hand"
]

if CACHING_AVAILABLE:
    __all__.extend([
        # Caching components
        "HandCache", "BoardTextureCache", "PreflopRangeCache",
        "CacheConfig", "CacheStats", "create_cache_key"
    ])

if PREPOPULATION_AVAILABLE:
    __all__.extend([
        # Cache prepopulation components
        "StartupCachePopulator", "StartupPopulationConfig",
        "prepopulate_cache"  # Convenience function
    ])


def prepopulate_cache(comprehensive: bool = False, time_limit: float = 30.0) -> dict:
    """Convenience function to prepopulate cache for faster performance.
    
    This function prepopulates the cache with common poker scenarios to provide
    near-instant results for subsequent queries. It can run in two modes:
    
    Args:
        comprehensive: If True, populate all common scenarios (2-3 minutes).
                      If False, populate only priority hands (30 seconds).
        time_limit: Maximum time to spend on prepopulation in seconds.
                   Default is 30 seconds for quick mode.
    
    Returns:
        Dictionary with population statistics:
        - scenarios_populated: Number of scenarios added to cache
        - scenarios_skipped: Number of scenarios already cached
        - population_time_seconds: Time taken for population
        - success: Whether population completed successfully
    
    Example:
        >>> # Quick prepopulation (30 seconds)
        >>> stats = prepopulate_cache()
        >>> print(f"Populated {stats['scenarios_populated']} scenarios")
        
        >>> # Comprehensive prepopulation (2-3 minutes)
        >>> stats = prepopulate_cache(comprehensive=True)
        >>> print(f"Cache coverage: {stats['final_coverage']:.1f}%")
    """
    if not PREPOPULATION_AVAILABLE:
        return {
            'success': False,
            'error': 'Cache prepopulation not available',
            'scenarios_populated': 0
        }
    
    # Create solver instance
    solver = MonteCarloSolver(
        enable_caching=True,
        skip_cache_warming=False,
        force_cache_regeneration=comprehensive
    )
    
    # Initialize cache if needed
    if not solver._initialize_cache_if_needed():
        return {
            'success': False,
            'error': 'Failed to initialize cache',
            'scenarios_populated': 0
        }
    
    # Get cache instances
    unified_cache = solver._unified_cache
    preflop_cache = solver._preflop_cache
    
    # Use StartupCachePopulator for both modes (shares solver's cache instances)
    populator = StartupCachePopulator(
        unified_cache=unified_cache,
        preflop_cache=preflop_cache
    )
    
    # Configure based on mode
    if comprehensive:
        # Comprehensive mode: all hands, more opponents, longer time
        populator.config.max_population_time_seconds = int(time_limit)
        populator.config.priority_hands_only = False
        populator.config.hand_categories = ["premium", "strong", "medium", "weak"]
        populator.config.max_opponents = 6
        populator.config.simulation_modes = ["fast", "default", "precision"]
        populator.config.force_repopulation = True
    else:
        # Quick mode: priority hands only
        populator.config.max_population_time_seconds = int(time_limit)
        populator.config.priority_hands_only = True
        populator.config.hand_categories = ["premium", "strong"]
        populator.config.max_opponents = 4
        populator.config.simulation_modes = ["fast", "default"]
    
    # Define simulation callback for both modes
    def simulation_callback(hand_notation: str, num_opponents: int, simulation_mode: str):
        """Run actual simulations during prepopulation."""
        # Parse hand notation
        if len(hand_notation) == 2:
            # Pocket pair
            rank = hand_notation[0]
            # Convert T to 10
            if rank == 'T':
                rank = '10'
            hero_hand = [f"{rank}♠", f"{rank}♥"]
        elif len(hand_notation) == 3:
            # Non-pair
            rank1 = hand_notation[0]
            rank2 = hand_notation[1]
            suited = hand_notation[2] == 's'
            
            # Convert T to 10
            if rank1 == 'T':
                rank1 = '10'
            if rank2 == 'T':
                rank2 = '10'
            
            if suited:
                hero_hand = [f"{rank1}♠", f"{rank2}♠"]
            else:
                hero_hand = [f"{rank1}♠", f"{rank2}♥"]
        else:
            return None
        
        # Run simulation
        return solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=[],
            simulation_mode=simulation_mode
        )
    
    # Run prepopulation
    result = populator.populate_startup_cache(simulation_callback)
    
    return {
        'success': result.success,
        'scenarios_populated': result.scenarios_populated,
        'scenarios_skipped': result.scenarios_skipped,
        'scenarios_failed': result.scenarios_failed,
        'population_time_seconds': result.population_time_seconds,
        'initial_coverage': result.initial_coverage,
        'final_coverage': result.final_coverage,
        'performance_improvement': result.performance_improvement
    }

__version__ = "1.6.0"
__author__ = "hildolfr"
__license__ = "MIT" 