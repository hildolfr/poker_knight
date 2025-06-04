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
    
    __all__ = [
        "Card", "HandEvaluator", "Deck", "SimulationResult", 
        "MonteCarloSolver", "solve_poker_hand",
        # Caching components
        "HandCache", "BoardTextureCache", "PreflopRangeCache",
        "CacheConfig", "CacheStats", "create_cache_key"
    ]
except ImportError:
    CACHING_AVAILABLE = False
    
    __all__ = [
        "Card", "HandEvaluator", "Deck", "SimulationResult", 
        "MonteCarloSolver", "solve_poker_hand"
    ]

__version__ = "1.5.1"
__author__ = "hildolfr"
__license__ = "MIT" 