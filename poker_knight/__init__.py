"""
♞ Poker Knight - High-Performance Monte Carlo Texas Hold'em Poker Solver

A high-performance, statistically validated Monte Carlo poker analysis library
optimized for AI applications and poker strategy research.

Key Features:
- Monte Carlo simulation engine for precise hand analysis
- Configurable simulation modes (fast/default/precision)
- Parallel processing with intelligent thread management
- Comprehensive statistical validation and confidence intervals
- Memory-optimized algorithms for high-throughput analysis
- Raw performance without caching overhead
"""

from .core import Card, HandEvaluator, Deck, SimulationResult
from .solver import MonteCarloSolver, solve_poker_hand

__all__ = [
    "Card", "HandEvaluator", "Deck", "SimulationResult", 
    "MonteCarloSolver", "solve_poker_hand"
]

__version__ = "1.7.0"
__author__ = "hildolfr"
__license__ = "MIT"