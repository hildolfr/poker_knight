"""
Poker Knight - Monte Carlo Texas Hold'em Poker Solver

A high-performance Monte Carlo poker simulation engine that calculates win probabilities
for Texas Hold'em scenarios with accurate card removal effects and statistical confidence intervals.

Optimized for AI poker systems requiring fast, reliable hand strength analysis.
"""

from .solver import (
    Card, HandEvaluator, Deck, MonteCarloSolver, 
    solve_poker_hand, SimulationResult
)

__version__ = "1.3.0"
__author__ = "hildolfr"
__license__ = "MIT"
__all__ = [
    "Card", "HandEvaluator", "Deck", "SimulationResult", 
    "MonteCarloSolver", "solve_poker_hand"
] 