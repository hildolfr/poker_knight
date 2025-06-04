#!/usr/bin/env python3
"""Debug cache hit/miss behavior."""

from poker_knight.solver import MonteCarloSolver

# Create solver with caching
solver = MonteCarloSolver(enable_caching=True)

# Get initial stats
stats1 = solver.get_cache_stats()
print(f"Initial stats: {stats1}")

# First analysis (should be a miss)
result1 = solver.analyze_hand(
    hero_hand=["A♠", "K♠"],
    num_opponents=2,
    simulation_mode="fast"
)
print(f"Result 1: win={result1.win_probability:.3f}")

# Get stats after first call
stats2 = solver.get_cache_stats()
print(f"After first call: {stats2}")

# Second identical analysis (should be a hit)
result2 = solver.analyze_hand(
    hero_hand=["A♠", "K♠"],
    num_opponents=2,
    simulation_mode="fast"
)
print(f"Result 2: win={result2.win_probability:.3f}")

# Get stats after second call
stats3 = solver.get_cache_stats()
print(f"After second call: {stats3}")

# Check if results are identical (they should be if cache hit)
print(f"\nResults identical? {result1.win_probability == result2.win_probability}")
print(f"Simulations run: result1={result1.simulations_run}, result2={result2.simulations_run}")

solver.close()