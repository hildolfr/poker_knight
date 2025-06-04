#!/usr/bin/env python3
"""Debug first analyze_hand call."""

from poker_knight.solver import MonteCarloSolver
from poker_knight.storage.unified_cache import clear_unified_cache

# Clear cache
clear_unified_cache()

# Create solver
solver = MonteCarloSolver(enable_caching=True)

# Clear all caches
if hasattr(solver, '_unified_cache') and solver._unified_cache:
    solver._unified_cache.clear()
if hasattr(solver, '_board_cache') and solver._board_cache:
    if hasattr(solver._board_cache, 'unified_cache'):
        solver._board_cache.unified_cache.clear()

print("All caches cleared")

# Check initial stats
stats1 = solver.get_cache_stats()
uc_stats1 = stats1['unified_cache']
print(f"Before first call: requests={uc_stats1['total_requests']}, hits={uc_stats1['cache_hits']}, misses={uc_stats1['cache_misses']}")

# Make first call - this should be a MISS
print("\nCalling analyze_hand...")
result = solver.analyze_hand(
    hero_hand=["A♠", "K♠"],
    num_opponents=2,
    simulation_mode="fast"
)
print(f"Result: win={result.win_probability}")

# Check stats after first call
stats2 = solver.get_cache_stats()
uc_stats2 = stats2['unified_cache']
print(f"\nAfter first call: requests={uc_stats2['total_requests']}, hits={uc_stats2['cache_hits']}, misses={uc_stats2['cache_misses']}")

# Also check board cache stats if available
if hasattr(solver, '_board_cache') and solver._board_cache:
    bc_stats = solver._board_cache._stats
    print(f"Board cache stats: requests={bc_stats['total_requests']}, hits={bc_stats['cache_hits']}, misses={bc_stats['cache_misses']}")

solver.close()