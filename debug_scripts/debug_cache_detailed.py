#!/usr/bin/env python3
"""Debug detailed cache behavior."""

from poker_knight.solver import MonteCarloSolver
from poker_knight.storage.unified_cache import clear_unified_cache

# Clear any existing cache
clear_unified_cache()

# Create solver with caching
solver = MonteCarloSolver(enable_caching=True)

# Clear caches explicitly
if hasattr(solver, '_unified_cache') and solver._unified_cache:
    solver._unified_cache.clear()
if hasattr(solver, '_legacy_hand_cache') and solver._legacy_hand_cache:
    solver._legacy_hand_cache.clear()

print("Cleared all caches")

# Get initial stats
stats1 = solver.get_cache_stats()
print(f"Initial stats: hits={stats1['unified_cache']['cache_hits']}, misses={stats1['unified_cache']['cache_misses']}")

# First analysis (should be a miss)
result1 = solver.analyze_hand(
    hero_hand=["A♠", "K♠"],
    num_opponents=2,
    simulation_mode="fast"
)

# Get stats after first call
stats2 = solver.get_cache_stats()
print(f"After first call: hits={stats2['unified_cache']['cache_hits']}, misses={stats2['unified_cache']['cache_misses']}")
print(f"  Cache size: {stats2['unified_cache']['cache_size']}")
print(f"  Total requests: {stats2['unified_cache']['total_requests']}")

solver.close()