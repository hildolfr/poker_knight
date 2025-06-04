#!/usr/bin/env python3
"""Debug initial cache population issue."""

import json
from poker_knight.solver import MonteCarloSolver

# Create solver with caching
solver = MonteCarloSolver(enable_caching=True)

# Check if caches are empty
print("Checking initial cache state...")

# Check unified cache
if hasattr(solver, '_unified_cache') and solver._unified_cache:
    stats = solver._unified_cache.get_stats()
    print(f"Unified cache size: {stats.cache_size}")
    print(f"Unified cache requests: {stats.total_requests}")
    
# Check board cache
if hasattr(solver, '_board_cache') and solver._board_cache:
    print(f"Board cache exists")
    # Try to access its unified cache
    if hasattr(solver._board_cache, 'unified_cache'):
        board_unified_stats = solver._board_cache.unified_cache.get_stats()
        print(f"Board's unified cache size: {board_unified_stats.cache_size}")
        print(f"Board's unified cache requests: {board_unified_stats.total_requests}")
        
        # Check if they're the same instance
        print(f"Same cache instance? {solver._unified_cache is solver._board_cache.unified_cache}")

solver.close()