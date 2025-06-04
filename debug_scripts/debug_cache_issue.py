#!/usr/bin/env python3
"""
Debug script to investigate cache hit/miss issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver
from poker_knight.storage.cache import create_cache_key, CacheConfig

def debug_cache_issue():
    """Debug the cache key generation and storage/retrieval."""
    print("🔍 Debugging Cache Issue")
    print("=" * 50)
    
    # Create solver with caching enabled
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    try:
        # Test scenario
        hero_hand = ["AS", "KS"]
        num_opponents = 2
        simulation_mode = "fast"
        
        print(f"Testing scenario: {hero_hand} vs {num_opponents} opponents, mode: {simulation_mode}")
        
        # Create cache key directly
        cache_key_1 = create_cache_key(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=None,
            simulation_mode=simulation_mode,
            hero_position=None,
            stack_depth=None,
            config=CacheConfig()
        )
        print(f"Cache key 1: {cache_key_1}")
        
        # Clear cache first
        if hasattr(solver, '_hand_cache') and solver._hand_cache:
            solver._hand_cache.clear()
        
        # Get initial cache stats
        initial_stats = solver.get_cache_stats()
        if initial_stats:
            print(f"Initial cache stats: hits={initial_stats['hand_cache']['cache_hits']}, misses={initial_stats['hand_cache']['cache_misses']}")
        
        # First analysis
        print("\n🔸 First analysis...")
        result1 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"Result 1: win={result1.win_probability:.6f}, sims={result1.simulations_run}")
        
        # Check cache stats after first call
        stats_after_1 = solver.get_cache_stats()
        if stats_after_1:
            print(f"After 1st call: hits={stats_after_1['hand_cache']['cache_hits']}, misses={stats_after_1['hand_cache']['cache_misses']}")
        
        # Create cache key again
        cache_key_2 = create_cache_key(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=None,
            simulation_mode=simulation_mode,
            hero_position=None,
            stack_depth=None,
            config=CacheConfig()
        )
        print(f"Cache key 2: {cache_key_2}")
        print(f"Keys match: {cache_key_1 == cache_key_2}")
        
        # Try to get from cache directly
        if hasattr(solver, '_hand_cache') and solver._hand_cache:
            print(f"Direct cache lookup result: {solver._hand_cache.get_result(cache_key_2) is not None}")
        
        # Second analysis
        print("\n🔸 Second analysis...")
        result2 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"Result 2: win={result2.win_probability:.6f}, sims={result2.simulations_run}")
        
        # Check cache stats after second call
        stats_after_2 = solver.get_cache_stats()
        if stats_after_2:
            print(f"After 2nd call: hits={stats_after_2['hand_cache']['cache_hits']}, misses={stats_after_2['hand_cache']['cache_misses']}")
        
        # Compare results
        print(f"\n📊 Results comparison:")
        print(f"Win probabilities: {result1.win_probability:.6f} vs {result2.win_probability:.6f}")
        print(f"Identical: {result1.win_probability == result2.win_probability}")
        print(f"Simulations: {result1.simulations_run} vs {result2.simulations_run}")
        
        # Check if second call was really a cache hit
        if stats_after_2 and stats_after_1:
            cache_hits_diff = stats_after_2['hand_cache']['cache_hits'] - stats_after_1['hand_cache']['cache_hits']
            print(f"Cache hits increased by: {cache_hits_diff}")
            if cache_hits_diff == 0:
                print("❌ PROBLEM: Second call was NOT a cache hit!")
            else:
                print("✅ Second call was a cache hit")
        
    finally:
        solver.close()

if __name__ == "__main__":
    debug_cache_issue()