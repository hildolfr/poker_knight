#!/usr/bin/env python3
"""
Debug script to investigate preflop vs hand cache fallback logic
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver
from poker_knight.storage.cache import create_cache_key, CacheConfig

def debug_fallback_logic():
    """Debug the preflop cache vs hand cache logic."""
    print("🔍 Debugging Preflop Cache Fallback Logic")
    print("=" * 60)
    
    # Create solver with caching enabled
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    try:
        # Force cache initialization
        solver._initialize_cache_if_needed()
        
        # Test scenario
        hero_hand = ["AS", "KS"]
        num_opponents = 2
        simulation_mode = "fast"
        
        print(f"Testing scenario: {hero_hand} vs {num_opponents} opponents, mode: {simulation_mode}")
        print(f"Board cards: None (preflop)")
        print(f"Preflop cache enabled: {solver._cache_config.preflop_cache_enabled}")
        
        # Create cache key
        cache_key = create_cache_key(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=None,
            simulation_mode=simulation_mode,
            hero_position=None,
            stack_depth=None,
            config=solver._cache_config or CacheConfig()
        )
        print(f"Hand cache key: {cache_key}")
        
        # Check what preflop cache key would be
        preflop_hand_key = solver._preflop_cache._normalize_preflop_hand(hero_hand)
        preflop_cache_key = f"{preflop_hand_key}_{num_opponents}"
        print(f"Preflop cache key: {preflop_cache_key}")
        
        # Clear both caches
        if solver._hand_cache:
            solver._hand_cache.clear()
        if solver._preflop_cache:
            solver._preflop_cache._preflop_cache.clear()
        
        print(f"\n🔸 First analysis...")
        result1 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"Result 1: win={result1.win_probability:.6f}, sims={result1.simulations_run}")
        
        # Check what got stored where
        print(f"\n📁 Cache storage check:")
        hand_cache_result = solver._hand_cache.get_result(cache_key) if solver._hand_cache else None
        preflop_cache_result = solver._preflop_cache.get_preflop_result(hero_hand, num_opponents) if solver._preflop_cache else None
        
        print(f"Stored in hand cache: {hand_cache_result is not None}")
        print(f"Stored in preflop cache: {preflop_cache_result is not None}")
        
        if hand_cache_result:
            print(f"Hand cache win prob: {hand_cache_result['win_probability']}")
        if preflop_cache_result:
            print(f"Preflop cache win prob: {preflop_cache_result['win_probability']}")
        
        # Check cache stats
        stats = solver.get_cache_stats()
        if stats:
            print(f"Cache stats: hits={stats['hand_cache']['cache_hits']}, misses={stats['hand_cache']['cache_misses']}")
        
        print(f"\n🔸 Second analysis...")
        result2 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"Result 2: win={result2.win_probability:.6f}, sims={result2.simulations_run}")
        
        # Check cache stats again
        stats2 = solver.get_cache_stats()
        if stats2:
            print(f"Cache stats after 2nd: hits={stats2['hand_cache']['cache_hits']}, misses={stats2['hand_cache']['cache_misses']}")
            
            hit_diff = stats2['hand_cache']['cache_hits'] - stats['hand_cache']['cache_hits']
            miss_diff = stats2['hand_cache']['cache_misses'] - stats['hand_cache']['cache_misses']
            print(f"Hits increased by: {hit_diff}")
            print(f"Misses increased by: {miss_diff}")
        
        print(f"\n📊 Final comparison:")
        print(f"Results identical: {result1.win_probability == result2.win_probability}")
        print(f"Simulations same: {result1.simulations_run == result2.simulations_run}")
        
        if result1.win_probability == result2.win_probability and result1.simulations_run == result2.simulations_run:
            print("✅ Likely cache hit (identical results)")
        else:
            print("❌ Likely cache miss (different results)")
        
    finally:
        solver.close()

if __name__ == "__main__":
    debug_fallback_logic()