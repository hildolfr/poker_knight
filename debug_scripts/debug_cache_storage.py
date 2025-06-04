#!/usr/bin/env python3
"""
Debug script to investigate cache storage issues
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver
from poker_knight.storage.unified_cache import create_cache_key, CacheResult
import time

def debug_cache_storage():
    """Debug the cache storage and retrieval process."""
    print("🔍 Debugging Cache Storage")
    print("=" * 50)
    
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
        
        # Check which cache is active
        print(f"\nCache architecture:")
        print(f"Using unified cache: {solver._unified_cache is not None}")
        print(f"Using legacy cache: {solver._legacy_hand_cache is not None}")
        
        if solver._unified_cache:
            print("\n--- Testing with Unified Cache ---")
            cache = solver._unified_cache
            
            # Create cache key
            cache_key = create_cache_key(hero_hand, num_opponents, [], simulation_mode)
            print(f"Cache key: {cache_key}")
            
            # Clear cache
            cache.clear()
            
            # Create a test result
            test_result = CacheResult(
                win_probability=0.777777,
                tie_probability=0.111111,
                loss_probability=0.111112,
                confidence_interval=(0.77, 0.78),
                simulations_run=10000,
                execution_time_ms=100.0,
                hand_categories={'high_card': 0.5, 'pair': 0.5},
                metadata={'test': True},
                timestamp=time.time(),
                ttl=3600
            )
            
            # Store manually
            print("\nManual cache operations:")
            success = cache.store(cache_key, test_result)
            print(f"Store success: {success}")
            
            # Retrieve manually
            retrieved = cache.get(cache_key)
            if retrieved:
                print(f"Retrieved win prob: {retrieved.win_probability}")
                print(f"Matches stored: {retrieved.win_probability == test_result.win_probability}")
            else:
                print("ERROR: Failed to retrieve!")
            
            # Clear and re-test with analyze_hand
            cache.clear()
            
        # Test with actual analysis
        print(f"\n🔸 Actual analysis test:")
        
        # First analysis
        print("First analysis...")
        result1 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"Result 1: win={result1.win_probability:.6f}")
        
        # Second analysis
        print("\nSecond analysis...")
        result2 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"Result 2: win={result2.win_probability:.6f}")
        
        # Check cache stats
        stats = solver.get_cache_stats()
        if stats:
            print(f"\nCache stats:")
            cache_type = stats.get('cache_type', 'unknown')
            print(f"Cache type: {cache_type}")
            if cache_type == 'unified':
                print(f"Hits: {stats['unified_cache']['cache_hits']}")
                print(f"Misses: {stats['unified_cache']['cache_misses']}")
            else:
                print(f"Hits: {stats.get('hand_cache', {}).get('cache_hits', 0)}")
                print(f"Misses: {stats.get('hand_cache', {}).get('cache_misses', 0)}")
        
        print(f"\n📊 Final comparison:")
        print(f"Results identical: {result1.win_probability == result2.win_probability}")
        print(f"Difference: {abs(result1.win_probability - result2.win_probability):.6f}")
        
    finally:
        solver.close()

if __name__ == "__main__":
    debug_cache_storage()