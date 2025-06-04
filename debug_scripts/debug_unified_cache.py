#!/usr/bin/env python3
"""Debug script to investigate unified cache behavior."""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight import MonteCarloSolver
from poker_knight.storage.unified_cache import get_unified_cache, create_cache_key, clear_unified_cache

# Clear any existing cache before starting
clear_unified_cache()

def test_unified_cache_consistency():
    """Test if unified cache returns identical results for repeated queries."""
    print("Testing unified cache consistency...\n")
    
    # Initialize solver with caching enabled
    solver = MonteCarloSolver(enable_caching=True)
    solver._initialize_cache_if_needed()
    
    # Test parameters
    hero_hand = ['A♠', 'K♠']
    num_opponents = 2
    board_cards = None  # Preflop
    simulation_mode = "default"
    
    print(f"Test scenario: {hero_hand} vs {num_opponents} opponents (preflop)")
    print(f"Simulation mode: {simulation_mode}")
    
    # Create cache key for inspection
    cache_key = create_cache_key(hero_hand, num_opponents, board_cards, simulation_mode)
    print(f"\nNormalized cache key: {cache_key}")
    
    # First analysis - should run simulation and cache result
    print("\n1. First analysis (should compute and cache):")
    start = time.time()
    result1 = solver.analyze_hand(hero_hand, num_opponents, board_cards, simulation_mode)
    elapsed1 = time.time() - start
    print(f"   Win probability: {result1.win_probability}")
    print(f"   Simulations run: {result1.simulations_run}")
    print(f"   Execution time: {elapsed1:.3f}s")
    
    # Check cache stats after first run
    cache_stats = solver.get_cache_stats()
    if cache_stats:
        print(f"\nCache stats after first run:")
        if 'unified_cache' in cache_stats:
            stats = cache_stats['unified_cache']
            print(f"   Total requests: {stats['total_requests']}")
            print(f"   Cache hits: {stats['cache_hits']}")
            print(f"   Cache misses: {stats['cache_misses']}")
            print(f"   Hit rate: {stats['hit_rate']:.2%}")
    
    # Direct cache lookup
    print("\n2. Direct cache lookup:")
    if solver._unified_cache:
        cached = solver._unified_cache.get(cache_key)
        if cached:
            print(f"   Found in cache: win_prob={cached.win_probability}")
            print(f"   Cached simulations: {cached.simulations_run}")
        else:
            print("   NOT found in unified cache!")
    
    # Second analysis - should retrieve from cache
    print("\n3. Second analysis (should use cache):")
    start = time.time()
    result2 = solver.analyze_hand(hero_hand, num_opponents, board_cards, simulation_mode)
    elapsed2 = time.time() - start
    print(f"   Win probability: {result2.win_probability}")
    print(f"   Simulations run: {result2.simulations_run}")
    print(f"   Execution time: {elapsed2:.3f}s")
    
    # Check cache stats after second run
    cache_stats = solver.get_cache_stats()
    if cache_stats:
        print(f"\nCache stats after second run:")
        if 'unified_cache' in cache_stats:
            stats = cache_stats['unified_cache']
            print(f"   Total requests: {stats['total_requests']}")
            print(f"   Cache hits: {stats['cache_hits']}")
            print(f"   Cache misses: {stats['cache_misses']}")
            print(f"   Hit rate: {stats['hit_rate']:.2%}")
    
    # Third analysis for good measure
    print("\n4. Third analysis (should also use cache):")
    start = time.time()
    result3 = solver.analyze_hand(hero_hand, num_opponents, board_cards, simulation_mode)
    elapsed3 = time.time() - start
    print(f"   Win probability: {result3.win_probability}")
    print(f"   Simulations run: {result3.simulations_run}")
    print(f"   Execution time: {elapsed3:.3f}s")
    
    # Compare results
    print("\n5. Result comparison:")
    print(f"   Result 1: win={result1.win_probability}, sims={result1.simulations_run}")
    print(f"   Result 2: win={result2.win_probability}, sims={result2.simulations_run}")
    print(f"   Result 3: win={result3.win_probability}, sims={result3.simulations_run}")
    
    identical_12 = (result1.win_probability == result2.win_probability and 
                    result1.simulations_run == result2.simulations_run)
    identical_23 = (result2.win_probability == result3.win_probability and 
                    result2.simulations_run == result3.simulations_run)
    
    print(f"\n   Results 1 & 2 identical: {identical_12}")
    print(f"   Results 2 & 3 identical: {identical_23}")
    
    if not (identical_12 and identical_23):
        print("\n⚠️  WARNING: Cache is not returning identical results!")
        print("   This suggests the cache lookup or storage is failing.")
    else:
        print("\n✓ SUCCESS: Cache is returning identical results!")
    
    # Check execution times
    print(f"\n6. Performance analysis:")
    print(f"   First run:  {elapsed1:.3f}s (compute + cache)")
    print(f"   Second run: {elapsed2:.3f}s (should be cache hit)")
    print(f"   Third run:  {elapsed3:.3f}s (should be cache hit)")
    
    if elapsed2 < elapsed1 * 0.1 and elapsed3 < elapsed1 * 0.1:
        print("   ✓ Cache hits are significantly faster")
    else:
        print("   ⚠️  Cache hits are not as fast as expected")
    
    # Test with different scenarios
    print("\n\n7. Testing with board cards (flop):")
    board = ['Q♠', 'J♠', '10♥']
    
    print(f"   Scenario: {hero_hand} vs {num_opponents} opponents")
    print(f"   Board: {board}")
    
    # First run with board
    result4 = solver.analyze_hand(hero_hand, num_opponents, board, simulation_mode)
    print(f"   First run: win={result4.win_probability}")
    
    # Second run with board
    result5 = solver.analyze_hand(hero_hand, num_opponents, board, simulation_mode)
    print(f"   Second run: win={result5.win_probability}")
    
    board_identical = result4.win_probability == result5.win_probability
    print(f"   Board results identical: {board_identical}")
    
    return identical_12 and identical_23 and board_identical


if __name__ == "__main__":
    test_unified_cache_consistency()