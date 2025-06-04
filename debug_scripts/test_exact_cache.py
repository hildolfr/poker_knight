#!/usr/bin/env python3
"""Test exact cache behavior with pre-stored results."""

from poker_knight import MonteCarloSolver
from poker_knight.storage.unified_cache import create_cache_key, CacheResult
import time

def test_with_prestored_result():
    """Test by pre-storing a known result."""
    print("Testing with pre-stored cache result...")
    
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    solver._initialize_cache_if_needed()
    
    # Disable other caches to isolate unified cache
    solver._board_cache = None
    solver._preflop_cache = None
    
    if not solver._unified_cache:
        print("ERROR: No unified cache!")
        return
    
    # Clear cache
    solver._unified_cache.clear()
    
    # Create known result
    known_result = CacheResult(
        win_probability=0.888888,
        tie_probability=0.055555,
        loss_probability=0.055557,
        confidence_interval=(0.88, 0.89),
        simulations_run=99999,
        execution_time_ms=123.45,
        hand_categories={
            'high_card': 0.3,
            'pair': 0.4,
            'two_pair': 0.2,
            'three_of_a_kind': 0.05,
            'straight': 0.02,
            'flush': 0.02,
            'full_house': 0.008,
            'four_of_a_kind': 0.001,
            'straight_flush': 0.001
        },
        metadata={
            'convergence_achieved': True,
            'stopped_early': False
        },
        timestamp=time.time(),
        ttl=3600
    )
    
    # Store it with the exact key the solver will use
    cache_key = create_cache_key(["AS", "KS"], 2, [], "fast")
    print(f"Storing result with key: {cache_key}")
    print(f"Known win probability: {known_result.win_probability}")
    
    success = solver._unified_cache.store(cache_key, known_result)
    print(f"Store success: {success}")
    
    # Verify it's stored
    retrieved = solver._unified_cache.get(cache_key)
    print(f"Can retrieve manually: {retrieved is not None}")
    if retrieved:
        print(f"Retrieved win prob: {retrieved.win_probability}")
    
    # Now analyze - should return our pre-stored result
    print("\n--- Running analyze_hand ---")
    result = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    
    print(f"\nAnalysis result: win={result.win_probability}")
    print(f"Expected: win={known_result.win_probability}")
    print(f"Match? {result.win_probability == known_result.win_probability}")
    
    # Check if it ran simulations
    print(f"\nSimulations run: {result.simulations_run}")
    print(f"Expected: {known_result.simulations_run}")
    
    # Check stats
    stats = solver.get_cache_stats()
    if stats and 'unified_cache' in stats:
        print(f"\nCache hits: {stats['unified_cache']['cache_hits']}")
        print(f"Cache misses: {stats['unified_cache']['cache_misses']}")
    
    solver.close()

if __name__ == '__main__':
    test_with_prestored_result()