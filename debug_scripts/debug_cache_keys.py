#!/usr/bin/env python3
"""Debug why different caches return different results."""

from poker_knight import MonteCarloSolver
from poker_knight.storage.unified_cache import create_cache_key

def test_cache_key_differences():
    """Test if different cache keys are being generated."""
    print("Testing cache key generation...")
    
    hero_hand = ["AS", "KS"]
    num_opponents = 2
    board_cards = []
    simulation_mode = "fast"
    
    # Generate unified cache key
    unified_key = create_cache_key(hero_hand, num_opponents, board_cards, simulation_mode)
    print(f"\nUnified cache key: {unified_key}")
    
    # Test with solver
    print("\n--- Testing with solver ---")
    solver = MonteCarloSolver(enable_caching=True)
    
    # Force initialization
    solver._initialize_cache_if_needed()
    
    # Disable board and preflop caches to test unified cache alone
    print("Disabling board and preflop caches...")
    solver._board_cache = None
    solver._preflop_cache = None
    
    print("\nFirst analysis (unified cache only):")
    result1 = solver.analyze_hand(
        hero_hand=hero_hand,
        num_opponents=num_opponents,
        board_cards=board_cards,
        simulation_mode=simulation_mode
    )
    print(f"Result 1: win={result1.win_probability:.6f}")
    
    print("\nSecond analysis (should be identical):")
    result2 = solver.analyze_hand(
        hero_hand=hero_hand,
        num_opponents=num_opponents,
        board_cards=board_cards,
        simulation_mode=simulation_mode
    )
    print(f"Result 2: win={result2.win_probability:.6f}")
    
    print(f"\nResults identical? {result1.win_probability == result2.win_probability}")
    
    # Check cache stats
    stats = solver.get_cache_stats()
    if stats:
        print(f"\nCache type: {stats.get('cache_type')}")
        if 'unified_cache' in stats:
            print(f"Unified cache hits: {stats['unified_cache']['cache_hits']}")
            print(f"Unified cache misses: {stats['unified_cache']['cache_misses']}")
    
    # Now test direct cache access
    if solver._unified_cache:
        print("\n--- Direct cache test ---")
        cached = solver._unified_cache.get(unified_key)
        if cached:
            print(f"Direct cache lookup: win={cached.win_probability:.6f}")
            print(f"Matches result1? {cached.win_probability == result1.win_probability}")
        else:
            print("No cached result found with unified key")
    
    solver.close()

if __name__ == '__main__':
    test_cache_key_differences()