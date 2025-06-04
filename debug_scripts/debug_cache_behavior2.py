#!/usr/bin/env python3
"""Debug cache behavior in detail."""

from poker_knight import MonteCarloSolver

def test_board_cache_behavior():
    """Test if board cache is causing issues."""
    print("Creating solver with caching enabled...")
    solver = MonteCarloSolver(enable_caching=True)
    
    # Force cache initialization
    solver._initialize_cache_if_needed()
    
    # Disable board cache to test
    print("\nDisabling board cache temporarily...")
    board_cache_backup = solver._board_cache
    solver._board_cache = None
    
    print("\nFirst analysis (without board cache):")
    result1 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 1: win={result1.win_probability:.6f}")
    
    print("\nSecond analysis (should use unified cache):")
    result2 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 2: win={result2.win_probability:.6f}")
    
    print(f"\nResults identical? {result1.win_probability == result2.win_probability}")
    print(f"Exact match: {result1.win_probability} == {result2.win_probability}")
    
    # Re-enable board cache
    solver._board_cache = board_cache_backup
    
    # Try with board cache enabled
    print("\n--- With board cache enabled ---")
    # Clear caches first
    if solver._unified_cache:
        solver._unified_cache.clear()
    if solver._board_cache and hasattr(solver._board_cache, 'clear'):
        solver._board_cache.clear()
    
    print("\nThird analysis (with board cache):")
    result3 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 3: win={result3.win_probability:.6f}")
    
    print("\nFourth analysis (should hit board cache):")
    result4 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 4: win={result4.win_probability:.6f}")
    
    print(f"\nResults 3 and 4 identical? {result3.win_probability == result4.win_probability}")
    
    # Get cache stats
    stats = solver.get_cache_stats()
    if stats:
        print("\nFinal cache stats:")
        print(f"Cache type: {stats.get('cache_type', 'unknown')}")
        if 'unified_cache' in stats:
            print(f"Unified cache hits: {stats['unified_cache']['cache_hits']}")
            print(f"Unified cache misses: {stats['unified_cache']['cache_misses']}")
    
    solver.close()

if __name__ == '__main__':
    test_board_cache_behavior()