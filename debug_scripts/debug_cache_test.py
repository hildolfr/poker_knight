#!/usr/bin/env python3
"""Debug cache behavior to understand why tests are failing."""

from poker_knight import MonteCarloSolver

def test_cache_behavior():
    """Test if cache is actually working."""
    print("Creating solver with caching enabled...")
    solver = MonteCarloSolver(enable_caching=True)
    
    # Force cache initialization
    solver._initialize_cache_if_needed()
    
    print("\nFirst analysis (should be cache miss):")
    result1 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 1: win={result1.win_probability:.4f}")
    
    # Get cache stats
    stats1 = solver.get_cache_stats()
    if stats1:
        cache_type = stats1.get('cache_type', 'legacy')
        print(f"Cache type: {cache_type}")
        if cache_type == 'unified':
            cache_stats = stats1['unified_cache']
        else:
            cache_stats = stats1.get('hand_cache', {})
        print(f"Cache hits: {cache_stats.get('cache_hits', 0)}")
        print(f"Cache misses: {cache_stats.get('cache_misses', 0)}")
    
    print("\nSecond analysis (should be cache hit):")
    result2 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 2: win={result2.win_probability:.4f}")
    
    # Get cache stats again
    stats2 = solver.get_cache_stats()
    if stats2:
        cache_type = stats2.get('cache_type', 'legacy')
        if cache_type == 'unified':
            cache_stats = stats2['unified_cache']
        else:
            cache_stats = stats2.get('hand_cache', {})
        print(f"Cache hits: {cache_stats.get('cache_hits', 0)}")
        print(f"Cache misses: {cache_stats.get('cache_misses', 0)}")
    
    print(f"\nResults identical? {result1.win_probability == result2.win_probability}")
    print(f"Difference: {abs(result1.win_probability - result2.win_probability):.6f}")
    
    # Check specific cache behavior
    if hasattr(solver, '_board_cache') and solver._board_cache:
        print("\nSolver has board cache")
    if hasattr(solver, '_unified_cache') and solver._unified_cache:
        print("Solver has unified cache")
    if hasattr(solver, '_preflop_cache') and solver._preflop_cache:
        print("Solver has preflop cache")
    
    # Let's try another test with explicit cache clear
    print("\n--- Testing with cache clear ---")
    if hasattr(solver, '_unified_cache') and solver._unified_cache:
        solver._unified_cache.clear()
    
    print("Third analysis (after cache clear):")
    result3 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 3: win={result3.win_probability:.4f}")
    
    print("Fourth analysis (should be cache hit again):")
    result4 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 4: win={result4.win_probability:.4f}")
    
    print(f"\nResults 3 and 4 identical? {result3.win_probability == result4.win_probability}")
    
    solver.close()

if __name__ == '__main__':
    test_cache_behavior()