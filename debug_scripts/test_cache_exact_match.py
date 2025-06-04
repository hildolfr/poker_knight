#!/usr/bin/env python3
"""Test if cache returns exact values when decimal precision is high."""

from poker_knight import MonteCarloSolver

def test_high_precision_cache():
    """Test cache with high decimal precision."""
    print("Testing cache with high decimal precision...")
    
    # Create solver with high precision
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    solver._initialize_cache_if_needed()
    
    # Set high decimal precision
    solver.config["output_settings"]["decimal_precision"] = 10
    
    print(f"Decimal precision: {solver.config['output_settings']['decimal_precision']}")
    
    # Clear all caches
    if solver._unified_cache:
        solver._unified_cache.clear()
    if solver._board_cache and hasattr(solver._board_cache, 'clear_cache'):
        solver._board_cache.clear_cache()
    
    # First analysis
    print("\nFirst analysis...")
    result1 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 1: win={result1.win_probability:.10f}")
    
    # Second analysis
    print("\nSecond analysis...")
    result2 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 2: win={result2.win_probability:.10f}")
    
    print(f"\nExact match? {result1.win_probability == result2.win_probability}")
    print(f"Difference: {abs(result1.win_probability - result2.win_probability):.10f}")
    
    # Check cache stats
    stats = solver.get_cache_stats()
    if stats:
        cache_type = stats.get('cache_type', 'unknown')
        if cache_type == 'unified':
            print(f"\nCache hits: {stats['unified_cache']['cache_hits']}")
            print(f"Cache misses: {stats['unified_cache']['cache_misses']}")
    
    solver.close()

if __name__ == '__main__':
    test_high_precision_cache()