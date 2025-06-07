#!/usr/bin/env python3
"""Debug cache statistics tracking issue."""

from poker_knight import MonteCarloSolver

def main():
    # Create solver with caching enabled
    solver = MonteCarloSolver(enable_caching=True)
    
    # Clear caches
    if hasattr(solver, '_unified_cache') and solver._unified_cache:
        solver._unified_cache.clear()
        print("Cleared unified cache")
    
    # Get initial stats
    stats1 = solver.get_cache_stats()
    print("Initial stats:", stats1)
    
    # First analysis
    print("\nRunning first analysis...")
    result1 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 1: Win={result1.win_probability:.2%}")
    
    # Get stats after first call
    stats2 = solver.get_cache_stats()
    print("\nStats after first call:", stats2)
    
    # Second identical analysis
    print("\nRunning second analysis (should hit cache)...")
    result2 = solver.analyze_hand(
        hero_hand=["AS", "KS"],
        num_opponents=2,
        simulation_mode="fast"
    )
    print(f"Result 2: Win={result2.win_probability:.2%}")
    
    # Get stats after second call
    stats3 = solver.get_cache_stats()
    print("\nStats after second call:", stats3)
    
    # Check if total requests increased
    if stats3 and 'unified_cache' in stats3:
        cache_stats1 = stats2['unified_cache']
        cache_stats2 = stats3['unified_cache']
        print(f"\nTotal requests after first: {cache_stats1.get('total_requests', 0)}")
        print(f"Total requests after second: {cache_stats2.get('total_requests', 0)}")
        print(f"Did total requests increase? {cache_stats2.get('total_requests', 0) > cache_stats1.get('total_requests', 0)}")

if __name__ == "__main__":
    main()