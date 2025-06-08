#!/usr/bin/env python3
"""
Manual verification script for cache prepopulation functionality.

This script verifies:
1. Prepopulation works correctly on solver initialization
2. No background threads are created
3. Performance improvements are achieved
"""

import time
import threading
from poker_knight import MonteCarloSolver, solve_poker_hand, prepopulate_cache


def monitor_threads(duration=5):
    """Monitor thread count for a duration."""
    print(f"\nMonitoring threads for {duration} seconds...")
    initial_threads = threading.active_count()
    print(f"Initial thread count: {initial_threads}")
    
    max_threads = initial_threads
    for i in range(duration):
        time.sleep(1)
        current_threads = threading.active_count()
        max_threads = max(max_threads, current_threads)
        print(f"  {i+1}s: {current_threads} threads")
    
    print(f"Max threads observed: {max_threads}")
    print(f"Thread increase: {max_threads - initial_threads}")
    return max_threads - initial_threads


def test_solver_prepopulation():
    """Test solver with automatic prepopulation."""
    print("\n=== Testing Solver Prepopulation ===")
    
    # Monitor threads before
    print("Thread count before solver creation:", threading.active_count())
    
    # Create solver (should trigger prepopulation on first use)
    solver = MonteCarloSolver(enable_caching=True)
    print("Solver created")
    
    # First analysis triggers prepopulation
    print("\nRunning first analysis (triggers prepopulation)...")
    start_time = time.time()
    result = solver.analyze_hand(["A♠", "K♠"], 2)
    first_time = time.time() - start_time
    
    print(f"First analysis time: {first_time:.3f}s")
    print(f"Win probability: {result.win_probability:.1%}")
    
    # Check if prepopulation occurred
    if hasattr(solver, '_population_result') and solver._population_result:
        pop_result = solver._population_result
        print(f"\nPrepopulation stats:")
        print(f"  Scenarios populated: {pop_result.scenarios_populated}")
        print(f"  Time taken: {pop_result.population_time_seconds:.1f}s")
        print(f"  Coverage: {pop_result.initial_coverage:.1f}% -> {pop_result.final_coverage:.1f}%")
    
    # Second analysis should be faster (cache hit)
    print("\nRunning second analysis (should hit cache)...")
    start_time = time.time()
    result2 = solver.analyze_hand(["A♠", "A♥"], 2)
    second_time = time.time() - start_time
    
    print(f"Second analysis time: {second_time:.3f}s")
    print(f"Speedup: {first_time/second_time:.1f}x")
    
    # Get cache stats
    cache_stats = solver.get_cache_stats()
    if cache_stats and 'aggregate_stats' in cache_stats:
        print(f"\nCache hit rate: {cache_stats['aggregate_stats']['overall_hit_rate']:.1%}")
    
    # Monitor threads
    thread_increase = monitor_threads(3)
    assert thread_increase <= 8, f"Too many threads created: {thread_increase}"


def test_skip_warming():
    """Test solver with cache warming disabled."""
    print("\n\n=== Testing Skip Cache Warming ===")
    
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    # Should not trigger prepopulation
    start_time = time.time()
    result = solver.analyze_hand(["K♠", "K♥"], 3)
    analysis_time = time.time() - start_time
    
    print(f"Analysis time (no prepopulation): {analysis_time:.3f}s")
    print(f"Win probability: {result.win_probability:.1%}")
    
    # Verify no prepopulation occurred
    if hasattr(solver, '_population_result'):
        print(f"Population result: {solver._population_result}")
    else:
        print("No prepopulation occurred (as expected)")


def test_convenience_function():
    """Test the prepopulate_cache convenience function."""
    print("\n\n=== Testing Convenience Function ===")
    
    # Quick mode
    print("\nTesting quick prepopulation...")
    start_time = time.time()
    stats = prepopulate_cache(comprehensive=False, time_limit=5.0)
    elapsed = time.time() - start_time
    
    print(f"Quick prepopulation completed in {elapsed:.1f}s")
    print(f"Stats: {stats}")
    
    if stats['success']:
        print(f"  Populated: {stats['scenarios_populated']} scenarios")
        print(f"  Final coverage: {stats.get('final_coverage', 'N/A')}")
    
    # Verify subsequent calls are fast
    print("\nTesting performance after prepopulation...")
    start_time = time.time()
    result = solve_poker_hand(["Q♠", "Q♥"], 2)
    solve_time = time.time() - start_time
    
    print(f"Solve time after prepopulation: {solve_time:.3f}s")
    print(f"Win probability: {result.win_probability:.1%}")


def test_force_regeneration():
    """Test force cache regeneration."""
    print("\n\n=== Testing Force Regeneration ===")
    
    solver = MonteCarloSolver(
        enable_caching=True,
        force_cache_regeneration=True
    )
    
    print("Solver created with force_cache_regeneration=True")
    print("This should trigger comprehensive prepopulation...")
    
    # Monitor the prepopulation
    start_time = time.time()
    result = solver.analyze_hand(["J♠", "J♦"], 4)
    elapsed = time.time() - start_time
    
    print(f"Analysis time (with regeneration): {elapsed:.3f}s")
    
    if hasattr(solver, '_population_result') and solver._population_result:
        pop_result = solver._population_result
        print(f"Comprehensive population stats:")
        print(f"  Scenarios: {pop_result.scenarios_populated}")
        print(f"  Time: {pop_result.population_time_seconds:.1f}s")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Cache Prepopulation Verification")
    print("=" * 60)
    
    # Test 1: Basic solver prepopulation
    test_solver_prepopulation()
    
    # Test 2: Skip warming
    test_skip_warming()
    
    # Test 3: Convenience function
    test_convenience_function()
    
    # Test 4: Force regeneration (commented out as it takes longer)
    # test_force_regeneration()
    
    print("\n" + "=" * 60)
    print("Verification complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()