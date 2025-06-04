#!/usr/bin/env python3
"""Debug script showing the cache behavior issue and solution."""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight import MonteCarloSolver
from poker_knight.storage.unified_cache import clear_unified_cache

def demonstrate_issue_and_solution():
    """Demonstrate the cache issue and show the solution."""
    
    print("DEMONSTRATING THE CACHE ISSUE AND SOLUTION")
    print("=" * 50)
    
    hero_hand = ['A♠', 'K♠']
    num_opponents = 2
    
    # ISSUE: Multiple solver instances or cache clears
    print("\n1. THE ISSUE - Multiple solver instances (simulating test isolation):")
    print("   Each test creates a new solver, leading to different Monte Carlo results")
    
    for i in range(3):
        # Simulate test isolation by clearing cache and creating new solver
        clear_unified_cache()
        solver = MonteCarloSolver(enable_caching=True)
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
        print(f"   Test {i+1}: win_probability = {result.win_probability:.4f} (sims: {result.simulations_run})")
        solver.close()
    
    print("\n   ⚠️  Notice: Different results due to Monte Carlo randomness!")
    
    # SOLUTION 1: Use a single solver instance
    print("\n2. SOLUTION 1 - Reuse solver instance within test suite:")
    clear_unified_cache()
    solver = MonteCarloSolver(enable_caching=True)
    
    for i in range(3):
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
        print(f"   Test {i+1}: win_probability = {result.win_probability:.4f} (sims: {result.simulations_run})")
    
    print("\n   ✓ Success: Consistent results from cache after first computation")
    solver.close()
    
    # SOLUTION 2: Pre-warm the cache
    print("\n3. SOLUTION 2 - Pre-warm cache before tests:")
    clear_unified_cache()
    
    # Pre-warm phase
    print("   Pre-warming cache...")
    warmup_solver = MonteCarloSolver(enable_caching=True)
    warmup_result = warmup_solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
    print(f"   Cached value: {warmup_result.win_probability:.4f}")
    warmup_solver.close()
    
    # Now run "isolated" tests
    print("\n   Running isolated tests with pre-warmed cache:")
    for i in range(3):
        # Each test creates its own solver (simulating test isolation)
        solver = MonteCarloSolver(enable_caching=True)
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
        print(f"   Test {i+1}: win_probability = {result.win_probability:.4f}")
        solver.close()
    
    print("\n   ✓ Success: All tests get consistent cached results")
    
    # EXPLANATION
    print("\n4. EXPLANATION:")
    print("   - Monte Carlo simulations have inherent randomness")
    print("   - Convergence analysis stops at different points")
    print("   - Results can vary by ~1-2% between runs")
    print("   - Cache stores the first computed result")
    print("   - Subsequent calls return the exact cached value")
    
    # RECOMMENDATION
    print("\n5. RECOMMENDATIONS FOR TESTS:")
    print("   a) Use approximate assertions: assertAlmostEqual(result, expected, delta=0.02)")
    print("   b) Pre-warm cache in setUpClass() for deterministic tests")
    print("   c) Disable cache for statistical tests that need randomness")
    print("   d) Use a shared solver instance within a test class")
    
    # Example of approximate assertion
    print("\n6. EXAMPLE TEST ASSERTION:")
    clear_unified_cache()
    
    # Run multiple times to show variance
    results = []
    for i in range(5):
        solver = MonteCarloSolver(enable_caching=False)  # Disable cache to show variance
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
        results.append(result.win_probability)
        solver.close()
    
    mean_result = sum(results) / len(results)
    max_variance = max(results) - min(results)
    
    print(f"   Results without cache: {[f'{r:.4f}' for r in results]}")
    print(f"   Mean: {mean_result:.4f}, Variance: {max_variance:.4f}")
    print(f"\n   Appropriate test assertion:")
    print(f"   self.assertAlmostEqual(result.win_probability, 0.495, delta=0.02)")
    print(f"   # Allows for Monte Carlo variance of ±2%")


if __name__ == "__main__":
    demonstrate_issue_and_solution()