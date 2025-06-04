#!/usr/bin/env python3
"""Debug script to reproduce the exact cache issue with different results."""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight import MonteCarloSolver, solve_poker_hand
from poker_knight.storage.unified_cache import get_unified_cache, create_cache_key, clear_unified_cache

def test_exact_issue():
    """Test the exact issue where cache returns different results (0.4928 vs 0.5026)."""
    print("Testing exact cache issue with different results...\n")
    
    # Clear cache to start fresh
    clear_unified_cache()
    
    # Test 1: Using solve_poker_hand (the convenience function)
    print("1. Testing with solve_poker_hand():")
    hero_hand = ['A♠', 'K♠']
    num_opponents = 2
    
    # Multiple runs
    results = []
    for i in range(5):
        result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        results.append(result.win_probability)
        print(f"   Run {i+1}: win_probability = {result.win_probability}")
    
    # Check if all results are identical
    all_same = all(r == results[0] for r in results)
    print(f"\n   All results identical: {all_same}")
    if not all_same:
        print(f"   Unique values: {set(results)}")
    
    # Test 2: Using MonteCarloSolver directly with fresh instance each time
    print("\n2. Testing with fresh MonteCarloSolver instances:")
    clear_unified_cache()
    
    results2 = []
    for i in range(5):
        solver = MonteCarloSolver(enable_caching=True)
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
        results2.append(result.win_probability)
        print(f"   Run {i+1}: win_probability = {result.win_probability}")
        solver.close()
    
    all_same2 = all(r == results2[0] for r in results2)
    print(f"\n   All results identical: {all_same2}")
    if not all_same2:
        print(f"   Unique values: {set(results2)}")
    
    # Test 3: Check if the issue is with convergence settings
    print("\n3. Testing with fast mode (fewer simulations):")
    clear_unified_cache()
    
    results3 = []
    for i in range(5):
        result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="fast")
        results3.append(result.win_probability)
        print(f"   Run {i+1}: win_probability = {result.win_probability}")
    
    all_same3 = all(r == results3[0] for r in results3)
    print(f"\n   All results identical: {all_same3}")
    if not all_same3:
        print(f"   Unique values: {set(results3)}")
    
    # Test 4: Check cache behavior with a single solver instance
    print("\n4. Testing with single solver instance:")
    clear_unified_cache()
    
    solver = MonteCarloSolver(enable_caching=True)
    results4 = []
    
    for i in range(5):
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
        results4.append(result.win_probability)
        print(f"   Run {i+1}: win_probability = {result.win_probability}, sims = {result.simulations_run}")
        
        # Check cache stats
        stats = solver.get_cache_stats()
        if stats and 'unified_cache' in stats:
            cache_info = stats['unified_cache']
            print(f"         Cache: requests={cache_info['total_requests']}, hits={cache_info['cache_hits']}, misses={cache_info['cache_misses']}")
    
    all_same4 = all(r == results4[0] for r in results4)
    print(f"\n   All results identical: {all_same4}")
    if not all_same4:
        print(f"   Unique values: {set(results4)}")
    
    # Test 5: Check if convergence analysis is causing variability
    print("\n5. Checking convergence details:")
    clear_unified_cache()
    
    solver = MonteCarloSolver(enable_caching=True)
    
    # First run
    result1 = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
    print(f"   First run:")
    print(f"     Win probability: {result1.win_probability}")
    print(f"     Simulations: {result1.simulations_run}")
    print(f"     Stopped early: {result1.stopped_early}")
    print(f"     Convergence achieved: {result1.convergence_achieved}")
    
    # Second run (should be cached)
    result2 = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
    print(f"\n   Second run:")
    print(f"     Win probability: {result2.win_probability}")
    print(f"     Simulations: {result2.simulations_run}")
    print(f"     Stopped early: {result2.stopped_early}")
    print(f"     Convergence achieved: {result2.convergence_achieved}")
    
    print(f"\n   Results identical: {result1.win_probability == result2.win_probability}")
    
    return all_same or all_same2 or all_same3 or all_same4


if __name__ == "__main__":
    test_exact_issue()