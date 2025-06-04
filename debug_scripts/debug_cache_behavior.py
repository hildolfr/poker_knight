#!/usr/bin/env python3
"""Debug script to understand cache behavior."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver

def test_cache_consistency():
    """Test if cached results are identical to original results."""
    print("Testing cache consistency...")
    
    # Create solver with caching enabled
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    # Test scenario
    hero_hand = ['A♠', 'K♠']
    num_opponents = 2
    
    # First run - should create cache entry
    print("\nFirst run (no cache):")
    result1 = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="fast")
    print(f"Win probability: {result1.win_probability}")
    print(f"Raw value: {repr(result1.win_probability)}")
    print(f"Simulations: {result1.simulations_run}")
    
    # Second run - should use cache
    print("\nSecond run (from cache):")
    result2 = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="fast")
    print(f"Win probability: {result2.win_probability}")
    print(f"Raw value: {repr(result2.win_probability)}")
    print(f"Simulations: {result2.simulations_run}")
    
    # Compare results
    print(f"\nResults equal: {result1.win_probability == result2.win_probability}")
    print(f"Difference: {abs(result1.win_probability - result2.win_probability)}")
    
    # Third run - clear cache and run again
    print("\nThird run (no cache, new simulation):")
    solver._hand_cache.clear()
    result3 = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="fast")
    print(f"Win probability: {result3.win_probability}")
    print(f"Raw value: {repr(result3.win_probability)}")
    print(f"Simulations: {result3.simulations_run}")
    
    print(f"\nFirst vs Third equal: {result1.win_probability == result3.win_probability}")
    print(f"Difference: {abs(result1.win_probability - result3.win_probability)}")
    
    solver.close()

if __name__ == "__main__":
    test_cache_consistency()