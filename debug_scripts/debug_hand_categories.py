#!/usr/bin/env python3
"""
Debug script to check if hand categories affect cache behavior
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver

def debug_hand_categories():
    """Debug hand categories and cache interaction."""
    print("🔍 Debugging Hand Categories and Cache")
    print("=" * 50)
    
    # Create solver with caching enabled
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    try:
        # Test scenario
        hero_hand = ["AS", "KS"]
        num_opponents = 2
        simulation_mode = "fast"
        
        print(f"Testing scenario: {hero_hand} vs {num_opponents} opponents")
        print(f"Config includes hand categories: {solver.config['output_settings']['include_hand_categories']}")
        
        # Clear cache
        if solver._hand_cache:
            solver._hand_cache.clear()
        
        # First analysis
        print("\n🔸 First analysis...")
        result1 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        
        print(f"Result 1:")
        print(f"  Win probability: {result1.win_probability:.6f}")
        print(f"  Simulations: {result1.simulations_run}")
        print(f"  Hand categories: {result1.hand_category_frequencies is not None}")
        if result1.hand_category_frequencies:
            print(f"  Categories: {list(result1.hand_category_frequencies.keys())}")
        
        # Second analysis
        print("\n🔸 Second analysis...")
        result2 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        
        print(f"Result 2:")
        print(f"  Win probability: {result2.win_probability:.6f}")
        print(f"  Simulations: {result2.simulations_run}")
        print(f"  Hand categories: {result2.hand_category_frequencies is not None}")
        if result2.hand_category_frequencies:
            print(f"  Categories: {list(result2.hand_category_frequencies.keys())}")
        
        # Compare results
        print(f"\n📊 Comparison:")
        print(f"  Win probabilities match: {result1.win_probability == result2.win_probability}")
        print(f"  Simulations match: {result1.simulations_run == result2.simulations_run}")
        print(f"  Hand categories match: {result1.hand_category_frequencies == result2.hand_category_frequencies}")
        
        # Check cache stats
        stats = solver.get_cache_stats()
        if stats:
            print(f"\n📈 Cache stats:")
            print(f"  Hits: {stats['hand_cache']['cache_hits']}")
            print(f"  Misses: {stats['hand_cache']['cache_misses']}")
            print(f"  Hit rate: {stats['hand_cache']['hit_rate']:.1%}")
        
    finally:
        solver.close()

if __name__ == "__main__":
    debug_hand_categories()