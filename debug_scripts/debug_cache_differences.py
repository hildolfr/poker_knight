#!/usr/bin/env python3
"""Debug script to find cache differences."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver
from poker_knight.storage import create_cache_key

def test_multiple_scenarios():
    """Test multiple scenarios to find differences."""
    print("Testing multiple scenarios for cache differences...")
    
    scenarios = [
        (['A♠', 'A♥'], 2, None, "Pocket Aces vs 2"),
        (['K♠', 'K♥'], 1, None, "Pocket Kings vs 1"),
        (['A♠', 'K♠'], 3, None, "AK suited vs 3"),
        (['Q♠', 'Q♥'], 2, ['A♠', 'K♥', '7♣'], "Queens on AK7 flop"),
        (['J♠', '10♠'], 1, ['9♠', '8♠', '2♥'], "JT suited on 982 flop"),
    ]
    
    # Create solver with caching enabled
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    differences = []
    
    for hero_hand, num_opponents, board_cards, description in scenarios:
        print(f"\n{description}:")
        
        # Initialize cache if needed
        solver._initialize_cache_if_needed()
        
        # Clear cache before each test
        if solver._hand_cache:
            solver._hand_cache.clear()
        
        # First run - no cache
        result1 = solver.analyze_hand(hero_hand, num_opponents, board_cards, simulation_mode="fast")
        
        # Force store in cache (simulate what would happen if cache was pre-populated)
        cache_key = create_cache_key(hero_hand, num_opponents, board_cards, "fast")
        
        # Second run - should use cache
        result2 = solver.analyze_hand(hero_hand, num_opponents, board_cards, simulation_mode="fast")
        
        # Third run - clear cache and run again
        if solver._hand_cache:
            solver._hand_cache.clear()
        result3 = solver.analyze_hand(hero_hand, num_opponents, board_cards, simulation_mode="fast")
        
        diff_1_2 = abs(result1.win_probability - result2.win_probability)
        diff_1_3 = abs(result1.win_probability - result3.win_probability)
        
        print(f"  Result 1: {result1.win_probability:.4f} (no cache)")
        print(f"  Result 2: {result2.win_probability:.4f} (from cache)")
        print(f"  Result 3: {result3.win_probability:.4f} (no cache, new sim)")
        print(f"  Diff 1-2: {diff_1_2:.4f}")
        print(f"  Diff 1-3: {diff_1_3:.4f}")
        
        if diff_1_2 > 0:
            differences.append((description, diff_1_2))
        if diff_1_3 > 0.001:
            print(f"  ⚠️  Significant difference between independent runs!")
    
    if differences:
        print(f"\n⚠️  Found {len(differences)} scenarios with cache differences:")
        for desc, diff in differences:
            print(f"  - {desc}: {diff:.4f}")
    else:
        print(f"\n✅ No cache differences found!")
    
    solver.close()

def test_precision_modes():
    """Test different precision modes."""
    print("\nTesting different simulation modes...")
    
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    hero_hand = ['A♠', 'K♦']
    num_opponents = 2
    
    for mode in ["fast", "default", "precision"]:
        print(f"\n{mode.upper()} mode:")
        solver._initialize_cache_if_needed()
        if solver._hand_cache:
            solver._hand_cache.clear()
        
        # Run multiple times
        results = []
        for i in range(5):
            if i > 0 and solver._hand_cache:  # Clear cache for subsequent runs
                solver._hand_cache.clear()
            result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode=mode)
            results.append(result.win_probability)
            print(f"  Run {i+1}: {result.win_probability:.4f} (sims: {result.simulations_run})")
        
        # Calculate variance
        avg = sum(results) / len(results)
        variance = sum((r - avg) ** 2 for r in results) / len(results)
        std_dev = variance ** 0.5
        max_diff = max(results) - min(results)
        
        print(f"  Average: {avg:.4f}")
        print(f"  Std Dev: {std_dev:.4f}")
        print(f"  Max Diff: {max_diff:.4f}")
    
    solver.close()

if __name__ == "__main__":
    test_multiple_scenarios()
    test_precision_modes()