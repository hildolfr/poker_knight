#!/usr/bin/env python3
"""Debug script to test if convergence analysis is causing cache variability."""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight import MonteCarloSolver
from poker_knight.storage.unified_cache import clear_unified_cache

def test_convergence_variability():
    """Test if convergence analysis early stopping causes different results."""
    print("Testing if convergence analysis causes result variability...\n")
    
    # Test scenarios
    test_cases = [
        (['A♠', 'K♠'], 2, None, "default"),      # AKs preflop
        (['Q♠', 'Q♥'], 3, None, "default"),      # QQ preflop
        (['7♥', '2♦'], 1, None, "default"),      # 72o heads-up
        (['A♠', 'K♠'], 2, ['Q♠', 'J♠', '10♥'], "default"),  # Royal flush draw
    ]
    
    for hero_hand, num_opponents, board, mode in test_cases:
        board_str = f"{board}" if board else "preflop"
        print(f"\nTesting: {hero_hand} vs {num_opponents} opponents, {board_str}")
        
        # Clear cache for each test case
        clear_unified_cache()
        
        # Create solver with caching disabled to see raw Monte Carlo variability
        solver_no_cache = MonteCarloSolver(enable_caching=False)
        
        # Run multiple times without cache
        print("  Without cache (raw Monte Carlo):")
        results_no_cache = []
        for i in range(5):
            result = solver_no_cache.analyze_hand(hero_hand, num_opponents, board, mode)
            results_no_cache.append({
                'win': result.win_probability,
                'sims': result.simulations_run,
                'stopped_early': result.stopped_early,
                'converged': result.convergence_achieved
            })
            print(f"    Run {i+1}: win={result.win_probability:.4f}, sims={result.simulations_run}, "
                  f"early_stop={result.stopped_early}, converged={result.convergence_achieved}")
        
        # Calculate variance
        wins = [r['win'] for r in results_no_cache]
        if len(set(wins)) > 1:
            variance = max(wins) - min(wins)
            print(f"  Variance without cache: {variance:.4f} (range: {min(wins):.4f} - {max(wins):.4f})")
        else:
            print(f"  All results identical: {wins[0]:.4f}")
        
        solver_no_cache.close()
        
        # Now test with cache enabled
        print("\n  With cache enabled:")
        solver_with_cache = MonteCarloSolver(enable_caching=True)
        
        results_with_cache = []
        for i in range(5):
            result = solver_with_cache.analyze_hand(hero_hand, num_opponents, board, mode)
            results_with_cache.append({
                'win': result.win_probability,
                'sims': result.simulations_run,
                'from_cache': i > 0  # First run computes, rest should be cached
            })
            print(f"    Run {i+1}: win={result.win_probability:.4f}, sims={result.simulations_run}")
        
        # Check cache consistency
        wins_cached = [r['win'] for r in results_with_cache]
        if len(set(wins_cached[1:])) == 1:  # All cached results should be identical
            print(f"  ✓ Cached results consistent: {wins_cached[1]:.4f}")
        else:
            print(f"  ⚠️  Cached results vary: {wins_cached}")
        
        # Check if first result matches cached results
        if wins_cached[0] == wins_cached[1]:
            print(f"  ✓ First computation matches cached results")
        else:
            print(f"  ⚠️  First computation ({wins_cached[0]:.4f}) differs from cache ({wins_cached[1]:.4f})")
        
        solver_with_cache.close()
    
    # Test with different simulation modes
    print("\n\nTesting simulation modes effect on caching:")
    clear_unified_cache()
    
    solver = MonteCarloSolver(enable_caching=True)
    hero_hand = ['A♠', 'K♠']
    num_opponents = 2
    
    for mode in ["fast", "default", "precision"]:
        print(f"\n  Mode: {mode}")
        
        # First run
        result1 = solver.analyze_hand(hero_hand, num_opponents, None, mode)
        print(f"    First:  win={result1.win_probability:.4f}, sims={result1.simulations_run}")
        
        # Second run (cached)
        result2 = solver.analyze_hand(hero_hand, num_opponents, None, mode)
        print(f"    Second: win={result2.win_probability:.4f}, sims={result2.simulations_run}")
        
        if result1.win_probability == result2.win_probability:
            print(f"    ✓ Cache working correctly")
        else:
            print(f"    ⚠️  Cache mismatch!")
    
    solver.close()


if __name__ == "__main__":
    test_convergence_variability()