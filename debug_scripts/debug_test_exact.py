#!/usr/bin/env python3
"""
Test exact cache behavior that was failing in tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver

def test_exact_failing_pattern():
    """Test the exact pattern that was failing in the tests."""
    print("🔍 Testing Exact Failing Pattern")
    print("=" * 50)
    
    solver = MonteCarloSolver(enable_caching=True)
    
    try:
        # Clear cache completely
        if hasattr(solver, '_hand_cache') and solver._hand_cache:
            solver._hand_cache.clear()
        
        # Test the exact same pattern as the failing tests
        print("Testing cache hit and miss behavior (like failing test)...")
        
        # First analysis should be a cache miss
        result1 = solver.analyze_hand(
            hero_hand=["AS", "KS"],
            num_opponents=2,
            simulation_mode="fast"
        )
        
        # Get initial stats
        stats = solver.get_cache_stats()
        if stats:  # Only test if caching is available
            initial_hits = stats['hand_cache']['cache_hits']
            initial_misses = stats['hand_cache']['cache_misses']
            print(f"After first call: hits={initial_hits}, misses={initial_misses}")
            
            # Second identical analysis should be a cache hit
            result2 = solver.analyze_hand(
                hero_hand=["AS", "KS"],
                num_opponents=2,
                simulation_mode="fast"
            )
            
            # Get updated stats
            stats_after = solver.get_cache_stats()
            
            print(f"After second call: hits={stats_after['hand_cache']['cache_hits']}, misses={stats_after['hand_cache']['cache_misses']}")
            
            # Verify cache hit occurred (this was failing before)
            hit_increase = stats_after['hand_cache']['cache_hits'] - initial_hits
            miss_increase = stats_after['hand_cache']['cache_misses'] - initial_misses
            
            print(f"\n📊 Analysis:")
            print(f"   Cache hits increased by: {hit_increase}")
            print(f"   Cache misses increased by: {miss_increase}")
            print(f"   Expected: hits +1, misses +0")
            
            # Check if this matches test expectations
            cache_hit_occurred = hit_increase == 1 and miss_increase == 0
            print(f"   Cache hit occurred correctly: {cache_hit_occurred}")
            
            # Results should be IDENTICAL when from cache (not just close)
            results_identical = (result1.win_probability == result2.win_probability and
                               result1.tie_probability == result2.tie_probability and
                               result1.loss_probability == result2.loss_probability)
            print(f"   Results identical: {results_identical}")
            
            if cache_hit_occurred and results_identical:
                print("\n✅ TEST PATTERN NOW PASSES!")
            else:
                print("\n❌ Test pattern still fails")
                print(f"     Win prob: {result1.win_probability} vs {result2.win_probability}")
                print(f"     Tie prob: {result1.tie_probability} vs {result2.tie_probability}")
                print(f"     Loss prob: {result1.loss_probability} vs {result2.loss_probability}")
        
    finally:
        solver.close()

if __name__ == "__main__":
    test_exact_failing_pattern()