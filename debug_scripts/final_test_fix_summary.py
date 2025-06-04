#!/usr/bin/env python3
"""
Final test to confirm cache fixes are working
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from poker_knight.solver import MonteCarloSolver

def final_cache_test():
    """Final test of cache functionality to confirm fixes."""
    print("🔍 Final Cache Fix Verification")
    print("=" * 50)
    
    # Test the exact scenario from the failing tests
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    try:
        # Clear cache to start fresh
        if solver._hand_cache:
            solver._hand_cache.clear()
        
        # Test scenario that was failing
        hero_hand = ["AS", "KS"]
        num_opponents = 2
        simulation_mode = "fast"
        
        print(f"Testing: {hero_hand} vs {num_opponents} opponents, mode: {simulation_mode}")
        
        # Get initial cache stats
        initial_stats = solver.get_cache_stats()
        print(f"Initial: hits={initial_stats['hand_cache']['cache_hits']}, misses={initial_stats['hand_cache']['cache_misses']}")
        
        # First call - should be cache miss
        print("\n1️⃣ First analysis (expecting cache miss)...")
        result1 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"   Result: win={result1.win_probability:.6f}, sims={result1.simulations_run}")
        
        # Check stats after first call
        stats1 = solver.get_cache_stats()
        print(f"   After 1st: hits={stats1['hand_cache']['cache_hits']}, misses={stats1['hand_cache']['cache_misses']}")
        
        # Second call - should be cache hit
        print("\n2️⃣ Second analysis (expecting cache hit)...")
        result2 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"   Result: win={result2.win_probability:.6f}, sims={result2.simulations_run}")
        
        # Check stats after second call
        stats2 = solver.get_cache_stats()
        print(f"   After 2nd: hits={stats2['hand_cache']['cache_hits']}, misses={stats2['hand_cache']['cache_misses']}")
        
        # Verify the fix
        print(f"\n✅ Verification:")
        cache_hits_increased = stats2['hand_cache']['cache_hits'] > stats1['hand_cache']['cache_hits']
        results_identical = (result1.win_probability == result2.win_probability and 
                           result1.simulations_run == result2.simulations_run)
        
        print(f"   Cache hits increased: {cache_hits_increased}")
        print(f"   Results identical: {results_identical}")
        
        if cache_hits_increased and results_identical:
            print("   ✅ CACHE IS WORKING CORRECTLY!")
        else:
            print("   ❌ Cache issue still exists")
            
        # Test different scenarios to ensure no cross-contamination
        print(f"\n3️⃣ Testing different scenario...")
        result3 = solver.analyze_hand(
            hero_hand=["AH", "KH"],  # Different suits
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"   Different hand result: win={result3.win_probability:.6f}")
        
        # Test same scenario again
        print(f"\n4️⃣ Testing original scenario again...")
        result4 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode=simulation_mode
        )
        print(f"   Same hand again: win={result4.win_probability:.6f}")
        print(f"   Still identical to first: {result1.win_probability == result4.win_probability}")
        
        # Final stats
        final_stats = solver.get_cache_stats()
        print(f"\n📊 Final stats: hits={final_stats['hand_cache']['cache_hits']}, misses={final_stats['hand_cache']['cache_misses']}")
        print(f"   Hit rate: {final_stats['hand_cache']['hit_rate']:.1%}")
        
    finally:
        solver.close()

if __name__ == "__main__":
    final_cache_test()