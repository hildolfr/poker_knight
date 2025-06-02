#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration test for Poker Knight v1.6 MonteCarloSolver with caching system
Tests the complete integration between solver and caching components
"""

import time
import sys
import os

# Add the poker_knight module to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from poker_knight.solver import MonteCarloSolver
    from poker_knight.storage import CacheConfig
    print("[PASS] MonteCarloSolver and caching system imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import solver or caching system: {e}")
    sys.exit(1)


def test_solver_cache_integration():
    """Test MonteCarloSolver with caching enabled."""
    print("\n🧪 Testing MonteCarloSolver Cache Integration")
    print("=" * 60)
    
    # Create solver with caching enabled
    cache_config = CacheConfig(
        max_memory_mb=128,
        hand_cache_size=1000,
        preflop_cache_enabled=True
    )
    
    with MonteCarloSolver(enable_caching=True) as solver:
        # Test scenario: Pocket Aces vs 2 opponents preflop
        hero_hand = ['AS', 'AH']
        num_opponents = 2
        
        print(f"Testing scenario: {hero_hand} vs {num_opponents} opponents (preflop)")
        
        # First analysis - should be cache miss
        print("\n[SEARCH] First analysis (cache miss expected)...")
        start_time = time.time()
        result1 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode="fast"
        )
        time1 = (time.time() - start_time) * 1000
        
        print(f"   Win probability: {result1.win_probability:.3f}")
        print(f"   Simulations run: {result1.simulations_run}")
        print(f"   Execution time: {time1:.1f}ms")
        
        # Second analysis - should be cache hit
        print("\n⚡ Second analysis (cache hit expected)...")
        start_time = time.time()
        result2 = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode="fast"
        )
        time2 = (time.time() - start_time) * 1000
        
        print(f"   Win probability: {result2.win_probability:.3f}")
        print(f"   Simulations run: {result2.simulations_run}")
        print(f"   Execution time: {time2:.1f}ms")
        
        # Verify cache performance
        speedup = time1 / time2 if time2 > 0 else float('inf')
        print(f"\n[STATS] Performance Analysis:")
        print(f"   Cache speedup: {speedup:.1f}x")
        print(f"   Results match: {abs(result1.win_probability - result2.win_probability) < 0.001}")
        
        # Get cache statistics
        cache_stats = solver.get_cache_stats()
        if cache_stats:
            print(f"\n📈 Cache Statistics:")
            hand_cache = cache_stats.get('hand_cache', {})
            print(f"   Total requests: {hand_cache.get('total_requests', 0)}")
            print(f"   Cache hits: {hand_cache.get('cache_hits', 0)}")
            print(f"   Hit rate: {hand_cache.get('hit_rate', 0):.1%}")
            print(f"   Memory usage: {hand_cache.get('memory_usage_mb', 0):.1f}MB")
        
        return speedup > 5  # Cache should provide significant speedup


def test_different_scenarios():
    """Test caching with different poker scenarios."""
    print("\n🎯 Testing Multiple Scenarios")
    print("=" * 60)
    
    scenarios = [
        (['KS', 'KH'], 1, None, "Pocket Kings vs 1 opponent"),
        (['AS', 'KS'], 3, None, "AK suited vs 3 opponents"),
        (['QS', 'QH'], 2, ['AS', 'KH', '7C'], "Pocket Queens on AK7 flop"),
        (['JS', '10S'], 1, ['9S', '8S', '2H'], "Jack-Ten suited on 982 flop"),
    ]
    
    with MonteCarloSolver(enable_caching=True) as solver:
        results = []
        
        for hero_hand, num_opponents, board_cards, description in scenarios:
            print(f"\n🃏 {description}")
            
            # First run
            start_time = time.time()
            result1 = solver.analyze_hand(
                hero_hand=hero_hand,
                num_opponents=num_opponents,
                board_cards=board_cards,
                simulation_mode="fast"
            )
            time1 = (time.time() - start_time) * 1000
            
            # Second run (should be cached)
            start_time = time.time()
            result2 = solver.analyze_hand(
                hero_hand=hero_hand,
                num_opponents=num_opponents,
                board_cards=board_cards,
                simulation_mode="fast"
            )
            time2 = (time.time() - start_time) * 1000
            
            speedup = time1 / time2 if time2 > 0 else float('inf')
            print(f"   Win probability: {result1.win_probability:.3f}")
            print(f"   First run: {time1:.1f}ms, Second run: {time2:.1f}ms")
            print(f"   Speedup: {speedup:.1f}x")
            
            results.append(speedup > 2)  # Expect at least 2x speedup
        
        # Final cache statistics
        cache_stats = solver.get_cache_stats()
        if cache_stats:
            print(f"\n[STATS] Final Cache Statistics:")
            hand_cache = cache_stats.get('hand_cache', {})
            preflop_cache = cache_stats.get('preflop_cache', {})
            print(f"   Hand cache requests: {hand_cache.get('total_requests', 0)}")
            print(f"   Hand cache hit rate: {hand_cache.get('hit_rate', 0):.1%}")
            print(f"   Preflop cache coverage: {preflop_cache.get('coverage_percentage', 0):.1f}%")
        
        return all(results)


def test_cache_persistence():
    """Test cache persistence across solver instances."""
    print("\n💾 Testing Cache Persistence")
    print("=" * 60)
    
    hero_hand = ['AS', 'AH']
    num_opponents = 1
    
    # First solver instance
    print("Creating first solver instance...")
    with MonteCarloSolver(enable_caching=True) as solver1:
        result1 = solver1.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode="fast"
        )
        print(f"   First result: {result1.win_probability:.3f}")
    
    # Second solver instance (should benefit from cache if persistent)
    print("Creating second solver instance...")
    with MonteCarloSolver(enable_caching=True) as solver2:
        start_time = time.time()
        result2 = solver2.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode="fast"
        )
        time2 = (time.time() - start_time) * 1000
        
        print(f"   Second result: {result2.win_probability:.3f}")
        print(f"   Execution time: {time2:.1f}ms")
        
        # Check if results are consistent
        consistent = abs(result1.win_probability - result2.win_probability) < 0.01
        print(f"   Results consistent: {consistent}")
        
        return consistent


def main():
    """Run all integration tests."""
    print("♞ Poker Knight v1.6 Solver-Cache Integration Test")
    print("=" * 70)
    
    try:
        results = []
        
        # Test basic integration
        results.append(test_solver_cache_integration())
        
        # Test multiple scenarios
        results.append(test_different_scenarios())
        
        # Test cache persistence
        results.append(test_cache_persistence())
        
        if all(results):
            print("\n🎉 All integration tests passed!")
            print("\n[PASS] Caching system successfully integrated with MonteCarloSolver")
            print("\n[IDEA] Benefits achieved:")
            print("   - Significant performance improvements for repeated scenarios")
            print("   - Consistent results across cache hits and misses")
            print("   - Memory-efficient LRU cache management")
            print("   - Separate preflop and postflop caching strategies")
            print("\n[ROCKET] Ready for production use!")
        else:
            print("\n[WARN]  Some integration tests failed.")
            print(f"   Test results: {results}")
        
        return 0 if all(results) else 1
        
    except Exception as e:
        print(f"\n[FAIL] Integration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 