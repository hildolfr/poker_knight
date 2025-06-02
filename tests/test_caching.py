#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Poker Knight v1.6 caching system (Task 1.3)

This script validates:
- Cache components functionality 
- Cache hit/miss functionality
- Cache statistics
"""

import time
import sys
import os

# Add the poker_knight module to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from poker_knight.storage import HandCache, PreflopRangeCache, CacheConfig, create_cache_key
    print("[PASS] Caching system imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import caching system: {e}")
    sys.exit(1)


def test_cache_components():
    """Test cache components directly."""
    print("\n🧪 Testing Cache Components")
    print("=" * 50)
    
    # Test cache configuration
    config = CacheConfig(
        max_memory_mb=64,
        hand_cache_size=100,
        preflop_cache_enabled=True
    )
    
    # Test hand cache
    hand_cache = HandCache(config)
    print(f"Hand cache created successfully")
    
    # Test preflop cache  
    preflop_cache = PreflopRangeCache(config)
    print(f"Preflop cache created successfully")
    
    # Test cache key generation
    cache_key = create_cache_key(
        hero_hand=['AS', 'AH'],
        num_opponents=2,
        board_cards=None,
        simulation_mode="fast"
    )
    print(f"Cache key generated: {cache_key[:16]}...")
    
    # Test storing and retrieving from hand cache
    test_result = {
        'win_probability': 0.85,
        'tie_probability': 0.02,
        'loss_probability': 0.13,
        'simulations_run': 10000,
        'execution_time_ms': 150.5
    }
    
    print("\nTesting hand cache storage and retrieval:")
    stored = hand_cache.store_result(cache_key, test_result)
    print(f"[PASS] Result stored: {stored}")
    
    retrieved = hand_cache.get_result(cache_key)
    print(f"[PASS] Result retrieved: {retrieved is not None}")
    
    if retrieved:
        print(f"   Win probability: {retrieved['win_probability']}")
        print(f"   Simulations: {retrieved['simulations_run']}")
    
    # Test cache statistics
    stats = hand_cache.get_stats()
    print(f"\n[STATS] Cache Statistics:")
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache misses: {stats.cache_misses}")
    print(f"   Hit rate: {stats.hit_rate:.1%}")
    print(f"   Memory usage: {stats.memory_usage_mb:.1f}MB")
    
    return stats.total_requests > 0 and stats.cache_hits > 0


def test_preflop_cache():
    """Test preflop cache functionality."""
    print("\n🎯 Testing Preflop Cache")
    print("=" * 50)
    
    config = CacheConfig()
    preflop_cache = PreflopRangeCache(config)
    
    # Test preflop result storage and retrieval
    hero_hand = ['AS', 'AH']
    test_result = {
        'win_probability': 0.85,
        'tie_probability': 0.02,
        'loss_probability': 0.13,
        'simulations_run': 10000
    }
    
    print(f"Testing with hand: {hero_hand}")
    
    # Store result
    stored = preflop_cache.store_preflop_result(hero_hand, 2, test_result, "button")
    print(f"[PASS] Preflop result stored: {stored}")
    
    # Retrieve result
    retrieved = preflop_cache.get_preflop_result(hero_hand, 2, "button")
    print(f"[PASS] Preflop result retrieved: {retrieved is not None}")
    
    if retrieved:
        print(f"   Win probability: {retrieved['win_probability']}")
        
    # Test cache coverage
    coverage = preflop_cache.get_cache_coverage()
    print(f"\n[STATS] Preflop Cache Coverage:")
    print(f"   Cached combinations: {coverage['cached_combinations']}")
    print(f"   Coverage: {coverage['coverage_percentage']:.1f}%")
    
    return retrieved is not None


def test_cache_performance():
    """Test cache performance improvement."""
    print("\n⚡ Testing Cache Performance")
    print("=" * 50)
    
    config = CacheConfig()
    hand_cache = HandCache(config)
    
    # Simulate expensive computation
    cache_key = create_cache_key(['KS', 'KH'], 1, None, "default")
    
    expensive_result = {
        'win_probability': 0.83,
        'simulations_run': 100000,
        'execution_time_ms': 2500.0  # Simulated expensive computation
    }
    
    # First "computation" - cache miss
    print("Testing cache miss...")
    start_time = time.time()
    result1 = hand_cache.get_result(cache_key)
    if result1 is None:
        # Simulate computation time (but don't actually sleep to avoid hanging)
        print("Cache miss - simulating computation...")
        hand_cache.store_result(cache_key, expensive_result)
        result1 = expensive_result
    time1 = (time.time() - start_time) * 1000
    
    # Second "computation" - cache hit
    print("Testing cache hit...")
    start_time = time.time()
    result2 = hand_cache.get_result(cache_key)
    time2 = (time.time() - start_time) * 1000
    
    print(f"First run (cache miss): {time1:.1f}ms")
    print(f"Second run (cache hit): {time2:.1f}ms")
    
    # Cache hit should be much faster (even without sleep, should be sub-millisecond)
    if result2 is not None and time2 < 10:  # Cache hit should be very fast
        print(f"[PASS] Cache working! Cache hit was {time2:.1f}ms")
        return True
    else:
        print(f"[WARN]  Cache may not be working optimally")
        print(f"   Result2 is not None: {result2 is not None}")
        print(f"   Time2: {time2:.1f}ms")
        return False


def main():
    """Run all caching tests."""
    print("♞ Poker Knight v1.6 Caching System Test")
    print("=" * 60)
    sys.stdout.flush()
    
    try:
        results = []
        
        print("Running cache components test...")
        sys.stdout.flush()
        results.append(test_cache_components())
        
        print("Running preflop cache test...")  
        sys.stdout.flush()
        results.append(test_preflop_cache())
        
        print("Running cache performance test...")
        sys.stdout.flush()
        results.append(test_cache_performance())
        
        print("\nAll tests completed. Processing results...")
        sys.stdout.flush()
        
        if all(results):
            print("\n🎉 All cache component tests passed!")
            print("\n[IDEA] Next steps:")
            print("   - Integrate caching with MonteCarloSolver")
            print("   - Add performance monitoring")
            print("   - Consider Redis persistence for enterprise deployment")
        else:
            print("\n[WARN]  Some tests failed. Check cache implementation.")
            print(f"   Test results: {results}")
        
        sys.stdout.flush()
        return 0 if all(results) else 1
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 