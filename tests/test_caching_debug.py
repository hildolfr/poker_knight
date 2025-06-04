#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug test script for Poker Knight v1.6 caching system
Simplified version to isolate hanging issues
"""

import time
import sys
import os

# Add the poker_knight module to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from poker_knight.storage import HandCache, CacheConfig, create_cache_key
    print("[PASS] Caching system imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import caching system: {e}")
    sys.exit(1)


def test_basic_cache():
    """Test basic cache functionality."""
    print("\n🧪 Testing Basic Cache Operations")
    print("=" * 50)
    
    config = CacheConfig()
    hand_cache = HandCache(config)
    
    # Test cache key generation
    cache_key = create_cache_key(['KS', 'KH'], 1, None, "default")
    print(f"Cache key generated: {cache_key[:16]}...")
    
    # Test storing and retrieving
    test_result = {
        'win_probability': 0.83,
        'simulations_run': 1000,
        'execution_time_ms': 250.0
    }
    
    print("Testing store...")
    stored = hand_cache.store_result(cache_key, test_result)
    print(f"[PASS] Result stored: {stored}")
    
    print("Testing retrieve...")
    retrieved = hand_cache.get_result(cache_key)
    print(f"[PASS] Result retrieved: {retrieved is not None}")
    
    if retrieved:
        print(f"   Win probability: {retrieved['win_probability']}")
    
    assert stored, "Result should have been stored successfully"
    assert retrieved is not None, "Retrieved result should not be None"
    assert retrieved['win_probability'] == test_result['win_probability'], "Retrieved data should match stored data"


def test_cache_miss_then_hit():
    """Test cache miss followed by cache hit."""
    print("\n⚡ Testing Cache Miss -> Hit")
    print("=" * 50)
    
    config = CacheConfig()
    hand_cache = HandCache(config)
    
    cache_key = create_cache_key(['AS', 'AH'], 2, None, "test")
    
    # First call - should be cache miss
    print("First call (should be cache miss)...")
    start_time = time.time()
    result1 = hand_cache.get_result(cache_key)
    time1 = (time.time() - start_time) * 1000
    print(f"Result: {result1}, Time: {time1:.1f}ms")
    
    # Store result
    if result1 is None:
        print("Storing result...")
        test_result = {
            'win_probability': 0.85,
            'simulations_run': 10000,
            'execution_time_ms': 1500.0
        }
        stored = hand_cache.store_result(cache_key, test_result)
        print(f"Stored: {stored}")
    
    # Second call - should be cache hit
    print("Second call (should be cache hit)...")
    start_time = time.time()
    result2 = hand_cache.get_result(cache_key)
    time2 = (time.time() - start_time) * 1000
    print(f"Result: {result2 is not None}, Time: {time2:.1f}ms")
    
    if result2:
        print(f"   Win probability: {result2['win_probability']}")
    
    # Check stats
    stats = hand_cache.get_stats()
    print(f"\n[STATS] Cache Statistics:")
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache misses: {stats.cache_misses}")
    print(f"   Hit rate: {stats.hit_rate:.1%}")
    
    assert result1 is None, "First call should be a cache miss"
    assert result2 is not None, "Second call should be a cache hit"
    assert stats.cache_hits == 1, "Should have exactly 1 cache hit"
    assert stats.cache_misses == 1, "Should have exactly 1 cache miss"


def main():
    """Run debug tests."""
    print("♞ Poker Knight v1.6 Caching System Debug Test")
    print("=" * 60)
    
    try:
        test_basic_cache()
        test_cache_miss_then_hit()
        
        print("\n🎉 Debug tests completed!")
        return 0
        
    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 