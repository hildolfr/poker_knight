#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Knight Redis Integration Test

Comprehensive test for Redis-based cache persistence in the Poker Knight system.
Tests both basic functionality and enterprise-grade scenarios.

Author: hildolfr
"""

import sys
import os
import time
import json
from typing import Dict, Any, Optional

# Add the poker_knight package to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from poker_knight.storage.cache import (
        CacheConfig, HandCache, PreflopRangeCache, 
        create_cache_key, REDIS_AVAILABLE
    )
    print("Successfully imported caching system")
except ImportError as e:
    print(f"Failed to import caching system: {e}")
    sys.exit(1)

def check_redis_availability():
    """Check if Redis is available and connectable."""
    print("\nChecking Redis Availability")
    print("=" * 50)
    
    if not REDIS_AVAILABLE:
        print("Redis library not available")
        print("   Install with: pip install redis>=4.0.0")
        return False
    
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=False)
        client.ping()
        print("Redis server is running and accessible")
        
        # Check Redis version
        info = client.info()
        redis_version = info.get('redis_version', 'unknown')
        print(f"   Redis version: {redis_version}")
        print(f"   Memory usage: {info.get('used_memory_human', 'unknown')}")
        
        return True
        
    except redis.ConnectionError:
        print("Redis server not running")
        print("   Start Redis with: redis-server")
        return False
    except Exception as e:
        print(f"Redis connection error: {e}")
        return False

def test_redis_cache_persistence():
    """Test Redis cache persistence functionality."""
    print("\nTesting Redis Cache Persistence")
    print("=" * 50)
    
    # Create config with Redis enabled
    config = CacheConfig(
        max_memory_mb=64,
        hand_cache_size=100,
        enable_persistence=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0
    )
    
    # Test hand cache with persistence
    print("Creating hand cache with Redis persistence...")
    hand_cache = HandCache(config)
    
    # Verify Redis client is connected
    if hand_cache._redis_client is None:
        print("Redis client not initialized")
        assert False, "Redis client should be initialized"
    
    print("Redis client initialized successfully")
    
    # Clear any existing test data
    hand_cache.clear()
    print("Cache cleared")
    
    # Test data
    test_scenarios = [
        {
            'hero_hand': ['As', 'Ah'],
            'num_opponents': 2,
            'board_cards': None,
            'expected_win_rate': 0.85
        },
        {
            'hero_hand': ['Ks', 'Kh'],
            'num_opponents': 1,
            'board_cards': ['As', '7h', '2c'],
            'expected_win_rate': 0.15
        },
        {
            'hero_hand': ['Qs', 'Qh'],
            'num_opponents': 3,
            'board_cards': None,
            'expected_win_rate': 0.60
        }
    ]
    
    # Store test scenarios
    cache_keys = []
    for i, scenario in enumerate(test_scenarios):
        cache_key = create_cache_key(
            hero_hand=scenario['hero_hand'],
            num_opponents=scenario['num_opponents'],
            board_cards=scenario['board_cards'],
            simulation_mode="redis_test"
        )
        cache_keys.append(cache_key)
        
        result = {
            'win_probability': scenario['expected_win_rate'],
            'tie_probability': 0.02,
            'loss_probability': 1.0 - scenario['expected_win_rate'] - 0.02,
            'simulations_run': 10000,
            'execution_time_ms': 150.5,
            'scenario_id': i,
            'redis_test': True
        }
        
        stored = hand_cache.store_result(cache_key, result)
        print(f"Scenario {i+1} stored: {stored}")
    
    # Verify immediate retrieval (from memory cache)
    print("\nTesting immediate retrieval (memory cache):")
    for i, cache_key in enumerate(cache_keys):
        result = hand_cache.get_result(cache_key)
        if result and result.get('scenario_id') == i:
            print(f"Scenario {i+1} retrieved from memory cache")
        else:
            print(f"Scenario {i+1} not found in memory cache")
            assert False, f"Scenario {i+1} should be found in memory cache"
    
    # Test Redis persistence by creating new cache instance
    print("\nTesting Redis persistence (new cache instance):")
    hand_cache_new = HandCache(config)
    
    # Memory cache should be empty, but Redis should have the data
    for i, cache_key in enumerate(cache_keys):
        result = hand_cache_new.get_result(cache_key)
        if result and result.get('scenario_id') == i and result.get('redis_test'):
            print(f"Scenario {i+1} retrieved from Redis persistence")
        else:
            print(f"Scenario {i+1} not found in Redis persistence")
            assert False, f"Scenario {i+1} should be found in Redis persistence"
    
    # Get statistics
    stats = hand_cache_new.get_stats()
    print(f"\nCache Statistics:")
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache misses: {stats.cache_misses}")
    print(f"   Hit rate: {stats.hit_rate:.1%}")
    print(f"   Persistence loads: {stats.persistence_loads}")
    print(f"   Persistence saves: {stats.persistence_saves}")
    
    # Clean up test data
    hand_cache_new.clear()
    print("Test data cleaned up")
    
    # Add final assertions
    assert stats.total_requests >= len(cache_keys), "Total requests should be at least the number of cache keys"
    assert stats.cache_hits >= len(cache_keys), "Cache hits should be at least the number of cache keys"

def test_redis_performance():
    """Test Redis cache performance."""
    print("\nTesting Redis Cache Performance")
    print("=" * 50)
    
    config = CacheConfig(
        max_memory_mb=128,
        hand_cache_size=1000,
        enable_persistence=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0
    )
    
    hand_cache = HandCache(config)
    hand_cache.clear()
    
    # Performance test data
    test_hands = [
        ['As', 'Ah'], ['Ks', 'Kh'], ['Qs', 'Qh'], ['Js', 'Jh'],
        ['Ts', 'Th'], ['9s', '9h'], ['8s', '8h'], ['7s', '7h'],
        ['As', 'Ks'], ['As', 'Qs'], ['As', 'Js'], ['As', 'Ts']
    ]
    
    num_scenarios = 50
    print(f"Storing {num_scenarios} scenarios to Redis...")
    
    # Store scenarios
    start_time = time.time()
    cache_keys = []
    
    for i in range(num_scenarios):
        hero_hand = test_hands[i % len(test_hands)]
        cache_key = create_cache_key(
            hero_hand=hero_hand,
            num_opponents=(i % 5) + 1,
            simulation_mode=f"perf_test_{i}"
        )
        cache_keys.append(cache_key)
        
        result = {
            'win_probability': 0.5 + (i * 0.01) % 0.4,
            'simulations_run': 10000,
            'scenario_id': i
        }
        
        hand_cache.store_result(cache_key, result)
    
    store_time = time.time() - start_time
    print(f"Stored {num_scenarios} scenarios in {store_time:.3f}s")
    print(f"   Average store time: {(store_time/num_scenarios)*1000:.1f}ms per scenario")
    
    # Test retrieval performance
    print("\nTesting retrieval performance...")
    
    # Create new cache instance to test Redis retrieval
    hand_cache_new = HandCache(config)
    
    start_time = time.time()
    retrieved_count = 0
    
    for cache_key in cache_keys:
        result = hand_cache_new.get_result(cache_key)
        if result:
            retrieved_count += 1
    
    retrieve_time = time.time() - start_time
    print(f"Retrieved {retrieved_count}/{num_scenarios} scenarios in {retrieve_time:.3f}s")
    print(f"   Average retrieval time: {(retrieve_time/num_scenarios)*1000:.1f}ms per scenario")
    
    # Performance targets
    avg_store_ms = (store_time/num_scenarios) * 1000
    avg_retrieve_ms = (retrieve_time/num_scenarios) * 1000
    
    performance_ok = True
    if avg_store_ms > 50:  # Target: <50ms per store
        print(f"Store performance slower than target (50ms): {avg_store_ms:.1f}ms")
        performance_ok = False
    
    if avg_retrieve_ms > 10:  # Target: <10ms per retrieval
        print(f"Retrieval performance slower than target (10ms): {avg_retrieve_ms:.1f}ms")
        performance_ok = False
    
    if performance_ok:
        print("Performance targets met")
    
    # Clean up
    hand_cache_new.clear()
    
    # Add assertions for performance
    assert retrieved_count == num_scenarios, f"Should retrieve all {num_scenarios} scenarios"
    assert avg_store_ms <= 50, f"Store performance should be <50ms, but was {avg_store_ms:.1f}ms"
    assert avg_retrieve_ms <= 10, f"Retrieval performance should be <10ms, but was {avg_retrieve_ms:.1f}ms"

def test_redis_failover():
    """Test Redis failover behavior when Redis is unavailable."""
    print("\nTesting Redis Failover Behavior")
    print("=" * 50)
    
    # Test with invalid Redis configuration
    config = CacheConfig(
        enable_persistence=True,
        redis_host="localhost",
        redis_port=9999,  # Non-existent port
        redis_db=0
    )
    
    print("Creating cache with invalid Redis config...")
    hand_cache = HandCache(config)
    
    # Should gracefully fall back to memory-only caching
    if hand_cache._redis_client is None:
        print("Cache gracefully fell back to memory-only mode")
    else:
        print("Cache should have fallen back to memory-only mode")
        assert False, "Cache should have fallen back to memory-only mode when Redis is unavailable"
    
    # Test normal caching operations still work
    cache_key = create_cache_key(['As', 'Ah'], 2, None, "failover_test")
    test_result = {
        'win_probability': 0.85,
        'simulations_run': 10000
    }
    
    stored = hand_cache.store_result(cache_key, test_result)
    retrieved = hand_cache.get_result(cache_key)
    
    if stored and retrieved and retrieved['win_probability'] == 0.85:
        print("Memory cache still works during Redis failover")
    else:
        print("Memory cache failed during Redis failover")
    
    assert stored, "Should be able to store in memory cache during Redis failover"
    assert retrieved is not None, "Should be able to retrieve from memory cache during Redis failover"
    assert retrieved['win_probability'] == 0.85, "Retrieved value should match stored value"

def main():
    """Run Redis integration tests."""
    print("Poker Knight Redis Integration Test")
    print("=" * 60)
    
    try:
        # Check Redis availability
        if not check_redis_availability():
            print("\nRedis Setup Instructions:")
            print("   1. Install Redis Server:")
            print("      - Windows: Download from https://github.com/microsoftarchive/redis/releases")
            print("      - Ubuntu: sudo apt-get install redis-server") 
            print("      - macOS: brew install redis")
            print("   2. Start Redis: redis-server")
            print("   3. Install Python Redis: pip install redis>=4.0.0")
            print("   4. Test connection: redis-cli ping")
            return 1
        
        results = []
        
        # Run tests
        print("\nRunning Redis integration tests...")
        results.append(test_redis_cache_persistence())
        results.append(test_redis_performance())
        results.append(test_redis_failover())
        
        # Summary
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        
        test_names = [
            "Redis Cache Persistence",
            "Redis Performance",
            "Redis Failover"
        ]
        
        for i, (test_name, result) in enumerate(zip(test_names, results)):
            status = "PASS" if result else "FAIL"
            print(f"{status:8} {test_name}")
        
        if all(results):
            print("\nAll Redis integration tests passed!")
            print("\nNext steps:")
            print("   - Enable Redis persistence in config.json")
            print("   - Run production performance benchmarks")
            print("   - Set up Redis clustering for high availability")
            print("   - Configure Redis memory management and expiration")
            return 0
        else:
            print("\nSome Redis tests failed. Check Redis configuration.")
            return 1
            
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 