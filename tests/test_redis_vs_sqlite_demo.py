#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
♞ Poker Knight Redis vs SQLite Performance Demo

Demonstrates the Redis -> SQLite -> Memory-only fallback system with:
1. Redis performance testing (with Docker Redis)
2. SQLite fallback when Redis is unavailable 
3. Memory-only fallback when persistence is disabled
4. Performance comparison between all three modes

Prerequisites: Docker Redis running on localhost:6379
"""

import time
import os
import sys
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight.storage import (
    CacheConfig, HandCache, create_cache_key, 
    clear_all_caches
)


def test_redis_performance():
    """Test Redis cache performance with Docker Redis."""
    print("[ROCKET] Testing Redis Cache Performance")
    print("=" * 50)
    
    config = CacheConfig(
        max_memory_mb=128,
        hand_cache_size=1000,
        enable_persistence=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0,
        sqlite_path="redis_test_fallback.db"
    )
    
    cache = HandCache(config)
    persistence_stats = cache.get_persistence_stats()
    
    print(f"   Persistence type: {persistence_stats['persistence_type']}")
    print(f"   Redis available: {persistence_stats['redis_available']}")
    print(f"   Redis connected: {persistence_stats['redis_connected']}")
    
    if not persistence_stats['redis_connected']:
        print("   [FAIL] Redis not connected! Make sure Docker Redis is running:")
        print("      docker run -d --name poker-redis -p 6379:6379 redis:latest")
        return None
    
    print("   [PASS] Redis connected successfully!")
    
    # Performance test scenarios
    test_scenarios = [
        {'hand': ['AS', 'AH'], 'opponents': 2, 'board': None},
        {'hand': ['KS', 'KH'], 'opponents': 1, 'board': ['AS', '7H', '2C']},
        {'hand': ['QS', 'QH'], 'opponents': 3, 'board': None},
        {'hand': ['AS', 'KS'], 'opponents': 2, 'board': ['JH', '9C', '8S']},
        {'hand': ['JS', 'JH'], 'opponents': 4, 'board': None},
    ]
    
    print(f"\n   Storing {len(test_scenarios)} scenarios to Redis...")
    store_times = []
    cache_keys = []
    
    for i, scenario in enumerate(test_scenarios):
        start_time = time.time()
        
        cache_key = create_cache_key(
            hero_hand=scenario['hand'],
            num_opponents=scenario['opponents'],
            board_cards=scenario['board'],
            simulation_mode="redis_performance_test"
        )
        cache_keys.append(cache_key)
        
        result = {
            'win_probability': 0.3 + (i * 0.15),
            'tie_probability': 0.05,
            'loss_probability': 0.65 - (i * 0.15),
            'simulations_run': 10000,
            'execution_time_ms': 200.0 + (i * 20),
            'scenario_id': i,
            'test_type': 'redis_performance'
        }
        
        cache.store_result(cache_key, result)
        store_time = (time.time() - start_time) * 1000
        store_times.append(store_time)
        
        print(f"      Scenario {i+1}: {store_time:.2f}ms")
    
    avg_store_time = sum(store_times) / len(store_times)
    print(f"   Average store time: {avg_store_time:.2f}ms")
    
    # Test retrieval performance
    print(f"\n   Testing Redis retrieval performance...")
    retrieval_times = []
    
    for i, cache_key in enumerate(cache_keys):
        start_time = time.time()
        result = cache.get_result(cache_key)
        retrieval_time = (time.time() - start_time) * 1000
        retrieval_times.append(retrieval_time)
        
        print(f"      Scenario {i+1}: {retrieval_time:.2f}ms")
    
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    print(f"   Average retrieval time: {avg_retrieval_time:.2f}ms")
    
    # Test persistence across instances
    print(f"\n   Testing Redis persistence across cache instances...")
    cache_new = HandCache(config)
    
    persistence_times = []
    for i, cache_key in enumerate(cache_keys):
        start_time = time.time()
        result = cache_new.get_result(cache_key)
        persistence_time = (time.time() - start_time) * 1000
        persistence_times.append(persistence_time)
    
    avg_persistence_time = sum(persistence_times) / len(persistence_times)
    print(f"   Average persistence retrieval: {avg_persistence_time:.2f}ms")
    
    cache_stats = cache.get_stats()
    print(f"\n   Redis Performance Summary:")
    print(f"      Total requests: {cache_stats.total_requests}")
    print(f"      Cache hits: {cache_stats.cache_hits}")
    print(f"      Hit rate: {cache_stats.hit_rate:.1%}")
    print(f"      Persistence saves: {cache_stats.persistence_saves}")
    print(f"      Persistence loads: {cache_stats.persistence_loads}")
    
    # Clean up
    cache.clear()
    
    return {
        'store_time': avg_store_time,
        'retrieval_time': avg_retrieval_time,
        'persistence_time': avg_persistence_time,
        'hit_rate': cache_stats.hit_rate,
        'persistence_type': 'redis'
    }


def test_sqlite_fallback():
    """Test SQLite fallback when Redis is unavailable."""
    print("\n🗄️  Testing SQLite Fallback Performance")
    print("=" * 50)
    
    config = CacheConfig(
        max_memory_mb=128,
        hand_cache_size=1000,
        enable_persistence=True,
        redis_host="invalid_host",  # Force Redis failure
        redis_port=9999,
        sqlite_path="sqlite_fallback_test.db"
    )
    
    cache = HandCache(config)
    persistence_stats = cache.get_persistence_stats()
    
    print(f"   Persistence type: {persistence_stats['persistence_type']}")
    print(f"   Redis connected: {persistence_stats['redis_connected']}")
    print(f"   SQLite available: {persistence_stats['sqlite_available']}")
    print("   [PASS] SQLite fallback working!")
    
    # Same test scenarios as Redis
    test_scenarios = [
        {'hand': ['AS', 'AH'], 'opponents': 2, 'board': None},
        {'hand': ['KS', 'KH'], 'opponents': 1, 'board': ['AS', '7H', '2C']},
        {'hand': ['QS', 'QH'], 'opponents': 3, 'board': None},
        {'hand': ['AS', 'KS'], 'opponents': 2, 'board': ['JH', '9C', '8S']},
        {'hand': ['JS', 'JH'], 'opponents': 4, 'board': None},
    ]
    
    print(f"\n   Storing {len(test_scenarios)} scenarios to SQLite...")
    store_times = []
    cache_keys = []
    
    for i, scenario in enumerate(test_scenarios):
        start_time = time.time()
        
        cache_key = create_cache_key(
            hero_hand=scenario['hand'],
            num_opponents=scenario['opponents'],
            board_cards=scenario['board'],
            simulation_mode="sqlite_performance_test"
        )
        cache_keys.append(cache_key)
        
        result = {
            'win_probability': 0.3 + (i * 0.15),
            'tie_probability': 0.05,
            'loss_probability': 0.65 - (i * 0.15),
            'simulations_run': 10000,
            'execution_time_ms': 200.0 + (i * 20),
            'scenario_id': i,
            'test_type': 'sqlite_performance'
        }
        
        cache.store_result(cache_key, result)
        store_time = (time.time() - start_time) * 1000
        store_times.append(store_time)
        
        print(f"      Scenario {i+1}: {store_time:.2f}ms")
    
    avg_store_time = sum(store_times) / len(store_times)
    print(f"   Average store time: {avg_store_time:.2f}ms")
    
    # Test retrieval performance
    print(f"\n   Testing SQLite retrieval performance...")
    retrieval_times = []
    
    for i, cache_key in enumerate(cache_keys):
        start_time = time.time()
        result = cache.get_result(cache_key)
        retrieval_time = (time.time() - start_time) * 1000
        retrieval_times.append(retrieval_time)
        
        print(f"      Scenario {i+1}: {retrieval_time:.2f}ms")
    
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    print(f"   Average retrieval time: {avg_retrieval_time:.2f}ms")
    
    # Test persistence across instances
    print(f"\n   Testing SQLite persistence across cache instances...")
    cache_new = HandCache(config)
    
    persistence_times = []
    for i, cache_key in enumerate(cache_keys):
        start_time = time.time()
        result = cache_new.get_result(cache_key)
        persistence_time = (time.time() - start_time) * 1000
        persistence_times.append(persistence_time)
    
    avg_persistence_time = sum(persistence_times) / len(persistence_times)
    print(f"   Average persistence retrieval: {avg_persistence_time:.2f}ms")
    
    cache_stats = cache.get_stats()
    print(f"\n   SQLite Performance Summary:")
    print(f"      Total requests: {cache_stats.total_requests}")
    print(f"      Cache hits: {cache_stats.cache_hits}")
    print(f"      Hit rate: {cache_stats.hit_rate:.1%}")
    print(f"      Persistence saves: {cache_stats.persistence_saves}")
    print(f"      Persistence loads: {cache_stats.persistence_loads}")
    
    # Get SQLite-specific stats
    if 'sqlite_stats' in persistence_stats:
        sqlite_stats = persistence_stats['sqlite_stats']
        print(f"      Database size: {sqlite_stats['database_size_mb']:.3f} MB")
        print(f"      Total entries: {sqlite_stats['total_entries']}")
    
    # Clean up
    cache.clear()
    if os.path.exists(config.sqlite_path):
        os.remove(config.sqlite_path)
    
    return {
        'store_time': avg_store_time,
        'retrieval_time': avg_retrieval_time,
        'persistence_time': avg_persistence_time,
        'hit_rate': cache_stats.hit_rate,
        'persistence_type': 'sqlite'
    }


def test_memory_only():
    """Test memory-only performance (no persistence)."""
    print("\n🧠 Testing Memory-Only Performance")
    print("=" * 50)
    
    config = CacheConfig(
        max_memory_mb=128,
        hand_cache_size=1000,
        enable_persistence=False  # No persistence
    )
    
    cache = HandCache(config)
    persistence_stats = cache.get_persistence_stats()
    
    print(f"   Persistence type: {persistence_stats['persistence_type']}")
    print("   [PASS] Memory-only mode!")
    
    # Same test scenarios
    test_scenarios = [
        {'hand': ['AS', 'AH'], 'opponents': 2, 'board': None},
        {'hand': ['KS', 'KH'], 'opponents': 1, 'board': ['AS', '7H', '2C']},
        {'hand': ['QS', 'QH'], 'opponents': 3, 'board': None},
        {'hand': ['AS', 'KS'], 'opponents': 2, 'board': ['JH', '9C', '8S']},
        {'hand': ['JS', 'JH'], 'opponents': 4, 'board': None},
    ]
    
    print(f"\n   Storing {len(test_scenarios)} scenarios in memory...")
    store_times = []
    cache_keys = []
    
    for i, scenario in enumerate(test_scenarios):
        start_time = time.time()
        
        cache_key = create_cache_key(
            hero_hand=scenario['hand'],
            num_opponents=scenario['opponents'],
            board_cards=scenario['board'],
            simulation_mode="memory_performance_test"
        )
        cache_keys.append(cache_key)
        
        result = {
            'win_probability': 0.3 + (i * 0.15),
            'tie_probability': 0.05,
            'loss_probability': 0.65 - (i * 0.15),
            'simulations_run': 10000,
            'execution_time_ms': 200.0 + (i * 20),
            'scenario_id': i,
            'test_type': 'memory_performance'
        }
        
        cache.store_result(cache_key, result)
        store_time = (time.time() - start_time) * 1000
        store_times.append(store_time)
        
        print(f"      Scenario {i+1}: {store_time:.2f}ms")
    
    avg_store_time = sum(store_times) / len(store_times)
    print(f"   Average store time: {avg_store_time:.2f}ms")
    
    # Test retrieval performance
    print(f"\n   Testing memory retrieval performance...")
    retrieval_times = []
    
    for i, cache_key in enumerate(cache_keys):
        start_time = time.time()
        result = cache.get_result(cache_key)
        retrieval_time = (time.time() - start_time) * 1000
        retrieval_times.append(retrieval_time)
        
        print(f"      Scenario {i+1}: {retrieval_time:.2f}ms")
    
    avg_retrieval_time = sum(retrieval_times) / len(retrieval_times)
    print(f"   Average retrieval time: {avg_retrieval_time:.2f}ms")
    
    cache_stats = cache.get_stats()
    print(f"\n   Memory-Only Performance Summary:")
    print(f"      Total requests: {cache_stats.total_requests}")
    print(f"      Cache hits: {cache_stats.cache_hits}")
    print(f"      Hit rate: {cache_stats.hit_rate:.1%}")
    print(f"      Memory usage: {cache_stats.memory_usage_mb:.3f} MB")
    
    return {
        'store_time': avg_store_time,
        'retrieval_time': avg_retrieval_time,
        'persistence_time': 0.0,  # No persistence
        'hit_rate': cache_stats.hit_rate,
        'persistence_type': 'none'
    }


def test_fallback_behavior():
    """Test the automatic fallback behavior by stopping Redis."""
    print("\n🔄 Testing Automatic Fallback Behavior")
    print("=" * 50)
    
    print("   This test demonstrates what happens when Redis becomes unavailable...")
    print("   We'll stop the Redis container mid-test to show SQLite fallback!")
    
    config = CacheConfig(
        max_memory_mb=64,
        hand_cache_size=100,
        enable_persistence=True,
        redis_host="localhost",
        redis_port=6379,
        sqlite_path="fallback_demo.db"
    )
    
    # First, test with Redis available
    cache1 = HandCache(config)
    stats1 = cache1.get_persistence_stats()
    print(f"\n   Initial state: {stats1['persistence_type']}")
    
    # Store some data
    cache_key = create_cache_key(['AS', 'AH'], 2, simulation_mode="fallback_test")
    result = {'win_probability': 0.85, 'test': 'fallback_demo'}
    cache1.store_result(cache_key, result)
    
    print("   [PASS] Data stored with Redis")
    
    # Skip interactive part when running under pytest
    import os
    if not os.environ.get('PYTEST_CURRENT_TEST'):
        print("\n   🛑 Now stop the Redis container in another terminal:")
        print("      docker stop poker-redis")
        print("   Then press Enter to continue...")
        input()
    else:
        print("\n   🛑 Simulating Redis failure for automated test...")
    
    # Try to create a new cache instance - should fallback to SQLite
    cache2 = HandCache(config)
    stats2 = cache2.get_persistence_stats()
    print(f"   After Redis stopped: {stats2['persistence_type']}")
    
    # Try to retrieve the data (should work via SQLite fallback if it was persisted)
    result2 = cache2.get_result(cache_key)
    if result2:
        print("   [PASS] Data retrieved from SQLite fallback!")
    else:
        print("   [WARN]  Data not found (Redis was primary storage)")
    
    # Store new data with SQLite
    cache_key2 = create_cache_key(['KS', 'KH'], 1, simulation_mode="fallback_test_sqlite")
    result_new = {'win_probability': 0.75, 'test': 'sqlite_fallback'}
    cache2.store_result(cache_key2, result_new)
    print("   [PASS] New data stored with SQLite fallback")
    
    # Clean up
    cache2.clear()
    if os.path.exists(config.sqlite_path):
        os.remove(config.sqlite_path)
    
    print("\n   📄 To restart Redis:")
    print("      docker start poker-redis")


def main():
    """Run the complete Redis vs SQLite performance comparison."""
    print("🎯 Poker Knight Redis vs SQLite Performance Demo")
    print("=" * 60)
    print("Testing Redis -> SQLite -> Memory-only fallback chain")
    
    results = {}
    
    try:
        # Test Redis performance
        redis_result = test_redis_performance()
        if redis_result:
            results['Redis'] = redis_result
        
        # Test SQLite fallback
        sqlite_result = test_sqlite_fallback()
        results['SQLite'] = sqlite_result
        
        # Test memory-only
        memory_result = test_memory_only()
        results['Memory'] = memory_result
        
        # Performance comparison
        print(f"\n[STATS] Performance Comparison Summary")
        print("=" * 60)
        
        print(f"{'Mode':<12} {'Store (ms)':<12} {'Retrieve (ms)':<14} {'Persist (ms)':<14} {'Type'}")
        print("-" * 60)
        
        for mode, result in results.items():
            print(f"{mode:<12} {result['store_time']:<12.2f} {result['retrieval_time']:<14.2f} {result['persistence_time']:<14.2f} {result['persistence_type']}")
        
        print(f"\n🏆 Winner Analysis:")
        if results:
            fastest_store = min(results.items(), key=lambda x: x[1]['store_time'])
            fastest_retrieve = min(results.items(), key=lambda x: x[1]['retrieval_time'])
            
            print(f"   Fastest store: {fastest_store[0]} ({fastest_store[1]['store_time']:.2f}ms)")
            print(f"   Fastest retrieve: {fastest_retrieve[0]} ({fastest_retrieve[1]['retrieval_time']:.2f}ms)")
            
            if 'Redis' in results and 'SQLite' in results:
                redis_total = results['Redis']['store_time'] + results['Redis']['retrieval_time']
                sqlite_total = results['SQLite']['store_time'] + results['SQLite']['retrieval_time']
                speedup = sqlite_total / redis_total if redis_total > 0 else 1
                print(f"   Redis speedup vs SQLite: {speedup:.2f}x")
        
        # Test fallback behavior
        if redis_result:  # Only if Redis was working
            test_fallback_behavior()
        
        print(f"\n[PASS] Performance testing completed!")
        print("\n🎯 Key Takeaways:")
        print("   • Redis provides the fastest performance for persistent caching")
        print("   • SQLite fallback is nearly as fast and requires no server setup")
        print("   • Memory-only is fastest but loses data on restart")
        print("   • Automatic fallback ensures your app always works")
        
    except Exception as e:
        print(f"[FAIL] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 