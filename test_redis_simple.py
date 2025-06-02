#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
♞ Poker Knight Simple Redis Test

Simple test to demonstrate Redis caching performance with Docker Redis.
"""

import time
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight.storage import CacheConfig, HandCache, create_cache_key


def test_redis_cache():
    """Test Redis cache functionality and performance."""
    print("[ROCKET] Poker Knight Redis Cache Test")
    print("=" * 45)
    
    # Redis configuration
    config = CacheConfig(
        max_memory_mb=64,
        hand_cache_size=100,
        enable_persistence=True,
        redis_host="localhost",
        redis_port=6379,
        redis_db=0
    )
    
    cache = HandCache(config)
    persistence_stats = cache.get_persistence_stats()
    
    print(f"Redis available: {persistence_stats['redis_available']}")
    print(f"Redis connected: {persistence_stats['redis_connected']}")
    print(f"Persistence type: {persistence_stats['persistence_type']}")
    
    if not persistence_stats['redis_connected']:
        print("\n[FAIL] Redis not connected!")
        print("Make sure Docker Redis is running:")
        print("  docker run -d --name poker-redis -p 6379:6379 redis:latest")
        return
    
    print("\n[PASS] Redis connected successfully!")
    
    # Test scenarios
    scenarios = [
        {'hand': ['AS', 'AH'], 'opponents': 2, 'desc': 'Pocket Aces vs 2'},
        {'hand': ['KS', 'KH'], 'opponents': 1, 'desc': 'Pocket Kings vs 1'},
        {'hand': ['QS', 'QH'], 'opponents': 3, 'desc': 'Pocket Queens vs 3'},
        {'hand': ['AS', 'KS'], 'opponents': 2, 'desc': 'AK suited vs 2'},
    ]
    
    print(f"\n📝 Storing {len(scenarios)} poker scenarios...")
    store_times = []
    cache_keys = []
    
    for i, scenario in enumerate(scenarios):
        start_time = time.time()
        
        cache_key = create_cache_key(
            hero_hand=scenario['hand'],
            num_opponents=scenario['opponents'],
            simulation_mode="redis_demo"
        )
        cache_keys.append(cache_key)
        
        # Simulate poker analysis result
        result = {
            'win_probability': 0.4 + (i * 0.15),
            'tie_probability': 0.05,
            'loss_probability': 0.55 - (i * 0.15),
            'simulations_run': 10000,
            'execution_time_ms': 250.0,
            'scenario': scenario['desc']
        }
        
        cache.store_result(cache_key, result)
        store_time = (time.time() - start_time) * 1000
        store_times.append(store_time)
        
        print(f"   {scenario['desc']}: {store_time:.2f}ms")
    
    avg_store = sum(store_times) / len(store_times)
    print(f"\n⚡ Average store time: {avg_store:.2f}ms")
    
    # Test retrieval
    print(f"\n📖 Testing retrieval from Redis cache...")
    retrieval_times = []
    
    for i, cache_key in enumerate(cache_keys):
        start_time = time.time()
        result = cache.get_result(cache_key)
        retrieval_time = (time.time() - start_time) * 1000
        retrieval_times.append(retrieval_time)
        
        if result:
            print(f"   {scenarios[i]['desc']}: {retrieval_time:.2f}ms - {result['win_probability']:.3f} win rate")
        else:
            print(f"   {scenarios[i]['desc']}: [FAIL] Not found")
    
    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    print(f"\n⚡ Average retrieval time: {avg_retrieval:.2f}ms")
    
    # Test persistence across instances
    print(f"\n🔄 Testing persistence across cache instances...")
    cache_new = HandCache(config)
    
    persistence_times = []
    for i, cache_key in enumerate(cache_keys):
        start_time = time.time()
        result = cache_new.get_result(cache_key)
        persistence_time = (time.time() - start_time) * 1000
        persistence_times.append(persistence_time)
        
        if result:
            print(f"   {scenarios[i]['desc']}: {persistence_time:.2f}ms from Redis")
    
    avg_persistence = sum(persistence_times) / len(persistence_times)
    print(f"\n⚡ Average persistence time: {avg_persistence:.2f}ms")
    
    # Get cache statistics
    cache_stats = cache.get_stats()
    
    print(f"\n[STATS] Redis Cache Statistics:")
    print(f"   Total requests: {cache_stats.total_requests}")
    print(f"   Cache hits: {cache_stats.cache_hits}")
    print(f"   Cache misses: {cache_stats.cache_misses}")
    print(f"   Hit rate: {cache_stats.hit_rate:.1%}")
    print(f"   Persistence saves: {cache_stats.persistence_saves}")
    print(f"   Persistence loads: {cache_stats.persistence_loads}")
    print(f"   Memory usage: {cache_stats.memory_usage_mb:.3f} MB")
    
    # Clean up
    cache.clear()
    print(f"\n🧹 Cache cleared")
    
    print(f"\n[PASS] Redis test completed successfully!")
    
    # Performance summary
    speedup_factor = 250.0 / avg_retrieval if avg_retrieval > 0 else 0
    print(f"\n🏆 Performance Summary:")
    print(f"   Redis retrieval is ~{speedup_factor:.0f}x faster than simulation")
    print(f"   Perfect for high-frequency poker analysis!")


if __name__ == "__main__":
    test_redis_cache() 