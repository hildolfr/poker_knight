#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
♞ Poker Knight Fallback Demo

Shows Redis -> SQLite fallback behavior.
"""

import time
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight.storage import CacheConfig, HandCache, create_cache_key


def test_fallback():
    """Test fallback from Redis to SQLite."""
    print("🔄 Poker Knight Cache Fallback Demo")
    print("=" * 50)
    
    # Configuration that tries Redis first, falls back to SQLite
    config = CacheConfig(
        max_memory_mb=64,
        hand_cache_size=100,
        enable_persistence=True,
        redis_host="localhost",
        redis_port=6379,
        sqlite_path="fallback_demo.db"
    )
    
    cache = HandCache(config)
    stats = cache.get_persistence_stats()
    
    print(f"Redis available (library): {stats['redis_available']}")
    print(f"Redis connected: {stats['redis_connected']}")
    print(f"SQLite available: {stats['sqlite_available']}")
    print(f"Active persistence: {stats['persistence_type']}")
    
    if stats['persistence_type'] == 'redis':
        print("\n[PASS] Using Redis for persistence")
    elif stats['persistence_type'] == 'sqlite':
        print("\n🗄️  Using SQLite fallback (Redis unavailable)")
    else:
        print("\n🧠 Using memory-only (no persistence)")
    
    # Test cache functionality regardless of backend
    print(f"\n📝 Testing cache with {stats['persistence_type']} backend...")
    
    scenarios = [
        {'hand': ['AS', 'AH'], 'opponents': 2, 'desc': 'AA vs 2'},
        {'hand': ['KS', 'KH'], 'opponents': 1, 'desc': 'KK vs 1'},
    ]
    
    # Store data
    cache_keys = []
    for scenario in scenarios:
        cache_key = create_cache_key(
            hero_hand=scenario['hand'],
            num_opponents=scenario['opponents'],
            simulation_mode="fallback_demo"
        )
        cache_keys.append(cache_key)
        
        result = {
            'win_probability': 0.85,
            'scenario': scenario['desc'],
            'backend': stats['persistence_type']
        }
        
        cache.store_result(cache_key, result)
        print(f"   Stored: {scenario['desc']}")
    
    # Retrieve data
    print(f"\n📖 Retrieving from {stats['persistence_type']} cache...")
    for i, cache_key in enumerate(cache_keys):
        result = cache.get_result(cache_key)
        if result:
            print(f"   Retrieved: {scenarios[i]['desc']} - backend: {result['backend']}")
        else:
            print(f"   [FAIL] Failed to retrieve: {scenarios[i]['desc']}")
    
    # Test persistence across instances
    print(f"\n🔄 Testing persistence across cache instances...")
    cache_new = HandCache(config)
    
    for i, cache_key in enumerate(cache_keys):
        result = cache_new.get_result(cache_key)
        if result:
            print(f"   Persisted: {scenarios[i]['desc']} - backend: {result['backend']}")
        else:
            print(f"   [FAIL] Not persisted: {scenarios[i]['desc']}")
    
    # Get statistics
    cache_stats = cache.get_stats()
    print(f"\n[STATS] Cache Statistics:")
    print(f"   Persistence type: {cache_stats.persistence_type}")
    print(f"   Total requests: {cache_stats.total_requests}")
    print(f"   Cache hits: {cache_stats.cache_hits}")
    print(f"   Hit rate: {cache_stats.hit_rate:.1%}")
    print(f"   Persistence saves: {cache_stats.persistence_saves}")
    
    # Clean up
    cache.clear()
    if os.path.exists(config.sqlite_path):
        try:
            os.remove(config.sqlite_path)
            print(f"\n🧹 Cleaned up: {config.sqlite_path}")
        except:
            print(f"\n🧹 Note: {config.sqlite_path} will be cleaned up later")
    
    print(f"\n[PASS] Fallback demo completed!")
    
    if stats['persistence_type'] == 'sqlite':
        print(f"\n[IDEA] To test Redis, start the container:")
        print(f"   docker start poker-redis")
        print(f"   Then run this test again!")


if __name__ == "__main__":
    test_fallback() 