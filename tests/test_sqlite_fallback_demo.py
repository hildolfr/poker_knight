#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
♞ Poker Knight SQLite Fallback Cache Demo

Demonstrates the automatic fallback system:
1. Redis (if available)
2. SQLite (if Redis unavailable but persistence enabled)
3. Memory-only (if neither available)

Run this script to see the cache fallback behavior in action.
"""

import time
import os
import sys
import tempfile
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight.storage import (
    CacheConfig, HandCache, create_cache_key, 
    clear_all_caches
)


def test_cache_fallback_system():
    """Test the complete cache fallback system."""
    print("🎯 Poker Knight Cache Fallback System Demo")
    print("=" * 60)
    
    # Test scenarios
    test_scenarios = [
        {
            'hero_hand': ['AS', 'AH'],
            'num_opponents': 2,
            'board_cards': None,
            'description': 'Pocket Aces preflop vs 2 opponents'
        },
        {
            'hero_hand': ['KS', 'KH'],
            'num_opponents': 1,
            'board_cards': ['AS', '7H', '2C'],
            'description': 'Pocket Kings on A72 board'
        },
        {
            'hero_hand': ['QS', 'QH'],
            'num_opponents': 3,
            'board_cards': None,
            'description': 'Pocket Queens preflop vs 3 opponents'
        },
        {
            'hero_hand': ['AS', 'KS'],
            'num_opponents': 2,
            'board_cards': ['JH', '9C', '8S'],
            'description': 'AK suited on J98 board'
        }
    ]
    
    # Test different configurations
    configs = [
        {
            'name': 'Redis Enabled (may fallback to SQLite)',
            'config': CacheConfig(
                max_memory_mb=64,
                hand_cache_size=100,
                enable_persistence=True,
                redis_host="localhost",
                redis_port=6379,
                sqlite_path="demo_cache_redis_attempt.db"
            )
        },
        {
            'name': 'SQLite Only (Redis disabled)',
            'config': CacheConfig(
                max_memory_mb=64,
                hand_cache_size=100,
                enable_persistence=True,
                redis_host="invalid_host",  # Force Redis failure
                redis_port=9999,
                sqlite_path="demo_cache_sqlite_only.db"
            )
        },
        {
            'name': 'Memory Only (Persistence disabled)',
            'config': CacheConfig(
                max_memory_mb=64,
                hand_cache_size=100,
                enable_persistence=False
            )
        }
    ]
    
    results = {}
    
    for config_info in configs:
        config_name = config_info['name']
        config = config_info['config']
        
        print(f"\n📋 Testing: {config_name}")
        print("-" * 50)
        
        # Create cache instance
        cache = HandCache(config)
        
        # Check what persistence backend is being used
        persistence_stats = cache.get_persistence_stats()
        print(f"   Persistence type: {persistence_stats['persistence_type']}")
        print(f"   Redis available: {persistence_stats['redis_available']}")
        print(f"   Redis connected: {persistence_stats['redis_connected']}")
        print(f"   SQLite available: {persistence_stats['sqlite_available']}")
        
        # Store test scenarios
        print("\n   Storing test scenarios...")
        cache_keys = []
        store_times = []
        
        for i, scenario in enumerate(test_scenarios):
            start_time = time.time()
            
            cache_key = create_cache_key(
                hero_hand=scenario['hero_hand'],
                num_opponents=scenario['num_opponents'],
                board_cards=scenario['board_cards'],
                simulation_mode=f"demo_{config_name.lower().replace(' ', '_')}"
            )
            cache_keys.append(cache_key)
            
            # Simulate poker analysis result
            result = {
                'win_probability': 0.4 + (i * 0.15),
                'tie_probability': 0.05,
                'loss_probability': 0.55 - (i * 0.15),
                'simulations_run': 10000,
                'execution_time_ms': 120.0 + (i * 10),
                'scenario_description': scenario['description'],
                'cache_test': True,
                'config_name': config_name
            }
            
            success = cache.store_result(cache_key, result)
            store_time = (time.time() - start_time) * 1000
            store_times.append(store_time)
            
            print(f"      Scenario {i+1}: {scenario['description'][:30]}... stored in {store_time:.2f}ms")
        
        # Test retrieval performance
        print("\n   Testing retrieval performance...")
        retrieval_times = []
        successful_retrievals = 0
        
        for i, cache_key in enumerate(cache_keys):
            start_time = time.time()
            result = cache.get_result(cache_key)
            retrieval_time = (time.time() - start_time) * 1000
            retrieval_times.append(retrieval_time)
            
            if result and result.get('cache_test'):
                successful_retrievals += 1
                print(f"      Scenario {i+1}: Retrieved in {retrieval_time:.2f}ms")
            else:
                print(f"      Scenario {i+1}: [FAIL] Not found")
        
        # Get cache statistics
        cache_stats = cache.get_stats()
        
        # Store results
        results[config_name] = {
            'persistence_type': persistence_stats['persistence_type'],
            'successful_stores': len(store_times),
            'successful_retrievals': successful_retrievals,
            'avg_store_time_ms': sum(store_times) / len(store_times) if store_times else 0,
            'avg_retrieval_time_ms': sum(retrieval_times) / len(retrieval_times) if retrieval_times else 0,
            'hit_rate': cache_stats.hit_rate,
            'total_requests': cache_stats.total_requests,
            'cache_hits': cache_stats.cache_hits,
            'persistence_saves': cache_stats.persistence_saves,
            'persistence_loads': cache_stats.persistence_loads
        }
        
        print(f"\n   Results:")
        print(f"      Hit rate: {cache_stats.hit_rate:.1%}")
        print(f"      Avg store time: {results[config_name]['avg_store_time_ms']:.2f}ms")
        print(f"      Avg retrieval time: {results[config_name]['avg_retrieval_time_ms']:.2f}ms")
        print(f"      Persistence saves: {cache_stats.persistence_saves}")
        print(f"      Persistence loads: {cache_stats.persistence_loads}")
        
        # Test persistence across instances
        if persistence_stats['persistence_type'] in ['redis', 'sqlite']:
            print("\n   Testing persistence across cache instances...")
            
            # Create new cache instance
            cache_new = HandCache(config)
            persistence_retrieval_times = []
            persistence_successful = 0
            
            for i, cache_key in enumerate(cache_keys):
                start_time = time.time()
                result = cache_new.get_result(cache_key)
                retrieval_time = (time.time() - start_time) * 1000
                persistence_retrieval_times.append(retrieval_time)
                
                if result and result.get('cache_test'):
                    persistence_successful += 1
            
            avg_persistence_time = sum(persistence_retrieval_times) / len(persistence_retrieval_times)
            print(f"      Persistence test: {persistence_successful}/{len(cache_keys)} scenarios retrieved")
            print(f"      Avg persistence retrieval: {avg_persistence_time:.2f}ms")
        
        # Clean up test data
        cache.clear()
        
        # Close SQLite connections before file cleanup
        if cache._sqlite_cache:
            cache._sqlite_cache.close()
        
        if config.sqlite_path and os.path.exists(config.sqlite_path):
            try:
                os.remove(config.sqlite_path)
                print(f"   Cleaned up demo file: {config.sqlite_path}")
            except OSError:
                print(f"   Note: Demo file {config.sqlite_path} will be cleaned up when process exits")
    
    # Summary comparison
    print(f"\n[STATS] Performance Comparison Summary")
    print("=" * 60)
    
    for config_name, result in results.items():
        print(f"\n{config_name}:")
        print(f"   Persistence Type: {result['persistence_type']}")
        print(f"   Store Performance: {result['avg_store_time_ms']:.2f}ms average")
        print(f"   Retrieval Performance: {result['avg_retrieval_time_ms']:.2f}ms average")
        print(f"   Success Rate: {result['successful_retrievals']}/{result['successful_stores']} scenarios")
        print(f"   Cache Hit Rate: {result['hit_rate']:.1%}")
    
    # Add assertions to verify the tests
    for config_name, result in results.items():
        assert result['successful_stores'] > 0, f"{config_name}: Should have successful stores"
        assert result['successful_retrievals'] > 0, f"{config_name}: Should have successful retrievals"
        assert result['avg_store_time_ms'] > 0, f"{config_name}: Store time should be positive"
        assert result['avg_retrieval_time_ms'] > 0, f"{config_name}: Retrieval time should be positive"
        assert result['hit_rate'] > 0, f"{config_name}: Hit rate should be positive"
    
    # Only return dict when not running under pytest
    if not os.environ.get('PYTEST_CURRENT_TEST'):
        return results


def test_sqlite_specific_features():
    """Test SQLite-specific features like statistics and cleanup."""
    print(f"\n🗄️  SQLite Cache Specific Features Demo")
    print("=" * 60)
    
    # Create SQLite-only configuration
    config = CacheConfig(
        enable_persistence=True,
        redis_host="invalid_host",  # Force SQLite fallback
        sqlite_path="sqlite_features_demo.db"
    )
    
    cache = HandCache(config)
    persistence_stats = cache.get_persistence_stats()
    
    if persistence_stats['sqlite_available']:
        print("[PASS] SQLite cache initialized successfully")
        
        # Store some test data
        print("\nStoring test data...")
        for i in range(10):
            cache_key = f"test_key_{i}"
            result = {
                'test_data': f"Test result {i}",
                'value': i * 10,
                'timestamp': time.time()
            }
            cache.store_result(cache_key, result)
        
        # Get SQLite statistics
        if 'sqlite_stats' in persistence_stats:
            sqlite_stats = persistence_stats['sqlite_stats']
            print(f"\nSQLite Statistics:")
            print(f"   Database file: {sqlite_stats['database_path']}")
            print(f"   Total entries: {sqlite_stats['total_entries']}")
            print(f"   Database size: {sqlite_stats['database_size_mb']:.3f} MB")
            print(f"   Average access count: {sqlite_stats['avg_access_count']:.1f}")
        
        # Test cleanup
        print("\nTesting cleanup functionality...")
        if cache._sqlite_cache:
            deleted_count = cache._sqlite_cache.cleanup_expired(max_age_hours=0)  # Clean everything
            print(f"   Cleaned up {deleted_count} expired entries")
        
        # Clean up demo file
        cache.clear()
        
        # Close SQLite connections before file cleanup
        if cache._sqlite_cache:
            cache._sqlite_cache.close()
        
        if os.path.exists(config.sqlite_path):
            os.remove(config.sqlite_path)
            print(f"   Cleaned up demo file: {config.sqlite_path}")
    else:
        print("[FAIL] SQLite cache not available")


def main():
    """Run the complete demo."""
    print("Starting Poker Knight Cache Fallback Demo...")
    
    try:
        # Test fallback system
        test_cache_fallback_system()
        
        # Test SQLite features
        test_sqlite_specific_features()
        
        print(f"\n[PASS] Demo completed successfully!")
        print("\nKey takeaways:")
        print("   • Cache automatically falls back: Redis -> SQLite -> Memory-only")
        print("   • SQLite provides lightweight persistence without server setup")
        print("   • Performance remains excellent across all modes")
        print("   • Your application works regardless of Redis availability")
        
    except Exception as e:
        print(f"[FAIL] Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 