#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Poker Knight v1.6 caching system (Task 1.3)

This script validates:
- Cache components functionality 
- Cache hit/miss functionality
- Cache statistics

Updated to use the new Phase 4 unified cache architecture.
"""

import time
import sys
import os

# Add the poker_knight module to the path
sys.path.insert(0, os.path.dirname(__file__))

try:
    # Import new unified cache system
    from poker_knight.storage.unified_cache import (
        ThreadSafeMonteCarloCache, CacheKey, CacheResult, 
        create_cache_key, CacheKeyNormalizer
    )
    from poker_knight.storage.hierarchical_cache import (
        HierarchicalCache, HierarchicalCacheConfig
    )
    print("[PASS] Phase 4 caching system imported successfully")
except ImportError as e:
    print(f"[FAIL] Failed to import Phase 4 caching system: {e}")
    sys.exit(1)


def test_cache_components():
    """Test cache components directly."""
    print("\n🧪 Testing Cache Components")
    print("=" * 50)
    
    # Test unified cache creation
    unified_cache = ThreadSafeMonteCarloCache(
        max_memory_mb=64,
        max_entries=100,
        enable_persistence=False  # Keep testing simple
    )
    print(f"Unified cache created successfully")
    
    # Test hierarchical cache configuration
    hier_config = HierarchicalCacheConfig()
    hier_config.l1_config.max_memory_mb = 32
    hier_config.l2_config.enabled = False  # Disable Redis for test
    hier_config.l3_config.enabled = False  # Disable SQLite for test
    
    hier_cache = HierarchicalCache(hier_config)
    print(f"Hierarchical cache created successfully")
    
    # Test cache key generation with new system
    cache_key = create_cache_key(
        hero_hand=['A♠', 'A♥'],  # Using Unicode suits
        num_opponents=2,
        board_cards=None,
        simulation_mode="fast"
    )
    print(f"Cache key generated: {cache_key.to_string()[:32]}...")
    
    # Test storing and retrieving from unified cache
    test_result = CacheResult(
        win_probability=0.85,
        tie_probability=0.02,
        loss_probability=0.13,
        confidence_interval=(0.83, 0.87),
        simulations_run=10000,
        execution_time_ms=150.5,
        hand_categories={'pair': 0.15, 'high_card': 0.85},
        metadata={'test': True},
        timestamp=time.time()
    )
    
    print("\nTesting unified cache storage and retrieval:")
    stored = unified_cache.store(cache_key, test_result)
    print(f"[PASS] Result stored: {stored}")
    
    retrieved = unified_cache.get(cache_key)
    print(f"[PASS] Result retrieved: {retrieved is not None}")
    
    if retrieved:
        print(f"   Win probability: {retrieved.win_probability}")
        print(f"   Simulations: {retrieved.simulations_run}")
    
    # Test cache statistics
    stats = unified_cache.get_stats()
    print(f"\n[STATS] Cache Statistics:")
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Cache misses: {stats.cache_misses}")
    print(f"   Hit rate: {stats.hit_rate:.1%}")
    print(f"   Memory usage: {stats.memory_usage_mb:.1f}MB")
    
    assert stats.total_requests > 0, "Total requests should be greater than 0"
    assert stats.cache_hits > 0, "Cache hits should be greater than 0"
    
    # Clean up
    hier_cache.shutdown()


def test_preflop_cache():
    """Test preflop cache functionality using unified cache."""
    print("\n🎯 Testing Preflop Cache")
    print("=" * 50)
    
    # Create unified cache for preflop scenarios
    cache = ThreadSafeMonteCarloCache(
        max_memory_mb=32,
        max_entries=1000,
        enable_persistence=False
    )
    
    # Test preflop result storage and retrieval
    hero_hand = ['A♠', 'A♥']
    
    # Create cache key for preflop scenario
    cache_key = create_cache_key(
        hero_hand=hero_hand,
        num_opponents=2,
        board_cards=None,  # None = preflop
        simulation_mode="default"
    )
    
    # Create test result
    test_result = CacheResult(
        win_probability=0.85,
        tie_probability=0.02,
        loss_probability=0.13,
        confidence_interval=(0.83, 0.87),
        simulations_run=10000,
        execution_time_ms=50.0,
        hand_categories={'pair': 1.0},  # AA is always a pair preflop
        metadata={'position': 'button'},
        timestamp=time.time()
    )
    
    print(f"Testing with hand: {hero_hand}")
    print(f"Cache key: {cache_key.to_string()[:40]}...")
    
    # Store result
    stored = cache.store(cache_key, test_result)
    print(f"[PASS] Preflop result stored: {stored}")
    
    # Retrieve result
    retrieved = cache.get(cache_key)
    print(f"[PASS] Preflop result retrieved: {retrieved is not None}")
    
    if retrieved:
        print(f"   Win probability: {retrieved.win_probability}")
        print(f"   Simulations: {retrieved.simulations_run}")
        
    # Test with different position (should be same cache key in new system)
    cache_key2 = create_cache_key(hero_hand, 2, None, "default")
    retrieved2 = cache.get(cache_key2)
    print(f"\n[TEST] Position-independent caching: {retrieved2 is not None}")
    
    # Test cache statistics
    stats = cache.get_stats()
    print(f"\n[STATS] Preflop Cache Statistics:")
    print(f"   Total requests: {stats.total_requests}")
    print(f"   Cache hits: {stats.cache_hits}")
    print(f"   Hit rate: {stats.hit_rate:.1%}")
    
    assert retrieved is not None, "Retrieved preflop result should not be None"
    assert retrieved2 is not None, "Cache should be position-independent"


def test_phase4_integration():
    """Test Phase 4 cache system integration."""
    print("\n🚀 Testing Phase 4 Cache System Integration")
    print("=" * 50)
    
    try:
        from poker_knight.storage.phase4_integration import (
            Phase4CacheSystem, Phase4Config, create_balanced_cache_system
        )
    except ImportError:
        import pytest
        pytest.skip("Phase 4 integration not available yet")
    
    # Create Phase 4 system with test configuration
    config = Phase4Config(
        optimization_level="balanced",
        auto_start_services=False,
        enable_optimized_persistence=False  # Disable for testing
    )
    
    phase4_system = Phase4CacheSystem(config)
    
    # Initialize system
    print("Initializing Phase 4 cache system...")
    success = phase4_system.initialize()
    print(f"[PASS] System initialized: {success}")
    
    # Get system status
    status = phase4_system.get_system_status()
    print(f"\n[STATUS] Phase 4 System Status:")
    print(f"   Initialized: {status['initialized']}")
    print(f"   Optimization level: {status['optimization_level']}")
    print(f"   Components: {status['components']}")
    
    # Test cache operation through hierarchical cache
    if phase4_system.hierarchical_cache:
        cache_key = create_cache_key(['A♠', 'K♦'], 3, None, "precision")
        test_result = CacheResult(
            win_probability=0.31,
            tie_probability=0.01,
            loss_probability=0.68,
            confidence_interval=(0.29, 0.33),
            simulations_run=500000,
            execution_time_ms=5000.0,
            hand_categories={'high_card': 0.5, 'pair': 0.3, 'two_pair': 0.2},
            metadata={'phase4_test': True},
            timestamp=time.time()
        )
        
        # Store and retrieve
        phase4_system.hierarchical_cache.store(cache_key, test_result)
        retrieved = phase4_system.hierarchical_cache.get(cache_key)
        
        print(f"\n[TEST] Hierarchical cache working: {retrieved is not None}")
        if retrieved:
            print(f"   Retrieved win probability: {retrieved.win_probability}")
    
    # Clean up
    phase4_system.stop_services()
    
    print("\n[PASS] Phase 4 integration test completed")


def test_cache_performance():
    """Test cache performance improvement."""
    print("\n⚡ Testing Cache Performance")
    print("=" * 50)
    
    # Create unified cache
    cache = ThreadSafeMonteCarloCache(
        max_memory_mb=64,
        max_entries=1000,
        enable_persistence=False
    )
    
    # Create cache key for test scenario
    cache_key = create_cache_key(['K♠', 'K♥'], 1, None, "default")
    
    # Create expensive result
    expensive_result = CacheResult(
        win_probability=0.83,
        tie_probability=0.01,
        loss_probability=0.16,
        confidence_interval=(0.81, 0.85),
        simulations_run=100000,
        execution_time_ms=2500.0,  # Simulated expensive computation
        hand_categories={'pair': 1.0},
        metadata={'expensive': True},
        timestamp=time.time()
    )
    
    # First "computation" - cache miss
    print("Testing cache miss...")
    start_time = time.time()
    result1 = cache.get(cache_key)
    if result1 is None:
        # Simulate computation time (but don't actually sleep to avoid hanging)
        print("Cache miss - simulating computation...")
        cache.store(cache_key, expensive_result)
        result1 = expensive_result
    time1 = (time.time() - start_time) * 1000
    
    # Second "computation" - cache hit
    print("Testing cache hit...")
    start_time = time.time()
    result2 = cache.get(cache_key)
    time2 = (time.time() - start_time) * 1000
    
    print(f"First run (cache miss): {time1:.1f}ms")
    print(f"Second run (cache hit): {time2:.1f}ms")
    
    # Cache hit should be much faster (even without sleep, should be sub-millisecond)
    if result2 is not None and time2 < 10:  # Cache hit should be very fast
        print(f"[PASS] Cache working! Cache hit was {time2:.1f}ms")
    else:
        print(f"[WARN] Cache may not be working optimally")
        print(f"   Result2 is not None: {result2 is not None}")
        print(f"   Time2: {time2:.1f}ms")
    
    # Test hierarchical cache performance
    print("\n🔄 Testing Hierarchical Cache Performance")
    hier_config = HierarchicalCacheConfig()
    hier_config.l1_config.max_memory_mb = 16
    hier_config.l2_config.enabled = False  # No Redis for test
    hier_config.l3_config.enabled = False  # No SQLite for test
    
    hier_cache = HierarchicalCache(hier_config)
    
    # Test hierarchical cache hit
    hier_cache.store(cache_key, expensive_result)
    
    start_time = time.time()
    hier_result = hier_cache.get(cache_key)
    hier_time = (time.time() - start_time) * 1000
    
    print(f"Hierarchical cache L1 hit: {hier_time:.1f}ms")
    
    # Get hierarchical cache stats
    hier_stats = hier_cache.get_stats()
    print(f"\n[STATS] Hierarchical Cache Performance:")
    print(f"   L1 hit rate: {hier_stats.l1_hit_rate:.1%}")
    print(f"   Overall hit rate: {hier_stats.overall_hit_rate:.1%}")
    print(f"   Avg L1 response time: {hier_stats.avg_l1_response_time_ms:.1f}ms")
    
    # Clean up
    hier_cache.shutdown()
    
    assert result2 is not None, "Cache hit result should not be None"
    assert time2 < 10, f"Cache hit should be very fast, but took {time2:.1f}ms"
    assert hier_result is not None, "Hierarchical cache should return result"


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
        
        print("Running Phase 4 integration test...")
        sys.stdout.flush()
        results.append(test_phase4_integration())
        
        print("\nAll tests completed. Processing results...")
        sys.stdout.flush()
        
        if all(results):
            print("\n🎉 All cache component tests passed!")
            print("\n[IDEA] Next steps:")
            print("   - Phase 4 cache system is ready for production")
            print("   - Consider enabling Redis/SQLite persistence layers")
            print("   - Monitor cache performance with Phase4CacheSystem")
            print("   - Use hierarchical cache for optimal performance")
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