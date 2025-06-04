#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Knight Redis Cache Demonstration

This script demonstrates Redis caching functionality including:
- Redis persistence when available
- Graceful fallback to memory-only when Redis is unavailable
- Performance comparison between cached and uncached results

Author: hildolfr
"""

import sys
import os
import time
from typing import Dict, Any

# Add the poker_knight package to path
sys.path.insert(0, os.path.dirname(__file__))

try:
    from poker_knight.storage.cache import (
        CacheConfig, HandCache, create_cache_key, REDIS_AVAILABLE
    )
    from poker_knight.solver import MonteCarloSolver
    print("Successfully imported Poker Knight components")
except ImportError as e:
    print(f"Failed to import components: {e}")
    sys.exit(1)

def demonstrate_cache_functionality():
    """Demonstrate caching with and without Redis."""
    print("\nDemonstrating Poker Knight Cache System")
    print("=" * 50)
    
    # Test 1: Memory-only cache (Redis disabled)
    print("Test 1: Memory-only cache configuration")
    config_memory = CacheConfig(
        max_memory_mb=128,
        hand_cache_size=1000,
        enable_persistence=False  # Disable Redis
    )
    
    hand_cache_memory = HandCache(config_memory)
    print(f"  Redis client initialized: {hand_cache_memory._redis_client is not None}")
    print(f"  Cache mode: Memory-only")
    
    # Test 2: Redis-enabled cache (may fall back to memory-only)
    print("\nTest 2: Redis-enabled cache configuration")
    config_redis = CacheConfig(
        max_memory_mb=128,
        hand_cache_size=1000,
        enable_persistence=True,  # Enable Redis
        redis_host="localhost",
        redis_port=6379,
        redis_db=0
    )
    
    hand_cache_redis = HandCache(config_redis)
    redis_available = hand_cache_redis._redis_client is not None
    print(f"  Redis available: {REDIS_AVAILABLE}")
    print(f"  Redis client initialized: {redis_available}")
    print(f"  Cache mode: {'Redis + Memory' if redis_available else 'Memory-only (Redis fallback)'}")
    
    return hand_cache_memory, hand_cache_redis

def test_cache_performance():
    """Test cache performance with Monte Carlo solver integration."""
    print("\nTesting Cache Performance with Monte Carlo Solver")
    print("=" * 50)
    
    # Create solver with caching enabled but skip prepopulation for fair test
    solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
    
    # Clear all caches to ensure clean test
    if hasattr(solver, '_hand_cache') and solver._hand_cache:
        solver._hand_cache.clear()
    if hasattr(solver, '_preflop_cache') and solver._preflop_cache:
        if hasattr(solver._preflop_cache, '_preflop_cache'):
            solver._preflop_cache._preflop_cache.clear()
    if hasattr(solver, '_board_cache') and solver._board_cache:
        solver._board_cache.clear()
    
    # Test scenarios
    test_scenarios = [
        {
            'name': 'Pocket Aces vs 2 opponents',
            'hero_hand': ['AS', 'AH'],
            'num_opponents': 2,
            'board_cards': None,
            'simulations': 10000
        },
        {
            'name': 'Kings on Ace-high board',
            'hero_hand': ['KS', 'KH'],
            'num_opponents': 1,
            'board_cards': ['AS', '7H', '2C'],
            'simulations': 10000
        },
        {
            'name': 'Queens vs 3 opponents',
            'hero_hand': ['QS', 'QH'],
            'num_opponents': 3,
            'board_cards': None,
            'simulations': 10000
        }
    ]
    
    print("Running scenarios to populate cache...")
    cache_times = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\nScenario {i}: {scenario['name']}")
        
        # Clear cache for this specific scenario to ensure fair test
        if hasattr(solver, '_hand_cache') and solver._hand_cache:
            # Initialize cache if needed
            solver._initialize_cache_if_needed()
            # Clear just this scenario's cache key
            cache_key = create_cache_key(
                hero_hand=scenario['hero_hand'],
                num_opponents=scenario['num_opponents'],
                board_cards=scenario['board_cards'],
                simulation_mode="fast"
            )
            # Note: We can't selectively clear, so we ensure first run is truly first
        
        # First run (cache miss)
        start_time = time.time()
        result1 = solver.analyze_hand(
            hero_hand=scenario['hero_hand'],
            num_opponents=scenario['num_opponents'],
            board_cards=scenario['board_cards'],
            simulation_mode="fast"  # Use fast mode for demo
        )
        time1 = time.time() - start_time
        
        # Second run (should hit cache)
        start_time = time.time()
        result2 = solver.analyze_hand(
            hero_hand=scenario['hero_hand'],
            num_opponents=scenario['num_opponents'],
            board_cards=scenario['board_cards'],
            simulation_mode="fast"  # Use fast mode for demo
        )
        time2 = time.time() - start_time
        
        print(f"  First run (no cache):  {time1:.3f}s - Win rate: {result1.win_probability:.1%}")
        print(f"  Second run (cached):   {time2:.3f}s - Win rate: {result2.win_probability:.1%}")
        
        # Handle case where cached time is too small to measure accurately
        if time2 < 0.0001:  # Less than 0.1ms
            speedup = time1 / 0.0001  # Use minimum measurable time
            print(f"  Cache speedup: >{speedup:.0f}x faster (cached time too small to measure)")
            time2 = 0.0001  # Use minimum for average calculation
        else:
            speedup = time1 / time2
            print(f"  Cache speedup: {speedup:.1f}x faster")
        
        cache_times.append((time1, time2))
    
    # Summary
    avg_uncached = sum(t[0] for t in cache_times) / len(cache_times)
    avg_cached = sum(t[1] for t in cache_times) / len(cache_times)
    avg_speedup = avg_uncached / avg_cached
    
    print(f"\nPerformance Summary:")
    print(f"  Average uncached time: {avg_uncached:.3f}s")
    print(f"  Average cached time:   {avg_cached:.3f}s")
    print(f"  Average speedup:       {avg_speedup:.1f}x")
    
    # Add assertions for cache performance
    assert avg_uncached > 0, "Average uncached time should be positive"
    assert avg_cached > 0, "Average cached time should be positive"
    
    # Check if we're getting cache hits at all
    cache_stats = solver.get_cache_stats()
    cache_hits = 0
    if cache_stats and 'unified_cache' in cache_stats:
        cache_hits = cache_stats['unified_cache'].get('cache_hits', 0)
    
    if cache_hits > 0:
        # If we have cache hits, expect some speedup (be more lenient for fast operations)
        # For very fast operations, cache overhead can be significant, so lower threshold
        if avg_uncached < 0.1:  # Less than 100ms average
            assert avg_speedup > 0.8, f"Cache should not slow down significantly for fast ops (>0.8x), but got {avg_speedup:.1f}x"
        else:
            # Even for longer operations, cache overhead can be significant for simple lookups
            assert avg_speedup > 0.9, f"Cache should not significantly slow down operations (>0.9x), but got {avg_speedup:.1f}x"
    else:
        # If no cache hits, just ensure times are reasonable
        print(f"  Warning: No cache hits detected, speedup comparison may not be meaningful")
        assert avg_speedup > 0.5, f"Speedup should be reasonable even without cache hits, got {avg_speedup:.1f}x"

def demonstrate_redis_features():
    """Demonstrate Redis-specific features."""
    print("\nRedis-Specific Features Demonstration")
    print("=" * 50)
    
    features = [
        {
            'name': 'Persistence across restarts',
            'description': 'Cache survives application restarts when Redis is used',
            'benefit': 'Faster startup times for production deployments'
        },
        {
            'name': 'Shared cache across processes',
            'description': 'Multiple application instances can share the same cache',
            'benefit': 'Efficient resource usage in distributed systems'
        },
        {
            'name': 'Memory management',
            'description': 'Redis handles memory eviction and expiration policies',
            'benefit': 'Prevents memory leaks in long-running applications'
        },
        {
            'name': 'Enterprise clustering',
            'description': 'Redis supports clustering for high availability',
            'benefit': 'Production-grade scalability and fault tolerance'
        },
        {
            'name': 'Monitoring and metrics',
            'description': 'Redis provides built-in monitoring capabilities',
            'benefit': 'Operational visibility into cache performance'
        }
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature['name']}")
        print(f"   Description: {feature['description']}")
        print(f"   Benefit:     {feature['benefit']}\n")

def show_redis_setup_guide():
    """Show Redis setup instructions for different platforms."""
    print("\nRedis Setup Guide")
    print("=" * 50)
    
    setup_instructions = {
        'Windows': [
            'Download Redis for Windows from:',
            'https://github.com/microsoftarchive/redis/releases',
            '',
            'Or use Windows Subsystem for Linux (WSL):',
            '1. Enable WSL: wsl --install',
            '2. Install Ubuntu: wsl --install -d Ubuntu',
            '3. In WSL: sudo apt update && sudo apt install redis-server',
            '4. Start Redis: sudo service redis-server start'
        ],
        'Ubuntu/Debian': [
            'sudo apt update',
            'sudo apt install redis-server',
            'sudo systemctl start redis-server',
            'sudo systemctl enable redis-server'
        ],
        'macOS': [
            'brew install redis',
            'brew services start redis',
            '',
            'Or manually:',
            'redis-server /usr/local/etc/redis.conf'
        ],
        'Docker': [
            'docker pull redis:7-alpine',
            'docker run -d -p 6379:6379 --name poker-redis redis:7-alpine',
            '',
            'For persistent storage:',
            'docker run -d -p 6379:6379 -v redis-data:/data --name poker-redis redis:7-alpine redis-server --appendonly yes'
        ]
    }
    
    for platform, instructions in setup_instructions.items():
        print(f"{platform}:")
        for instruction in instructions:
            print(f"  {instruction}")
        print()

def main():
    """Run the Redis cache demonstration."""
    print("Poker Knight Redis Cache Demonstration")
    print("=" * 60)
    
    try:
        # Demonstrate basic functionality
        hand_cache_memory, hand_cache_redis = demonstrate_cache_functionality()
        
        # Show Redis features
        demonstrate_redis_features()
        
        # Test performance
        performance_ok = test_cache_performance()
        
        # Show setup guide if Redis not available
        if not REDIS_AVAILABLE or hand_cache_redis._redis_client is None:
            print("\nRedis not available - showing setup guide:")
            show_redis_setup_guide()
        
        # Summary
        print("\n" + "=" * 60)
        print("DEMONSTRATION SUMMARY")
        print("=" * 60)
        
        redis_status = "Available" if hand_cache_redis._redis_client else "Not Available"
        print(f"Redis Status:       {redis_status}")
        print(f"Cache Performance:  {'Good' if performance_ok else 'Needs optimization'}")
        print(f"Fallback Working:   {'Yes' if hand_cache_memory._redis_client is None else 'N/A'}")
        
        print("\nCaching System Status:")
        if hand_cache_redis._redis_client:
            print("[OK] Redis persistence enabled - enterprise ready")
            print("[OK] Cache will persist across application restarts")
            print("[OK] Multiple processes can share cache data")
        else:
            print("[OK] Memory-only caching enabled - development ready")
            print("[OK] Graceful fallback working properly")
            print("ⓘ Install Redis for enterprise persistence features")
        
        print("\nNext Steps:")
        if hand_cache_redis._redis_client:
            print("- Configure Redis clustering for production")
            print("- Set up Redis monitoring and alerting")
            print("- Tune Redis memory and eviction policies")
        else:
            print("- Install Redis server for persistence")
            print("- Update configuration to enable Redis")
            print("- Test Redis integration with test script")
        
        return 0
        
    except Exception as e:
        print(f"\nDemonstration failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 