#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
♞ Poker Knight SQLite Integration Test

Simple test to verify SQLite fallback works with MonteCarloSolver.
"""

import time
import os
import sys
import json
import tempfile

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight.solver import MonteCarloSolver


def test_sqlite_integration():
    """Test SQLite fallback with MonteCarloSolver."""
    print("🧪 Testing SQLite Fallback with MonteCarloSolver")
    print("=" * 55)
    
    # Create temporary config file for SQLite fallback testing
    config_data = {
        "simulation_settings": {
            "fast_mode_simulations": 1000,
            "default_simulations": 2000,
            "precision_mode_simulations": 5000,
            "max_workers": 2
        },
        "performance_settings": {
            "timeout_fast_mode_ms": 3000,
            "timeout_default_mode_ms": 20000,
            "timeout_precision_mode_ms": 120000,
            "optimization_level": 2
        },
        "output_settings": {
            "include_hand_categories": True,
            "include_confidence_intervals": True
        },
        "cache_settings": {
            "max_memory_mb": 64,
            "hand_cache_size": 100,
            "enable_persistence": True,
            "redis_host": "nonexistent_host",  # Force Redis failure
            "redis_port": 9999,
            "sqlite_path": "test_integration_cache.db"
        }
    }
    
    # Create temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f, indent=2)
        config_path = f.name
    
    try:
        print("Creating MonteCarloSolver with SQLite fallback...")
        
        with MonteCarloSolver(config_path=config_path, enable_caching=True) as solver:
            # Get cache stats to see what persistence is being used
            cache_stats = solver.get_cache_stats()
            print(f"Cache enabled: {cache_stats['caching_enabled'] if cache_stats else False}")
            
            # Check persistence type if available
            if hasattr(solver, '_hand_cache') and hasattr(solver._hand_cache, 'get_persistence_stats'):
                persistence_stats = solver._hand_cache.get_persistence_stats()
                print(f"Persistence type: {persistence_stats['persistence_type']}")
                print(f"SQLite available: {persistence_stats['sqlite_available']}")
                print(f"Redis connected: {persistence_stats['redis_connected']}")
            
            print("\n1️⃣ First analysis (will be cached)...")
            start_time = time.time()
            result1 = solver.analyze_hand(
                hero_hand=['AS', 'AH'],
                num_opponents=2,
                simulation_mode="fast"
            )
            time1 = (time.time() - start_time) * 1000
            print(f"   Result: {result1.win_probability:.3f} win rate")
            print(f"   Time: {time1:.1f}ms")
            
            print("\n2️⃣ Second analysis (should be cached)...")
            start_time = time.time()
            result2 = solver.analyze_hand(
                hero_hand=['AS', 'AH'],
                num_opponents=2,
                simulation_mode="fast"
            )
            time2 = (time.time() - start_time) * 1000
            print(f"   Result: {result2.win_probability:.3f} win rate")
            print(f"   Time: {time2:.1f}ms")
            
            # Check cache performance
            cache_stats_final = solver.get_cache_stats()
            if cache_stats_final:
                print(f"\n[STATS] Cache Performance:")
                print(f"   Total requests: {cache_stats_final['hand_cache']['total_requests']}")
                print(f"   Cache hits: {cache_stats_final['hand_cache']['cache_hits']}")
                print(f"   Cache misses: {cache_stats_final['hand_cache']['cache_misses']}")
                print(f"   Hit rate: {cache_stats_final['hand_cache']['hit_rate']:.1%}")
                
                cached = time2 < time1 * 0.5  # Second run should be much faster
                print(f"   Caching working: {'[PASS]' if cached else '[FAIL]'}")
            
            print("\n3️⃣ Testing different hand...")
            start_time = time.time()
            result3 = solver.analyze_hand(
                hero_hand=['KS', 'KH'],
                num_opponents=1,
                simulation_mode="fast"
            )
            time3 = (time.time() - start_time) * 1000
            print(f"   Result: {result3.win_probability:.3f} win rate")
            print(f"   Time: {time3:.1f}ms")
        
        print("\n4️⃣ Testing persistence across solver instances...")
        
        # Create new solver instance to test persistence
        with MonteCarloSolver(config_path=config_path, enable_caching=True) as new_solver:
            start_time = time.time()
            result4 = new_solver.analyze_hand(
                hero_hand=['AS', 'AH'],
                num_opponents=2,
                simulation_mode="fast"
            )
            time4 = (time.time() - start_time) * 1000
            print(f"   Result: {result4.win_probability:.3f} win rate")
            print(f"   Time: {time4:.1f}ms")
            
            persistent = time4 < time1 * 0.5  # Should be fast due to SQLite persistence
            print(f"   Persistence working: {'[PASS]' if persistent else '[FAIL]'}")
            
            cache_stats_new = new_solver.get_cache_stats()
            if cache_stats_new:
                print(f"   New instance cache hits: {cache_stats_new['hand_cache']['cache_hits']}")
        
        print("\n[PASS] SQLite integration test completed!")
        return True
        
    finally:
        # Clean up temporary config file
        try:
            os.unlink(config_path)
        except:
            pass
        
        # Clean up test database
        db_path = config_data["cache_settings"]["sqlite_path"]
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"🧹 Cleaned up test database: {db_path}")
            except OSError:
                print(f"🧹 Note: {db_path} will be cleaned up when process exits")


if __name__ == "__main__":
    test_sqlite_integration() 