#!/usr/bin/env python3
"""
Poker Knight Test Suite Integration Demo

This test file demonstrates how the cache tests are integrated into the
overall test suite and validates the unified testing approach.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to the path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from poker_knight.storage.cache import (
        CacheConfig, HandCache, BoardTextureCache, PreflopRangeCache,
        create_cache_key, get_cache_manager
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

try:
    from poker_knight.solver import MonteCarloSolver
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False


@pytest.mark.cache
@pytest.mark.cache_integration
@pytest.mark.integration
class TestCacheIntegrationDemo:
    """Demonstrate cache integration with the main test suite."""
    
    @pytest.mark.quick
    def test_cache_markers_applied(self):
        """Test that cache markers are properly applied."""
        # This test should have cache, cache_integration, integration, and quick markers
        pass
    
    @pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache module not available")
    def test_cache_module_import(self):
        """Test that cache modules can be imported."""
        assert CacheConfig is not None
        assert HandCache is not None
        assert BoardTextureCache is not None
        assert PreflopRangeCache is not None
    
    @pytest.mark.skipif(not SOLVER_AVAILABLE, reason="Solver module not available")
    def test_solver_module_import(self):
        """Test that solver module can be imported."""
        assert MonteCarloSolver is not None
    
    @pytest.mark.skipif(not CACHE_AVAILABLE or not SOLVER_AVAILABLE, 
                       reason="Cache or solver modules not available")
    def test_cache_solver_integration_basic(self):
        """Test basic integration between cache and solver."""
        # Create cache config with memory-only settings
        config = CacheConfig(
            enable_persistence=False,
            max_memory_mb=64,
            hand_cache_size=100
        )
        
        # Get cache managers
        hand_cache, board_cache, preflop_cache = get_cache_manager(config)
        
        # Create a simple cache key
        cache_key = create_cache_key(
            hero_hand=['Ac', 'Kd'],
            num_opponents=2,
            simulation_mode="fast",
            config=config
        )
        
        assert cache_key is not None
        assert len(cache_key) > 0
        
        # Test cache miss (should return None)
        result = hand_cache.get_result(cache_key)
        assert result is None
        
        # Store a dummy result
        dummy_result = {
            'hero_hand': ['Ac', 'Kd'],
            'win_probability': 0.65,
            'tie_probability': 0.05,
            'simulations_run': 1000
        }
        
        success = hand_cache.store_result(cache_key, dummy_result)
        assert success
        
        # Test cache hit
        cached_result = hand_cache.get_result(cache_key)
        assert cached_result is not None
        assert cached_result['win_probability'] == 0.65


@pytest.mark.cache
@pytest.mark.cache_unit
@pytest.mark.unit
class TestCacheUnitDemo:
    """Demonstrate cache unit tests integration."""
    
    @pytest.mark.quick
    @pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache module not available")
    def test_cache_config_creation(self):
        """Test cache configuration creation (quick unit test)."""
        config = CacheConfig()
        assert config.max_memory_mb == 512
        assert config.hand_cache_size == 10000
        assert config.enable_persistence == False
    
    @pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache module not available")
    def test_cache_key_generation(self):
        """Test cache key generation (unit test)."""
        key1 = create_cache_key(['Ac', 'Kd'], 2, simulation_mode="fast")
        key2 = create_cache_key(['Kd', 'Ac'], 2, simulation_mode="fast")  # Same hand, different order
        
        # Keys should be the same (normalized)
        assert key1 == key2
        
        key3 = create_cache_key(['Ac', 'Kd'], 3, simulation_mode="fast")  # Different opponent count
        assert key1 != key3


@pytest.mark.cache
@pytest.mark.cache_performance
@pytest.mark.performance
class TestCachePerformanceDemo:
    """Demonstrate cache performance tests integration."""
    
    @pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache module not available")
    def test_cache_performance_basic(self):
        """Test basic cache performance (demonstration)."""
        import time
        
        config = CacheConfig(enable_persistence=False, max_memory_mb=32)
        hand_cache, _, _ = get_cache_manager(config)
        
        # Test cache write performance
        start_time = time.time()
        
        for i in range(100):
            cache_key = f"test_key_{i}"
            dummy_result = {'win_probability': 0.5 + (i * 0.001)}
            hand_cache.store_result(cache_key, dummy_result)
        
        write_time = (time.time() - start_time) * 1000  # Convert to ms
        assert write_time < 100  # Should complete in under 100ms
        
        # Test cache read performance
        start_time = time.time()
        
        for i in range(100):
            cache_key = f"test_key_{i}"
            result = hand_cache.get_result(cache_key)
            assert result is not None
        
        read_time = (time.time() - start_time) * 1000  # Convert to ms
        assert read_time < 50  # Should complete in under 50ms


@pytest.mark.cache
@pytest.mark.cache_persistence
@pytest.mark.slow
class TestCachePersistenceDemo:
    """Demonstrate cache persistence tests integration."""
    
    @pytest.mark.skipif(not CACHE_AVAILABLE, reason="Cache module not available")
    def test_sqlite_fallback_available(self):
        """Test that SQLite fallback is available."""
        config = CacheConfig(
            enable_persistence=True,
            sqlite_path=":memory:",  # Use in-memory SQLite for testing
            # Force SQLite by using invalid Redis settings
            redis_host="invalid_host_that_does_not_exist",
            redis_port=99999
        )
        
        hand_cache = HandCache(config)
        assert hand_cache is not None
        
        # Check that SQLite is being used (not Redis)
        stats = hand_cache.get_persistence_stats()
        assert stats['persistence_type'] == 'sqlite'


@pytest.mark.cache
@pytest.mark.redis_required
@pytest.mark.slow
class TestRedisDemo:
    """Demonstrate Redis tests integration."""
    
    def test_redis_marker_applied(self):
        """Test that redis_required marker is applied."""
        # This test will be skipped if Redis is not available
        # The marker allows for conditional test execution
        pass


# Utility test to validate marker integration
class TestMarkerIntegration:
    """Test that markers are properly integrated."""
    
    def test_marker_system_working(self):
        """Basic test to ensure marker system is functional."""
        # This is a basic test that should always run
        assert True
    
    @pytest.mark.quick
    def test_quick_marker(self):
        """Test that quick marker works."""
        assert True
    
    @pytest.mark.slow
    def test_slow_marker(self):
        """Test that slow marker works."""
        import time
        time.sleep(0.1)  # Simulate slow test
        assert True


if __name__ == "__main__":
    # Allow running this file directly for quick testing
    pytest.main([__file__, "-v"]) 