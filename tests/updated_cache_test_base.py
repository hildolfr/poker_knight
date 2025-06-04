#!/usr/bin/env python3
"""
♞ Poker Knight Updated Cache Test Base Classes

Updated base classes for cache testing with the new Phase 4 unified cache system.
Provides proper isolation, setup/teardown, and helper methods for testing
the hierarchical cache architecture.

Replaces the old cache_test_base.py with modern testing patterns.

Author: hildolfr
License: MIT
"""

import unittest
import tempfile
import time
import threading
import os
import shutil
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from unittest.mock import patch, MagicMock
from abc import ABC, abstractmethod

# Import Phase 4 cache components
try:
    from poker_knight.storage.unified_cache import (
        CacheKey, CacheResult, ThreadSafeMonteCarloCache, CacheKeyNormalizer
    )
    from poker_knight.storage.hierarchical_cache import (
        HierarchicalCache, HierarchicalCacheConfig, CacheLayer
    )
    from poker_knight.storage.phase4_integration import (
        Phase4CacheSystem, Phase4Config
    )
    PHASE4_AVAILABLE = True
except ImportError:
    PHASE4_AVAILABLE = False


class BaseCacheTest(unittest.TestCase, ABC):
    """Abstract base class for cache testing with proper isolation."""
    
    def setUp(self):
        """Set up test environment with proper isolation."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        
        # Create temporary directory for test isolation
        self.temp_dir = tempfile.mkdtemp(prefix="poker_cache_test_")
        self.temp_path = Path(self.temp_dir)
        
        # Initialize test-specific cache
        self.setup_cache()
        
        # Track test start time for performance measurements
        self.test_start_time = time.time()
    
    def tearDown(self):
        """Clean up test environment."""
        try:
            # Clean up cache
            self.cleanup_cache()
            
            # Remove temporary directory
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"Warning: Cleanup failed: {e}")
    
    @abstractmethod
    def setup_cache(self):
        """Set up cache for testing. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def cleanup_cache(self):
        """Clean up cache after testing. Must be implemented by subclasses."""
        pass
    
    def create_test_cache_key(self, 
                             hand: str = "AK_suited",
                             opponents: int = 2,
                             board: str = "preflop",
                             mode: str = "default") -> CacheKey:
        """Create test cache key with default values."""
        return CacheKey(
            hero_hand=hand,
            num_opponents=opponents,
            board_cards=board,
            simulation_mode=mode
        )
    
    def create_test_cache_result(self,
                                win_prob: float = 0.65,
                                tie_prob: float = 0.02,
                                loss_prob: float = 0.33,
                                simulations: int = 10000,
                                exec_time: float = 100.0) -> CacheResult:
        """Create test cache result with default values."""
        return CacheResult(
            win_probability=win_prob,
            tie_probability=tie_prob,
            loss_probability=loss_prob,
            confidence_interval=(win_prob - 0.03, win_prob + 0.03),
            simulations_run=simulations,
            execution_time_ms=exec_time,
            hand_categories={'pair': 0.15, 'high_card': 0.85},
            metadata={'test': True},
            timestamp=time.time()
        )
    
    def assert_cache_result_equal(self, result1: CacheResult, result2: CacheResult, places: int = 3):
        """Assert two cache results are equal within tolerance."""
        self.assertAlmostEqual(result1.win_probability, result2.win_probability, places=places)
        self.assertAlmostEqual(result1.tie_probability, result2.tie_probability, places=places)
        self.assertAlmostEqual(result1.loss_probability, result2.loss_probability, places=places)
        self.assertEqual(result1.simulations_run, result2.simulations_run)
    
    def assert_cache_stats_valid(self, stats):
        """Assert cache statistics are valid."""
        self.assertIsNotNone(stats)
        
        # Check for required attributes
        required_attrs = ['total_requests', 'cache_hits', 'cache_misses']
        for attr in required_attrs:
            self.assertTrue(hasattr(stats, attr), f"Missing attribute: {attr}")
            self.assertGreaterEqual(getattr(stats, attr), 0)
        
        # Validate hit rate calculation
        total = stats.cache_hits + stats.cache_misses
        if total > 0:
            expected_hit_rate = stats.cache_hits / total
            self.assertAlmostEqual(stats.hit_rate, expected_hit_rate, places=3)
    
    def measure_operation_time(self, operation: Callable) -> float:
        """Measure operation execution time in milliseconds."""
        start_time = time.time()
        operation()
        return (time.time() - start_time) * 1000


class UnifiedCacheTestBase(BaseCacheTest):
    """Base class for testing unified cache components."""
    
    def setup_cache(self):
        """Set up unified cache for testing."""
        self.cache = ThreadSafeMonteCarloCache(
            max_memory_mb=32,  # Small for testing
            max_entries=100,
            enable_persistence=False  # Disable for test isolation
        )
    
    def cleanup_cache(self):
        """Clean up unified cache."""
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()
    
    def test_basic_cache_operations(self):
        """Test basic cache operations."""
        # Create test data
        key = self.create_test_cache_key()
        result = self.create_test_cache_result()
        
        # Test storage
        success = self.cache.store(key, result)
        self.assertTrue(success)
        
        # Test retrieval
        retrieved = self.cache.get(key)
        self.assertIsNotNone(retrieved)
        self.assert_cache_result_equal(result, retrieved)
        
        # Test miss
        miss_key = self.create_test_cache_key(hand="72_offsuit")
        miss_result = self.cache.get(miss_key)
        self.assertIsNone(miss_result)
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        initial_stats = self.cache.get_stats()
        initial_requests = initial_stats.total_requests
        
        # Perform operations
        key1 = self.create_test_cache_key(hand="AA")
        key2 = self.create_test_cache_key(hand="KK")
        result = self.create_test_cache_result()
        
        self.cache.store(key1, result)
        self.cache.get(key1)  # Hit
        self.cache.get(key2)  # Miss
        
        # Check statistics
        final_stats = self.cache.get_stats()
        self.assertGreater(final_stats.total_requests, initial_requests)
        self.assertGreater(final_stats.cache_hits, initial_stats.cache_hits)
        self.assertGreater(final_stats.cache_misses, initial_stats.cache_misses)
        self.assert_cache_stats_valid(final_stats)


class HierarchicalCacheTestBase(BaseCacheTest):
    """Base class for testing hierarchical cache components."""
    
    def setup_cache(self):
        """Set up hierarchical cache for testing."""
        self.config = HierarchicalCacheConfig()
        # Small sizes for testing
        self.config.l1_config.max_memory_mb = 16
        self.config.l2_config.max_memory_mb = 32
        self.config.l3_config.max_memory_mb = 64
        # Disable persistence for test isolation
        self.config.l2_config.enabled = False  # Disable Redis for tests
        self.config.l3_config.enabled = False  # Disable SQLite for tests
        
        self.cache = HierarchicalCache(self.config)
    
    def cleanup_cache(self):
        """Clean up hierarchical cache."""
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()
            self.cache.shutdown()
    
    def test_hierarchical_operations(self):
        """Test hierarchical cache operations."""
        # Create test data
        key = self.create_test_cache_key()
        result = self.create_test_cache_result()
        
        # Test storage across layers
        success = self.cache.store(key, result)
        self.assertTrue(success)
        
        # Test retrieval
        retrieved = self.cache.get(key)
        self.assertIsNotNone(retrieved)
        self.assert_cache_result_equal(result, retrieved)
    
    def test_layer_statistics(self):
        """Test per-layer statistics."""
        stats = self.cache.get_stats()
        self.assertIsNotNone(stats)
        
        # Check layer-specific stats
        self.assertIsNotNone(stats.l1_stats)
        self.assert_cache_stats_valid(stats.l1_stats)


class Phase4IntegrationTestBase(BaseCacheTest):
    """Base class for testing complete Phase 4 integration."""
    
    def setup_cache(self):
        """Set up Phase 4 cache system for testing."""
        self.config = Phase4Config(
            optimization_level="balanced",
            auto_start_services=False,  # Manual control for testing
            enable_optimized_persistence=False  # Disable for test isolation
        )
        
        # Override cache directory for test isolation
        if self.config.persistence_config:
            self.config.persistence_config.cache_directory = str(self.temp_path)
        
        self.system = Phase4CacheSystem(self.config)
        self.system.initialize()
    
    def cleanup_cache(self):
        """Clean up Phase 4 system."""
        if hasattr(self, 'system') and self.system:
            self.system.stop_services()
    
    def test_system_integration(self):
        """Test Phase 4 system integration."""
        # Test system status
        status = self.system.get_system_status()
        self.assertIsInstance(status, dict)
        self.assertIn('initialized', status)
        self.assertTrue(status['initialized'])
    
    def test_service_lifecycle(self):
        """Test service start/stop lifecycle."""
        # Start services
        success = self.system.start_services()
        self.assertTrue(success)
        self.assertTrue(self.system._started)
        
        # Stop services
        success = self.system.stop_services()
        self.assertTrue(success)
        self.assertFalse(self.system._started)


class PerformanceCacheTestBase(BaseCacheTest):
    """Base class for cache performance testing."""
    
    def setup_cache(self):
        """Set up cache for performance testing."""
        self.cache = ThreadSafeMonteCarloCache(max_memory_mb=64)
        self.performance_data = {}
    
    def cleanup_cache(self):
        """Clean up performance cache."""
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def benchmark_cache_operation(self, operation_name: str, operation: Callable, iterations: int = 100) -> Dict[str, float]:
        """Benchmark cache operation performance."""
        times = []
        
        for _ in range(iterations):
            start_time = time.time()
            operation()
            execution_time = (time.time() - start_time) * 1000  # Convert to ms
            times.append(execution_time)
        
        # Calculate statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        performance = {
            'avg_time_ms': avg_time,
            'min_time_ms': min_time,
            'max_time_ms': max_time,
            'iterations': iterations
        }
        
        self.performance_data[operation_name] = performance
        return performance
    
    def assert_performance_acceptable(self, operation_name: str, max_avg_time_ms: float):
        """Assert that operation performance is within acceptable limits."""
        self.assertIn(operation_name, self.performance_data)
        performance = self.performance_data[operation_name]
        
        self.assertLess(
            performance['avg_time_ms'], 
            max_avg_time_ms,
            f"{operation_name} average time {performance['avg_time_ms']:.2f}ms exceeds limit {max_avg_time_ms}ms"
        )
    
    def test_cache_hit_performance(self):
        """Test cache hit performance."""
        # Setup data
        key = self.create_test_cache_key()
        result = self.create_test_cache_result()
        self.cache.store(key, result)
        
        # Benchmark cache hits
        def cache_hit():
            self.cache.get(key)
        
        performance = self.benchmark_cache_operation("cache_hit", cache_hit, iterations=1000)
        
        # Cache hits should be very fast
        self.assert_performance_acceptable("cache_hit", max_avg_time_ms=1.0)
    
    def test_cache_miss_performance(self):
        """Test cache miss performance."""
        # Benchmark cache misses
        def cache_miss():
            miss_key = self.create_test_cache_key(hand=f"random_{time.time()}")
            self.cache.get(miss_key)
        
        performance = self.benchmark_cache_operation("cache_miss", cache_miss, iterations=100)
        
        # Cache misses should still be reasonably fast
        self.assert_performance_acceptable("cache_miss", max_avg_time_ms=5.0)


class MockSolverTestBase(BaseCacheTest):
    """Base class for testing cache integration with mocked solver."""
    
    def setup_cache(self):
        """Set up mock solver environment."""
        self.cache = ThreadSafeMonteCarloCache()
        self.mock_solver = self.create_mock_solver()
    
    def cleanup_cache(self):
        """Clean up mock solver environment."""
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def create_mock_solver(self):
        """Create mock solver for testing."""
        mock_solver = MagicMock()
        mock_solver.get_cache_stats.return_value = {
            'caching_enabled': True,
            'cache_type': 'unified',
            'unified_cache': {
                'total_requests': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'hit_rate': 0.0
            }
        }
        return mock_solver
    
    def simulate_solver_analysis(self, hand: List[str], opponents: int, use_cache: bool = True) -> Dict[str, Any]:
        """Simulate solver analysis with cache integration."""
        # Create cache key
        cache_key = CacheKey(
            hero_hand=CacheKeyNormalizer.normalize_hand(hand),
            num_opponents=opponents,
            board_cards="preflop",
            simulation_mode="default"
        )
        
        if use_cache:
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return {
                    'win_probability': cached_result.win_probability,
                    'execution_time_ms': 1.0,  # Fast cache hit
                    'cached': True
                }
        
        # Simulate computation
        time.sleep(0.01)  # Simulate computation time
        result_data = {
            'win_probability': 0.65,
            'execution_time_ms': 100.0,  # Slow computation
            'cached': False
        }
        
        if use_cache:
            # Store in cache
            cache_result = self.create_test_cache_result(win_prob=result_data['win_probability'])
            self.cache.store(cache_key, cache_result)
        
        return result_data
    
    def test_cache_speedup(self):
        """Test that cache provides significant speedup."""
        hand = ['A♠', 'K♠']
        opponents = 2
        
        # First run (no cache)
        result1 = self.simulate_solver_analysis(hand, opponents, use_cache=True)
        self.assertFalse(result1['cached'])
        
        # Second run (cache hit)
        result2 = self.simulate_solver_analysis(hand, opponents, use_cache=True)
        self.assertTrue(result2['cached'])
        
        # Cache should be significantly faster
        speedup_factor = result1['execution_time_ms'] / result2['execution_time_ms']
        self.assertGreater(speedup_factor, 10, "Cache should provide at least 10x speedup")


# Utility functions for test data generation
def generate_test_hands(count: int = 10) -> List[List[str]]:
    """Generate test poker hands."""
    hands = [
        ['A♠', 'A♥'],  # Pocket aces
        ['K♠', 'K♥'],  # Pocket kings
        ['Q♠', 'Q♥'],  # Pocket queens
        ['J♠', 'J♥'],  # Pocket jacks
        ['A♠', 'K♠'],  # AK suited
        ['A♠', 'K♥'],  # AK offsuit
        ['A♠', 'Q♠'],  # AQ suited
        ['K♠', 'Q♠'],  # KQ suited
        ['7♠', '2♥'],  # 72 offsuit (worst hand)
        ['9♠', '8♠'],  # 98 suited
    ]
    return hands[:count]


def generate_test_scenarios(hand_count: int = 5, opponent_range: tuple = (1, 4)) -> List[Dict[str, Any]]:
    """Generate test scenarios for comprehensive testing."""
    scenarios = []
    hands = generate_test_hands(hand_count)
    
    for hand in hands:
        for opponents in range(opponent_range[0], opponent_range[1] + 1):
            for mode in ['fast', 'default']:
                scenarios.append({
                    'hand': hand,
                    'opponents': opponents,
                    'mode': mode,
                    'expected_cache_key': f"{hand[0]}{hand[1]}_{opponents}_{mode}"
                })
    
    return scenarios


# Test suite helper functions
def create_cache_test_suite(test_base_class, test_methods: Optional[List[str]] = None):
    """Create test suite for cache testing."""
    suite = unittest.TestSuite()
    
    if test_methods:
        for method in test_methods:
            suite.addTest(test_base_class(method))
    else:
        suite.addTest(unittest.makeSuite(test_base_class))
    
    return suite


def run_cache_test_suite(suite, verbosity: int = 2):
    """Run cache test suite with proper reporting."""
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return {
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success': result.wasSuccessful(),
        'failure_details': result.failures,
        'error_details': result.errors
    }