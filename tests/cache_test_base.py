"""
♞ Poker Knight Cache Test Base Classes

Base classes for testing the unified cache system with proper isolation,
setup/teardown, and helper methods for cache behavior verification.

Author: hildolfr
License: MIT
"""

import unittest
import tempfile
import os
import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Import unified cache system
try:
    from poker_knight.storage.unified_cache import (
        ThreadSafeMonteCarloCache, CacheKey, CacheResult, CacheStats,
        create_cache_key, CacheKeyNormalizer, clear_unified_cache
    )
    UNIFIED_CACHE_AVAILABLE = True
except ImportError:
    UNIFIED_CACHE_AVAILABLE = False

# Import legacy cache for comparison tests
try:
    from poker_knight.storage.cache import (
        HandCache, BoardTextureCache, PreflopRangeCache,
        CacheConfig, create_cache_key as legacy_create_cache_key
    )
    LEGACY_CACHE_AVAILABLE = True
except ImportError:
    LEGACY_CACHE_AVAILABLE = False

# Import solver for integration tests
try:
    from poker_knight import MonteCarloSolver
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False


class CacheTestBase(unittest.TestCase):
    """
    Base class for all cache-related tests.
    
    Provides common setup/teardown, cache isolation, and helper methods
    for testing cache behavior in a predictable manner.
    """
    
    def setUp(self):
        """Set up isolated cache environment for each test."""
        # Create temporary directory for test databases
        self.temp_dir = tempfile.mkdtemp()
        self.test_sqlite_path = os.path.join(self.temp_dir, "test_cache.db")
        
        # Clear any global cache state
        clear_unified_cache()
        
        # Initialize test-specific cache instances
        self.unified_cache = None
        self.legacy_cache = None
        
        # Test data for consistent scenarios
        self.test_hands = [
            ["A♠", "K♠"],    # Premium suited
            ["Q♥", "Q♦"],    # Pocket pair
            ["J♠", "10♠"],   # Suited connector
            ["A♥", "K♣"],    # Premium offsuit
            ["7♠", "6♠"],    # Low suited connector
            ["2♥", "7♣"]     # Weak hand
        ]
        
        self.test_boards = [
            None,                           # Preflop
            ["A♠", "K♥", "Q♦"],            # High rainbow flop
            ["9♠", "8♠", "7♠"],            # Flush flop
            ["A♠", "A♥", "K♦"],            # Paired flop
            ["2♠", "7♥", "J♦", "5♣"],      # Turn
            ["K♠", "Q♥", "J♦", "10♣", "9♠"] # Straight river
        ]
        
        self.test_opponent_counts = [1, 2, 3, 4, 5, 6]
        self.test_simulation_modes = ["fast", "default", "precision"]
    
    def tearDown(self):
        """Clean up test environment."""
        # Close cache connections
        if self.unified_cache and hasattr(self.unified_cache, '_sqlite_connection'):
            try:
                self.unified_cache._sqlite_connection.close()
            except:
                pass
        
        if self.legacy_cache:
            try:
                self.legacy_cache.clear()
            except:
                pass
        
        # Clean up temporary files
        try:
            if os.path.exists(self.test_sqlite_path):
                os.remove(self.test_sqlite_path)
            os.rmdir(self.temp_dir)
        except:
            pass
        
        # Clear global cache state
        clear_unified_cache()
    
    def create_test_unified_cache(self, 
                                 enable_persistence: bool = False,
                                 max_memory_mb: int = 64) -> ThreadSafeMonteCarloCache:
        """Create isolated unified cache for testing."""
        if not UNIFIED_CACHE_AVAILABLE:
            self.skipTest("Unified cache not available")
        
        self.unified_cache = ThreadSafeMonteCarloCache(
            max_memory_mb=max_memory_mb,
            max_entries=1000,
            enable_persistence=enable_persistence,
            sqlite_path=self.test_sqlite_path
        )
        return self.unified_cache
    
    def create_test_legacy_cache(self) -> Tuple[Any, Any, Any]:
        """Create isolated legacy cache for comparison tests."""
        if not LEGACY_CACHE_AVAILABLE:
            self.skipTest("Legacy cache not available")
        
        config = CacheConfig(
            max_memory_mb=64,
            hand_cache_size=1000,
            enable_persistence=False
        )
        
        # Create separate instances to avoid global state
        hand_cache = HandCache(config)
        board_cache = BoardTextureCache(config)
        preflop_cache = PreflopRangeCache(config)
        
        self.legacy_cache = (hand_cache, board_cache, preflop_cache)
        return self.legacy_cache
    
    def create_test_cache_key(self, 
                             hero_hand: List[str] = None,
                             num_opponents: int = 2,
                             board_cards: List[str] = None,
                             simulation_mode: str = "default") -> CacheKey:
        """Create test cache key with defaults."""
        if hero_hand is None:
            hero_hand = ["A♠", "K♠"]
        
        return create_cache_key(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=board_cards,
            simulation_mode=simulation_mode
        )
    
    def create_test_cache_result(self,
                                win_prob: float = 0.65,
                                tie_prob: float = 0.02,
                                simulations: int = 10000) -> CacheResult:
        """Create test cache result with realistic values."""
        loss_prob = round(1.0 - win_prob - tie_prob, 10)  # Round to avoid floating point precision issues
        
        return CacheResult(
            win_probability=win_prob,
            tie_probability=tie_prob,
            loss_probability=loss_prob,
            confidence_interval=(win_prob - 0.02, win_prob + 0.02),
            simulations_run=simulations,
            execution_time_ms=25.5,
            hand_categories={
                'pair': 0.42,
                'two_pair': 0.23,
                'three_of_a_kind': 0.04,
                'straight': 0.03,
                'flush': 0.03,
                'full_house': 0.02,
                'high_card': 0.23
            },
            metadata={
                'test_generated': True,
                'convergence_achieved': True,
                'effective_sample_size': simulations * 0.85
            },
            timestamp=time.time()
        )
    
    def assert_cache_hit(self, cache, key: CacheKey, expected_result: CacheResult):
        """Assert that cache returns expected result for key."""
        result = cache.get(key)
        self.assertIsNotNone(result, "Cache should return result for stored key")
        self.assertAlmostEqual(result.win_probability, expected_result.win_probability, places=4)
        self.assertAlmostEqual(result.tie_probability, expected_result.tie_probability, places=4)
        self.assertAlmostEqual(result.loss_probability, expected_result.loss_probability, places=4)
    
    def assert_cache_miss(self, cache, key: CacheKey):
        """Assert that cache returns None for key."""
        result = cache.get(key)
        self.assertIsNone(result, "Cache should return None for non-existent key")
    
    def assert_cache_stats_updated(self, cache, 
                                  expected_hits: int = None,
                                  expected_misses: int = None,
                                  expected_requests: int = None):
        """Assert cache statistics are as expected."""
        stats = cache.get_stats()
        
        if expected_hits is not None:
            self.assertEqual(stats.cache_hits, expected_hits, 
                           f"Expected {expected_hits} cache hits, got {stats.cache_hits}")
        
        if expected_misses is not None:
            self.assertEqual(stats.cache_misses, expected_misses,
                           f"Expected {expected_misses} cache misses, got {stats.cache_misses}")
        
        if expected_requests is not None:
            self.assertEqual(stats.total_requests, expected_requests,
                           f"Expected {expected_requests} total requests, got {stats.total_requests}")
    
    def wait_for_cache_operation(self, timeout_ms: int = 100):
        """Wait for asynchronous cache operations to complete."""
        time.sleep(timeout_ms / 1000.0)


class UnifiedCacheTestBase(CacheTestBase):
    """
    Base class specifically for unified cache tests.
    
    Provides unified cache setup and specialized helper methods.
    """
    
    def setUp(self):
        """Set up unified cache test environment."""
        super().setUp()
        if not UNIFIED_CACHE_AVAILABLE:
            self.skipTest("Unified cache not available")
        
        # Create default unified cache instance
        self.cache = self.create_test_unified_cache()
    
    def test_cache_key_normalization(self):
        """Test that cache key normalization works correctly."""
        # Test hand normalization
        key1 = create_cache_key(["A♠", "K♠"], 2)
        key2 = create_cache_key(["AS", "KS"], 2)  # Legacy format
        key3 = create_cache_key(["K♠", "A♠"], 2)  # Different order
        
        # All should normalize to the same key
        self.assertEqual(key1.hero_hand, key2.hero_hand)
        self.assertEqual(key1.hero_hand, key3.hero_hand)
        self.assertEqual(key1.to_string(), key2.to_string())
        self.assertEqual(key1.to_string(), key3.to_string())
    
    def test_basic_cache_operations(self):
        """Test basic cache get/store operations."""
        key = self.create_test_cache_key()
        result = self.create_test_cache_result()
        
        # Test cache miss
        self.assert_cache_miss(self.cache, key)
        
        # Store result
        success = self.cache.store(key, result)
        self.assertTrue(success, "Cache store should succeed")
        
        # Test cache hit
        self.assert_cache_hit(self.cache, key, result)
        
        # Verify statistics
        self.assert_cache_stats_updated(self.cache, 
                                       expected_hits=1, 
                                       expected_misses=1, 
                                       expected_requests=2)


class LegacyCacheTestBase(CacheTestBase):
    """
    Base class for legacy cache compatibility tests.
    """
    
    def setUp(self):
        """Set up legacy cache test environment."""
        super().setUp()
        if not LEGACY_CACHE_AVAILABLE:
            self.skipTest("Legacy cache not available")
        
        # Create legacy cache instances
        self.hand_cache, self.board_cache, self.preflop_cache = self.create_test_legacy_cache()


class SolverCacheIntegrationTestBase(CacheTestBase):
    """
    Base class for solver cache integration tests.
    
    Tests cache behavior within the context of the full solver.
    """
    
    def setUp(self):
        """Set up solver with cache for integration testing."""
        super().setUp()
        if not SOLVER_AVAILABLE:
            self.skipTest("Solver not available for integration tests")
        
        # Create test config with fixed random seed
        import tempfile
        import json
        
        self.test_dir = tempfile.mkdtemp()
        self.config_path = f"{self.test_dir}/test_config.json"
        
        config = {
            "simulation_settings": {
                "random_seed": 42,  # Fixed seed for deterministic tests
                "fast_mode_simulations": 1000,
                "default_simulations": 10000,
                "max_workers": 2
            },
            "performance_settings": {
                "max_simulation_time_ms": 5000,
                "early_convergence_threshold": 0.001,
                "min_simulations_for_convergence": 1000,
                "timeout_fast_mode_ms": 3000,
                "timeout_default_mode_ms": 20000,
                "timeout_precision_mode_ms": 120000,
                "parallel_processing_threshold": 1000
            },
            "output_settings": {
                "decimal_precision": 4,
                "include_confidence_interval": True,
                "include_hand_categories": True
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        # Create solver with caching enabled
        self.solver = MonteCarloSolver(
            config_path=self.config_path,
            enable_caching=True
        )
        
        # Force cache initialization for testing
        self.solver._initialize_cache_if_needed()
    
    def tearDown(self):
        """Clean up solver resources."""
        if hasattr(self, 'solver'):
            self.solver.close()
        
        # Clean up temp directory
        if hasattr(self, 'test_dir'):
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
        
        super().tearDown()
    
    def run_deterministic_scenario(self, 
                                  hero_hand: List[str],
                                  num_opponents: int = 2,
                                  board_cards: List[str] = None,
                                  simulation_mode: str = "fast") -> Any:
        """Run a deterministic scenario for cache testing."""
        return self.solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=board_cards,
            simulation_mode=simulation_mode
        )
    
    def assert_solver_cache_behavior(self, 
                                   hero_hand: List[str],
                                   expected_cache_hit: bool = True):
        """Assert expected cache behavior in solver."""
        # Get initial cache stats
        initial_stats = self.solver.get_cache_stats()
        initial_hits = initial_stats.get('unified_cache', {}).get('cache_hits', 0) if initial_stats else 0
        
        # Run simulation twice with identical parameters
        result1 = self.run_deterministic_scenario(hero_hand)
        result2 = self.run_deterministic_scenario(hero_hand)
        
        # Get final cache stats
        final_stats = self.solver.get_cache_stats()
        final_hits = final_stats.get('unified_cache', {}).get('cache_hits', 0) if final_stats else 0
        
        if expected_cache_hit:
            # Second call should be a cache hit
            self.assertGreater(final_hits, initial_hits, 
                             "Expected cache hit on second identical call")
            # Results should be identical for cache hits
            self.assertEqual(result1.win_probability, result2.win_probability,
                           "Cache hits should return identical results")
        else:
            # Verify results are within statistical tolerance for cache misses
            tolerance = 0.02  # 2% tolerance for Monte Carlo variance
            self.assertAlmostEqual(result1.win_probability, result2.win_probability, 
                                 delta=tolerance,
                                 msg="Non-cached results should be within statistical tolerance")


class CachePerformanceTestBase(CacheTestBase):
    """
    Base class for cache performance tests.
    
    Provides timing utilities and performance assertion helpers.
    """
    
    def setUp(self):
        """Set up performance testing environment."""
        super().setUp()
        self.timing_tolerance_ms = 5.0  # 5ms tolerance for timing measurements
    
    def time_operation(self, operation_func, *args, **kwargs) -> Tuple[Any, float]:
        """Time an operation and return result and execution time in ms."""
        start_time = time.time()
        result = operation_func(*args, **kwargs)
        execution_time_ms = (time.time() - start_time) * 1000
        return result, execution_time_ms
    
    def assert_cache_speedup(self, 
                           cache_time_ms: float, 
                           no_cache_time_ms: float,
                           min_speedup: float = 2.0):
        """Assert that cache provides expected speedup."""
        if cache_time_ms < 1.0:  # Sub-millisecond operations are hard to measure
            self.skipTest("Cache operation too fast to measure reliably")
        
        speedup = no_cache_time_ms / cache_time_ms
        self.assertGreater(speedup, min_speedup,
                         f"Expected cache speedup >{min_speedup}x, got {speedup:.1f}x")
    
    def assert_timing_within_tolerance(self, 
                                     actual_time_ms: float,
                                     expected_time_ms: float,
                                     tolerance_factor: float = 2.0):
        """Assert timing is within expected tolerance."""
        max_allowed = expected_time_ms * tolerance_factor
        self.assertLess(actual_time_ms, max_allowed,
                       f"Operation took {actual_time_ms:.1f}ms, expected <{max_allowed:.1f}ms")


class ThreadSafetyCacheTestBase(CacheTestBase):
    """
    Base class for cache thread safety tests.
    """
    
    def setUp(self):
        """Set up thread safety testing environment."""
        super().setUp()
        self.num_threads = 4
        self.operations_per_thread = 100
        self.thread_results = []
        self.thread_exceptions = []
    
    def run_concurrent_operations(self, operation_func, *args, **kwargs):
        """Run operations concurrently across multiple threads."""
        def thread_worker(thread_id):
            try:
                for i in range(self.operations_per_thread):
                    result = operation_func(thread_id, i, *args, **kwargs)
                    self.thread_results.append((thread_id, i, result))
            except Exception as e:
                self.thread_exceptions.append((thread_id, e))
        
        # Start threads
        threads = []
        for thread_id in range(self.num_threads):
            thread = threading.Thread(target=thread_worker, args=(thread_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check for exceptions
        if self.thread_exceptions:
            raise Exception(f"Thread exceptions occurred: {self.thread_exceptions}")
    
    def assert_thread_safety(self, cache):
        """Assert that cache operations are thread-safe."""
        def cache_operation(thread_id, operation_id):
            key = self.create_test_cache_key(
                hero_hand=["A♠", "K♠"],
                num_opponents=thread_id + 1  # Vary by thread
            )
            result = self.create_test_cache_result()
            
            # Store and immediately retrieve
            cache.store(key, result)
            retrieved = cache.get(key)
            
            return retrieved is not None
        
        # Run concurrent operations
        self.run_concurrent_operations(cache_operation)
        
        # Verify no exceptions occurred
        self.assertEqual(len(self.thread_exceptions), 0, 
                        "Thread safety test should not raise exceptions")
        
        # Verify all operations completed
        expected_operations = self.num_threads * self.operations_per_thread
        self.assertEqual(len(self.thread_results), expected_operations,
                        "All thread operations should complete")


# Export base classes for use in other test files
__all__ = [
    'CacheTestBase',
    'UnifiedCacheTestBase', 
    'LegacyCacheTestBase',
    'SolverCacheIntegrationTestBase',
    'CachePerformanceTestBase',
    'ThreadSafetyCacheTestBase'
]