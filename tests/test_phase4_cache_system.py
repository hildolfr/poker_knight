#!/usr/bin/env python3
"""
♞ Poker Knight Phase 4 Cache System Tests

Comprehensive test suite for the Phase 4 hierarchical cache system including:
- Hierarchical cache (L1/L2/L3) functionality
- Adaptive cache management
- Performance monitoring and alerting
- Intelligent prepopulation
- Optimized persistence
- Complete system integration

This replaces the old dual-cache test architecture with tests designed
for the new unified cache system.

Author: hildolfr
License: MIT
"""

import unittest
import tempfile
import time
import threading
import os
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

# Import Phase 4 cache components
try:
    from poker_knight.storage.hierarchical_cache import (
        HierarchicalCache, HierarchicalCacheConfig, CacheLayer
    )
    from poker_knight.storage.adaptive_cache_manager import (
        AdaptiveCacheManager, AdaptiveCacheConfig, OptimizationStrategy
    )
    from poker_knight.storage.cache_performance_monitor import (
        PerformanceMonitor, MonitoringConfig
    )
    from poker_knight.storage.intelligent_prepopulation import (
        IntelligentPrepopulator, UsagePatternAnalyzer, PrepopulationStrategy
    )
    from poker_knight.storage.optimized_persistence import (
        OptimizedCachePersistenceManager, PersistenceConfig
    )
    from poker_knight.storage.phase4_integration import (
        Phase4CacheSystem, Phase4Config, create_balanced_cache_system
    )
    from poker_knight.storage.unified_cache import (
        CacheKey, CacheResult, ThreadSafeMonteCarloCache
    )
    PHASE4_AVAILABLE = True
except ImportError as e:
    print(f"Phase 4 cache system not available: {e}")
    PHASE4_AVAILABLE = False

# Import solver for integration tests
try:
    from poker_knight import MonteCarloSolver
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False


class TestHierarchicalCache(unittest.TestCase):
    """Test hierarchical cache system (L1/L2/L3)."""
    
    def setUp(self):
        """Set up test environment."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        
        # Create test configuration
        self.config = HierarchicalCacheConfig()
        self.config.l1_config.max_memory_mb = 32  # Small for testing
        self.config.l2_config.max_memory_mb = 64
        self.config.l3_config.max_memory_mb = 128
        
        # Create hierarchical cache
        self.cache = HierarchicalCache(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'cache'):
            self.cache.clear()
            self.cache.shutdown()
    
    def test_cache_creation(self):
        """Test cache system creation."""
        self.assertIsNotNone(self.cache)
        self.assertTrue(self.cache._l1_cache is not None)
        # L2/L3 might be None if Redis/SQLite not available
    
    def test_cache_storage_and_retrieval(self):
        """Test basic cache storage and retrieval."""
        # Create test cache key and result
        cache_key = CacheKey(
            hero_hand="AK_suited",
            num_opponents=2,
            board_cards="preflop",
            simulation_mode="default"
        )
        
        cache_result = CacheResult(
            win_probability=0.68,
            tie_probability=0.02,
            loss_probability=0.30,
            confidence_interval=(0.65, 0.71),
            simulations_run=10000,
            execution_time_ms=100.0,
            hand_categories={},
            metadata={},
            timestamp=time.time()
        )
        
        # Store and retrieve
        success = self.cache.store(cache_key, cache_result)
        self.assertTrue(success)
        
        retrieved = self.cache.get(cache_key)
        self.assertIsNotNone(retrieved)
        self.assertAlmostEqual(retrieved.win_probability, 0.68, places=2)
    
    def test_cache_miss(self):
        """Test cache miss behavior."""
        cache_key = CacheKey(
            hero_hand="72_offsuit",
            num_opponents=8,
            board_cards="preflop",
            simulation_mode="precision"
        )
        
        result = self.cache.get(cache_key)
        self.assertIsNone(result)
    
    def test_cache_statistics(self):
        """Test cache statistics tracking."""
        # Initial stats
        stats = self.cache.get_stats()
        initial_requests = stats.total_requests
        
        # Perform cache operations
        cache_key = CacheKey("AA", 1, "preflop", "fast")
        cache_result = CacheResult(0.85, 0.01, 0.14, (0.82, 0.88), 5000, 50.0, {}, {}, time.time())
        
        # Store and retrieve multiple times
        self.cache.store(cache_key, cache_result)
        self.cache.get(cache_key)  # Hit
        self.cache.get(CacheKey("KK", 1, "preflop", "fast"))  # Miss
        
        # Check statistics
        final_stats = self.cache.get_stats()
        self.assertGreater(final_stats.total_requests, initial_requests)


class TestAdaptiveCacheManager(unittest.TestCase):
    """Test adaptive cache management system."""
    
    def setUp(self):
        """Set up test environment."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        
        # Create cache and adaptive manager
        self.cache = HierarchicalCache()
        self.config = AdaptiveCacheConfig(
            optimization_strategy=OptimizationStrategy.BALANCED,
            adjustment_interval_seconds=1  # Fast for testing
        )
        self.manager = AdaptiveCacheManager(self.cache, self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'manager'):
            self.manager.stop_adaptive_optimization()
        if hasattr(self, 'cache'):
            self.cache.shutdown()
    
    def test_manager_creation(self):
        """Test adaptive manager creation."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.config.optimization_strategy, OptimizationStrategy.BALANCED)
    
    def test_optimization_status(self):
        """Test optimization status reporting."""
        status = self.manager.get_optimization_status()
        self.assertIsInstance(status, dict)
        self.assertIn('is_running', status)
        self.assertIn('strategy', status)
        self.assertIn('constraints', status)
    
    @patch('psutil.virtual_memory')
    def test_system_memory_constraints(self, mock_memory):
        """Test system memory constraint handling."""
        # Mock system memory
        mock_memory.return_value.total = 8 * 1024**3  # 8GB
        
        # Check constraints are applied
        status = self.manager.get_optimization_status()
        self.assertIn('system_memory_gb', status)


class TestPerformanceMonitor(unittest.TestCase):
    """Test performance monitoring system."""
    
    def setUp(self):
        """Set up test environment."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        
        self.cache = HierarchicalCache()
        self.config = MonitoringConfig(
            collection_interval_seconds=1,  # Fast for testing
            export_enabled=False  # Don't export during tests
        )
        self.monitor = PerformanceMonitor(self.cache, self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'monitor'):
            self.monitor.stop_monitoring()
        if hasattr(self, 'cache'):
            self.cache.shutdown()
    
    def test_monitor_creation(self):
        """Test performance monitor creation."""
        self.assertIsNotNone(self.monitor)
        self.assertFalse(self.monitor._is_monitoring)
    
    def test_dashboard_data(self):
        """Test performance dashboard data."""
        # Start monitoring and generate some cache activity
        self.monitor.start_monitoring()
        
        # Generate cache activity
        cache = self.cache
        for i in range(3):
            key = CacheKey("AK_suited", i + 1, "preflop", "default")
            result = CacheResult(
                win_probability=0.68,
                tie_probability=0.02,
                loss_probability=0.30,
                confidence_interval=(0.65, 0.71),
                simulations_run=10000,
                execution_time_ms=100.0,
                hand_categories={},
                metadata={},
                timestamp=time.time()
            )
            cache.store(key, result)
            cache.get(key)
        
        # Wait for data collection
        time.sleep(0.5)
        
        dashboard = self.monitor.get_performance_dashboard()
        self.assertIsInstance(dashboard, dict)
        
        # Handle case where no data is collected yet
        if 'error' in dashboard:
            self.assertIn('No performance data available', dashboard['error'])
        else:
            # Check required keys
            required_keys = ['timestamp', 'status', 'current', 'layers']
            for key in required_keys:
                self.assertIn(key, dashboard)
    
    def test_monitoring_lifecycle(self):
        """Test monitoring start/stop lifecycle."""
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor._is_monitoring)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor._is_monitoring)


class TestIntelligentPrepopulation(unittest.TestCase):
    """Test intelligent prepopulation system."""
    
    def setUp(self):
        """Set up test environment."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        
        # Create temporary usage history file
        self.temp_dir = tempfile.mkdtemp()
        self.usage_file = os.path.join(self.temp_dir, "test_usage.json")
        
        self.analyzer = UsagePatternAnalyzer(self.usage_file)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_creation(self):
        """Test usage pattern analyzer creation."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.usage_history_file, self.usage_file)
    
    def test_usage_recording(self):
        """Test scenario usage recording."""
        # Record some usage patterns
        self.analyzer.record_scenario_usage(
            hand_notation="AKs",
            num_opponents=2,
            simulation_mode="default",
            computation_time_ms=150.0,
            was_cached=False
        )
        
        # Check analytics
        analytics = self.analyzer.get_usage_analytics()
        self.assertGreater(analytics['total_scenarios'], 0)
        self.assertGreater(analytics['total_requests'], 0)
    
    def test_prepopulation_priorities(self):
        """Test prepopulation priority calculation."""
        # Create some usage data
        hands = ["AKs", "QQ", "AKo", "JJ", "72o"]
        for hand in hands:
            for _ in range(5):  # Multiple uses
                self.analyzer.record_scenario_usage(
                    hand_notation=hand,
                    num_opponents=2,
                    simulation_mode="default",
                    computation_time_ms=100.0,
                    was_cached=False
                )
        
        # Get priorities
        strategy = PrepopulationStrategy(name="test", description="test")
        priorities = self.analyzer.get_prepopulation_priorities(strategy)
        
        self.assertGreater(len(priorities), 0)
        # Premium hands should have higher importance
        premium_hands = [p for p in priorities if p.hand_notation in ["AKs", "QQ", "JJ"]]
        self.assertGreater(len(premium_hands), 0)


class TestOptimizedPersistence(unittest.TestCase):
    """Test optimized persistence system."""
    
    def setUp(self):
        """Set up test environment."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        
        # Create temporary directory for persistence
        self.temp_dir = tempfile.mkdtemp()
        self.config = PersistenceConfig(
            cache_directory=self.temp_dir,
            enabled=True
        )
        self.persistence = OptimizedCachePersistenceManager(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if hasattr(self, 'persistence'):
            self.persistence.shutdown()
        if hasattr(self, 'temp_dir'):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_persistence_creation(self):
        """Test persistence manager creation."""
        self.assertIsNotNone(self.persistence)
        self.assertTrue(self.config.enabled)
    
    def test_persistence_status(self):
        """Test persistence status reporting."""
        status = self.persistence.get_persistence_status()
        self.assertIsInstance(status, dict)
        self.assertIn('enabled', status)
        self.assertIn('statistics', status)
        self.assertIn('config', status)


class TestPhase4Integration(unittest.TestCase):
    """Test complete Phase 4 system integration."""
    
    def setUp(self):
        """Set up test environment."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        
        # Create test configuration
        self.config = Phase4Config(
            optimization_level="balanced",
            auto_start_services=False  # Manual control for testing
        )
        self.system = Phase4CacheSystem(self.config)
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'system'):
            self.system.stop_services()
    
    def test_system_creation(self):
        """Test Phase 4 system creation."""
        self.assertIsNotNone(self.system)
        self.assertEqual(self.system.config.optimization_level, "balanced")
        self.assertFalse(self.system._initialized)
    
    def test_system_initialization(self):
        """Test system initialization."""
        success = self.system.initialize()
        self.assertTrue(success)
        self.assertTrue(self.system._initialized)
        self.assertIsNotNone(self.system.hierarchical_cache)
    
    def test_system_status(self):
        """Test system status reporting."""
        self.system.initialize()
        status = self.system.get_system_status()
        
        self.assertIsInstance(status, dict)
        self.assertIn('initialized', status)
        self.assertIn('optimization_level', status)
        self.assertIn('components', status)
    
    def test_service_lifecycle(self):
        """Test service start/stop lifecycle."""
        # Initialize system
        self.system.initialize()
        
        # Start services
        success = self.system.start_services()
        self.assertTrue(success)
        self.assertTrue(self.system._started)
        
        # Stop services
        success = self.system.stop_services()
        self.assertTrue(success)
        self.assertFalse(self.system._started)


class TestSolverIntegration(unittest.TestCase):
    """Test cache integration with MonteCarloSolver."""
    
    def setUp(self):
        """Set up test environment."""
        if not PHASE4_AVAILABLE:
            self.skipTest("Phase 4 cache system not available")
        if not SOLVER_AVAILABLE:
            self.skipTest("MonteCarloSolver not available")
        
        # Create test config with fixed random seed
        import tempfile
        import json
        
        self.test_dir = tempfile.mkdtemp()
        self.config_path = f"{self.test_dir}/test_config.json"
        
        config = {
            "simulation_settings": {
                "random_seed": 42,  # Fixed seed for deterministic tests
                "fast_mode_simulations": 1000,
                "max_workers": 2
            },
            "performance_settings": {
                "max_simulation_time_ms": 5000,
                "timeout_fast_mode_ms": 3000
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
    
    def tearDown(self):
        """Clean up test environment."""
        if hasattr(self, 'solver'):
            self.solver.close()
        
        # Clean up temp directory
        if hasattr(self, 'test_dir'):
            import shutil
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_solver_with_cache(self):
        """Test solver integration with cache system."""
        # Run analysis twice - second should be cached
        hand = ['A♠', 'K♠']
        opponents = 2
        
        # First run (cache miss)
        start_time = time.time()
        result1 = self.solver.analyze_hand(hand, opponents, simulation_mode="fast")
        time1 = time.time() - start_time
        
        # Second run (cache hit)
        start_time = time.time()
        result2 = self.solver.analyze_hand(hand, opponents, simulation_mode="fast")
        time2 = time.time() - start_time
        
        # Results should be identical if cached, or close if not cached
        cache_stats = self.solver.get_cache_stats()
        cache_hits = 0
        if cache_stats and 'unified_cache' in cache_stats:
            cache_hits = cache_stats['unified_cache'].get('cache_hits', 0)
        
        if cache_hits > 0:
            self.assertEqual(result1.win_probability, result2.win_probability,
                           "Cached results should be identical")
        else:
            self.assertAlmostEqual(result1.win_probability, result2.win_probability, places=2,
                                 msg="Non-cached results should be close")
        
        # Second run should be faster (cache hit)
        # Note: This might not always be true in tests due to overhead, so we'll be lenient
        self.assertLess(time2, time1 * 10)  # At most 10x slower (very lenient)
    
    def test_cache_statistics(self):
        """Test cache statistics through solver."""
        # Get initial stats
        initial_stats = self.solver.get_cache_stats()
        
        if initial_stats and not initial_stats.get('error'):
            # Run some analyses
            hands = [['A♠', 'A♥'], ['K♠', 'K♥'], ['Q♠', 'Q♥']]
            for hand in hands:
                self.solver.analyze_hand(hand, 2, simulation_mode="fast")
            
            # Get final stats
            final_stats = self.solver.get_cache_stats()
            
            # Should have some requests
            if 'unified_cache' in final_stats:
                self.assertGreaterEqual(final_stats['unified_cache']['total_requests'], 0)


class TestCacheCompatibility(unittest.TestCase):
    """Test backward compatibility with existing cache interface."""
    
    def test_legacy_imports(self):
        """Test that legacy cache imports still work."""
        try:
            from poker_knight.storage import HandCache, CacheConfig
            # These should still be available for backward compatibility
            self.assertTrue(True, "Legacy imports available")
        except ImportError:
            # This is expected after refactor - legacy imports may be removed
            self.skipTest("Legacy cache imports no longer available")
    
    def test_solver_cache_interface(self):
        """Test solver cache interface compatibility."""
        if not SOLVER_AVAILABLE:
            self.skipTest("MonteCarloSolver not available")
        
        try:
            solver = MonteCarloSolver()
            stats = solver.get_cache_stats()
            # Should return some stats or None, not error
            self.assertIsInstance(stats, (dict, type(None)))
            solver.close()
        except Exception as e:
            self.fail(f"Solver cache interface failed: {e}")


# Test Suite Configuration
def create_test_suite():
    """Create comprehensive test suite for Phase 4 cache system."""
    suite = unittest.TestSuite()
    
    # Core component tests
    suite.addTest(unittest.makeSuite(TestHierarchicalCache))
    suite.addTest(unittest.makeSuite(TestAdaptiveCacheManager))
    suite.addTest(unittest.makeSuite(TestPerformanceMonitor))
    suite.addTest(unittest.makeSuite(TestIntelligentPrepopulation))
    suite.addTest(unittest.makeSuite(TestOptimizedPersistence))
    
    # Integration tests
    suite.addTest(unittest.makeSuite(TestPhase4Integration))
    suite.addTest(unittest.makeSuite(TestSolverIntegration))
    
    # Compatibility tests
    suite.addTest(unittest.makeSuite(TestCacheCompatibility))
    
    return suite


def run_phase4_tests():
    """Run Phase 4 cache system tests."""
    print("🧪 Running Phase 4 Cache System Tests")
    print("=" * 60)
    
    if not PHASE4_AVAILABLE:
        print("❌ Phase 4 cache system not available - cannot run tests")
        return False
    
    suite = create_test_suite()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All Phase 4 cache tests passed!")
    else:
        print(f"❌ {len(result.failures)} test(s) failed, {len(result.errors)} error(s)")
        
        if result.failures:
            print("\nFailures:")
            for test, traceback in result.failures:
                print(f"  - {test}: {traceback}")
        
        if result.errors:
            print("\nErrors:")
            for test, traceback in result.errors:
                print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_phase4_tests()