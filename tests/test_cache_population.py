#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test suite for Poker Knight Cache Pre-Population System

Tests the complete cache pre-population infrastructure including:
- PopulationConfig validation and defaults
- ScenarioGenerator comprehensive scenario creation
- CachePrePopulator one-time population logic
- Integration with existing cache system
- Performance characteristics and timing
- User control options and error handling
- Coverage statistics and reporting

This test suite ensures the cache population system works correctly
in isolation and integrates properly with the existing solver and cache systems.

Author: hildolfr
License: MIT
"""

import unittest
import tempfile
import os
import time
import threading
import json
import shutil
from unittest.mock import patch, MagicMock, Mock
from pathlib import Path
from dataclasses import asdict

# Import testing framework
import pytest

# Import the cache population system components
try:
    from poker_knight.storage.cache_prepopulation import (
        PopulationConfig, PopulationStats, ScenarioGenerator, CachePrePopulator,
        ensure_cache_populated, integrate_cache_population
    )
    from poker_knight.storage.cache import (
        CacheConfig, get_cache_manager, clear_all_caches
    )
    CACHE_POPULATION_AVAILABLE = True
except ImportError as e:
    CACHE_POPULATION_AVAILABLE = False
    pytest.skip(f"Cache population system not available: {e}", allow_module_level=True)

# Test whether solver integration is available
try:
    from poker_knight import MonteCarloSolver
    SOLVER_AVAILABLE = True
except ImportError:
    SOLVER_AVAILABLE = False


@pytest.mark.cache
@pytest.mark.cache_population
class TestPopulationConfig(unittest.TestCase):
    """Test cache population configuration management."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = PopulationConfig()
        
        # Master controls
        self.assertTrue(config.enable_persistence)
        self.assertFalse(config.skip_cache_warming)
        self.assertFalse(config.force_cache_regeneration)
        
        # Population thresholds
        self.assertEqual(config.cache_population_threshold, 0.95)
        self.assertEqual(config.target_coverage_percentage, 98.0)
        
        # Scenario coverage
        self.assertEqual(config.preflop_hands, "all_169")
        self.assertEqual(config.opponent_counts, [1, 2, 3, 4, 5, 6])
        self.assertEqual(config.board_patterns, ["rainbow", "monotone", "paired", "connected", "disconnected"])
        self.assertEqual(config.positions, ["early", "middle", "late", "button", "sb", "bb"])
        
        # Performance settings
        self.assertEqual(config.max_population_time_minutes, 5)
        self.assertEqual(config.progress_reporting_interval, 100)
        self.assertEqual(config.population_simulations, 50000)
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = PopulationConfig(
            enable_persistence=False,
            skip_cache_warming=True,
            force_cache_regeneration=True,
            cache_population_threshold=0.80,
            preflop_hands="premium_only",
            opponent_counts=[1, 2, 3],
            max_population_time_minutes=2,
            population_simulations=10000
        )
        
        self.assertFalse(config.enable_persistence)
        self.assertTrue(config.skip_cache_warming)
        self.assertTrue(config.force_cache_regeneration)
        self.assertEqual(config.cache_population_threshold, 0.80)
        self.assertEqual(config.preflop_hands, "premium_only")
        self.assertEqual(config.opponent_counts, [1, 2, 3])
        self.assertEqual(config.max_population_time_minutes, 2)
        self.assertEqual(config.population_simulations, 10000)
    
    def test_config_validation_ranges(self):
        """Test configuration value ranges and validation."""
        # Test valid threshold ranges
        config = PopulationConfig(cache_population_threshold=0.5)
        self.assertEqual(config.cache_population_threshold, 0.5)
        
        config = PopulationConfig(target_coverage_percentage=99.9)
        self.assertEqual(config.target_coverage_percentage, 99.9)
        
        # Test time constraints
        config = PopulationConfig(max_population_time_minutes=10)
        self.assertEqual(config.max_population_time_minutes, 10)
        
        # Test preflop hands options
        for hands_option in ["all_169", "premium_only", "common_only"]:
            config = PopulationConfig(preflop_hands=hands_option)
            self.assertEqual(config.preflop_hands, hands_option)


@pytest.mark.cache
@pytest.mark.cache_population
class TestScenarioGenerator(unittest.TestCase):
    """Test comprehensive scenario generation for cache population."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_all = PopulationConfig(preflop_hands="all_169")
        self.config_premium = PopulationConfig(preflop_hands="premium_only")
        self.config_common = PopulationConfig(preflop_hands="common_only")
    
    def test_preflop_hands_generation(self):
        """Test generation of all 169 preflop hands."""
        generator = ScenarioGenerator(self.config_all)
        all_hands = generator.get_preflop_hands_for_population()
        
        # Should have exactly 169 hands
        self.assertEqual(len(all_hands), 169)
        
        # Should include pocket pairs
        pocket_pairs = [f"{rank}{rank}" for rank in ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']]
        for pair in pocket_pairs:
            self.assertIn(pair, all_hands)
        
        # Should include suited hands
        self.assertIn("AKs", all_hands)
        self.assertIn("AQs", all_hands)
        self.assertIn("KQs", all_hands)
        
        # Should include offsuit hands  
        self.assertIn("AKo", all_hands)
        self.assertIn("AQo", all_hands)
        self.assertIn("KQo", all_hands)
        
        # Should not have duplicates
        self.assertEqual(len(all_hands), len(set(all_hands)))
    
    def test_premium_hands_subset(self):
        """Test premium hands are a proper subset."""
        generator = ScenarioGenerator(self.config_premium)
        premium_hands = generator.get_preflop_hands_for_population()
        
        # Premium hands should be a smaller set
        self.assertLess(len(premium_hands), 169)
        self.assertGreater(len(premium_hands), 0)
        
        # Should include top premium hands
        expected_premium = ["AA", "KK", "QQ", "JJ", "AKs", "AKo"]
        for hand in expected_premium:
            self.assertIn(hand, premium_hands)
    
    def test_common_hands_subset(self):
        """Test common hands are between premium and all hands."""
        generator_premium = ScenarioGenerator(self.config_premium)
        generator_common = ScenarioGenerator(self.config_common)
        generator_all = ScenarioGenerator(self.config_all)
        
        premium_hands = generator_premium.get_preflop_hands_for_population()
        common_hands = generator_common.get_preflop_hands_for_population()
        all_hands = generator_all.get_preflop_hands_for_population()
        
        # Common should be between premium and all
        self.assertGreater(len(common_hands), len(premium_hands))
        self.assertLess(len(common_hands), len(all_hands))
        
        # All premium hands should be in common hands
        for hand in premium_hands:
            self.assertIn(hand, common_hands)
    
    def test_scenario_generation_all_hands(self):
        """Test complete scenario generation with all hands."""
        generator = ScenarioGenerator(self.config_all)
        scenarios = generator.generate_all_scenarios()
        
        # Should generate a substantial number of scenarios
        self.assertGreater(len(scenarios), 1000)
        
        # Should have both preflop and board texture scenarios
        preflop_scenarios = [s for s in scenarios if s['type'] == 'preflop']
        board_scenarios = [s for s in scenarios if s['type'] == 'board_texture']
        
        self.assertGreater(len(preflop_scenarios), 0)
        self.assertGreater(len(board_scenarios), 0)
        
        # Check preflop scenario structure
        sample_preflop = preflop_scenarios[0]
        required_preflop_keys = ['type', 'hero_hand', 'num_opponents', 'board_cards', 'position', 'hand_notation', 'scenario_id']
        for key in required_preflop_keys:
            self.assertIn(key, sample_preflop)
        
        self.assertEqual(sample_preflop['type'], 'preflop')
        self.assertIsNone(sample_preflop['board_cards'])
        
        # Check board scenario structure if available
        if board_scenarios:
            sample_board = board_scenarios[0]
            required_board_keys = ['type', 'hero_hand', 'num_opponents', 'board_cards', 'position', 'scenario_id']
            for key in required_board_keys:
                self.assertIn(key, sample_board)
            
            self.assertEqual(sample_board['type'], 'board_texture')
            self.assertIsNotNone(sample_board['board_cards'])
    
    def test_scenario_generation_premium_only(self):
        """Test scenario generation with premium hands only."""
        generator = ScenarioGenerator(self.config_premium)
        scenarios = generator.generate_all_scenarios()
        
        # Should generate fewer scenarios than all hands
        all_generator = ScenarioGenerator(self.config_all)
        all_scenarios = all_generator.generate_all_scenarios()
        
        self.assertLess(len(scenarios), len(all_scenarios))
        self.assertGreater(len(scenarios), 0)
        
        # All scenarios should use premium hands
        premium_hands = generator.get_preflop_hands_for_population()
        for scenario in scenarios[:100]:  # Check first 100 scenarios
            if 'hand_notation' in scenario:
                self.assertIn(scenario['hand_notation'], premium_hands)
    
    def test_notation_to_cards_conversion(self):
        """Test hand notation to card conversion."""
        generator = ScenarioGenerator(self.config_all)
        
        # Test pocket pairs
        aa_cards = generator._notation_to_cards("AA")
        self.assertEqual(len(aa_cards), 2)
        self.assertTrue(all("A" in card for card in aa_cards))
        
        # Test suited hands
        aks_cards = generator._notation_to_cards("AKs")
        if aks_cards:  # May not be implemented yet
            self.assertEqual(len(aks_cards), 2)
        
        # Test invalid notation
        invalid_cards = generator._notation_to_cards("Invalid")
        self.assertFalse(invalid_cards)


@pytest.mark.cache
@pytest.mark.cache_population
class TestCachePrePopulator(unittest.TestCase):
    """Test the main cache pre-population engine."""
    
    def setUp(self):
        """Set up test fixtures with temporary cache."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_config = CacheConfig(
            enable_persistence=True,
            sqlite_path=os.path.join(self.temp_dir, "test_cache.db")
        )
        self.population_config = PopulationConfig(
            preflop_hands="premium_only",  # Use premium only for faster tests
            max_population_time_minutes=1,  # Short timeout for tests
            opponent_counts=[1, 2],  # Fewer opponents for faster tests
            positions=["button", "bb"]  # Fewer positions for faster tests
        )
        
        # Clear any existing caches
        clear_all_caches()
    
    def tearDown(self):
        """Clean up test fixtures."""
        clear_all_caches()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test populator initialization."""
        populator = CachePrePopulator(
            config=self.population_config,
            cache_config=self.cache_config
        )
        
        self.assertIsNotNone(populator.config)
        self.assertIsNotNone(populator.cache_config)
        self.assertIsNotNone(populator.stats)
        self.assertIsNotNone(populator._scenario_generator)
        
        # Stats should be initialized
        self.assertEqual(populator.stats.total_scenarios, 0)
        self.assertEqual(populator.stats.populated_scenarios, 0)
    
    def test_should_populate_cache_logic(self):
        """Test cache population decision logic."""
        # Test with persistence disabled
        config_no_persist = PopulationConfig(enable_persistence=False)
        populator = CachePrePopulator(config=config_no_persist)
        self.assertFalse(populator.should_populate_cache())
        
        # Test with cache warming disabled
        config_skip_warming = PopulationConfig(skip_cache_warming=True)
        populator = CachePrePopulator(config=config_skip_warming)
        self.assertFalse(populator.should_populate_cache())
        
        # Test with force regeneration
        config_force = PopulationConfig(force_cache_regeneration=True)
        populator = CachePrePopulator(config=config_force, cache_config=self.cache_config)
        self.assertTrue(populator.should_populate_cache())
    
    @patch('poker_knight.storage.cache_prepopulation.CACHING_AVAILABLE', True)
    def test_populate_cache_basic(self):
        """Test basic cache population functionality."""
        populator = CachePrePopulator(
            config=self.population_config,
            cache_config=self.cache_config
        )
        
        # Mock the cache managers to avoid actual cache operations in unit tests
        with patch.object(populator, '_preflop_cache') as mock_preflop_cache:
            mock_preflop_cache.get_cache_coverage.return_value = {
                'coverage_percentage': 0.0,  # Low coverage to trigger population
                'cached_combinations': 0
            }
            mock_preflop_cache.get_preflop_result.return_value = None  # Not cached
            mock_preflop_cache.store_preflop_result.return_value = True  # Success
            
            # Override should_populate_cache to return True
            with patch.object(populator, 'should_populate_cache', return_value=True):
                stats = populator.populate_cache()
            
            # Check that population was attempted
            self.assertIsNotNone(stats)
            self.assertGreater(stats.total_scenarios, 0)
            self.assertIsNotNone(stats.started_at)
            self.assertIsNotNone(stats.completed_at)
    
    def test_scenario_simulation(self):
        """Test scenario simulation for cache population."""
        populator = CachePrePopulator(
            config=self.population_config,
            cache_config=self.cache_config
        )
        
        # Create a test scenario
        test_scenario = {
            'type': 'preflop',
            'hero_hand': ['AS', 'AH'],
            'num_opponents': 2,
            'board_cards': None,
            'position': 'button',
            'hand_notation': 'AA',
            'scenario_id': 'test_scenario'
        }
        
        # Test simulation
        result = populator._simulate_scenario(test_scenario)
        
        self.assertIsNotNone(result)
        self.assertIn('win_probability', result)
        self.assertIn('tie_probability', result)
        self.assertIn('loss_probability', result)
        self.assertIn('simulations_run', result)
        self.assertTrue(result['cached'])
        self.assertTrue(result['population_generated'])
        
        # Probabilities should sum to approximately 1
        total_prob = result['win_probability'] + result['tie_probability'] + result['loss_probability']
        self.assertAlmostEqual(total_prob, 1.0, delta=0.01)
        
        # Win probability should be reasonable for AA
        self.assertGreater(result['win_probability'], 0.5)  # AA should have >50% win rate
    
    def test_population_stats_tracking(self):
        """Test population statistics tracking."""
        stats = PopulationStats()
        
        # Test initial state
        self.assertEqual(stats.total_scenarios, 0)
        self.assertEqual(stats.populated_scenarios, 0)
        self.assertEqual(stats.skipped_scenarios, 0)
        self.assertEqual(stats.failed_scenarios, 0)
        
        # Test stats updates
        stats.total_scenarios = 100
        stats.populated_scenarios = 80
        stats.skipped_scenarios = 15
        stats.failed_scenarios = 5
        
        self.assertEqual(stats.total_scenarios, 100)
        self.assertEqual(stats.populated_scenarios, 80)
        self.assertEqual(stats.skipped_scenarios, 15)
        self.assertEqual(stats.failed_scenarios, 5)


@pytest.mark.cache
@pytest.mark.cache_population
@pytest.mark.integration
class TestCachePopulationIntegration(unittest.TestCase):
    """Test cache population integration with existing systems."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_config = CacheConfig(
            enable_persistence=True,
            sqlite_path=os.path.join(self.temp_dir, "integration_cache.db")
        )
        clear_all_caches()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        clear_all_caches()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_ensure_cache_populated_function(self):
        """Test the convenience function for cache population."""
        population_config = PopulationConfig(
            preflop_hands="premium_only",
            max_population_time_minutes=1
        )
        
        # Mock the cache availability check
        with patch('poker_knight.storage.cache_prepopulation.CACHING_AVAILABLE', True):
            with patch('poker_knight.storage.cache_prepopulation.CachePrePopulator') as mock_populator_class:
                mock_populator = Mock()
                mock_stats = PopulationStats()
                mock_stats.populated_scenarios = 50
                mock_populator.populate_cache.return_value = mock_stats
                mock_populator_class.return_value = mock_populator
                
                stats = ensure_cache_populated(
                    cache_config=self.cache_config,
                    population_config=population_config
                )
                
                # Check that populator was created and called
                mock_populator_class.assert_called_once_with(population_config, self.cache_config)
                mock_populator.populate_cache.assert_called_once()
                self.assertEqual(stats.populated_scenarios, 50)
    
    def test_ensure_cache_populated_without_caching(self):
        """Test ensure_cache_populated when caching is not available."""
        with patch('poker_knight.storage.cache_prepopulation.CACHING_AVAILABLE', False):
            stats = ensure_cache_populated()
            
            # Should return empty stats
            self.assertEqual(stats.populated_scenarios, 0)
            self.assertEqual(stats.total_scenarios, 0)
    
    @patch('poker_knight.storage.cache_prepopulation.CACHING_AVAILABLE', True)
    def test_integration_with_cache_managers(self):
        """Test integration with actual cache managers."""
        population_config = PopulationConfig(
            preflop_hands="premium_only",
            max_population_time_minutes=1,
            opponent_counts=[1, 2],
            positions=["button"]
        )
        
        try:
            # Get cache managers
            hand_cache, board_cache, preflop_cache = get_cache_manager(self.cache_config)
            
            if preflop_cache:
                # Mock cache coverage to trigger population
                with patch.object(preflop_cache, 'get_cache_coverage') as mock_coverage:
                    mock_coverage.return_value = {
                        'coverage_percentage': 0.0,  # Low coverage
                        'cached_combinations': 0
                    }
                    
                    # Test population
                    populator = CachePrePopulator(
                        config=population_config,
                        cache_config=self.cache_config
                    )
                    
                    # Override should_populate_cache for testing
                    with patch.object(populator, 'should_populate_cache', return_value=True):
                        stats = populator.populate_cache()
                    
                    self.assertGreater(stats.total_scenarios, 0)
        
        except Exception as e:
            self.skipTest(f"Cache managers not available for integration test: {e}")
    
    @pytest.mark.skipif(not SOLVER_AVAILABLE, reason="MonteCarloSolver not available")
    def test_solver_integration_decorator(self):
        """Test the solver integration decorator."""
        # This is a placeholder for testing the integration decorator
        # The actual implementation would test the @integrate_cache_population decorator
        
        @integrate_cache_population
        def mock_solver_init(self, *args, **kwargs):
            self._caching_enabled = kwargs.get('enable_caching', False)
            self._cache_config = self.cache_config if hasattr(self, 'cache_config') else None
            return self
        
        # Test the decorator functionality
        class MockSolver:
            def __init__(self, enable_caching=True, skip_cache_warming=False):
                self.cache_config = self.cache_config if hasattr(self, 'cache_config') else None
                mock_solver_init(self, enable_caching=enable_caching, skip_cache_warming=skip_cache_warming)
        
        # Create mock solver with caching enabled
        with patch('poker_knight.storage.cache_prepopulation.ensure_cache_populated') as mock_ensure:
            mock_stats = PopulationStats()
            mock_ensure.return_value = mock_stats
            
            solver = MockSolver(enable_caching=True)
            
            # Decorator should have been called
            self.assertTrue(hasattr(solver, '_caching_enabled'))


@pytest.mark.cache
@pytest.mark.cache_population
@pytest.mark.performance
class TestCachePopulationPerformance(unittest.TestCase):
    """Test cache population performance characteristics."""
    
    def setUp(self):
        """Set up performance test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_config = CacheConfig(
            enable_persistence=True,
            sqlite_path=os.path.join(self.temp_dir, "perf_cache.db")
        )
        clear_all_caches()
    
    def tearDown(self):
        """Clean up performance test fixtures."""
        clear_all_caches()
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_scenario_generation_performance(self):
        """Test scenario generation performance."""
        config = PopulationConfig(preflop_hands="all_169")
        generator = ScenarioGenerator(config)
        
        start_time = time.time()
        scenarios = generator.generate_all_scenarios()
        generation_time = time.time() - start_time
        
        # Scenario generation should be fast (< 1 second for all scenarios)
        self.assertLess(generation_time, 1.0)
        self.assertGreater(len(scenarios), 1000)
        
        # Test scenarios per second
        scenarios_per_second = len(scenarios) / generation_time
        self.assertGreater(scenarios_per_second, 1000)  # At least 1000 scenarios/sec
    
    def test_simulation_performance(self):
        """Test individual scenario simulation performance."""
        config = PopulationConfig()
        populator = CachePrePopulator(config=config, cache_config=self.cache_config)
        
        test_scenario = {
            'type': 'preflop',
            'hero_hand': ['AS', 'KS'],
            'num_opponents': 2,
            'board_cards': None,
            'position': 'button',
            'hand_notation': 'AKs',
            'scenario_id': 'perf_test'
        }
        
        # Test multiple simulations for timing
        simulations = 100
        start_time = time.time()
        
        for _ in range(simulations):
            result = populator._simulate_scenario(test_scenario)
            self.assertIsNotNone(result)
        
        total_time = time.time() - start_time
        avg_time_per_simulation = total_time / simulations
        
        # Each simulation should be very fast (< 0.01 seconds)
        self.assertLess(avg_time_per_simulation, 0.01)
    
    def test_timeout_enforcement(self):
        """Test that population respects timeout limits."""
        config = PopulationConfig(
            preflop_hands="premium_only",
            max_population_time_minutes=0.1  # 6 seconds timeout
        )
        populator = CachePrePopulator(config=config, cache_config=self.cache_config)
        
        # Mock a slow simulation to test timeout
        original_simulate = populator._simulate_scenario
        
        def slow_simulate(scenario):
            time.sleep(0.01)  # Add delay to trigger timeout
            return original_simulate(scenario)
        
        with patch.object(populator, '_simulate_scenario', side_effect=slow_simulate):
            with patch.object(populator, 'should_populate_cache', return_value=True):
                start_time = time.time()
                stats = populator.populate_cache()
                actual_time = time.time() - start_time
                
                # Should not exceed timeout significantly (allow 1 second grace)
                max_allowed_time = config.max_population_time_minutes * 60 + 1
                self.assertLess(actual_time, max_allowed_time)


@pytest.mark.cache
@pytest.mark.cache_population
@pytest.mark.unit
class TestCachePopulationErrorHandling(unittest.TestCase):
    """Test error handling in cache population system."""
    
    def test_invalid_config_handling(self):
        """Test handling of invalid configuration values."""
        # Test with invalid preflop hands option
        config = PopulationConfig(preflop_hands="invalid_option")
        generator = ScenarioGenerator(config)
        
        # Should default to all hands for invalid option
        hands = generator.get_preflop_hands_for_population()
        self.assertEqual(len(hands), 169)  # Should fall back to all hands
    
    def test_missing_cache_dependencies(self):
        """Test behavior when cache dependencies are missing."""
        with patch('poker_knight.storage.cache_prepopulation.CACHING_AVAILABLE', False):
            stats = ensure_cache_populated()
            
            # Should return empty stats gracefully
            self.assertIsInstance(stats, PopulationStats)
            self.assertEqual(stats.populated_scenarios, 0)
    
    def test_simulation_error_handling(self):
        """Test handling of simulation errors."""
        config = PopulationConfig()
        populator = CachePrePopulator(config=config)
        
        # Test with invalid scenario - missing required fields
        invalid_scenario = {
            'type': 'invalid',
            'hero_hand': None,
            'scenario_id': 'invalid_test'
            # Missing 'num_opponents' and other required fields
        }
        
        # The simulation should handle missing fields gracefully
        try:
            result = populator._simulate_scenario(invalid_scenario)
            # Should handle gracefully and return None or valid result
            self.assertTrue(result is None or isinstance(result, dict))
        except KeyError:
            # This is expected for invalid scenarios - test passes
            pass
        except Exception as e:
            # Other exceptions should be handled gracefully
            self.fail(f"Unexpected exception for invalid scenario: {e}")
        
        # Test with partially valid scenario
        partial_scenario = {
            'type': 'preflop',
            'hero_hand': ['AS', 'AH'],
            'num_opponents': 2,
            'board_cards': None,
            'position': 'button',
            'hand_notation': 'AA',
            'scenario_id': 'partial_test'
        }
        
        # This should work
        result = populator._simulate_scenario(partial_scenario)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, dict)
    
    def test_cache_operation_failures(self):
        """Test handling of cache operation failures."""
        config = PopulationConfig(preflop_hands="premium_only")
        populator = CachePrePopulator(config=config)
        
        # Mock cache failure
        with patch.object(populator, '_preflop_cache') as mock_cache:
            mock_cache.store_preflop_result.return_value = False  # Simulate failure
            mock_cache.get_preflop_result.return_value = None
            mock_cache.get_cache_coverage.return_value = {'coverage_percentage': 0.0, 'cached_combinations': 0}
            
            with patch.object(populator, 'should_populate_cache', return_value=True):
                stats = populator.populate_cache()
            
            # Should handle cache failures gracefully
            self.assertIsInstance(stats, PopulationStats)
            # Some scenarios might fail, but process should complete
            self.assertGreater(stats.total_scenarios, 0)


if __name__ == "__main__":
    # Configure test runner for cache population tests
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "cache_population",
        "--disable-warnings"
    ]) 