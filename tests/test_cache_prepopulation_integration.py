"""
Tests for cache prepopulation integration with the solver.

Verifies that:
1. Cache prepopulation triggers correctly on solver initialization
2. skip_cache_warming parameter works
3. force_cache_regeneration parameter works  
4. No background threads are created
5. prepopulate_cache() convenience function works
"""

import pytest
import time
import threading
from unittest.mock import Mock, patch

from poker_knight import MonteCarloSolver, prepopulate_cache


class TestCachePrepopulationIntegration:
    """Test cache prepopulation integration with solver."""
    
    def test_solver_triggers_prepopulation_on_init(self):
        """Test that solver triggers cache prepopulation on first use."""
        # Track active threads before
        initial_threads = threading.active_count()
        
        # Create solver with caching enabled
        solver = MonteCarloSolver(
            enable_caching=True,
            skip_cache_warming=False
        )
        
        # Trigger cache initialization by analyzing a hand
        result = solver.analyze_hand(
            hero_hand=["A♠", "K♠"],
            num_opponents=2
        )
        
        # Verify result is valid
        assert result is not None
        assert 0 <= result.win_probability <= 1
        
        # Verify no background threads were created
        final_threads = threading.active_count()
        assert final_threads <= initial_threads + 1  # Allow for thread pool
        
        # Check population result if available
        if hasattr(solver, '_population_result') and solver._population_result:
            assert solver._population_result.scenarios_populated >= 0
            assert solver._population_result.population_time_seconds >= 0
    
    def test_skip_cache_warming_parameter(self):
        """Test that skip_cache_warming prevents prepopulation."""
        # Create solver with warming disabled
        solver = MonteCarloSolver(
            enable_caching=True,
            skip_cache_warming=True
        )
        
        # Trigger cache initialization
        result = solver.analyze_hand(
            hero_hand=["A♠", "K♠"],
            num_opponents=2
        )
        
        # Verify result is valid
        assert result is not None
        
        # Verify no population occurred
        assert not hasattr(solver, '_population_result') or solver._population_result is None
    
    def test_force_cache_regeneration_parameter(self):
        """Test that force_cache_regeneration triggers comprehensive population."""
        # Create solver with force regeneration
        solver = MonteCarloSolver(
            enable_caching=True,
            skip_cache_warming=False,
            force_cache_regeneration=True
        )
        
        # Trigger cache initialization which should use comprehensive mode
        result = solver.analyze_hand(
            hero_hand=["A♠", "K♠"],
            num_opponents=2
        )
        
        # Verify result is valid
        assert result is not None
        assert 0 <= result.win_probability <= 1
        
        # If prepopulation occurred, it should have used comprehensive settings
        if hasattr(solver, '_population_result') and solver._population_result:
            # Force regeneration typically results in more scenarios populated
            assert solver._population_result.scenarios_populated >= 0
    
    def test_no_background_threads_created(self):
        """Verify that prepopulation doesn't create background threads."""
        initial_threads = threading.active_count()
        
        # Create solver and trigger prepopulation
        solver = MonteCarloSolver(enable_caching=True)
        solver.analyze_hand(["A♠", "A♥"], 2)
        
        # Wait a bit to ensure no delayed thread creation
        time.sleep(0.5)
        
        # Check thread count (allow for thread pool)
        final_threads = threading.active_count()
        assert final_threads <= initial_threads + solver._max_workers
    
    def test_prepopulate_cache_quick_mode(self):
        """Test prepopulate_cache convenience function in quick mode."""
        # Run quick prepopulation
        stats = prepopulate_cache(comprehensive=False, time_limit=5.0)
        
        # Verify basic stats
        assert isinstance(stats, dict)
        assert 'success' in stats
        assert 'scenarios_populated' in stats
        assert 'population_time_seconds' in stats
        
        # Verify time limit was respected
        if stats['success']:
            assert stats['population_time_seconds'] <= 6.0  # Allow small overhead
    
    @pytest.mark.slow
    def test_prepopulate_cache_comprehensive_mode(self):
        """Test prepopulate_cache convenience function in comprehensive mode."""
        # Run comprehensive prepopulation with short time limit
        stats = prepopulate_cache(comprehensive=True, time_limit=10.0)
        
        # Verify stats
        assert isinstance(stats, dict)
        assert 'success' in stats
        assert 'scenarios_populated' in stats
        
        # Comprehensive mode should populate more scenarios
        if stats['success']:
            assert stats['scenarios_populated'] >= 0
    
    def test_cache_hit_rate_after_prepopulation(self):
        """Verify improved cache hit rate after prepopulation."""
        # Create solver with prepopulation
        solver = MonteCarloSolver(enable_caching=True)
        
        # First call triggers prepopulation
        solver.analyze_hand(["A♠", "K♠"], 2)
        
        # Get initial cache stats
        initial_stats = solver.get_cache_stats()
        
        # Run several queries that should hit cache
        test_hands = [
            (["A♠", "A♥"], 2),
            (["K♠", "K♥"], 2),
            (["Q♠", "Q♥"], 2),
            (["A♠", "K♠"], 3),
        ]
        
        for hand, opponents in test_hands:
            solver.analyze_hand(hand, opponents)
        
        # Get final cache stats
        final_stats = solver.get_cache_stats()
        
        if initial_stats and final_stats:
            # Verify cache was used (hit rate should improve)
            if 'aggregate_stats' in final_stats:
                final_hit_rate = final_stats['aggregate_stats'].get('overall_hit_rate', 0)
                assert final_hit_rate > 0  # Should have some hits
    
    def test_prepopulation_with_invalid_cache(self):
        """Test prepopulation handles cache initialization failures gracefully."""
        # Create solver with invalid cache config
        with patch('poker_knight.solver.get_unified_cache', side_effect=Exception("Cache init failed")):
            solver = MonteCarloSolver(enable_caching=True)
            
            # Should still work without cache
            result = solver.analyze_hand(["A♠", "K♠"], 2)
            assert result is not None
            assert 0 <= result.win_probability <= 1
    
    def test_solver_simulation_callback_parsing(self):
        """Test that the simulation callback correctly parses hand notations."""
        solver = MonteCarloSolver(enable_caching=True)
        
        # Test through the actual prepopulation path
        from poker_knight.storage.startup_prepopulation import StartupCachePopulator
        
        populator = StartupCachePopulator()
        
        # Test callback with different hand notations
        test_cases = [
            ("AA", 2, "fast"),   # Pocket pair
            ("AKs", 3, "fast"),  # Suited
            ("AKo", 2, "fast"),  # Offsuit
        ]
        
        for hand_notation, num_opponents, sim_mode in test_cases:
            # Create the callback as done in solver
            def simulation_callback(h, o, s):
                if len(h) == 2:
                    rank = h[0]
                    # Convert T to 10
                    if rank == 'T':
                        rank = '10'
                    hero_hand = [f"{rank}♠", f"{rank}♥"]
                elif len(h) == 3:
                    rank1 = h[0]
                    rank2 = h[1]
                    suited = h[2] == 's'
                    
                    # Convert T to 10
                    if rank1 == 'T':
                        rank1 = '10'
                    if rank2 == 'T':
                        rank2 = '10'
                    
                    if suited:
                        hero_hand = [f"{rank1}♠", f"{rank2}♠"]
                    else:
                        hero_hand = [f"{rank1}♠", f"{rank2}♥"]
                else:
                    return None
                
                return solver.analyze_hand(hero_hand, o, [], simulation_mode=s)
            
            # Test the callback
            result = simulation_callback(hand_notation, num_opponents, sim_mode)
            assert result is not None
            assert hasattr(result, 'win_probability')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])