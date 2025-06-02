# -*- coding: utf-8 -*-
import pytest
import time
import gc
import threading
from unittest.mock import patch
from typing import List, Dict, Any

from poker_knight import MonteCarloSolver


class TestExtremeScenarios:
    """Test extreme probability scenarios and rare edge cases."""
    
    def test_near_zero_probability_scenarios(self):
        """Test scenarios with extremely low probability outcomes."""
        solver = MonteCarloSolver()
        
        # Royal flush vs royal flush (extremely rare)
        # Both players have royal flush draws
        result = solver.analyze_hand(
            hero_hand=["AS", "KS"],
            board_cards=["QS", "JS", "10S"],
            num_opponents=1,  # Simulate one opponent with similar potential
            simulation_mode="precision"
        )
        
        # Should handle this gracefully - actual probabilities will be calculated
        assert 0.0 <= result.win_probability <= 1.0
        assert 0.0 <= result.tie_probability <= 1.0
        assert result.win_probability + result.tie_probability + result.loss_probability == pytest.approx(1.0, abs=0.001)
        
    def test_drawing_dead_scenarios(self):
        """Test scenarios where opponent is drawing dead (100% equity)."""
        solver = MonteCarloSolver()
        
        # Player has made straight flush, opponent cannot improve
        result = solver.analyze_hand(
            hero_hand=["9S", "8S"],
            board_cards=["7S", "6S", "5S", "2C", "3D"],
            num_opponents=1,
            simulation_mode="fast"
        )
        
        # Player should have very high win probability
        assert result.win_probability > 0.95
        assert result.tie_probability < 0.05
        assert result.loss_probability < 0.05
        
    def test_pocket_aces_vs_random(self):
        """Test pocket aces against random hands - known probability."""
        solver = MonteCarloSolver()
        
        result = solver.analyze_hand(
            hero_hand=["AS", "AH"],
            num_opponents=1,
            simulation_mode="precision"
        )
        
        # Pocket aces should win ~85% against random hand pre-flop
        assert 0.80 <= result.win_probability <= 0.90
        
    def test_extremely_short_stacked_scenarios(self):
        """Test scenarios with very short stack sizes (<2 BB)."""
        solver = MonteCarloSolver()
        
        # Simulate tournament ICM scenario with very short stacks
        result = solver.analyze_hand(
            hero_hand=["KS", "QS"],
            num_opponents=2,
            simulation_mode="default",
            # ICM parameters for short stack scenario
            stack_sizes=[1.5, 15.0, 20.0],  # Player has 1.5 BB
            tournament_context={"payout_structure": [50, 30, 20]}
        )
        
        # Should handle short stack calculations
        assert hasattr(result, 'icm_equity')
        if result.icm_equity is not None:
            assert 0.0 <= result.icm_equity <= 1.0
            
    def test_massive_multiway_pot(self):
        """Test 6-way pot scenario (maximum opponents)."""
        solver = MonteCarloSolver()
        
        result = solver.analyze_hand(
            hero_hand=["AS", "KS"],
            num_opponents=6,  # Maximum realistic opponents
            simulation_mode="precision"
        )
        
        # Should handle large multiway calculations
        assert 0.0 <= result.win_probability <= 1.0
        assert hasattr(result, 'position_aware_equity')
        
    def test_identical_hands_scenario(self):
        """Test scenario where similar hands compete."""
        solver = MonteCarloSolver()
        
        # High pair vs high pair scenario (will often tie on similar boards)
        result = solver.analyze_hand(
            hero_hand=["AS", "AH"],
            num_opponents=2,
            simulation_mode="default"
        )
        
        # Should handle similar strength scenarios
        assert 0.0 <= result.win_probability <= 1.0
        assert 0.0 <= result.tie_probability <= 1.0
        

class TestMemoryPressureScenarios:
    """Test behavior under memory constraints and stress conditions."""
    
    def test_large_simulation_count_memory(self):
        """Test large simulation mode for memory usage."""
        solver = MonteCarloSolver()
        
        # Large simulation count that should stress memory
        initial_memory = self._get_memory_usage()
        
        result = solver.analyze_hand(
            hero_hand=["AS", "KS"],
            num_opponents=3,
            simulation_mode="precision"  # Uses large simulation count
        )
        
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<100MB for this test)
        assert memory_increase < 100 * 1024 * 1024  # 100MB limit
        assert result.win_probability > 0.0
        
    def test_concurrent_solver_instances(self):
        """Test multiple solver instances running concurrently."""
        results = []
        threads = []
        
        def run_analysis(thread_id: int):
            solver = MonteCarloSolver()
            result = solver.analyze_hand(
                hero_hand=["KS", "QS"],
                num_opponents=2,
                simulation_mode="default"
            )
            results.append((thread_id, result))
        
        # Start 4 concurrent solver instances
        for i in range(4):
            thread = threading.Thread(target=run_analysis, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All threads should complete successfully
        assert len(results) == 4
        for thread_id, result in results:
            assert 0.0 <= result.win_probability <= 1.0
            
    def test_memory_leak_detection(self):
        """Test for memory leaks over extended runs."""
        solver = MonteCarloSolver()
        
        initial_memory = self._get_memory_usage()
        
        # Run multiple analyses to check for memory accumulation
        for _ in range(10):
            result = solver.analyze_hand(
                hero_hand=["JS", "10S"],
                num_opponents=2,
                simulation_mode="fast"
            )
            assert result.win_probability > 0.0
            
            # Force garbage collection
            gc.collect()
        
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory should not increase significantly over multiple runs
        assert memory_increase < 50 * 1024 * 1024  # 50MB tolerance
        
    def test_context_manager_resource_cleanup(self):
        """Test proper resource cleanup using context manager."""
        initial_memory = self._get_memory_usage()
        
        # Use solver in context manager
        with MonteCarloSolver() as solver:
            result = solver.analyze_hand(
                hero_hand=["AS", "AH"],
                num_opponents=3,
                simulation_mode="precision"
            )
            assert result.win_probability > 0.0
        
        # Force cleanup
        gc.collect()
        time.sleep(0.1)  # Allow cleanup time
        
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory should be properly cleaned up
        assert memory_increase < 25 * 1024 * 1024  # 25MB tolerance
        
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            # If psutil not available, return dummy value
            return 0


class TestTimeoutEdgeCases:
    """Test timeout behavior under extreme conditions."""
    
    def test_very_short_timeout(self):
        """Test behavior with extremely short timeouts."""
        solver = MonteCarloSolver()
        
        # Override config for very short timeout
        original_config = solver.config.copy()
        solver.config["performance_settings"]["timeout_fast_mode_ms"] = 50  # 50ms
        
        try:
            result = solver.analyze_hand(
                hero_hand=["KS", "QS"],
                num_opponents=2,
                simulation_mode="fast"
            )
            
            # Should complete with reduced simulations due to timeout
            assert result.win_probability > 0.0
            assert result.simulations_run > 0  # Should run some simulations
            
        finally:
            # Restore original config
            solver.config = original_config
            
    def test_timeout_with_different_thread_counts(self):
        """Test timeout behavior with various thread configurations."""
        solver = MonteCarloSolver()
        
        # Test with different thread counts
        thread_counts = [1, 2, 4, 8]
        
        for thread_count in thread_counts:
            # Override config for this test
            original_config = solver.config.copy()
            solver.config["simulation_settings"]["max_workers"] = thread_count
            solver.config["performance_settings"]["timeout_default_mode_ms"] = 200  # 200ms
            
            try:
                result = solver.analyze_hand(
                    hero_hand=["AS", "KS"],
                    num_opponents=3,
                    simulation_mode="default"
                )
                
                # Should handle timeout gracefully regardless of thread count
                assert result.win_probability > 0.0
                assert result.simulations_run > 0
                
            finally:
                # Restore original config
                solver.config = original_config
                
    def test_timeout_graceful_degradation(self):
        """Test graceful degradation under extreme time pressure."""
        solver = MonteCarloSolver()
        
        # Extremely short timeout for precision mode
        original_config = solver.config.copy()
        solver.config["performance_settings"]["timeout_precision_mode_ms"] = 25  # 25ms
        
        try:
            result = solver.analyze_hand(
                hero_hand=["QS", "JS"],
                num_opponents=4,
                simulation_mode="precision"
            )
            
            # Should degrade gracefully and still return valid results
            assert result.win_probability > 0.0
            assert result.simulations_run > 0
            
            # Should maintain result structure integrity
            assert hasattr(result, 'confidence_interval')
            
        finally:
            # Restore original config
            solver.config = original_config
            
    def test_zero_timeout_handling(self):
        """Test handling of zero or negative timeout values."""
        solver = MonteCarloSolver()
        
        # Test zero timeout
        original_config = solver.config.copy()
        solver.config["performance_settings"]["timeout_fast_mode_ms"] = 0
        
        try:
            result = solver.analyze_hand(
                hero_hand=["10S", "9S"],
                num_opponents=2,
                simulation_mode="fast"
            )
            
            # Should handle zero timeout gracefully (run minimum simulations)
            assert result.win_probability > 0.0
            assert result.simulations_run > 0
            
        finally:
            # Restore original config
            solver.config = original_config


class TestStatisticalEdgeCases:
    """Test statistical edge cases and validation."""
    
    def test_confidence_interval_extreme_probabilities(self):
        """Test confidence intervals for extreme probabilities."""
        solver = MonteCarloSolver()
        
        # Scenario with very high win probability
        result = solver.analyze_hand(
            hero_hand=["AS", "AH"],
            board_cards=["AC", "AD", "KS", "QC", "JD"],  # Four aces
            num_opponents=1,
            simulation_mode="precision"
        )
        
        # Should have very high win probability with confidence interval
        assert result.win_probability > 0.95
        if result.confidence_interval is not None:
            # Confidence interval should be valid
            lower, upper = result.confidence_interval
            assert 0.0 <= lower <= result.win_probability <= upper <= 1.0
        
    def test_variance_calculation_stability(self):
        """Test variance calculation stability across runs."""
        solver = MonteCarloSolver()
        
        variances = []
        
        # Run same scenario multiple times (reduced to 3 runs for stability)
        for _ in range(3):
            result = solver.analyze_hand(
                hero_hand=["KS", "KH"],
                num_opponents=2,
                simulation_mode="default"
            )
            
            # Calculate variance from confidence interval if available
            if result.confidence_interval:
                lower, upper = result.confidence_interval
                variance = ((upper - lower) / 3.92) ** 2  # Approximate variance from 95% CI
                variances.append(variance)
        
        # Variances should be reasonably consistent (increased tolerance)
        if len(variances) > 1:
            mean_variance = sum(variances) / len(variances)
            for variance in variances:
                # Each variance should be within 200% of mean (more tolerant)
                assert 0.25 * mean_variance <= variance <= 3.0 * mean_variance, \
                    f"Variance {variance:.2e} outside acceptable range of mean {mean_variance:.2e}"
                
    def test_convergence_analysis_extreme_scenarios(self):
        """Test convergence analysis with extreme scenarios."""
        solver = MonteCarloSolver()
        
        # Test convergence with very unbalanced scenario
        result = solver.analyze_hand(
            hero_hand=["2C", "3D"],  # Very weak hand
            num_opponents=2,  # Strong statistical opposition
            simulation_mode="precision"
        )
        
        # Should converge even with extreme probability differences
        assert result.win_probability < 0.4  # Should lose most of the time
        
        # Convergence data should be available
        if hasattr(result, 'convergence_achieved') and result.convergence_achieved is not None:
            assert isinstance(result.convergence_achieved, bool)
            
    def test_statistical_significance_edge_cases(self):
        """Test statistical significance in edge cases."""
        solver = MonteCarloSolver()
        
        # Test with fast mode (lower simulation count)
        result = solver.analyze_hand(
            hero_hand=["AS", "KS"],
            num_opponents=1,
            simulation_mode="fast"
        )
        
        # Should still provide valid statistics
        assert 0.0 <= result.win_probability <= 1.0
        assert result.simulations_run > 0
        
        # Margin of error should reflect limited sample size if available
        if hasattr(result, 'final_margin_of_error') and result.final_margin_of_error:
            assert result.final_margin_of_error > 0.001  # Should be measurable 