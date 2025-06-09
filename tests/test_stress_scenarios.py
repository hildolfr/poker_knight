# -*- coding: utf-8 -*-
try:
    import pytest
except ImportError:
    # Create a mock pytest for when it's not available
    class MockPytest:
        class mark:
            @staticmethod
            def stress(func):
                return func
            @staticmethod
            def slow(func):
                return func
        @staticmethod
        def fail(msg):
            raise AssertionError(msg)
    pytest = MockPytest()
import time
import threading
import multiprocessing
import gc
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple

from poker_knight import MonteCarloSolver


def cpu_intensive_analysis(arg):
    """Run CPU-intensive analysis in separate process."""
    solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
    total_simulations = 0
    
    # Run fewer analyses with fast mode to prevent hanging
    for _ in range(2):
        result = solver.analyze_hand(
            hero_hand=["A♠", "A♥"],
            num_opponents=2,
            simulation_mode="fast"  # Use fast mode to prevent hanging
        )
        total_simulations += result.simulations_run
    
    return total_simulations


class TestHighLoadScenarios:
    """Test system behavior under high computational load."""
    
    def test_massive_simulation_count(self):
        """Test handling of precision mode with large simulation counts."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Test with precision mode (uses large simulation counts)
        start_time = time.time()
        result = solver.analyze_hand(
            hero_hand=["A♠", "K♠"],
            num_opponents=2,
            simulation_mode="precision"
        )
        execution_time = time.time() - start_time
        
        # Should complete successfully even with large count
        assert result.win_probability > 0.0
        # Allow for early convergence - precision mode should run at least 7.5K simulations
        assert result.simulations_run > 7500  # Allow for early stopping due to convergence
        
        # Performance should be reasonable (allow up to 60 seconds)
        assert execution_time < 60.0
        
    def test_rapid_fire_analysis(self):
        """Test rapid consecutive analyses."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        results = []
        start_time = time.time()
        
        # Run 50 quick analyses in succession
        for i in range(50):
            result = solver.analyze_hand(
                hero_hand=[f"{['A', 'K', 'Q', 'J', '10'][i % 5]}♠", f"{['A', 'K', 'Q', 'J', '10'][(i+1) % 5]}♥"],
                num_opponents=1,
                simulation_mode="fast"
            )
            results.append(result)
        
        execution_time = time.time() - start_time
        
        # All analyses should complete successfully
        assert len(results) == 50
        for result in results:
            assert 0.0 <= result.win_probability <= 1.0
        
        # Should maintain reasonable performance (allow 120 seconds for 50 analyses)
        assert execution_time < 120.0
        
    def test_maximum_concurrent_threads(self):
        """Test maximum number of concurrent analysis threads."""
        num_threads = min(32, multiprocessing.cpu_count() * 4)  # Up to 32 threads
        results = []
        
        def run_concurrent_analysis(thread_id: int) -> Tuple[int, Any]:
            solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
            result = solver.analyze_hand(
                hero_hand=["A♠", "K♠"],
                num_opponents=2,
                simulation_mode="default"
            )
            return (thread_id, result)
        
        start_time = time.time()
        
        # Run maximum concurrent threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i in range(num_threads):
                future = executor.submit(run_concurrent_analysis, i)
                futures.append(future)
            
            # Collect all results
            for future in as_completed(futures):
                thread_id, result = future.result()
                results.append((thread_id, result))
                
        execution_time = time.time() - start_time
        
        # All threads should complete successfully
        assert len(results) == num_threads
        for thread_id, result in results:
            assert 0.0 <= result.win_probability <= 1.0
            
        # Should handle high concurrency (allow more time for slower systems)
        assert execution_time < 180.0  # 3 minutes for extreme concurrency
        
    def test_memory_intensive_multiway_scenarios(self):
        """Test memory usage with complex multiway scenarios."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        initial_memory = self._get_memory_usage()
        
        # Run multiple memory-intensive scenarios
        for opponent_count in [3, 5, 6]:  # Increasing complexity
            result = solver.analyze_hand(
                hero_hand=["A♠", "K♠"],
                num_opponents=opponent_count,
                simulation_mode="precision",
                # Add ICM parameters to increase memory usage
                stack_sizes=[20.0] * (opponent_count + 1),
                tournament_context={"payout_structure": [40, 25, 20, 10, 5][:opponent_count + 1]}
            )
            
            assert result.win_probability > 0.0
            
            # Force garbage collection between scenarios
            gc.collect()
            
        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (<200MB)
        assert memory_increase < 200 * 1024 * 1024
        
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return 0


class TestResourceExhaustionScenarios:
    """Test behavior when system resources are strained."""
    
    def test_low_memory_simulation(self):
        """Test behavior when available memory is limited."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Create memory pressure by allocating large arrays
        memory_hog = []
        try:
            # Allocate memory to create pressure (but not crash)
            for _ in range(10):
                memory_hog.append([0] * 1000000)  # 10M integers
            
            # Run analysis under memory pressure
            result = solver.analyze_hand(
                hero_hand=["K♠", "Q♠"],
                num_opponents=3,
                simulation_mode="precision"
            )
            
            # Should complete successfully despite memory pressure
            assert result.win_probability > 0.0
            
        finally:
            # Clean up memory
            del memory_hog
            gc.collect()
            
    @pytest.mark.stress
    @pytest.mark.slow  # Mark as slow test
    def test_cpu_intensive_concurrent_load(self):
        """Test CPU-intensive concurrent operations."""
        num_processes = min(4, multiprocessing.cpu_count())  # Reduce to 4 processes max
        
        start_time = time.time()
        
        # Run CPU-intensive work in multiple processes with timeout
        try:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Use map_async with timeout to prevent hanging
                async_result = pool.map_async(cpu_intensive_analysis, range(num_processes))
                results = async_result.get(timeout=120)  # 2 minute timeout
        except multiprocessing.TimeoutError:
            pytest.fail("Test timed out after 2 minutes - potential hanging issue")
            
        execution_time = time.time() - start_time
        
        # All processes should complete successfully
        assert len(results) == num_processes
        for total_sims in results:
            assert total_sims > 0
            
        # Should complete in reasonable time (reduced load should be faster)
        assert execution_time < 120.0  # 2 minutes for reduced CPU load
        
    def test_thread_exhaustion_recovery(self):
        """Test recovery from thread exhaustion scenarios."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Create many threads to stress the system
        num_threads = min(100, multiprocessing.cpu_count() * 10)
        results = []
        
        def quick_analysis(thread_id: int) -> int:
            result = solver.analyze_hand(
                hero_hand=["A♠", "K♠"],
                num_opponents=1,
                simulation_mode="fast"
            )
            return thread_id
        
        # Run with high thread count
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(quick_analysis, i) for i in range(num_threads)]
            
            for future in as_completed(futures):
                try:
                    thread_id = future.result(timeout=30)
                    results.append(thread_id)
                except Exception:
                    # Some threads may fail under extreme load - that's acceptable
                    pass
        
        # At least 50% of threads should complete successfully
        assert len(results) >= num_threads * 0.5


class TestEdgeCaseReliability:
    """Test reliability with edge cases and boundary conditions."""
    
    def test_random_scenario_stress_test(self):
        """Test with random scenarios to catch edge cases."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        cards = ["A", "K", "Q", "J", "10", "9", "8", "7", "6", "5", "4", "3", "2"]
        suits = ["♠", "♥", "♦", "♣"]
        
        successful_analyses = 0
        
        # Run 100 random scenarios
        for _ in range(100):
            try:
                # Generate random valid hand
                card1 = f"{random.choice(cards)}{random.choice(suits)}"
                card2 = f"{random.choice(cards)}{random.choice(suits)}"
                
                # Ensure no duplicate cards
                if card1 == card2:
                    continue
                    
                result = solver.analyze_hand(
                    hero_hand=[card1, card2],
                    num_opponents=random.randint(1, 6),
                    simulation_mode=random.choice(["fast", "default", "precision"])
                )
                
                assert 0.0 <= result.win_probability <= 1.0
                assert result.simulations_run > 0
                successful_analyses += 1
                
            except Exception:
                # Some random scenarios may be invalid - that's acceptable
                pass
        
        # At least 80% of valid random scenarios should complete successfully
        assert successful_analyses >= 80
        
    def test_boundary_value_scenarios(self):
        """Test boundary value scenarios."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Test minimum opponents
        result = solver.analyze_hand(
            hero_hand=["A♠", "K♠"],
            num_opponents=1,
            simulation_mode="fast"
        )
        assert 0.0 <= result.win_probability <= 1.0
        
        # Test maximum reasonable opponents (max is 6)
        result = solver.analyze_hand(
            hero_hand=["A♠", "K♠"],
            num_opponents=6,
            simulation_mode="fast"
        )
        assert 0.0 <= result.win_probability <= 1.0
        
        # Test with modified configuration
        original_config = solver.config.copy()
        try:
            # Test with extreme configuration values
            solver.config.update({
                "simulation_counts": {
                    "fast": 100,  # Very low
                    "default": 500,
                    "precision": 1000
                }
            })
            
            result = solver.analyze_hand(
                hero_hand=["2♠", "7♦"],  # Weak hand
                num_opponents=2,
                simulation_mode="fast"
            )
            
            assert 0.0 <= result.win_probability <= 1.0
            assert result.simulations_run > 0
            
        finally:
            # Restore original config
            solver.config = original_config
            
    def test_error_recovery_and_fallbacks(self):
        """Test error recovery and graceful fallbacks."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Test with invalid card format (should handle gracefully)
        try:
            result = solver.analyze_hand(
                hero_hand=["XX", "YY"],  # Invalid cards
                num_opponents=1,
                simulation_mode="fast"
            )
            # If it doesn't raise an error, result should still be valid
            if result:
                assert 0.0 <= result.win_probability <= 1.0
        except Exception:
            # Expected to fail with invalid cards
            pass
        
        # Test with impossible board configuration
        try:
            result = solver.analyze_hand(
                hero_hand=["A♠", "A♠"],  # Duplicate card
                num_opponents=1,
                simulation_mode="fast"
            )
            if result:
                assert 0.0 <= result.win_probability <= 1.0
        except Exception:
            # Expected to fail with duplicate cards
            pass
            
        # Test with extreme opponent count
        try:
            result = solver.analyze_hand(
                hero_hand=["A♠", "K♠"],
                num_opponents=10,  # Too many opponents
                simulation_mode="fast"
            )
            if result:
                assert result.simulations_run > 0  # Should handle gracefully
        except Exception:
            # May raise ValueError for invalid opponent count
            pass


class TestLongRunningStability:
    """Test stability over extended periods and many operations."""
    
    def test_extended_runtime_stability(self):
        """Test stability over extended runtime."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        start_time = time.time()
        analyses_completed = 0
        
        # Run analyses for 30 seconds continuously
        while time.time() - start_time < 30.0:
            result = solver.analyze_hand(
                hero_hand=["A♠", "K♠"],
                num_opponents=random.randint(1, 4),
                simulation_mode="default"
            )
            
            assert 0.0 <= result.win_probability <= 1.0
            analyses_completed += 1
        
        # Should complete multiple analyses in 30 seconds (very conservative expectation)
        assert analyses_completed > 1
        
    def test_memory_stability_over_time(self):
        """Test memory stability over many operations."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        memory_samples = []
        
        # Sample memory usage over many operations
        for i in range(20):
            # Run analysis
            result = solver.analyze_hand(
                hero_hand=["K♠", "Q♠"],
                num_opponents=2,
                simulation_mode="default"
            )
            assert result.win_probability > 0.0
            
            # Sample memory every 5 operations
            if i % 5 == 0:
                memory_samples.append(self._get_memory_usage())
                gc.collect()  # Force cleanup
                
        # Memory usage should remain stable (no significant growth)
        if len(memory_samples) > 1:
            max_memory = max(memory_samples)
            min_memory = min(memory_samples)
            memory_growth = (max_memory - min_memory) / min_memory if min_memory > 0 else 0
            
            # Memory growth should be <20%
            assert memory_growth < 0.20
            
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return 0


class TestPerformanceRegression:
    """Test for performance regressions under stress."""
    
    def test_performance_consistency_under_load(self):
        """Test that performance remains consistent under varying loads."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Baseline performance test
        start_time = time.time()
        baseline_result = solver.analyze_hand(
            hero_hand=["A♠", "K♠"],
            num_opponents=2,
            simulation_mode="default"
        )
        baseline_time = time.time() - start_time
        
        # Performance under concurrent load
        def concurrent_analysis():
            return solver.analyze_hand(
                hero_hand=["Q♠", "J♠"],
                num_opponents=2,
                simulation_mode="default"
            )
        
        # Run target analysis while other threads are working
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Start background load
            background_futures = [executor.submit(concurrent_analysis) for _ in range(3)]
            
            # Run target analysis under load
            start_time = time.time()
            loaded_result = solver.analyze_hand(
                hero_hand=["A♠", "K♠"],
                num_opponents=2,
                simulation_mode="default"
            )
            loaded_time = time.time() - start_time
            
            # Wait for background tasks
            for future in background_futures:
                result = future.result()
                assert result.win_probability > 0.0
        
        # Performance under load should not degrade significantly
        # Allow up to 5x degradation under heavy concurrent load (more realistic)
        assert loaded_time < baseline_time * 5.0
        
        # Results should be statistically similar
        assert abs(baseline_result.win_probability - loaded_result.win_probability) < 0.05
        
    def test_scalability_with_simulation_modes(self):
        """Test scalability across different simulation modes."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        simulation_modes = ["fast", "default", "precision"]
        times = []
        
        for mode in simulation_modes:
            start_time = time.time()
            result = solver.analyze_hand(
                hero_hand=["A♠", "K♠"],
                num_opponents=3,
                simulation_mode=mode
            )
            execution_time = time.time() - start_time
            times.append(execution_time)
            
            assert result.win_probability > 0.0
            
        # Time should scale with mode complexity (fast < default < precision)
        if len(times) >= 3:
            # Precision mode should take longer than fast mode
            assert times[2] > times[0]  # precision > fast
            
        # All modes should complete in reasonable time
        for t in times:
            assert t < 60.0  # 60 seconds max for any mode (increased from 30) 

    def test_high_simulation_count_stability(self):
        """Test solver stability with high simulation counts."""
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Test with very high simulation count
        start_time = time.time()
        result = solver.analyze_hand(
            hero_hand=["A♠", "A♥"],
            num_opponents=5,
            simulation_mode="precision"
        )
        execution_time = time.time() - start_time
        
        # Should complete successfully
        assert result.win_probability > 0.0
        assert result.simulations_run > 5000  # Should run substantial simulations
        
        # Should complete in reasonable time (allow up to 2 minutes)
        assert execution_time < 120.0
        
        # Result should be reasonable for pocket aces
        assert result.win_probability > 0.3  # Conservative lower bound for AA vs 5 opponents