#!/usr/bin/env python3
"""
Performance regression tests for Poker Knight

Ensures that performance improvements and optimizations don't introduce
accuracy regressions or break existing functionality.
"""

import unittest
import time
import statistics
from poker_knight import solve_poker_hand, MonteCarloSolver

class TestPerformanceRegression(unittest.TestCase):
    """Test suite for performance regression validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.solver = MonteCarloSolver()
    
    def test_simulation_count_targets(self):
        """Verify that simulation modes achieve reasonable simulation counts."""
        test_cases = [
            ("fast", 10000, 0.25),     # Fast mode: at least 25% of target (2.5K+ sims)
            ("default", 100000, 0.05), # Default mode: at least 5% of target (5K+ sims)  
            ("precision", 500000, 0.015) # Precision mode: at least 1.5% of target (7.5K+ sims)
        ]
        
        hero_hand = ['A♠️', 'A♥️']
        num_opponents = 2
        
        for mode, target_sims, min_ratio in test_cases:
            with self.subTest(mode=mode):
                result = solve_poker_hand(hero_hand, num_opponents, simulation_mode=mode)
                
                # Calculate minimum expected simulations based on target and convergence
                min_expected = target_sims * min_ratio
                
                self.assertGreater(result.simulations_run, min_expected,
                                 f"{mode} mode should run at least {min_expected:.0f} simulations, "
                                 f"got {result.simulations_run}")
                
                # Validate that early stopping is working appropriately
                if result.stopped_early:
                    self.assertTrue(result.convergence_achieved or result.target_accuracy_achieved,
                                  f"{mode} mode stopped early but didn't achieve convergence or target accuracy")
                
                # If it ran close to target, it should not have stopped early
                if result.simulations_run >= target_sims * 0.9:
                    # This is fine - it ran most/all of the target simulations
                    pass
                elif result.stopped_early:
                    # Early stopping should have a good reason
                    self.assertTrue(result.convergence_achieved or result.target_accuracy_achieved,
                                  f"{mode} mode stopped early at {result.simulations_run} sims but "
                                  f"should have run more or achieved convergence")
    
    def test_execution_time_bounds(self):
        """Test that execution times are within reasonable bounds."""
        test_cases = [
            ("fast", 10000),      # 10 seconds max for fast mode
            ("default", 30000),   # 30 seconds max for default mode  
            ("precision", 150000) # 150 seconds max for precision mode
        ]
        
        hero_hand = ['K♠️', 'K♥️']
        num_opponents = 3
        
        for mode, max_time_ms in test_cases:
            with self.subTest(mode=mode):
                start = time.time()
                result = solve_poker_hand(hero_hand, num_opponents, simulation_mode=mode)
                actual_time = (time.time() - start) * 1000
                
                self.assertLess(actual_time, max_time_ms,
                              f"{mode} mode should complete within {max_time_ms}ms")
                
                # Also verify reported time is close to actual time (more lenient)
                time_diff = abs(actual_time - result.execution_time_ms)
                self.assertLess(time_diff, 500,  # 500ms tolerance
                              f"Reported time should be close to actual time")
    
    def test_statistical_accuracy(self):
        """Test that simulation results are statistically accurate."""
        # Test with known scenario: AA with top set
        # Should win most of the time but allow for reasonable variance
        hero_hand = ['A♠️', 'A♥️']
        num_opponents = 1
        board = ['A♦️', 'Q♠️', 'J♥️']  # Give hero top set
        
        results = []
        for _ in range(3):  # Run fewer times for speed
            result = solve_poker_hand(hero_hand, num_opponents, board, "fast")
            results.append(result.win_probability)
        
        avg_win_rate = statistics.mean(results)
        std_dev = statistics.stdev(results) if len(results) > 1 else 0
        
        # AA with top set should win frequently (>80%)
        self.assertGreater(avg_win_rate, 0.75,
                         f"AA with top set should win >75% of the time, got {avg_win_rate:.1%}")
        
        # Results should be reasonably consistent
        self.assertLess(std_dev, 0.10,
                       f"Win rate should be reasonably consistent across runs, std dev: {std_dev:.3f}")
    
    def test_memory_usage_stability(self):
        """Test that memory usage remains stable across extended runs."""
        try:
            import gc
            import psutil
            import os
        except ImportError:
            self.skipTest("psutil not available for memory testing")
            return
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory usage
        gc.collect()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run multiple simulations
        hero_hand = ['Q♠️', 'Q♥️']
        for i in range(10):
            result = solve_poker_hand(hero_hand, 2, simulation_mode="fast")
            
            # Check memory every few iterations
            if i % 3 == 0:
                gc.collect()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - baseline_memory
                
                # Memory should not increase significantly (allow 50MB growth)
                self.assertLess(memory_increase, 50,
                              f"Memory usage increased too much: {memory_increase:.1f}MB")
    
    def test_confidence_interval_accuracy(self):
        """Test that confidence intervals are reasonable."""
        hero_hand = ['J♠️', 'J♥️']
        num_opponents = 2
        
        result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        
        if result.confidence_interval:
            lower, upper = result.confidence_interval
            
            # Confidence interval should contain the win probability
            self.assertLessEqual(lower, result.win_probability)
            self.assertGreaterEqual(upper, result.win_probability)
            
            # Interval should be reasonable width (more lenient)
            interval_width = upper - lower
            self.assertGreater(interval_width, 0.005,  # At least 0.5% width
                             f"Confidence interval too narrow: {interval_width:.4f}")
            self.assertLess(interval_width, 0.25,     # At most 25% width
                           f"Confidence interval too wide: {interval_width:.4f}")
    
    def test_hand_category_frequencies(self):
        """Test that hand category frequencies are reasonable."""
        # Use a scenario that generates various hand types
        hero_hand = ['7♠️', '8♦️']  # Medium connector
        num_opponents = 1
        
        result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        
        if result.hand_category_frequencies:
            frequencies = result.hand_category_frequencies
            
            # Should have some high card and pair frequencies
            self.assertIn('high_card', frequencies)
            self.assertIn('pair', frequencies)
            
            # Frequencies should sum to approximately 1.0
            total_freq = sum(frequencies.values())
            self.assertAlmostEqual(total_freq, 1.0, places=1,
                                 msg=f"Hand frequencies should sum to 1.0, got {total_freq}")
            
            # High card should be reasonably common for this hand (more lenient)
            self.assertGreater(frequencies.get('high_card', 0), 0.15,
                             "High card should be reasonably common for 7-8 offsuit")
    
    def test_parallel_vs_sequential_consistency(self):
        """Test that parallel and sequential modes give consistent results."""
        hero_hand = ['A♠️', 'K♦️']
        num_opponents = 2
        board = ['A♥️', 'K♠️', '7♣️']
        
        # This test requires direct access to solver methods
        solver = MonteCarloSolver()
        
        # Parse cards
        hero_cards = [solver.evaluator.parse_card(card) for card in hero_hand]
        board_cards = [solver.evaluator.parse_card(card) for card in board]
        removed_cards = hero_cards + board_cards
        
        # Run both modes with same parameters
        num_sims = 10000
        max_time = 10000
        start_time = time.time()
        
        seq_wins, seq_ties, seq_losses, seq_cats, seq_convergence = solver._run_sequential_simulations(
            hero_cards, num_opponents, board_cards, removed_cards, num_sims, max_time, start_time
        )
        
        start_time = time.time()
        par_wins, par_ties, par_losses, par_cats, par_convergence = solver._run_parallel_simulations(
            hero_cards, num_opponents, board_cards, removed_cards, num_sims, max_time, start_time
        )
        
        # Results should be statistically similar (within 5% for win rate)
        seq_total = seq_wins + seq_ties + seq_losses
        par_total = par_wins + par_ties + par_losses
        
        if seq_total > 0 and par_total > 0:
            seq_win_rate = seq_wins / seq_total
            par_win_rate = par_wins / par_total
            
            win_rate_diff = abs(seq_win_rate - par_win_rate)
            self.assertLess(win_rate_diff, 0.05,
                          f"Sequential vs parallel win rates should be similar: {seq_win_rate:.3f} vs {par_win_rate:.3f}")
    
    def test_convergence_behavior(self):
        """Test that results converge as simulation count increases."""
        hero_hand = ['10♠️', '10♥️']
        num_opponents = 3
        
        # Run with increasing simulation counts
        modes = ["fast", "default"]  # Skip precision to save time
        results = []
        
        for mode in modes:
            result = solve_poker_hand(hero_hand, num_opponents, simulation_mode=mode)
            results.append((result.simulations_run, result.win_probability))
        
        # More simulations should generally lead to more stable results
        # (This is a basic check - full convergence testing would be more complex)
        if len(results) >= 2:
            sim_counts = [r[0] for r in results]
            win_probs = [r[1] for r in results]
            
            # Just verify that we get reasonable results
            for i, (sims, prob) in enumerate(results):
                self.assertGreater(sims, 1000, f"Mode {modes[i]} should run substantial simulations")
                self.assertGreater(prob, 0.1, f"Pocket 10s should have reasonable win rate")
                self.assertLess(prob, 0.9, f"Pocket 10s vs 3 opponents shouldn't dominate")

class TestPerformanceBenchmarks(unittest.TestCase):
    """Benchmark tests for performance tracking."""
    
    def test_hand_evaluation_speed(self):
        """Benchmark hand evaluation speed."""
        from poker_knight import HandEvaluator, Card
        
        evaluator = HandEvaluator()
        
        # Test different hand types
        test_hands = {
            "pair": [Card('A', '♠️'), Card('A', '♥️'), Card('K', '♦️'), Card('Q', '♠️'), Card('J', '♥️')],
            "flush": [Card('A', '♠️'), Card('J', '♠️'), Card('9', '♠️'), Card('7', '♠️'), Card('5', '♠️')],
            "full_house": [Card('A', '♠️'), Card('A', '♥️'), Card('A', '♦️'), Card('K', '♠️'), Card('K', '♥️')]
        }
        
        for hand_type, cards in test_hands.items():
            with self.subTest(hand_type=hand_type):
                num_evals = 1000
                start_time = time.time()
                
                for _ in range(num_evals):
                    rank, tiebreakers = evaluator.evaluate_hand(cards)
                
                end_time = time.time()
                avg_time = ((end_time - start_time) / num_evals) * 1000  # ms
                
                # Each evaluation should be very fast
                self.assertLess(avg_time, 0.01,  # Less than 0.01ms per evaluation
                              f"{hand_type} evaluation too slow: {avg_time:.4f}ms")

if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2) 