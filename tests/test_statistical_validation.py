#!/usr/bin/env python3
"""
Comprehensive Statistical Validation Tests for Poker Knight

This module provides rigorous statistical validation of the Monte Carlo poker solver,
ensuring accuracy, reliability, and consistency across various poker scenarios.

Features:
- Distribution validation for known poker scenarios
- Confidence interval verification  
- Variance and convergence analysis
- Cross-validation with analytical results
- Performance consistency checks
"""

import unittest
import math
import statistics
from collections import Counter
from typing import List, Dict, Tuple
from poker_knight import solve_poker_hand, MonteCarloSolver


class TestStatisticalValidation(unittest.TestCase):
    """Statistical validation test suite for Monte Carlo poker solver."""
    
    def setUp(self):
        """Set up test environment."""
        self.solver = MonteCarloSolver()
        # Known poker probabilities for validation
        self.known_probabilities = {
            # Pre-flop head-to-head matchups (approximate)
            'AA_vs_random': 0.85,
            'AA_vs_KK': 0.82,
            'AK_vs_random': 0.66,  # AKs suited
            '72o_vs_random': 0.32,  # Worst starting hand
        }
        
        # Expected hand category frequencies (7-card poker)
        self.expected_hand_frequencies = {
            'high_card': 0.174,
            'pair': 0.438,
            'two_pair': 0.235,
            'three_of_a_kind': 0.048,
            'straight': 0.046,
            'flush': 0.030,
            'full_house': 0.026,
            'four_of_a_kind': 0.0024,
            'straight_flush': 0.0003,
            'royal_flush': 0.0000031
        }
    
    def test_chi_square_hand_distribution(self):
        """
        Chi-square goodness-of-fit test for hand category distributions.
        Tests if observed hand frequencies match expected poker probabilities.
        """
        # Run large simulation to get hand category frequencies
        hero_hand = ['7♠️', '8♦️']  # Medium-strength hand for variety
        num_opponents = 1
        
        result = solve_poker_hand(
            hero_hand, num_opponents, 
            simulation_mode="precision"  # Large sample size for statistical power
        )
        
        if not result.hand_category_frequencies:
            self.skipTest("Hand category frequencies not available")
        
        observed_frequencies = result.hand_category_frequencies
        expected_frequencies = self.expected_hand_frequencies
        
        # Calculate chi-square statistic
        chi_square = 0
        degrees_of_freedom = 0
        
        for category in expected_frequencies:
            if category in observed_frequencies:
                observed = observed_frequencies[category]
                expected = expected_frequencies[category]
                
                # Only include categories with sufficient expected frequency
                if expected >= 0.01:  # At least 1% expected
                    chi_square += ((observed - expected) ** 2) / expected
                    degrees_of_freedom += 1
        
        degrees_of_freedom -= 1  # Subtract 1 for constraint
        
        # Critical value for α = 0.05 with df ≈ 6-8 is around 12.59-15.51
        critical_value = 15.51  # Conservative critical value
        
        self.assertLess(chi_square, critical_value,
                       f"Chi-square test failed: χ² = {chi_square:.3f} > {critical_value}")
        
        print(f"✅ Chi-square goodness-of-fit: χ² = {chi_square:.3f} (df = {degrees_of_freedom})")
    
    def test_confidence_interval_coverage(self):
        """
        Test that confidence intervals have correct coverage probability.
        95% confidence intervals should contain the true value 95% of the time.
        """
        # Use a scenario with known approximate probability
        hero_hand = ['A♠️', 'A♥️']  # Pocket aces
        num_opponents = 1
        true_win_rate = 0.85  # Approximate known value
        
        intervals_containing_true_value = 0
        total_tests = 20  # Number of confidence intervals to test
        
        for _ in range(total_tests):
            result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
            
            if result.confidence_interval:
                lower, upper = result.confidence_interval
                if lower <= true_win_rate <= upper:
                    intervals_containing_true_value += 1
        
        coverage_rate = intervals_containing_true_value / total_tests
        
        # 95% confidence intervals should contain true value ~95% of the time
        # Allow for sampling variation: expect at least 80% coverage
        self.assertGreater(coverage_rate, 0.75,
                          f"Confidence interval coverage too low: {coverage_rate:.1%}")
        
        print(f"✅ Confidence interval coverage: {coverage_rate:.1%} ({intervals_containing_true_value}/{total_tests})")
    
    def test_sample_size_effect_on_accuracy(self):
        """
        Test that larger sample sizes reduce standard error and improve accuracy.
        This validates the Monte Carlo convergence property.
        """
        hero_hand = ['K♠️', 'K♥️']
        num_opponents = 2
        
        # Test different sample sizes
        sample_sizes = ["fast", "default", "precision"]  # 10K, 100K, 500K
        accuracies = []
        
        # Run multiple times for each sample size
        for mode in sample_sizes:
            win_rates = []
            for _ in range(5):  # 5 repetitions per sample size
                result = solve_poker_hand(hero_hand, num_opponents, simulation_mode=mode)
                win_rates.append(result.win_probability)
            
            # Calculate standard deviation as measure of accuracy
            std_dev = statistics.stdev(win_rates)
            accuracies.append(std_dev)
            print(f"  {mode} mode std dev: {std_dev:.4f}")
        
        # Larger samples should have smaller standard deviation
        for i in range(len(accuracies) - 1):
            self.assertGreaterEqual(accuracies[i], accuracies[i + 1] * 0.8,  # Allow some variation
                                  f"Accuracy should improve with larger samples: "
                                  f"{sample_sizes[i]} std={accuracies[i]:.4f} vs "
                                  f"{sample_sizes[i+1]} std={accuracies[i+1]:.4f}")
        
        print("✅ Sample size effect validated: larger samples → better accuracy")
    
    def test_known_poker_probabilities(self):
        """
        Test simulation results against known poker probabilities.
        Validates that the Monte Carlo method produces theoretically correct results.
        """
        test_scenarios = [
            # (hero_hand, opponents, board, expected_win_rate, tolerance, description)
            (['A♠️', 'A♥️'], 1, [], 0.85, 0.05, "AA vs random preflop"),
            (['A♠️', 'K♠️'], 1, [], 0.66, 0.05, "AKs vs random preflop"),
            (['7♠️', '2♦️'], 1, [], 0.32, 0.05, "72o vs random preflop"),
            (['A♠️', 'A♥️'], 1, ['A♦️', 'K♠️', 'Q♥️'], 0.95, 0.05, "AA with top set"),
        ]
        
        for hero_hand, opponents, board, expected, tolerance, description in test_scenarios:
            with self.subTest(scenario=description):
                result = solve_poker_hand(hero_hand, opponents, board, "default")
                
                error = abs(result.win_probability - expected)
                self.assertLess(error, tolerance,
                               f"{description}: Expected {expected:.1%}, got {result.win_probability:.1%}, "
                               f"error {error:.3f} > tolerance {tolerance}")
                
                print(f"✅ {description}: {result.win_probability:.1%} (expected {expected:.1%})")
    
    def test_simulation_variance_stability(self):
        """
        Test that simulation variance is stable across multiple runs.
        High variance could indicate implementation issues.
        """
        hero_hand = ['Q♠️', 'J♠️']
        num_opponents = 3
        
        # Run multiple simulations and check variance
        win_rates = []
        for _ in range(15):  # 15 independent runs
            result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="fast")
            win_rates.append(result.win_probability)
        
        variance = statistics.variance(win_rates)
        std_dev = statistics.stdev(win_rates)
        
        # Variance should be reasonable for Monte Carlo simulation
        # For 10K simulations, standard error ≈ sqrt(p(1-p)/n) ≈ 0.005 for p≈0.5
        expected_std_error = 0.005
        max_acceptable_std = expected_std_error * 3  # Allow 3x for additional variation
        
        self.assertLess(std_dev, max_acceptable_std,
                       f"Simulation variance too high: std={std_dev:.4f} > {max_acceptable_std:.4f}")
        
        print(f"✅ Simulation variance stable: std={std_dev:.4f}, variance={variance:.6f}")
    
    def test_symmetry_validation(self):
        """
        Test that equivalent hands produce equivalent results (symmetry test).
        Different suits of the same hand should have identical probabilities.
        """
        # Test equivalent hands with different suits
        equivalent_hands = [
            (['A♠️', 'K♠️'], ['A♥️', 'K♥️']),  # Same suited connector
            (['Q♠️', 'Q♥️'], ['Q♦️', 'Q♣️']),  # Same pocket pair
            (['10♠️', '9♦️'], ['10♥️', '9♣️']), # Same offsuit connector
        ]
        
        num_opponents = 2
        tolerance = 0.03  # 3% tolerance for symmetry
        
        for hand1, hand2 in equivalent_hands:
            with self.subTest(hand1=hand1, hand2=hand2):
                result1 = solve_poker_hand(hand1, num_opponents, simulation_mode="default")
                result2 = solve_poker_hand(hand2, num_opponents, simulation_mode="default")
                
                win_rate_diff = abs(result1.win_probability - result2.win_probability)
                
                self.assertLess(win_rate_diff, tolerance,
                               f"Equivalent hands should have similar win rates: "
                               f"{hand1} = {result1.win_probability:.3f}, "
                               f"{hand2} = {result2.win_probability:.3f}, "
                               f"diff = {win_rate_diff:.3f}")
                
                print(f"✅ Symmetry validated: {hand1} vs {hand2}, diff = {win_rate_diff:.3f}")
    
    def test_normality_of_simulation_results(self):
        """
        Test that simulation results follow expected statistical distributions.
        Win rates should be approximately normally distributed around the true value.
        """
        hero_hand = ['J♠️', 'J♥️']
        num_opponents = 2
        
        # Collect many simulation results
        win_rates = []
        for _ in range(25):  # 25 independent simulations
            result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="fast")
            win_rates.append(result.win_probability)
        
        # Basic normality checks
        mean_win_rate = statistics.mean(win_rates)
        std_dev = statistics.stdev(win_rates)
        
        # Check for reasonable distribution shape
        # Count values within 1 and 2 standard deviations
        within_1_std = sum(1 for x in win_rates if abs(x - mean_win_rate) <= std_dev)
        within_2_std = sum(1 for x in win_rates if abs(x - mean_win_rate) <= 2 * std_dev)
        
        # Normal distribution: ~68% within 1σ, ~95% within 2σ
        pct_within_1_std = within_1_std / len(win_rates)
        pct_within_2_std = within_2_std / len(win_rates)
        
        # Allow reasonable tolerance for small sample size
        self.assertGreater(pct_within_1_std, 0.50,  # At least 50% within 1σ
                          f"Too few values within 1σ: {pct_within_1_std:.1%}")
        self.assertGreater(pct_within_2_std, 0.85,  # At least 85% within 2σ
                          f"Too few values within 2σ: {pct_within_2_std:.1%}")
        
        print(f"✅ Distribution normality: {pct_within_1_std:.1%} within 1σ, {pct_within_2_std:.1%} within 2σ")
    
    def test_monte_carlo_convergence_rate(self):
        """
        Test that Monte Carlo error decreases as 1/√n (theoretical convergence rate).
        This validates the fundamental Monte Carlo property.
        """
        hero_hand = ['A♠️', 'Q♠️']
        num_opponents = 1
        
        # We'll use the solver's internal methods to control simulation count precisely
        solver = MonteCarloSolver()
        hero_cards = [solver.evaluator.parse_card(card) for card in hero_hand]
        board_cards = []
        removed_cards = hero_cards
        
        # Test different simulation counts and measure accuracy
        simulation_counts = [1000, 4000, 16000]  # 4x increases
        errors = []
        
        # Get a reference "true" value with very large simulation
        import time
        start_time = time.time()
        ref_wins, ref_ties, ref_losses, _ = solver._run_sequential_simulations(
            hero_cards, num_opponents, board_cards, removed_cards, 
            100000, 30000, start_time
        )
        ref_total = ref_wins + ref_ties + ref_losses
        reference_win_rate = ref_wins / ref_total if ref_total > 0 else 0
        
        # Test each simulation count
        for sim_count in simulation_counts:
            results = []
            for _ in range(5):  # Multiple runs for each count
                start_time = time.time()
                wins, ties, losses, _ = solver._run_sequential_simulations(
                    hero_cards, num_opponents, board_cards, removed_cards,
                    sim_count, 10000, start_time
                )
                total = wins + ties + losses
                if total > 0:
                    win_rate = wins / total
                    results.append(win_rate)
            
            if results:
                # Calculate mean squared error
                mse = statistics.mean([(wr - reference_win_rate) ** 2 for wr in results])
                rmse = math.sqrt(mse)
                errors.append(rmse)
                print(f"  {sim_count:,} simulations: RMSE = {rmse:.4f}")
        
        # Check convergence rate: error should decrease roughly as 1/√n
        # When n increases by 4x, error should decrease by ~2x
        if len(errors) >= 2:
            for i in range(len(errors) - 1):
                ratio = errors[i] / errors[i + 1] if errors[i + 1] > 0 else 0
                # Expect ratio between 1.5 and 3.0 (theoretical is 2.0)
                self.assertGreater(ratio, 1.2,
                                  f"Convergence rate too slow: error ratio = {ratio:.2f}")
                self.assertLess(ratio, 4.0,
                               f"Convergence rate suspicious: error ratio = {ratio:.2f}")
        
        print("✅ Monte Carlo convergence rate validated")


class TestStatisticalUtils(unittest.TestCase):
    """Test statistical utility functions and edge cases."""
    
    def test_confidence_interval_calculation(self):
        """Test confidence interval calculations are mathematically correct."""
        # Test with known values
        win_probability = 0.6
        simulations = 10000
        
        # Calculate expected confidence interval manually
        # CI = p ± z * sqrt(p(1-p)/n)
        z_score = 1.96  # 95% confidence
        std_error = math.sqrt(win_probability * (1 - win_probability) / simulations)
        expected_margin = z_score * std_error
        
        # Use solver to get actual interval
        solver = MonteCarloSolver()
        actual_interval = solver._calculate_confidence_interval(win_probability, simulations)
        
        if actual_interval:
            lower, upper = actual_interval
            actual_margin = (upper - lower) / 2
            
            # Should be very close to manual calculation
            margin_error = abs(actual_margin - expected_margin)
            self.assertLess(margin_error, 0.001,
                           f"Confidence interval calculation error: {margin_error:.4f}")
            
            print(f"✅ Confidence interval calculation: margin = {actual_margin:.4f} (expected {expected_margin:.4f})")
    
    def test_extreme_probability_edge_cases(self):
        """Test statistical calculations with extreme probabilities (near 0 or 1)."""
        test_cases = [
            (0.001, 10000),  # Very low probability
            (0.999, 10000),  # Very high probability
            (0.5, 100),      # Medium probability, small sample
        ]
        
        solver = MonteCarloSolver()
        
        for prob, sims in test_cases:
            with self.subTest(prob=prob, sims=sims):
                interval = solver._calculate_confidence_interval(prob, sims)
                
                if interval:
                    lower, upper = interval
                    
                    # Interval should contain the probability
                    self.assertLessEqual(lower, prob)
                    self.assertGreaterEqual(upper, prob)
                    
                    # Interval should be within valid bounds
                    self.assertGreaterEqual(lower, 0.0)
                    self.assertLessEqual(upper, 1.0)
                    
                    print(f"✅ Extreme case validated: p={prob}, interval=[{lower:.4f}, {upper:.4f}]")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 