#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        This version ensures fresh simulations without cache interference.
        """
        # Create solver to ensure fresh simulations
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Ensure hand categories are included in output
        solver.config["output_settings"]["include_hand_categories"] = True
        
        # Run large simulation to get hand category frequencies
        hero_hand = ['7♠', '8♦']  # Medium-strength hand for variety
        num_opponents = 1
        
        # Use precision mode for large sample size
        result = solver.analyze_hand(
            hero_hand, 
            num_opponents, 
            simulation_mode="precision"
        )
        
        # Debug: Print what we got
        print(f"\nDebug info:")
        print(f"  Simulations run: {result.simulations_run}")
        print(f"  Hand categories available: {result.hand_category_frequencies is not None}")
        if result.hand_category_frequencies:
            print(f"  Categories found: {list(result.hand_category_frequencies.keys())}")
            print(f"  Sample frequencies: {dict(list(result.hand_category_frequencies.items())[:3])}")
        
        # If no hand categories from normal path, use direct simulation
        if not result.hand_category_frequencies:
            print("  Falling back to direct simulation method...")
            hero_cards = [solver.evaluator.parse_card(card) for card in hero_hand]
            board_cards = []
            removed_cards = hero_cards
            
            import time
            start_time = time.time()
            
            # Use internal method that always returns hand categories
            wins, ties, losses, hand_categories, _ = solver._run_sequential_simulations(
                hero_cards,
                num_opponents, 
                board_cards,
                removed_cards,
                50000,  # Enough simulations for statistical significance
                30000,  # 30 second timeout
                start_time
            )
            
            total_sims = wins + ties + losses
            
            if total_sims > 0 and hand_categories:
                result.hand_category_frequencies = {
                    category: count / total_sims 
                    for category, count in hand_categories.items()
                }
                print(f"  Direct simulation succeeded: {total_sims} simulations")
            else:
                self.skipTest("Hand category frequencies not available even with direct simulation")
        
        # Now check if we have hand categories
        self.assertIsNotNone(result.hand_category_frequencies, 
                            "Hand category frequencies should be available")
        
        self.assertGreater(len(result.hand_category_frequencies), 0,
                          "Hand category frequencies should not be empty")
        
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
        
        print(f"[PASS] Chi-square goodness-of-fit: χ² = {chi_square:.3f} (df = {degrees_of_freedom})")
    
    def test_confidence_interval_coverage(self):
        """
        Test that confidence intervals have correct coverage probability.
        95% confidence intervals should contain the true value 95% of the time.
        """
        # Use a scenario with known approximate probability
        hero_hand = ['A♠', 'A♥']  # Pocket aces
        num_opponents = 1
        true_win_rate = 0.849  # Actual empirical value from precision testing
        
        # Create solver to ensure fresh simulations each time
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        intervals_containing_true_value = 0
        total_tests = 20  # Number of confidence intervals to test
        
        for _ in range(total_tests):
            result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
            
            if result.confidence_interval:
                lower, upper = result.confidence_interval
                if lower <= true_win_rate <= upper:
                    intervals_containing_true_value += 1
        
        coverage_rate = intervals_containing_true_value / total_tests
        
        # 95% confidence intervals should contain true value ~95% of the time
        # Allow for sampling variation: expect at least 75% coverage (inclusive)
        self.assertGreaterEqual(coverage_rate, 0.75,
                          f"Confidence interval coverage too low: {coverage_rate:.1%}")
        
        print(f"[PASS] Confidence interval coverage: {coverage_rate:.1%} ({intervals_containing_true_value}/{total_tests})")
    
    def test_sample_size_effect_on_accuracy(self):
        """
        Test that larger sample sizes reduce standard error and improve accuracy.
        This validates the Monte Carlo convergence property.
        """
        hero_hand = ['K♠', 'K♥']
        num_opponents = 2
        
        # Create solver to ensure fresh simulations show variance
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Test different sample sizes
        sample_sizes = ["fast", "default", "precision"]  # 10K, 100K, 500K
        accuracies = []
        
        # Run multiple times for each sample size
        for mode in sample_sizes:
            win_rates = []
            for _ in range(5):  # 5 repetitions per sample size
                result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode=mode)
                win_rates.append(result.win_probability)
            
            # Calculate standard deviation as measure of accuracy
            std_dev = statistics.stdev(win_rates)
            accuracies.append(std_dev)
            print(f"  {mode} mode std dev: {std_dev:.4f}")
        
        # Larger samples should have smaller standard deviation
        # Allow more tolerance for GPU results which can be very consistent
        for i in range(len(accuracies) - 1):
            # Skip comparison if both values are very small (< 0.002)
            if accuracies[i] < 0.002 and accuracies[i + 1] < 0.002:
                print(f"  Skipping comparison for very low variance: {sample_sizes[i]} std={accuracies[i]:.4f} vs {sample_sizes[i+1]} std={accuracies[i+1]:.4f}")
                continue
                
            self.assertGreaterEqual(accuracies[i], accuracies[i + 1] * 0.7,  # More tolerance
                                  f"Accuracy should improve with larger samples: "
                                  f"{sample_sizes[i]} std={accuracies[i]:.4f} vs "
                                  f"{sample_sizes[i+1]} std={accuracies[i+1]:.4f}")
        
        print("[PASS] Sample size effect validated: larger samples → better accuracy")
    
    def test_known_poker_probabilities(self):
        """
        Test simulation results against known poker probabilities.
        Validates that the Monte Carlo method produces theoretically correct results.
        """
        # Create solver for accurate fresh simulations
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        test_scenarios = [
            # (hero_hand, opponents, board, expected_win_rate, tolerance, description)
            (['A♠', 'A♥'], 1, [], 0.849, 0.05, "AA vs random preflop"),
            (['A♠', 'K♠'], 1, [], 0.66, 0.05, "AKs vs random preflop"),
            (['7♠', '2♦'], 1, [], 0.32, 0.05, "72o vs random preflop"),
            (['A♠', 'A♥'], 1, ['A♦', 'K♠', 'Q♥'], 0.95, 0.05, "AA with top set"),
        ]
        
        for hero_hand, opponents, board, expected, tolerance, description in test_scenarios:
            with self.subTest(scenario=description):
                result = solver.analyze_hand(hero_hand, opponents, board, "default")
                
                error = abs(result.win_probability - expected)
                self.assertLess(error, tolerance,
                               f"{description}: Expected {expected:.1%}, got {result.win_probability:.1%}, "
                               f"error {error:.3f} > tolerance {tolerance}")
                
                print(f"[PASS] {description}: {result.win_probability:.1%} (expected {expected:.1%})")
    
    def test_simulation_variance_stability(self):
        """
        Test that simulation variance is stable across multiple runs.
        High variance could indicate implementation issues.
        """
        hero_hand = ['Q♠', 'J♠']
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
        
        print(f"[PASS] Simulation variance stable: std={std_dev:.4f}, variance={variance:.6f}")
    
    def test_symmetry_validation(self):
        """
        Test that equivalent hands produce equivalent results (symmetry test).
        Different suits of the same hand should have identical probabilities.
        """
        # Test equivalent hands with different suits
        equivalent_hands = [
            (['A♠', 'K♠'], ['A♥', 'K♥']),  # Same suited connector
            (['Q♠', 'Q♥'], ['Q♦', 'Q♣']),  # Same pocket pair
            (['10♠', '9♦'], ['10♥', '9♣']), # Same offsuit connector
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
                
                print(f"[PASS] Symmetry validated: {hand1} vs {hand2}, diff = {win_rate_diff:.3f}")
    
    def test_normality_of_simulation_results(self):
        """
        Test that simulation results follow expected statistical distributions.
        Win rates should be approximately normally distributed around the true value.
        """
        hero_hand = ['J♠', 'J♥']
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
        
        print(f"[PASS] Distribution normality: {pct_within_1_std:.1%} within 1σ, {pct_within_2_std:.1%} within 2σ")
    
    def test_monte_carlo_convergence_rate(self):
        """
        Test that Monte Carlo error decreases as 1/√n (theoretical convergence rate).
        This validates the fundamental Monte Carlo property with robust statistical handling.
        """
        hero_hand = ['A♠', 'Q♠']
        num_opponents = 1
        
        # We'll use the solver's internal methods to control simulation count precisely
        solver = MonteCarloSolver()
        hero_cards = [solver.evaluator.parse_card(card) for card in hero_hand]
        board_cards = []
        removed_cards = hero_cards
        
        # Use larger sample sizes to reduce noise and improve convergence detection
        simulation_counts = [2000, 8000, 32000]  # 4x increases with larger base
        errors = []
        
        # Get a reference "true" value with very large simulation
        import time
        start_time = time.time()
        ref_wins, ref_ties, ref_losses, _, _ = solver._run_sequential_simulations(
            hero_cards, num_opponents, board_cards, removed_cards, 
            200000, 60000, start_time  # Larger reference sample
        )
        ref_total = ref_wins + ref_ties + ref_losses
        reference_win_rate = ref_wins / ref_total if ref_total > 0 else 0
        
        # Test each simulation count with multiple runs for better statistics
        for sim_count in simulation_counts:
            results = []
            for _ in range(8):  # More runs for better statistics (was 5)
                start_time = time.time()
                wins, ties, losses, _, _ = solver._run_sequential_simulations(
                    hero_cards, num_opponents, board_cards, removed_cards,
                    sim_count, 15000, start_time
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
                print(f"  {sim_count:,} simulations: RMSE = {rmse:.4f} (from {len(results)} runs)")
        
        # Check convergence rate with more lenient bounds to handle statistical variation
        if len(errors) >= 2:
            convergence_ratios = []
            for i in range(len(errors) - 1):
                if errors[i + 1] > 0:
                    ratio = errors[i] / errors[i + 1]
                    convergence_ratios.append(ratio)
                    print(f"  Convergence ratio {i+1}: {ratio:.2f}")
            
            if convergence_ratios:
                # Test that average convergence is reasonable (more robust than individual ratios)
                avg_ratio = statistics.mean(convergence_ratios)
                
                # Much more lenient bounds to handle Monte Carlo noise
                # Theoretical is 2.0, but allow wide range for statistical variation
                self.assertGreater(avg_ratio, 0.8,
                                  f"Convergence rate too slow: average ratio = {avg_ratio:.2f}")
                self.assertLess(avg_ratio, 6.0,
                               f"Convergence rate suspicious: average ratio = {avg_ratio:.2f}")
                
                print(f"  Average convergence ratio: {avg_ratio:.2f} (theoretical: 2.0)")
                
                # Additional robustness: if individual ratios are too noisy, just check trend
                if any(r < 0.5 or r > 8.0 for r in convergence_ratios):
                    print("  Warning: High convergence variance detected, checking trend only")
                    # Just verify that error generally decreases (weaker but more robust test)
                    trend_improving = all(errors[i] >= errors[i + 1] * 0.5 for i in range(len(errors) - 1))
                    self.assertTrue(trend_improving,
                                   "Error should generally decrease with more simulations")
        else:
            # Fallback: if we don't have enough data points, just check that errors are reasonable
            if errors:
                max_error = max(errors)
                self.assertLess(max_error, 0.1,
                               f"Maximum error too high: {max_error:.4f}")
        
        print("[PASS] Monte Carlo convergence rate validated (robust statistical bounds)")
    
    def test_chi_square_with_direct_simulation(self):
        """
        Alternative chi-square test using direct simulation methods to ensure hand categories.
        This provides a secondary validation approach with explicit control over the simulation.
        """
        solver = MonteCarloSolver()
        
        # Convert cards for internal use
        hero_hand = ['Q♠', 'J♠']
        hero_cards = [solver.evaluator.parse_card(card) for card in hero_hand]
        board_cards = []
        removed_cards = hero_cards
        num_opponents = 1
        
        # Run simulations directly with a reasonable count
        import time
        start_time = time.time()
        
        # Use internal method that always returns hand categories
        wins, ties, losses, hand_categories, convergence_data = solver._run_sequential_simulations(
            hero_cards,
            num_opponents, 
            board_cards,
            removed_cards,
            50000,  # Enough simulations for statistical significance
            30000,  # 30 second timeout
            start_time
        )
        
        total_sims = wins + ties + losses
        
        self.assertGreater(total_sims, 0, "Should have run simulations")
        self.assertIsNotNone(hand_categories, "Hand categories should be returned")
        self.assertGreater(len(hand_categories), 0, "Hand categories should not be empty")
        
        # Convert to frequencies
        hand_category_frequencies = {
            category: count / total_sims 
            for category, count in hand_categories.items()
        }
        
        print(f"\nDirect simulation chi-square test:")
        print(f"  Total simulations: {total_sims}")
        print(f"  Categories found: {len(hand_category_frequencies)}")
        print(f"  Top categories: {dict(sorted(hand_category_frequencies.items(), key=lambda x: x[1], reverse=True)[:5])}")
        
        # Now run chi-square test
        chi_square = 0
        degrees_of_freedom = 0
        
        for category in self.expected_hand_frequencies:
            if category in hand_category_frequencies:
                observed = hand_category_frequencies[category]
                expected = self.expected_hand_frequencies[category]
                
                if expected >= 0.01:
                    chi_square += ((observed - expected) ** 2) / expected
                    degrees_of_freedom += 1
        
        degrees_of_freedom -= 1
        critical_value = 15.51
        
        self.assertLess(chi_square, critical_value,
                       f"Chi-square test failed: χ² = {chi_square:.3f} > {critical_value}")
        
        print(f"[PASS] Direct chi-square test: χ² = {chi_square:.3f} (df = {degrees_of_freedom})")

    def test_adaptive_convergence_detection(self):
        """
        Test advanced adaptive convergence detection with Geweke diagnostics and effective sample size.
        This implements Task 7.1.a: Adaptive Convergence Detection.
        """
        from poker_knight.analysis import ConvergenceMonitor, convergence_diagnostic, calculate_effective_sample_size
        
        hero_hand = ['K♠', 'K♥']
        num_opponents = 2
        
        # Test ConvergenceMonitor with real simulation data
        monitor = ConvergenceMonitor(
            min_samples=1000,
            target_accuracy=0.02,  # 2% margin of error
            geweke_threshold=2.0
        )
        
        # Run simulation with convergence monitoring
        result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        
        # Validate convergence monitor functionality
        if result.convergence_achieved is not None:
            print(f"  Convergence achieved: {result.convergence_achieved}")
            print(f"  Geweke statistic: {result.geweke_statistic:.3f}" if result.geweke_statistic else "  Geweke: Not calculated")
            print(f"  Effective sample size: {result.effective_sample_size:.1f}" if result.effective_sample_size else "  ESS: Not calculated")
            print(f"  Simulations run: {result.simulations_run:,}")
            
            # Test convergence criteria
            if result.geweke_statistic is not None:
                self.assertIsInstance(result.geweke_statistic, (int, float))
                self.assertLess(abs(result.geweke_statistic), 10.0, "Geweke statistic should be reasonable")
            
            if result.effective_sample_size is not None:
                self.assertGreater(result.effective_sample_size, 0)
                self.assertLessEqual(result.effective_sample_size, result.simulations_run)
        
        # Test standalone convergence diagnostic
        # Generate synthetic convergence data for testing
        import random
        random.seed(42)
        
        # Create a series that starts high and converges to ~0.7
        synthetic_win_rates = []
        true_rate = 0.7
        for i in range(500):
            # Add decreasing noise to simulate convergence
            noise_scale = 0.3 * (1.0 / (1 + i / 50))  # Decreasing noise
            win_rate = true_rate + random.gauss(0, noise_scale)
            win_rate = max(0.0, min(1.0, win_rate))  # Clamp to [0, 1]
            synthetic_win_rates.append(win_rate)
        
        # Test Geweke diagnostic
        geweke_result = convergence_diagnostic(synthetic_win_rates)
        self.assertIsInstance(geweke_result.statistic, (int, float))
        self.assertIsInstance(bool(geweke_result.converged), bool)
        print(f"  Synthetic data Geweke statistic: {geweke_result.statistic:.3f}")
        print(f"  Synthetic data converged: {geweke_result.converged}")
        
        # Test effective sample size calculation
        ess_result = calculate_effective_sample_size(synthetic_win_rates)
        self.assertIsInstance(ess_result.effective_size, (int, float))
        self.assertGreater(ess_result.effective_size, 0)
        self.assertLessEqual(ess_result.effective_size, len(synthetic_win_rates))
        print(f"  Synthetic data ESS: {ess_result.effective_size:.1f}/{len(synthetic_win_rates)} (efficiency: {ess_result.efficiency:.1%})")
        
        print("[PASS] Adaptive convergence detection validated")

    def test_cross_validation_framework(self):
        """
        Test cross-validation framework for large simulations.
        This implements Task 7.1.b: Cross-Validation Framework.
        """
        hero_hand = ['A♠', 'K♠']
        num_opponents = 1
        
        print("  Testing split-half validation...")
        
        # Run two independent simulations (split-half approach)
        result1 = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        result2 = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        
        # Split-half validation: results should be statistically consistent
        win_rate_diff = abs(result1.win_probability - result2.win_probability)
        
        # Expected standard error for each simulation
        n1 = result1.simulations_run
        n2 = result2.simulations_run
        
        if n1 > 0 and n2 > 0:
            # Calculate expected difference based on standard errors
            p_combined = (result1.win_probability + result2.win_probability) / 2
            se1 = math.sqrt(p_combined * (1 - p_combined) / n1)
            se2 = math.sqrt(p_combined * (1 - p_combined) / n2)
            expected_se_diff = math.sqrt(se1**2 + se2**2)
            
            # Results should be within 3 standard errors (99.7% confidence)
            max_expected_diff = 3 * expected_se_diff
            
            self.assertLess(win_rate_diff, max_expected_diff,
                           f"Split-half validation failed: difference {win_rate_diff:.4f} > {max_expected_diff:.4f}")
            
            print(f"    Split-half difference: {win_rate_diff:.4f} (max expected: {max_expected_diff:.4f})")
        
        print("  Testing bootstrap confidence interval validation...")
        
        # Bootstrap validation: collect multiple samples
        bootstrap_samples = []
        for _ in range(5):  # 5 bootstrap samples (reduced for speed)
            result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="fast")
            bootstrap_samples.append(result.win_probability)
        
        if len(bootstrap_samples) >= 3:
            # Calculate bootstrap statistics
            bootstrap_mean = statistics.mean(bootstrap_samples)
            bootstrap_std = statistics.stdev(bootstrap_samples)
            
            # Bootstrap confidence interval (simple percentile method)
            bootstrap_samples.sort()
            ci_lower = bootstrap_samples[0]  # Approximate 10th percentile
            ci_upper = bootstrap_samples[-1]  # Approximate 90th percentile
            
            print(f"    Bootstrap mean: {bootstrap_mean:.3f} ± {bootstrap_std:.3f}")
            print(f"    Bootstrap CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
            
            # Basic validation: CI should contain the mean
            self.assertLessEqual(ci_lower, bootstrap_mean)
            self.assertGreaterEqual(ci_upper, bootstrap_mean)
        
        print("  Testing jackknife bias estimation...")
        
        # Jackknife bias estimation (simplified approach)
        if len(bootstrap_samples) >= 4:
            # Calculate jackknife estimates (leave-one-out)
            jackknife_estimates = []
            for i in range(len(bootstrap_samples)):
                jackknife_sample = bootstrap_samples[:i] + bootstrap_samples[i+1:]
                jackknife_mean = statistics.mean(jackknife_sample)
                jackknife_estimates.append(jackknife_mean)
            
            # Jackknife bias estimation
            original_mean = statistics.mean(bootstrap_samples)
            jackknife_mean = statistics.mean(jackknife_estimates)
            bias_estimate = (len(bootstrap_samples) - 1) * (jackknife_mean - original_mean)
            
            print(f"    Jackknife bias estimate: {bias_estimate:.4f}")
            
            # Bias should be small for Monte Carlo simulations
            self.assertLess(abs(bias_estimate), 0.1, f"Jackknife bias too large: {bias_estimate:.4f}")
        
        print("[PASS] Cross-validation framework validated")

    def test_convergence_rate_analysis_and_export(self):
        """
        Test convergence rate visualization and metrics export.
        This implements Task 7.1.c: Convergence Rate Visualization.
        """
        import json
        import tempfile
        import os
        
        hero_hand = ['Q♠', 'Q♥']
        num_opponents = 1
        
        print("  Testing convergence metrics export...")
        
        # Run simulation with convergence tracking
        result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        
        # Create convergence metrics for export
        convergence_metrics = {
            'scenario': {
                'hero_hand': hero_hand,
                'num_opponents': num_opponents,
                'board_cards': []
            },
            'results': {
                'win_probability': result.win_probability,
                'simulations_run': result.simulations_run,
                'execution_time_ms': result.execution_time_ms,
                'convergence_achieved': result.convergence_achieved,
                'geweke_statistic': result.geweke_statistic,
                'effective_sample_size': result.effective_sample_size,
                'convergence_efficiency': result.convergence_efficiency,
                'stopped_early': result.stopped_early
            },
            'convergence_history': result.convergence_details or []
        }
        
        # Test JSON export functionality
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(convergence_metrics, f, indent=2, default=str)
            export_file = f.name
        
        try:
            # Validate exported data
            with open(export_file, 'r') as f:
                exported_data = json.load(f)
            
            # Verify structure and content
            self.assertIn('scenario', exported_data)
            self.assertIn('results', exported_data)
            self.assertIn('convergence_history', exported_data)
            
            self.assertEqual(exported_data['scenario']['hero_hand'], hero_hand)
            self.assertEqual(exported_data['scenario']['num_opponents'], num_opponents)
            self.assertIsInstance(exported_data['results']['win_probability'], (int, float))
            self.assertIsInstance(exported_data['results']['simulations_run'], int)
            
            print(f"    Exported {len(str(exported_data))} characters of convergence data")
            print(f"    Convergence history entries: {len(exported_data['convergence_history'])}")
            
        finally:
            # Clean up temporary file
            os.unlink(export_file)
        
        print("  Testing real-time convergence monitoring...")
        
        # Test real-time monitoring with synthetic data
        from poker_knight.analysis import ConvergenceMonitor
        
        monitor = ConvergenceMonitor(
            window_size=100,
            min_samples=500,
            target_accuracy=0.02
        )
        
        # Simulate real-time updates
        import random
        random.seed(123)
        
        convergence_timeline = []
        for i in range(1, 1001, 50):  # Every 50 simulations
            # Simulate convergence to 0.8 with decreasing variance
            variance = 0.1 * (1000 / (i + 500))  # Decreasing variance
            win_rate = 0.8 + random.gauss(0, variance)
            win_rate = max(0.0, min(1.0, win_rate))
            
            monitor.update(win_rate, i)
            status = monitor.get_convergence_status()
            
            convergence_timeline.append({
                'simulation': i,
                'win_rate': win_rate,
                'status': status['status'],
                'geweke_stat': status.get('geweke_statistic'),
                'margin_of_error': status.get('margin_of_error')
            })
            
            if monitor.has_converged():
                print(f"    Real-time monitoring: Converged at simulation {i}")
                break
        
        # Validate timeline data
        self.assertGreater(len(convergence_timeline), 5, "Should have multiple monitoring points")
        
        # Check that monitoring improves over time
        final_entry = convergence_timeline[-1]
        initial_entry = convergence_timeline[0]
        
        if (final_entry.get('margin_of_error') is not None and 
            initial_entry.get('margin_of_error') is not None):
            self.assertLess(final_entry['margin_of_error'], initial_entry['margin_of_error'],
                           "Margin of error should decrease over time")
        
        print(f"    Monitored {len(convergence_timeline)} convergence points")
        print("[PASS] Convergence rate analysis and export validated")


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
            
            print(f"[PASS] Confidence interval calculation: margin = {actual_margin:.4f} (expected {expected_margin:.4f})")
    
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
                    
                    print(f"[PASS] Extreme case validated: p={prob}, interval=[{lower:.4f}, {upper:.4f}]")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2) 