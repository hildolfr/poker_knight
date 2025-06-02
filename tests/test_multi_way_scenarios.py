#!/usr/bin/env python3
"""
Comprehensive Multi-Way Pot Analysis Tests (Task 7.2)

This module tests the advanced multi-way pot analysis features added to Poker Knight v1.5.0:
- Position-aware equity calculation with fold equity estimates
- ICM (Independent Chip Model) integration for tournament play  
- Multi-way hand range analysis and coordination effects
- Stack-to-pot ratio calculations and tournament pressure metrics

Author: hildolfr
Version: 1.5.0
"""

import unittest
import math
from typing import Dict, List, Any
from poker_knight import solve_poker_hand, MonteCarloSolver
import random


class TestMultiWayPotAnalysis(unittest.TestCase):
    """Test suite for multi-way pot analysis features."""
    
    def setUp(self):
        """Set up test environment."""
        self.solver = MonteCarloSolver()
    
    def test_position_aware_equity_calculation(self):
        """
        Test position-aware equity calculation with different table positions.
        Validates Task 7.2.a: Position-Aware Equity Calculation.
        """
        hero_hand = ['A♠️', 'K♠️']  # Strong hand for position testing
        num_opponents = 2
        
        positions = ['early', 'middle', 'late', 'button', 'sb', 'bb']
        position_results = {}
        
        print("Testing position-aware equity calculation...")
        
        for position in positions:
            result = solve_poker_hand(
                hero_hand, num_opponents, 
                simulation_mode="fast",
                hero_position=position
            )
            
            # Validate position-aware equity exists
            self.assertIsNotNone(result.position_aware_equity, 
                               f"Position-aware equity missing for {position}")
            
            # Validate fold equity estimates
            self.assertIsNotNone(result.fold_equity_estimates,
                               f"Fold equity estimates missing for {position}")
            
            position_equity = result.position_aware_equity
            baseline_equity = position_equity.get('baseline_equity')
            position_adjusted = position_equity.get(position)
            
            # Validate equity structure
            self.assertIsInstance(baseline_equity, float)
            self.assertIsInstance(position_adjusted, float)
            self.assertGreaterEqual(baseline_equity, 0.0)
            self.assertLessEqual(baseline_equity, 1.0)
            self.assertGreaterEqual(position_adjusted, 0.0)
            self.assertLessEqual(position_adjusted, 1.0)
            
            position_results[position] = {
                'baseline': baseline_equity,
                'adjusted': position_adjusted,
                'advantage': position_adjusted - baseline_equity
            }
            
            print(f"  {position:6}: {baseline_equity:.3f} → {position_adjusted:.3f} "
                  f"(advantage: {position_adjusted - baseline_equity:+.3f})")
        
        # Test positional advantages are in expected order
        # Button should have highest advantage, early position should have lowest
        button_advantage = position_results['button']['advantage']
        early_advantage = position_results['early']['advantage']
        
        self.assertGreater(button_advantage, early_advantage,
                          "Button position should have better equity than early position")
        
        # Late position should be better than early
        late_advantage = position_results['late']['advantage']
        self.assertGreater(late_advantage, early_advantage,
                          "Late position should have better equity than early position")
        
        print("✅ Position-aware equity calculation validated")
    
    def test_multi_way_statistics_3_opponents(self):
        """
        Test multi-way statistics calculation for 3+ opponents.
        Validates advanced multi-way pot metrics and coordination effects.
        """
        hero_hand = ['Q♠️', 'Q♥️']  # Premium pair for multi-way testing
        num_opponents = 3  # 4-way pot
        
        result = solve_poker_hand(
            hero_hand, num_opponents,
            simulation_mode="fast"
        )
        
        print(f"Testing 4-way pot statistics (QQ vs {num_opponents} opponents)...")
        
        # Validate multi-way statistics exist
        self.assertIsNotNone(result.multi_way_statistics,
                           "Multi-way statistics missing for 3+ opponents")
        self.assertIsNotNone(result.coordination_effects,
                           "Coordination effects missing for multi-way pot")
        self.assertIsNotNone(result.defense_frequencies,
                           "Defense frequencies missing for multi-way analysis")
        
        multiway_stats = result.multi_way_statistics
        coordination = result.coordination_effects
        defense_freq = result.defense_frequencies
        
        # Validate multi-way statistics structure
        self.assertEqual(multiway_stats['total_opponents'], num_opponents)
        self.assertIn('individual_win_rate', multiway_stats)
        self.assertIn('conditional_win_probability', multiway_stats)
        self.assertIn('expected_position_finish', multiway_stats)
        
        # Validate coordination effects
        self.assertIn('coordination_factor', coordination)
        self.assertIn('total_coordination_effect', coordination)
        self.assertIn('isolation_difficulty', coordination)
        
        # Validate defense frequencies
        self.assertIn('optimal_defense_frequency', defense_freq)
        self.assertIn('minimum_defense_frequency', defense_freq)
        
        # Test statistical relationships
        self.assertGreater(multiway_stats['individual_win_rate'], result.win_probability,
                          "Individual win rate should be higher than beating all opponents")
        
        self.assertGreaterEqual(defense_freq['optimal_defense_frequency'], 
                               defense_freq['minimum_defense_frequency'],
                               "Optimal defense frequency should be >= minimum")
        
        print(f"  Win vs all: {result.win_probability:.3f}")
        print(f"  Win vs 1: {multiway_stats['individual_win_rate']:.3f}")
        print(f"  Expected finish: {multiway_stats['expected_position_finish']:.1f}")
        print(f"  Coordination effect: {coordination['total_coordination_effect']:.3f}")
        print(f"  Optimal defense freq: {defense_freq['optimal_defense_frequency']:.3f}")
        
        print("✅ Multi-way statistics (3+ opponents) validated")
    
    def test_icm_integration_tournament_context(self):
        """
        Test ICM (Independent Chip Model) integration for tournament play.
        Validates Task 7.2.b: ICM Integration.
        """
        hero_hand = ['J♠️', 'J♥️']  # Medium strength hand for ICM testing
        num_opponents = 2
        
        # Tournament context: bubble situation
        tournament_context = {
            'bubble_factor': 1.5,  # Increased pressure near bubble
            'blinds': 1000,
            'ante': 100
        }
        
        # Stack sizes: [hero, opponent1, opponent2]
        stack_sizes = [15000, 25000, 10000]  # Hero is medium stack
        pot_size = 3000
        
        result = solve_poker_hand(
            hero_hand, num_opponents,
            simulation_mode="fast",
            stack_sizes=stack_sizes,
            pot_size=pot_size,
            tournament_context=tournament_context
        )
        
        print("Testing ICM integration with tournament context...")
        
        # Validate ICM fields exist
        self.assertIsNotNone(result.icm_equity, "ICM equity missing")
        self.assertIsNotNone(result.bubble_factor, "Bubble factor missing")
        self.assertIsNotNone(result.stack_to_pot_ratio, "SPR missing")
        self.assertIsNotNone(result.tournament_pressure, "Tournament pressure missing")
        
        # Validate ICM calculations
        self.assertEqual(result.bubble_factor, 1.5)
        self.assertAlmostEqual(result.stack_to_pot_ratio, 15000 / 3000, places=2)
        
        # Validate tournament pressure metrics
        tournament_pressure = result.tournament_pressure
        self.assertIn('hero_chip_percentage', tournament_pressure)
        self.assertIn('stack_pressure', tournament_pressure)
        
        expected_chip_percentage = 15000 / (15000 + 25000 + 10000)
        self.assertAlmostEqual(tournament_pressure['hero_chip_percentage'], 
                              expected_chip_percentage, places=3)
        
        # ICM equity should be different from raw equity due to bubble pressure
        self.assertNotEqual(result.icm_equity, result.win_probability,
                           "ICM equity should differ from raw win probability")
        
        print(f"  Raw equity: {result.win_probability:.3f}")
        print(f"  ICM equity: {result.icm_equity:.3f}")
        print(f"  SPR: {result.stack_to_pot_ratio:.1f}")
        print(f"  Chip %: {tournament_pressure['hero_chip_percentage']:.1%}")
        print(f"  Stack pressure: {tournament_pressure['stack_pressure']:.3f}")
        
        print("✅ ICM integration validated")
    
    def test_multi_way_range_analysis(self):
        """
        Test multi-way hand range analysis and coordination effects.
        Validates Task 7.2.c: Multi-Way Range Analysis.
        """
        test_scenarios = [
            # (hand, opponents, expected_characteristics)
            (['A♠️', 'A♥️'], 4, {'strong_vs_many': True}),   # Premium hand vs many
            (['7♠️', '6♠️'], 3, {'drawing_hand': True}),      # Drawing hand multiway
            (['K♠️', 'Q♦️'], 5, {'marginal_multiway': True}), # Marginal hand vs many
        ]
        
        print("Testing multi-way range analysis...")
        
        for hero_hand, num_opponents, characteristics in test_scenarios:
            result = solve_poker_hand(
                hero_hand, num_opponents,
                simulation_mode="fast"
            )
            
            # All multi-way analysis fields should exist
            self.assertIsNotNone(result.bluff_catching_frequency,
                               f"Bluff catching frequency missing for {hero_hand}")
            self.assertIsNotNone(result.range_coordination_score,
                               f"Range coordination score missing for {hero_hand}")
            
            bluff_catch_freq = result.bluff_catching_frequency
            coordination_score = result.range_coordination_score
            
            # Validate ranges
            self.assertGreaterEqual(bluff_catch_freq, 0.1)
            self.assertLessEqual(bluff_catch_freq, 0.5)
            self.assertGreaterEqual(coordination_score, 0.0)
            self.assertLessEqual(coordination_score, 1.0)
            
            hand_desc = ' '.join(hero_hand)
            print(f"  {hand_desc:8} vs {num_opponents}: "
                  f"bluff catch {bluff_catch_freq:.3f}, "
                  f"coordination {coordination_score:.3f}")
            
            # Test specific characteristics
            if characteristics.get('strong_vs_many'):
                # Strong hands should have higher coordination scores (more opposition)
                self.assertGreater(coordination_score, 0.4,
                                 "Strong hands should face more coordination")
            
            if characteristics.get('marginal_multiway'):
                # Marginal hands vs many opponents should have lower bluff catching frequency
                self.assertLess(bluff_catch_freq, 0.3,
                               "Marginal hands should bluff catch less vs many opponents")
        
        print("✅ Multi-way range analysis validated")
    
    def test_position_and_stack_interaction(self):
        """
        Test interaction between position and stack size in multi-way analysis.
        """
        hero_hand = ['A♠️', 'Q♠️']  # Strong but not premium hand
        num_opponents = 2
        
        # Test different position/stack combinations
        test_combinations = [
            ('button', [50000, 20000, 30000], 'big_stack_button'),
            ('early', [50000, 20000, 30000], 'big_stack_early'),  
            ('button', [10000, 20000, 30000], 'short_stack_button'),
            ('early', [10000, 20000, 30000], 'short_stack_early'),
        ]
        
        print("Testing position and stack size interactions...")
        
        results = {}
        for position, stacks, scenario in test_combinations:
            result = solve_poker_hand(
                hero_hand, num_opponents,
                simulation_mode="fast",
                hero_position=position,
                stack_sizes=stacks,
                pot_size=5000,
                tournament_context={'bubble_factor': 1.0}
            )
            
            results[scenario] = {
                'raw_equity': result.win_probability,
                'position_equity': result.position_aware_equity[position],
                'icm_equity': result.icm_equity,
                'spr': result.stack_to_pot_ratio
            }
            
            print(f"  {scenario:18}: "
                  f"raw {result.win_probability:.3f}, "
                  f"pos {result.position_aware_equity[position]:.3f}, "
                  f"icm {result.icm_equity:.3f}")
        
        # Validate expected relationships
        # Big stack + button should be better than big stack + early
        self.assertGreater(
            results['big_stack_button']['position_equity'],
            results['big_stack_early']['position_equity'],
            "Big stack on button should have better equity than big stack early"
        )
        
        # Short stack scenarios should show different ICM pressures
        self.assertNotEqual(
            results['short_stack_button']['icm_equity'],
            results['big_stack_button']['icm_equity'],
            "Stack size should affect ICM equity"
        )
        
        print("✅ Position and stack interactions validated")
    
    def test_backward_compatibility(self):
        """
        Test that existing code without multi-way parameters still works.
        Ensures backward compatibility while adding new features.
        """
        hero_hand = ['K♠️', 'K♥️']
        num_opponents = 2
        
        # Use a fixed seed for reproducible results in this test
        random.seed(42)
        
        # Test old-style call (should work without multi-way analysis)
        result_old = solve_poker_hand(hero_hand, num_opponents, simulation_mode="fast")
        
        # Reset seed to ensure same random state
        random.seed(42)
        
        # Test new-style call with explicit None parameters
        result_new = solve_poker_hand(
            hero_hand, num_opponents, simulation_mode="fast",
            hero_position=None, stack_sizes=None, pot_size=None, tournament_context=None
        )
        
        # Core results should be very similar (within Monte Carlo variance tolerance)
        win_prob_diff = abs(result_old.win_probability - result_new.win_probability)
        self.assertLess(win_prob_diff, 0.01,  # Allow 1% variance due to Monte Carlo simulation
                       f"Win probabilities should be similar: {result_old.win_probability:.4f} vs {result_new.win_probability:.4f}")
        
        # Simulation counts should be similar
        sim_diff = abs(result_old.simulations_run - result_new.simulations_run)
        max_sim_diff = max(result_old.simulations_run, result_new.simulations_run) * 0.1  # 10% tolerance
        self.assertLess(sim_diff, max_sim_diff,
                       f"Simulation counts should be similar: {result_old.simulations_run} vs {result_new.simulations_run}")
        
        # Multi-way fields should be None for old-style calls
        self.assertIsNone(result_old.position_aware_equity)
        self.assertIsNone(result_old.icm_equity)
        self.assertIsNone(result_old.multi_way_statistics)
        
        print("✅ Backward compatibility validated")
    
    def test_edge_cases_multiway_analysis(self):
        """
        Test edge cases in multi-way analysis.
        """
        hero_hand = ['2♠️', '3♠️']  # Weak hand for edge case testing
        
        # Test maximum opponents (6)
        result_max = solve_poker_hand(
            hero_hand, 6, simulation_mode="fast",
            hero_position="early"
        )
        
        # Should handle maximum opponents gracefully
        self.assertIsNotNone(result_max.multi_way_statistics)
        self.assertEqual(result_max.multi_way_statistics['total_opponents'], 6)
        
        # Test with extreme stack ratios
        extreme_stacks = [100, 50000, 1000, 2000]  # Hero very short-stacked
        result_extreme = solve_poker_hand(
            hero_hand, 3, simulation_mode="fast",
            stack_sizes=extreme_stacks,
            pot_size=200,
            tournament_context={'bubble_factor': 2.0}  # High bubble pressure
        )
        
        # Should handle extreme ratios without crashing
        self.assertIsNotNone(result_extreme.stack_to_pot_ratio)
        self.assertIsNotNone(result_extreme.icm_equity)
        
        # SPR should be very low (short stack)
        self.assertLess(result_extreme.stack_to_pot_ratio, 1.0)
        
        print("✅ Edge cases in multi-way analysis validated")


class TestMultiWayPerformance(unittest.TestCase):
    """Test performance characteristics of multi-way analysis."""
    
    def test_multiway_analysis_performance(self):
        """Test that multi-way analysis doesn't significantly impact performance."""
        import time
        
        hero_hand = ['A♠️', 'K♠️']
        num_opponents = 3
        
        # Test without multi-way analysis
        start_time = time.time()
        result_simple = solve_poker_hand(hero_hand, num_opponents, simulation_mode="fast")
        simple_time = time.time() - start_time
        
        # Test with multi-way analysis
        start_time = time.time()
        result_multiway = solve_poker_hand(
            hero_hand, num_opponents, simulation_mode="fast",
            hero_position="button",
            stack_sizes=[25000, 20000, 15000, 30000],
            pot_size=3000,
            tournament_context={'bubble_factor': 1.2}
        )
        multiway_time = time.time() - start_time
        
        # Multi-way analysis should not add more than 50% overhead
        performance_ratio = multiway_time / simple_time
        self.assertLess(performance_ratio, 1.5,
                       f"Multi-way analysis overhead too high: {performance_ratio:.2f}x")
        
        print(f"Performance test:")
        print(f"  Simple analysis: {simple_time:.3f}s")
        print(f"  Multi-way analysis: {multiway_time:.3f}s")
        print(f"  Overhead: {(performance_ratio - 1) * 100:.1f}%")
        print("✅ Multi-way analysis performance acceptable")


if __name__ == '__main__':
    # Run with high verbosity to see detailed test output
    unittest.main(verbosity=2) 