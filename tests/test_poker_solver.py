#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Knight - Test Suite
Comprehensive tests for the Poker Knight Monte Carlo poker solver.
"""

import unittest
from poker_knight import (
    Card, HandEvaluator, Deck, MonteCarloSolver, 
    solve_poker_hand, SimulationResult
)
from poker_knight.constants import HAND_RANKINGS

class TestCard(unittest.TestCase):
    """Test Card class functionality."""
    
    def test_card_creation(self):
        """Test card creation and validation."""
        card = Card('A', 'S')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, '♠')  # Suits are normalized to unicode
        self.assertEqual(str(card), 'A♠')
        
    def test_card_value(self):
        """Test card value property."""
        self.assertEqual(Card('2', 'S').value, 0)
        self.assertEqual(Card('A', 'S').value, 12)
        self.assertEqual(Card('K', 'S').value, 11)
        
    def test_invalid_card(self):
        """Test invalid card creation."""
        with self.assertRaises(ValueError):
            Card('X', 'S')
        with self.assertRaises(ValueError):
            Card('A', '🎯')

class TestHandEvaluator(unittest.TestCase):
    """Test hand evaluation functionality."""
    
    def setUp(self):
        self.evaluator = HandEvaluator()
    
    def test_parse_card(self):
        """Test card parsing from string."""
        card = self.evaluator.parse_card('AS')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, '♠')  # Normalized to unicode
        
        # Test 10 parsing
        card = self.evaluator.parse_card('10H')
        self.assertEqual(card.rank, '10')
        self.assertEqual(card.suit, '♥')  # Normalized to unicode
    
    def test_royal_flush(self):
        """Test royal flush evaluation."""
        cards = [
            Card('A', 'S'), Card('K', 'S'), Card('Q', 'S'),
            Card('J', 'S'), Card('10', 'S')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['royal_flush'])
    
    def test_straight_flush(self):
        """Test straight flush evaluation."""
        cards = [
            Card('9', 'S'), Card('8', 'S'), Card('7', 'S'),
            Card('6', 'S'), Card('5', 'S')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['straight_flush'])
    
    def test_four_of_a_kind(self):
        """Test four of a kind evaluation."""
        cards = [
            Card('A', 'S'), Card('A', 'H'), Card('A', 'D'),
            Card('A', 'C'), Card('K', 'S')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['four_of_a_kind'])
        self.assertEqual(tiebreakers[0], 12)  # Aces
    
    def test_full_house(self):
        """Test full house evaluation."""
        cards = [
            Card('A', 'S'), Card('A', 'H'), Card('A', 'D'),
            Card('K', 'S'), Card('K', 'H')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['full_house'])
    
    def test_flush(self):
        """Test flush evaluation."""
        cards = [
            Card('A', 'S'), Card('J', 'S'), Card('9', 'S'),
            Card('7', 'S'), Card('5', 'S')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['flush'])
    
    def test_straight(self):
        """Test straight evaluation."""
        cards = [
            Card('A', 'S'), Card('K', 'H'), Card('Q', 'D'),
            Card('J', 'S'), Card('10', 'C')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['straight'])
        
        # Test wheel straight (A-2-3-4-5)
        cards = [
            Card('A', 'S'), Card('2', 'H'), Card('3', 'D'),
            Card('4', 'S'), Card('5', 'C')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['straight'])
        self.assertEqual(tiebreakers[0], 3)  # 5-high straight
    
    def test_three_of_a_kind(self):
        """Test three of a kind evaluation."""
        cards = [
            Card('A', 'S'), Card('A', 'H'), Card('A', 'D'),
            Card('K', 'S'), Card('Q', 'H')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['three_of_a_kind'])
    
    def test_two_pair(self):
        """Test two pair evaluation."""
        cards = [
            Card('A', 'S'), Card('A', 'H'), Card('K', 'D'),
            Card('K', 'S'), Card('Q', 'H')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['two_pair'])
    
    def test_pair(self):
        """Test pair evaluation."""
        cards = [
            Card('A', 'S'), Card('A', 'H'), Card('K', 'D'),
            Card('Q', 'S'), Card('J', 'H')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['pair'])
    
    def test_high_card(self):
        """Test high card evaluation."""
        cards = [
            Card('A', 'S'), Card('K', 'H'), Card('Q', 'D'),
            Card('J', 'S'), Card('9', 'H')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['high_card'])
    
    def test_seven_card_evaluation(self):
        """Test evaluation with 7 cards (finds best 5)."""
        cards = [
            Card('A', 'S'), Card('A', 'H'), Card('A', 'D'),
            Card('K', 'S'), Card('K', 'H'), Card('2', 'C'), Card('3', 'D')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['full_house'])

class TestDeck(unittest.TestCase):
    """Test deck functionality."""
    
    def test_deck_creation(self):
        """Test deck creation."""
        deck = Deck()
        self.assertEqual(deck.remaining_cards(), 52)
    
    def test_deck_with_removed_cards(self):
        """Test deck creation with removed cards."""
        removed = [Card('A', 'S'), Card('K', 'S')]
        deck = Deck(removed)
        self.assertEqual(deck.remaining_cards(), 50)
    
    def test_deal_cards(self):
        """Test dealing cards."""
        deck = Deck()
        cards = deck.deal(5)
        self.assertEqual(len(cards), 5)
        self.assertEqual(deck.remaining_cards(), 47)
    
    def test_deal_too_many_cards(self):
        """Test dealing more cards than available."""
        deck = Deck()
        deck.deal(50)
        with self.assertRaises(ValueError):
            deck.deal(5)

class TestMonteCarloSolver(unittest.TestCase):
    """Test Monte Carlo solver functionality."""
    
    def setUp(self):
        self.solver = MonteCarloSolver()
    
    def test_analyze_pocket_aces(self):
        """Test analysis of pocket aces (should have high win rate)."""
        result = self.solver.analyze_hand(['AS', 'AH'], 1, simulation_mode="fast")
        
        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.win_probability, 0.8)  # AA should win >80% vs 1 opponent
        self.assertGreater(result.simulations_run, 0)
        self.assertGreater(result.execution_time_ms, 0)
    
    def test_analyze_with_board(self):
        """Test analysis with board cards."""
        # Hero has pocket aces, board has A-K-Q (giving hero trips)
        result = self.solver.analyze_hand(
            ['AS', 'AH'], 
            2, 
            ['AD', 'KS', 'QH'],
            simulation_mode="fast"
        )
        
        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.win_probability, 0.8)  # Trip aces should be very strong
    
    def test_analyze_weak_hand(self):
        """Test analysis of weak hand."""
        result = self.solver.analyze_hand(['2S', '7H'], 5, simulation_mode="fast")
        
        self.assertIsInstance(result, SimulationResult)
        self.assertLess(result.win_probability, 0.3)  # 2-7 offsuit should be weak
    
    def test_invalid_inputs(self):
        """Test invalid input handling."""
        # Wrong number of hole cards
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS'], 1)
        
        # Too many opponents
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'KS'], 7)
        
        # Wrong number of board cards
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'KS'], 1, ['AD', 'KD'])
        
        # Invalid simulation mode
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'KS'], 1, simulation_mode="invalid")
        
        # Duplicate cards in hero hand and board
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'AS'], 1)  # Duplicate in hero hand
        
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'KS'], 1, ['AS', 'QH', 'JD'])  # Hero card on board
        
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'KS'], 1, ['QH', 'QH', 'JD'])  # Duplicate on board
        
        # Invalid card formats
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'XS'], 1)  # Invalid rank
        
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'A🎯'], 1)  # Invalid suit
        
        # Zero opponents
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'KS'], 0)
        
        # Too many hole cards
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['AS', 'KS', 'QS'], 1)
        
        # Empty hero hand
        with self.assertRaises(ValueError):
            self.solver.analyze_hand([], 1)
    
    def test_boundary_conditions(self):
        """Test boundary conditions."""
        # Exactly 5 board cards (river)
        result = self.solver.analyze_hand(
            ['AS', 'KS'], 
            1, 
            ['QS', 'JS', '10S', '9H', '8D'],  # 5 cards
            simulation_mode="fast"
        )
        self.assertIsInstance(result, SimulationResult)
        
        # Exactly 3 board cards (flop)
        result = self.solver.analyze_hand(
            ['AS', 'KS'], 
            1, 
            ['QS', 'JS', '10S'],  # 3 cards
            simulation_mode="fast"
        )
        self.assertIsInstance(result, SimulationResult)
        
        # Maximum opponents (6)
        result = self.solver.analyze_hand(['AS', 'AH'], 6, simulation_mode="fast")
        self.assertIsInstance(result, SimulationResult)
        
        # Minimum opponents (1)
        result = self.solver.analyze_hand(['AS', 'AH'], 1, simulation_mode="fast")
        self.assertIsInstance(result, SimulationResult)
    
    def test_card_format_edge_cases(self):
        """Test edge cases in card format parsing."""
        # Test 10 card parsing
        result = self.solver.analyze_hand(['10S', '10H'], 1, simulation_mode="fast")
        self.assertIsInstance(result, SimulationResult)
        
        # Test all suits
        suits = ['S', 'H', 'D', 'C']
        for suit in suits:
            result = self.solver.analyze_hand([f'A{suit}', f'K{suit}'], 1, simulation_mode="fast")
            self.assertIsInstance(result, SimulationResult)
        
        # Test all ranks
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
        for rank in ranks:
            result = self.solver.analyze_hand([f'{rank}S', 'AH'], 1, simulation_mode="fast")
            self.assertIsInstance(result, SimulationResult)

class TestEdgeCaseHandEvaluation(unittest.TestCase):
    """Test edge cases in hand evaluation."""
    
    def setUp(self):
        self.evaluator = HandEvaluator()
    
    def test_wheel_straight_edge_cases(self):
        """Test wheel straight (A-2-3-4-5) in various scenarios."""
        # Basic wheel straight
        cards = [Card('A', 'S'), Card('2', 'H'), Card('3', 'D'), Card('4', 'S'), Card('5', 'C')]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['straight'])
        self.assertEqual(tiebreakers[0], 3)  # 5-high straight
        
        # Wheel straight flush
        cards = [Card('A', 'S'), Card('2', 'S'), Card('3', 'S'), Card('4', 'S'), Card('5', 'S')]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['straight_flush'])
        self.assertEqual(tiebreakers[0], 3)  # 5-high straight flush
        
        # Wheel vs higher straight (wheel should lose)
        wheel = [Card('A', 'S'), Card('2', 'H'), Card('3', 'D'), Card('4', 'S'), Card('5', 'C')]
        higher = [Card('6', 'S'), Card('7', 'H'), Card('8', 'D'), Card('9', 'S'), Card('10', 'C')]
        
        wheel_rank, wheel_tie = self.evaluator.evaluate_hand(wheel)
        higher_rank, higher_tie = self.evaluator.evaluate_hand(higher)
        
        self.assertEqual(wheel_rank, higher_rank)  # Both straights
        self.assertLess(wheel_tie[0], higher_tie[0])  # Wheel should have lower tiebreaker
    
    def test_identical_hands(self):
        """Test true tie scenarios."""
        # Identical full houses
        hand1 = [Card('A', 'S'), Card('A', 'H'), Card('A', 'D'), Card('K', 'S'), Card('K', 'H')]
        hand2 = [Card('A', 'C'), Card('A', 'S'), Card('A', 'H'), Card('K', 'D'), Card('K', 'C')]
        
        rank1, tie1 = self.evaluator.evaluate_hand(hand1)
        rank2, tie2 = self.evaluator.evaluate_hand(hand2)
        
        self.assertEqual(rank1, rank2)
        self.assertEqual(tie1, tie2)
        
        # Identical high card hands
        hand1 = [Card('A', 'S'), Card('K', 'H'), Card('Q', 'D'), Card('J', 'S'), Card('9', 'H')]
        hand2 = [Card('A', 'H'), Card('K', 'S'), Card('Q', 'S'), Card('J', 'D'), Card('9', 'S')]
        
        rank1, tie1 = self.evaluator.evaluate_hand(hand1)
        rank2, tie2 = self.evaluator.evaluate_hand(hand2)
        
        self.assertEqual(rank1, rank2)
        self.assertEqual(tie1, tie2)
    
    def test_kicker_comparison_edge_cases(self):
        """Test complex kicker scenarios."""
        # Two pair with different kickers
        hand1 = [Card('A', 'S'), Card('A', 'H'), Card('K', 'D'), Card('K', 'S'), Card('Q', 'H')]  # AA KK Q
        hand2 = [Card('A', 'D'), Card('A', 'C'), Card('K', 'H'), Card('K', 'C'), Card('J', 'D')]  # AA KK J
        
        rank1, tie1 = self.evaluator.evaluate_hand(hand1)
        rank2, tie2 = self.evaluator.evaluate_hand(hand2)
        
        self.assertEqual(rank1, rank2)  # Both two pair
        self.assertGreater(tie1, tie2)  # Q kicker beats J kicker
        
        # One pair with multiple kickers
        hand1 = [Card('A', 'S'), Card('A', 'H'), Card('K', 'D'), Card('Q', 'S'), Card('J', 'H')]  # AA K Q J
        hand2 = [Card('A', 'D'), Card('A', 'C'), Card('K', 'H'), Card('Q', 'C'), Card('10', 'D')]  # AA K Q 10
        
        rank1, tie1 = self.evaluator.evaluate_hand(hand1)
        rank2, tie2 = self.evaluator.evaluate_hand(hand2)
        
        self.assertEqual(rank1, rank2)  # Both one pair
        self.assertGreater(tie1, tie2)  # J kicker beats 10 kicker
    
    def test_seven_card_edge_cases(self):
        """Test 7-card evaluation edge cases."""
        # 7 cards where best 5 is not obvious
        cards = [
            Card('A', 'S'), Card('A', 'H'),  # Pair of aces
            Card('K', 'D'), Card('K', 'S'),  # Pair of kings  
            Card('Q', 'H'), Card('J', 'D'), Card('9', 'S')  # High cards (no straight)
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['two_pair'])
        
        # 7 cards with potential straight and flush
        cards = [
            Card('A', 'S'), Card('K', 'S'), Card('Q', 'S'),  # Potential royal flush
            Card('J', 'S'), Card('10', 'S'),  # Completes royal flush
            Card('9', 'H'), Card('8', 'D')   # Extra cards
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, HAND_RANKINGS['royal_flush'])

class TestConfigurationEdgeCases(unittest.TestCase):
    """Test configuration-related edge cases."""
    
    def test_missing_config_file(self):
        """Test behavior with missing config file."""
        with self.assertRaises(FileNotFoundError):
            MonteCarloSolver("nonexistent_config.json")
    
    def test_config_with_missing_keys(self):
        """Test behavior with incomplete config."""
        import tempfile
        import json
        import os
        
        # Create a minimal config file
        minimal_config = {
            "simulation_settings": {
                "default_simulations": 1000
            },
            "performance_settings": {
                "max_simulation_time_ms": 5000
            },
            "output_settings": {
                "include_confidence_interval": True,
                "include_hand_categories": True,
                "decimal_precision": 4
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(minimal_config, f)
            temp_config_path = f.name
        
        try:
            solver = MonteCarloSolver(temp_config_path)
            result = solver.analyze_hand(['AS', 'KS'], 1, simulation_mode="fast")
            self.assertIsInstance(result, SimulationResult)
        finally:
            os.unlink(temp_config_path)

class TestStatisticalEdgeCases(unittest.TestCase):
    """Test statistical edge cases."""
    
    def setUp(self):
        self.solver = MonteCarloSolver()
    
    def test_extreme_scenarios(self):
        """Test extreme win/loss scenarios."""
        # Royal flush vs high card (should be 100% win)
        # Note: This is hard to test directly since we can't control opponent cards
        # But we can test that very strong hands have very high win rates
        result = self.solver.analyze_hand(
            ['AS', 'KS'], 
            1, 
            ['QS', 'JS', '10S'],  # Royal flush
            simulation_mode="fast"
        )
        self.assertGreater(result.win_probability, 0.95)  # Should be very high
    
    def test_confidence_interval_edge_cases(self):
        """Test confidence interval calculations."""
        # Very few simulations should have wide confidence intervals
        # Very many simulations should have narrow confidence intervals
        # This is tested in the performance regression tests
        pass
    
    def test_hand_category_edge_cases(self):
        """Test hand category frequency edge cases."""
        # Test with a hand that can make many different types
        result = self.solver.analyze_hand(['AS', 'KS'], 1, simulation_mode="fast")
        
        if result.hand_category_frequencies:
            # Should have various hand types
            self.assertGreater(len(result.hand_category_frequencies), 1)
            
            # Frequencies should sum to approximately 1.0
            total = sum(result.hand_category_frequencies.values())
            self.assertAlmostEqual(total, 1.0, places=1)

class TestConvenienceFunction(unittest.TestCase):
    """Test the convenience function."""
    
    def test_solve_poker_hand(self):
        """Test the convenience function."""
        result = solve_poker_hand(['AS', 'AH'], 1, simulation_mode="fast")
        
        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.win_probability, 0.8)

class TestKnownScenarios(unittest.TestCase):
    """Test against known poker scenarios."""
    
    def test_pocket_pairs_vs_overcards(self):
        """Test pocket pair vs overcards (classic coin flip)."""
        solver = MonteCarloSolver()
        
        # This is hard to test exactly since we can't control opponent cards
        # But we can test that the solver runs without error
        result = solver.analyze_hand(['QS', 'QH'], 1, simulation_mode="fast")
        self.assertIsInstance(result, SimulationResult)
    
    def test_dominated_hands(self):
        """Test dominated hand scenarios."""
        solver = MonteCarloSolver()
        
        # AK should beat AQ most of the time (when they don't both make pairs)
        ak_result = solver.analyze_hand(['AS', 'KH'], 1, simulation_mode="fast")
        aq_result = solver.analyze_hand(['AS', 'QH'], 1, simulation_mode="fast")
        
        # Both should be reasonable hands, but this is hard to test precisely
        # without controlling opponent cards
        self.assertIsInstance(ak_result, SimulationResult)
        self.assertIsInstance(aq_result, SimulationResult)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 