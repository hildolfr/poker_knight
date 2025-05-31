#!/usr/bin/env python3
"""
Poker Knight - Test Suite
Comprehensive tests for the Poker Knight Monte Carlo poker solver.
"""

import unittest
from poker_solver import (
    Card, HandEvaluator, Deck, MonteCarloSolver, 
    solve_poker_hand, SimulationResult
)

class TestCard(unittest.TestCase):
    """Test Card class functionality."""
    
    def test_card_creation(self):
        """Test card creation and validation."""
        card = Card('A', '♠️')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, '♠️')
        self.assertEqual(str(card), 'A♠️')
        
    def test_card_value(self):
        """Test card value property."""
        self.assertEqual(Card('2', '♠️').value, 0)
        self.assertEqual(Card('A', '♠️').value, 12)
        self.assertEqual(Card('K', '♠️').value, 11)
        
    def test_invalid_card(self):
        """Test invalid card creation."""
        with self.assertRaises(ValueError):
            Card('X', '♠️')
        with self.assertRaises(ValueError):
            Card('A', '🎯')

class TestHandEvaluator(unittest.TestCase):
    """Test hand evaluation functionality."""
    
    def setUp(self):
        self.evaluator = HandEvaluator()
    
    def test_parse_card(self):
        """Test card parsing from string."""
        card = self.evaluator.parse_card('A♠️')
        self.assertEqual(card.rank, 'A')
        self.assertEqual(card.suit, '♠️')
        
        # Test 10 parsing
        card = self.evaluator.parse_card('10♥️')
        self.assertEqual(card.rank, '10')
        self.assertEqual(card.suit, '♥️')
    
    def test_royal_flush(self):
        """Test royal flush evaluation."""
        cards = [
            Card('A', '♠️'), Card('K', '♠️'), Card('Q', '♠️'),
            Card('J', '♠️'), Card('10', '♠️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['royal_flush'])
    
    def test_straight_flush(self):
        """Test straight flush evaluation."""
        cards = [
            Card('9', '♠️'), Card('8', '♠️'), Card('7', '♠️'),
            Card('6', '♠️'), Card('5', '♠️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['straight_flush'])
    
    def test_four_of_a_kind(self):
        """Test four of a kind evaluation."""
        cards = [
            Card('A', '♠️'), Card('A', '♥️'), Card('A', '♦️'),
            Card('A', '♣️'), Card('K', '♠️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['four_of_a_kind'])
        self.assertEqual(tiebreakers[0], 12)  # Aces
    
    def test_full_house(self):
        """Test full house evaluation."""
        cards = [
            Card('A', '♠️'), Card('A', '♥️'), Card('A', '♦️'),
            Card('K', '♠️'), Card('K', '♥️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['full_house'])
    
    def test_flush(self):
        """Test flush evaluation."""
        cards = [
            Card('A', '♠️'), Card('J', '♠️'), Card('9', '♠️'),
            Card('7', '♠️'), Card('5', '♠️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['flush'])
    
    def test_straight(self):
        """Test straight evaluation."""
        cards = [
            Card('A', '♠️'), Card('K', '♥️'), Card('Q', '♦️'),
            Card('J', '♠️'), Card('10', '♣️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['straight'])
        
        # Test wheel straight (A-2-3-4-5)
        cards = [
            Card('A', '♠️'), Card('2', '♥️'), Card('3', '♦️'),
            Card('4', '♠️'), Card('5', '♣️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['straight'])
        self.assertEqual(tiebreakers[0], 3)  # 5-high straight
    
    def test_three_of_a_kind(self):
        """Test three of a kind evaluation."""
        cards = [
            Card('A', '♠️'), Card('A', '♥️'), Card('A', '♦️'),
            Card('K', '♠️'), Card('Q', '♥️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['three_of_a_kind'])
    
    def test_two_pair(self):
        """Test two pair evaluation."""
        cards = [
            Card('A', '♠️'), Card('A', '♥️'), Card('K', '♦️'),
            Card('K', '♠️'), Card('Q', '♥️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['two_pair'])
    
    def test_pair(self):
        """Test pair evaluation."""
        cards = [
            Card('A', '♠️'), Card('A', '♥️'), Card('K', '♦️'),
            Card('Q', '♠️'), Card('J', '♥️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['pair'])
    
    def test_high_card(self):
        """Test high card evaluation."""
        cards = [
            Card('A', '♠️'), Card('K', '♥️'), Card('Q', '♦️'),
            Card('J', '♠️'), Card('9', '♥️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['high_card'])
    
    def test_seven_card_evaluation(self):
        """Test evaluation with 7 cards (finds best 5)."""
        cards = [
            Card('A', '♠️'), Card('A', '♥️'), Card('A', '♦️'),
            Card('K', '♠️'), Card('K', '♥️'), Card('2', '♣️'), Card('3', '♦️')
        ]
        rank, tiebreakers = self.evaluator.evaluate_hand(cards)
        self.assertEqual(rank, self.evaluator.HAND_RANKINGS['full_house'])

class TestDeck(unittest.TestCase):
    """Test deck functionality."""
    
    def test_deck_creation(self):
        """Test deck creation."""
        deck = Deck()
        self.assertEqual(deck.remaining_cards(), 52)
    
    def test_deck_with_removed_cards(self):
        """Test deck creation with removed cards."""
        removed = [Card('A', '♠️'), Card('K', '♠️')]
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
        result = self.solver.analyze_hand(['A♠️', 'A♥️'], 1, simulation_mode="fast")
        
        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.win_probability, 0.8)  # AA should win >80% vs 1 opponent
        self.assertGreater(result.simulations_run, 0)
        self.assertGreater(result.execution_time_ms, 0)
    
    def test_analyze_with_board(self):
        """Test analysis with board cards."""
        # Hero has pocket aces, board has A-K-Q (giving hero trips)
        result = self.solver.analyze_hand(
            ['A♠️', 'A♥️'], 
            2, 
            ['A♦️', 'K♠️', 'Q♥️'],
            simulation_mode="fast"
        )
        
        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.win_probability, 0.8)  # Trip aces should be very strong
    
    def test_analyze_weak_hand(self):
        """Test analysis of weak hand."""
        result = self.solver.analyze_hand(['2♠️', '7♥️'], 5, simulation_mode="fast")
        
        self.assertIsInstance(result, SimulationResult)
        self.assertLess(result.win_probability, 0.3)  # 2-7 offsuit should be weak
    
    def test_invalid_inputs(self):
        """Test invalid input handling."""
        # Wrong number of hole cards
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['A♠️'], 1)
        
        # Too many opponents
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['A♠️', 'K♠️'], 7)
        
        # Wrong number of board cards
        with self.assertRaises(ValueError):
            self.solver.analyze_hand(['A♠️', 'K♠️'], 1, ['A♦️', 'K♦️'])
    
    def test_simulation_modes(self):
        """Test different simulation modes."""
        hand = ['A♠️', 'K♠️']
        
        fast_result = self.solver.analyze_hand(hand, 1, simulation_mode="fast")
        default_result = self.solver.analyze_hand(hand, 1, simulation_mode="default")
        
        # Both should complete successfully (may hit time limits)
        self.assertIsInstance(fast_result, SimulationResult)
        self.assertIsInstance(default_result, SimulationResult)
        self.assertGreater(fast_result.simulations_run, 0)
        self.assertGreater(default_result.simulations_run, 0)
    
    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        result = self.solver.analyze_hand(['A♠️', 'A♥️'], 1, simulation_mode="fast")
        
        if result.confidence_interval:
            lower, upper = result.confidence_interval
            self.assertLessEqual(lower, result.win_probability)
            self.assertGreaterEqual(upper, result.win_probability)
            self.assertLessEqual(lower, upper)

class TestConvenienceFunction(unittest.TestCase):
    """Test the convenience function."""
    
    def test_solve_poker_hand(self):
        """Test the convenience function."""
        result = solve_poker_hand(['A♠️', 'A♥️'], 1, simulation_mode="fast")
        
        self.assertIsInstance(result, SimulationResult)
        self.assertGreater(result.win_probability, 0.8)

class TestKnownScenarios(unittest.TestCase):
    """Test against known poker scenarios."""
    
    def test_pocket_pairs_vs_overcards(self):
        """Test pocket pair vs overcards (classic coin flip)."""
        solver = MonteCarloSolver()
        
        # This is hard to test exactly since we can't control opponent cards
        # But we can test that the solver runs without error
        result = solver.analyze_hand(['Q♠️', 'Q♥️'], 1, simulation_mode="fast")
        self.assertIsInstance(result, SimulationResult)
    
    def test_dominated_hands(self):
        """Test dominated hand scenarios."""
        solver = MonteCarloSolver()
        
        # AK should beat AQ most of the time (when they don't both make pairs)
        ak_result = solver.analyze_hand(['A♠️', 'K♥️'], 1, simulation_mode="fast")
        aq_result = solver.analyze_hand(['A♠️', 'Q♥️'], 1, simulation_mode="fast")
        
        # Both should be reasonable hands, but this is hard to test precisely
        # without controlling opponent cards
        self.assertIsInstance(ak_result, SimulationResult)
        self.assertIsInstance(aq_result, SimulationResult)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 