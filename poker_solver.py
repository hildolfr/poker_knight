#!/usr/bin/env python3
"""
Poker Knight v1.0.0 - Monte Carlo Texas Hold'em Poker Solver

A high-performance Monte Carlo simulation engine for analyzing Texas Hold'em poker hands.
Designed for integration into AI poker players and real-time gameplay decision making.

Author: AI Assistant
License: MIT
Version: 1.0.0
"""

import json
import random
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import Counter
import itertools
from concurrent.futures import ThreadPoolExecutor
import math

# Card suits using Unicode emojis
SUITS = ['♠️', '♥️', '♦️', '♣️']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}

@dataclass
class Card:
    """Represents a playing card with suit and rank."""
    rank: str
    suit: str
    
    def __post_init__(self):
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit}")
    
    def __str__(self):
        return f"{self.rank}{self.suit}"
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    @property
    def value(self):
        """Numeric value for comparison (2=0, 3=1, ..., A=12)."""
        return RANK_VALUES[self.rank]

class HandEvaluator:
    """Fast Texas Hold'em hand evaluation."""
    
    # Hand rankings (higher number = better hand)
    HAND_RANKINGS = {
        'high_card': 1,
        'pair': 2,
        'two_pair': 3,
        'three_of_a_kind': 4,
        'straight': 5,
        'flush': 6,
        'full_house': 7,
        'four_of_a_kind': 8,
        'straight_flush': 9,
        'royal_flush': 10
    }
    
    @staticmethod
    def parse_card(card_str: str) -> Card:
        """Parse card string like 'A♠️' into Card object."""
        # Handle 10 specially since it's two characters
        if card_str.startswith('10'):
            rank = '10'
            suit = card_str[2:]
        else:
            rank = card_str[0]
            suit = card_str[1:]
        
        return Card(rank, suit)
    
    @staticmethod
    def evaluate_hand(cards: List[Card]) -> Tuple[int, List[int]]:
        """
        Evaluate a 5-7 card hand and return (hand_rank, tiebreakers).
        Returns tuple where first element is hand ranking, second is list of tiebreaker values.
        """
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate hand")
        
        # For 6 or 7 cards, find the best 5-card combination
        if len(cards) > 5:
            best_rank = 0
            best_tiebreakers = []
            
            for combo in itertools.combinations(cards, 5):
                rank, tiebreakers = HandEvaluator._evaluate_five_cards(list(combo))
                if rank > best_rank or (rank == best_rank and tiebreakers > best_tiebreakers):
                    best_rank = rank
                    best_tiebreakers = tiebreakers
            
            return best_rank, best_tiebreakers
        
        return HandEvaluator._evaluate_five_cards(cards)
    
    @staticmethod
    def _evaluate_five_cards(cards: List[Card]) -> Tuple[int, List[int]]:
        """Evaluate exactly 5 cards."""
        ranks = [card.value for card in cards]
        suits = [card.suit for card in cards]
        
        rank_counts = Counter(ranks)
        suit_counts = Counter(suits)
        
        # Check for flush
        is_flush = len(suit_counts) == 1
        
        # Check for straight
        sorted_ranks = sorted(set(ranks))
        is_straight = False
        straight_high = 0
        
        if len(sorted_ranks) == 5 and sorted_ranks[-1] - sorted_ranks[0] == 4:
            is_straight = True
            straight_high = sorted_ranks[-1]
        # Check for A-2-3-4-5 straight (wheel)
        elif sorted_ranks == [0, 1, 2, 3, 12]:  # 2,3,4,5,A
            is_straight = True
            straight_high = 3  # 5-high straight
        
        # Get rank frequency groups
        rank_groups = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Determine hand type and tiebreakers
        if is_straight and is_flush:
            if straight_high == 12:  # A-high straight flush
                return HandEvaluator.HAND_RANKINGS['royal_flush'], [straight_high]
            else:
                return HandEvaluator.HAND_RANKINGS['straight_flush'], [straight_high]
        
        elif rank_groups[0][1] == 4:  # Four of a kind
            quad_rank = rank_groups[0][0]
            kicker = rank_groups[1][0]
            return HandEvaluator.HAND_RANKINGS['four_of_a_kind'], [quad_rank, kicker]
        
        elif rank_groups[0][1] == 3 and rank_groups[1][1] == 2:  # Full house
            trips_rank = rank_groups[0][0]
            pair_rank = rank_groups[1][0]
            return HandEvaluator.HAND_RANKINGS['full_house'], [trips_rank, pair_rank]
        
        elif is_flush:
            sorted_ranks_desc = sorted(ranks, reverse=True)
            return HandEvaluator.HAND_RANKINGS['flush'], sorted_ranks_desc
        
        elif is_straight:
            return HandEvaluator.HAND_RANKINGS['straight'], [straight_high]
        
        elif rank_groups[0][1] == 3:  # Three of a kind
            trips_rank = rank_groups[0][0]
            kickers = sorted([rank_groups[1][0], rank_groups[2][0]], reverse=True)
            return HandEvaluator.HAND_RANKINGS['three_of_a_kind'], [trips_rank] + kickers
        
        elif rank_groups[0][1] == 2 and rank_groups[1][1] == 2:  # Two pair
            high_pair = max(rank_groups[0][0], rank_groups[1][0])
            low_pair = min(rank_groups[0][0], rank_groups[1][0])
            kicker = rank_groups[2][0]
            return HandEvaluator.HAND_RANKINGS['two_pair'], [high_pair, low_pair, kicker]
        
        elif rank_groups[0][1] == 2:  # One pair
            pair_rank = rank_groups[0][0]
            kickers = sorted([rank_groups[i][0] for i in range(1, 4)], reverse=True)
            return HandEvaluator.HAND_RANKINGS['pair'], [pair_rank] + kickers
        
        else:  # High card
            sorted_ranks_desc = sorted(ranks, reverse=True)
            return HandEvaluator.HAND_RANKINGS['high_card'], sorted_ranks_desc

class Deck:
    """Deck management with card removal tracking."""
    
    def __init__(self, removed_cards: Optional[List[Card]] = None):
        self.cards = []
        for suit in SUITS:
            for rank in RANKS:
                card = Card(rank, suit)
                if removed_cards is None or card not in removed_cards:
                    self.cards.append(card)
        
        self.shuffle()
    
    def shuffle(self):
        """Shuffle the deck."""
        random.shuffle(self.cards)
    
    def deal(self, num_cards: int) -> List[Card]:
        """Deal specified number of cards."""
        if num_cards > len(self.cards):
            raise ValueError(f"Cannot deal {num_cards} cards, only {len(self.cards)} remaining")
        
        dealt = self.cards[:num_cards]
        self.cards = self.cards[num_cards:]
        return dealt
    
    def remaining_cards(self) -> int:
        """Number of cards remaining in deck."""
        return len(self.cards)

@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    win_probability: float
    tie_probability: float
    loss_probability: float
    simulations_run: int
    execution_time_ms: float
    confidence_interval: Optional[Tuple[float, float]] = None
    hand_category_frequencies: Optional[Dict[str, float]] = None

class MonteCarloSolver:
    """Monte Carlo poker solver for Texas Hold'em."""
    
    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.evaluator = HandEvaluator()
    
    def analyze_hand(self, 
                    hero_hand: List[str], 
                    num_opponents: int,
                    board_cards: Optional[List[str]] = None,
                    simulation_mode: str = "default") -> SimulationResult:
        """
        Analyze hand strength using Monte Carlo simulation.
        
        Args:
            hero_hand: List of 2 card strings (e.g., ['A♠️', 'K♥️'])
            num_opponents: Number of opponents (2-7 total players)
            board_cards: Optional board cards (3-5 cards for flop/turn/river)
            simulation_mode: "fast", "default", or "precision"
        
        Returns:
            SimulationResult with win probability and additional metrics
        """
        start_time = time.time()
        
        # Validate inputs
        if len(hero_hand) != 2:
            raise ValueError("Hero hand must contain exactly 2 cards")
        if not (1 <= num_opponents <= 6):
            raise ValueError("Number of opponents must be between 1 and 6")
        if board_cards and not (3 <= len(board_cards) <= 5):
            raise ValueError("Board cards must be 3-5 cards if provided")
        
        # Parse cards
        hero_cards = [self.evaluator.parse_card(card) for card in hero_hand]
        board = [self.evaluator.parse_card(card) for card in board_cards] if board_cards else []
        
        # Determine simulation count
        sim_key = f"{simulation_mode}_simulations"
        if sim_key in self.config["simulation_settings"]:
            num_simulations = self.config["simulation_settings"][sim_key]
        else:
            num_simulations = self.config["simulation_settings"]["default_simulations"]
        
        # Track removed cards for accurate simulation
        removed_cards = hero_cards + board
        
        # Run simulations
        wins = 0
        ties = 0
        losses = 0
        hand_categories = Counter()
        
        max_time_ms = self.config["performance_settings"]["max_simulation_time_ms"]
        
        for sim in range(num_simulations):
            # Check time limit
            if (time.time() - start_time) * 1000 > max_time_ms:
                num_simulations = sim
                break
            
            result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
            
            if result["result"] == "win":
                wins += 1
            elif result["result"] == "tie":
                ties += 1
            else:
                losses += 1
            
            if self.config["output_settings"]["include_hand_categories"]:
                hand_categories[result["hero_hand_type"]] += 1
        
        execution_time = (time.time() - start_time) * 1000
        
        # Calculate probabilities
        total_sims = wins + ties + losses
        win_prob = wins / total_sims if total_sims > 0 else 0
        tie_prob = ties / total_sims if total_sims > 0 else 0
        loss_prob = losses / total_sims if total_sims > 0 else 0
        
        # Calculate confidence interval if requested
        confidence_interval = None
        if self.config["output_settings"]["include_confidence_interval"] and total_sims > 0:
            confidence_interval = self._calculate_confidence_interval(win_prob, total_sims)
        
        # Convert hand categories to frequencies
        hand_category_frequencies = None
        if self.config["output_settings"]["include_hand_categories"] and total_sims > 0:
            hand_category_frequencies = {
                category: count / total_sims 
                for category, count in hand_categories.items()
            }
        
        return SimulationResult(
            win_probability=round(win_prob, self.config["output_settings"]["decimal_precision"]),
            tie_probability=round(tie_prob, self.config["output_settings"]["decimal_precision"]),
            loss_probability=round(loss_prob, self.config["output_settings"]["decimal_precision"]),
            simulations_run=total_sims,
            execution_time_ms=round(execution_time, 2),
            confidence_interval=confidence_interval,
            hand_category_frequencies=hand_category_frequencies
        )
    
    def _simulate_hand(self, hero_cards: List[Card], num_opponents: int, 
                      board: List[Card], removed_cards: List[Card]) -> Dict[str, Any]:
        """Simulate a single hand."""
        # Create deck with removed cards
        deck = Deck(removed_cards)
        
        # Deal opponent hands
        opponent_hands = []
        for _ in range(num_opponents):
            opponent_hands.append(deck.deal(2))
        
        # Complete the board if needed
        current_board = board.copy()
        cards_needed = 5 - len(current_board)
        if cards_needed > 0:
            current_board.extend(deck.deal(cards_needed))
        
        # Evaluate all hands
        hero_full_hand = hero_cards + current_board
        hero_rank, hero_tiebreakers = self.evaluator.evaluate_hand(hero_full_hand)
        
        opponent_results = []
        for opp_cards in opponent_hands:
            opp_full_hand = opp_cards + current_board
            opp_rank, opp_tiebreakers = self.evaluator.evaluate_hand(opp_full_hand)
            opponent_results.append((opp_rank, opp_tiebreakers))
        
        # Determine result
        hero_better_count = 0
        hero_tied_count = 0
        
        for opp_rank, opp_tiebreakers in opponent_results:
            if hero_rank > opp_rank:
                hero_better_count += 1
            elif hero_rank == opp_rank:
                if hero_tiebreakers > opp_tiebreakers:
                    hero_better_count += 1
                elif hero_tiebreakers == opp_tiebreakers:
                    hero_tied_count += 1
        
        # Determine overall result
        if hero_better_count == num_opponents:
            result = "win"
        elif hero_better_count + hero_tied_count == num_opponents and hero_tied_count > 0:
            result = "tie"
        else:
            result = "loss"
        
        # Get hand category name
        hand_type = self._get_hand_type_name(hero_rank)
        
        return {
            "result": result,
            "hero_hand_type": hand_type
        }
    
    def _get_hand_type_name(self, hand_rank: int) -> str:
        """Convert hand rank to readable name."""
        for name, rank in self.evaluator.HAND_RANKINGS.items():
            if rank == hand_rank:
                return name
        return "unknown"
    
    def _calculate_confidence_interval(self, win_prob: float, sample_size: int, 
                                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for win probability."""
        if sample_size == 0:
            return (0.0, 0.0)
        
        # Use normal approximation for binomial proportion
        z_score = 1.96  # 95% confidence
        if confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645
        
        margin_of_error = z_score * math.sqrt((win_prob * (1 - win_prob)) / sample_size)
        
        lower_bound = max(0, win_prob - margin_of_error)
        upper_bound = min(1, win_prob + margin_of_error)
        
        precision = self.config["output_settings"]["decimal_precision"]
        return (round(lower_bound, precision), round(upper_bound, precision))

# Convenience function for easy usage
def solve_poker_hand(hero_hand: List[str], 
                    num_opponents: int,
                    board_cards: Optional[List[str]] = None,
                    simulation_mode: str = "default") -> SimulationResult:
    """
    Convenience function to analyze a poker hand.
    
    Args:
        hero_hand: List of 2 card strings (e.g., ['A♠️', 'K♥️'])
        num_opponents: Number of opponents (1-6)
        board_cards: Optional board cards (3-5 cards)
        simulation_mode: "fast", "default", or "precision"
    
    Returns:
        SimulationResult with analysis
    """
    solver = MonteCarloSolver()
    return solver.analyze_hand(hero_hand, num_opponents, board_cards, simulation_mode) 