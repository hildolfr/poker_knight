#!/usr/bin/env python3
"""
Poker Knight v1.2.1 - Monte Carlo Texas Hold'em Poker Solver

A high-performance Monte Carlo poker simulation engine that calculates win probabilities
for Texas Hold'em scenarios with accurate card removal effects and statistical confidence intervals.

Optimized for AI poker systems requiring fast, reliable hand strength analysis.

Author: Assistant
Version: 1.2.1
License: MIT
"""

import json
import random
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from collections import Counter
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import math

# Module metadata
__version__ = "1.2.1"
__author__ = "Assistant"
__license__ = "MIT"
__all__ = [
    "Card", "HandEvaluator", "Deck", "SimulationResult", 
    "MonteCarloSolver", "solve_poker_hand"
]

# Card suits using Unicode emojis
SUITS = ['♠️', '♥️', '♦️', '♣️']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
RANK_VALUES = {rank: i for i, rank in enumerate(RANKS)}

@dataclass
class Card:
    """Represents a playing card with suit and rank."""
    rank: str
    suit: str
    
    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")
        if self.suit not in SUITS:
            raise ValueError(f"Invalid suit: {self.suit}")
    
    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def __eq__(self, other) -> bool:
        return self.rank == other.rank and self.suit == other.suit
    
    @property
    def value(self) -> int:
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
    
    # Precomputed straight patterns for optimization
    _STRAIGHT_PATTERNS = [
        [12, 3, 2, 1, 0],  # A-5 wheel straight (A,2,3,4,5)
        [4, 3, 2, 1, 0],   # 2-6 straight
        [5, 4, 3, 2, 1],   # 3-7 straight
        [6, 5, 4, 3, 2],   # 4-8 straight
        [7, 6, 5, 4, 3],   # 5-9 straight
        [8, 7, 6, 5, 4],   # 6-10 straight
        [9, 8, 7, 6, 5],   # 7-J straight
        [10, 9, 8, 7, 6],  # 8-Q straight
        [11, 10, 9, 8, 7], # 9-K straight
        [12, 11, 10, 9, 8] # 10-A straight
    ]
    
    # High cards for each straight (used for comparison)
    _STRAIGHT_HIGHS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
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
        """Evaluate exactly 5 cards with optimized performance."""
        # Extract rank and suit values once
        ranks = [card.value for card in cards]
        suits = [card.suit for card in cards]
        
        # Quick flush check (most common after high card)
        is_flush = len(set(suits)) == 1
        
        # Count rank frequencies with manual counting for better performance
        rank_counts = [0] * 13
        for rank in ranks:
            rank_counts[rank] += 1
        
        # Find the rank frequency pattern
        counts = sorted(rank_counts, reverse=True)
        four_kind = counts[0] == 4
        three_kind = counts[0] == 3
        pair = counts[0] == 2
        two_pair = counts[0] == 2 and counts[1] == 2
        full_house = three_kind and counts[1] == 2
        
        # Fast path for four of a kind
        if four_kind:
            quad_rank = rank_counts.index(4)
            kicker = rank_counts.index(1)
            return HandEvaluator.HAND_RANKINGS['four_of_a_kind'], [quad_rank, kicker]
        
        # Fast path for full house
        if full_house:
            trips_rank = rank_counts.index(3)
            pair_rank = rank_counts.index(2)
            return HandEvaluator.HAND_RANKINGS['full_house'], [trips_rank, pair_rank]
        
        # Check for straight using precomputed patterns
        rank_set = set(ranks)
        is_straight = False
        straight_high = 0
        
        for i, pattern in enumerate(HandEvaluator._STRAIGHT_PATTERNS):
            if all(rank in rank_set for rank in pattern):
                is_straight = True
                straight_high = HandEvaluator._STRAIGHT_HIGHS[i]
                break
        
        # Handle straight flush and royal flush
        if is_straight and is_flush:
            if straight_high == 12:  # A-high straight flush
                return HandEvaluator.HAND_RANKINGS['royal_flush'], [straight_high]
            else:
                return HandEvaluator.HAND_RANKINGS['straight_flush'], [straight_high]
        
        # Fast path for flush
        if is_flush:
            sorted_ranks_desc = sorted(ranks, reverse=True)
            return HandEvaluator.HAND_RANKINGS['flush'], sorted_ranks_desc
        
        # Fast path for straight
        if is_straight:
            return HandEvaluator.HAND_RANKINGS['straight'], [straight_high]
        
        # Fast path for three of a kind
        if three_kind:
            trips_rank = rank_counts.index(3)
            kickers = [i for i, count in enumerate(rank_counts) if count == 1]
            kickers.sort(reverse=True)
            return HandEvaluator.HAND_RANKINGS['three_of_a_kind'], [trips_rank] + kickers
        
        # Fast path for two pair
        if two_pair:
            pairs = [i for i, count in enumerate(rank_counts) if count == 2]
            pairs.sort(reverse=True)
            kicker = rank_counts.index(1)
            return HandEvaluator.HAND_RANKINGS['two_pair'], pairs + [kicker]
        
        # Fast path for one pair
        if pair:
            pair_rank = rank_counts.index(2)
            kickers = [i for i, count in enumerate(rank_counts) if count == 1]
            kickers.sort(reverse=True)
            return HandEvaluator.HAND_RANKINGS['pair'], [pair_rank] + kickers
        
        # High card
        sorted_ranks_desc = sorted(ranks, reverse=True)
        return HandEvaluator.HAND_RANKINGS['high_card'], sorted_ranks_desc

class Deck:
    """Deck management with card removal tracking."""
    
    def __init__(self, removed_cards: Optional[List[Card]] = None):
        # Pre-allocate full deck for better memory efficiency
        self._full_deck = []
        for suit in SUITS:
            for rank in RANKS:
                self._full_deck.append(Card(rank, suit))
        
        # Create working deck by filtering removed cards
        if removed_cards is None:
            self.cards = self._full_deck.copy()
        else:
            removed_set = set(removed_cards)  # Use set for O(1) lookup
            self.cards = [card for card in self._full_deck if card not in removed_set]
        
        self.shuffle()
    
    def shuffle(self) -> None:
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
    
    def reset_with_removed(self, removed_cards: List[Card]) -> None:
        """Reset deck with new removed cards (avoids object creation)."""
        removed_set = set(removed_cards)
        self.cards = [card for card in self._full_deck if card not in removed_set]
        self.shuffle()

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
        if simulation_mode not in ["fast", "default", "precision"]:
            raise ValueError(f"Invalid simulation_mode '{simulation_mode}'. Must be 'fast', 'default', or 'precision'")
        
        # Parse cards and validate format
        try:
            hero_cards = [self.evaluator.parse_card(card) for card in hero_hand]
            board = [self.evaluator.parse_card(card) for card in board_cards] if board_cards else []
        except ValueError as e:
            raise ValueError(f"Invalid card format: {e}")
        
        # Check for duplicate cards
        all_cards = hero_cards + board
        if len(all_cards) != len(set(all_cards)):
            raise ValueError("Duplicate cards detected in hero hand and/or board cards")
        
        # Determine simulation count
        if simulation_mode == "fast":
            sim_key = "fast_mode_simulations"
        elif simulation_mode == "precision":
            sim_key = "precision_mode_simulations"
        else:
            sim_key = "default_simulations"
        
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
        
        # Get timeout settings - use realistic timeouts based on actual performance
        # Each simulation takes ~0.15ms, so calculate realistic timeouts
        base_timeout = self.config["performance_settings"]["max_simulation_time_ms"]
        if simulation_mode == "fast":
            max_time_ms = 3000  # 3 seconds for 10K sims (~1.5s needed)
        elif simulation_mode == "precision":
            max_time_ms = 120000  # 120 seconds for 500K sims (~75s needed + buffer)
        else:
            max_time_ms = 20000  # 20 seconds for 100K sims (~15s needed)
        
        # Run the target number of simulations with timeout as safety fallback
        if self.config["simulation_settings"].get("parallel_processing", False) and num_simulations >= 1000:
            # Use parallel processing for large simulation counts
            wins, ties, losses, hand_categories = self._run_parallel_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
            )
        else:
            # Use sequential processing for small simulation counts or when disabled
            wins, ties, losses, hand_categories = self._run_sequential_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
            )
        
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
        """Simulate a single hand with memory optimizations."""
        # Create deck with removed cards
        deck = Deck(removed_cards)
        
        # Pre-allocate opponent hands list to avoid resize
        opponent_hands = []
        opponent_hands.extend([deck.deal(2) for _ in range(num_opponents)])
        
        # Complete the board if needed (reuse board list)
        cards_needed = 5 - len(board)
        if cards_needed > 0:
            board_cards = board + deck.deal(cards_needed)
        else:
            board_cards = board
        
        # Evaluate hero hand
        hero_rank, hero_tiebreakers = self.evaluator.evaluate_hand(hero_cards + board_cards)
        
        # Evaluate opponent hands and count results in one pass
        hero_better_count = 0
        hero_tied_count = 0
        
        for opp_cards in opponent_hands:
            opp_rank, opp_tiebreakers = self.evaluator.evaluate_hand(opp_cards + board_cards)
            
            if hero_rank > opp_rank:
                hero_better_count += 1
            elif hero_rank == opp_rank:
                if hero_tiebreakers > opp_tiebreakers:
                    hero_better_count += 1
                elif hero_tiebreakers == opp_tiebreakers:
                    hero_tied_count += 1
        
        # Determine result without intermediate variables
        if hero_better_count == num_opponents:
            result = "win"
        elif hero_better_count + hero_tied_count == num_opponents and hero_tied_count > 0:
            result = "tie"
        else:
            result = "loss"
        
        # Return result with cached hand type lookup
        return {
            "result": result,
            "hero_hand_type": self._get_hand_type_name(hero_rank)
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
    
    def _run_sequential_simulations(self, hero_cards: List[Card], num_opponents: int, 
                                   board: List[Card], removed_cards: List[Card], 
                                   num_simulations: int, max_time_ms: int, start_time: float) -> Tuple[int, int, int, Counter]:
        """Run simulations sequentially with memory optimizations."""
        wins = 0
        ties = 0
        losses = 0
        hand_categories = Counter() if self.config["output_settings"]["include_hand_categories"] else None
        
        # Check timeout every N simulations (configurable for performance)
        timeout_check_interval = min(5000, max(1000, num_simulations // 20))
        
        for sim in range(num_simulations):
            # Optimized timeout check
            if sim > 0 and sim % timeout_check_interval == 0:
                if (time.time() - start_time) * 1000 > max_time_ms:
                    break
            
            result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
            
            # Direct assignment without string comparison
            result_type = result["result"]
            if result_type == "win":
                wins += 1
            elif result_type == "tie":
                ties += 1
            else:
                losses += 1
            
            # Only track hand categories if needed
            if hand_categories is not None:
                hand_categories[result["hero_hand_type"]] += 1
        
        return wins, ties, losses, hand_categories or Counter()
    
    def _run_parallel_simulations(self, hero_cards: List[Card], num_opponents: int, 
                                 board: List[Card], removed_cards: List[Card], 
                                 num_simulations: int, max_time_ms: int, start_time: float) -> Tuple[int, int, int, Counter]:
        """Run simulations in parallel using ThreadPoolExecutor with memory optimizations."""
        import concurrent.futures
        import threading
        
        wins = 0
        ties = 0
        losses = 0
        hand_categories = Counter()
        
        # Determine batch size and number of workers
        num_workers = min(4, max(1, num_simulations // 1000))  # 1-4 workers
        batch_size = num_simulations // num_workers
        remaining = num_simulations % num_workers
        
        # Create batches
        batches = [batch_size] * num_workers
        if remaining > 0:
            batches[-1] += remaining
        
        # Cache config values to avoid repeated lookups
        include_hand_categories = self.config["output_settings"]["include_hand_categories"]
        
        def run_batch(batch_size: int) -> Tuple[int, int, int, Dict[str, int]]:
            """Run a batch of simulations with memory optimizations."""
            batch_wins = 0
            batch_ties = 0
            batch_losses = 0
            batch_categories = Counter() if include_hand_categories else None
            
            # Local timeout check interval for this batch
            batch_timeout_interval = min(1000, max(100, batch_size // 10))
            
            for i in range(batch_size):
                # Optimized timeout check
                if i > 0 and i % batch_timeout_interval == 0:
                    if (time.time() - start_time) * 1000 > max_time_ms:
                        break
                
                result = self._simulate_hand(hero_cards, num_opponents, board, removed_cards)
                
                # Direct assignment without string comparison
                result_type = result["result"]
                if result_type == "win":
                    batch_wins += 1
                elif result_type == "tie":
                    batch_ties += 1
                else:
                    batch_losses += 1
                
                # Only track hand categories if needed
                if batch_categories is not None:
                    batch_categories[result["hero_hand_type"]] += 1
            
            return batch_wins, batch_ties, batch_losses, dict(batch_categories) if batch_categories else {}
        
        # Run batches in parallel
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(run_batch, batch_size) for batch_size in batches]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    batch_wins, batch_ties, batch_losses, batch_categories = future.result()
                    wins += batch_wins
                    ties += batch_ties
                    losses += batch_losses
                    
                    # Merge hand categories efficiently
                    if include_hand_categories and batch_categories:
                        for category, count in batch_categories.items():
                            hand_categories[category] += count
                            
                except Exception as e:
                    # Log error but continue with other batches
                    print(f"Warning: Batch simulation failed: {e}")
        
        return wins, ties, losses, hand_categories

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