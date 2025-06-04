"""
Hand evaluation logic for Poker Knight.

This module contains the HandEvaluator class for evaluating poker hands.
"""

import itertools
from collections import Counter
from typing import List, Tuple

from .cards import Card
from ..constants import HAND_RANKINGS, STRAIGHT_PATTERNS, STRAIGHT_HIGHS


class HandEvaluator:
    """Fast Texas Hold'em hand evaluation."""
    
    # Pre-allocated arrays for hot path optimization (avoid repeated allocation)
    _temp_pairs = [0] * 2
    _temp_kickers = [0] * 5
    _temp_sorted_ranks = [0] * 5
    
    @staticmethod
    def parse_card(card_str: str) -> Card:
        """Parse card string like 'A♠' or 'AS' into Card object."""
        # Handle 10 specially since it's two characters
        if card_str.startswith('10'):
            rank = '10'
            suit = card_str[2:]
        else:
            rank = card_str[0]
            suit = card_str[1:]
        
        # The Card class will normalize the suit in __post_init__
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
        
        # Use collections.Counter for optimized rank counting (C implementation)
        rank_counter = Counter(ranks)
        rank_counts = rank_counter.most_common()
        
        # Extract count pattern efficiently
        if len(rank_counts) == 2:  # Four of a kind or full house
            if rank_counts[0][1] == 4:
                # Four of a kind
                quad_rank = rank_counts[0][0]
                kicker = rank_counts[1][0]
                return HAND_RANKINGS['four_of_a_kind'], [quad_rank, kicker]
            else:
                # Full house (3,2)
                trips_rank = rank_counts[0][0]
                pair_rank = rank_counts[1][0]
                return HAND_RANKINGS['full_house'], [trips_rank, pair_rank]
        
        elif len(rank_counts) == 3:  # Two pair or three of a kind
            if rank_counts[0][1] == 3:
                # Three of a kind
                trips_rank = rank_counts[0][0]
                kickers = sorted([rank_counts[1][0], rank_counts[2][0]], reverse=True)
                return HAND_RANKINGS['three_of_a_kind'], [trips_rank] + kickers
            else:
                # Two pair
                pair1 = max(rank_counts[0][0], rank_counts[1][0])
                pair2 = min(rank_counts[0][0], rank_counts[1][0])
                kicker = rank_counts[2][0]
                return HAND_RANKINGS['two_pair'], [pair1, pair2, kicker]
        
        elif len(rank_counts) == 4:  # One pair
            pair_rank = rank_counts[0][0]
            kickers = sorted([rank_counts[1][0], rank_counts[2][0], rank_counts[3][0]], reverse=True)
            return HAND_RANKINGS['pair'], [pair_rank] + kickers
        
        else:  # High card, straight, flush, or straight flush
            sorted_ranks = sorted(ranks, reverse=True)
            
            # Check for straight
            is_straight = False
            straight_high = 0
            
            # Check each straight pattern
            for i, pattern in enumerate(STRAIGHT_PATTERNS):
                if all(rank in ranks for rank in pattern):
                    is_straight = True
                    straight_high = STRAIGHT_HIGHS[i]
                    break
            
            if is_straight and is_flush:
                if straight_high == 12:  # A-high straight flush = royal flush
                    return HAND_RANKINGS['royal_flush'], [straight_high]
                else:
                    return HAND_RANKINGS['straight_flush'], [straight_high]
            elif is_flush:
                return HAND_RANKINGS['flush'], sorted_ranks
            elif is_straight:
                return HAND_RANKINGS['straight'], [straight_high]
            else:
                return HAND_RANKINGS['high_card'], sorted_ranks