"""
Core card representations for Poker Knight.

This module contains the Card and Deck classes used throughout the poker solver.
"""

import random
from dataclasses import dataclass
from typing import List, Optional, Set

from ..constants import SUITS, SUIT_MAPPING, RANKS, RANK_VALUES


@dataclass
class Card:
    """Represents a playing card with suit and rank."""
    rank: str
    suit: str
    
    def __post_init__(self) -> None:
        if self.rank not in RANKS:
            raise ValueError(f"Invalid rank: {self.rank}")
        
        # Normalize suit format for backward compatibility
        if self.suit in SUIT_MAPPING:
            self.suit = SUIT_MAPPING[self.suit]
        elif self.suit not in SUITS:
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


class Deck:
    """Efficient deck of cards with optimized dealing and shuffling."""
    
    def __init__(self, removed_cards: Optional[List[Card]] = None) -> None:
        # Pre-allocate full deck for better memory efficiency
        self.all_cards = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        self.removed_cards: Set[Card] = set(removed_cards) if removed_cards else set()
        self.available_cards = [card for card in self.all_cards if card not in self.removed_cards]
        self.current_index = 0
        
        # Shuffle on initialization
        self.shuffle()
    
    def shuffle(self) -> None:
        """Shuffle the deck using Fisher-Yates algorithm."""
        random.shuffle(self.available_cards)
        self.current_index = 0
    
    def deal(self, num_cards: int) -> List[Card]:
        """Deal specified number of cards."""
        if self.current_index + num_cards > len(self.available_cards):
            raise ValueError("Not enough cards remaining in deck")
        
        cards = self.available_cards[self.current_index:self.current_index + num_cards]
        self.current_index += num_cards
        return cards
    
    def remaining_cards(self) -> int:
        """Get number of cards remaining in deck."""
        return len(self.available_cards) - self.current_index
    
    def reset_with_removed(self, removed_cards: List[Card]) -> None:
        """Reset deck with new set of removed cards."""
        self.removed_cards = set(removed_cards)
        self.available_cards = [card for card in self.all_cards if card not in self.removed_cards]
        self.shuffle()