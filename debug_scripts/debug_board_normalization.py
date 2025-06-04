#!/usr/bin/env python3
"""Debug board normalization issue."""

from poker_knight.storage.unified_cache import CacheKeyNormalizer, create_cache_key

# Test the normalization
board = ["Q♥", "J♦", "10♣"]
print(f"Original board: {board}")

# Normalize each card
normalized_cards = []
for card in board:
    normalized = CacheKeyNormalizer._normalize_card(card)
    print(f"  {card} -> {normalized}")
    normalized_cards.append(normalized)

print(f"Normalized cards: {normalized_cards}")
print(f"Joined: {'_'.join(normalized_cards)}")

# Test with create_cache_key
key = create_cache_key(["A♠", "K♠"], 2, ["Q♥", "J♦", "10♣"])
print(f"\nKey board_cards: {key.board_cards}")
print(f"Expected: T♣_J♦_Q♥")