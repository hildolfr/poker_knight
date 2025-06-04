#!/usr/bin/env python3
"""Debug board cache key generation and normalization."""

from poker_knight.storage.board_cache import BoardAnalyzer, get_board_cache
from poker_knight.storage.unified_cache import create_cache_key, CacheKeyNormalizer

def test_board_cache_keys():
    """Test how board cache creates keys for different scenarios."""
    print("=== Board Cache Key Analysis ===\n")
    
    board_cache = get_board_cache()
    
    # Test different board representations that should be the same
    test_cases = [
        # Same board, different order
        (['A♠', 'K♥', 'Q♦'], ['Q♦', 'A♠', 'K♥']),
        (['A♠', 'K♥', 'Q♦'], ['K♥', 'Q♦', 'A♠']),
        
        # Same ranks, different suits (should these be same?)
        (['A♠', 'K♥', 'Q♦'], ['A♥', 'K♦', 'Q♠']),
        
        # Flush boards
        (['A♠', 'K♠', 'Q♠'], ['Q♠', 'K♠', 'A♠']),
        
        # Paired boards
        (['A♠', 'A♥', 'K♦'], ['A♥', 'K♦', 'A♠']),
    ]
    
    for i, (board1, board2) in enumerate(test_cases):
        print(f"Test case {i+1}:")
        print(f"  Board 1: {board1}")
        print(f"  Board 2: {board2}")
        
        # Analyze patterns
        pattern1 = BoardAnalyzer.analyze_board(board1)
        pattern2 = BoardAnalyzer.analyze_board(board2)
        
        print(f"  Pattern 1: {pattern1.rank_pattern}, {pattern1.suit_pattern}")
        print(f"  Pattern 2: {pattern2.rank_pattern}, {pattern2.suit_pattern}")
        
        # Create cache keys
        hero_hand = ['J♠', 'T♠']
        num_opponents = 2
        
        cache_key1 = board_cache._create_board_cache_key(
            hero_hand, num_opponents, pattern1, "default"
        )
        cache_key2 = board_cache._create_board_cache_key(
            hero_hand, num_opponents, pattern2, "default"
        )
        
        print(f"  Cache key 1: {cache_key1.board_cards}")
        print(f"  Cache key 2: {cache_key2.board_cards}")
        print(f"  Keys match: {cache_key1.board_cards == cache_key2.board_cards}")
        
        # Also check unified cache keys
        unified_key1 = create_cache_key(hero_hand, num_opponents, board1, "default")
        unified_key2 = create_cache_key(hero_hand, num_opponents, board2, "default")
        
        print(f"  Unified key 1: {unified_key1.board_cards}")
        print(f"  Unified key 2: {unified_key2.board_cards}")
        print(f"  Unified keys match: {unified_key1.board_cards == unified_key2.board_cards}")
        print()
    
    # Test preflop scenarios
    print("Testing preflop scenarios:")
    preflop_boards = [None, []]
    
    for board in preflop_boards:
        pattern = BoardAnalyzer.analyze_board(board)
        print(f"  Board: {board}")
        print(f"  Pattern: stage={pattern.stage}, texture={pattern.texture}")
        print(f"  Rank pattern: {pattern.rank_pattern}")
        
        cache_key = board_cache._create_board_cache_key(
            ['A♠', 'K♠'], 2, pattern, "default"
        )
        print(f"  Cache key: {cache_key.board_cards}")
        print()
    
    # Test how normalize_board handles different inputs
    print("Testing board normalization:")
    test_boards = [
        ['A♠', 'K♥', 'Q♦'],
        ['Q♦', 'K♥', 'A♠'],  # Different order
        ['AS', 'KH', 'QD'],   # Letter suits
        ['10♠', 'J♥', 'Q♦'],  # With 10
    ]
    
    for board in test_boards:
        try:
            normalized = CacheKeyNormalizer.normalize_board(board)
            print(f"  {board} -> {normalized}")
        except Exception as e:
            print(f"  {board} -> ERROR: {e}")

if __name__ == "__main__":
    test_board_cache_keys()