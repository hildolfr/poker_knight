#!/usr/bin/env python3
"""Debug cache mismatch between board cache and unified cache."""

from poker_knight import solve_poker_hand
from poker_knight.storage.unified_cache import get_unified_cache
from poker_knight.storage.board_cache import get_board_cache

def test_cache_mismatch():
    """Test if board cache abstraction causes mismatched results."""
    print("=== Cache Mismatch Analysis ===\n")
    
    # Clear all caches
    unified_cache = get_unified_cache()
    board_cache = get_board_cache()
    unified_cache.clear()
    board_cache.clear_cache()
    
    # Test scenario 1: Same rank pattern, different suits
    print("Test 1: Same rank pattern, different suits")
    hero_hand = ['J♠', '10♠']
    num_opponents = 2
    
    # Board 1: Rainbow AKQ
    board1 = ['A♠', 'K♥', 'Q♦']
    print(f"\nBoard 1: {board1} (rainbow)")
    result1 = solve_poker_hand(hero_hand, num_opponents, board1)
    print(f"Result 1: Win={result1.win_probability:.4f}, Tie={result1.tie_probability:.4f}")
    
    # Board 2: Different rainbow AKQ (should have different equity due to suit interactions)
    board2 = ['A♥', 'K♦', 'Q♠']
    print(f"\nBoard 2: {board2} (rainbow, different suits)")
    result2 = solve_poker_hand(hero_hand, num_opponents, board2)
    print(f"Result 2: Win={result2.win_probability:.4f}, Tie={result2.tie_probability:.4f}")
    
    print(f"\nResults identical: {result1.win_probability == result2.win_probability}")
    if result1.win_probability != result2.win_probability:
        print(f"Difference: {abs(result1.win_probability - result2.win_probability):.6f}")
    
    # Check what happened in the caches
    print("\nCache analysis:")
    stats = board_cache.get_cache_stats()
    print(f"Board cache: requests={stats['total_requests']}, hits={stats['cache_hits']}")
    
    unified_stats = unified_cache.get_stats()
    print(f"Unified cache: requests={unified_stats.total_requests}, hits={unified_stats.cache_hits}")
    
    # Test scenario 2: Flush vs rainbow (definitely different)
    print("\n\nTest 2: Flush board vs rainbow board")
    
    # Clear caches again
    unified_cache.clear()
    board_cache.clear_cache()
    
    # Board 3: Flush board
    board3 = ['A♠', 'K♠', 'Q♠']
    print(f"\nBoard 3: {board3} (flush)")
    result3 = solve_poker_hand(hero_hand, num_opponents, board3)
    print(f"Result 3: Win={result3.win_probability:.4f}, Tie={result3.tie_probability:.4f}")
    
    # Board 4: Rainbow with same ranks
    board4 = ['A♥', 'K♦', 'Q♣']
    print(f"\nBoard 4: {board4} (rainbow)")
    result4 = solve_poker_hand(hero_hand, num_opponents, board4)
    print(f"Result 4: Win={result4.win_probability:.4f}, Tie={result4.tie_probability:.4f}")
    
    print(f"\nResults identical: {result3.win_probability == result4.win_probability}")
    print(f"Expected different: True (flush vs rainbow)")
    if result3.win_probability != result4.win_probability:
        print(f"Difference: {abs(result3.win_probability - result4.win_probability):.6f}")
    
    # Test scenario 3: Check cache lookup order
    print("\n\nTest 3: Cache lookup order and fallback")
    
    # Clear only board cache, keep unified cache
    board_cache.clear_cache()
    
    # Re-run board4 (should hit unified cache)
    print(f"\nRe-running Board 4: {board4}")
    result4_cached = solve_poker_hand(hero_hand, num_opponents, board4)
    print(f"Result 4 (cached): Win={result4_cached.win_probability:.4f}")
    print(f"Matches original: {result4.win_probability == result4_cached.win_probability}")
    
    # Check if unified cache was hit
    new_unified_stats = unified_cache.get_stats()
    print(f"Unified cache hits increased: {new_unified_stats.cache_hits > unified_stats.cache_hits}")
    
    # Test with slightly different hero hand to see if board cache causes issues
    print("\n\nTest 4: Different hero hand, same board pattern")
    
    hero_hand2 = ['9♠', '8♠']
    
    # Run with board1 
    print(f"\nHero: {hero_hand2}, Board: {board1}")
    result5 = solve_poker_hand(hero_hand2, num_opponents, board1)
    print(f"Result 5: Win={result5.win_probability:.4f}")
    
    # Run with board2 (same pattern, different suits)
    print(f"\nHero: {hero_hand2}, Board: {board2}")  
    result6 = solve_poker_hand(hero_hand2, num_opponents, board2)
    print(f"Result 6: Win={result6.win_probability:.4f}")
    
    print(f"\nResults identical: {result5.win_probability == result6.win_probability}")
    if result5.win_probability != result6.win_probability:
        print(f"Difference: {abs(result5.win_probability - result6.win_probability):.6f}")
        print("This suggests board cache abstraction may be too aggressive")

if __name__ == "__main__":
    test_cache_mismatch()