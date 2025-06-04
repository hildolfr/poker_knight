#!/usr/bin/env python3
"""Debug why cached results are not identical."""

import pickle
import time
from poker_knight import solve_poker_hand
from poker_knight.storage.unified_cache import create_cache_key, CacheKeyNormalizer, ThreadSafeMonteCarloCache
from poker_knight.storage.board_cache import get_board_cache, BoardAnalyzer
from poker_knight.storage.preflop_cache import get_preflop_cache

def analyze_cache_behavior():
    """Analyze cache storage and retrieval consistency."""
    print("=== Cache Consistency Analysis ===\n")
    
    # Test scenario
    hero_hand = ['A♠', 'K♠']
    num_opponents = 2
    board_cards = ['Q♠', 'J♠', '10♥']
    simulation_mode = "default"
    
    print(f"Test Scenario:")
    print(f"  Hero: {hero_hand}")
    print(f"  Opponents: {num_opponents}")
    print(f"  Board: {board_cards}")
    print(f"  Mode: {simulation_mode}\n")
    
    # Clear all caches to start fresh
    unified_cache = ThreadSafeMonteCarloCache(enable_persistence=True)
    board_cache = get_board_cache(unified_cache=unified_cache)
    
    unified_cache.clear()
    board_cache.clear_cache()
    
    # Run simulation to populate cache
    print("1. Running initial simulation to populate cache...")
    result1 = solve_poker_hand(hero_hand, num_opponents, board_cards, simulation_mode)
    print(f"   Win: {result1.win_probability:.4f}")
    print(f"   Tie: {result1.tie_probability:.4f}")
    print(f"   Loss: {result1.loss_probability:.4f}")
    print(f"   Simulations: {result1.simulations_run}\n")
    
    # Check what's in the caches
    print("2. Checking cache contents...")
    
    # Check unified cache
    cache_key = create_cache_key(hero_hand, num_opponents, board_cards, simulation_mode)
    print(f"   Cache key: {cache_key}")
    
    unified_result = unified_cache.get(cache_key)
    if unified_result:
        print(f"   Unified cache hit:")
        print(f"     Win: {unified_result.win_probability:.4f}")
        print(f"     Tie: {unified_result.tie_probability:.4f}")
        print(f"     Loss: {unified_result.loss_probability:.4f}")
        print(f"     Timestamp: {unified_result.timestamp}")
    else:
        print(f"   Unified cache miss!")
    
    # Check board cache
    board_result = board_cache.get_board_result(hero_hand, num_opponents, board_cards, simulation_mode)
    if board_result:
        print(f"\n   Board cache hit:")
        print(f"     Win: {board_result.win_probability:.4f}")
        print(f"     Tie: {board_result.tie_probability:.4f}")
        print(f"     Loss: {board_result.loss_probability:.4f}")
        print(f"     Timestamp: {board_result.timestamp}")
    else:
        print(f"\n   Board cache miss!")
    
    # Analyze board pattern normalization
    print("\n3. Analyzing board pattern normalization...")
    board_pattern = BoardAnalyzer.analyze_board(board_cards)
    print(f"   Board stage: {board_pattern.stage}")
    print(f"   Board texture: {board_pattern.texture}")
    print(f"   Rank pattern: {board_pattern.rank_pattern}")
    print(f"   Suit pattern: {board_pattern.suit_pattern}")
    
    # Check how board cache creates keys
    print("\n4. Checking board cache key generation...")
    board_cache_key = board_cache._create_board_cache_key(
        hero_hand, num_opponents, board_pattern, simulation_mode
    )
    print(f"   Board cache key: {board_cache_key}")
    print(f"   Board representation: {board_cache_key.board_cards}")
    
    # Run second simulation to check cache hit
    print("\n5. Running second simulation (should be cached)...")
    result2 = solve_poker_hand(hero_hand, num_opponents, board_cards, simulation_mode)
    print(f"   Win: {result2.win_probability:.4f}")
    print(f"   Tie: {result2.tie_probability:.4f}")
    print(f"   Loss: {result2.loss_probability:.4f}")
    print(f"   Simulations: {result2.simulations_run}")
    
    # Compare results
    print("\n6. Comparing results...")
    if (result1.win_probability == result2.win_probability and 
        result1.tie_probability == result2.tie_probability and
        result1.loss_probability == result2.loss_probability):
        print("   ✓ Results are IDENTICAL")
    else:
        print("   ✗ Results are DIFFERENT!")
        print(f"   Win diff: {abs(result1.win_probability - result2.win_probability):.6f}")
        print(f"   Tie diff: {abs(result1.tie_probability - result2.tie_probability):.6f}")
        print(f"   Loss diff: {abs(result1.loss_probability - result2.loss_probability):.6f}")
    
    # Check cache statistics
    print("\n7. Cache statistics...")
    unified_stats = unified_cache.get_stats()
    print(f"   Unified cache:")
    print(f"     Total requests: {unified_stats.total_requests}")
    print(f"     Cache hits: {unified_stats.cache_hits}")
    print(f"     Cache misses: {unified_stats.cache_misses}")
    print(f"     Hit rate: {unified_stats.hit_rate:.2%}")
    
    board_stats = board_cache.get_cache_stats()
    print(f"\n   Board cache:")
    print(f"     Total requests: {board_stats['total_requests']}")
    print(f"     Cache hits: {board_stats['cache_hits']}")
    print(f"     Cache misses: {board_stats['cache_misses']}")
    print(f"     Hit rate: {board_stats['hit_rate']:.2%}")
    
    # Test multiple runs to see if results vary
    print("\n8. Testing multiple cache retrievals...")
    results = []
    for i in range(5):
        r = solve_poker_hand(hero_hand, num_opponents, board_cards, simulation_mode)
        results.append((r.win_probability, r.tie_probability, r.loss_probability))
        print(f"   Run {i+1}: Win={r.win_probability:.4f}, Tie={r.tie_probability:.4f}, Loss={r.loss_probability:.4f}")
    
    # Check if all results are identical
    first_result = results[0]
    all_identical = all(r == first_result for r in results)
    print(f"\n   All results identical: {all_identical}")
    
    # Check for floating point precision issues
    if not all_identical:
        print("\n9. Analyzing floating point precision...")
        for i, r in enumerate(results):
            if r != first_result:
                win_diff = abs(r[0] - first_result[0])
                tie_diff = abs(r[1] - first_result[1])
                loss_diff = abs(r[2] - first_result[2])
                print(f"   Result {i+1} differences:")
                print(f"     Win: {win_diff:.10f}")
                print(f"     Tie: {tie_diff:.10f}")
                print(f"     Loss: {loss_diff:.10f}")

if __name__ == "__main__":
    analyze_cache_behavior()