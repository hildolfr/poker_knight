#!/usr/bin/env python3
"""Debug cache instance management."""

import time
import gc
from poker_knight import MonteCarloSolver
from poker_knight.storage.unified_cache import ThreadSafeMonteCarloCache, get_unified_cache
from poker_knight.storage.board_cache import get_board_cache
from poker_knight.storage.preflop_cache import get_preflop_cache

def check_cache_instances():
    """Check if multiple cache instances are being created."""
    print("=== Cache Instance Analysis ===\n")
    
    # Force garbage collection to clean up any existing instances
    gc.collect()
    
    print("1. Creating first solver instance...")
    solver1 = MonteCarloSolver()
    solver1._initialize_cache_if_needed()
    
    print(f"   Unified cache: {id(solver1._unified_cache)}")
    print(f"   Board cache: {id(solver1._board_cache)}")
    print(f"   Preflop cache: {id(solver1._preflop_cache)}")
    
    # Get global cache instances
    print("\n2. Getting global cache instances...")
    global_unified = get_unified_cache()
    global_board = get_board_cache()
    global_preflop = get_preflop_cache()
    
    print(f"   Global unified: {id(global_unified)}")
    print(f"   Global board: {id(global_board)}")
    print(f"   Global preflop: {id(global_preflop)}")
    
    # Check if they're the same
    print("\n3. Checking if instances match...")
    print(f"   Unified match: {solver1._unified_cache is global_unified}")
    print(f"   Board match: {solver1._board_cache is global_board}")
    print(f"   Preflop match: {solver1._preflop_cache is global_preflop}")
    
    # Create second solver
    print("\n4. Creating second solver instance...")
    solver2 = MonteCarloSolver()
    solver2._initialize_cache_if_needed()
    
    print(f"   Solver2 unified cache: {id(solver2._unified_cache)}")
    print(f"   Solver2 board cache: {id(solver2._board_cache)}")
    print(f"   Solver2 preflop cache: {id(solver2._preflop_cache)}")
    
    # Check if second solver uses same caches
    print("\n5. Checking if solver2 uses same cache instances...")
    print(f"   Unified same: {solver1._unified_cache is solver2._unified_cache}")
    print(f"   Board same: {solver1._board_cache is solver2._board_cache}")
    print(f"   Preflop same: {solver1._preflop_cache is solver2._preflop_cache}")
    
    # Test with actual simulation
    print("\n6. Testing with actual simulations...")
    hero_hand = ['K♠', 'Q♠']
    num_opponents = 2
    board_cards = ['A♠', 'J♠', '10♥']
    
    # Clear caches
    if solver1._unified_cache:
        solver1._unified_cache.clear()
    if solver1._board_cache:
        solver1._board_cache.clear_cache()
    
    # Run with solver1
    print("\n   Running simulation with solver1...")
    result1 = solver1.analyze_hand(hero_hand, num_opponents, board_cards)
    print(f"   Result1: Win={result1.win_probability:.4f}")
    
    # Check cache stats
    if solver1._unified_cache:
        stats1 = solver1._unified_cache.get_stats()
        print(f"   Solver1 cache stats: hits={stats1.cache_hits}, misses={stats1.cache_misses}")
    
    # Run with solver2 (should hit cache)
    print("\n   Running simulation with solver2...")
    result2 = solver2.analyze_hand(hero_hand, num_opponents, board_cards)
    print(f"   Result2: Win={result2.win_probability:.4f}")
    
    # Check cache stats again
    if solver2._unified_cache:
        stats2 = solver2._unified_cache.get_stats()
        print(f"   Solver2 cache stats: hits={stats2.cache_hits}, misses={stats2.cache_misses}")
    
    # Check if results are identical
    print(f"\n   Results identical: {result1.win_probability == result2.win_probability}")
    
    # Check board cache unified cache references
    print("\n7. Checking board cache's unified cache reference...")
    if solver1._board_cache:
        board_unified = solver1._board_cache.unified_cache
        print(f"   Board cache's unified cache: {id(board_unified)}")
        print(f"   Same as solver's unified: {board_unified is solver1._unified_cache}")
        print(f"   Same as global unified: {board_unified is global_unified}")

if __name__ == "__main__":
    check_cache_instances()