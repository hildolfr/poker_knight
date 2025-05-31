#!/usr/bin/env python3
"""
Test enhanced input validation
"""

from poker_solver import solve_poker_hand

def test_validation():
    print("Testing enhanced input validation...")
    
    # Test duplicate cards
    try:
        result = solve_poker_hand(['A♠️', 'A♠️'], 1)
        print("ERROR: Duplicate cards should have been caught!")
    except ValueError as e:
        print(f"✅ Duplicate cards caught: {e}")
    
    # Test invalid simulation mode
    try:
        result = solve_poker_hand(['A♠️', 'K♥️'], 1, simulation_mode='invalid')
        print("ERROR: Invalid simulation mode should have been caught!")
    except ValueError as e:
        print(f"✅ Invalid simulation mode caught: {e}")
    
    # Test duplicate between hero and board
    try:
        result = solve_poker_hand(['A♠️', 'K♥️'], 1, ['A♠️', 'Q♦️', 'J♣️'])
        print("ERROR: Duplicate between hero and board should have been caught!")
    except ValueError as e:
        print(f"✅ Hero/board duplicate caught: {e}")
    
    # Test valid input still works
    try:
        result = solve_poker_hand(['A♠️', 'K♥️'], 1, simulation_mode='fast')
        print(f"✅ Valid input works: {result.win_probability:.1%} win rate")
    except Exception as e:
        print(f"ERROR: Valid input failed: {e}")

if __name__ == "__main__":
    test_validation() 