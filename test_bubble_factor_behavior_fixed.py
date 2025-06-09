#!/usr/bin/env python3
"""
Test bubble_factor behavior under different tournament conditions.
This script verifies whether bubble_factor actually affects equity calculations
differently based on tournament context.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight import solve_poker_hand

def test_bubble_factor_impact():
    """Test if bubble_factor produces different results in different tournament contexts."""
    
    print("Testing bubble_factor behavior under different tournament conditions\n")
    print("=" * 70)
    
    # Test scenario: AA vs 3 opponents on the bubble
    hero_hand = ['A♠', 'A♥']
    num_opponents = 3
    board = []  # Pre-flop
    
    # Test 1: No bubble factor (normal play)
    print("\nTest 1: No bubble factor (bubble_factor = 1.0)")
    print("-" * 50)
    result_no_bubble = solve_poker_hand(
        hero_hand,
        num_opponents,
        board,
        simulation_mode="fast",
        stack_sizes=[5000, 4000, 3000, 2000],
        pot_size=300,  # Add pot size for ICM calculations
        tournament_context={'bubble_factor': 1.0}
    )
    print(f"Win probability: {result_no_bubble.win_probability:.4f}")
    print(f"ICM equity: {result_no_bubble.icm_equity:.4f}" if result_no_bubble.icm_equity else "ICM equity: Not calculated")
    print(f"Bubble factor: {result_no_bubble.bubble_factor}" if result_no_bubble.bubble_factor else "Bubble factor: Not set")
    
    # Test 2: Moderate bubble pressure
    print("\nTest 2: Moderate bubble pressure (bubble_factor = 1.5)")
    print("-" * 50)
    result_moderate_bubble = solve_poker_hand(
        hero_hand,
        num_opponents,
        board,
        simulation_mode="fast",
        stack_sizes=[5000, 4000, 3000, 2000],
        pot_size=300,
        tournament_context={'bubble_factor': 1.5}
    )
    print(f"Win probability: {result_moderate_bubble.win_probability:.4f}")
    print(f"ICM equity: {result_moderate_bubble.icm_equity:.4f}" if result_moderate_bubble.icm_equity else "ICM equity: Not calculated")
    print(f"Bubble factor: {result_moderate_bubble.bubble_factor}" if result_moderate_bubble.bubble_factor else "Bubble factor: Not set")
    
    # Test 3: High bubble pressure
    print("\nTest 3: High bubble pressure (bubble_factor = 2.0)")
    print("-" * 50)
    result_high_bubble = solve_poker_hand(
        hero_hand,
        num_opponents,
        board,
        simulation_mode="fast",
        stack_sizes=[5000, 4000, 3000, 2000],
        pot_size=300,
        tournament_context={'bubble_factor': 2.0}
    )
    print(f"Win probability: {result_high_bubble.win_probability:.4f}")
    print(f"ICM equity: {result_high_bubble.icm_equity:.4f}" if result_high_bubble.icm_equity else "ICM equity: Not calculated")
    print(f"Bubble factor: {result_high_bubble.bubble_factor}" if result_high_bubble.bubble_factor else "Bubble factor: Not set")
    
    # Test 4: Extreme bubble pressure
    print("\nTest 4: Extreme bubble pressure (bubble_factor = 3.0)")
    print("-" * 50)
    result_extreme_bubble = solve_poker_hand(
        hero_hand,
        num_opponents,
        board,
        simulation_mode="fast",
        stack_sizes=[5000, 4000, 3000, 2000],
        pot_size=300,
        tournament_context={'bubble_factor': 3.0}
    )
    print(f"Win probability: {result_extreme_bubble.win_probability:.4f}")
    print(f"ICM equity: {result_extreme_bubble.icm_equity:.4f}" if result_extreme_bubble.icm_equity else "ICM equity: Not calculated")
    print(f"Bubble factor: {result_extreme_bubble.bubble_factor}" if result_extreme_bubble.bubble_factor else "Bubble factor: Not set")
    
    # Test 5: Different stack distributions with same bubble factor
    print("\n\nTest 5: Same bubble_factor (2.0) with different stack distributions")
    print("=" * 70)
    
    # Scenario A: Hero is chip leader
    print("\nScenario A: Hero is chip leader")
    print("-" * 50)
    result_leader = solve_poker_hand(
        hero_hand,
        num_opponents,
        board,
        simulation_mode="fast",
        stack_sizes=[10000, 3000, 2000, 1000],  # Hero has most chips
        pot_size=300,
        tournament_context={'bubble_factor': 2.0}
    )
    print(f"Stack sizes: [10000, 3000, 2000, 1000] (Hero first)")
    print(f"Win probability: {result_leader.win_probability:.4f}")
    print(f"ICM equity: {result_leader.icm_equity:.4f}" if result_leader.icm_equity else "ICM equity: Not calculated")
    print(f"Tournament pressure: {result_leader.tournament_pressure}" if result_leader.tournament_pressure else "Tournament pressure: Not calculated")
    
    # Scenario B: Hero is short stack
    print("\nScenario B: Hero is short stack")
    print("-" * 50)
    result_short = solve_poker_hand(
        hero_hand,
        num_opponents,
        board,
        simulation_mode="fast",
        stack_sizes=[1000, 10000, 3000, 2000],  # Hero has least chips
        pot_size=300,
        tournament_context={'bubble_factor': 2.0}
    )
    print(f"Stack sizes: [1000, 10000, 3000, 2000] (Hero first)")
    print(f"Win probability: {result_short.win_probability:.4f}")
    print(f"ICM equity: {result_short.icm_equity:.4f}" if result_short.icm_equity else "ICM equity: Not calculated")
    print(f"Tournament pressure: {result_short.tournament_pressure}" if result_short.tournament_pressure else "Tournament pressure: Not calculated")
    
    # Scenario C: Hero is medium stack
    print("\nScenario C: Hero is medium stack")
    print("-" * 50)
    result_medium = solve_poker_hand(
        hero_hand,
        num_opponents,
        board,
        simulation_mode="fast",
        stack_sizes=[3000, 10000, 2000, 1000],  # Hero has medium chips
        pot_size=300,
        tournament_context={'bubble_factor': 2.0}
    )
    print(f"Stack sizes: [3000, 10000, 2000, 1000] (Hero first)")
    print(f"Win probability: {result_medium.win_probability:.4f}")
    print(f"ICM equity: {result_medium.icm_equity:.4f}" if result_medium.icm_equity else "ICM equity: Not calculated")
    print(f"Tournament pressure: {result_medium.tournament_pressure}" if result_medium.tournament_pressure else "Tournament pressure: Not calculated")
    
    # Summary
    print("\n\nSUMMARY")
    print("=" * 70)
    print(f"\nBubble factor impact on AA vs 3 opponents:")
    print(f"No bubble (1.0):       Win = {result_no_bubble.win_probability:.4f}, ICM = {result_no_bubble.icm_equity:.4f if result_no_bubble.icm_equity else 'N/A'}")
    print(f"Moderate bubble (1.5): Win = {result_moderate_bubble.win_probability:.4f}, ICM = {result_moderate_bubble.icm_equity:.4f if result_moderate_bubble.icm_equity else 'N/A'}")
    print(f"High bubble (2.0):     Win = {result_high_bubble.win_probability:.4f}, ICM = {result_high_bubble.icm_equity:.4f if result_high_bubble.icm_equity else 'N/A'}")
    print(f"Extreme bubble (3.0):  Win = {result_extreme_bubble.win_probability:.4f}, ICM = {result_extreme_bubble.icm_equity:.4f if result_extreme_bubble.icm_equity else 'N/A'}")
    
    print(f"\nStack position impact (bubble_factor = 2.0):")
    print(f"Chip leader:  Win = {result_leader.win_probability:.4f}, ICM = {result_leader.icm_equity:.4f if result_leader.icm_equity else 'N/A'}")
    print(f"Short stack:  Win = {result_short.win_probability:.4f}, ICM = {result_short.icm_equity:.4f if result_short.icm_equity else 'N/A'}")
    print(f"Medium stack: Win = {result_medium.win_probability:.4f}, ICM = {result_medium.icm_equity:.4f if result_medium.icm_equity else 'N/A'}")
    
    # Verify that bubble_factor actually changes results
    print("\n\nVERIFICATION")
    print("=" * 70)
    
    # Check if we have ICM equity values to compare
    if all(r.icm_equity is not None for r in [result_no_bubble, result_moderate_bubble, result_high_bubble, result_extreme_bubble]):
        if (result_no_bubble.icm_equity != result_moderate_bubble.icm_equity or
            result_moderate_bubble.icm_equity != result_high_bubble.icm_equity):
            print("✓ CONFIRMED: bubble_factor DOES affect ICM equity calculations")
            print("  Different bubble_factor values produce different ICM equity values")
        else:
            print("✗ WARNING: bubble_factor does NOT seem to affect ICM calculations")
            print("  All bubble_factor values produced the same ICM equity")
    else:
        print("! NOTE: ICM equity not calculated for all scenarios")
        print("  Ensure pot_size is provided for ICM calculations")
    
    if all(r.icm_equity is not None for r in [result_leader, result_short, result_medium]):
        if (result_leader.icm_equity != result_short.icm_equity or
            result_short.icm_equity != result_medium.icm_equity):
            print("✓ CONFIRMED: Stack position DOES affect ICM equity with bubble_factor")
            print("  Different stack positions produce different ICM equity values")
        else:
            print("✗ WARNING: Stack position does NOT affect ICM equity calculations")
            print("  All stack positions produced the same ICM equity")
    else:
        print("! NOTE: ICM equity not calculated for all stack position scenarios")
    
    # Additional check: print bubble_factor values to ensure they're being set
    print("\n\nDEBUG INFO")
    print("=" * 70)
    print("Bubble factor values from results:")
    print(f"No bubble result: bubble_factor = {result_no_bubble.bubble_factor}")
    print(f"Moderate bubble result: bubble_factor = {result_moderate_bubble.bubble_factor}")
    print(f"High bubble result: bubble_factor = {result_high_bubble.bubble_factor}")
    print(f"Extreme bubble result: bubble_factor = {result_extreme_bubble.bubble_factor}")

if __name__ == "__main__":
    test_bubble_factor_impact()