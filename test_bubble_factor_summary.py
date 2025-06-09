#!/usr/bin/env python3
"""
Clean summary of bubble_factor behavior testing results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from poker_knight import solve_poker_hand

def test_bubble_factor_summary():
    """Generate a clean summary of bubble_factor behavior."""
    
    print("BUBBLE FACTOR BEHAVIOR ANALYSIS")
    print("=" * 70)
    print("\nTest Setup: AA vs 3 opponents, various tournament conditions")
    print("Stack sizes: [5000, 4000, 3000, 2000], Pot size: 300")
    
    # Run tests with different bubble factors
    hero_hand = ['A♠', 'A♥']
    num_opponents = 3
    board = []
    
    bubble_factors = [1.0, 1.5, 2.0, 3.0]
    results = []
    
    print("\n1. BUBBLE FACTOR IMPACT ON ICM EQUITY")
    print("-" * 50)
    print(f"{'Bubble Factor':<15} {'Win Prob':<10} {'ICM Equity':<12} {'ICM/Win Ratio':<15}")
    print("-" * 50)
    
    for bf in bubble_factors:
        result = solve_poker_hand(
            hero_hand,
            num_opponents,
            board,
            simulation_mode="fast",
            stack_sizes=[5000, 4000, 3000, 2000],
            pot_size=300,
            tournament_context={'bubble_factor': bf}
        )
        results.append((bf, result))
        
        icm_win_ratio = result.icm_equity / result.win_probability if result.icm_equity else 0
        print(f"{bf:<15.1f} {result.win_probability:<10.4f} {result.icm_equity:<12.4f} {icm_win_ratio:<15.4f}")
    
    # Test stack position impact
    print("\n\n2. STACK POSITION IMPACT (bubble_factor = 2.0)")
    print("-" * 50)
    print(f"{'Stack Position':<20} {'Hero Stack':<12} {'Win Prob':<10} {'ICM Equity':<12} {'Stack Pressure':<15}")
    print("-" * 50)
    
    stack_configs = [
        ("Chip Leader", [10000, 3000, 2000, 1000]),
        ("Medium Stack", [3000, 10000, 2000, 1000]),
        ("Short Stack", [1000, 10000, 3000, 2000])
    ]
    
    for position, stacks in stack_configs:
        result = solve_poker_hand(
            hero_hand,
            num_opponents,
            board,
            simulation_mode="fast",
            stack_sizes=stacks,
            pot_size=300,
            tournament_context={'bubble_factor': 2.0}
        )
        
        stack_pressure = result.tournament_pressure['stack_pressure'] if result.tournament_pressure else 0
        print(f"{position:<20} {stacks[0]:<12} {result.win_probability:<10.4f} {result.icm_equity:<12.4f} {stack_pressure:<15.4f}")
    
    # Analysis
    print("\n\n3. KEY FINDINGS")
    print("-" * 50)
    
    # Finding 1: Bubble factor effect
    base_icm = results[0][1].icm_equity
    high_bubble_icm = results[2][1].icm_equity
    extreme_bubble_icm = results[3][1].icm_equity
    
    print(f"• Bubble factor DOES affect ICM equity calculations:")
    print(f"  - No bubble (1.0): ICM = {base_icm:.4f}")
    print(f"  - High bubble (2.0): ICM = {high_bubble_icm:.4f} ({(high_bubble_icm/base_icm - 1)*100:.1f}% change)")
    print(f"  - Extreme bubble (3.0): ICM = {extreme_bubble_icm:.4f} ({(extreme_bubble_icm/base_icm - 1)*100:.1f}% change)")
    
    # Finding 2: Diminishing returns
    if high_bubble_icm == extreme_bubble_icm:
        print(f"\n• WARNING: Bubble factor appears to have diminishing returns!")
        print(f"  - bubble_factor 2.0 and 3.0 produce the same ICM equity ({high_bubble_icm:.4f})")
        print(f"  - This suggests a cap or floor in the bubble adjustment calculation")
    
    # Finding 3: Win probability unchanged
    print(f"\n• Win probability remains constant across all bubble factors:")
    print(f"  - This is correct: bubble factor affects ICM equity, not raw win probability")
    
    # Finding 4: Current implementation
    print(f"\n• Current bubble_factor implementation (from multiway.py):")
    print(f"  - Only uses bubble_factor from tournament_context")
    print(f"  - Does NOT automatically calculate based on stack distributions")
    print(f"  - Adjustment formula: max(0.7, 1.0 - (bubble_factor - 1.0) * 0.3)")
    print(f"  - This creates a floor at 0.7x the base equity")
    
    print("\n\n4. RECOMMENDATIONS")
    print("-" * 50)
    print("• The proposed automatic bubble_factor calculation would:")
    print("  - Calculate bubble_factor automatically when not explicitly provided")
    print("  - Use stack distributions, BB counts, and player counts")
    print("  - Provide more nuanced tournament pressure modeling")
    print("  - Remove the current hard floor limitation")

if __name__ == "__main__":
    test_bubble_factor_summary()