#!/usr/bin/env python3
"""Quick GPU validation test."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

from poker_knight import solve_poker_hand

print("=== Quick GPU Validation ===\n")

# Test cases with expected ranges
tests = [
    (['A♠', 'A♥'], 1, (0.80, 0.90), "AA vs 1"),
    (['A♠', 'A♥'], 4, (0.45, 0.65), "AA vs 4"),
    (['K♠', 'K♥'], 2, (0.60, 0.75), "KK vs 2"),
    (['A♠', 'K♠'], 2, (0.45, 0.55), "AKs vs 2"),
    (['7♠', '2♥'], 1, (0.25, 0.40), "72o vs 1"),
]

all_pass = True

for hand, opps, (min_win, max_win), desc in tests:
    result = solve_poker_hand(hand, opps, simulation_mode='fast')
    in_range = min_win <= result.win_probability <= max_win
    status = "✅" if in_range else "❌"
    
    print(f"{desc}: {result.win_probability:.1%} (expected {min_win:.0%}-{max_win:.0%}) {status}")
    print(f"  GPU: {result.gpu_used}, Backend: {result.backend}")
    
    if not in_range:
        all_pass = False

print(f"\nOverall: {'✅ PASS' if all_pass else '❌ FAIL'}")