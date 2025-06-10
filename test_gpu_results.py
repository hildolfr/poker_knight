#!/usr/bin/env python3
"""Test GPU results for reasonableness."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

from poker_knight import solve_poker_hand

print("=== GPU Results Test ===\n")

# Test scenarios with expected approximate results
test_cases = [
    # (hero_hand, num_opponents, expected_win_range, description)
    (['A♠', 'A♥'], 1, (0.80, 0.90), "Pocket Aces vs 1 opponent"),
    (['A♠', 'A♥'], 4, (0.45, 0.65), "Pocket Aces vs 4 opponents"),
    (['A♠', 'K♠'], 2, (0.45, 0.55), "AK suited vs 2 opponents"),
    (['7♠', '2♥'], 1, (0.25, 0.40), "7-2 offsuit vs 1 opponent"),
    (['K♠', 'K♥'], 2, (0.60, 0.75), "Pocket Kings vs 2 opponents"),
]

for hero_hand, num_opponents, (min_win, max_win), description in test_cases:
    print(f"Test: {description}")
    print(f"Hand: {' '.join(hero_hand)}")
    
    result = solve_poker_hand(hero_hand, num_opponents, simulation_mode='default')
    
    in_range = min_win <= result.win_probability <= max_win
    status = "✅ PASS" if in_range else "❌ FAIL"
    
    print(f"Win probability: {result.win_probability:.1%}")
    print(f"Expected range: {min_win:.0%}-{max_win:.0%}")
    print(f"GPU used: {result.gpu_used}")
    print(f"Backend: {result.backend}")
    print(f"Status: {status}")
    print()

# Also test the demo scenario
print("\nDemo test: A♠ K♠ vs 2 opponents")
result = solve_poker_hand(['A♠', 'K♠'], 2, simulation_mode='default')
print(f"Win probability: {result.win_probability:.1%}")
print(f"Simulations: {result.simulations_run:,}")
print(f"GPU used: {result.gpu_used}")
print(f"Execution time: {result.execution_time_ms:.1f}ms")