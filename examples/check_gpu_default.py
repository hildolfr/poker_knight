#!/usr/bin/env python3
"""Check if GPU is used by default."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

from poker_knight import solve_poker_hand

print("=== GPU Default Usage Check ===\n")

# Test different simulation modes
modes = ['fast', 'default', 'precision']

for mode in modes:
    result = solve_poker_hand(['A♠', 'K♠'], 2, simulation_mode=mode)
    print(f"Mode: {mode}")
    print(f"  GPU used: {result.gpu_used}")
    print(f"  Backend: {result.backend}")
    print(f"  Simulations: {result.simulations_run:,}")
    print()