#!/usr/bin/env python3
"""Test with debug logging enabled."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from poker_knight import solve_poker_hand

print("=== Test with Debug Logging ===\n")

result = solve_poker_hand(['A♠', 'A♥'], 1, simulation_mode='fast')
print(f"\nResult: {result.win_probability:.1%} win probability")
print(f"Simulations: {result.simulations_run}")
print(f"GPU used: {result.gpu_used}")