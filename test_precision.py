#!/usr/bin/env python3
"""
Test precision mode specifically
"""

from poker_solver import solve_poker_hand
import time

def test_precision_mode():
    print("Testing precision mode...")
    
    start_time = time.time()
    result = solve_poker_hand(['A♠️', 'A♥️'], 1, simulation_mode='precision')
    total_time = time.time() - start_time
    
    print(f"Precision mode results:")
    print(f"  Simulations run: {result.simulations_run:,}")
    print(f"  Target: 500,000")
    print(f"  Efficiency: {result.simulations_run/500000*100:.1f}% of target")
    print(f"  Execution time: {result.execution_time_ms:.1f}ms")
    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Win probability: {result.win_probability:.1%}")

if __name__ == "__main__":
    test_precision_mode() 