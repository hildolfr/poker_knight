#!/usr/bin/env python3
"""
Quick test script to verify the simulation timeout fix
"""

from poker_solver import solve_poker_hand

def test_simulation_modes():
    print("Testing simulation timeout fix...")
    
    # Test fast mode
    result_fast = solve_poker_hand(['A♠️', 'A♥️'], 1, simulation_mode='fast')
    print(f"Fast mode: {result_fast.simulations_run} sims in {result_fast.execution_time_ms:.1f}ms (target: 10,000)")
    
    # Test default mode  
    result_default = solve_poker_hand(['A♠️', 'A♥️'], 1, simulation_mode='default')
    print(f"Default mode: {result_default.simulations_run} sims in {result_default.execution_time_ms:.1f}ms (target: 100,000)")
    
    # Test precision mode
    result_precision = solve_poker_hand(['A♠️', 'A♥️'], 1, simulation_mode='precision')
    print(f"Precision mode: {result_precision.simulations_run} sims in {result_precision.execution_time_ms:.1f}ms (target: 500,000)")
    
    print("\nAnalysis:")
    print(f"Fast mode efficiency: {result_fast.simulations_run/10000*100:.1f}% of target")
    print(f"Default mode efficiency: {result_default.simulations_run/100000*100:.1f}% of target")
    print(f"Precision mode efficiency: {result_precision.simulations_run/500000*100:.1f}% of target")

if __name__ == "__main__":
    test_simulation_modes() 