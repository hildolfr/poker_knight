#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Debug script for advanced parallel processing"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poker_knight'))

from poker_knight.solver import MonteCarloSolver

def main():
    print("[SEARCH] Debugging Advanced Parallel Processing")
    print("=" * 50)
    
    # Test 1: No cache, no optimization, precision mode
    print("Test 1: Precision mode, no cache, no optimization")
    solver1 = MonteCarloSolver(enable_caching=False)
    result1 = solver1.analyze_hand(
        hero_hand=["AS", "AH"], 
        num_opponents=3, 
        board_cards=["KD", "QD", "JD"], 
        simulation_mode="precision",
        intelligent_optimization=False
    )
    print(f"  Simulations: {result1.simulations_run:,}")
    print(f"  Win rate: {result1.win_probability:.1%}")
    parallel_used = result1.optimization_data and 'parallel_execution' in result1.optimization_data
    print(f"  Advanced parallel: {parallel_used}")
    solver1.close()
    
    # Test 2: Default mode with higher simulation count
    print("\nTest 2: Default mode, high sim count")
    solver2 = MonteCarloSolver(enable_caching=False)
    # Override config to force higher simulations
    solver2.config["simulation_settings"]["default_simulations"] = 50000
    result2 = solver2.analyze_hand(
        hero_hand=["AS", "AH"], 
        num_opponents=4, 
        board_cards=["KD", "QD", "JD", "10S"], 
        simulation_mode="default",
        intelligent_optimization=False
    )
    print(f"  Simulations: {result2.simulations_run:,}")
    parallel_used2 = result2.optimization_data and 'parallel_execution' in result2.optimization_data
    print(f"  Advanced parallel: {parallel_used2}")
    solver2.close()
    
    print("\n[PASS] Debug completed!")

if __name__ == "__main__":
    main() 