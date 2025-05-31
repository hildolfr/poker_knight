#!/usr/bin/env python3
"""
Test parallel processing performance
"""

import sys
import os
# Add parent directory to path to allow importing poker_knight
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poker_knight import MonteCarloSolver
import time

def test_parallel_performance():
    print("Testing parallel processing performance...")
    
    # Test with parallel processing enabled
    solver_parallel = MonteCarloSolver()
    solver_parallel.config["simulation_settings"]["parallel_processing"] = True
    
    # Test with parallel processing disabled
    solver_sequential = MonteCarloSolver()
    solver_sequential.config["simulation_settings"]["parallel_processing"] = False
    
    hand = ['A♠️', 'K♥️']
    opponents = 2
    
    # Test default mode (100K simulations)
    print("\nTesting default mode (100,000 simulations):")
    
    # Sequential
    start_time = time.time()
    result_seq = solver_sequential.analyze_hand(hand, opponents, simulation_mode='default')
    seq_time = time.time() - start_time
    
    # Parallel
    start_time = time.time()
    result_par = solver_parallel.analyze_hand(hand, opponents, simulation_mode='default')
    par_time = time.time() - start_time
    
    print(f"Sequential: {result_seq.simulations_run:,} sims in {seq_time:.2f}s ({result_seq.win_probability:.1%})")
    print(f"Parallel:   {result_par.simulations_run:,} sims in {par_time:.2f}s ({result_par.win_probability:.1%})")
    print(f"Speedup:    {seq_time/par_time:.2f}x")
    
    # Test precision mode (500K simulations) - only if we have time
    print("\nTesting precision mode (500,000 simulations):")
    
    # Sequential
    start_time = time.time()
    result_seq = solver_sequential.analyze_hand(hand, opponents, simulation_mode='precision')
    seq_time = time.time() - start_time
    
    # Parallel
    start_time = time.time()
    result_par = solver_parallel.analyze_hand(hand, opponents, simulation_mode='precision')
    par_time = time.time() - start_time
    
    print(f"Sequential: {result_seq.simulations_run:,} sims in {seq_time:.2f}s ({result_seq.win_probability:.1%})")
    print(f"Parallel:   {result_par.simulations_run:,} sims in {par_time:.2f}s ({result_par.win_probability:.1%})")
    print(f"Speedup:    {seq_time/par_time:.2f}x")

if __name__ == "__main__":
    test_parallel_performance() 