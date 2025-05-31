#!/usr/bin/env python3
"""
Debug test to understand precision mode timeout issue
"""

from poker_solver import MonteCarloSolver
import time

def debug_precision_mode():
    print("Debugging precision mode...")
    
    solver = MonteCarloSolver()
    
    # Check the config values
    print(f"Config loaded:")
    print(f"  fast_mode_simulations: {solver.config['simulation_settings']['fast_mode_simulations']}")
    print(f"  default_simulations: {solver.config['simulation_settings']['default_simulations']}")
    print(f"  precision_mode_simulations: {solver.config['simulation_settings']['precision_mode_simulations']}")
    print(f"  max_simulation_time_ms: {solver.config['performance_settings']['max_simulation_time_ms']}")
    
    # Test what simulation count is determined for precision mode
    simulation_mode = "precision"
    if simulation_mode == "fast":
        sim_key = "fast_mode_simulations"
    elif simulation_mode == "precision":
        sim_key = "precision_mode_simulations"
    else:
        sim_key = "default_simulations"
    
    if sim_key in solver.config["simulation_settings"]:
        num_simulations = solver.config["simulation_settings"][sim_key]
    else:
        num_simulations = solver.config["simulation_settings"]["default_simulations"]
    
    print(f"\nSimulation count lookup:")
    print(f"  sim_key: {sim_key}")
    print(f"  num_simulations determined: {num_simulations}")
    
    # Test what timeout is calculated for precision mode
    base_timeout = solver.config["performance_settings"]["max_simulation_time_ms"]
    if simulation_mode == "fast":
        max_time_ms = 3000
    elif simulation_mode == "precision":
        max_time_ms = 120000  # 120 seconds
    else:
        max_time_ms = 20000
    
    print(f"  Calculated timeout: {max_time_ms}ms ({max_time_ms/1000}s)")
    
    # Test the simulation
    start_time = time.time()
    result = solver.analyze_hand(['A♠️', 'A♥️'], 1, simulation_mode='precision')
    total_time = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Simulations run: {result.simulations_run:,}")
    print(f"  Target: {num_simulations:,}")
    print(f"  Execution time: {result.execution_time_ms:.1f}ms")
    print(f"  Total time: {total_time*1000:.1f}ms")
    print(f"  Timeout was: {max_time_ms}ms")
    print(f"  Did it timeout? {result.execution_time_ms >= max_time_ms}")

if __name__ == "__main__":
    debug_precision_mode() 