#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to force Advanced Parallel Processing to trigger
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poker_knight'))

from poker_knight.solver import MonteCarloSolver


def test_force_advanced_parallel():
    """Force advanced parallel processing by using high simulation counts."""
    print("[ROCKET] Testing Forced Advanced Parallel Processing")
    print("=" * 60)
    
    # Create solver to avoid cache hits
    solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
    
    # Test scenario: 72o vs 5 opponents (worst hand, many opponents)
    # This should have high complexity and avoid cache hits
    hero_hand = ["7S", "2C"]  # Worst possible hand
    board = ["AD", "KD", "QD", "JD"]  # Very coordinated board (turn)
    num_opponents = 5  # Maximum opponents for high complexity
    
    print(f"Scenario: {hero_hand} vs {num_opponents} opponents")
    print(f"Board: {board}")
    print()
    
    # Manually override config to force high simulation counts
    solver.config["simulation_settings"]["default_simulations"] = 50000
    solver.config["simulation_settings"]["precision_mode_simulations"] = 100000
    
    # Test with precision mode and no intelligent optimization
    print("Testing precision mode with 100,000 simulations (no intelligent optimization, no cache)...")
    
    start_time = time.time()
    result = solver.analyze_hand(
        hero_hand=hero_hand,
        num_opponents=num_opponents,
        board_cards=board,
        simulation_mode="precision",
        intelligent_optimization=False  # Disable to force high sim count
    )
    execution_time = time.time() - start_time
    
    print(f"  [PASS] Win Rate: {result.win_probability:.1%}")
    print(f"  ⏱️  Execution Time: {execution_time:.2f}s")
    print(f"  🔢 Simulations: {result.simulations_run:,}")
    
    # Check if optimization data includes parallel execution info
    if result.optimization_data and 'parallel_execution' in result.optimization_data:
        parallel_info = result.optimization_data['parallel_execution']
        print(f"  [FIX] Engine: {parallel_info['engine_type']}")
        print(f"  👥 Workers: {parallel_info['total_workers']} "
              f"(MP: {parallel_info['multiprocessing_workers']}, "
              f"T: {parallel_info['threading_workers']})")
        print(f"  📈 Speedup: {parallel_info['speedup_factor']:.1f}x")
        print(f"  ⚡ Efficiency: {parallel_info['efficiency_percentage']:.1f}%")
        print(f"  💻 CPU Util: {parallel_info['cpu_utilization']:.1f}%")
        print("  [PASS] ADVANCED PARALLEL PROCESSING ACTIVATED!")
    else:
        print("  [WARN]  Standard processing used (not advanced parallel)")
        
        # Debug why advanced parallel wasn't used
        print("\n  [SEARCH] Debug Info:")
        print(f"     - Parallel engine available: {solver._parallel_engine is not None}")
        print(f"     - Simulations: {result.simulations_run:,}")
        print(f"     - Parallel processing enabled: {solver.config['simulation_settings'].get('parallel_processing', False)}")
        print(f"     - Parallel threshold: {solver.config['performance_settings'].get('parallel_processing_threshold', 1000)}")
        
        # Calculate what the complexity score would be
        base_complexity = 8.0  # precision mode
        opponent_factor = min(2.0, num_opponents * 0.5)  # 5 * 0.5 = 2.5, min(2.0, 2.5) = 2.0
        board_factor = len(board) * 0.3  # 4 * 0.3 = 1.2
        calculated_complexity = base_complexity + opponent_factor + board_factor
        print(f"     - Calculated complexity score: {calculated_complexity:.1f}")
        print(f"     - Complexity threshold for advanced parallel: 3.0")
        print(f"     - Minimum simulations for advanced parallel: 5,000")
        
        # Check if convergence analysis is interfering
        print(f"     - Convergence analysis available: {hasattr(solver, 'CONVERGENCE_ANALYSIS_AVAILABLE')}")
    
    print()
    
    # Clean up
    solver.close()
    print("[PASS] Test completed!")


if __name__ == "__main__":
    try:
        test_force_advanced_parallel()
        
    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 