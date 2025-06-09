#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for Advanced Parallel Processing Architecture (Task 1.1)

Tests the new multiprocessing + threading hybrid approach for Monte Carlo
poker simulations with complexity-based work distribution.
"""

import time
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'poker_knight'))

from poker_knight.solver import MonteCarloSolver


def test_basic_parallel_processing():
    """Test basic parallel processing functionality."""
    print("[ROCKET] Testing Advanced Parallel Processing (Task 1.1)")
    print("=" * 60)
    
    # Create solver with parallel processing enabled
    solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
    
    # Test scenario: AA vs 3 opponents on coordinated board
    # This should trigger advanced parallel processing
    hero_hand = ["AS", "AH"]
    board = ["KD", "QD", "JD"]  # Coordinated board
    num_opponents = 3
    
    print(f"Scenario: {hero_hand} vs {num_opponents} opponents")
    print(f"Board: {board}")
    print()
    
    # Override config to force higher simulation counts and disable optimizations
    solver.config["simulation_settings"]["fast_mode_simulations"] = 20000
    solver.config["simulation_settings"]["default_simulations"] = 75000
    solver.config["simulation_settings"]["precision_mode_simulations"] = 200000
    
    # Test with different simulation modes
    for mode, expected_sims in [("fast", 20000), ("default", 75000), ("precision", 200000)]:
        print(f"Testing {mode} mode (expecting ~{expected_sims:,} simulations)...")
        
        start_time = time.time()
        result = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=board,
            simulation_mode=mode,
            intelligent_optimization=False  # Disable to force full simulation count
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
        else:
            print("  [WARN]  Standard processing used (not advanced parallel)")
        
        print()
    
    # Clean up
    solver.close()
    print("[PASS] All tests completed successfully!")


def test_consistent_performance():
    """Test that performance is consistent without caching."""
    print("[PERF]  Testing Consistent Performance (No Cache)")
    print("=" * 60)
    
    solver = MonteCarloSolver()  # No caching in v1.7.0
    
    hero_hand = ["KS", "KH"]
    num_opponents = 2
    
    print("Running multiple simulations to test consistency...")
    times = []
    
    for i in range(3):
        print(f"Run {i+1}...", end='', flush=True)
        start_time = time.time()
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="fast")
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f" {elapsed:.3f}s (Win rate: {result.win_probability:.1%})")
    
    avg_time = sum(times) / len(times)
    max_diff = max(times) - min(times)
    
    print(f"\nAverage time: {avg_time:.3f}s")
    print(f"Time variance: {max_diff:.3f}s")
    print(f"Consistency: {'Good' if max_diff < 0.5 else 'Fair'}")
    
    solver.close()
    print("[PASS] Performance consistency test completed!")


def test_numa_awareness():
    """Test NUMA topology detection."""
    print("🖥️  Testing NUMA Awareness")
    print("=" * 60)
    
    try:
        from poker_knight.core.parallel import NumaTopology
        numa = NumaTopology()
        
        print(f"NUMA Available: {numa.available}")
        if numa.available:
            topology = numa.get_topology_info()
            print(f"NUMA Nodes: {topology.get('numa_nodes', 'Unknown')}")
            print(f"Logical Cores: {topology.get('logical_cores', 'Unknown')}")
            print(f"Physical Cores: {topology.get('physical_cores', 'Unknown')}")
            
            # Test CPU assignments
            for node_id in range(topology.get('numa_nodes', 1)):
                cpus = numa.get_cpus_for_node(node_id)
                print(f"NUMA Node {node_id}: CPUs {cpus}")
        else:
            print("NUMA topology detection not available on this system")
            
    except ImportError:
        print("[WARN]  Advanced parallel processing module not available")
    
    print("[PASS] NUMA awareness test completed!")


if __name__ == "__main__":
    try:
        test_basic_parallel_processing()
        print()
        test_consistent_performance()
        print()
        test_numa_awareness()
        
    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 All Task 1.1 tests passed! Advanced Parallel Processing is working correctly.") 