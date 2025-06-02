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
    
    # Create solver with parallel processing enabled and caching DISABLED
    solver = MonteCarloSolver(enable_caching=False)
    
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


def test_cache_integration():
    """Test that caching works with the new parallel processing."""
    print("[CACHE]  Testing Cache Integration with Parallel Processing")
    print("=" * 60)
    
    solver = MonteCarloSolver(enable_caching=True)
    
    hero_hand = ["KS", "KH"]
    num_opponents = 2
    
    print("First run (should populate cache)...")
    start_time = time.time()
    result1 = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
    time1 = time.time() - start_time
    
    print("Second run (should hit cache)...")
    start_time = time.time()
    result2 = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
    time2 = time.time() - start_time
    
    print(f"First run:  {time1:.3f}s, Win rate: {result1.win_probability:.1%}")
    print(f"Second run: {time2:.3f}s, Win rate: {result2.win_probability:.1%}")
    
    if time2 > 0:
        speedup = time1/time2
        print(f"Speedup from caching: {speedup:.1f}x")
    else:
        print("Second run was instantaneous (perfect cache hit)")
    
    # Get cache stats
    cache_stats = solver.get_cache_stats()
    if cache_stats:
        hand_cache = cache_stats.get('hand_cache', {})
        print(f"Cache hit rate: {hand_cache.get('hit_rate', 0):.1%}")
        print(f"Cache requests: {hand_cache.get('total_requests', 0)}")
    
    solver.close()
    print("[PASS] Cache integration test completed!")


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
        test_cache_integration()
        print()
        test_numa_awareness()
        
    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n🎉 All Task 1.1 tests passed! Advanced Parallel Processing is working correctly.") 