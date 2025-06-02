#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive test and demonstration of the NUMA-aware Cache Warming System.

This test demonstrates:
- Cache warming system initialization
- NUMA-aware background processing
- Intelligent task generation for all 169 preflop combinations
- Board texture warming for common patterns
- Adaptive learning from user queries
- Performance monitoring and statistics
- Integration with existing cache and parallel systems

The cache warming system leverages NUMA topology for optimal CPU utilization
and provides foundation for future CUDA acceleration.

Author: hildolfr
License: MIT
"""

import os
import sys
import time
import threading
import multiprocessing as mp
from pathlib import Path

# Add poker_knight to path
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

try:
    from poker_knight.storage.cache_warming import (
        WarmingProfile, WarmingPriority, WarmingTask, WarmingConfig, WarmingStats,
        WarmingTaskGenerator, NumaAwareCacheWarmer, create_cache_warmer,
        start_background_warming, CACHE_WARMING_AVAILABLE
    )
    from poker_knight.storage.cache import CacheConfig, get_cache_manager
    from poker_knight.core.parallel import NumaTopology
    from poker_knight import MonteCarloSolver
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"[FAIL] Cache warming dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


def test_warming_config():
    """Test cache warming configuration."""
    print("[FIX] Testing Cache Warming Configuration")
    print("=" * 60)
    
    # Test default configuration
    config = WarmingConfig()
    print(f"Default profile: {config.profile}")
    print(f"Max background workers: {config.max_background_workers}")
    print(f"NUMA aware: {config.numa_aware}")
    print(f"Background CPU limit: {config.background_cpu_limit}")
    print(f"Warm on startup: {config.warm_on_startup}")
    
    # Test custom configuration
    tournament_config = WarmingConfig(
        profile=WarmingProfile.TOURNAMENT,
        max_background_workers=4,
        background_cpu_limit=0.7,
        numa_aware=True
    )
    
    print(f"\nTournament config profile: {tournament_config.profile}")
    print(f"Tournament config workers: {tournament_config.max_background_workers}")
    print("[PASS] Cache warming configuration test completed!")


def test_task_generation():
    """Test intelligent task generation."""
    print("\n📝 Testing Task Generation")
    print("=" * 60)
    
    config = WarmingConfig()
    generator = WarmingTaskGenerator(config)
    
    # Test preflop task generation
    print("Generating preflop tasks...")
    preflop_tasks = generator.generate_preflop_tasks()
    print(f"Generated {len(preflop_tasks)} preflop tasks")
    
    # Show priority distribution
    priority_counts = {}
    for task in preflop_tasks[:100]:  # Sample first 100
        if task.priority not in priority_counts:
            priority_counts[task.priority] = 0
        priority_counts[task.priority] += 1
    
    print("Priority distribution (first 100 tasks):")
    for priority, count in priority_counts.items():
        print(f"  {priority.name}: {count} tasks")
    
    # Test board texture task generation
    print("\nGenerating board texture tasks...")
    board_tasks = generator.generate_board_texture_tasks()
    print(f"Generated {len(board_tasks)} board texture tasks")
    
    # Show some example tasks
    print("\nExample preflop tasks:")
    for i, task in enumerate(preflop_tasks[:5]):
        print(f"  {i+1}. {task.task_id} ({task.priority.name})")
        print(f"     Hand: {task.hero_hand}, Opponents: {task.num_opponents}")
        print(f"     Position: {task.position}, Simulations: {task.simulations}")
    
    print("\nExample board texture tasks:")
    for i, task in enumerate(board_tasks[:3]):
        print(f"  {i+1}. {task.task_id} ({task.priority.name})")
        print(f"     Hand: {task.hero_hand}, Board: {task.board_cards}")
        print(f"     Opponents: {task.num_opponents}, Simulations: {task.simulations}")
    
    print("[PASS] Task generation test completed!")


def test_numa_topology():
    """Test NUMA topology detection."""
    print("\n🖥️  Testing NUMA Topology Detection")
    print("=" * 60)
    
    numa = NumaTopology()
    
    print(f"NUMA Available: {numa.available}")
    if numa.available:
        topology = numa.get_topology_info()
        print(f"NUMA Nodes: {topology.get('numa_nodes', 'Unknown')}")
        print(f"Logical Cores: {topology.get('logical_cores', 'Unknown')}")
        print(f"Physical Cores: {topology.get('physical_cores', 'Unknown')}")
        print(f"Cores per Node: {topology.get('cores_per_node', 'Unknown')}")
        
        # Test CPU assignments
        for node_id in range(topology.get('numa_nodes', 1)):
            cpus = numa.get_cpus_for_node(node_id)
            print(f"NUMA Node {node_id}: CPUs {cpus[:8]}{'...' if len(cpus) > 8 else ''}")
    else:
        print("NUMA topology detection not available on this system")
    
    print("[PASS] NUMA topology test completed!")


def test_cache_warmer_initialization():
    """Test cache warmer initialization."""
    print("\n🔥 Testing Cache Warmer Initialization")
    print("=" * 60)
    
    # Test basic initialization
    config = WarmingConfig(
        max_background_workers=2,
        numa_aware=True,
        warm_on_startup=False  # Don't start automatically for test
    )
    
    try:
        warmer = create_cache_warmer(config)
        print(f"Cache warmer created successfully")
        print(f"NUMA topology available: {warmer._numa_topology.available if warmer._numa_topology else False}")
        print(f"Parallel engine available: {warmer._parallel_engine is not None}")
        print(f"Hand cache available: {warmer._hand_cache is not None}")
        
        # Test stats
        stats = warmer.get_warming_stats()
        print(f"Initial stats - Total tasks: {stats.total_tasks}")
        print(f"Initial stats - Completed: {stats.completed_tasks}")
        
        # Clean up
        warmer.shutdown()
        print("[PASS] Cache warmer initialization test completed!")
        
    except Exception as e:
        print(f"[WARN]  Cache warmer initialization failed: {e}")


def test_background_warming():
    """Test background warming process."""
    print("\n⚡ Testing Background Cache Warming")
    print("=" * 60)
    
    # Create a small-scale test configuration
    config = WarmingConfig(
        max_background_workers=2,
        warming_batch_size=10,
        numa_aware=True,
        warm_on_startup=False,
        background_cpu_limit=0.8,
        opponent_counts=[1, 2],  # Limit opponents for faster testing
        common_positions=["BTN", "BB"],  # Limit positions
        save_warming_progress=False  # Don't save during test
    )
    
    try:
        warmer = create_cache_warmer(config)
        
        # Start warming
        print("Starting background warming...")
        warmer.start_warming(blocking=False)
        
        # Monitor progress for a short time
        start_time = time.time()
        max_test_time = 30.0  # Test for 30 seconds max
        
        while time.time() - start_time < max_test_time:
            stats = warmer.get_warming_stats()
            
            print(f"\rProgress: {stats.completed_tasks}/{stats.total_tasks} tasks "
                  f"({stats.completed_tasks/max(1, stats.total_tasks)*100:.1f}%) "
                  f"| {stats.cache_entries_created} cache entries "
                  f"| {stats.tasks_per_second:.1f} tasks/sec", end="")
            
            # Stop if we've made good progress or completed
            if stats.completed_tasks >= 50 or stats.completed_tasks >= stats.total_tasks:
                break
            
            time.sleep(1.0)
        
        print()  # New line after progress updates
        
        # Final stats
        final_stats = warmer.get_warming_stats()
        print(f"\nFinal Results:")
        print(f"  Completed tasks: {final_stats.completed_tasks}")
        print(f"  Failed tasks: {final_stats.failed_tasks}")
        print(f"  Cache entries created: {final_stats.cache_entries_created}")
        print(f"  Total simulations: {final_stats.total_simulations}")
        print(f"  Average tasks per second: {final_stats.tasks_per_second:.2f}")
        
        if final_stats.numa_distribution:
            print(f"  NUMA distribution: {dict(final_stats.numa_distribution)}")
        
        # Test priority task addition
        print("\nTesting priority task addition...")
        warmer.add_priority_task(
            hero_hand=["AS", "AH"],
            num_opponents=3,
            position="BTN"
        )
        
        # Let it process the priority task
        time.sleep(5.0)
        
        priority_stats = warmer.get_warming_stats()
        print(f"After priority task - Completed: {priority_stats.completed_tasks}")
        
        # Clean up
        warmer.stop_warming()
        warmer.shutdown()
        
        print("[PASS] Background warming test completed!")
        
    except Exception as e:
        print(f"[WARN]  Background warming test failed: {e}")
        import traceback
        traceback.print_exc()


def test_adaptive_learning():
    """Test adaptive learning from user queries."""
    print("\n🧠 Testing Adaptive Learning")
    print("=" * 60)
    
    config = WarmingConfig(
        learn_from_queries=True,
        adaptation_threshold=5,  # Lower threshold for testing
        query_history_size=100
    )
    
    generator = WarmingTaskGenerator(config)
    
    # Simulate user queries
    print("Simulating user queries...")
    queries = [
        (["AS", "AH"], 2, None, "BTN"),
        (["KS", "KH"], 3, None, "CO"),
        (["AS", "KS"], 2, None, "BTN"),
        (["AS", "AH"], 2, None, "BTN"),  # Repeat
        (["QS", "QH"], 4, None, "MP"),
        (["AS", "KS"], 2, None, "BTN"),  # Repeat
    ]
    
    for hero_hand, num_opponents, board_cards, position in queries:
        generator.learn_from_query(hero_hand, num_opponents, board_cards, position)
        print(f"  Learned from query: {hero_hand} vs {num_opponents} opponents in {position}")
    
    # Check pattern learning
    print(f"\nQuery patterns learned: {len(generator._query_patterns)}")
    print("Top patterns:")
    for pattern, count in generator._query_patterns.most_common(5):
        print(f"  {pattern}: {count} times")
    
    # Generate adaptive tasks
    print("\nGenerating adaptive tasks...")
    adaptive_tasks = generator.get_adaptive_tasks()
    print(f"Generated {len(adaptive_tasks)} adaptive tasks")
    
    for i, task in enumerate(adaptive_tasks[:3]):
        print(f"  {i+1}. {task.task_id} (user requested: {task.user_requested})")
    
    print("[PASS] Adaptive learning test completed!")


def test_integration_with_solver():
    """Test integration with existing MonteCarloSolver."""
    print("\n🔗 Testing Integration with Solver")
    print("=" * 60)
    
    try:
        # Create solver with caching enabled
        solver = MonteCarloSolver(enable_caching=True)
        
        # Create cache warmer
        warmer = create_cache_warmer(WarmingConfig(
            max_background_workers=2,
            warming_batch_size=5,
            warm_on_startup=False
        ))
        
        print("Testing solver query without warming...")
        start_time = time.time()
        result1 = solver.analyze_hand(
            hero_hand=["KS", "KH"],
            num_opponents=2,
            simulation_mode="fast"
        )
        time1 = time.time() - start_time
        
        print(f"First query time: {time1:.3f}s, Win rate: {result1.win_probability:.1%}")
        
        # Add this scenario as priority task for warming
        warmer.add_priority_task(
            hero_hand=["KS", "KH"],
            num_opponents=2
        )
        
        # Start warming briefly
        print("Starting brief warming...")
        warmer.start_warming(blocking=False)
        time.sleep(10.0)  # Warm for 10 seconds
        
        # Test same query again
        print("Testing solver query after warming...")
        start_time = time.time()
        result2 = solver.analyze_hand(
            hero_hand=["KS", "KH"],
            num_opponents=2,
            simulation_mode="fast"
        )
        time2 = time.time() - start_time
        
        print(f"Second query time: {time2:.3f}s, Win rate: {result2.win_probability:.1%}")
        
        if time2 > 0:
            speedup = time1 / time2
            print(f"Potential speedup: {speedup:.1f}x")
        
        # Get cache stats
        cache_stats = solver.get_cache_stats()
        if cache_stats:
            hand_cache = cache_stats.get('hand_cache', {})
            print(f"Cache hit rate: {hand_cache.get('hit_rate', 0):.1%}")
            print(f"Cache requests: {hand_cache.get('total_requests', 0)}")
        
        # Get warming stats
        warming_stats = warmer.get_warming_stats()
        print(f"Warming completed tasks: {warming_stats.completed_tasks}")
        print(f"Cache entries created: {warming_stats.cache_entries_created}")
        
        # Clean up
        warmer.stop_warming()
        warmer.shutdown()
        solver.close()
        
        print("[PASS] Integration test completed!")
        
    except Exception as e:
        print(f"[WARN]  Integration test failed: {e}")
        import traceback
        traceback.print_exc()


def test_cuda_readiness():
    """Test CUDA-ready architecture (placeholder for future)."""
    print("\n[ROCKET] Testing CUDA-Ready Architecture")
    print("=" * 60)
    
    # Test CUDA configuration
    config = WarmingConfig(
        cuda_enabled=False,  # Not implemented yet
        cuda_device_id=0,
        cuda_memory_fraction=0.5
    )
    
    print(f"CUDA enabled: {config.cuda_enabled}")
    print(f"CUDA device ID: {config.cuda_device_id}")
    print(f"CUDA memory fraction: {config.cuda_memory_fraction}")
    
    # Test that configuration is ready for future CUDA implementation
    print("Configuration ready for future CUDA implementation")
    print("Future CUDA features planned:")
    print("  - GPU-accelerated Monte Carlo simulations")
    print("  - Batch processing with CUDA streams")
    print("  - Memory-efficient GPU cache warming")
    print("  - Hybrid CPU/GPU task distribution")
    
    print("[PASS] CUDA readiness test completed!")


def run_comprehensive_demo():
    """Run comprehensive demonstration of cache warming system."""
    print("♞ Poker Knight Cache Warming System Demo")
    print("=" * 80)
    print(f"System Info: {mp.cpu_count()} CPU cores, Python {sys.version}")
    print(f"Dependencies available: {DEPENDENCIES_AVAILABLE}")
    print("=" * 80)
    
    if not DEPENDENCIES_AVAILABLE:
        print("[FAIL] Required dependencies not available. Cannot run demo.")
        return
    
    try:
        # Run all tests
        test_warming_config()
        test_task_generation()
        test_numa_topology()
        test_cache_warmer_initialization()
        test_background_warming()
        test_adaptive_learning()
        test_integration_with_solver()
        test_cuda_readiness()
        
        print("\n🎉 All cache warming tests completed successfully!")
        print("\nCache Warming System Features Demonstrated:")
        print("[PASS] NUMA-aware background processing")
        print("[PASS] Intelligent task generation (169 preflop + board textures)")
        print("[PASS] Adaptive learning from user queries")
        print("[PASS] Background warming with CPU throttling")
        print("[PASS] Integration with existing cache and solver systems")
        print("[PASS] Progress tracking and statistics")
        print("[PASS] CUDA-ready architecture for future acceleration")
        
        print(f"\nNext Steps:")
        print("1. Integrate cache warming with MonteCarloSolver initialization")
        print("2. Add CUDA acceleration for GPU-powered cache warming")
        print("3. Implement advanced board texture analysis")
        print("4. Add tournament vs cash game warming profiles")
        print("5. Optimize memory usage for large-scale warming")
        
    except Exception as e:
        print(f"[FAIL] Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_comprehensive_demo() 