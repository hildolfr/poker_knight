#!/usr/bin/env python3
"""
CUDA Acceleration Demo for Poker Knight

Demonstrates GPU-accelerated poker hand analysis with performance comparisons.
"""

import time
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from poker_knight import solve_poker_hand, MonteCarloSolver
from poker_knight.cuda import CUDA_AVAILABLE, get_device_info
from poker_knight.cuda.benchmark import run_quick_benchmark


def print_header(text):
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(f"{text:^60}")
    print('=' * 60)


def demo_cuda_availability():
    """Check and display CUDA availability."""
    print_header("CUDA Availability Check")
    
    if CUDA_AVAILABLE:
        print("✅ CUDA is available!")
        
        device_info = get_device_info()
        if device_info:
            print(f"\nGPU Device Information:")
            print(f"  Name: {device_info['name']}")
            print(f"  Compute Capability: {device_info['compute_capability']}")
            print(f"  Total Memory: {device_info['total_memory'] / (1024**3):.2f} GB")
            print(f"  Multiprocessors: {device_info['multiprocessors']}")
    else:
        print("❌ CUDA is not available")
        print("\nTo enable GPU acceleration:")
        print("  1. Ensure you have an NVIDIA GPU")
        print("  2. Install CUDA Toolkit 11.0+")
        print("  3. Install CuPy: pip install cupy-cuda11x")


def demo_simple_gpu_analysis():
    """Demonstrate simple GPU-accelerated analysis."""
    print_header("Simple GPU Analysis Demo")
    
    if not CUDA_AVAILABLE:
        print("Skipping GPU demo - CUDA not available")
        return
    
    print("\nAnalyzing pocket Aces vs 3 opponents...")
    
    # Warm up
    solve_poker_hand(['A♠', 'A♥'], 3, simulation_mode='fast')
    
    # Time GPU analysis
    start = time.time()
    result = solve_poker_hand(
        ['A♠', 'A♥'], 
        3,
        simulation_mode='default'  # 100k simulations
    )
    gpu_time = time.time() - start
    
    print(f"\nResults:")
    print(f"  Win Probability: {result.win_probability:.1%}")
    print(f"  Tie Probability: {result.tie_probability:.1%}")
    print(f"  Loss Probability: {result.loss_probability:.1%}")
    print(f"  Simulations: {result.simulations_run:,}")
    print(f"  Execution Time: {gpu_time:.3f}s")
    
    # Check if GPU was used
    if hasattr(result, 'optimization_data') and result.optimization_data:
        if 'gpu_execution' in result.optimization_data:
            gpu_info = result.optimization_data['gpu_execution']
            print(f"\n  🚀 GPU Acceleration Used!")
            print(f"  Backend: {gpu_info['backend']}")
            print(f"  Device: {gpu_info['device']}")


def demo_cpu_vs_gpu_comparison():
    """Compare CPU and GPU performance."""
    print_header("CPU vs GPU Performance Comparison")
    
    if not CUDA_AVAILABLE:
        print("Skipping comparison - CUDA not available")
        return
    
    scenarios = [
        {
            'name': 'Pre-flop Analysis',
            'hero_hand': ['A♠', 'K♠'],
            'opponents': 2,
            'board': None,
            'simulations': 100000
        },
        {
            'name': 'Flop Analysis - Flush Draw',
            'hero_hand': ['A♠', 'K♠'],
            'opponents': 3,
            'board': ['Q♠', 'J♥', '5♠'],
            'simulations': 100000
        },
        {
            'name': 'Complex Multi-way Pot',
            'hero_hand': ['Q♥', 'Q♦'],
            'opponents': 5,
            'board': ['K♠', '9♥', '7♦'],
            'simulations': 200000
        }
    ]
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Hand: {' '.join(scenario['hero_hand'])}")
        print(f"  Opponents: {scenario['opponents']}")
        if scenario['board']:
            print(f"  Board: {' '.join(scenario['board'])}")
        print(f"  Simulations: {scenario['simulations']:,}")
        
        # Create solvers
        cpu_solver = MonteCarloSolver()
        cpu_solver.gpu_solver = None  # Force CPU
        
        gpu_solver = MonteCarloSolver()
        if gpu_solver.gpu_solver is None:
            print("  ❌ GPU solver not available")
            continue
        
        # CPU timing
        start = time.time()
        cpu_result = cpu_solver.analyze_hand(
            scenario['hero_hand'],
            scenario['opponents'],
            scenario['board'],
            num_simulations=scenario['simulations']
        )
        cpu_time = time.time() - start
        
        # GPU timing (force GPU by using large simulation count)
        start = time.time()
        gpu_result = gpu_solver.analyze_hand(
            scenario['hero_hand'],
            scenario['opponents'],
            scenario['board'],
            num_simulations=scenario['simulations']
        )
        gpu_time = time.time() - start
        
        # Results
        print(f"\n  CPU Results:")
        print(f"    Win: {cpu_result.win_probability:.1%}")
        print(f"    Time: {cpu_time:.3f}s")
        
        print(f"\n  GPU Results:")
        print(f"    Win: {gpu_result.win_probability:.1%}")
        print(f"    Time: {gpu_time:.3f}s")
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        print(f"\n  🚀 Speedup: {speedup:.1f}x")


def demo_batch_processing():
    """Demonstrate batch processing capabilities."""
    print_header("Batch Processing Demo")
    
    if not CUDA_AVAILABLE:
        print("Skipping batch demo - CUDA not available")
        return
    
    print("\nProcessing multiple hands in batch...")
    
    hands = [
        {'hero': ['A♠', 'A♥'], 'opponents': 2, 'board': None},
        {'hero': ['K♠', 'K♥'], 'opponents': 3, 'board': ['Q♠', 'J♥', '10♦']},
        {'hero': ['A♠', 'K♠'], 'opponents': 4, 'board': ['Q♠', 'J♠', '10♥']},
        {'hero': ['Q♥', 'Q♦'], 'opponents': 2, 'board': ['K♠', '9♥', '7♦']},
        {'hero': ['J♠', 'J♥'], 'opponents': 5, 'board': None},
    ]
    
    start = time.time()
    results = []
    
    for i, hand in enumerate(hands):
        result = solve_poker_hand(
            hand['hero'],
            hand['opponents'],
            hand['board'],
            simulation_mode='fast'  # Quick analysis
        )
        results.append(result)
        
        print(f"\nHand {i+1}: {' '.join(hand['hero'])}")
        if hand['board']:
            print(f"  Board: {' '.join(hand['board'])}")
        print(f"  Win Probability: {result.win_probability:.1%}")
    
    total_time = time.time() - start
    print(f"\nTotal batch processing time: {total_time:.3f}s")
    print(f"Average time per hand: {total_time/len(hands):.3f}s")


def demo_advanced_features():
    """Demonstrate advanced GPU features."""
    print_header("Advanced GPU Features")
    
    if not CUDA_AVAILABLE:
        print("Skipping advanced features - CUDA not available")
        return
    
    print("\n1. Adaptive Simulation Count")
    print("   GPU automatically adjusts workload for optimal performance")
    
    print("\n2. Memory Pooling")
    print("   Reuses GPU memory allocations for faster execution")
    
    print("\n3. Kernel Caching")
    print("   Compiled CUDA kernels are cached for instant reuse")
    
    print("\n4. Automatic Fallback")
    print("   Seamlessly falls back to CPU if GPU encounters errors")
    
    # Run a quick benchmark
    print("\n5. Quick Benchmark")
    try:
        benchmark_results = run_quick_benchmark()
        
        for bench in benchmark_results['benchmarks']:
            if bench['backend'] == 'gpu':
                gpu_results = bench['results']
                print(f"\n   GPU Performance:")
                for res in gpu_results:
                    print(f"     {res['simulations']:,} sims: {res['throughput']:.0f} sims/sec")
    except Exception as e:
        print(f"   Benchmark failed: {e}")


def main():
    """Run all demos."""
    print("\n🎯 Poker Knight CUDA Acceleration Demo")
    print("=" * 60)
    
    # Check CUDA availability
    demo_cuda_availability()
    
    if CUDA_AVAILABLE:
        # Run demos
        demo_simple_gpu_analysis()
        demo_cpu_vs_gpu_comparison()
        demo_batch_processing()
        demo_advanced_features()
        
        print_header("Demo Complete!")
        print("\nGPU acceleration provides significant speedups for:")
        print("  • Large simulation counts (>10,000)")
        print("  • Multi-opponent scenarios")
        print("  • Batch processing of multiple hands")
        print("  • Tournament simulations with ICM calculations")
    else:
        print("\n⚠️  Install CUDA and CuPy to unlock GPU acceleration!")
        print("\nRunning CPU-only demo...")
        
        # Show CPU performance
        result = solve_poker_hand(['A♠', 'A♥'], 3, simulation_mode='fast')
        print(f"\nCPU Analysis (10k simulations):")
        print(f"  Win Probability: {result.win_probability:.1%}")
        print(f"  Execution Time: {result.execution_time_ms:.0f}ms")


if __name__ == "__main__":
    main()