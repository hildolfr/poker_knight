#!/usr/bin/env python3
"""
Example showing GPU usage information in Poker Knight results.

This demonstrates how to check if GPU acceleration was used for your analysis.
"""

from poker_knight import solve_poker_hand, MonteCarloSolver
from poker_knight.cuda import CUDA_AVAILABLE, get_device_info


def example_automatic_gpu():
    """Show automatic GPU usage."""
    print("=== Automatic GPU Usage Example ===\n")
    
    # This will automatically use GPU if available and beneficial
    result = solve_poker_hand(
        ['A♠', 'K♠'],  # Your hand
        3,              # Number of opponents
        ['Q♠', 'J♠', '10♥'],  # Board cards
        simulation_mode='default'  # 100k simulations
    )
    
    # Display results
    print(f"Hand: A♠ K♠")
    print(f"Board: Q♠ J♠ 10♥")
    print(f"Opponents: 3")
    print(f"\nResults:")
    print(f"  Win probability: {result.win_probability:.1%}")
    print(f"  Tie probability: {result.tie_probability:.1%}")
    print(f"  Loss probability: {result.loss_probability:.1%}")
    print(f"  Simulations run: {result.simulations_run:,}")
    print(f"  Execution time: {result.execution_time_ms:.0f}ms")
    
    # GPU usage information
    print(f"\nAcceleration Info:")
    if result.gpu_used:
        print(f"  GPU Used: ✅ Yes")
        print(f"  Backend: {result.backend}")
        print(f"  Device: {result.device}")
    else:
        print(f"  GPU Used: ❌ No (CPU mode)")
        print(f"  Backend: {result.backend}")


def example_force_gpu():
    """Force GPU usage for all simulations."""
    print("\n\n=== Force GPU Usage Example ===\n")
    
    # Create solver with always_use_gpu enabled
    import json
    from pathlib import Path
    
    # Create temporary config
    config = {
        "simulation_settings": {
            "default_simulations": 100000,
            "parallel_processing": True
        },
        "cuda_settings": {
            "enable_cuda": True,
            "always_use_gpu": True,  # Force GPU for all simulations
            "min_simulations_for_gpu": 100
        },
        "performance_settings": {
            "parallel_processing_threshold": 1000
        },
        "output_settings": {
            "include_confidence_interval": True,
            "include_hand_categories": True,
            "decimal_precision": 4
        }
    }
    
    # Save temporary config
    temp_config = Path("temp_gpu_config.json")
    with open(temp_config, 'w') as f:
        json.dump(config, f)
    
    try:
        # Create solver with GPU forced on
        solver = MonteCarloSolver(str(temp_config))
        
        # Even small simulations will use GPU
        result = solver.analyze_hand(
            ['7♥', '7♦'],
            2,
            simulation_mode='fast'  # Only 10k simulations
        )
        
        print(f"Small simulation (10k) with forced GPU:")
        print(f"  Win probability: {result.win_probability:.1%}")
        print(f"  GPU Used: {'✅ Yes' if result.gpu_used else '❌ No'}")
        print(f"  Backend: {result.backend}")
        
    finally:
        # Clean up temp config
        if temp_config.exists():
            temp_config.unlink()


def example_check_gpu_availability():
    """Check GPU availability and info."""
    print("\n\n=== GPU Availability Check ===\n")
    
    if CUDA_AVAILABLE:
        print("✅ CUDA is available!")
        
        info = get_device_info()
        if info:
            print(f"\nGPU Information:")
            print(f"  Name: {info['name']}")
            print(f"  Compute Capability: {info['compute_capability']}")
            print(f"  Total Memory: {info['total_memory'] / (1024**3):.2f} GB")
            print(f"  Multiprocessors: {info['multiprocessors']}")
    else:
        print("❌ CUDA is not available")
        print("\nTo enable GPU acceleration:")
        print("  1. Ensure you have an NVIDIA GPU")
        print("  2. Install CUDA Toolkit 11.0+")
        print("  3. Install CuPy: pip install cupy-cuda11x")


def example_compare_backends():
    """Compare CPU vs GPU performance."""
    print("\n\n=== Backend Comparison ===\n")
    
    if not CUDA_AVAILABLE:
        print("GPU not available for comparison")
        return
    
    # Test scenario
    hand = ['A♠', 'A♥']
    opponents = 4
    board = ['K♠', 'Q♥', 'J♦']
    simulations = 100000
    
    print(f"Test: {' '.join(hand)} vs {opponents} opponents")
    print(f"Board: {' '.join(board)}")
    print(f"Simulations: {simulations:,}\n")
    
    # CPU test - force CPU by disabling GPU
    solver_cpu = MonteCarloSolver()
    if solver_cpu.gpu_solver:
        solver_cpu.gpu_solver = None  # Disable GPU
    
    import time
    start = time.time()
    # For CPU test, we need to use a specific simulation mode
    # Since we want 100k simulations, use 'default' mode
    cpu_result = solver_cpu.analyze_hand(hand, opponents, board, simulation_mode='default')
    cpu_time = time.time() - start
    
    print(f"CPU Results:")
    print(f"  Win: {cpu_result.win_probability:.2%}")
    print(f"  Time: {cpu_time:.3f}s")
    print(f"  Backend: {cpu_result.backend}")
    
    # GPU test - use default solver which should use GPU
    solver_gpu = MonteCarloSolver()
    
    start = time.time()
    gpu_result = solver_gpu.analyze_hand(hand, opponents, board, simulation_mode='default')
    gpu_time = time.time() - start
    
    print(f"\nGPU Results:")
    print(f"  Win: {gpu_result.win_probability:.2%}")
    print(f"  Time: {gpu_time:.3f}s")
    print(f"  Backend: {gpu_result.backend}")
    print(f"  Device: {gpu_result.device}")
    
    if gpu_result.gpu_used:
        speedup = cpu_time / gpu_time
        print(f"\n🚀 GPU Speedup: {speedup:.1f}x faster!")


def main():
    """Run all examples."""
    print("Poker Knight GPU Usage Examples")
    print("=" * 40)
    
    # Check GPU availability first
    example_check_gpu_availability()
    
    # Show automatic GPU usage
    example_automatic_gpu()
    
    # Show forced GPU usage
    if CUDA_AVAILABLE:
        example_force_gpu()
    
    # Compare backends
    example_compare_backends()
    
    print("\n" + "=" * 40)
    print("Examples complete!")


if __name__ == "__main__":
    main()