#!/usr/bin/env python3
"""
Simple demonstration of GPU usage in Poker Knight.
Shows that GPU detection and reporting works correctly.
"""

from poker_knight import solve_poker_hand, MonteCarloSolver
from poker_knight.cuda import CUDA_AVAILABLE, get_device_info


def main():
    print("=== Poker Knight GPU Demo ===\n")
    
    # Check GPU availability
    if CUDA_AVAILABLE:
        print("✅ GPU acceleration is available!")
        info = get_device_info()
        if info:
            print(f"   Device: {info['name']}")
            print(f"   Memory: {info['total_memory'] / (1024**3):.2f} GB")
    else:
        print("❌ GPU acceleration not available")
        print("   The solver will use CPU mode")
    
    print("\n=== Running Analysis ===")
    
    # Run a simple analysis
    result = solve_poker_hand(
        ['A♠', 'K♠'],  # Ace-King suited
        2,              # vs 2 opponents
        simulation_mode='default'  # 100k simulations
    )
    
    # Display results
    print(f"\nHand: A♠ K♠ vs 2 opponents")
    print(f"Win probability: {result.win_probability:.1%}")
    print(f"Simulations run: {result.simulations_run:,}")
    
    # Display GPU usage info
    print(f"\n=== Execution Details ===")
    if hasattr(result, 'gpu_used'):
        if result.gpu_used:
            print(f"GPU Acceleration: ✅ Used")
            print(f"Backend: {result.backend}")
            if result.device:
                print(f"Device: {result.device}")
        else:
            print(f"GPU Acceleration: ❌ Not used")
            print(f"Backend: {result.backend} (CPU mode)")
            
            # Explain why GPU wasn't used
            if CUDA_AVAILABLE:
                print("\nNote: GPU is available but wasn't used because:")
                print("- The CUDA kernels are still being optimized")
                print("- For now, the solver falls back to CPU mode")
                print("- This ensures reliable results while GPU support is improved")
    
    print("\n=== Configuration ===")
    print("To force GPU usage (when fully implemented):")
    print('1. Set "always_use_gpu": true in config.json')
    print("2. Ensure simulation count is >= 1,000")
    
    # Show how to check if specific simulation would use GPU
    from poker_knight.cuda import should_use_gpu
    
    scenarios = [
        (1000, 2, "Minimum for GPU"),
        (10000, 2, "Small simulation"),
        (100000, 4, "Large simulation"),
        (500000, 6, "Precision mode")
    ]
    
    print("\n=== GPU Usage Guidelines ===")
    for sims, opps, desc in scenarios:
        would_use = should_use_gpu(sims, opps)
        status = "✅ Would use GPU" if would_use and CUDA_AVAILABLE else "❌ Would use CPU"
        print(f"{desc:20} ({sims:,} sims, {opps} opponents): {status}")


if __name__ == "__main__":
    main()