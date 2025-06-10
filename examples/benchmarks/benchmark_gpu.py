#!/usr/bin/env python3
"""Benchmark GPU vs CPU performance."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

import time
from poker_knight import solve_poker_hand
import poker_knight.solver

print("=== GPU vs CPU Performance Benchmark ===\n")

# Test scenarios
scenarios = [
    (['A♠', 'K♠'], 2, 10000, "AKs vs 2 - Fast"),
    (['Q♥', 'Q♦'], 3, 100000, "QQ vs 3 - Default"),
    (['7♠', '2♥'], 1, 100000, "72o vs 1 - Default"),
    (['A♠', 'A♥'], 4, 500000, "AA vs 4 - Precision"),
]

# Store results
results = []

for hero_hand, num_opponents, num_sims, description in scenarios:
    print(f"\nTest: {description}")
    print(f"Simulations: {num_sims:,}")
    
    # Test with GPU
    poker_knight.solver.CUDA_AVAILABLE = True
    start = time.perf_counter()
    gpu_result = solve_poker_hand(hero_hand, num_opponents, simulation_mode='default')
    gpu_time = time.perf_counter() - start
    
    # Test with CPU (force disable GPU)
    poker_knight.solver.CUDA_AVAILABLE = False
    start = time.perf_counter()
    cpu_result = solve_poker_hand(hero_hand, num_opponents, simulation_mode='default')
    cpu_time = time.perf_counter() - start
    
    # Restore GPU availability
    poker_knight.solver.CUDA_AVAILABLE = True
    
    # Calculate speedup
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0
    
    print(f"GPU: {gpu_time*1000:.1f}ms (Win: {gpu_result.win_probability:.1%})")
    print(f"CPU: {cpu_time*1000:.1f}ms (Win: {cpu_result.win_probability:.1%})")
    print(f"Speedup: {speedup:.1f}x")
    
    # Check accuracy
    diff = abs(gpu_result.win_probability - cpu_result.win_probability)
    print(f"Difference: {diff:.1%}")
    
    results.append({
        'description': description,
        'gpu_time': gpu_time * 1000,
        'cpu_time': cpu_time * 1000,
        'speedup': speedup,
        'difference': diff
    })

# Summary
print("\n" + "="*50)
print("PERFORMANCE SUMMARY")
print("="*50)

if results:
    avg_speedup = sum(r['speedup'] for r in results) / len(results)
    max_speedup = max(r['speedup'] for r in results)
    avg_diff = sum(r['difference'] for r in results) / len(results)
    
    print(f"\nAverage speedup: {avg_speedup:.1f}x")
    print(f"Maximum speedup: {max_speedup:.1f}x")
    print(f"Average difference: {avg_diff:.2%}")
    
    print("\nDetailed Results:")
    print(f"{'Scenario':<25} {'GPU (ms)':<10} {'CPU (ms)':<10} {'Speedup':<10} {'Diff':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['description']:<25} {r['gpu_time']:<10.1f} {r['cpu_time']:<10.1f} {r['speedup']:<10.1f} {r['difference']:<10.1%}")