#!/usr/bin/env python3
"""Detailed GPU benchmark with timing breakdown."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

import time
import cupy as cp
from poker_knight import solve_poker_hand, MonteCarloSolver
from poker_knight.cuda.gpu_solver import GPUSolver

print("=== Detailed GPU Performance Analysis ===\n")

# Create GPU solver directly
gpu_solver = GPUSolver()

# Test scenarios
scenarios = [
    (100_000, "100K simulations"),
    (1_000_000, "1M simulations"),
    (5_000_000, "5M simulations"),
]

for num_sims, description in scenarios:
    print(f"\n{description}")
    print("-" * 40)
    
    hero_hand = ['A♠', 'K♠']
    num_opponents = 2
    
    # Time GPU solve
    start = time.perf_counter()
    
    # Breakdown timing
    t1 = time.perf_counter()
    gpu_result = gpu_solver.analyze_hand(hero_hand, num_opponents, None, num_sims)
    t2 = time.perf_counter()
    
    total_gpu_time = (t2 - t1) * 1000
    
    print(f"GPU Total Time: {total_gpu_time:.1f}ms")
    print(f"Simulations: {gpu_result.simulations_run:,}")
    print(f"Win rate: {gpu_result.win_probability:.1%}")
    print(f"Simulations/second: {gpu_result.simulations_run/(total_gpu_time/1000):,.0f}")
    
    # For comparison, time CPU version
    import poker_knight.solver
    original_cuda = poker_knight.solver.CUDA_AVAILABLE
    poker_knight.solver.CUDA_AVAILABLE = False
    
    t1 = time.perf_counter()
    cpu_result = solve_poker_hand(hero_hand, num_opponents, simulation_mode='default')
    t2 = time.perf_counter()
    
    cpu_time = (t2 - t1) * 1000
    
    poker_knight.solver.CUDA_AVAILABLE = original_cuda
    
    print(f"\nCPU Time: {cpu_time:.1f}ms")
    print(f"CPU Win rate: {cpu_result.win_probability:.1%}")
    print(f"Speedup: {cpu_time/total_gpu_time:.1f}x")
    
    # Check if we're memory or compute bound
    print(f"\nAnalysis:")
    bytes_per_sim = 7 * 2  # 7 cards * 2 bytes roughly
    bandwidth_used = (bytes_per_sim * gpu_result.simulations_run) / (total_gpu_time / 1000) / 1e9
    print(f"Effective bandwidth: {bandwidth_used:.1f} GB/s")
    
    # GTX 1060 theoretical: 192 GB/s
    print(f"Bandwidth utilization: {bandwidth_used/192*100:.1f}%")

# Direct kernel test
print("\n\n=== Direct Kernel Performance ===")
from poker_knight.cuda.kernels import KernelWrapper

wrapper = KernelWrapper()
hero_gpu = gpu_solver._cards_to_gpu(['A♠', 'A♥'])
board_gpu = cp.zeros(5, dtype=cp.uint8)

# Warm up
for _ in range(5):
    wrapper.monte_carlo(hero_gpu, board_gpu, 0, 1, 10000, 256)

# Time large simulation
num_sims = 10_000_000
print(f"\nTesting {num_sims:,} simulations directly...")

start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

start_event.record()
wins, ties, total = wrapper.monte_carlo(
    hero_gpu, board_gpu, 0, 1, num_sims, 256
)
end_event.record()
end_event.synchronize()

kernel_time = cp.cuda.get_elapsed_time(start_event, end_event)

print(f"Kernel time: {kernel_time:.2f}ms")
print(f"Simulations: {total:,}")
print(f"Simulations/second: {total/(kernel_time/1000):,.0f}")
print(f"Win rate: {wins/total*100:.1f}%")