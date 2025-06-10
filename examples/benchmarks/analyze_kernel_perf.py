#!/usr/bin/env python3
"""Analyze kernel performance issues."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

import cupy as cp
from poker_knight.cuda.kernels import KernelWrapper
from poker_knight.cuda.gpu_solver import GPUSolver

print("=== Kernel Performance Analysis ===\n")

# Create kernel wrapper
wrapper = KernelWrapper()
gpu_solver = GPUSolver()

# Test parameters
num_simulations = 1000000  # 1M simulations
num_opponents = 2
hero_hand = gpu_solver._cards_to_gpu(['A♠', 'K♠'])
board = cp.zeros(5, dtype=cp.uint8)

print(f"Target simulations: {num_simulations:,}")

# Check grid configuration
threads_per_block = 256

# Calculate what the kernel wrapper will use
if num_simulations <= 10000:
    simulations_per_thread = 100
    total_threads = (num_simulations + simulations_per_thread - 1) // simulations_per_thread
else:
    simulations_per_thread = 1000
    total_threads = (num_simulations + simulations_per_thread - 1) // simulations_per_thread

num_blocks = (total_threads + threads_per_block - 1) // threads_per_block
total_threads = num_blocks * threads_per_block
simulations_per_thread = num_simulations // total_threads
if simulations_per_thread == 0:
    simulations_per_thread = 1
    
actual_total_sims = total_threads * simulations_per_thread

print(f"\nGrid configuration:")
print(f"  Blocks: {num_blocks}")
print(f"  Threads per block: {threads_per_block}")
print(f"  Total threads: {total_threads}")
print(f"  Simulations per thread: {simulations_per_thread}")
print(f"  Actual total simulations: {actual_total_sims:,}")

# Calculate theoretical occupancy
sm_count = 10  # GTX 1060 has 10 SMs
max_threads_per_sm = 2048
max_blocks_per_sm = 32

threads_per_sm = min(total_threads // sm_count, max_threads_per_sm)
blocks_per_sm = min(num_blocks // sm_count, max_blocks_per_sm)

print(f"\nOccupancy analysis:")
print(f"  SMs on device: {sm_count}")
print(f"  Threads per SM: {threads_per_sm} / {max_threads_per_sm} ({threads_per_sm/max_threads_per_sm*100:.1f}%)")
print(f"  Blocks per SM: {blocks_per_sm} / {max_blocks_per_sm}")

# Time the kernel
print(f"\nRunning kernel...")

# Warm up
for _ in range(3):
    wins, ties, total = wrapper.monte_carlo(
        hero_hand, board, 0, num_opponents, 10000, threads_per_block
    )

# Measure
start_event = cp.cuda.Event()
end_event = cp.cuda.Event()

start_event.record()
wins, ties, total = wrapper.monte_carlo(
    hero_hand, board, 0, num_opponents, num_simulations, threads_per_block
)
end_event.record()
end_event.synchronize()

kernel_time = cp.cuda.get_elapsed_time(start_event, end_event)

print(f"\nKernel execution time: {kernel_time:.2f}ms")
print(f"Simulations completed: {total:,}")
print(f"Simulations per second: {total/kernel_time*1000:,.0f}")
print(f"Win rate: {wins/total*100:.1f}%")

# Calculate work efficiency
work_per_thread = simulations_per_thread
print(f"\nWork distribution:")
print(f"  Work per thread: {work_per_thread} simulations")
print(f"  Estimated cycles per simulation: ~1000")
print(f"  Total cycles per thread: ~{work_per_thread * 1000:,}")

# Suggestions
print(f"\n⚠️ Issues identified:")
if simulations_per_thread < 1000:
    print(f"  - Low work per thread ({simulations_per_thread} simulations)")
if threads_per_sm < max_threads_per_sm * 0.5:
    print(f"  - Low occupancy ({threads_per_sm/max_threads_per_sm*100:.1f}%)")
print(f"  - Simple RNG may be limiting performance")
print(f"  - No shared memory usage for deck")
print(f"  - Memory access patterns not optimized")