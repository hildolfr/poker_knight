#!/usr/bin/env python3
"""Direct CUDA test to isolate kernel issues."""

import cupy as cp
import numpy as np

# Simple test kernel
simple_kernel = cp.RawKernel(r'''
extern "C" __global__
void test_kernel(
    const unsigned char* hero_hand,
    int num_opponents,
    int simulations_per_thread,
    unsigned int* wins,
    unsigned int* ties
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Simple calculation based on hero hand
    unsigned int hero_strength = (hero_hand[0] & 0xF) + (hero_hand[1] & 0xF);
    
    // Simple win calculation
    unsigned int thread_wins = 0;
    unsigned int thread_ties = 0;
    
    for (int i = 0; i < simulations_per_thread; i++) {
        // Simple pseudo-random based on thread index
        unsigned int random_val = (idx * simulations_per_thread + i) % 100;
        
        if (hero_strength > 15) {  // Strong hand
            if (random_val < 70) thread_wins++;
            else if (random_val < 80) thread_ties++;
        } else if (hero_strength > 10) {  // Medium hand
            if (random_val < 50) thread_wins++;
            else if (random_val < 60) thread_ties++;
        } else {  // Weak hand
            if (random_val < 30) thread_wins++;
            else if (random_val < 35) thread_ties++;
        }
    }
    
    // Use atomicAdd to accumulate results
    atomicAdd(&wins[0], thread_wins);
    atomicAdd(&ties[0], thread_ties);
}
''', 'test_kernel')

# Test the kernel
print("=== Direct CUDA Kernel Test ===")

# Create test data
hero_hand = cp.array([0x8C, 0x8D], dtype=cp.uint8)  # AK in GPU format
num_opponents = 2
simulations_per_thread = 100
threads_per_block = 64
num_blocks = 10

# Allocate output arrays
wins = cp.zeros(1, dtype=cp.uint32)
ties = cp.zeros(1, dtype=cp.uint32)

print(f"Hero hand: {hero_hand}")
print(f"Blocks: {num_blocks}, Threads: {threads_per_block}")
print(f"Simulations per thread: {simulations_per_thread}")

# Launch kernel
try:
    simple_kernel(
        (num_blocks,), (threads_per_block,),
        (hero_hand, num_opponents, simulations_per_thread, wins, ties)
    )
    
    # Synchronize
    cp.cuda.Stream.null.synchronize()
    
    # Get results
    total_sims = num_blocks * threads_per_block * simulations_per_thread
    win_count = int(wins[0])
    tie_count = int(ties[0])
    
    print(f"\n✅ Kernel executed successfully!")
    print(f"Total simulations: {total_sims}")
    print(f"Wins: {win_count} ({win_count/total_sims*100:.1f}%)")
    print(f"Ties: {tie_count} ({tie_count/total_sims*100:.1f}%)")
    
except Exception as e:
    print(f"\n❌ Kernel failed: {e}")
    import traceback
    traceback.print_exc()