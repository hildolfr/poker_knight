#!/usr/bin/env python3
"""Direct test of improved kernel."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

import cupy as cp
import numpy as np
from poker_knight.cuda.kernels import get_kernel

print("=== Direct Kernel Test ===\n")

# Get the kernel
kernel = get_kernel('monte_carlo_improved')
if kernel is None:
    print("Failed to get kernel!")
    exit(1)

print(f"Got kernel: {kernel}")

# Test parameters
hero_hand = cp.array([140, 156], dtype=cp.uint8)  # A♠ A♥
board_cards = cp.zeros(5, dtype=cp.uint8)  # No board cards
board_size = 0
num_opponents = 1
simulations_per_thread = 100
threads_per_block = 64
num_blocks = 10

# Output arrays
wins = cp.zeros(1, dtype=cp.uint32)
ties = cp.zeros(1, dtype=cp.uint32)

print(f"Hero hand: {hero_hand} (should be pocket aces)")
print(f"Opponents: {num_opponents}")
print(f"Blocks: {num_blocks}, Threads: {threads_per_block}")
print(f"Simulations per thread: {simulations_per_thread}")
print(f"Total simulations: {num_blocks * threads_per_block * simulations_per_thread}")

# Launch kernel
try:
    kernel(
        (num_blocks,), (threads_per_block,),
        (hero_hand, board_cards, board_size, num_opponents,
         simulations_per_thread, wins, ties, np.uint32(42))
    )
    
    cp.cuda.Stream.null.synchronize()
    
    # Get results
    total_sims = num_blocks * threads_per_block * simulations_per_thread
    win_count = int(wins[0])
    tie_count = int(ties[0])
    loss_count = total_sims - win_count - tie_count
    
    print(f"\n✅ Kernel executed successfully!")
    print(f"Wins: {win_count} ({win_count/total_sims*100:.1f}%)")
    print(f"Ties: {tie_count} ({tie_count/total_sims*100:.1f}%)")
    print(f"Losses: {loss_count} ({loss_count/total_sims*100:.1f}%)")
    print(f"Total counted: {win_count + tie_count + loss_count}")
    
    # Sanity check
    if win_count/total_sims > 0.75 and win_count/total_sims < 0.90:
        print("\n✅ Results look reasonable for AA vs 1 opponent!")
    else:
        print(f"\n❌ Results don't look right. Expected ~85% win rate for AA vs 1.")
        
except Exception as e:
    print(f"\n❌ Kernel failed: {e}")
    import traceback
    traceback.print_exc()