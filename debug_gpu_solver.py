#!/usr/bin/env python3
"""Debug GPU solver issues."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

from poker_knight.cuda.gpu_solver import GPUSolver
from poker_knight.cuda.kernels import KernelWrapper

print("=== Debug GPU Solver ===\n")

# Create GPU solver
gpu_solver = GPUSolver()

# Test analyze_hand
hero_hand = ['A♠', 'A♥']
num_opponents = 1
num_simulations = 10000

print(f"Testing: {hero_hand} vs {num_opponents} opponent")
print(f"Requested simulations: {num_simulations}")

# Get the result
result = gpu_solver.analyze_hand(hero_hand, num_opponents, None, num_simulations)

print(f"\nResult:")
print(f"Win probability: {result.win_probability:.1%}")
print(f"Tie probability: {result.tie_probability:.1%}")
print(f"Loss probability: {result.loss_probability:.1%}")
print(f"Simulations run: {result.simulations_run}")
print(f"GPU used: {result.gpu_used}")

# Now test kernel wrapper directly
print("\n=== Testing Kernel Wrapper Directly ===")

import cupy as cp

wrapper = KernelWrapper()
hero_gpu = gpu_solver._cards_to_gpu(hero_hand)
board_gpu = cp.zeros(5, dtype=cp.uint8)

print(f"Hero GPU format: {hero_gpu}")

wins, ties, total = wrapper.monte_carlo(
    hero_gpu, board_gpu, 0, num_opponents, num_simulations, 256
)

print(f"\nKernel results:")
print(f"Wins: {wins}")
print(f"Ties: {ties}")
print(f"Total: {total}")
print(f"Win %: {wins/total*100:.1f}%")

# Check the math
print(f"\nChecking probabilities:")
print(f"Win probability from kernel: {wins/total:.3f}")
print(f"Loss probability from kernel: {(total-wins-ties)/total:.3f}")
print(f"Sum of probabilities: {(wins+ties+(total-wins-ties))/total:.3f}")