#!/usr/bin/env python3
"""Debug the CUDA kernel to understand the issue."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

import cupy as cp
import numpy as np
from poker_knight.cuda.gpu_solver import GPUSolver

# Create GPU solver
gpu_solver = GPUSolver()

# Test specific hands
print("=== Debugging CUDA Kernel ===\n")

# Test 1: Convert cards to GPU format
test_hands = [
    (['A♠', 'A♥'], "Pocket Aces"),
    (['K♠', 'K♥'], "Pocket Kings"),
    (['A♠', 'K♠'], "AK suited"),
    (['7♠', '2♥'], "7-2 offsuit"),
]

for hand, desc in test_hands:
    print(f"\nTest: {desc} - {hand}")
    gpu_cards = gpu_solver._cards_to_gpu(hand)
    print(f"GPU format: {gpu_cards}")
    
    # Decode the GPU format
    for i, card in enumerate(gpu_cards):
        card_val = int(card)
        valid = (card_val & 0x80) != 0
        suit = (card_val >> 4) & 0x3
        rank = card_val & 0xF
        print(f"  Card {i}: valid={valid}, suit={suit}, rank={rank} (binary: {card_val:08b})")

# Test 2: Check kernel calculations
print("\n=== Testing Kernel Logic ===")

# Manually calculate what the kernel should produce
from poker_knight.cuda.kernels import KernelWrapper

wrapper = KernelWrapper()

# Test with pocket aces
hero_hand = cp.array([0x8C, 0x8C], dtype=cp.uint8)  # Two aces
board = cp.zeros(5, dtype=cp.uint8)
num_opponents = 2
num_simulations = 1000

print(f"\nTesting with hero_hand: {hero_hand}")
print(f"Opponents: {num_opponents}")
print(f"Simulations: {num_simulations}")

try:
    wins, ties, total = wrapper.monte_carlo(
        hero_hand, board, 0, num_opponents, num_simulations, 64
    )
    print(f"\nResults:")
    print(f"Wins: {wins} ({wins/total*100:.1f}%)")
    print(f"Ties: {ties} ({ties/total*100:.1f}%)")
    print(f"Total: {total}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Direct kernel test with known values
print("\n=== Direct Kernel Test ===")

# Create a simple test kernel to verify our understanding
test_kernel = cp.RawKernel(r'''
extern "C" __global__
void debug_kernel(
    const unsigned char* hero_hand,
    unsigned int* debug_output
) {
    int idx = threadIdx.x;
    if (idx < 2) {
        // Output hero hand values
        debug_output[idx] = hero_hand[idx];
        
        // Extract and output components
        unsigned char card = hero_hand[idx];
        debug_output[2 + idx] = card & 0xF;  // rank
        debug_output[4 + idx] = (card >> 4) & 0x3;  // suit
        debug_output[6 + idx] = (card & 0x80) ? 1 : 0;  // valid flag
    }
    
    if (idx == 0) {
        // Calculate hand strength as kernel does
        unsigned int hero_strength = (hero_hand[0] & 0xF) + (hero_hand[1] & 0xF);
        debug_output[8] = hero_strength;
    }
}
''', 'debug_kernel')

# Test the debug kernel
hero_aces = cp.array([0x8C, 0x8C], dtype=cp.uint8)  # Two aces (rank 12 each)
debug_output = cp.zeros(10, dtype=cp.uint32)

test_kernel((1,), (32,), (hero_aces, debug_output))
cp.cuda.Stream.null.synchronize()

print("Debug output:")
print(f"Card 0 raw: {debug_output[0]} (0x{int(debug_output[0]):02X})")
print(f"Card 1 raw: {debug_output[1]} (0x{int(debug_output[1]):02X})")
print(f"Card 0 rank: {debug_output[2]}")
print(f"Card 1 rank: {debug_output[3]}")
print(f"Card 0 suit: {debug_output[4]}")
print(f"Card 1 suit: {debug_output[5]}")
print(f"Card 0 valid: {debug_output[6]}")
print(f"Card 1 valid: {debug_output[7]}")
print(f"Hero strength: {debug_output[8]}")