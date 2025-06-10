#!/usr/bin/env python3
"""Validate GPU results against CPU implementation."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

from poker_knight import solve_poker_hand
import time

print("=== GPU vs CPU Validation Test ===\n")

# Test scenarios
test_cases = [
    # (hero_hand, num_opponents, board, description)
    (['A♠', 'A♥'], 1, None, "Pocket Aces vs 1 opponent"),
    (['7♠', '2♥'], 4, None, "7-2 offsuit vs 4 opponents"),
    (['K♥', 'Q♥'], 2, ['A♥', 'J♥', '10♥'], "Royal flush on flop"),
    (['9♠', '9♦'], 3, ['9♥', '9♣', '2♦'], "Quads on flop"),
    (['A♣', 'K♣'], 2, ['Q♣', 'J♣', '10♣'], "Straight flush on flop"),
]

# Force CPU mode by temporarily disabling CUDA
import poker_knight.solver
original_cuda_available = poker_knight.solver.CUDA_AVAILABLE

results = []

for hero_hand, num_opponents, board, description in test_cases:
    print(f"\nTest: {description}")
    print(f"Hand: {' '.join(hero_hand)} vs {num_opponents} opponent(s)")
    if board:
        print(f"Board: {' '.join(board)}")
    
    # Run with GPU (if available)
    poker_knight.solver.CUDA_AVAILABLE = original_cuda_available
    start = time.time()
    gpu_result = solve_poker_hand(hero_hand, num_opponents, board, simulation_mode='fast')
    gpu_time = (time.time() - start) * 1000
    
    # Run with CPU
    poker_knight.solver.CUDA_AVAILABLE = False
    start = time.time()
    cpu_result = solve_poker_hand(hero_hand, num_opponents, board, simulation_mode='fast')
    cpu_time = (time.time() - start) * 1000
    
    # Compare results
    win_diff = abs(gpu_result.win_probability - cpu_result.win_probability)
    
    print(f"GPU: Win={gpu_result.win_probability:.1%}, Time={gpu_time:.1f}ms, Used={gpu_result.gpu_used}")
    print(f"CPU: Win={cpu_result.win_probability:.1%}, Time={cpu_time:.1f}ms")
    print(f"Difference: {win_diff:.1%}")
    print(f"Speedup: {cpu_time/gpu_time:.1f}x" if gpu_result.gpu_used else "N/A (GPU not used)")
    
    results.append({
        'description': description,
        'gpu_win': gpu_result.win_probability,
        'cpu_win': cpu_result.win_probability,
        'difference': win_diff,
        'gpu_time': gpu_time,
        'cpu_time': cpu_time,
        'speedup': cpu_time/gpu_time if gpu_result.gpu_used else None
    })

# Restore original CUDA availability
poker_knight.solver.CUDA_AVAILABLE = original_cuda_available

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)

max_diff = max(r['difference'] for r in results)
avg_diff = sum(r['difference'] for r in results) / len(results)

print(f"\nAccuracy:")
print(f"  Maximum difference: {max_diff:.2%}")
print(f"  Average difference: {avg_diff:.2%}")
print(f"  All within 5%: {'✅ Yes' if max_diff < 0.05 else '❌ No'}")

if any(r['speedup'] for r in results):
    speedups = [r['speedup'] for r in results if r['speedup']]
    print(f"\nPerformance:")
    print(f"  Average speedup: {sum(speedups)/len(speedups):.1f}x")
    print(f"  Best speedup: {max(speedups):.1f}x")
else:
    print(f"\nPerformance: GPU not used in tests")