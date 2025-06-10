#!/usr/bin/env python3
"""Minimal test of GPU functionality."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

# Test 1: Check CUDA availability
print("=== Test 1: CUDA Availability ===")
from poker_knight.cuda import CUDA_AVAILABLE, get_device_info

print(f"CUDA Available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    info = get_device_info()
    print(f"Device: {info}")

# Test 2: Try to create GPU solver
print("\n=== Test 2: GPU Solver Creation ===")
try:
    from poker_knight.cuda.gpu_solver import GPUSolver
    gpu_solver = GPUSolver()
    print("✅ GPU Solver created successfully")
except Exception as e:
    print(f"❌ Failed to create GPU solver: {e}")

# Test 3: Simple analysis
print("\n=== Test 3: Simple GPU Analysis ===")
try:
    if 'gpu_solver' in locals():
        result = gpu_solver.analyze_hand(['A♠', 'K♠'], 2, None, 10000)
        print(f"✅ Analysis completed: Win={result.win_probability:.1%}")
except Exception as e:
    print(f"❌ GPU analysis failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check kernel compilation
print("\n=== Test 4: Kernel Compilation ===")
try:
    from poker_knight.cuda.kernels import compile_kernels
    kernels = compile_kernels()
    print(f"✅ Compiled kernels: {list(kernels.keys())}")
except Exception as e:
    print(f"❌ Kernel compilation failed: {e}")
    import traceback
    traceback.print_exc()