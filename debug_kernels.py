#!/usr/bin/env python3
"""Debug kernel loading."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knight')

from poker_knight.cuda.kernels import get_kernel_manager, compile_kernels

print("=== Debugging Kernel Loading ===\n")

# Force recompile
print("Compiling kernels...")
try:
    kernels = compile_kernels(force_recompile=True)
    print(f"Successfully compiled {len(kernels)} kernels:")
    for name, kernel in kernels.items():
        print(f"  - {name}: {kernel}")
except Exception as e:
    print(f"Failed to compile: {e}")
    import traceback
    traceback.print_exc()

# Check kernel source
print("\n=== Checking Kernel Source ===")
manager = get_kernel_manager()
source = manager.get_kernel_source()
print(f"Source length: {len(source)} chars")
print("First 200 chars:")
print(source[:200])

# Check if improved kernel is there
if "monte_carlo_improved" in source:
    print("\n✅ Found monte_carlo_improved in source")
else:
    print("\n❌ monte_carlo_improved NOT found in source")
    
# Look for kernel signatures
import re
kernel_funcs = re.findall(r'extern "C" __global__ void (\w+)\(', source)
print(f"\nFound kernel functions: {kernel_funcs}")