#!/usr/bin/env python3
"""Clear CUDA kernel cache to force recompilation."""

import shutil
from pathlib import Path

cache_dir = Path("/home/user/Documents/poker_knight/poker_knight/cuda/.cuda_cache")
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print(f"Cleared CUDA cache at {cache_dir}")
else:
    print("No CUDA cache found")

# Also clear CuPy's cache
import os
cupy_cache = Path(os.path.expanduser("~/.cupy/kernel_cache"))
if cupy_cache.exists():
    shutil.rmtree(cupy_cache)
    print(f"Cleared CuPy cache at {cupy_cache}")
else:
    print("No CuPy cache found")