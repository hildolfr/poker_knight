#!/usr/bin/env python3
"""Debug cache initialization."""

import sys
sys.stdout = sys.stderr  # Force unbuffered output

from poker_knight.solver import MonteCarloSolver

print("Creating solver...", flush=True)
solver = MonteCarloSolver(enable_caching=True)
print("Solver created", flush=True)

# Get stats
stats = solver.get_cache_stats()
print(f"Initial cache stats: {stats}", flush=True)

solver.close()
print("Done", flush=True)