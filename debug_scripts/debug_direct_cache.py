#!/usr/bin/env python3
"""Debug cache counting issue."""

# Directly test the unified cache
from poker_knight.storage.unified_cache import ThreadSafeMonteCarloCache, create_cache_key, CacheResult

# Create a fresh cache
cache = ThreadSafeMonteCarloCache(max_memory_mb=64, max_entries=100)

# Get initial stats
stats1 = cache.get_stats()
print(f"Initial: requests={stats1.total_requests}, hits={stats1.cache_hits}, misses={stats1.cache_misses}")

# Create a key
key = create_cache_key(["A♠", "K♠"], 2)

# Try to get (should be a miss)
result = cache.get(key)
print(f"Result from get: {result}")

# Get stats after miss
stats2 = cache.get_stats()
print(f"After get: requests={stats2.total_requests}, hits={stats2.cache_hits}, misses={stats2.cache_misses}")

# Store a result
test_result = CacheResult(
    win_probability=0.5,
    tie_probability=0.1,
    loss_probability=0.4,
    simulations_run=10000
)
cache.put(key, test_result)

# Get stats after put
stats3 = cache.get_stats()
print(f"After put: requests={stats3.total_requests}, hits={stats3.cache_hits}, misses={stats3.cache_misses}")

# Try to get again (should be a hit)
result2 = cache.get(key)
print(f"Result from second get: {result2.win_probability if result2 else None}")

# Final stats
stats4 = cache.get_stats()
print(f"After second get: requests={stats4.total_requests}, hits={stats4.cache_hits}, misses={stats4.cache_misses}")