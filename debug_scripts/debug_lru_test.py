#!/usr/bin/env python3
"""Debug LRU eviction behavior."""

from collections import OrderedDict

# Test OrderedDict LRU behavior
cache = OrderedDict()

# Add items
for i in range(5):
    cache[f"key{i}"] = f"value{i}"

print("Initial cache:", list(cache.keys()))

# Access key0 (should move to end)
value = cache.pop("key0")
cache["key0"] = value

print("After accessing key0:", list(cache.keys()))

# Add more items and evict
for i in range(5, 8):
    if len(cache) >= 5:
        # Remove oldest (first)
        removed = cache.popitem(last=False)
        print(f"  Evicted: {removed[0]}")
    cache[f"key{i}"] = f"value{i}"
    print(f"  Added key{i}, cache: {list(cache.keys())}")

print("\nFinal cache:", list(cache.keys()))
print("Is key0 still in cache?", "key0" in cache)