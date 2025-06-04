"""
♞ Poker Knight Storage Module

High-performance caching and storage solutions for Monte Carlo poker simulations.
Provides LRU caching, memoization, and optional persistent storage for enterprise deployment.

Author: hildolfr
License: MIT
"""

from .cache import (
    HandCache, BoardTextureCache, PreflopRangeCache, 
    CacheConfig, create_cache_key, get_cache_manager, REDIS_AVAILABLE,
    clear_all_caches, ThreadSafeLRUCache, SQLiteCache
)

# Import CacheStats from unified_cache where it actually exists
try:
    from .unified_cache import CacheStats
except ImportError:
    CacheStats = None

__all__ = [
    "HandCache",
    "BoardTextureCache", 
    "PreflopRangeCache",
    "CacheConfig",
    "CacheStats", 
    "create_cache_key",
    "get_cache_manager",
    "REDIS_AVAILABLE",
    "clear_all_caches",
    "ThreadSafeLRUCache",
    "SQLiteCache"
] 