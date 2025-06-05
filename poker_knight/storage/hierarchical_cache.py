"""
♞ Poker Knight Hierarchical Cache System

Advanced multi-layer cache architecture with intelligent routing and eviction.
Implements L1 (memory), L2 (Redis), L3 (SQLite) cache hierarchy with
frequency-weighted eviction and memory pressure handling.

Author: hildolfr
License: MIT
"""

import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import weakref

# Import cache components
from .unified_cache import (
    CacheKey, CacheResult, CacheStats, CacheInterface,
    ThreadSafeMonteCarloCache
)

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layer types in hierarchy."""
    L1_MEMORY = "l1_memory"       # In-memory hot cache
    L2_REDIS = "l2_redis"         # Redis warm cache
    L3_SQLITE = "l3_sqlite"       # SQLite cold cache


@dataclass
class CacheLayerConfig:
    """Configuration for individual cache layer."""
    enabled: bool = True
    max_memory_mb: int = 128
    max_entries: int = 10000
    ttl_seconds: Optional[int] = None
    eviction_policy: str = "lru_frequency"  # "lru", "lfu", "lru_frequency"
    promotion_threshold: int = 2  # Promote to higher layer after N accesses


@dataclass
class HierarchicalCacheConfig:
    """Configuration for hierarchical cache system."""
    # Layer configurations
    l1_config: CacheLayerConfig = None
    l2_config: CacheLayerConfig = None
    l3_config: CacheLayerConfig = None
    
    # Global settings
    enable_auto_promotion: bool = True
    enable_background_demotion: bool = True
    memory_pressure_threshold: float = 0.85  # System memory threshold
    cache_warming_enabled: bool = True
    
    # Performance settings
    max_promotion_batch_size: int = 100
    demotion_check_interval_seconds: int = 300  # 5 minutes
    stats_update_interval_seconds: int = 60
    
    def __post_init__(self):
        if self.l1_config is None:
            self.l1_config = CacheLayerConfig(
                max_memory_mb=128,
                max_entries=5000,
                ttl_seconds=3600,  # 1 hour
                eviction_policy="lru_frequency"
            )
        
        if self.l2_config is None:
            self.l2_config = CacheLayerConfig(
                max_memory_mb=256,
                max_entries=20000,
                ttl_seconds=7200,  # 2 hours
                eviction_policy="lru_frequency"
            )
        
        if self.l3_config is None:
            self.l3_config = CacheLayerConfig(
                max_memory_mb=512,
                max_entries=100000,
                ttl_seconds=86400,  # 24 hours
                eviction_policy="lru"
            )


@dataclass
class AccessStats:
    """Track access patterns for intelligent caching."""
    access_count: int = 0
    last_access_time: float = 0.0
    access_frequency: float = 0.0  # Accesses per hour
    cache_layer: Optional[CacheLayer] = None
    promotion_candidate: bool = False
    
    def update_access(self):
        """Update access statistics."""
        current_time = time.time()
        self.access_count += 1
        
        # Calculate frequency (accesses per hour)
        if self.last_access_time > 0:
            time_diff_hours = (current_time - self.last_access_time) / 3600
            if time_diff_hours > 0:
                # Exponential moving average for frequency
                new_frequency = 1.0 / time_diff_hours
                self.access_frequency = (0.8 * self.access_frequency + 0.2 * new_frequency)
        
        self.last_access_time = current_time


@dataclass
class HierarchicalCacheStats:
    """Statistics for hierarchical cache system."""
    # Layer statistics
    l1_stats: CacheStats
    l2_stats: CacheStats
    l3_stats: CacheStats
    
    # Hierarchy statistics
    total_requests: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    total_misses: int = 0
    
    # Performance statistics
    promotions: int = 0
    demotions: int = 0
    auto_evictions: int = 0
    memory_pressure_events: int = 0
    
    # Timing statistics
    avg_l1_response_time_ms: float = 0.0
    avg_l2_response_time_ms: float = 0.0
    avg_l3_response_time_ms: float = 0.0
    
    @property
    def overall_hit_rate(self) -> float:
        """Calculate overall hit rate across all layers."""
        total_hits = self.l1_hits + self.l2_hits + self.l3_hits
        if self.total_requests == 0:
            return 0.0
        return total_hits / self.total_requests
    
    @property
    def l1_hit_rate(self) -> float:
        """L1 cache hit rate."""
        if self.total_requests == 0:
            return 0.0
        return self.l1_hits / self.total_requests


class MemoryPressureMonitor:
    """Monitors system memory pressure and triggers cache management."""
    
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self._last_check = 0
        self._check_interval = 5.0  # Check every 5 seconds
        self._pressure_events = 0
    
    def check_memory_pressure(self) -> Tuple[bool, float, str]:
        """
        Check if system is under memory pressure.
        
        Returns:
            (is_under_pressure, current_usage, recommendation)
        """
        current_time = time.time()
        
        # Don't check too frequently
        if current_time - self._last_check < self._check_interval:
            return False, 0.0, ""
        
        self._last_check = current_time
        
        try:
            memory = psutil.virtual_memory()
            usage_ratio = memory.percent / 100.0
            
            if usage_ratio > self.threshold:
                self._pressure_events += 1
                severity = "moderate" if usage_ratio < 0.95 else "critical"
                recommendation = f"Aggressive eviction needed - {severity} memory pressure"
                return True, usage_ratio, recommendation
            
            return False, usage_ratio, ""
            
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
            return False, 0.0, ""


class FrequencyWeightedLRU:
    """LRU cache with frequency weighting for intelligent eviction."""
    
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._cache = OrderedDict()
        self._access_stats = {}
        self._lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get item and update access patterns."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                
                # Update access stats
                if key not in self._access_stats:
                    self._access_stats[key] = AccessStats()
                self._access_stats[key].update_access()
                
                return value
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item and handle eviction if needed."""
        with self._lock:
            if key in self._cache:
                # Update existing
                self._cache.pop(key)
                self._cache[key] = value
                return True
            
            # Check if eviction needed
            if len(self._cache) >= self.max_size:
                evicted_key = self._select_eviction_candidate()
                if evicted_key:
                    self._cache.pop(evicted_key, None)
                    self._access_stats.pop(evicted_key, None)
            
            # Add new item
            self._cache[key] = value
            if key not in self._access_stats:
                self._access_stats[key] = AccessStats()
            
            return True
    
    def _select_eviction_candidate(self) -> Optional[str]:
        """Select item for eviction using frequency-weighted LRU."""
        if not self._cache:
            return None
        
        # Calculate eviction scores (lower = more likely to evict)
        candidates = []
        current_time = time.time()
        
        for key in list(self._cache.keys()):
            stats = self._access_stats.get(key)
            if not stats:
                # No stats = immediate eviction candidate
                return key
            
            # Score based on recency and frequency
            recency_score = current_time - stats.last_access_time
            frequency_penalty = 1.0 / (stats.access_frequency + 1.0)
            
            score = recency_score * frequency_penalty
            candidates.append((score, key))
        
        # Sort by score (highest score = oldest and least frequent)
        candidates.sort(reverse=True)
        return candidates[0][1] if candidates else None
    
    def size(self) -> int:
        """Current cache size."""
        return len(self._cache)
    
    def clear(self):
        """Clear cache."""
        with self._lock:
            self._cache.clear()
            self._access_stats.clear()


class HierarchicalCache(CacheInterface):
    """
    Multi-layer cache with intelligent routing and promotion/demotion.
    
    L1: Hot memory cache - fastest access, smallest size
    L2: Warm Redis cache - fast network access, medium size  
    L3: Cold SQLite cache - persistent storage, largest size
    """
    
    def __init__(self, config: Optional[HierarchicalCacheConfig] = None):
        self.config = config or HierarchicalCacheConfig()
        
        # Initialize cache layers
        self._l1_cache = None
        self._l2_cache = None
        self._l3_cache = None
        
        # Access tracking
        self._access_stats = {}
        self._promotion_candidates = set()
        
        # Memory monitoring
        self._memory_monitor = MemoryPressureMonitor(
            self.config.memory_pressure_threshold
        )
        
        # Statistics
        self._stats = HierarchicalCacheStats(
            l1_stats=CacheStats(),
            l2_stats=CacheStats(),
            l3_stats=CacheStats()
        )
        
        # Threading
        self._lock = threading.RLock()
        self._background_thread = None
        self._stop_background = threading.Event()
        
        # Initialize layers
        self._initialize_cache_layers()
        self._start_background_management()
        
        logger.info("Hierarchical cache system initialized")
    
    def _initialize_cache_layers(self):
        """Initialize individual cache layers."""
        try:
            # L1: Memory cache
            if self.config.l1_config.enabled:
                self._l1_cache = ThreadSafeMonteCarloCache(
                    max_memory_mb=self.config.l1_config.max_memory_mb,
                    max_entries=self.config.l1_config.max_entries,
                    enable_persistence=False  # L1 is memory only
                )
                logger.info(f"L1 memory cache initialized ({self.config.l1_config.max_memory_mb}MB)")
            
            # L2: Redis cache (using SQLite for now since Redis params not supported)
            if self.config.l2_config.enabled:
                self._l2_cache = ThreadSafeMonteCarloCache(
                    max_memory_mb=self.config.l2_config.max_memory_mb,
                    max_entries=self.config.l2_config.max_entries,
                    enable_persistence=True,
                    sqlite_path="poker_knight_l2_cache.db"
                )
                logger.info(f"L2 cache initialized ({self.config.l2_config.max_memory_mb}MB)")
            
            # L3: SQLite cache
            if self.config.l3_config.enabled:
                self._l3_cache = ThreadSafeMonteCarloCache(
                    max_memory_mb=self.config.l3_config.max_memory_mb,
                    max_entries=self.config.l3_config.max_entries,
                    enable_persistence=True,
                    sqlite_path="poker_knight_l3_cache.db"
                )
                logger.info(f"L3 SQLite cache initialized ({self.config.l3_config.max_memory_mb}MB)")
        
        except Exception as e:
            logger.error(f"Failed to initialize cache layers: {e}")
    
    def get(self, key: CacheKey) -> Optional[CacheResult]:
        """Get from cache with intelligent layer routing."""
        start_time = time.time()
        key_str = key.to_string()
        
        with self._lock:
            self._stats.total_requests += 1
            
            # Try L1 first (fastest)
            if self._l1_cache:
                result = self._l1_cache.get(key)
                if result:
                    self._stats.l1_hits += 1
                    self._update_access_stats(key_str, CacheLayer.L1_MEMORY)
                    self._stats.avg_l1_response_time_ms = self._update_avg_time(
                        self._stats.avg_l1_response_time_ms, start_time
                    )
                    return result
            
            # Try L2 (medium speed)
            if self._l2_cache:
                result = self._l2_cache.get(key)
                if result:
                    self._stats.l2_hits += 1
                    self._update_access_stats(key_str, CacheLayer.L2_REDIS)
                    self._stats.avg_l2_response_time_ms = self._update_avg_time(
                        self._stats.avg_l2_response_time_ms, start_time
                    )
                    
                    # Promote to L1 if configured
                    if self.config.enable_auto_promotion and self._l1_cache:
                        self._promote_to_l1(key, result)
                    
                    return result
            
            # Try L3 (slowest but persistent)
            if self._l3_cache:
                result = self._l3_cache.get(key)
                if result:
                    self._stats.l3_hits += 1
                    self._update_access_stats(key_str, CacheLayer.L3_SQLITE)
                    self._stats.avg_l3_response_time_ms = self._update_avg_time(
                        self._stats.avg_l3_response_time_ms, start_time
                    )
                    
                    # Consider promotion based on access patterns
                    if self.config.enable_auto_promotion:
                        self._consider_promotion(key_str, key, result)
                    
                    return result
            
            # Cache miss
            self._stats.total_misses += 1
            return None
    
    def store(self, key: CacheKey, result: CacheResult) -> bool:
        """Store in appropriate cache layer(s)."""
        key_str = key.to_string()
        success = False
        
        with self._lock:
            # Always try to store in L1 for immediate access
            if self._l1_cache:
                try:
                    # Use store method if available, otherwise put (put doesn't return bool)
                    if hasattr(self._l1_cache, 'store'):
                        if self._l1_cache.store(key, result):
                            self._update_access_stats(key_str, CacheLayer.L1_MEMORY)
                            success = True
                    else:
                        self._l1_cache.put(key, result)
                        self._update_access_stats(key_str, CacheLayer.L1_MEMORY)
                        success = True
                except Exception as e:
                    logger.warning(f"L1 storage failed: {e}")
            
            # Store in L2 for warm cache
            if self._l2_cache:
                try:
                    # Use store method if available, otherwise put (put doesn't return bool)
                    if hasattr(self._l2_cache, 'store'):
                        if self._l2_cache.store(key, result):
                            success = True
                    else:
                        self._l2_cache.put(key, result)
                        success = True
                except Exception as e:
                    logger.warning(f"L2 storage failed: {e}")
            
            # Store in L3 for persistence
            if self._l3_cache:
                try:
                    # Use store method if available, otherwise put (put doesn't return bool)
                    if hasattr(self._l3_cache, 'store'):
                        if self._l3_cache.store(key, result):
                            success = True
                    else:
                        self._l3_cache.put(key, result)
                        success = True
                except Exception as e:
                    logger.warning(f"L3 storage failed: {e}")
        
        return success
    
    def _update_access_stats(self, key_str: str, layer: CacheLayer):
        """Update access statistics for intelligent management."""
        if key_str not in self._access_stats:
            self._access_stats[key_str] = AccessStats()
        
        stats = self._access_stats[key_str]
        stats.update_access()
        stats.cache_layer = layer
        
        # Check for promotion candidates
        if (layer != CacheLayer.L1_MEMORY and 
            stats.access_count >= self.config.l1_config.promotion_threshold):
            stats.promotion_candidate = True
            self._promotion_candidates.add(key_str)
    
    def _promote_to_l1(self, key: CacheKey, result: CacheResult):
        """Promote result to L1 cache."""
        if self._l1_cache:
            try:
                self._l1_cache.put(key, result)
                self._stats.promotions += 1
            except Exception as e:
                logger.warning(f"L1 promotion failed: {e}")
    
    def _consider_promotion(self, key_str: str, key: CacheKey, result: CacheResult):
        """Consider promoting based on access patterns."""
        stats = self._access_stats.get(key_str)
        if not stats:
            return
        
        # Promote if accessed frequently
        if stats.access_frequency > 1.0:  # More than once per hour
            if stats.cache_layer == CacheLayer.L3_SQLITE and self._l2_cache:
                # Promote L3 -> L2
                try:
                    self._l2_cache.put(key, result)
                    self._stats.promotions += 1
                except Exception as e:
                    logger.warning(f"L2 promotion failed: {e}")
            
            elif stats.cache_layer == CacheLayer.L2_REDIS and self._l1_cache:
                # Promote L2 -> L1
                self._promote_to_l1(key, result)
    
    def _update_avg_time(self, current_avg: float, start_time: float) -> float:
        """Update average response time with exponential moving average."""
        response_time_ms = (time.time() - start_time) * 1000
        return 0.9 * current_avg + 0.1 * response_time_ms
    
    def _start_background_management(self):
        """Start background thread for cache management."""
        def background_worker():
            while not self._stop_background.is_set():
                try:
                    # Check memory pressure
                    is_pressure, usage, recommendation = self._memory_monitor.check_memory_pressure()
                    if is_pressure:
                        self._handle_memory_pressure(usage, recommendation)
                    
                    # Process promotion candidates
                    if self.config.enable_auto_promotion:
                        self._process_promotion_candidates()
                    
                    # Update cache statistics
                    self._update_cache_statistics()
                    
                    # Sleep until next check
                    self._stop_background.wait(self.config.demotion_check_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Background cache management error: {e}")
                    self._stop_background.wait(60)  # Wait 1 minute on error
        
        self._background_thread = threading.Thread(target=background_worker, daemon=True)
        self._background_thread.start()
    
    def _handle_memory_pressure(self, usage: float, recommendation: str):
        """Handle system memory pressure by aggressive eviction."""
        logger.warning(f"Memory pressure detected ({usage:.1%}): {recommendation}")
        self._stats.memory_pressure_events += 1
        
        # Aggressive L1 eviction
        if self._l1_cache and usage > 0.9:
            try:
                # Clear 50% of L1 cache
                self._l1_cache.clear()
                self._stats.auto_evictions += 1
                logger.info("Cleared L1 cache due to critical memory pressure")
            except Exception as e:
                logger.error(f"Emergency L1 eviction failed: {e}")
    
    def _process_promotion_candidates(self):
        """Process queued promotion candidates."""
        if not self._promotion_candidates:
            return
        
        candidates = list(self._promotion_candidates)[:self.config.max_promotion_batch_size]
        
        for key_str in candidates:
            try:
                # Remove from candidates
                self._promotion_candidates.discard(key_str)
                
                # Process promotion logic here if needed
                # (Current implementation promotes immediately on access)
                
            except Exception as e:
                logger.warning(f"Promotion processing failed for {key_str}: {e}")
    
    def _update_cache_statistics(self):
        """Update cache layer statistics."""
        try:
            if self._l1_cache:
                self._stats.l1_stats = self._l1_cache.get_stats()
            if self._l2_cache:
                self._stats.l2_stats = self._l2_cache.get_stats()
            if self._l3_cache:
                self._stats.l3_stats = self._l3_cache.get_stats()
        except Exception as e:
            logger.warning(f"Statistics update failed: {e}")
    
    def invalidate(self, pattern: str) -> int:
        """Invalidate entries across all layers."""
        total_invalidated = 0
        
        for cache in [self._l1_cache, self._l2_cache, self._l3_cache]:
            if cache:
                try:
                    invalidated = cache.invalidate(pattern)
                    total_invalidated += invalidated
                except Exception as e:
                    logger.warning(f"Invalidation failed for cache layer: {e}")
        
        return total_invalidated
    
    def get_stats(self) -> HierarchicalCacheStats:
        """Get comprehensive hierarchical cache statistics."""
        self._update_cache_statistics()
        return self._stats
    
    def clear(self) -> bool:
        """Clear all cache layers."""
        success = True
        
        for cache in [self._l1_cache, self._l2_cache, self._l3_cache]:
            if cache:
                try:
                    if not cache.clear():
                        success = False
                except Exception as e:
                    logger.error(f"Cache clear failed: {e}")
                    success = False
        
        # Clear tracking data
        with self._lock:
            self._access_stats.clear()
            self._promotion_candidates.clear()
        
        return success
    
    def shutdown(self):
        """Shutdown hierarchical cache system."""
        logger.info("Shutting down hierarchical cache system")
        
        # Stop background thread
        if self._background_thread:
            self._stop_background.set()
            self._background_thread.join(timeout=10)
        
        # Close cache layers
        for cache in [self._l1_cache, self._l2_cache, self._l3_cache]:
            if cache and hasattr(cache, 'close'):
                try:
                    cache.close()
                except Exception as e:
                    logger.warning(f"Cache layer shutdown failed: {e}")


# Global hierarchical cache instance
_hierarchical_cache = None
_hierarchical_cache_lock = threading.Lock()


def get_hierarchical_cache(config: Optional[HierarchicalCacheConfig] = None) -> HierarchicalCache:
    """Get global hierarchical cache instance."""
    global _hierarchical_cache, _hierarchical_cache_lock
    
    with _hierarchical_cache_lock:
        if _hierarchical_cache is None:
            _hierarchical_cache = HierarchicalCache(config)
        return _hierarchical_cache


def clear_hierarchical_cache():
    """Clear global hierarchical cache."""
    global _hierarchical_cache, _hierarchical_cache_lock
    
    with _hierarchical_cache_lock:
        if _hierarchical_cache:
            _hierarchical_cache.clear()
            _hierarchical_cache.shutdown()
            _hierarchical_cache = None