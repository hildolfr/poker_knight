"""
♞ Poker Knight Intelligent Caching System

High-performance caching system for Monte Carlo poker simulations with:
- LRU hand cache for frequently analyzed scenarios
- Board texture memoization for common patterns  
- Preflop range cache (169 hand combinations)
- Optional Redis/SQLite persistence for enterprise deployment
- Automatic fallback: Redis -> SQLite -> Memory-only

Performance targets:
- Sub-10ms response for cached scenarios
- 80%+ cache hit rate in typical usage
- Configurable memory limits with intelligent eviction

Author: hildolfr
License: MIT
"""

import time
import json
import hashlib
import pickle
import sqlite3
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from functools import lru_cache
from collections import OrderedDict
import threading
import weakref

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    # Memory cache settings
    max_memory_mb: int = 512  # Maximum memory usage in MB
    hand_cache_size: int = 10000  # Number of hand scenarios to cache
    board_texture_cache_size: int = 5000  # Number of board textures to cache
    preflop_cache_enabled: bool = True  # Enable preflop cache (169 combinations)
    
    # Performance settings
    cache_hit_rate_target: float = 0.8  # Target cache hit rate
    eviction_batch_size: int = 100  # Number of items to evict at once
    cache_cleanup_interval: int = 300  # Cleanup interval in seconds
    
    # Persistence settings (optional)
    enable_persistence: bool = False
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    sqlite_path: str = "poker_knight_cache.db"  # SQLite fallback database path
    disk_cache_path: Optional[str] = None  # Legacy parameter, kept for compatibility
    
    # Cache key settings
    include_position_in_key: bool = True
    include_stack_depth_in_key: bool = True
    key_precision_digits: int = 3  # Decimal precision for float values in keys


@dataclass
class CacheStats:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    memory_usage_mb: float = 0.0
    hit_rate: float = 0.0
    evictions: int = 0
    persistence_saves: int = 0
    persistence_loads: int = 0
    last_cleanup: Optional[float] = None
    persistence_type: str = "none"  # Track which persistence backend is being used


def create_cache_key(hero_hand: List[str], 
                    num_opponents: int,
                    board_cards: Optional[List[str]] = None,
                    simulation_mode: str = "default",
                    hero_position: Optional[str] = None,
                    stack_depth: Optional[float] = None,
                    config: Optional[CacheConfig] = None) -> str:
    """
    Create a consistent cache key for poker scenarios.
    
    Args:
        hero_hand: Hero's hole cards
        num_opponents: Number of opponents
        board_cards: Community cards (if any)
        simulation_mode: Simulation mode
        hero_position: Position (if position-aware caching enabled)
        stack_depth: Stack depth (if stack-aware caching enabled)
        config: Cache configuration
        
    Returns:
        Hash-based cache key
    """
    if config is None:
        config = CacheConfig()
    
    # Create normalized key components
    key_parts = [
        "|".join(sorted(hero_hand)),  # Sort hole cards for consistency
        str(num_opponents),
        simulation_mode
    ]
    
    # Add board cards if present
    if board_cards:
        key_parts.append("|".join(sorted(board_cards)))
    else:
        key_parts.append("preflop")
    
    # Add position if enabled
    if config.include_position_in_key and hero_position:
        key_parts.append(hero_position)
    
    # Add stack depth if enabled (rounded for consistency)
    if config.include_stack_depth_in_key and stack_depth is not None:
        rounded_depth = round(stack_depth, config.key_precision_digits)
        key_parts.append(f"depth_{rounded_depth}")
    
    # Create hash of key components
    key_string = "|".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


class ThreadSafeLRUCache:
    """Thread-safe LRU cache implementation with memory management."""
    
    def __init__(self, max_size: int, max_memory_mb: float):
        self.max_size = max_size
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self._cache = OrderedDict()
        self._memory_usage = 0
        self._lock = threading.RLock()
        self._access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache, moving it to end (most recently used)."""
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                value = self._cache.pop(key)
                self._cache[key] = value
                self._access_times[key] = time.time()
                return value
            return None
    
    def put(self, key: str, value: Any) -> bool:
        """Put item in cache, evicting old items if necessary."""
        with self._lock:
            # Estimate memory usage
            value_size = self._estimate_size(value)
            
            # Check if single item exceeds memory limit
            if value_size > self.max_memory_bytes:
                return False
            
            # Evict items if necessary
            while (len(self._cache) >= self.max_size or 
                   self._memory_usage + value_size > self.max_memory_bytes):
                if not self._cache:
                    break
                self._evict_oldest()
            
            # Add new item
            if key in self._cache:
                # Update existing item
                old_size = self._estimate_size(self._cache[key])
                self._memory_usage -= old_size
            
            self._cache[key] = value
            self._memory_usage += value_size
            self._access_times[key] = time.time()
            
            return True
    
    def _evict_oldest(self) -> None:
        """Evict the least recently used item."""
        if self._cache:
            key, value = self._cache.popitem(last=False)
            self._memory_usage -= self._estimate_size(value)
            self._access_times.pop(key, None)
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate memory size of object."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback estimate
            return 1024  # 1KB default estimate
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'memory_usage_mb': self._memory_usage / (1024 * 1024),
                'max_memory_mb': self.max_memory_bytes / (1024 * 1024),
                'memory_utilization': self._memory_usage / self.max_memory_bytes if self.max_memory_bytes > 0 else 0
            }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._memory_usage = 0


class SQLiteCache:
    """
    SQLite-based persistent cache as fallback for Redis.
    
    Provides lightweight persistent caching when Redis is not available.
    Thread-safe with connection pooling and automatic table creation.
    """
    
    def __init__(self, db_path: str, table_prefix: str = "cache"):
        self.db_path = db_path
        self.table_name = f"{table_prefix}_entries"
        self._lock = threading.RLock()
        self._local = threading.local()  # Thread-local storage for connections
        
        # Initialize database
        self._init_database()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path, 
                check_same_thread=False,
                timeout=30.0
            )
            # Enable WAL mode for better concurrent access
            self._local.connection.execute("PRAGMA journal_mode=WAL")
            self._local.connection.execute("PRAGMA synchronous=NORMAL")
            self._local.connection.execute("PRAGMA cache_size=10000")
        return self._local.connection
    
    def _init_database(self) -> None:
        """Initialize SQLite database and tables."""
        try:
            conn = self._get_connection()
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    cache_key TEXT PRIMARY KEY,
                    data BLOB NOT NULL,
                    created_at REAL NOT NULL,
                    accessed_at REAL NOT NULL,
                    access_count INTEGER DEFAULT 1
                )
            """)
            
            # Create index for cleanup operations
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_accessed 
                ON {self.table_name}(accessed_at)
            """)
            
            conn.commit()
        except sqlite3.Error:
            pass  # Fail gracefully if database setup fails
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get cached data from SQLite."""
        try:
            conn = self._get_connection()
            cursor = conn.execute(f"""
                SELECT data FROM {self.table_name} 
                WHERE cache_key = ?
            """, (cache_key,))
            
            row = cursor.fetchone()
            if row:
                # Update access time and count
                current_time = time.time()
                conn.execute(f"""
                    UPDATE {self.table_name} 
                    SET accessed_at = ?, access_count = access_count + 1
                    WHERE cache_key = ?
                """, (current_time, cache_key))
                conn.commit()
                
                # Deserialize data
                return pickle.loads(row[0])
            
            return None
        except (sqlite3.Error, pickle.PickleError):
            return None
    
    def set(self, cache_key: str, data: Any, expiration_hours: int = 24) -> bool:
        """Store data in SQLite cache."""
        try:
            serialized_data = pickle.dumps(data)
            current_time = time.time()
            
            conn = self._get_connection()
            conn.execute(f"""
                INSERT OR REPLACE INTO {self.table_name} 
                (cache_key, data, created_at, accessed_at, access_count)
                VALUES (?, ?, ?, ?, 1)
            """, (cache_key, serialized_data, current_time, current_time))
            
            conn.commit()
            return True
        except (sqlite3.Error, pickle.PickleError):
            return False
    
    def delete(self, cache_key: str) -> bool:
        """Delete entry from SQLite cache."""
        try:
            conn = self._get_connection()
            conn.execute(f"DELETE FROM {self.table_name} WHERE cache_key = ?", (cache_key,))
            conn.commit()
            return True
        except sqlite3.Error:
            return False
    
    def clear(self) -> bool:
        """Clear all entries from SQLite cache."""
        try:
            conn = self._get_connection()
            conn.execute(f"DELETE FROM {self.table_name}")
            conn.commit()
            return True
        except sqlite3.Error:
            return False
    
    def cleanup_expired(self, max_age_hours: int = 168) -> int:
        """Remove old entries from cache (default: 1 week)."""
        try:
            cutoff_time = time.time() - (max_age_hours * 3600)
            conn = self._get_connection()
            
            cursor = conn.execute(f"""
                DELETE FROM {self.table_name} 
                WHERE accessed_at < ?
            """, (cutoff_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count
        except sqlite3.Error:
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get SQLite cache statistics."""
        try:
            conn = self._get_connection()
            
            # Get basic stats
            cursor = conn.execute(f"""
                SELECT 
                    COUNT(*) as total_entries,
                    AVG(access_count) as avg_access_count,
                    MAX(accessed_at) as last_access,
                    MIN(created_at) as oldest_entry
                FROM {self.table_name}
            """)
            
            stats = cursor.fetchone()
            
            # Get database file size
            try:
                db_size_mb = os.path.getsize(self.db_path) / (1024 * 1024)
            except OSError:
                db_size_mb = 0.0
            
            return {
                'total_entries': stats[0] if stats else 0,
                'avg_access_count': stats[1] if stats and stats[1] else 0.0,
                'last_access': stats[2] if stats and stats[2] else None,
                'oldest_entry': stats[3] if stats and stats[3] else None,
                'database_size_mb': db_size_mb,
                'database_path': self.db_path
            }
        except sqlite3.Error:
            return {
                'total_entries': 0,
                'database_size_mb': 0.0,
                'database_path': self.db_path,
                'error': 'Unable to get statistics'
            }
    
    def close(self) -> None:
        """Close database connections."""
        if hasattr(self._local, 'connection'):
            try:
                self._local.connection.close()
                delattr(self._local, 'connection')
            except:
                pass


class HandCache:
    """
    Main cache for poker hand analysis results.
    
    Implements Task 1.3.1: LRU hand cache for frequently analyzed scenarios
    With Redis -> SQLite -> Memory-only fallback strategy
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self.stats = CacheStats()
        
        # Initialize memory cache
        memory_per_cache = self.config.max_memory_mb // 3  # Split between hand, board, preflop
        self._memory_cache = ThreadSafeLRUCache(
            max_size=self.config.hand_cache_size,
            max_memory_mb=memory_per_cache
        )
        
        # Initialize persistence if enabled
        self._redis_client = None
        self._sqlite_cache = None
        
        if self.config.enable_persistence:
            # Try Redis first
            if REDIS_AVAILABLE:
                try:
                    self._redis_client = redis.Redis(
                        host=self.config.redis_host,
                        port=self.config.redis_port,
                        db=self.config.redis_db,
                        password=self.config.redis_password,
                        decode_responses=False  # We'll handle pickle data
                    )
                    # Test connection
                    self._redis_client.ping()
                    self.stats.persistence_type = "redis"
                except Exception:
                    self._redis_client = None
            
            # Fall back to SQLite if Redis not available
            if self._redis_client is None:
                try:
                    self._sqlite_cache = SQLiteCache(
                        db_path=self.config.sqlite_path,
                        table_prefix="poker_knight_hand"
                    )
                    self.stats.persistence_type = "sqlite"
                except Exception:
                    self._sqlite_cache = None
                    self.stats.persistence_type = "none"
        
        # Last cleanup time
        self._last_cleanup = time.time()
    
    def get_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached simulation result.
        
        Lookup order: Memory -> Redis -> SQLite -> None
        
        Args:
            cache_key: Cache key for the scenario
            
        Returns:
            Cached result or None if not found
        """
        self.stats.total_requests += 1
        
        # Try memory cache first
        result = self._memory_cache.get(cache_key)
        if result is not None:
            self.stats.cache_hits += 1
            self._update_hit_rate()
            return result
        
        # Try Redis if available
        if self._redis_client:
            try:
                data = self._redis_client.get(f"poker_knight:hand:{cache_key}")
                if data:
                    result = pickle.loads(data)
                    # Promote to memory cache
                    self._memory_cache.put(cache_key, result)
                    self.stats.cache_hits += 1
                    self.stats.persistence_loads += 1
                    self._update_hit_rate()
                    return result
            except Exception:
                pass  # Fail gracefully
        
        # Try SQLite if available
        if self._sqlite_cache:
            try:
                result = self._sqlite_cache.get(cache_key)
                if result:
                    # Promote to memory cache
                    self._memory_cache.put(cache_key, result)
                    self.stats.cache_hits += 1
                    self.stats.persistence_loads += 1
                    self._update_hit_rate()
                    return result
            except Exception:
                pass  # Fail gracefully
        
        # Cache miss
        self.stats.cache_misses += 1
        self._update_hit_rate()
        return None
    
    def store_result(self, cache_key: str, result: Dict[str, Any]) -> bool:
        """
        Store simulation result in cache.
        
        Storage order: Memory + (Redis OR SQLite)
        
        Args:
            cache_key: Cache key for the scenario
            result: Simulation result to cache
            
        Returns:
            True if successfully stored
        """
        success = False
        
        # Store in memory cache
        if self._memory_cache.put(cache_key, result):
            success = True
        
        # Store in persistent storage if enabled
        if self._redis_client:
            try:
                data = pickle.dumps(result)
                self._redis_client.set(
                    f"poker_knight:hand:{cache_key}", 
                    data,
                    ex=86400  # 24 hour expiration
                )
                self.stats.persistence_saves += 1
            except Exception:
                pass  # Fail gracefully
        elif self._sqlite_cache:
            try:
                self._sqlite_cache.set(cache_key, result, expiration_hours=24)
                self.stats.persistence_saves += 1
            except Exception:
                pass  # Fail gracefully
        
        # Periodic cleanup
        if time.time() - self._last_cleanup > self.config.cache_cleanup_interval:
            self._cleanup()
        
        return success
    
    def _update_hit_rate(self) -> None:
        """Update cache hit rate statistics."""
        if self.stats.total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
    
    def _cleanup(self) -> None:
        """Perform periodic cache cleanup."""
        self._last_cleanup = time.time()
        self.stats.last_cleanup = self._last_cleanup
        
        # Update memory usage stats
        cache_stats = self._memory_cache.stats()
        self.stats.memory_usage_mb = cache_stats['memory_usage_mb']
        
        # Clean up SQLite cache if available
        if self._sqlite_cache:
            try:
                self._sqlite_cache.cleanup_expired(max_age_hours=168)  # 1 week
            except Exception:
                pass
    
    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        # Update memory stats
        cache_stats = self._memory_cache.stats()
        self.stats.memory_usage_mb = cache_stats['memory_usage_mb']
        return self.stats
    
    def get_persistence_stats(self) -> Dict[str, Any]:
        """Get detailed persistence statistics."""
        stats = {
            'persistence_type': self.stats.persistence_type,
            'redis_available': REDIS_AVAILABLE,
            'redis_connected': self._redis_client is not None,
            'sqlite_available': self._sqlite_cache is not None
        }
        
        if self._sqlite_cache:
            stats['sqlite_stats'] = self._sqlite_cache.get_stats()
        
        return stats
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._memory_cache.clear()
        
        if self._redis_client:
            try:
                # Clear all poker knight keys
                keys = self._redis_client.keys("poker_knight:hand:*")
                if keys:
                    self._redis_client.delete(*keys)
            except Exception:
                pass
        
        if self._sqlite_cache:
            try:
                self._sqlite_cache.clear()
            except Exception:
                pass


class BoardTextureCache:
    """
    Cache for board texture analysis and common patterns.
    
    Implements Task 1.3.2: Board texture memoization for common patterns
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        
        # Initialize cache for board texture analysis
        memory_per_cache = self.config.max_memory_mb // 3
        self._texture_cache = ThreadSafeLRUCache(
            max_size=self.config.board_texture_cache_size,
            max_memory_mb=memory_per_cache
        )
        
        # Pre-populate with common board textures
        self._populate_common_textures()
    
    def _populate_common_textures(self) -> None:
        """Pre-populate cache with analysis of common board textures."""
        # This will be populated during first analysis runs
        # For now, we'll compute on-demand and cache results
        pass
    
    def get_texture_analysis(self, board_cards: List[str]) -> Optional[Dict[str, Any]]:
        """Get cached board texture analysis."""
        if not board_cards or len(board_cards) < 3:
            return None
        
        # Create texture key (order matters less for texture analysis)
        texture_key = "|".join(sorted(board_cards))
        return self._texture_cache.get(texture_key)
    
    def store_texture_analysis(self, board_cards: List[str], analysis: Dict[str, Any]) -> bool:
        """Store board texture analysis."""
        if not board_cards or len(board_cards) < 3:
            return False
        
        texture_key = "|".join(sorted(board_cards))
        return self._texture_cache.put(texture_key, analysis)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get texture cache statistics."""
        return self._texture_cache.stats()


class PreflopRangeCache:
    """
    Cache for preflop hand combinations and range analysis.
    
    Implements Task 1.3.3: Preflop range cache (169 hand combinations)
    """
    
    # All 169 possible preflop hand combinations
    PREFLOP_HANDS = []
    
    @classmethod
    def _generate_preflop_hands(cls) -> None:
        """Generate all 169 unique preflop hand combinations."""
        if cls.PREFLOP_HANDS:
            return
        
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        hands = []
        
        # Pocket pairs
        for rank in ranks:
            hands.append(f"{rank}{rank}")
        
        # Suited hands
        for i, rank1 in enumerate(ranks):
            for rank2 in ranks[i+1:]:
                hands.append(f"{rank1}{rank2}s")
        
        # Offsuit hands  
        for i, rank1 in enumerate(ranks):
            for rank2 in ranks[i+1:]:
                hands.append(f"{rank1}{rank2}o")
        
        cls.PREFLOP_HANDS = hands
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._generate_preflop_hands()
        
        # Initialize preflop cache with full capacity for all 169 hands
        # across different opponent counts and positions
        max_entries = len(self.PREFLOP_HANDS) * 6 * 6  # 169 hands * 6 opponents * 6 positions
        memory_per_cache = self.config.max_memory_mb // 3
        
        self._preflop_cache = ThreadSafeLRUCache(
            max_size=max_entries,
            max_memory_mb=memory_per_cache
        )
        
        self._cache_coverage = {}  # Track which combinations are cached
    
    def get_preflop_result(self, hero_hand: List[str], num_opponents: int, 
                          position: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get cached preflop analysis result."""
        hand_key = self._normalize_preflop_hand(hero_hand)
        if not hand_key:
            return None
        
        cache_key = f"{hand_key}_{num_opponents}"
        if position:
            cache_key += f"_{position}"
        
        return self._preflop_cache.get(cache_key)
    
    def store_preflop_result(self, hero_hand: List[str], num_opponents: int,
                           result: Dict[str, Any], position: Optional[str] = None) -> bool:
        """Store preflop analysis result."""
        hand_key = self._normalize_preflop_hand(hero_hand)
        if not hand_key:
            return False
        
        cache_key = f"{hand_key}_{num_opponents}"
        if position:
            cache_key += f"_{position}"
        
        success = self._preflop_cache.put(cache_key, result)
        if success:
            self._cache_coverage[cache_key] = time.time()
        
        return success
    
    def _normalize_preflop_hand(self, hero_hand: List[str]) -> Optional[str]:
        """Normalize preflop hand to standard notation (e.g., AKs, QQ, 72o)."""
        if len(hero_hand) != 2:
            return None
        
        # Extract ranks and suits
        try:
            card1, card2 = hero_hand
            # Handle 10 specially since it's two characters
            if card1.startswith('10'):
                rank1 = '10'
                suit1 = card1[2:]
            else:
                rank1 = card1[0]
                suit1 = card1[1:]
            
            if card2.startswith('10'):
                rank2 = '10'
                suit2 = card2[2:]
            else:
                rank2 = card2[0]
                suit2 = card2[1:]
        except:
            return None
        
        # Normalize rank notation (10 -> T for compatibility with precomputed hands)
        rank1 = 'T' if rank1 == '10' else rank1
        rank2 = 'T' if rank2 == '10' else rank2
        
        # Handle pocket pairs
        if rank1 == rank2:
            return f"{rank1}{rank2}"
        
        # Order ranks by strength (A > K > Q > ... > 2)
        rank_order = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        if rank_order.index(rank1) > rank_order.index(rank2):
            rank1, rank2 = rank2, rank1
            suit1, suit2 = suit2, suit1
        
        # Determine if suited or offsuit
        suited = 's' if suit1 == suit2 else 'o'
        return f"{rank1}{rank2}{suited}"
    
    def get_cache_coverage(self) -> Dict[str, Any]:
        """Get statistics on preflop cache coverage."""
        total_combinations = len(self.PREFLOP_HANDS) * 6  # 6 opponent counts
        cached_combinations = len(self._cache_coverage)
        
        return {
            'total_possible_combinations': total_combinations,
            'cached_combinations': cached_combinations,
            'coverage_percentage': (cached_combinations / total_combinations) * 100,
            'cache_stats': self._preflop_cache.stats()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preflop cache statistics."""
        return self._preflop_cache.stats()


class CachingSimulationResult:
    """Wrapper for SimulationResult with caching metadata."""
    
    def __init__(self, result: Any, cached: bool = False, cache_key: str = ""):
        self.result = result
        self.cached = cached
        self.cache_key = cache_key
        self.cache_timestamp = time.time() if cached else None
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying result."""
        return getattr(self.result, name)


# Singleton cache manager for global access
_cache_manager = None
_cache_lock = threading.Lock()


def get_cache_manager(config: Optional[CacheConfig] = None) -> Tuple[HandCache, BoardTextureCache, PreflopRangeCache]:
    """Get global cache manager instances."""
    global _cache_manager, _cache_lock
    
    with _cache_lock:
        if _cache_manager is None:
            _cache_manager = (
                HandCache(config),
                BoardTextureCache(config),
                PreflopRangeCache(config)
            )
        return _cache_manager


def clear_all_caches() -> None:
    """Clear all cache instances."""
    global _cache_manager, _cache_lock
    
    with _cache_lock:
        if _cache_manager:
            hand_cache, board_cache, preflop_cache = _cache_manager
            hand_cache.clear()
            board_cache._texture_cache.clear()
            preflop_cache._preflop_cache.clear()


# Performance monitoring decorator
def cache_performance_monitor(func):
    """Decorator to monitor cache performance."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Log performance metrics if needed
        if hasattr(result, 'cached') and result.cached and execution_time < 10:
            # Successfully met sub-10ms target for cached results
            pass
        
        return result
    return wrapper 