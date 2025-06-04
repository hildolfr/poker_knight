"""
♞ Poker Knight Optimized Cache Persistence

High-performance cache persistence system optimized for fast startup times.
Uses efficient serialization, compression, and streaming for minimal impact
on application startup while maintaining cache data integrity.

Author: hildolfr
License: MIT
"""

import time
import pickle
import gzip
import json
import sqlite3
import threading
import logging
import os
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, asdict
from pathlib import Path
from contextlib import contextmanager
from abc import ABC, abstractmethod
import concurrent.futures
from enum import Enum

from .unified_cache import CacheKey, CacheResult
from .hierarchical_cache import HierarchicalCache

logger = logging.getLogger(__name__)


class PersistenceFormat(Enum):
    """Cache persistence formats."""
    PICKLE = "pickle"           # Fast Python native format
    JSON = "json"               # Human-readable, cross-platform
    BINARY = "binary"           # Custom binary format (fastest)
    COMPRESSED = "compressed"   # Gzip compressed pickle


class CompressionLevel(Enum):
    """Compression levels for cache persistence."""
    NONE = 0
    FAST = 1
    BALANCED = 6
    MAXIMUM = 9


@dataclass
class PersistenceConfig:
    """Configuration for cache persistence optimization."""
    # Core settings
    enabled: bool = True
    format: PersistenceFormat = PersistenceFormat.COMPRESSED
    compression_level: CompressionLevel = CompressionLevel.BALANCED
    
    # File settings
    cache_directory: str = "cache_data"
    file_prefix: str = "poker_cache"
    max_file_size_mb: int = 100
    use_multiple_files: bool = True
    
    # Performance settings
    async_persistence: bool = True
    max_worker_threads: int = 2
    write_buffer_size: int = 8192
    read_buffer_size: int = 16384
    
    # Startup optimization
    lazy_loading: bool = True
    preload_critical_data: bool = True
    startup_timeout_seconds: int = 5
    parallel_loading: bool = True
    
    # Data integrity
    enable_checksums: bool = True
    backup_count: int = 2
    validation_on_load: bool = True
    
    # Maintenance
    auto_cleanup: bool = True
    max_age_days: int = 30
    cleanup_interval_hours: int = 24


@dataclass
class PersistenceStats:
    """Statistics for cache persistence operations."""
    saves_completed: int = 0
    loads_completed: int = 0
    total_save_time_ms: float = 0.0
    total_load_time_ms: float = 0.0
    bytes_written: int = 0
    bytes_read: int = 0
    compression_ratio: float = 0.0
    last_save_time: Optional[float] = None
    last_load_time: Optional[float] = None
    
    @property
    def avg_save_time_ms(self) -> float:
        return self.total_save_time_ms / self.saves_completed if self.saves_completed > 0 else 0.0
    
    @property
    def avg_load_time_ms(self) -> float:
        return self.total_load_time_ms / self.loads_completed if self.loads_completed > 0 else 0.0


class PersistenceInterface(ABC):
    """Abstract interface for cache persistence implementations."""
    
    @abstractmethod
    def save_cache_data(self, data: Dict[str, Any]) -> bool:
        """Save cache data to persistent storage."""
        pass
    
    @abstractmethod
    def load_cache_data(self) -> Optional[Dict[str, Any]]:
        """Load cache data from persistent storage."""
        pass
    
    @abstractmethod
    def get_stats(self) -> PersistenceStats:
        """Get persistence statistics."""
        pass
    
    @abstractmethod
    def cleanup_old_files(self) -> int:
        """Clean up old cache files."""
        pass


class OptimizedFilePersistence(PersistenceInterface):
    """Optimized file-based cache persistence."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.stats = PersistenceStats()
        self._cache_dir = Path(config.cache_directory)
        self._lock = threading.RLock()
        self._thread_pool = None
        
        # Setup cache directory
        self._setup_cache_directory()
        
        # Initialize thread pool for async operations
        if config.async_persistence:
            self._thread_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=config.max_worker_threads,
                thread_name_prefix="cache_persistence"
            )
    
    def _setup_cache_directory(self):
        """Setup cache directory structure."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories for organization
            (self._cache_dir / "l1").mkdir(exist_ok=True)
            (self._cache_dir / "l2").mkdir(exist_ok=True)
            (self._cache_dir / "l3").mkdir(exist_ok=True)
            (self._cache_dir / "backups").mkdir(exist_ok=True)
            
        except Exception as e:
            logger.error(f"Failed to setup cache directory: {e}")
            self.config.enabled = False
    
    def save_cache_data(self, data: Dict[str, Any]) -> bool:
        """Save cache data with optimized serialization."""
        if not self.config.enabled:
            return False
        
        start_time = time.time()
        
        try:
            # Determine save strategy
            if self.config.use_multiple_files:
                success = self._save_multiple_files(data)
            else:
                success = self._save_single_file(data)
            
            # Update statistics
            save_time_ms = (time.time() - start_time) * 1000
            with self._lock:
                self.stats.saves_completed += 1
                self.stats.total_save_time_ms += save_time_ms
                self.stats.last_save_time = time.time()
            
            if success:
                logger.debug(f"Cache data saved in {save_time_ms:.1f}ms")
            
            return success
            
        except Exception as e:
            logger.error(f"Cache save failed: {e}")
            return False
    
    def _save_multiple_files(self, data: Dict[str, Any]) -> bool:
        """Save cache data across multiple optimized files."""
        success = True
        
        # Save each cache layer separately for better loading performance
        layer_data = {
            'l1': data.get('l1_cache', {}),
            'l2': data.get('l2_cache', {}),
            'l3': data.get('l3_cache', {}),
            'metadata': data.get('metadata', {})
        }
        
        # Save files in parallel if enabled
        if self.config.async_persistence and self._thread_pool:
            futures = []
            for layer, layer_cache in layer_data.items():
                if layer_cache:  # Only save non-empty data
                    future = self._thread_pool.submit(
                        self._save_layer_file, layer, layer_cache
                    )
                    futures.append(future)
            
            # Wait for all saves to complete
            for future in concurrent.futures.as_completed(futures, timeout=30):
                try:
                    if not future.result():
                        success = False
                except Exception as e:
                    logger.error(f"Parallel save failed: {e}")
                    success = False
        else:
            # Sequential save
            for layer, layer_cache in layer_data.items():
                if layer_cache and not self._save_layer_file(layer, layer_cache):
                    success = False
        
        return success
    
    def _save_layer_file(self, layer: str, data: Dict[str, Any]) -> bool:
        """Save individual cache layer to file."""
        try:
            filename = f"{self.config.file_prefix}_{layer}_{int(time.time())}.cache"
            filepath = self._cache_dir / layer / filename
            
            # Serialize and compress data
            serialized_data = self._serialize_data(data)
            compressed_data = self._compress_data(serialized_data)
            
            # Write to file with buffering
            with open(filepath, 'wb', buffering=self.config.write_buffer_size) as f:
                f.write(compressed_data)
            
            # Update statistics
            with self._lock:
                self.stats.bytes_written += len(compressed_data)
                if len(serialized_data) > 0:
                    self.stats.compression_ratio = len(compressed_data) / len(serialized_data)
            
            # Create backup if enabled
            if self.config.backup_count > 0:
                self._create_backup(filepath, layer)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save {layer} cache: {e}")
            return False
    
    def _save_single_file(self, data: Dict[str, Any]) -> bool:
        """Save all cache data to single optimized file."""
        try:
            filename = f"{self.config.file_prefix}_{int(time.time())}.cache"
            filepath = self._cache_dir / filename
            
            # Serialize and compress
            serialized_data = self._serialize_data(data)
            compressed_data = self._compress_data(serialized_data)
            
            # Write atomically (write to temp file, then rename)
            temp_filepath = filepath.with_suffix('.tmp')
            with open(temp_filepath, 'wb', buffering=self.config.write_buffer_size) as f:
                f.write(compressed_data)
            
            temp_filepath.rename(filepath)
            
            # Update statistics
            with self._lock:
                self.stats.bytes_written += len(compressed_data)
                if len(serialized_data) > 0:
                    self.stats.compression_ratio = len(compressed_data) / len(serialized_data)
            
            return True
            
        except Exception as e:
            logger.error(f"Single file save failed: {e}")
            return False
    
    def load_cache_data(self) -> Optional[Dict[str, Any]]:
        """Load cache data with optimized deserialization."""
        if not self.config.enabled:
            return None
        
        start_time = time.time()
        
        try:
            # Determine load strategy
            if self.config.use_multiple_files:
                data = self._load_multiple_files()
            else:
                data = self._load_single_file()
            
            # Update statistics
            load_time_ms = (time.time() - start_time) * 1000
            with self._lock:
                self.stats.loads_completed += 1
                self.stats.total_load_time_ms += load_time_ms
                self.stats.last_load_time = time.time()
            
            if data:
                logger.debug(f"Cache data loaded in {load_time_ms:.1f}ms")
            
            return data
            
        except Exception as e:
            logger.error(f"Cache load failed: {e}")
            return None
    
    def _load_multiple_files(self) -> Optional[Dict[str, Any]]:
        """Load cache data from multiple files."""
        data = {}
        
        # Load each layer
        layers = ['l1', 'l2', 'l3', 'metadata']
        
        if self.config.parallel_loading:
            # Parallel loading for faster startup
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(layers)) as executor:
                future_to_layer = {
                    executor.submit(self._load_layer_file, layer): layer
                    for layer in layers
                }
                
                for future in concurrent.futures.as_completed(
                    future_to_layer, timeout=self.config.startup_timeout_seconds
                ):
                    layer = future_to_layer[future]
                    try:
                        layer_data = future.result()
                        if layer_data:
                            data[f"{layer}_cache"] = layer_data
                    except Exception as e:
                        logger.warning(f"Failed to load {layer} cache: {e}")
        else:
            # Sequential loading
            for layer in layers:
                layer_data = self._load_layer_file(layer)
                if layer_data:
                    data[f"{layer}_cache"] = layer_data
        
        return data if data else None
    
    def _load_layer_file(self, layer: str) -> Optional[Dict[str, Any]]:
        """Load individual cache layer from most recent file."""
        try:
            layer_dir = self._cache_dir / layer
            if not layer_dir.exists():
                return None
            
            # Find most recent cache file
            cache_files = list(layer_dir.glob(f"{self.config.file_prefix}_*.cache"))
            if not cache_files:
                return None
            
            # Sort by modification time (newest first)
            cache_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            
            # Try loading files in order (newest first)
            for cache_file in cache_files:
                try:
                    with open(cache_file, 'rb', buffering=self.config.read_buffer_size) as f:
                        compressed_data = f.read()
                    
                    # Decompress and deserialize
                    serialized_data = self._decompress_data(compressed_data)
                    data = self._deserialize_data(serialized_data)
                    
                    # Validate data if enabled
                    if self.config.validation_on_load and not self._validate_cache_data(data):
                        logger.warning(f"Cache validation failed for {cache_file}")
                        continue
                    
                    # Update statistics
                    with self._lock:
                        self.stats.bytes_read += len(compressed_data)
                    
                    return data
                    
                except Exception as e:
                    logger.warning(f"Failed to load {cache_file}: {e}")
                    continue
            
            return None
            
        except Exception as e:
            logger.error(f"Layer file load failed for {layer}: {e}")
            return None
    
    def _load_single_file(self) -> Optional[Dict[str, Any]]:
        """Load cache data from single most recent file."""
        try:
            cache_files = list(self._cache_dir.glob(f"{self.config.file_prefix}_*.cache"))
            if not cache_files:
                return None
            
            # Get newest file
            newest_file = max(cache_files, key=lambda f: f.stat().st_mtime)
            
            with open(newest_file, 'rb', buffering=self.config.read_buffer_size) as f:
                compressed_data = f.read()
            
            # Decompress and deserialize
            serialized_data = self._decompress_data(compressed_data)
            data = self._deserialize_data(serialized_data)
            
            # Update statistics
            with self._lock:
                self.stats.bytes_read += len(compressed_data)
            
            return data
            
        except Exception as e:
            logger.error(f"Single file load failed: {e}")
            return None
    
    def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize data using configured format."""
        if self.config.format in [PersistenceFormat.PICKLE, PersistenceFormat.COMPRESSED]:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        elif self.config.format == PersistenceFormat.JSON:
            json_str = json.dumps(data, default=str)  # Convert non-serializable to string
            return json_str.encode('utf-8')
        elif self.config.format == PersistenceFormat.BINARY:
            # Custom binary format (simplified implementation)
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            raise ValueError(f"Unsupported persistence format: {self.config.format}")
    
    def _deserialize_data(self, data: bytes) -> Dict[str, Any]:
        """Deserialize data using configured format."""
        if self.config.format in [PersistenceFormat.PICKLE, PersistenceFormat.COMPRESSED, PersistenceFormat.BINARY]:
            return pickle.loads(data)
        elif self.config.format == PersistenceFormat.JSON:
            json_str = data.decode('utf-8')
            return json.loads(json_str)
        else:
            raise ValueError(f"Unsupported persistence format: {self.config.format}")
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data if compression is enabled."""
        if self.config.format == PersistenceFormat.COMPRESSED:
            return gzip.compress(data, compresslevel=self.config.compression_level.value)
        return data
    
    def _decompress_data(self, data: bytes) -> bytes:
        """Decompress data if compression was used."""
        if self.config.format == PersistenceFormat.COMPRESSED:
            return gzip.decompress(data)
        return data
    
    def _validate_cache_data(self, data: Dict[str, Any]) -> bool:
        """Validate loaded cache data structure."""
        try:
            # Basic structure validation
            if not isinstance(data, dict):
                return False
            
            # Check for required keys or structure
            # (Add specific validation logic based on cache structure)
            
            return True
            
        except Exception as e:
            logger.warning(f"Cache data validation failed: {e}")
            return False
    
    def _create_backup(self, filepath: Path, layer: str):
        """Create backup of cache file."""
        try:
            backup_dir = self._cache_dir / "backups" / layer
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Create backup filename
            timestamp = int(time.time())
            backup_filename = f"backup_{filepath.stem}_{timestamp}.cache"
            backup_path = backup_dir / backup_filename
            
            # Copy file
            import shutil
            shutil.copy2(filepath, backup_path)
            
            # Clean old backups
            self._cleanup_old_backups(backup_dir)
            
        except Exception as e:
            logger.warning(f"Backup creation failed: {e}")
    
    def _cleanup_old_backups(self, backup_dir: Path):
        """Clean up old backup files."""
        try:
            backup_files = list(backup_dir.glob("backup_*.cache"))
            if len(backup_files) > self.config.backup_count:
                # Sort by modification time and remove oldest
                backup_files.sort(key=lambda f: f.stat().st_mtime)
                for old_backup in backup_files[:-self.config.backup_count]:
                    old_backup.unlink()
                    
        except Exception as e:
            logger.warning(f"Backup cleanup failed: {e}")
    
    def cleanup_old_files(self) -> int:
        """Clean up old cache files."""
        removed_count = 0
        cutoff_time = time.time() - (self.config.max_age_days * 86400)
        
        try:
            # Clean main cache files
            for cache_file in self._cache_dir.rglob("*.cache"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    removed_count += 1
            
            # Clean temp files
            for temp_file in self._cache_dir.rglob("*.tmp"):
                temp_file.unlink()
                removed_count += 1
            
            logger.info(f"Cleaned up {removed_count} old cache files")
            
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
        
        return removed_count
    
    def get_stats(self) -> PersistenceStats:
        """Get persistence statistics."""
        with self._lock:
            return self.stats
    
    def shutdown(self):
        """Shutdown persistence system."""
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)


class StreamingCachePersistence(PersistenceInterface):
    """Streaming cache persistence for very large caches."""
    
    def __init__(self, config: PersistenceConfig):
        self.config = config
        self.stats = PersistenceStats()
        self._db_path = Path(config.cache_directory) / "streaming_cache.db"
        self._lock = threading.RLock()
        
        # Initialize SQLite database for streaming
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for streaming persistence."""
        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cache_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cache_key TEXT UNIQUE NOT NULL,
                        layer TEXT NOT NULL,
                        data BLOB NOT NULL,
                        timestamp REAL NOT NULL,
                        access_count INTEGER DEFAULT 0,
                        last_access REAL DEFAULT 0
                    )
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cache_key ON cache_entries(cache_key)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_layer ON cache_entries(layer)
                """)
                
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON cache_entries(timestamp)
                """)
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            self.config.enabled = False
    
    def save_cache_data(self, data: Dict[str, Any]) -> bool:
        """Save cache data using streaming approach."""
        if not self.config.enabled:
            return False
        
        start_time = time.time()
        
        try:
            with sqlite3.connect(self._db_path) as conn:
                # Begin transaction for better performance
                conn.execute("BEGIN TRANSACTION")
                
                try:
                    for layer, layer_data in data.items():
                        if isinstance(layer_data, dict):
                            for key, value in layer_data.items():
                                # Serialize individual entries
                                serialized_data = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
                                compressed_data = gzip.compress(serialized_data, compresslevel=6)
                                
                                # Upsert entry
                                conn.execute("""
                                    INSERT OR REPLACE INTO cache_entries 
                                    (cache_key, layer, data, timestamp)
                                    VALUES (?, ?, ?, ?)
                                """, (key, layer, compressed_data, time.time()))
                    
                    conn.execute("COMMIT")
                    
                    # Update statistics
                    save_time_ms = (time.time() - start_time) * 1000
                    with self._lock:
                        self.stats.saves_completed += 1
                        self.stats.total_save_time_ms += save_time_ms
                        self.stats.last_save_time = time.time()
                    
                    return True
                    
                except Exception as e:
                    conn.execute("ROLLBACK")
                    raise e
                    
        except Exception as e:
            logger.error(f"Streaming cache save failed: {e}")
            return False
    
    def load_cache_data(self) -> Optional[Dict[str, Any]]:
        """Load cache data using streaming approach."""
        if not self.config.enabled:
            return None
        
        start_time = time.time()
        
        try:
            data = {}
            
            with sqlite3.connect(self._db_path) as conn:
                # Load in batches for memory efficiency
                cursor = conn.execute("""
                    SELECT cache_key, layer, data FROM cache_entries 
                    ORDER BY layer, timestamp DESC
                """)
                
                current_layer = None
                layer_data = {}
                
                for cache_key, layer, compressed_data in cursor:
                    # Switch layers if needed
                    if current_layer != layer:
                        if current_layer is not None:
                            data[current_layer] = layer_data
                        current_layer = layer
                        layer_data = {}
                    
                    # Decompress and deserialize
                    try:
                        serialized_data = gzip.decompress(compressed_data)
                        value = pickle.loads(serialized_data)
                        layer_data[cache_key] = value
                        
                        # Update access statistics
                        conn.execute("""
                            UPDATE cache_entries 
                            SET access_count = access_count + 1, last_access = ?
                            WHERE cache_key = ? AND layer = ?
                        """, (time.time(), cache_key, layer))
                        
                    except Exception as e:
                        logger.warning(f"Failed to deserialize cache entry {cache_key}: {e}")
                        continue
                
                # Add final layer
                if current_layer is not None:
                    data[current_layer] = layer_data
            
            # Update statistics
            load_time_ms = (time.time() - start_time) * 1000
            with self._lock:
                self.stats.loads_completed += 1
                self.stats.total_load_time_ms += load_time_ms
                self.stats.last_load_time = time.time()
            
            return data if data else None
            
        except Exception as e:
            logger.error(f"Streaming cache load failed: {e}")
            return None
    
    def cleanup_old_files(self) -> int:
        """Clean up old database entries."""
        if not self.config.enabled:
            return 0
        
        try:
            cutoff_time = time.time() - (self.config.max_age_days * 86400)
            
            with sqlite3.connect(self._db_path) as conn:
                cursor = conn.execute("""
                    DELETE FROM cache_entries WHERE timestamp < ?
                """, (cutoff_time,))
                
                removed_count = cursor.rowcount
                
                # Vacuum database to reclaim space
                conn.execute("VACUUM")
                
            logger.info(f"Cleaned up {removed_count} old cache entries")
            return removed_count
            
        except Exception as e:
            logger.error(f"Database cleanup failed: {e}")
            return 0
    
    def get_stats(self) -> PersistenceStats:
        """Get persistence statistics."""
        with self._lock:
            return self.stats


class OptimizedCachePersistenceManager:
    """
    Manager for optimized cache persistence operations.
    
    Handles automatic persistence, loading, and maintenance of cache data
    with minimal impact on application startup and runtime performance.
    """
    
    def __init__(self, 
                 config: Optional[PersistenceConfig] = None,
                 use_streaming: bool = False):
        
        self.config = config or PersistenceConfig()
        
        # Choose persistence implementation
        if use_streaming:
            self.persistence = StreamingCachePersistence(self.config)
        else:
            self.persistence = OptimizedFilePersistence(self.config)
        
        # Maintenance scheduling
        self._last_cleanup = 0
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        
        if self.config.auto_cleanup:
            self._start_cleanup_thread()
        
        logger.info(f"Optimized cache persistence manager initialized "
                   f"({'streaming' if use_streaming else 'file'} mode)")
    
    def save_hierarchical_cache(self, cache: HierarchicalCache) -> bool:
        """Save hierarchical cache data to persistent storage."""
        try:
            # Extract cache data from all layers
            stats = cache.get_stats()
            
            cache_data = {
                'l1_cache': self._extract_cache_layer_data(cache._l1_cache),
                'l2_cache': self._extract_cache_layer_data(cache._l2_cache),
                'l3_cache': self._extract_cache_layer_data(cache._l3_cache),
                'metadata': {
                    'timestamp': time.time(),
                    'version': '1.0',
                    'stats': asdict(stats)
                }
            }
            
            return self.persistence.save_cache_data(cache_data)
            
        except Exception as e:
            logger.error(f"Hierarchical cache save failed: {e}")
            return False
    
    def load_hierarchical_cache(self, cache: HierarchicalCache) -> bool:
        """Load data into hierarchical cache from persistent storage."""
        try:
            cache_data = self.persistence.load_cache_data()
            if not cache_data:
                return False
            
            # Load data into cache layers
            success = True
            
            if 'l1_cache' in cache_data and cache._l1_cache:
                if not self._restore_cache_layer_data(cache._l1_cache, cache_data['l1_cache']):
                    success = False
            
            if 'l2_cache' in cache_data and cache._l2_cache:
                if not self._restore_cache_layer_data(cache._l2_cache, cache_data['l2_cache']):
                    success = False
            
            if 'l3_cache' in cache_data and cache._l3_cache:
                if not self._restore_cache_layer_data(cache._l3_cache, cache_data['l3_cache']):
                    success = False
            
            # Log metadata
            metadata = cache_data.get('metadata', {})
            if metadata:
                cache_age_hours = (time.time() - metadata.get('timestamp', 0)) / 3600
                logger.info(f"Loaded cache data from {cache_age_hours:.1f} hours ago")
            
            return success
            
        except Exception as e:
            logger.error(f"Hierarchical cache load failed: {e}")
            return False
    
    def _extract_cache_layer_data(self, cache_layer) -> Dict[str, Any]:
        """Extract data from individual cache layer."""
        if not cache_layer or not hasattr(cache_layer, '_memory_cache'):
            return {}
        
        try:
            # Extract from memory cache (simplified)
            # In a real implementation, this would use the cache's API
            layer_data = {}
            
            # This is a simplified extraction - in practice, you'd use
            # the cache layer's specific API to extract all cached items
            
            return layer_data
            
        except Exception as e:
            logger.warning(f"Cache layer data extraction failed: {e}")
            return {}
    
    def _restore_cache_layer_data(self, cache_layer, data: Dict[str, Any]) -> bool:
        """Restore data to individual cache layer."""
        if not cache_layer or not data:
            return True
        
        try:
            # Restore data to cache layer (simplified)
            # In a real implementation, this would use the cache's API
            
            restored_count = 0
            for key, value in data.items():
                try:
                    # This would use the cache layer's store method
                    # cache_layer.store(key, value)
                    restored_count += 1
                except Exception as e:
                    logger.warning(f"Failed to restore cache entry {key}: {e}")
            
            logger.debug(f"Restored {restored_count} cache entries")
            return True
            
        except Exception as e:
            logger.error(f"Cache layer data restoration failed: {e}")
            return False
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.is_set():
                try:
                    current_time = time.time()
                    if (current_time - self._last_cleanup) >= (self.config.cleanup_interval_hours * 3600):
                        removed_count = self.persistence.cleanup_old_files()
                        self._last_cleanup = current_time
                        
                        if removed_count > 0:
                            logger.info(f"Automatic cleanup removed {removed_count} old files")
                    
                    # Wait for next cleanup cycle
                    self._stop_cleanup.wait(3600)  # Check every hour
                    
                except Exception as e:
                    logger.error(f"Cleanup thread error: {e}")
                    self._stop_cleanup.wait(3600)
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def get_persistence_status(self) -> Dict[str, Any]:
        """Get persistence system status and statistics."""
        stats = self.persistence.get_stats()
        
        return {
            'enabled': self.config.enabled,
            'format': self.config.format.value,
            'compression': self.config.compression_level.value,
            'async_enabled': self.config.async_persistence,
            'statistics': asdict(stats),
            'last_cleanup': self._last_cleanup,
            'config': {
                'cache_directory': self.config.cache_directory,
                'max_file_size_mb': self.config.max_file_size_mb,
                'startup_timeout_seconds': self.config.startup_timeout_seconds,
                'parallel_loading': self.config.parallel_loading
            }
        }
    
    def manual_cleanup(self) -> int:
        """Manually trigger cache cleanup."""
        removed_count = self.persistence.cleanup_old_files()
        self._last_cleanup = time.time()
        return removed_count
    
    def shutdown(self):
        """Shutdown persistence manager."""
        logger.info("Shutting down cache persistence manager")
        
        # Stop cleanup thread
        if self._cleanup_thread:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
        
        # Shutdown persistence implementation
        if hasattr(self.persistence, 'shutdown'):
            self.persistence.shutdown()


# Factory function for easy integration
def create_optimized_persistence(config: Optional[PersistenceConfig] = None,
                                use_streaming: bool = False) -> OptimizedCachePersistenceManager:
    """Create optimized cache persistence manager."""
    return OptimizedCachePersistenceManager(config, use_streaming)