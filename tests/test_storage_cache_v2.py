#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test suite for Poker Knight v1.6 caching system - Current Architecture
Tests the actual cache implementation as it exists now.

This replaces test_storage_cache.py which was written for an older architecture.
"""

import unittest
import tempfile
import os
import time
import threading
from pathlib import Path

import pytest

from poker_knight.storage.cache import (
    CacheConfig, HandCache, BoardTextureCache, PreflopRangeCache,
    create_cache_key, get_cache_manager, clear_all_caches,
    ThreadSafeLRUCache, SQLiteCache, REDIS_AVAILABLE
)
from poker_knight.storage.unified_cache import (
    CacheKey, CacheResult, CacheStats, ThreadSafeMonteCarloCache,
    CacheKeyNormalizer
)


class TestCacheConfig(unittest.TestCase):
    """Test cache configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = CacheConfig()
        
        # Test actual attributes that exist
        self.assertEqual(config.max_memory_mb, 512)
        self.assertEqual(config.hand_cache_size, 10000)
        self.assertEqual(config.board_cache_size, 5000)
        self.assertFalse(config.enable_persistence)
        self.assertIsNone(config.sqlite_path)
        self.assertIsNone(config.redis_host)
        self.assertEqual(config.redis_port, 6379)
        self.assertEqual(config.redis_db, 0)
        self.assertEqual(config.redis_ttl, 86400)
        self.assertTrue(config.preflop_cache_enabled)
        self.assertTrue(config.board_cache_enabled)
        self.assertEqual(config.warmup_iterations, 10000)
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = CacheConfig(
            max_memory_mb=1024,
            hand_cache_size=20000,
            enable_persistence=True,
            sqlite_path="/tmp/test.db"
        )
        
        self.assertEqual(config.max_memory_mb, 1024)
        self.assertEqual(config.hand_cache_size, 20000)
        self.assertTrue(config.enable_persistence)
        self.assertEqual(config.sqlite_path, "/tmp/test.db")


class TestCacheKeyNormalizer(unittest.TestCase):
    """Test cache key normalization."""
    
    def test_normalize_hand_string(self):
        """Test hand normalization from string."""
        # Test pocket pairs
        self.assertEqual(CacheKeyNormalizer.normalize_hand("AA"), "AA")
        self.assertEqual(CacheKeyNormalizer.normalize_hand("KK"), "KK")
        
        # Test suited hands - normalizer returns strings as-is
        self.assertEqual(CacheKeyNormalizer.normalize_hand("AK suited"), "AK suited")
        self.assertEqual(CacheKeyNormalizer.normalize_hand("AK_suited"), "AK_suited")
        
        # Test offsuit hands - normalizer returns strings as-is
        self.assertEqual(CacheKeyNormalizer.normalize_hand("AK offsuit"), "AK offsuit")
        self.assertEqual(CacheKeyNormalizer.normalize_hand("AK_offsuit"), "AK_offsuit")
    
    def test_normalize_hand_list(self):
        """Test hand normalization from card list."""
        # Test with unicode suits
        result = CacheKeyNormalizer.normalize_hand(['A♠', 'K♠'])
        self.assertIn(result, ['AK_suited', 'KA_suited'])  # Order may vary
        
        result = CacheKeyNormalizer.normalize_hand(['A♠', 'K♥'])
        self.assertIn(result, ['AK_offsuit', 'KA_offsuit'])
        
        # Test pocket pair
        result = CacheKeyNormalizer.normalize_hand(['A♠', 'A♥'])
        self.assertEqual(result, 'AA')
    
    def test_normalize_board(self):
        """Test board normalization."""
        # Test preflop
        self.assertEqual(CacheKeyNormalizer.normalize_board("preflop"), "preflop")
        self.assertEqual(CacheKeyNormalizer.normalize_board(""), "preflop")
        self.assertEqual(CacheKeyNormalizer.normalize_board([]), "preflop")
        
        # Test with board cards
        board = CacheKeyNormalizer.normalize_board(['A♠', 'K♠', 'Q♥'])
        self.assertEqual(len(board.split('_')), 3)


class TestCacheKey(unittest.TestCase):
    """Test CacheKey dataclass."""
    
    def test_cache_key_creation(self):
        """Test creating cache keys."""
        key = CacheKey(
            hero_hand="AK_suited",
            num_opponents=2,
            board_cards="preflop",
            simulation_mode="default"
        )
        
        self.assertEqual(key.hero_hand, "AK_suited")
        self.assertEqual(key.num_opponents, 2)
        self.assertEqual(key.board_cards, "preflop")
        self.assertEqual(key.simulation_mode, "default")
    
    def test_cache_key_to_string(self):
        """Test cache key string conversion."""
        key = CacheKey(
            hero_hand="AA",
            num_opponents=3,
            board_cards="A♠_K♠_Q♥",
            simulation_mode="precision"
        )
        
        key_str = key.to_string()
        self.assertIn("AA", key_str)
        self.assertIn("3", key_str)
        self.assertIn("precision", key_str)


class TestCacheResult(unittest.TestCase):
    """Test CacheResult dataclass."""
    
    def test_cache_result_creation(self):
        """Test creating cache results."""
        result = CacheResult(
            win_probability=0.65,
            tie_probability=0.05,
            loss_probability=0.30,
            simulations_run=100000,
            execution_time_ms=250.5
        )
        
        self.assertEqual(result.win_probability, 0.65)
        self.assertEqual(result.tie_probability, 0.05)
        self.assertEqual(result.loss_probability, 0.30)
        self.assertEqual(result.simulations_run, 100000)
        self.assertEqual(result.execution_time_ms, 250.5)
        
        # Check defaults
        self.assertIsNotNone(result.hand_categories)
        self.assertIsNotNone(result.metadata)
        self.assertIsNone(result.confidence_interval)
    
    def test_cache_result_with_metadata(self):
        """Test cache result with metadata."""
        result = CacheResult(
            win_probability=0.75,
            simulations_run=50000,
            metadata={'test': True, 'version': '1.6'},
            hand_categories={'pair': 25000, 'high_card': 25000}
        )
        
        self.assertEqual(result.metadata['test'], True)
        self.assertEqual(result.metadata['version'], '1.6')
        self.assertEqual(result.hand_categories['pair'], 25000)


class TestThreadSafeMonteCarloCache(unittest.TestCase):
    """Test the main cache implementation."""
    
    def setUp(self):
        """Set up test cache."""
        self.cache = ThreadSafeMonteCarloCache(
            max_memory_mb=10,
            max_entries=100,
            enable_persistence=False
        )
    
    def test_basic_cache_operations(self):
        """Test basic get/put operations."""
        key = CacheKey("AA", 2, "preflop", "default")
        result = CacheResult(win_probability=0.85, simulations_run=10000)
        
        # Test put
        self.cache.put(key, result)
        
        # Test get
        cached = self.cache.get(key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.win_probability, 0.85)
        self.assertEqual(cached.simulations_run, 10000)
        
        # Test miss
        miss_key = CacheKey("KK", 2, "preflop", "default")
        self.assertIsNone(self.cache.get(miss_key))
    
    def test_cache_stats(self):
        """Test cache statistics."""
        stats = self.cache.get_stats()
        self.assertIsInstance(stats, CacheStats)
        self.assertEqual(stats.total_requests, 0)
        self.assertEqual(stats.cache_hits, 0)
        self.assertEqual(stats.cache_misses, 0)
        
        # Add item and check stats
        key = CacheKey("AK", 1, "preflop", "default")
        result = CacheResult(win_probability=0.65)
        self.cache.put(key, result)
        
        # Hit
        self.cache.get(key)
        
        # Miss
        self.cache.get(CacheKey("QQ", 1, "preflop", "default"))
        
        stats = self.cache.get_stats()
        self.assertEqual(stats.total_requests, 2)
        self.assertEqual(stats.cache_hits, 1)
        self.assertEqual(stats.cache_misses, 1)
        self.assertAlmostEqual(stats.hit_rate, 0.5)
    
    def test_clear_cache(self):
        """Test clearing cache."""
        key = CacheKey("JJ", 3, "preflop", "fast")
        result = CacheResult(win_probability=0.55)
        
        self.cache.put(key, result)
        self.assertIsNotNone(self.cache.get(key))
        
        self.cache.clear()
        self.assertIsNone(self.cache.get(key))


class TestHandCache(unittest.TestCase):
    """Test HandCache legacy wrapper."""
    
    def setUp(self):
        """Set up test cache."""
        config = CacheConfig(enable_persistence=False)
        self.cache = HandCache(config)
    
    def test_string_key_interface(self):
        """Test string-based key interface."""
        # Test with test key format
        test_result = {
            'win_probability': 0.75,
            'tie_probability': 0.10,
            'loss_probability': 0.15,
            'simulations_run': 50000,
            'execution_time_ms': 100.0,
            'hand_category_frequencies': {},
            'cached': True
        }
        
        # Store using internal cache directly for test
        test_key = CacheKey("test_key", 1, "test", "test")
        test_cache_result = CacheResult(
            win_probability=0.75,
            tie_probability=0.10,
            loss_probability=0.15,
            simulations_run=50000,
            execution_time_ms=100.0
        )
        self.cache._cache.put(test_key, test_cache_result)
        
        # Retrieve using string key
        result = self.cache.get_result("test_key")
        self.assertIsNotNone(result)
        self.assertEqual(result['win_probability'], 0.75)
        self.assertEqual(result['simulations_run'], 50000)
    
    def test_store_result(self):
        """Test storing results."""
        result_data = {
            'win_probability': 0.65,
            'tie_probability': 0.05,
            'loss_probability': 0.30,
            'simulations_run': 100000,
            'execution_time_ms': 200.0
        }
        
        self.cache.store_result("QQ_3_preflop_default", result_data)
        
        # Try to retrieve
        cached = self.cache.get_result("QQ_3_preflop_default")
        self.assertIsNotNone(cached)
        self.assertEqual(cached['win_probability'], 0.65)


class TestCacheIntegration(unittest.TestCase):
    """Test cache integration functions."""
    
    def test_create_cache_key(self):
        """Test legacy cache key creation."""
        # Test with list of cards
        key = create_cache_key(['A♠', 'K♠'], 2)
        self.assertIsInstance(key, str)
        self.assertIn('suited', key.lower())
        
        # Test with board cards
        key = create_cache_key(['Q♥', 'Q♦'], 3, ['A♠', 'K♠', 'J♥'])
        self.assertIsInstance(key, str)
        self.assertIn('QQ', key)
    
    def test_get_cache_manager(self):
        """Test cache manager factory."""
        config = CacheConfig(
            preflop_cache_enabled=True,
            board_cache_enabled=True
        )
        
        hand_cache, board_cache, preflop_cache = get_cache_manager(config)
        
        self.assertIsInstance(hand_cache, HandCache)
        self.assertIsInstance(board_cache, BoardTextureCache)
        self.assertIsInstance(preflop_cache, PreflopRangeCache)
    
    def test_clear_all_caches(self):
        """Test clearing all caches."""
        # This should not raise an error
        clear_all_caches()


class TestThreadSafety(unittest.TestCase):
    """Test thread safety of cache operations."""
    
    def test_concurrent_access(self):
        """Test concurrent cache access."""
        cache = ThreadSafeMonteCarloCache(max_entries=1000)
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(10):
                    key = CacheKey(f"hand_{thread_id}_{i}", 2, "preflop", "default")
                    result = CacheResult(win_probability=float(thread_id) / 10)
                    
                    cache.put(key, result)
                    cached = cache.get(key)
                    
                    if cached and cached.win_probability == result.win_probability:
                        results.append(True)
                    else:
                        results.append(False)
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        self.assertEqual(len(errors), 0)
        self.assertEqual(len(results), 50)
        self.assertTrue(all(results))


class TestBoardTextureCache(unittest.TestCase):
    """Test BoardTextureCache functionality."""
    
    def setUp(self):
        """Set up test cache."""
        config = CacheConfig(board_cache_size=100)
        self.cache = BoardTextureCache(config)
    
    def test_board_texture_cache_creation(self):
        """Test BoardTextureCache is created properly."""
        self.assertIsNotNone(self.cache)
        self.assertIsNotNone(self.cache._cache)
        self.assertEqual(self.cache.config.board_cache_size, 100)
    
    def test_clear_board_cache(self):
        """Test clearing board texture cache."""
        # Add some data to the internal cache
        key = CacheKey("test", 1, "A♠_K♠_Q♠", "default")
        result = CacheResult(win_probability=0.5)
        self.cache._cache.put(key, result)
        
        # Clear and verify
        self.cache.clear()
        self.assertIsNone(self.cache._cache.get(key))


class TestPreflopRangeCache(unittest.TestCase):
    """Test PreflopRangeCache functionality."""
    
    def setUp(self):
        """Set up test cache."""
        config = CacheConfig()
        self.cache = PreflopRangeCache(config)
    
    def test_169_hands_generation(self):
        """Test that all 169 preflop hands are generated."""
        self.assertEqual(len(PreflopRangeCache.PREFLOP_HANDS), 169)
        
        # Check we have 13 pairs
        pairs = [h for h in PreflopRangeCache.PREFLOP_HANDS if h[0] == h[1]]
        self.assertEqual(len(pairs), 13)
        
        # Check we have suited and offsuit hands
        suited = [h for h in PreflopRangeCache.PREFLOP_HANDS if h.endswith('s')]
        offsuit = [h for h in PreflopRangeCache.PREFLOP_HANDS if h.endswith('o')]
        self.assertEqual(len(suited), 78)
        self.assertEqual(len(offsuit), 78)
    
    def test_cache_coverage(self):
        """Test cache coverage statistics."""
        coverage = self.cache.get_cache_coverage()
        
        self.assertIn('cached_combinations', coverage)
        self.assertIn('coverage_percentage', coverage)
        self.assertIn('total_requests', coverage)
        self.assertIn('hit_rate', coverage)
        
        # Initially empty
        self.assertEqual(coverage['cached_combinations'], 0)
        self.assertEqual(coverage['coverage_percentage'], 0.0)
    
    def test_preflop_result_storage(self):
        """Test storing and retrieving preflop results."""
        hero_hand = ['A♠', 'K♠']
        num_opponents = 2
        position = "button"
        
        result_data = {
            'win_probability': 0.65,
            'tie_probability': 0.05,
            'loss_probability': 0.30,
            'simulations_run': 10000,
            'execution_time_ms': 100.0,
            'hand_category_frequencies': {'flush': 1000}
        }
        
        # Store result
        stored = self.cache.store_preflop_result(hero_hand, num_opponents, result_data, position)
        self.assertTrue(stored)
        
        # Retrieve result
        cached = self.cache.get_preflop_result(hero_hand, num_opponents, position)
        self.assertIsNotNone(cached)
        self.assertEqual(cached['win_probability'], 0.65)
        self.assertEqual(cached['simulations_run'], 10000)


class TestSQLitePersistence(unittest.TestCase):
    """Test SQLite persistence functionality."""
    
    def setUp(self):
        """Set up test cache with SQLite."""
        self.temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_file.close()
        
        self.cache = ThreadSafeMonteCarloCache(
            max_entries=100,
            enable_persistence=True,
            sqlite_path=self.temp_file.name
        )
    
    def tearDown(self):
        """Clean up temp file."""
        if hasattr(self, 'cache'):
            self.cache.clear()
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_persistence_basic_operations(self):
        """Test basic SQLite persistence operations."""
        key = CacheKey("AA", 2, "preflop", "default")
        result = CacheResult(
            win_probability=0.85,
            simulations_run=50000,
            execution_time_ms=200.0
        )
        
        # Store in cache
        self.cache.put(key, result)
        
        # Create new cache instance with same SQLite file
        cache2 = ThreadSafeMonteCarloCache(
            max_entries=100,
            enable_persistence=True,
            sqlite_path=self.temp_file.name
        )
        
        # Should be able to retrieve from persistence
        cached = cache2.get(key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached.win_probability, 0.85)
    
    def test_persistence_with_memory_limit(self):
        """Test persistence when memory limit is reached."""
        # Small memory limit to force eviction
        cache = ThreadSafeMonteCarloCache(
            max_memory_mb=0.001,  # Very small
            max_entries=2,
            enable_persistence=True,
            sqlite_path=self.temp_file.name
        )
        
        # Add multiple items
        for i in range(5):
            key = CacheKey(f"hand_{i}", 2, "preflop", "default")
            result = CacheResult(win_probability=float(i) / 10)
            cache.put(key, result)
        
        # Check stats
        stats = cache.get_stats()
        self.assertGreater(stats.evictions, 0)
        self.assertEqual(stats.persistence_type, "sqlite")


class TestMemoryManagement(unittest.TestCase):
    """Test memory management and LRU eviction."""
    
    def test_lru_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = ThreadSafeMonteCarloCache(max_entries=3)
        
        # Fill cache
        for i in range(3):
            key = CacheKey(f"hand_{i}", 2, "preflop", "default")
            result = CacheResult(win_probability=float(i) / 10)
            cache.put(key, result)
        
        # Access first two items to make them more recently used
        cache.get(CacheKey("hand_0", 2, "preflop", "default"))
        cache.get(CacheKey("hand_1", 2, "preflop", "default"))
        
        # Add new item - should evict hand_2
        key = CacheKey("hand_3", 2, "preflop", "default")
        result = CacheResult(win_probability=0.99)
        cache.put(key, result)
        
        # Check hand_2 was evicted
        self.assertIsNone(cache.get(CacheKey("hand_2", 2, "preflop", "default")))
        
        # Check others still exist
        self.assertIsNotNone(cache.get(CacheKey("hand_0", 2, "preflop", "default")))
        self.assertIsNotNone(cache.get(CacheKey("hand_1", 2, "preflop", "default")))
        self.assertIsNotNone(cache.get(CacheKey("hand_3", 2, "preflop", "default")))
    
    def test_memory_usage_tracking(self):
        """Test memory usage is tracked properly."""
        cache = ThreadSafeMonteCarloCache(max_memory_mb=10)
        
        # Add items
        for i in range(10):
            key = CacheKey(f"hand_{i}", 2, "preflop", "default")
            result = CacheResult(
                win_probability=0.5,
                hand_categories={'flush': 1000, 'straight': 500},
                metadata={'test': True}
            )
            cache.put(key, result)
        
        stats = cache.get_stats()
        self.assertGreater(stats.memory_usage_mb, 0)
        self.assertEqual(stats.cache_size, 10)


@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
class TestRedisIntegration(unittest.TestCase):
    """Test Redis integration when available."""
    
    def test_redis_availability_check(self):
        """Test Redis availability is detected correctly."""
        # This test runs only if Redis is available
        self.assertTrue(REDIS_AVAILABLE)
        
        # Try creating a cache with Redis config
        config = CacheConfig(
            redis_host='localhost',
            redis_port=6379
        )
        
        # HandCache should handle Redis gracefully
        cache = HandCache(config)
        self.assertIsNotNone(cache)


if __name__ == '__main__':
    unittest.main()