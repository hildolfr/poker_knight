# Test Storage Cache Rebase Plan

## Summary of Issue

The `tests/test_storage_cache.py` file was showing as a suspicious skip rather than a proper test failure. Upon investigation, the file contains a problematic import pattern that skips the entire test module if cache imports fail:

```python
try:
    from poker_knight.storage.cache import (
        CacheConfig, CacheStats, ThreadSafeLRUCache, SQLiteCache,
        HandCache, BoardTextureCache, PreflopRangeCache,
        create_cache_key, get_cache_manager, clear_all_caches,
        CachingSimulationResult
    )
    CACHE_AVAILABLE = True
except ImportError as e:
    CACHE_AVAILABLE = False
    pytest.skip(f"Caching system not available: {e}", allow_module_level=True)
```

This pattern is inappropriate for a test file that's specifically testing the caching system - import failures should be test failures, not skips.

## Root Cause Analysis

1. **Import Failures**: The test file was trying to import classes that don't exist in the current codebase:
   - `CacheStats` exists in `unified_cache.py`, not `cache.py`
   - `CachingSimulationResult` doesn't exist at all
   - Several other classes have different locations than expected

2. **Architecture Mismatch**: The test was written for a different cache architecture than what currently exists

3. **Masking Real Issues**: The skip pattern was hiding actual import errors that should be fixed

## Current Cache Architecture

Based on codebase analysis:

### Core Implementation (`unified_cache.py`)
- `CacheKey` - Dataclass for cache keys
- `CacheResult` - Dataclass for cache results  
- `CacheStats` - Dataclass for cache statistics
- `ThreadSafeMonteCarloCache` - Main cache implementation
- `CacheKeyNormalizer` - Key normalization utilities

### Legacy Compatibility Layer (`cache.py`)
- `CacheConfig` - Configuration class
- `HandCache` - Wrapper around ThreadSafeMonteCarloCache
- `BoardTextureCache` - Simplified board texture cache
- `PreflopRangeCache` - Preflop combinations cache
- `create_cache_key()` - Legacy key creation function
- `get_cache_manager()` - Cache manager factory
- Aliases: `ThreadSafeLRUCache = ThreadSafeMonteCarloCache`
- Aliases: `SQLiteCache = ThreadSafeMonteCarloCache`

## Planned Changes

### 1. Fix Import Structure
Replace the problematic try/except skip pattern with direct imports:

```python
# Direct imports - let failures be real failures
from poker_knight.storage.cache import (
    CacheConfig, HandCache, BoardTextureCache, PreflopRangeCache,
    create_cache_key, get_cache_manager, clear_all_caches,
    ThreadSafeLRUCache, SQLiteCache, REDIS_AVAILABLE
)
from poker_knight.storage.unified_cache import (
    CacheKey, CacheResult, CacheStats, ThreadSafeMonteCarloCache,
    CacheKeyNormalizer
)
```

### 2. Update Test Classes
- Remove tests for non-existent methods
- Update class usage to match current architecture
- Focus on testing actual functionality that exists
- Simplify over-specified tests that test implementation details

### 3. Key Test Fixes Needed

#### TestCacheConfig
- Remove tests for config options that don't exist
- Focus on testing actual CacheConfig attributes

#### TestCacheKeyGeneration  
- Update to use actual `create_cache_key()` function
- Test normalization behavior that actually exists

#### TestThreadSafeMonteCarloCache
- Test the actual `ThreadSafeMonteCarloCache` class
- Use `CacheKey` and `CacheResult` dataclasses properly
- Test real cache operations (get, put, clear, stats)

#### TestHandCache
- Test legacy wrapper functionality
- Focus on string key interface (how it's actually used)
- Test persistence configuration

#### Cache Integration Tests
- Test that cache managers work together
- Test global cache clearing
- Remove tests for functionality that doesn't exist

### 4. Architecture Alignment
The current system has:
- Unified cache core with dataclass-based keys/results
- Legacy string-based interface for backward compatibility  
- Optional Redis support (not required)
- SQLite persistence as fallback
- Memory-based LRU eviction

### 5. Conservative Approach
Instead of one large rewrite, we can:
1. First fix just the import issues to stop the false skip
2. Incrementally update individual test classes
3. Verify each change works before proceeding
4. Keep existing test patterns where they still work

## Expected Outcome

After the rebase:
- Import errors will properly fail tests instead of skipping
- Tests will validate actual cache functionality 
- No more suspicious skips hiding real issues
- Cache system will have proper test coverage
- Tests will be maintainable and match the actual architecture

## Files to Modify

1. `tests/test_storage_cache.py` - Main test file needing rebase
2. Potentially add integration tests if gaps are found
3. Update any related test documentation

## Verification Plan

1. Run the updated test file to ensure imports work
2. Verify all test classes execute properly
3. Check that cache functionality is properly covered
4. Ensure no regression in other cache-related tests
5. Run full test suite to check for integration issues

## Timeline

- **Phase 1**: Fix imports and basic test structure
- **Phase 2**: Update individual test classes incrementally  
- **Phase 3**: Add any missing test coverage
- **Phase 4**: Final verification and documentation