# Cache Testing Guide

## Overview

This guide explains the comprehensive test suite for the Poker Knight v1.6 caching system, including how to run tests, avoid conflicts, and understand test isolation.

## Test Structure

### Test Files

1. **`test_storage_cache.py`** - Comprehensive unit tests for all cache components
2. **`test_cache_integration.py`** - Integration tests with MonteCarloSolver
3. **`test_cache_markers.py`** - Test categorization and markers
4. **`run_cache_tests.py`** - Specialized test runner

### Test Categories

Tests are organized into several categories using pytest markers:

- `@pytest.mark.cache` - All cache-related tests
- `@pytest.mark.cache_unit` - Unit tests for individual cache components
- `@pytest.mark.cache_integration` - Integration tests with solver
- `@pytest.mark.cache_persistence` - Tests involving SQLite/Redis persistence
- `@pytest.mark.cache_performance` - Performance-focused tests
- `@pytest.mark.redis_required` - Tests requiring Redis server

## Running Tests

### Quick Start

```bash
# Run all cache tests
python tests/run_cache_tests.py full

# Run only unit tests (fast, no persistence)
python tests/run_cache_tests.py unit

# Run integration tests with persistence
python tests/run_cache_tests.py integration

# Run Redis tests (requires Redis server)
python tests/run_cache_tests.py redis --check-redis
```

### Using pytest directly

```bash
# Run all cache tests
pytest -m cache tests/

# Run only unit tests
pytest -m "cache_unit and not cache_persistence" tests/

# Run with coverage
pytest --cov=poker_knight.storage tests/test_storage_cache.py
```

## Test Components Covered

### 1. CacheConfig and CacheStats
- **File**: `test_storage_cache.py::TestCacheConfig`
- **Coverage**: Configuration validation, default values, custom settings
- **Isolation**: No persistence, memory-only tests

### 2. Cache Key Generation
- **File**: `test_storage_cache.py::TestCacheKeyGeneration`
- **Coverage**: Key normalization, collision avoidance, parameter inclusion
- **Isolation**: Pure functions, no side effects

### 3. ThreadSafeLRUCache
- **File**: `test_storage_cache.py::TestThreadSafeLRUCache`
- **Coverage**: LRU eviction, memory management, thread safety, statistics
- **Isolation**: Independent cache instances per test

### 4. SQLiteCache
- **File**: `test_storage_cache.py::TestSQLiteCache`
- **Coverage**: CRUD operations, thread safety, cleanup, statistics
- **Isolation**: Temporary databases, automatic cleanup

### 5. HandCache
- **File**: `test_storage_cache.py::TestHandCache`
- **Coverage**: Memory/SQLite/Redis fallback, statistics, cleanup
- **Isolation**: Temporary directories, isolated cache instances

### 6. BoardTextureCache
- **File**: `test_storage_cache.py::TestBoardTextureCache`
- **Coverage**: Texture analysis caching, normalization
- **Isolation**: Independent cache instances

### 7. PreflopRangeCache
- **File**: `test_storage_cache.py::TestPreflopRangeCache`
- **Coverage**: 169 hand combinations, normalization, coverage tracking
- **Isolation**: Independent cache instances

### 8. Integration Tests
- **File**: `test_cache_integration.py::TestCacheIntegration`
- **Coverage**: Solver integration, hit/miss behavior, statistics accuracy
- **Isolation**: Temporary cache files, global cache clearing

### 9. Redis Integration
- **File**: `test_storage_cache.py::TestRedisIntegration`
- **Coverage**: Redis operations, fallback behavior
- **Requirements**: Redis server on localhost:6379
- **Isolation**: Separate Redis database (db=15)

## Conflict Resolution

### Problem: Global Cache Manager Singleton

**Issue**: The cache system uses a global singleton that could cause interference between tests.

**Solution**: 
- `clear_all_caches()` function clears all global state
- Called in `setUp()` and `tearDown()` methods
- Test-specific cache configurations with isolated paths

### Problem: SQLite File Conflicts

**Issue**: Multiple tests using the same SQLite file could interfere.

**Solution**:
- Each test uses `tempfile.mkdtemp()` for isolated directories
- Automatic cleanup in `tearDown()` methods
- Unique database names per test

### Problem: Redis State Persistence

**Issue**: Redis data persists between test runs.

**Solution**:
- Use separate Redis database (db=15) for tests
- `flushdb()` in test cleanup
- Test isolation through unique key prefixes

### Problem: Memory Cache State

**Issue**: In-memory caches could retain data between tests.

**Solution**:
- Explicit cache clearing in test setup/teardown
- Independent cache instances where possible
- Memory usage verification

### Problem: Thread Safety Test Interference

**Issue**: Multi-threaded tests could interfere with each other.

**Solution**:
- Thread-local test data
- Proper thread synchronization
- Isolated test data per thread

## Best Practices

### 1. Test Isolation

```python
def setUp(self):
    # Clear global state
    clear_all_caches()
    
    # Create isolated environment
    self.temp_dir = tempfile.mkdtemp()
    self.config = CacheConfig(
        sqlite_path=os.path.join(self.temp_dir, "test.db")
    )

def tearDown(self):
    # Clean up instances
    self.cache.clear()
    
    # Clear global state
    clear_all_caches()
    
    # Remove temp files
    shutil.rmtree(self.temp_dir)
```

### 2. Redis Testing

```python
@pytest.mark.skipif(not REDIS_AVAILABLE, reason="Redis not available")
def test_redis_feature(self):
    # Use test database
    config = CacheConfig(redis_db=15)
    # Test implementation
```

### 3. Performance Testing

```python
def test_cache_performance(self):
    # Measure cache hit performance
    start_time = time.time()
    result = cache.get(key)
    hit_time = (time.time() - start_time) * 1000
    
    # Assert sub-10ms performance target
    self.assertLess(hit_time, 10.0)
```

### 4. Thread Safety Testing

```python
def test_thread_safety(self):
    results = []
    errors = []
    
    def worker(thread_id):
        # Thread-specific operations
        # Record results and errors
        
    threads = [Thread(target=worker, args=(i,)) for i in range(5)]
    # Start, join, verify results
```

## Environment Variables

The test runner sets environment variables for isolation:

- `POKER_KNIGHT_TEST_MODE=1` - Indicates test mode
- `POKER_KNIGHT_TEST_CACHE_DIR` - Temporary directory for test files

## Troubleshooting

### Redis Connection Issues

```bash
# Check Redis status
redis-cli ping

# Start Redis (if needed)
redis-server

# Run tests with Redis check
python tests/run_cache_tests.py redis --check-redis
```

### SQLite Permission Issues

```bash
# Ensure temp directory is writable
ls -la /tmp/

# Run with verbose output to see paths
python tests/run_cache_tests.py integration --verbose
```

### Import Issues

```bash
# Verify cache module can be imported
python -c "from poker_knight.storage.cache import HandCache; print('OK')"

# Check for missing dependencies
pip install redis  # If using Redis tests
```

### Performance Test Failures

Performance tests may be sensitive to system load:

```bash
# Run performance tests in isolation
pytest -m cache_performance -v tests/

# Check system resources
top
```

## Continuous Integration

For CI environments:

```bash
# Skip Redis tests if Redis not available
python tests/run_cache_tests.py integration

# Run with coverage
python tests/run_cache_tests.py full --coverage

# Parallel execution
python tests/run_cache_tests.py unit --parallel 4
```

## Extending Tests

When adding new cache functionality:

1. Add unit tests to `test_storage_cache.py`
2. Add integration tests to `test_cache_integration.py`
3. Use appropriate markers (`@pytest.mark.cache`, etc.)
4. Ensure proper isolation in setUp/tearDown
5. Document any new environment requirements
6. Update this guide if needed

## Test Coverage Goals

- **Unit Tests**: 100% line coverage for cache components
- **Integration Tests**: All solver-cache interaction paths
- **Performance Tests**: Sub-10ms cache hit times
- **Thread Safety**: Multi-threaded operation verification
- **Persistence**: Redis and SQLite fallback scenarios
- **Error Handling**: Graceful degradation testing 