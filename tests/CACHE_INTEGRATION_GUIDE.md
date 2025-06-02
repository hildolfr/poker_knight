# Poker Knight Cache Test Integration Guide

This guide documents how cache tests are integrated into the main Poker Knight test suite, providing a unified testing experience.

## Overview

The cache tests are now fully integrated into the main test suite using pytest markers and custom command-line options. This provides:

- **Unified test execution** - Run cache tests alongside other tests
- **Selective test running** - Run only specific types of cache tests  
- **Consistent workflow** - Same commands and patterns as other tests
- **Proper isolation** - Cache tests are properly marked and categorized

## Quick Start

```bash
# Run all cache tests
pytest --cache

# Run fast cache unit tests (development workflow)
pytest --cache-unit

# Run cache integration tests
pytest --cache-integration

# Run cache performance tests
pytest --cache-performance

# Run cache tests without Redis dependencies
pytest -m "cache and not redis_required"

# Run all tests (including cache tests)
pytest --all
```

## Test Categories

### 1. Cache Unit Tests (`--cache-unit`)
- **Purpose**: Fast, isolated tests of cache components
- **Dependencies**: None (memory-only)
- **Speed**: Very fast (< 10 seconds)
- **Use case**: Development workflow, CI fast checks

**Components tested:**
- `CacheConfig` creation and validation
- Cache key generation and normalization
- Memory cache operations (LRU behavior)
- Thread safety mechanisms
- Cache statistics and monitoring

**Run with:**
```bash
pytest --cache-unit
pytest -m cache_unit
pytest -m "cache_unit and quick"  # Extra fast subset
```

### 2. Cache Integration Tests (`--cache-integration`)
- **Purpose**: Test cache integration with MonteCarloSolver
- **Dependencies**: Main solver components
- **Speed**: Medium (< 30 seconds)
- **Use case**: Feature validation, integration testing

**Components tested:**
- End-to-end cache workflows
- Solver-cache interaction
- Cache hit/miss behavior in real scenarios
- Multi-component data flow

**Run with:**
```bash
pytest --cache-integration
pytest -m cache_integration
```

### 3. Cache Persistence Tests (`--cache-persistence`)
- **Purpose**: Test Redis and SQLite persistence layers
- **Dependencies**: SQLite (always), Redis (optional)
- **Speed**: Medium (< 45 seconds)
- **Use case**: Persistence validation, fallback testing

**Components tested:**
- SQLite cache operations
- Redis cache operations (if available)
- Fallback mechanisms (Redis → SQLite → Memory)
- Data integrity and recovery
- Persistence configuration

**Run with:**
```bash
pytest -m cache_persistence
pytest -m "cache_persistence and not redis_required"  # SQLite only
```

### 4. Cache Performance Tests (`--cache-performance`)
- **Purpose**: Performance benchmarking and regression detection
- **Dependencies**: Performance monitoring tools
- **Speed**: Medium (< 60 seconds)
- **Use case**: Performance validation, optimization

**Components tested:**
- Sub-10ms response time requirements
- Memory usage limits
- Cache hit rate optimization
- Throughput under load
- Performance regression detection

**Run with:**
```bash
pytest --cache-performance
pytest -m cache_performance
```

### 5. Redis Tests (`--redis`)
- **Purpose**: Redis-specific functionality
- **Dependencies**: Redis server running
- **Speed**: Medium (< 30 seconds)
- **Use case**: Redis deployment validation

**Components tested:**
- Redis connection and authentication
- Redis-specific cache operations
- Redis failover scenarios
- Redis configuration validation

**Run with:**
```bash
pytest --redis
pytest -m redis_required
```

## Command-Line Options

### Cache-Specific Options

| Option | Description | Equivalent Marker |
|--------|-------------|-------------------|
| `--cache` | All cache-related tests | `-m cache` |
| `--cache-unit` | Fast unit tests only | `-m cache_unit` |
| `--cache-integration` | Integration tests | `-m cache_integration` |
| `--cache-performance` | Performance tests | `-m cache_performance` |
| `--redis` | Redis-dependent tests | `-m redis_required` |

### General Options

| Option | Description | Use Case |
|--------|-------------|----------|
| `--quick` | Fast subset of all tests | Development workflow |
| `--unit` | Core unit tests | Basic validation |
| `--all` | Complete test suite | Pre-commit, CI |
| `--performance` | All performance tests | Performance monitoring |

## Marker System

### Cache Markers

| Marker | Applied To | Purpose |
|--------|------------|---------|
| `cache` | All cache tests | Select all cache functionality |
| `cache_unit` | Unit tests | Fast, isolated tests |
| `cache_integration` | Integration tests | Multi-component tests |
| `cache_persistence` | Persistence tests | Storage layer tests |
| `cache_performance` | Performance tests | Benchmarking tests |
| `redis_required` | Redis tests | Redis dependency tests |

### Combining Markers

```bash
# Cache tests suitable for quick development validation
pytest -m "cache and quick"

# Cache tests that don't require Redis
pytest -m "cache and not redis_required"

# All persistence tests (cache and non-cache)
pytest -m "cache_persistence or statistical"

# Performance tests (cache and general)
pytest -m "performance or cache_performance"
```

## Integration with Existing Tests

### Automatic Marker Application

The test system automatically applies markers based on:

1. **File names** - Tests in `test_storage_cache.py` get cache markers
2. **Class names** - Test classes get appropriate sub-markers  
3. **Test names** - Tests with performance keywords get performance markers
4. **Manual decoration** - Explicit `@pytest.mark.cache` decorations

### Test Discovery

Cache tests are discovered alongside other tests:

```bash
# See all collected tests
pytest --collect-only

# See only cache tests
pytest --cache --collect-only

# See test counts by category
pytest -m cache --collect-only -q
pytest -m cache_unit --collect-only -q
```

## Development Workflow

### Daily Development

```bash
# Quick validation (< 30 seconds)
pytest --quick

# Quick cache validation (< 10 seconds)  
pytest --cache-unit

# Specific cache feature testing
pytest tests/test_storage_cache.py::TestCacheConfig -v
```

### Feature Development

```bash
# Test specific cache functionality
pytest -m cache_unit -k "test_cache_key"

# Test integration after changes
pytest --cache-integration

# Performance impact assessment
pytest --cache-performance
```

### Pre-Commit Validation

```bash
# Full test suite
pytest --all

# Core functionality + cache
pytest -m "unit or cache"

# Performance regression check
pytest -m "performance or cache_performance"
```

## Configuration Files

### pytest.ini

Contains all marker definitions including cache markers:

```ini
markers =
    cache: Caching system tests (all cache-related tests)
    cache_unit: Cache unit tests (fast, no persistence)
    cache_integration: Cache integration tests (with solver)
    cache_persistence: Cache persistence tests (Redis/SQLite)
    cache_performance: Cache performance benchmarks
    redis_required: Tests requiring Redis server
```

### conftest.py

Handles:
- Cache command-line options (`--cache`, `--cache-unit`, etc.)
- Automatic marker application
- Test collection filtering
- Cache-specific fixtures

## File Organization

```
tests/
├── conftest.py                    # Main pytest configuration
├── pytest.ini                    # Marker definitions
├── run_tests.py                  # Unified test runner
├── test_storage_cache.py         # Cache unit tests
├── test_cache_integration.py     # Cache integration tests
├── test_integration_demo.py      # Integration demonstration
├── run_cache_tests.py           # Specialized cache runner (legacy)
├── cache_testing_guide.md       # Detailed cache test docs
└── CACHE_INTEGRATION_GUIDE.md   # This file
```

## Troubleshooting

### Common Issues

1. **No tests collected for cache markers**
   ```bash
   # Check marker definitions
   pytest --markers | grep cache
   
   # Verify test files exist
   ls tests/test_*cache*.py
   ```

2. **Redis tests failing**
   ```bash
   # Skip Redis tests
   pytest -m "cache and not redis_required"
   
   # Check Redis server
   redis-cli ping
   ```

3. **Performance tests timing out**
   ```bash
   # Run with increased timeout
   pytest --cache-performance --timeout=120
   
   # Run single performance test
   pytest -k "test_cache_performance_basic" -v
   ```

### Validation Script

Use the validation script to check integration:

```bash
python validate_test_integration.py
```

This will test:
- Marker functionality
- Command-line options
- Test execution
- Configuration files
- File organization

## Migration from Standalone Cache Tests

### Before (Standalone)

```bash
python tests/run_cache_tests.py unit
python tests/run_cache_tests.py performance
python tests/run_cache_tests.py redis
```

### After (Integrated)

```bash
pytest --cache-unit
pytest --cache-performance  
pytest --redis
```

### Backward Compatibility

The old `run_cache_tests.py` script still works but the new unified approach is recommended:

```bash
# Old way (still works)
python tests/run_cache_tests.py unit

# New way (recommended)
pytest --cache-unit
```

## Benefits of Integration

1. **Unified Experience** - Same commands and patterns for all tests
2. **Better Discovery** - Cache tests appear in main test collection
3. **Flexible Selection** - Combine cache tests with other test categories
4. **Consistent Reporting** - Same output format and result logging
5. **CI/CD Integration** - Cache tests work with existing CI pipelines
6. **IDE Support** - Better IDE integration and test discovery

## Best Practices

1. **Use appropriate markers** for test selection
2. **Run cache-unit tests** during active development
3. **Include cache tests** in pre-commit hooks
4. **Monitor cache performance** regularly
5. **Test Redis fallback** scenarios
6. **Validate cache hit rates** in integration tests

## Getting Help

- Review test output with `-v` flag for detailed information
- Use `--collect-only` to see which tests would run
- Check the cache testing guide: `tests/cache_testing_guide.md`
- Run the validation script: `python validate_test_integration.py` 