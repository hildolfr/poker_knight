# Poker Knight Test Suite

A comprehensive, pytest-based testing framework for the Poker Knight Monte Carlo poker solver with integrated caching system testing.

## Quick Start

```bash
# Run all tests (comprehensive suite)
pytest --all

# Run all tests (default - same as --all)
pytest

# Quick validation (fast subset)
pytest --quick

# Statistical validation
pytest --statistical

# Performance tests
pytest --performance

# Unit tests only
pytest --unit

# Cache tests (all cache-related)
pytest --cache

# Cache unit tests (fast, no persistence)
pytest --cache-unit

# Cache integration tests
pytest --cache-integration

# Redis tests (requires Redis server)
pytest --redis

# With coverage
pytest --all --with-coverage
```

## Test Categories

The test suite is organized using pytest markers:

| Marker | Description | Example Usage |
|--------|-------------|---------------|
| `unit` | Core functionality tests | `pytest -m unit` |
| `statistical` | Statistical validation tests | `pytest -m statistical` |
| `performance` | Performance and benchmarks | `pytest -m performance` |
| `stress` | Stress and load testing | `pytest -m stress` |
| `quick` | Fast validation subset | `pytest -m quick` |
| `integration` | Multi-component tests | `pytest -m integration` |
| `slow` | Long-running tests | `pytest -m "not slow"` |
| `cache` | All cache-related tests | `pytest -m cache` |
| `cache_unit` | Fast cache unit tests | `pytest -m cache_unit` |
| `cache_integration` | Cache integration tests | `pytest -m cache_integration` |
| `cache_persistence` | Cache persistence tests | `pytest -m cache_persistence` |
| `cache_performance` | Cache performance tests | `pytest -m cache_performance` |
| `redis_required` | Tests requiring Redis | `pytest -m redis_required` |

## Test Suite Options

| Option | Description | Usage |
|--------|-------------|-------|
| `--all` | Complete comprehensive test suite | `pytest --all` |
| `--quick` | Fast validation subset | `pytest --quick` |
| `--unit` | Core functionality only | `pytest --unit` |
| `--statistical` | Statistical validation | `pytest --statistical` |
| `--performance` | Performance and benchmarks | `pytest --performance` |
| `--stress` | Stress and load testing | `pytest --stress` |
| `--cache` | All cache tests | `pytest --cache` |
| `--cache-unit` | Fast cache unit tests | `pytest --cache-unit` |
| `--cache-integration` | Cache integration tests | `pytest --cache-integration` |
| `--cache-performance` | Cache performance tests | `pytest --cache-performance` |
| `--redis` | Redis-dependent tests | `pytest --redis` |
| (no option) | Same as `--all` | `pytest` |

## Cache Test Integration

The caching system tests are fully integrated into the main test suite:

### Cache Test Categories

1. **Cache Unit Tests** (`--cache-unit`)
   - Fast, in-memory tests
   - No external dependencies (Redis/SQLite)
   - Tests core cache logic and data structures
   - Ideal for development workflow

2. **Cache Integration Tests** (`--cache-integration`)
   - Integration with MonteCarloSolver
   - End-to-end cache workflows
   - Multi-component interaction testing

3. **Cache Persistence Tests** (`--cache-persistence`)
   - Redis and SQLite persistence testing
   - Fallback mechanism validation
   - Data integrity and recovery

4. **Cache Performance Tests** (`--cache-performance`)
   - Performance benchmarking
   - Memory usage validation
   - Cache hit rate optimization
   - Sub-10ms response time verification

### Running Cache Tests

```bash
# All cache tests
pytest --cache

# Fast development testing (no external dependencies)
pytest --cache-unit

# Full integration testing
pytest --cache-integration

# Performance validation
pytest --cache-performance

# Redis-specific tests (requires Redis server)
pytest --redis

# Cache tests without Redis dependencies
pytest -m "cache and not redis_required"

# Quick cache validation
pytest -m "cache and quick"
```

## Advanced Usage

```bash
# Combine markers
pytest -m "unit and not slow"
pytest -m "statistical or performance"
pytest -m "cache and not redis_required"

# Run specific test files
pytest tests/test_poker_solver.py
pytest tests/test_statistical_validation.py -v
pytest tests/test_storage_cache.py
pytest tests/test_cache_integration.py

# Run with coverage and HTML report
pytest --cov=poker_knight --cov-report=html

# Cache-specific coverage
pytest --cache --cov=poker_knight.storage --cov-report=html

# Parallel execution (if pytest-xdist installed)
pytest -n auto

# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l
```

## Test Structure

```
tests/
├── conftest.py                     # Pytest configuration and fixtures
├── run_tests.py                    # Simple wrapper script (optional)
├── test_poker_solver.py            # Core unit tests
├── test_statistical_validation.py  # Statistical accuracy tests
├── test_performance.py             # Performance benchmarks
├── test_stress_scenarios.py        # Stress testing
├── test_edge_cases_extended.py     # Edge case testing
├── test_storage_cache.py           # Cache unit tests
├── test_cache_integration.py       # Cache integration tests
├── run_cache_tests.py              # Specialized cache test runner
├── cache_testing_guide.md          # Cache testing documentation
└── ...                            # Additional test modules
```

## Legacy Compatibility

The original `run_tests.py` script is now a simple wrapper around pytest with full cache test integration:

```bash
# These are equivalent:
python tests/run_tests.py --cache-unit
pytest --cache-unit

# These are equivalent:
python tests/run_tests.py --redis
pytest --redis

# These are equivalent (default behavior):
python tests/run_tests.py
pytest
```

The specialized `run_cache_tests.py` script is still available for advanced cache testing scenarios but the main test suite now includes all cache functionality.

## Coverage Analysis

Generate test coverage reports:

```bash
# Terminal report for all tests
pytest --cov=poker_knight --cov-report=term-missing

# HTML report for all tests
pytest --cov=poker_knight --cov-report=html
open htmlcov/index.html

# Cache-specific coverage
pytest --cache --cov=poker_knight.storage --cov-report=html

# Combined with test filtering
pytest --quick --cov=poker_knight --cov-report=html
```

## Configuration

Test behavior is configured in:
- `pytest.ini` - Main pytest configuration with cache markers
- `tests/conftest.py` - Custom options and fixtures (includes cache integration)

## Development Workflow

### Quick Development Testing
```bash
# Fast validation during development
pytest --quick

# Fast cache testing (no external dependencies)
pytest --cache-unit

# Quick cache validation
pytest -m "cache and quick"
```

### Pre-Commit Validation
```bash
# Full test suite validation
pytest --all

# Core functionality + cache validation
pytest -m "unit or cache"
```

### Performance Monitoring
```bash
# All performance tests
pytest --performance

# Cache performance specifically
pytest --cache-performance

# Combined performance testing
pytest -m "performance or cache_performance"
```

## Tips

1. **Use markers** for efficient test selection
2. **Run quick tests** during development: `pytest --quick`
3. **Cache unit tests** for fast cache validation: `pytest --cache-unit`
4. **Full validation** before commits: `pytest`
5. **Statistical tests** for accuracy verification: `pytest --statistical`
6. **Performance monitoring**: `pytest --performance`
7. **Cache performance**: `pytest --cache-performance`
8. **Redis testing** (if available): `pytest --redis`

## Automatic Result Logging

All test runs are automatically logged to timestamped files in `tests/results/` for AI analysis and regression tracking:

### File Format
```
test_results_YYYY-MM-DD_HH-MM-SS_<test_type>.json
```

### Examples
- `test_results_2024-06-01_14-30-25_all.json`
- `test_results_2024-06-01_14-35-12_unit.json`
- `test_results_2024-06-01_14-40-18_cache.json`
- `test_results_2024-06-01_15-20-40_statistical.json`

### Contents
Each file contains:
- **Summary statistics**: Pass/fail counts, success rates
- **Failed test details**: Names, error messages, durations
- **Configuration used**: Which test type was run
- **Regression indicators**: Automated analysis flags
- **Overall status**: PASS/FAIL for easy AI parsing

### AI Analysis Features
- **Never overwritten**: Timestamp prevents data loss
- **Regression tracking**: Compare results across versions
- **Trend analysis**: Monitor test health over time
- **Structured data**: JSON format for easy parsing

The pytest framework provides excellent output formatting, progress indicators, and failure reporting - much cleaner than the previous custom implementation! 