# Poker Knight Test Suite

A comprehensive, pytest-based testing framework for the Poker Knight Monte Carlo poker solver.

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

## Test Suite Options

| Option | Description | Usage |
|--------|-------------|-------|
| `--all` | Complete comprehensive test suite | `pytest --all` |
| `--quick` | Fast validation subset | `pytest --quick` |
| `--unit` | Core functionality only | `pytest --unit` |
| `--statistical` | Statistical validation | `pytest --statistical` |
| `--performance` | Performance and benchmarks | `pytest --performance` |
| `--stress` | Stress and load testing | `pytest --stress` |
| (no option) | Same as `--all` | `pytest` |

## Advanced Usage

```bash
# Combine markers
pytest -m "unit and not slow"
pytest -m "statistical or performance"

# Run specific test files
pytest tests/test_poker_solver.py
pytest tests/test_statistical_validation.py -v

# Run with coverage and HTML report
pytest --cov=poker_knight --cov-report=html

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
├── conftest.py                    # Pytest configuration and fixtures
├── run_tests.py                   # Simple wrapper script (optional)
├── test_poker_solver.py           # Core unit tests
├── test_statistical_validation.py # Statistical accuracy tests
├── test_performance.py            # Performance benchmarks
├── test_stress_scenarios.py       # Stress testing
├── test_edge_cases_extended.py    # Edge case testing
└── ...                           # Additional test modules
```

## Legacy Compatibility

The original `run_tests.py` script is now a simple wrapper around pytest:

```bash
# These are equivalent:
python tests/run_tests.py --quick
pytest --quick

# These are equivalent:
python tests/run_tests.py --statistical
pytest --statistical

# These are equivalent:
python tests/run_tests.py --all
pytest --all

# These are equivalent (default behavior):
python tests/run_tests.py
pytest
```

## Coverage Analysis

Generate test coverage reports:

```bash
# Terminal report
pytest --cov=poker_knight --cov-report=term-missing

# HTML report (opens in browser)
pytest --cov=poker_knight --cov-report=html
open htmlcov/index.html

# Combined with test filtering
pytest --quick --cov=poker_knight --cov-report=html
```

## Configuration

Test behavior is configured in:
- `pytest.ini` - Main pytest configuration
- `tests/conftest.py` - Custom options and fixtures

## Tips

1. **Use markers** for efficient test selection
2. **Run quick tests** during development: `pytest --quick`
3. **Full validation** before commits: `pytest`
4. **Statistical tests** for accuracy verification: `pytest --statistical`
5. **Performance monitoring**: `pytest --performance`

## Automatic Result Logging

All test runs are automatically logged to timestamped files in `tests/results/` for AI analysis and regression tracking:

### File Format
```
test_results_YYYY-MM-DD_HH-MM-SS_<test_type>.json
```

### Examples
- `test_results_2024-06-01_14-30-25_all.json`
- `test_results_2024-06-01_14-35-12_unit.json`
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