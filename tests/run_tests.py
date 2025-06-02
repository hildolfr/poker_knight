#!/usr/bin/env python3
"""
Poker Knight Test Suite - Clean Pytest Interface

A simple helper script that provides convenient aliases for pytest commands
with the Poker Knight test markers and configuration.

Usage Examples:
    # Quick validation (fast subset of tests)
    python tests/run_tests.py --quick
    pytest --quick

    # Statistical validation tests
    python tests/run_tests.py --statistical  
    pytest --statistical

    # Performance and benchmark tests
    python tests/run_tests.py --performance
    pytest --performance

    # Stress and load testing
    python tests/run_tests.py --stress
    pytest --stress

    # Core unit tests only
    python tests/run_tests.py --unit
    pytest --unit

    # Cache tests (all cache-related tests)
    python tests/run_tests.py --cache
    pytest --cache

    # Cache unit tests (fast, no persistence)
    python tests/run_tests.py --cache-unit
    pytest --cache-unit

    # Cache integration tests (with solver)
    python tests/run_tests.py --cache-integration
    pytest --cache-integration

    # Cache performance tests
    python tests/run_tests.py --cache-performance
    pytest --cache-performance

    # Redis-dependent tests only
    python tests/run_tests.py --redis
    pytest --redis

    # NUMA (Non-Uniform Memory Access) tests only
    python tests/run_tests.py --numa
    pytest --numa

    # All tests (comprehensive suite)
    python tests/run_tests.py --all
    pytest --all

    # Only tests that failed in the last run
    python tests/run_tests.py --failed
    pytest --failed

    # All tests (default - same as --all)
    python tests/run_tests.py
    pytest

    # With coverage analysis
    python tests/run_tests.py --all --with-coverage
    pytest --all --with-coverage

    # Native pytest marker selection
    pytest -m "unit and not slow"
    pytest -m "statistical or performance"
    pytest -m "cache and not redis_required"
    pytest -m "numa and not slow"
    pytest -m "not stress"

Available Test Markers:
    - unit: Core functionality tests
    - statistical: Statistical validation tests  
    - performance: Performance and benchmark tests
    - stress: Stress and load testing
    - quick: Fast validation subset
    - edge_cases: Edge case and boundary testing
    - integration: Integration tests
    - slow: Long-running tests
    - cache: All cache-related tests
    - cache_unit: Cache unit tests (fast, no persistence)
    - cache_integration: Cache integration tests (with solver)
    - cache_persistence: Cache persistence tests (Redis/SQLite)
    - cache_performance: Cache performance benchmarks
    - redis_required: Tests requiring Redis server
    - numa: NUMA (Non-Uniform Memory Access) related tests

Available Test Suites:
    --quick             Fast validation (development workflow)
    --unit              Core functionality only
    --statistical       Statistical accuracy validation
    --performance       Performance and benchmarks
    --stress            Stress and load testing
    --cache             All cache tests
    --cache-unit        Cache unit tests (fast)
    --cache-integration Cache integration tests
    --cache-performance Cache performance tests
    --redis             Redis-dependent tests
    --numa              NUMA topology and parallel processing tests
    --all               Complete comprehensive test suite
    --failed            Only tests that failed in the last run
    (no flag)           Same as --all (runs everything)

Cache Test Organization:
    The cache tests are fully integrated into the main test suite:
    
    Cache Unit Tests:
    - Fast, in-memory tests
    - No external dependencies
    - Run with: pytest --cache-unit
    
    Cache Integration Tests:
    - Tests with MonteCarloSolver
    - Persistence layer testing
    - Run with: pytest --cache-integration
    
    Cache Performance Tests:
    - Benchmarking and optimization
    - Performance regression detection
    - Run with: pytest --cache-performance
    
    Redis Tests:
    - Require Redis server running
    - Automatic fallback testing
    - Run with: pytest --redis

NUMA Test Organization:
    The NUMA tests validate Non-Uniform Memory Access functionality:
    
    NUMA Unit Tests:
    - Topology detection and validation
    - CPU-to-node mapping verification
    - Run with: pytest --numa
    
    NUMA Integration Tests:
    - Integration with parallel processing engine
    - Work distribution and affinity testing
    - Run with: pytest --numa
    
    NUMA Performance Tests:
    - Performance comparison with/without NUMA awareness
    - Memory locality optimization validation
    - Run with: pytest --numa

Direct pytest usage is recommended for maximum flexibility!
"""

import sys
import subprocess
from pathlib import Path


def print_usage():
    """Print usage information and examples."""
    print(__doc__)


def main():
    """Simple wrapper that passes arguments to pytest."""
    
    # Change to project root
    project_root = Path(__file__).parent.parent
    
    # If no arguments or --help, show usage
    if len(sys.argv) == 1 or "--help" in sys.argv or "-h" in sys.argv:
        print_usage()
        return
    
    # Build pytest command
    pytest_args = ["python", "-m", "pytest"]
    
    # Add any additional arguments passed to this script
    pytest_args.extend(sys.argv[1:])
    
    print("Poker Knight Test Suite")
    print(f"   Running: {' '.join(pytest_args)}")
    print("   Use 'pytest --help' for full pytest options")
    print()
    
    # Execute pytest
    try:
        result = subprocess.run(pytest_args, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n[!] Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[X] Failed to run tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 