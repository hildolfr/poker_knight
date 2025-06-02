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

Available Test Suites:
    --quick       Fast validation (development workflow)
    --unit        Core functionality only
    --statistical Statistical accuracy validation
    --performance Performance and benchmarks
    --stress      Stress and load testing
    --all         Complete comprehensive test suite
    --failed      Only tests that failed in the last run
    (no flag)     Same as --all (runs everything)

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
    
    print("♞ Poker Knight Test Suite")
    print(f"   Running: {' '.join(pytest_args)}")
    print("   Use 'pytest --help' for full pytest options")
    print()
    
    # Execute pytest
    try:
        result = subprocess.run(pytest_args, cwd=project_root)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Failed to run tests: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 