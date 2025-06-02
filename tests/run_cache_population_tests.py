#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test runner for Poker Knight Cache Pre-Population System

This script provides easy access to run cache population tests with different
configurations and reporting options.

Usage:
    python run_cache_population_tests.py [options]

Examples:
    # Run all cache population tests
    python run_cache_population_tests.py
    
    # Run only unit tests
    python run_cache_population_tests.py --unit
    
    # Run performance tests
    python run_cache_population_tests.py --performance
    
    # Run with verbose output and coverage
    python run_cache_population_tests.py --verbose --coverage
    
    # Run specific test class
    python run_cache_population_tests.py --class TestPopulationConfig

Author: hildolfr
License: MIT
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_tests(test_args):
    """Run cache population tests with given arguments."""
    
    # Base pytest command for cache population tests
    base_cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_cache_population.py",
        "--cache-population",  # Use our custom marker
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--strict-markers"  # Ensure all markers are defined
    ]
    
    # Add custom arguments
    if test_args.verbose:
        base_cmd.append("-vv")  # Extra verbose
    
    if test_args.quiet:
        base_cmd.remove("-v")
        base_cmd.append("-q")
    
    if test_args.coverage:
        base_cmd.extend(["--cov=poker_knight.storage.cache_prepopulation", 
                        "--cov-report=term-missing", 
                        "--cov-report=html:tests/results/cache_population_coverage"])
    
    if test_args.unit:
        base_cmd.extend(["-m", "unit"])
    elif test_args.integration:
        base_cmd.extend(["-m", "integration"])
    elif test_args.performance:
        base_cmd.extend(["-m", "performance"])
    
    if test_args.class_name:
        base_cmd.extend(["-k", test_args.class_name])
    
    if test_args.test_name:
        base_cmd.extend(["-k", test_args.test_name])
    
    if test_args.fail_fast:
        base_cmd.append("-x")
    
    if test_args.parallel:
        try:
            import pytest_xdist
            base_cmd.extend(["-n", str(test_args.parallel)])
        except ImportError:
            print("Warning: pytest-xdist not available. Running tests sequentially.")
    
    if test_args.durations:
        base_cmd.extend(["--durations=10"])
    
    # Add any additional pytest args
    if test_args.pytest_args:
        base_cmd.extend(test_args.pytest_args.split())
    
    print("♞ Running Poker Knight Cache Pre-Population Tests")
    print("=" * 60)
    print(f"Command: {' '.join(base_cmd)}")
    print("=" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(base_cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"\nError running tests: {e}")
        return 1


def main():
    """Main entry point for the test runner."""
    
    parser = argparse.ArgumentParser(
        description="Test runner for Poker Knight Cache Pre-Population System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run all cache population tests
  %(prog)s --unit                    # Run only unit tests
  %(prog)s --performance             # Run only performance tests
  %(prog)s --integration             # Run only integration tests
  %(prog)s --class TestPopulationConfig  # Run specific test class
  %(prog)s --test "test_scenario_generation" # Run tests matching pattern
  %(prog)s --verbose --coverage      # Run with verbose output and coverage
  %(prog)s --fail-fast               # Stop on first failure
  %(prog)s --parallel 4              # Run tests in parallel (requires pytest-xdist)
        """
    )
    
    # Test selection options
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--unit", action="store_true",
        help="Run only unit tests (fast, no external dependencies)"
    )
    test_group.add_argument(
        "--integration", action="store_true", 
        help="Run only integration tests (with cache system)"
    )
    test_group.add_argument(
        "--performance", action="store_true",
        help="Run only performance tests (may take longer)"
    )
    
    # Specific test selection
    parser.add_argument(
        "--class", dest="class_name", metavar="CLASS_NAME",
        help="Run tests from specific test class (e.g. TestPopulationConfig)"
    )
    parser.add_argument(
        "--test", dest="test_name", metavar="TEST_PATTERN",
        help="Run tests matching pattern (e.g. 'test_scenario_generation')"
    )
    
    # Output options
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose output (show all test details)"
    )
    output_group.add_argument(
        "--quiet", "-q", action="store_true",
        help="Quiet output (minimal details)"
    )
    
    # Analysis options
    parser.add_argument(
        "--coverage", action="store_true",
        help="Run tests with code coverage analysis"
    )
    parser.add_argument(
        "--durations", action="store_true",
        help="Show test durations (slowest 10 tests)"
    )
    
    # Execution options
    parser.add_argument(
        "--fail-fast", "-x", action="store_true",
        help="Stop on first test failure"
    )
    parser.add_argument(
        "--parallel", "-n", type=int, metavar="N",
        help="Run tests in parallel using N workers (requires pytest-xdist)"
    )
    
    # Advanced options
    parser.add_argument(
        "--pytest-args", metavar="ARGS",
        help="Additional arguments to pass to pytest (quoted string)"
    )
    
    args = parser.parse_args()
    
    # Validate requirements
    test_file = Path(__file__).parent / "test_cache_population.py"
    if not test_file.exists():
        print(f"Error: Test file not found: {test_file}")
        return 1
    
    # Run the tests
    exit_code = run_tests(args)
    
    # Report results
    if exit_code == 0:
        print("\n[PASS] All cache population tests passed!")
    elif exit_code == 1:
        print("\n[FAIL] Some cache population tests failed")
    elif exit_code == 130:
        print("\n[WARN]  Tests interrupted by user")
    else:
        print(f"\n[FAIL] Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 