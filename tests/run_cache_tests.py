#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Specialized test runner for Poker Knight caching system tests.

This script provides different modes for running cache tests:
- Unit tests only (no persistence)
- Integration tests (with persistence)
- Performance tests
- Full test suite
- Redis tests (if Redis is available)

Ensures proper test isolation and conflict avoidance.
"""

import os
import sys
import subprocess
import argparse
import tempfile
import shutil
from pathlib import Path


def check_redis_available():
    """Check if Redis server is available for testing."""
    try:
        import redis
        client = redis.Redis(host='localhost', port=6379, db=15)
        client.ping()
        return True
    except:
        return False


def setup_test_environment():
    """Set up isolated test environment."""
    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp(prefix="poker_knight_cache_test_")
    
    # Set environment variables for test isolation
    os.environ['POKER_KNIGHT_TEST_MODE'] = '1'
    os.environ['POKER_KNIGHT_TEST_CACHE_DIR'] = temp_dir
    
    return temp_dir


def cleanup_test_environment(temp_dir):
    """Clean up test environment."""
    try:
        shutil.rmtree(temp_dir)
    except:
        pass
    
    # Clean up environment variables
    os.environ.pop('POKER_KNIGHT_TEST_MODE', None)
    os.environ.pop('POKER_KNIGHT_TEST_CACHE_DIR', None)


def run_pytest_command(args, test_selection=None):
    """Run pytest with specific arguments."""
    cmd = ['python', '-m', 'pytest']
    
    # Add verbosity
    cmd.extend(['-v', '--tb=short'])
    
    # Add test selection if specified
    if test_selection:
        cmd.extend(['-m', test_selection])
    
    # Add specific test files
    cmd.extend([
        'tests/test_storage_cache.py',
        'tests/test_cache_integration.py'
    ])
    
    # Add any additional arguments
    cmd.extend(args)
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=Path(__file__).parent.parent)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run Poker Knight caching system tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Modes:
  unit        Run unit tests only (no persistence, fast)
  integration Run integration tests (with persistence)
  performance Run performance tests
  redis       Run Redis tests (requires Redis server)
  full        Run all cache tests
  
Examples:
  python run_cache_tests.py unit
  python run_cache_tests.py integration --verbose
  python run_cache_tests.py redis --check-redis
  python run_cache_tests.py full --coverage
        """
    )
    
    parser.add_argument(
        'mode', 
        choices=['unit', 'integration', 'performance', 'redis', 'full'],
        help='Test mode to run'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Increase verbosity'
    )
    
    parser.add_argument(
        '--check-redis',
        action='store_true',
        help='Check Redis availability before running Redis tests'
    )
    
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Run tests with coverage analysis'
    )
    
    parser.add_argument(
        '--parallel', '-n',
        type=int,
        help='Run tests in parallel with N workers'
    )
    
    parser.add_argument(
        '--stop-on-first-failure', '-x',
        action='store_true',
        help='Stop on first test failure'
    )
    
    args = parser.parse_args()
    
    # Set up test environment
    temp_dir = setup_test_environment()
    
    try:
        # Check Redis availability if needed
        if args.mode == 'redis' or (args.mode == 'full' and args.check_redis):
            if not check_redis_available():
                print("[FAIL] Redis server not available!")
                print("   Please start Redis server or skip Redis tests")
                if args.mode == 'redis':
                    return 1
                else:
                    print("   Continuing without Redis tests...")
        
        # Build pytest arguments
        pytest_args = []
        
        if args.verbose:
            pytest_args.append('-vv')
        
        if args.coverage:
            pytest_args.extend(['--cov=poker_knight.storage', '--cov-report=term-missing'])
        
        if args.parallel:
            pytest_args.extend(['-n', str(args.parallel)])
        
        if args.stop_on_first_failure:
            pytest_args.append('-x')
        
        # Select test markers based on mode
        test_selection = None
        if args.mode == 'unit':
            test_selection = 'cache_unit and not cache_persistence'
            print("🧪 Running cache unit tests (no persistence)")
        elif args.mode == 'integration':
            test_selection = 'cache_integration or cache_persistence'
            print("🔗 Running cache integration tests (with persistence)")
        elif args.mode == 'performance':
            test_selection = 'cache_performance'
            print("⚡ Running cache performance tests")
        elif args.mode == 'redis':
            test_selection = 'redis_required'
            print("📡 Running Redis-specific tests")
        elif args.mode == 'full':
            test_selection = 'cache'
            print("🎯 Running full cache test suite")
        
        # Run the tests
        print(f"Test isolation directory: {temp_dir}")
        result = run_pytest_command(pytest_args, test_selection)
        
        # Report results
        if result.returncode == 0:
            print("\n[PASS] All cache tests passed!")
        else:
            print(f"\n[FAIL] Some cache tests failed (exit code: {result.returncode})")
        
        return result.returncode
        
    finally:
        # Clean up test environment
        cleanup_test_environment(temp_dir)


if __name__ == '__main__':
    sys.exit(main()) 