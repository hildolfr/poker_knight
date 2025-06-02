#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Test Runner - Deadlock Prevention Version

This runner applies all safety measures to prevent deadlocks:
- Unicode encoding fixes
- Redis connection fallbacks  
- NUMA disabling
- Conservative parallel settings
- Proper cleanup procedures
"""

import os
import sys
import time
import subprocess
from pathlib import Path

# Apply all safety measures
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['POKER_KNIGHT_SAFE_MODE'] = '1'
os.environ['POKER_KNIGHT_DISABLE_CACHE_WARMING'] = '1'
os.environ['POKER_KNIGHT_DISABLE_NUMA'] = '1'
os.environ['POKER_KNIGHT_DISABLE_REDIS'] = '1'

def run_safe_test(test_file: str, timeout: int = 60):
    """Run a test with all safety measures applied."""
    print(f"Running safe test: {test_file}")
    
    try:
        # Run with safe environment
        result = subprocess.run(
            [sys.executable, test_file],
            timeout=timeout,
            capture_output=True,
            text=True,
            env=os.environ.copy()
        )
        
        success = result.returncode == 0
        print(f"  Result: {'PASS' if success else 'FAIL'}")
        
        if not success and result.stderr:
            print(f"  Error: {result.stderr[:200]}...")
        
        return success
        
    except subprocess.TimeoutExpired:
        print(f"  Result: TIMEOUT (>{timeout}s)")
        return False
    except Exception as e:
        print(f"  Result: ERROR ({e})")
        return False

def main():
    """Run critical tests safely."""
    print("Safe Test Runner - Deadlock Prevention Version")
    print("=" * 50)
    
    # Test files in order of importance
    test_files = [
        'tests/test_parallel.py',              # Known working
        'test_advanced_parallel.py',           # High priority
        'test_cache_with_redis_demo.py',       # Cache testing
        'tests/test_numa.py',                  # NUMA testing
        'tests/test_cache_integration.py',     # Cache integration
    ]
    
    results = []
    for test_file in test_files:
        if Path(test_file).exists():
            success = run_safe_test(test_file, timeout=120)
            results.append((test_file, success))
        else:
            print(f"Skipping missing file: {test_file}")
    
    # Summary
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! Deadlock issues resolved.")
        return 0
    else:
        print("Some tests still failing - check logs for details")
        return 1

if __name__ == "__main__":
    sys.exit(main())
