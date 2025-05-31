#!/usr/bin/env python3
"""
Poker Knight Test Runner

Convenient script to run various test suites for the Poker Knight project.
"""

import sys
import subprocess
import argparse
import os

def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {result.returncode})")
            
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error running {description}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run Poker Knight tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--unit", action="store_true", help="Run unit tests")
    parser.add_argument("--statistical", action="store_true", help="Run statistical validation tests")
    parser.add_argument("--performance", action="store_true", help="Run performance tests")
    parser.add_argument("--regression", action="store_true", help="Run regression tests")
    parser.add_argument("--quick", action="store_true", help="Run quick validation tests")
    parser.add_argument("--example", action="store_true", help="Run example usage")
    
    args = parser.parse_args()
    
    # If no specific tests requested, run all
    if not any([args.unit, args.statistical, args.performance, args.regression, args.quick, args.example]):
        args.all = True
    
    print("♞ Poker Knight Test Runner v1.3.0")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    if args.all or args.unit:
        total_count += 1
        if run_command("python -m pytest tests/test_poker_solver.py -v", "Unit Tests"):
            success_count += 1
    
    if args.all or args.statistical:
        total_count += 1
        if run_command("python -m pytest tests/test_statistical_validation.py -v", "Statistical Validation Tests"):
            success_count += 1
    
    if args.all or args.performance:
        total_count += 1
        if run_command("python tests/test_performance.py", "Performance Benchmarks"):
            success_count += 1
    
    if args.all or args.regression:
        total_count += 1
        if run_command("python -m pytest tests/test_performance_regression.py -v", "Performance Regression Tests"):
            success_count += 1
    
    if args.all or args.quick:
        total_count += 1
        if run_command("python tests/test_validation.py", "Quick Validation Tests"):
            success_count += 1
    
    if args.example:
        total_count += 1
        if run_command("python examples/example_usage.py", "Example Usage"):
            success_count += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"🎯 Test Summary")
    print('='*60)
    print(f"Tests Passed: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 