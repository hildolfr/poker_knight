#!/usr/bin/env python3
"""
Poker Knight Test Suite Integration Validator

This script validates that the cache tests are properly integrated into
the main test suite and that all markers and options work correctly.
"""

import subprocess
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple


def run_command(cmd: List[str], description: str) -> Tuple[bool, str, str]:
    """Run a command and return success status, stdout, and stderr."""
    print(f"[Test] {description}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60,
            cwd=Path(__file__).parent,
            encoding='utf-8',
            errors='replace'
        )
        
        success = result.returncode == 0
        status = "[PASS]" if success else "[FAIL]"
        print(f"   Result: {status}")
        
        if not success:
            print(f"   Error: {result.stderr.strip()}")
        
        return success, result.stdout, result.stderr
    
    except subprocess.TimeoutExpired:
        print("   Result: [TIMEOUT]")
        return False, "", "Command timed out"
    except Exception as e:
        print(f"   Result: [ERROR] - {e}")
        return False, "", str(e)


def test_marker_functionality() -> Dict[str, bool]:
    """Test that all markers work correctly."""
    print("\n[Markers] Testing Marker Functionality")
    print("=" * 50)
    
    marker_tests = {
        # Cache markers
        "cache": ["python", "-m", "pytest", "-m", "cache", "--collect-only", "-q"],
        "cache_unit": ["python", "-m", "pytest", "-m", "cache_unit", "--collect-only", "-q"],
        "cache_integration": ["python", "-m", "pytest", "-m", "cache_integration", "--collect-only", "-q"],
        "cache_performance": ["python", "-m", "pytest", "-m", "cache_performance", "--collect-only", "-q"],
        "redis_required": ["python", "-m", "pytest", "-m", "redis_required", "--collect-only", "-q"],
        
        # Existing markers
        "unit": ["python", "-m", "pytest", "-m", "unit", "--collect-only", "-q"],
        "quick": ["python", "-m", "pytest", "-m", "quick", "--collect-only", "-q"],
        "statistical": ["python", "-m", "pytest", "-m", "statistical", "--collect-only", "-q"],
        "performance": ["python", "-m", "pytest", "-m", "performance", "--collect-only", "-q"],
        
        # Marker combinations
        "cache_and_quick": ["python", "-m", "pytest", "-m", "cache and quick", "--collect-only", "-q"],
        "cache_not_redis": ["python", "-m", "pytest", "-m", "cache and not redis_required", "--collect-only", "-q"],
    }
    
    results = {}
    
    for marker_name, cmd in marker_tests.items():
        success, stdout, stderr = run_command(cmd, f"Testing marker: {marker_name}")
        results[marker_name] = success
        
        if success and stdout:
            # Count collected tests
            lines = stdout.strip().split('\n')
            test_count = len([line for line in lines if line.startswith('tests/')])
            print(f"   Collected: {test_count} tests")
    
    return results


def test_command_line_options() -> Dict[str, bool]:
    """Test that all command-line options work correctly."""
    print("\n[Options] Testing Command-Line Options")
    print("=" * 50)
    
    option_tests = {
        # Cache options
        "cache": ["python", "-m", "pytest", "--cache", "--collect-only", "-q"],
        "cache-unit": ["python", "-m", "pytest", "--cache-unit", "--collect-only", "-q"],
        "cache-integration": ["python", "-m", "pytest", "--cache-integration", "--collect-only", "-q"],
        "cache-performance": ["python", "-m", "pytest", "--cache-performance", "--collect-only", "-q"],
        "redis": ["python", "-m", "pytest", "--redis", "--collect-only", "-q"],
        
        # Existing options
        "quick": ["python", "-m", "pytest", "--quick", "--collect-only", "-q"],
        "unit": ["python", "-m", "pytest", "--unit", "--collect-only", "-q"],
        "statistical": ["python", "-m", "pytest", "--statistical", "--collect-only", "-q"],
        "performance": ["python", "-m", "pytest", "--performance", "--collect-only", "-q"],
        "all": ["python", "-m", "pytest", "--all", "--collect-only", "-q"],
    }
    
    results = {}
    
    for option_name, cmd in option_tests.items():
        success, stdout, stderr = run_command(cmd, f"Testing option: --{option_name}")
        results[option_name] = success
        
        if success and stdout:
            # Count collected tests
            lines = stdout.strip().split('\n')
            test_count = len([line for line in lines if line.startswith('tests/')])
            print(f"   Collected: {test_count} tests")
    
    return results


def test_cache_test_execution() -> Dict[str, bool]:
    """Test actual execution of cache tests."""
    print("\n[Execute] Testing Cache Test Execution")
    print("=" * 50)
    
    execution_tests = {
        # Quick cache tests (should be fast)
        "cache_quick": ["python", "-m", "pytest", "-m", "cache and quick", "-v", "--tb=short"],
        
        # Demo integration test
        "integration_demo": ["python", "-m", "pytest", "tests/test_integration_demo.py", "-v"],
        
        # Cache unit tests (fast subset)
        "cache_unit_basic": ["python", "-m", "pytest", "-m", "cache_unit", "-k", "test_cache_config", "-v"],
    }
    
    results = {}
    
    for test_name, cmd in execution_tests.items():
        success, stdout, stderr = run_command(cmd, f"Executing: {test_name}")
        results[test_name] = success
        
        if success:
            # Look for test results in output
            if "passed" in stdout or "PASSED" in stdout:
                print(f"   Tests executed successfully")
            elif "collected 0 items" in stdout:
                print(f"   No tests collected (may be expected)")
    
    return results


def test_test_runner_wrapper() -> Dict[str, bool]:
    """Test the run_tests.py wrapper script."""
    print("\n[Wrapper] Testing run_tests.py Wrapper")
    print("=" * 50)
    
    wrapper_tests = {
        "help": ["python", "tests/run_tests.py", "--help"],
        "cache_option": ["python", "tests/run_tests.py", "--cache", "--collect-only", "-q"],
        "cache_unit_option": ["python", "tests/run_tests.py", "--cache-unit", "--collect-only", "-q"],
    }
    
    results = {}
    
    for test_name, cmd in wrapper_tests.items():
        success, stdout, stderr = run_command(cmd, f"Testing wrapper: {test_name}")
        results[test_name] = success
        
        if test_name == "help" and success:
            if "Cache Test Organization" in stdout:
                print(f"   Help text includes cache documentation")
    
    return results


def validate_configuration_files() -> Dict[str, bool]:
    """Validate that configuration files are properly set up."""
    print("\n[Config] Validating Configuration Files")
    print("=" * 50)
    
    results = {}
    
    # Check pytest.ini
    pytest_ini_path = Path("pytest.ini")
    if pytest_ini_path.exists():
        content = pytest_ini_path.read_text(encoding='utf-8', errors='replace')
        cache_markers_present = all(marker in content for marker in [
            "cache:", "cache_unit:", "cache_integration:", 
            "cache_persistence:", "cache_performance:", "redis_required:"
        ])
        results["pytest_ini_markers"] = cache_markers_present
        print(f"   pytest.ini cache markers: [PRESENT]" if cache_markers_present else "   pytest.ini cache markers: [MISSING]")
    else:
        results["pytest_ini_markers"] = False
        print("   pytest.ini: [NOT FOUND]")
    
    # Check conftest.py
    conftest_path = Path("tests/conftest.py")
    if conftest_path.exists():
        content = conftest_path.read_text(encoding='utf-8', errors='replace')
        cache_options_present = all(option in content for option in [
            "--cache", "--cache-unit", "--cache-integration", 
            "--cache-performance", "--redis"
        ])
        results["conftest_cache_options"] = cache_options_present
        print(f"   conftest.py cache options: [PRESENT]" if cache_options_present else "   conftest.py cache options: [MISSING]")
    else:
        results["conftest_cache_options"] = False
        print("   conftest.py: [NOT FOUND]")
    
    # Check that test_cache_markers.py was removed
    cache_markers_path = Path("tests/test_cache_markers.py")
    results["redundant_markers_removed"] = not cache_markers_path.exists()
    print(f"   Redundant markers file removed: [YES]" if not cache_markers_path.exists() else "   Redundant markers file removed: [NO]")
    
    return results


def main():
    """Main validation function."""
    print("[Integration] Poker Knight Test Suite Integration Validation")
    print("=" * 60)
    print()
    
    all_results = {}
    
    # Run all validation tests
    all_results["markers"] = test_marker_functionality()
    all_results["options"] = test_command_line_options()
    all_results["execution"] = test_cache_test_execution()
    all_results["wrapper"] = test_test_runner_wrapper()
    all_results["config"] = validate_configuration_files()
    
    # Generate summary report
    print("\n[Summary] Summary Report")
    print("=" * 50)
    
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        category_total = len(results)
        category_passed = sum(1 for success in results.values() if success)
        
        total_tests += category_total
        passed_tests += category_passed
        
        status = "[PASS]" if category_passed == category_total else "[FAIL]"
        print(f"{category.upper():15} {status} ({category_passed}/{category_total})")
        
        # Show failed tests
        failed_tests = [test for test, success in results.items() if not success]
        if failed_tests:
            for test in failed_tests:
                print(f"                  [X] {test}")
    
    print(f"\nOVERALL RESULT: [PASS] ({passed_tests}/{total_tests})" if passed_tests == total_tests else f"\nOVERALL RESULT: [FAIL] ({passed_tests}/{total_tests})")
    
    if passed_tests == total_tests:
        print("\n[Success] Cache test integration is working correctly!")
        print("   You can now use the unified test suite with cache tests.")
        print("\n   Try these commands:")
        print("   - pytest --cache          # All cache tests")
        print("   - pytest --cache-unit     # Fast cache unit tests")
        print("   - pytest --quick          # Quick validation including cache")
        print("   - pytest -m cache         # Cache tests via marker")
        return 0
    else:
        print("\n[Warning] Some integration issues were found.")
        print("   Please review the failed tests and fix any configuration issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 