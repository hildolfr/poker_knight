#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures for Poker Knight test suite.
Provides custom command-line options and test categorization.
"""

import pytest
import json
import os
from datetime import datetime
from pathlib import Path


def pytest_addoption(parser):
    """Add custom command-line options to pytest."""
    
    # Test suite selection options
    parser.addoption(
        "--quick", action="store_true", default=False,
        help="Run quick validation tests only"
    )
    parser.addoption(
        "--statistical", action="store_true", default=False,
        help="Run statistical validation tests only"
    )
    parser.addoption(
        "--performance", action="store_true", default=False,
        help="Run performance and benchmark tests only"
    )
    parser.addoption(
        "--stress", action="store_true", default=False,
        help="Run stress and load tests only"
    )
    parser.addoption(
        "--unit", action="store_true", default=False,
        help="Run unit tests only"
    )
    parser.addoption(
        "--numa", action="store_true", default=False,
        help="Run NUMA (Non-Uniform Memory Access) tests only"
    )
    parser.addoption(
        "--all", action="store_true", default=False,
        help="Run all tests (comprehensive test suite)"
    )
    parser.addoption(
        "--failed", action="store_true", default=False,
        help="Run only the tests that failed in the last test run"
    )
    parser.addoption(
        "--with-coverage", action="store_true", default=False,
        help="Run tests with coverage analysis"
    )


def pytest_configure(config):
    """Configure pytest based on command-line options."""
    config.addinivalue_line(
        "markers", "numa: marks tests as NUMA (Non-Uniform Memory Access) related tests"
    )
    
    # Check for conflicting options
    test_type_options = [
        config.getoption("--quick"),
        config.getoption("--statistical"), 
        config.getoption("--performance"),
        config.getoption("--stress"),
        config.getoption("--unit"),
        config.getoption("--numa"),
        config.getoption("--all"),
        config.getoption("--failed")
    ]
    
    if sum(test_type_options) > 1:
        pytest.exit("Error: Only one test type option can be specified at a time")
    
    # Handle --failed option
    if config.getoption("--failed"):
        failed_tests = _get_failed_tests_from_last_run()
        if not failed_tests:
            pytest.exit("No failed tests found in recent test results. Run tests first to generate results.")
        
        # Add the specific failed test node IDs to pytest's collection
        # This ensures pytest runs only these specific tests, not entire files
        if not config.args:
            config.args = failed_tests
        else:
            config.args.extend(failed_tests)
            
        print(f"\n[Knight] Running {len(failed_tests)} specific failed tests from last run:")
        for test in failed_tests:
            print(f"   - {test}")
        print()
    
    # Add markers based on command-line options
    markexpr_parts = []
    
    if config.getoption("--quick"):
        markexpr_parts.append("quick")
    elif config.getoption("--statistical"):
        markexpr_parts.append("statistical")
    elif config.getoption("--performance"):
        markexpr_parts.append("performance or regression")
    elif config.getoption("--stress"):
        markexpr_parts.append("stress or slow")
    elif config.getoption("--unit"):
        markexpr_parts.append("unit")
    elif config.getoption("--numa"):
        markexpr_parts.append("numa")
    elif config.getoption("--all"):
        # Explicitly run all tests - no marker filtering
        pass
    elif config.getoption("--failed"):
        # Failed tests already handled above - no marker filtering needed
        pass
    
    # Set marker expression if any specific test type was requested
    if markexpr_parts:
        config.option.markexpr = " or ".join(markexpr_parts)
    
    # Enable coverage if requested
    if config.getoption("--with-coverage"):
        config.option.cov = ["poker_knight"]
        config.option.cov_report = ["term-missing", "html"]


def pytest_collection_modifyitems(config, items):
    """Modify collected items based on markers and options."""
    
    # Auto-mark tests based on filename patterns
    for item in items:
        # Mark NUMA tests
        if "test_numa.py" in item.nodeid:
            item.add_marker(pytest.mark.numa)
            
            # Add specific markers based on test class names
            if "TestNumaTopology" in item.nodeid or "TestNumaConfiguration" in item.nodeid:
                item.add_marker(pytest.mark.unit)
            elif "TestNumaIntegration" in item.nodeid:
                item.add_marker(pytest.mark.integration)
            elif "TestNumaPerformance" in item.nodeid:
                item.add_marker(pytest.mark.performance)
            elif "TestNumaEdgeCases" in item.nodeid:
                item.add_marker(pytest.mark.edge_cases)
        
        # Mark tests based on file names (existing functionality)
        elif "test_poker_solver.py" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        elif "test_statistical_validation.py" in item.nodeid:
            item.add_marker(pytest.mark.statistical)
        elif "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "test_stress" in item.nodeid:
            item.add_marker(pytest.mark.stress)
        elif "test_edge_cases" in item.nodeid:
            item.add_marker(pytest.mark.edge_cases)
        elif "test_validation.py" in item.nodeid:
            item.add_marker(pytest.mark.validation)
        elif "test_precision.py" in item.nodeid:
            item.add_marker(pytest.mark.precision)
        elif "test_parallel.py" in item.nodeid:
            item.add_marker(pytest.mark.parallel)
        elif "test_multi_way" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark quick tests (subset of fast-running tests)
        quick_tests = [
            "test_basic_functionality", "test_card_creation", "test_card_value",
            "test_hand_evaluation", "test_fast_mode_performance", "test_basic_edge_cases"
        ]
        
        if any(quick_test in item.nodeid for quick_test in quick_tests):
            item.add_marker(pytest.mark.quick)


@pytest.fixture(scope="session")
def poker_test_config():
    """Provide test configuration for poker knight tests."""
    return {
        "simulation_modes": ["fast", "default", "precision"],
        "test_hands": {
            "strong": ["AS", "AH"],
            "medium": ["KS", "QS"],
            "weak": ["7S", "2D"]
        },
        "opponent_counts": [1, 2, 4, 8]
    }


def _get_test_type(config):
    """Determine the test type from configuration options."""
    if config.getoption("--quick"):
        return "quick"
    elif config.getoption("--statistical"):
        return "statistical"
    elif config.getoption("--performance"):
        return "performance"
    elif config.getoption("--stress"):
        return "stress"
    elif config.getoption("--unit"):
        return "unit"
    elif config.getoption("--numa"):
        return "numa"
    elif config.getoption("--all"):
        return "all"
    elif config.getoption("--failed"):
        return "failed"
    else:
        return "all"  # Default behavior


def _get_failed_tests_from_last_run():
    """Get the list of failed test node IDs from the most recent test results file."""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        return []
    
    # Find the most recent results file
    json_files = list(results_dir.glob("test_results_*.json"))
    if not json_files:
        return []
    
    # Sort by modification time, most recent first
    latest_file = max(json_files, key=lambda f: f.stat().st_mtime)
    
    try:
        with open(latest_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # Extract failed test node IDs (full paths including class and method names)
        failed_tests = []
        for test in results.get("failed_tests", []):
            test_name = test.get("name", "")
            if test_name:
                failed_tests.append(test_name)
        
        return failed_tests
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not read test results from {latest_file}: {e}")
        return []


def _write_test_results(config, stats, exitstatus):
    """Write test results to a timestamped file."""
    
    # Get test type and generate timestamp
    test_type = _get_test_type(config)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create results directory if it doesn't exist
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Generate filename
    filename = f"test_results_{timestamp}_{test_type}.json"
    filepath = results_dir / filename
    
    # Collect test statistics
    passed = len(stats.get('passed', []))
    failed = len(stats.get('failed', []))
    errors = len(stats.get('error', []))
    skipped = len(stats.get('skipped', []))
    total = passed + failed + errors + skipped
    
    # Collect failed test details
    failed_tests = []
    for test_report in stats.get('failed', []):
        failed_tests.append({
            "name": test_report.nodeid,
            "outcome": test_report.outcome,
            "duration": getattr(test_report, 'duration', 0),
            "longrepr": str(test_report.longrepr) if test_report.longrepr else None
        })
    
    # Collect error test details
    error_tests = []
    for test_report in stats.get('error', []):
        error_tests.append({
            "name": test_report.nodeid,
            "outcome": test_report.outcome,
            "duration": getattr(test_report, 'duration', 0),
            "longrepr": str(test_report.longrepr) if test_report.longrepr else None
        })
    
    # Collect skipped test details
    skipped_tests = []
    for test_report in stats.get('skipped', []):
        skipped_tests.append({
            "name": test_report.nodeid,
            "outcome": test_report.outcome,
            "duration": getattr(test_report, 'duration', 0),
            "reason": str(test_report.longrepr) if test_report.longrepr else None
        })
    
    # Prepare result data
    result_data = {
        "timestamp": timestamp,
        "test_type": test_type,
        "test_suite": "Poker Knight Monte Carlo Solver",
        "summary": {
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "skipped": skipped,
            "success_rate": (passed / (passed + failed) * 100) if (passed + failed) > 0 else 0,
            "exit_status": exitstatus
        },
        "configuration": {
            "quick": config.getoption("--quick"),
            "statistical": config.getoption("--statistical"),
            "performance": config.getoption("--performance"),
            "stress": config.getoption("--stress"),
            "unit": config.getoption("--unit"),
            "numa": config.getoption("--numa"),
            "all": config.getoption("--all"),
            "with_coverage": config.getoption("--with-coverage")
        },
        "failed_tests": failed_tests,
        "error_tests": error_tests,
        "skipped_tests": skipped_tests,
        "analysis": {
            "overall_status": "PASS" if failed == 0 and errors == 0 else "FAIL",
            "critical_failures": failed + errors,
            "regression_indicators": {
                "has_failures": failed > 0,
                "has_errors": errors > 0,
                "success_rate_below_threshold": (passed / (passed + failed) * 100) < 95 if (passed + failed) > 0 else False
            }
        }
    }
    
    # Write to file
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    except Exception as e:
        # Fallback: write to a simple text file if JSON fails
        txt_filepath = results_dir / f"test_results_{timestamp}_{test_type}.txt"
        with open(txt_filepath, 'w', encoding='utf-8') as f:
            f.write(f"Poker Knight Test Results - {test_type.upper()}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total: {total}, Passed: {passed}, Failed: {failed}, Errors: {errors}, Skipped: {skipped}\n")
            f.write(f"Success Rate: {(passed / total * 100):.1f}%\n")
            f.write(f"Exit Status: {exitstatus}\n")
            f.write(f"Overall Status: {'PASS' if failed == 0 and errors == 0 else 'FAIL'}\n")
            if failed_tests:
                f.write(f"\nFailed Tests:\n")
                for test in failed_tests:
                    f.write(f"  - {test['name']}\n")
            if error_tests:
                f.write(f"\nError Tests:\n")
                for test in error_tests:
                    f.write(f"  - {test['name']}\n")
        
        return txt_filepath


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom summary information to test output."""
    
    if hasattr(terminalreporter, 'stats'):
        # Determine test type for summary
        test_type = "Comprehensive Test Suite"
        if config.getoption("--quick"):
            test_type = "Quick Validation"
        elif config.getoption("--statistical"):
            test_type = "Statistical Validation"
        elif config.getoption("--performance"):
            test_type = "Performance Testing"
        elif config.getoption("--stress"):
            test_type = "Stress Testing"
        elif config.getoption("--unit"):
            test_type = "Unit Testing"
        elif config.getoption("--numa"):
            test_type = "NUMA Testing"
        elif config.getoption("--all"):
            test_type = "Complete Test Suite (--all)"
        
        # Add beautiful summary section
        terminalreporter.write_sep("=", f"🎯 Poker Knight Test Summary - {test_type}")
        
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        errors = len(terminalreporter.stats.get('error', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        total = passed + failed + errors + skipped
        
        if total > 0:
            # Calculate success rate excluding skipped tests
            tests_run = passed + failed
            success_rate = (passed / tests_run) * 100 if tests_run > 0 else 0
            terminalreporter.write(f"♞ Tests: {passed} passed, {failed} failed, {errors} errors, {skipped} skipped\n")
            terminalreporter.write(f"♞ Success Rate: {success_rate:.1f}% (excluding skipped)\n")
            
            if failed == 0 and errors == 0:
                terminalreporter.write("🎉 All tests passed!\n")
            else:
                terminalreporter.write("[WARN]  Some tests failed - see details above\n")
        
        # Show coverage info if available
        if config.getoption("--with-coverage"):
            terminalreporter.write("[STATS] Coverage report generated in htmlcov/\n")
        
        # Write results to file
        try:
            result_file = _write_test_results(config, terminalreporter.stats, exitstatus)
            terminalreporter.write(f"📝 Results logged to: {result_file.name}\n")
        except Exception as e:
            terminalreporter.write(f"[WARN]  Failed to write results file: {e}\n") 