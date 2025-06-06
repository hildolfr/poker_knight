#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker Knight Deadlock and Hang Analyzer

Systematic testing framework to identify and resolve deadlocks and hangs
in parallel processing tests. This tool:

1. Runs each test file individually with timeouts
2. Records detailed logs for each test
3. Captures system resource usage during tests
4. Identifies specific deadlock patterns
5. Provides comprehensive analysis and recommendations

Author: hildolfr
License: MIT
"""

import os
import sys
import time
import json
import subprocess
import threading
import queue
import signal
import psutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from collections import Counter
import traceback
import multiprocessing as mp

# Set up comprehensive logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('deadlock_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Result of running a single test."""
    test_name: str
    test_file: str
    status: str  # "success", "timeout", "error", "deadlock"
    execution_time: float
    exit_code: Optional[int]
    stdout: str
    stderr: str
    error_message: Optional[str]
    system_stats: Dict[str, Any]
    deadlock_indicators: List[str]
    timestamp: datetime


@dataclass
class SystemStats:
    """System resource statistics during test execution."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    processes_count: int
    threads_count: int
    open_files: int
    network_connections: int
    redis_running: bool


class DeadlockAnalyzer:
    """Main deadlock analysis engine."""
    
    def __init__(self, timeout_seconds: int = 300):
        self.timeout_seconds = timeout_seconds
        self.results: List[TestResult] = []
        self.analysis_log = []
        self.start_time = datetime.now()
        
        # Create results directory
        self.results_dir = Path("deadlock_analysis_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Test discovery
        self.test_files = self._discover_test_files()
        
        logger.info(f"Initialized DeadlockAnalyzer with {len(self.test_files)} test files")
        logger.info(f"Timeout set to {timeout_seconds} seconds per test")
    
    def _discover_test_files(self) -> List[Tuple[str, str]]:
        """Discover all test files in the project."""
        test_files = []
        
        # Root directory test files
        root_tests = [
            ("test_advanced_parallel.py", "Advanced Parallel Processing"),
            ("test_force_advanced_parallel.py", "Force Advanced Parallel"),
            ("test_cache_with_redis_demo.py", "Redis Cache Demo"),
            ("test_cache_warming_demo.py", "Cache Warming Demo"),
            ("test_cache_prepopulation_demo.py", "Cache Prepopulation Demo"),
            ("test_redis_integration.py", "Redis Integration"),
            ("test_redis_simple.py", "Simple Redis Test"),
            ("test_redis_vs_sqlite_demo.py", "Redis vs SQLite Demo"),
            ("test_sqlite_fallback_demo.py", "SQLite Fallback Demo"),
            ("test_sqlite_integration.py", "SQLite Integration"),
            ("test_solver_caching_integration.py", "Solver Caching Integration"),
            ("test_fallback_demo.py", "Fallback Demo"),
            ("test_caching.py", "Basic Caching"),
            ("test_caching_debug.py", "Caching Debug"),
            ("debug_parallel.py", "Debug Parallel Processing")
        ]
        
        for filename, description in root_tests:
            filepath = Path(filename)
            if filepath.exists():
                test_files.append((str(filepath), description))
        
        # Tests directory
        tests_dir = Path("tests")
        if tests_dir.exists():
            test_patterns = [
                ("test_parallel.py", "Basic Parallel Tests"),
                ("test_numa.py", "NUMA Tests"),
                ("test_cache_integration.py", "Cache Integration"),
                ("test_storage_cache_v2.py", "Storage Cache"),
                ("test_cache_population.py", "Cache Population"),
                ("test_poker_solver.py", "Poker Solver Tests"),
                ("test_statistical_validation.py", "Statistical Validation"),
                ("test_stress_scenarios.py", "Stress Scenarios"),
                ("test_performance_regression.py", "Performance Regression"),
                ("test_multi_way_scenarios.py", "Multi-way Scenarios")
            ]
            
            for filename, description in test_patterns:
                filepath = tests_dir / filename
                if filepath.exists():
                    test_files.append((str(filepath), description))
        
        return test_files
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all discovered tests with comprehensive analysis."""
        logger.info("Starting comprehensive deadlock analysis")
        logger.info(f"Found {len(self.test_files)} test files to analyze")
        
        # Record initial system state
        initial_stats = self._capture_system_stats()
        logger.info(f"Initial system state: CPU {initial_stats.cpu_percent}%, "
                   f"Memory {initial_stats.memory_percent}% ({initial_stats.memory_mb}MB)")
        
        # Run tests one by one
        for i, (test_file, description) in enumerate(self.test_files, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Test {i}/{len(self.test_files)}: {description}")
            logger.info(f"File: {test_file}")
            logger.info(f"{'='*60}")
            
            result = self._run_single_test(test_file, description)
            self.results.append(result)
            
            # Log result immediately
            self._log_test_result(result)
            
            # Brief pause between tests
            time.sleep(2.0)
        
        # Generate final analysis
        analysis = self._generate_analysis()
        
        # Save comprehensive results
        self._save_results(analysis)
        
        return analysis
    
    def _run_single_test(self, test_file: str, description: str) -> TestResult:
        """Run a single test with timeout and monitoring."""
        logger.info(f"Starting test: {description}")
        
        start_time = time.time()
        test_name = Path(test_file).stem
        
        # Prepare result object
        result = TestResult(
            test_name=test_name,
            test_file=test_file,
            status="unknown",
            execution_time=0.0,
            exit_code=None,
            stdout="",
            stderr="",
            error_message=None,
            system_stats={},
            deadlock_indicators=[],
            timestamp=datetime.now()
        )
        
        try:
            # Start system monitoring
            monitor_queue = queue.Queue()
            monitor_thread = threading.Thread(
                target=self._monitor_system_resources,
                args=(monitor_queue, test_name),
                daemon=True
            )
            monitor_thread.start()
            
            # Run the test with timeout
            process_result = self._execute_test_process(test_file)
            
            # Stop monitoring
            monitor_queue.put("STOP")
            
            # Process results
            result.execution_time = time.time() - start_time
            result.exit_code = process_result['exit_code']
            result.stdout = process_result['stdout']
            result.stderr = process_result['stderr']
            result.error_message = process_result['error_message']
            
            # Determine status
            if process_result['timed_out']:
                result.status = "timeout"
                result.deadlock_indicators.append("Process timed out")
            elif result.exit_code == 0:
                result.status = "success"
            else:
                result.status = "error"
            
            # Analyze for deadlock indicators
            result.deadlock_indicators.extend(
                self._analyze_deadlock_indicators(result.stdout, result.stderr)
            )
            
            # Capture final system stats
            result.system_stats = asdict(self._capture_system_stats())
            
        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            result.execution_time = time.time() - start_time
            logger.error(f"Exception running test {test_name}: {e}")
            logger.error(traceback.format_exc())
        
        return result
    
    def _execute_test_process(self, test_file: str) -> Dict[str, Any]:
        """Execute test in separate process with timeout."""
        logger.info(f"Executing: python {test_file}")
        
        try:
            # Use subprocess with timeout
            process = subprocess.Popen(
                [sys.executable, test_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                env=os.environ.copy()
            )
            
            try:
                stdout, stderr = process.communicate(timeout=self.timeout_seconds)
                return {
                    'exit_code': process.returncode,
                    'stdout': stdout,
                    'stderr': stderr,
                    'timed_out': False,
                    'error_message': None
                }
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Test timed out after {self.timeout_seconds} seconds")
                
                # Try to terminate gracefully
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    process.kill()
                    process.wait()
                
                # Get partial output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                except:
                    stdout, stderr = "", ""
                
                return {
                    'exit_code': -1,
                    'stdout': stdout,
                    'stderr': stderr,
                    'timed_out': True,
                    'error_message': f"Test timed out after {self.timeout_seconds} seconds"
                }
                
        except Exception as e:
            return {
                'exit_code': -1,
                'stdout': "",
                'stderr': str(e),
                'timed_out': False,
                'error_message': f"Failed to execute test: {e}"
            }
    
    def _monitor_system_resources(self, monitor_queue: queue.Queue, test_name: str):
        """Monitor system resources during test execution."""
        logger.debug(f"Starting resource monitoring for {test_name}")
        
        while True:
            try:
                # Check for stop signal
                try:
                    message = monitor_queue.get_nowait()
                    if message == "STOP":
                        break
                except queue.Empty:
                    pass
                
                # Capture stats
                stats = self._capture_system_stats()
                
                # Log if concerning levels detected
                if stats.cpu_percent > 95:
                    logger.warning(f"High CPU usage: {stats.cpu_percent}%")
                if stats.memory_percent > 90:
                    logger.warning(f"High memory usage: {stats.memory_percent}%")
                if stats.processes_count > 200:
                    logger.warning(f"High process count: {stats.processes_count}")
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                break
        
        logger.debug(f"Stopped resource monitoring for {test_name}")
    
    def _capture_system_stats(self) -> SystemStats:
        """Capture current system resource statistics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            # Process counts
            processes = list(psutil.process_iter(['pid', 'name']))
            process_count = len(processes)
            
            # Thread count (approximate)
            thread_count = sum(
                p.info.get('num_threads', 0) for p in psutil.process_iter(['num_threads'])
                if p.info.get('num_threads') is not None
            )
            
            # Open files
            try:
                open_files = len(psutil.Process().open_files())
            except:
                open_files = 0
            
            # Network connections
            try:
                connections = len(psutil.net_connections())
            except:
                connections = 0
            
            # Check if Redis is running
            redis_running = any(
                'redis' in p.info.get('name', '').lower()
                for p in psutil.process_iter(['name'])
            )
            
            return SystemStats(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_mb=memory.used / (1024 * 1024),
                processes_count=process_count,
                threads_count=thread_count,
                open_files=open_files,
                network_connections=connections,
                redis_running=redis_running
            )
            
        except Exception as e:
            logger.error(f"Error capturing system stats: {e}")
            return SystemStats(0, 0, 0, 0, 0, 0, 0, False)
    
    def _analyze_deadlock_indicators(self, stdout: str, stderr: str) -> List[str]:
        """Analyze output for deadlock indicators."""
        indicators = []
        
        # Common deadlock patterns
        deadlock_patterns = [
            ("multiprocessing", "Multiprocessing deadlock"),
            ("threading", "Threading deadlock"),
            ("pool", "Process/thread pool deadlock"),
            ("queue", "Queue deadlock"),
            ("lock", "Lock contention"),
            ("redis", "Redis connection issue"),
            ("sqlite", "SQLite locking issue"),
            ("cache", "Cache system deadlock"),
            ("numa", "NUMA allocation issue"),
            ("timeout", "Operation timeout"),
            ("hang", "Process hang"),
            ("zombie", "Zombie process"),
            ("blocked", "Blocked operation")
        ]
        
        combined_output = (stdout + stderr).lower()
        
        for pattern, description in deadlock_patterns:
            if pattern in combined_output:
                indicators.append(description)
        
        # Check for specific error patterns
        error_patterns = [
            ("broken pipe", "Broken pipe error"),
            ("connection refused", "Connection refused"),
            ("resource temporarily unavailable", "Resource exhaustion"),
            ("cannot allocate memory", "Memory allocation failure"),
            ("too many open files", "File descriptor exhaustion"),
            ("operation not permitted", "Permission/resource issue")
        ]
        
        for pattern, description in error_patterns:
            if pattern in combined_output:
                indicators.append(description)
        
        return indicators
    
    def _log_test_result(self, result: TestResult):
        """Log the result of a single test."""
        status_emoji = {
            "success": "[PASS]",
            "timeout": "[TIMEOUT]",
            "error": "[FAIL]",
            "deadlock": "[LOCK]",
            "unknown": "[UNKNOWN]"
        }
        
        emoji = status_emoji.get(result.status, "[UNKNOWN]")
        logger.info(f"{emoji} {result.test_name}: {result.status.upper()}")
        logger.info(f"   Execution time: {result.execution_time:.2f}s")
        
        if result.exit_code is not None:
            logger.info(f"   Exit code: {result.exit_code}")
        
        if result.deadlock_indicators:
            logger.warning(f"   Deadlock indicators: {', '.join(result.deadlock_indicators)}")
        
        if result.error_message:
            logger.error(f"   Error: {result.error_message}")
        
        # Save individual test log
        test_log_file = self.results_dir / f"{result.test_name}_log.txt"
        with open(test_log_file, 'w') as f:
            f.write(f"Test: {result.test_name}\n")
            f.write(f"File: {result.test_file}\n")
            f.write(f"Status: {result.status}\n")
            f.write(f"Execution time: {result.execution_time:.2f}s\n")
            f.write(f"Exit code: {result.exit_code}\n")
            f.write(f"Timestamp: {result.timestamp}\n")
            f.write(f"\nDeadlock indicators:\n")
            for indicator in result.deadlock_indicators:
                f.write(f"  - {indicator}\n")
            f.write(f"\nSTDOUT:\n{result.stdout}\n")
            f.write(f"\nSTDERR:\n{result.stderr}\n")
            if result.error_message:
                f.write(f"\nError message:\n{result.error_message}\n")
    
    def _generate_analysis(self) -> Dict[str, Any]:
        """Generate comprehensive analysis of all test results."""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.status == "success")
        timeout_tests = sum(1 for r in self.results if r.status == "timeout")
        error_tests = sum(1 for r in self.results if r.status == "error")
        
        # Collect all deadlock indicators
        all_indicators = []
        for result in self.results:
            all_indicators.extend(result.deadlock_indicators)
        
        # Count indicator frequency
        indicator_counts = Counter(all_indicators)
        
        # Identify problematic tests
        problematic_tests = [r for r in self.results if r.status in ["timeout", "error"]]
        
        # Calculate average execution times
        successful_times = [r.execution_time for r in self.results if r.status == "success"]
        avg_success_time = sum(successful_times) / len(successful_times) if successful_times else 0
        
        analysis = {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "timeout_tests": timeout_tests,
                "error_tests": error_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0,
                "average_success_time": avg_success_time
            },
            "deadlock_indicators": dict(indicator_counts),
            "problematic_tests": [
                {
                    "name": r.test_name,
                    "file": r.test_file,
                    "status": r.status,
                    "indicators": r.deadlock_indicators,
                    "execution_time": r.execution_time
                }
                for r in problematic_tests
            ],
            "recommendations": self._generate_recommendations(problematic_tests, indicator_counts),
            "timestamp": datetime.now().isoformat(),
            "analysis_duration": (datetime.now() - self.start_time).total_seconds()
        }
        
        return analysis
    
    def _generate_recommendations(self, problematic_tests: List[TestResult], 
                                 indicator_counts: Counter) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []
        
        # Based on most common deadlock indicators
        if "Multiprocessing deadlock" in indicator_counts:
            recommendations.append(
                "[FIX] Add proper process cleanup and join() calls in multiprocessing code"
            )
        
        if "Redis connection issue" in indicator_counts:
            recommendations.append(
                "[FIX] Implement Redis connection pooling and timeout handling"
            )
        
        if "Process timeout" in indicator_counts:
            recommendations.append(
                "[FIX] Reduce simulation counts in tests or increase timeout values"
            )
        
        if "Lock contention" in indicator_counts:
            recommendations.append(
                "[FIX] Review locking mechanisms and reduce lock scope"
            )
        
        # Based on specific test patterns
        parallel_tests = [t for t in problematic_tests if "parallel" in t.test_name.lower()]
        if parallel_tests:
            recommendations.append(
                "[FIX] Review parallel processing initialization and cleanup procedures"
            )
        
        cache_tests = [t for t in problematic_tests if "cache" in t.test_name.lower()]
        if cache_tests:
            recommendations.append(
                "[FIX] Implement proper cache connection lifecycle management"
            )
        
        numa_tests = [t for t in problematic_tests if "numa" in t.test_name.lower()]
        if numa_tests:
            recommendations.append(
                "[FIX] Add NUMA availability checks and graceful fallbacks"
            )
        
        # General recommendations
        if len(problematic_tests) > len(self.results) * 0.5:
            recommendations.append(
                "[FIX] Consider isolating tests in separate processes with proper cleanup"
            )
        
        return recommendations
    
    def _save_results(self, analysis: Dict[str, Any]):
        """Save comprehensive results to files."""
        # Save analysis JSON
        analysis_file = self.results_dir / "deadlock_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Save detailed results
        results_file = self.results_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump([asdict(r) for r in self.results], f, indent=2, default=str)
        
        # Save summary report
        report_file = self.results_dir / "summary_report.txt"
        with open(report_file, 'w') as f:
            f.write("POKER KNIGHT DEADLOCK ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Analysis completed at: {analysis['timestamp']}\n")
            f.write(f"Total analysis time: {analysis['analysis_duration']:.1f} seconds\n\n")
            
            # Summary
            summary = analysis['summary']
            f.write("TEST SUMMARY:\n")
            f.write(f"  Total tests: {summary['total_tests']}\n")
            f.write(f"  Successful: {summary['successful_tests']}\n")
            f.write(f"  Timeouts: {summary['timeout_tests']}\n")
            f.write(f"  Errors: {summary['error_tests']}\n")
            f.write(f"  Success rate: {summary['success_rate']:.1%}\n")
            f.write(f"  Avg success time: {summary['average_success_time']:.2f}s\n\n")
            
            # Deadlock indicators
            if analysis['deadlock_indicators']:
                f.write("DEADLOCK INDICATORS:\n")
                for indicator, count in analysis['deadlock_indicators'].items():
                    f.write(f"  {indicator}: {count} occurrences\n")
                f.write("\n")
            
            # Problematic tests
            if analysis['problematic_tests']:
                f.write("PROBLEMATIC TESTS:\n")
                for test in analysis['problematic_tests']:
                    f.write(f"  {test['name']} ({test['status']})\n")
                    f.write(f"    File: {test['file']}\n")
                    f.write(f"    Time: {test['execution_time']:.2f}s\n")
                    if test['indicators']:
                        f.write(f"    Issues: {', '.join(test['indicators'])}\n")
                    f.write("\n")
            
            # Recommendations
            if analysis['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                for rec in analysis['recommendations']:
                    f.write(f"  {rec}\n")
                f.write("\n")
        
        logger.info(f"Results saved to {self.results_dir}")


def main():
    """Main entry point for deadlock analysis."""
    print("[SEARCH] Poker Knight Deadlock and Hang Analyzer")
    print("=" * 50)
    
    # Create analyzer with 5-minute timeout per test
    analyzer = DeadlockAnalyzer(timeout_seconds=300)
    
    try:
        # Run comprehensive analysis
        analysis = analyzer.run_all_tests()
        
        # Print summary
        print("\n[STATS] ANALYSIS COMPLETE")
        print("=" * 50)
        summary = analysis['summary']
        print(f"Tests run: {summary['total_tests']}")
        print(f"Successful: {summary['successful_tests']} ({summary['success_rate']:.1%})")
        print(f"Timeouts: {summary['timeout_tests']}")
        print(f"Errors: {summary['error_tests']}")
        
        if analysis['deadlock_indicators']:
            print("\n🚨 Most common issues:")
            for indicator, count in list(analysis['deadlock_indicators'].items())[:5]:
                print(f"  • {indicator}: {count} occurrences")
        
        if analysis['recommendations']:
            print("\n[IDEA] Top recommendations:")
            for rec in analysis['recommendations'][:3]:
                print(f"  {rec}")
        
        print(f"\n📁 Detailed results saved to: {analyzer.results_dir}")
        
        return 0 if summary['success_rate'] > 0.8 else 1
        
    except KeyboardInterrupt:
        print("\n[WARN]  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Analysis failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 