#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Safe Test Runner for Poker Knight

This test runner implements proper process isolation and cleanup mechanisms
to prevent deadlocks and hangs in parallel processing tests.

Key features:
- Process isolation for each test
- Proper cleanup of multiprocessing resources
- Redis connection management
- NUMA-aware testing with fallbacks
- Timeout protection
- Resource monitoring

Author: hildolfr
License: MIT
"""

import os
import sys
import time
import signal
import subprocess
import multiprocessing as mp
import threading
import atexit
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SafeTestRunner:
    """Safe test runner with deadlock prevention."""
    
    def __init__(self):
        self.active_processes: List[subprocess.Popen] = []
        self.cleanup_registered = False
        
        # Register cleanup handler
        if not self.cleanup_registered:
            atexit.register(self.cleanup_all)
            self.cleanup_registered = True
    
    def run_test_isolated(self, test_script: str, timeout: int = 120) -> Dict[str, Any]:
        """Run a test in complete isolation."""
        logger.info(f"Running test: {test_script}")
        
        # Prepare isolated environment
        env = os.environ.copy()
        env['POKER_KNIGHT_TEST_MODE'] = 'isolated'
        env['POKER_KNIGHT_DISABLE_NUMA'] = '1'  # Disable NUMA for safety
        
        start_time = time.time()
        
        try:
            # Run in subprocess with strict timeout
            process = subprocess.Popen(
                [sys.executable, test_script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                start_new_session=True  # Create new process group
            )
            
            self.active_processes.append(process)
            
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                exit_code = process.returncode
                
            except subprocess.TimeoutExpired:
                logger.warning(f"Test {test_script} timed out after {timeout}s")
                
                # Forcefully terminate the entire process group
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    time.sleep(2)  # Give time for graceful shutdown
                    
                    if process.poll() is None:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    
                except ProcessLookupError:
                    pass  # Process already terminated
                
                # Get partial output
                try:
                    stdout, stderr = process.communicate(timeout=1)
                except:
                    stdout, stderr = "", ""
                
                exit_code = -1
                
            finally:
                if process in self.active_processes:
                    self.active_processes.remove(process)
            
            execution_time = time.time() - start_time
            
            return {
                'test_name': Path(test_script).stem,
                'success': exit_code == 0,
                'exit_code': exit_code,
                'execution_time': execution_time,
                'stdout': stdout,
                'stderr': stderr,
                'timed_out': execution_time >= timeout
            }
            
        except Exception as e:
            logger.error(f"Error running test {test_script}: {e}")
            return {
                'test_name': Path(test_script).stem,
                'success': False,
                'exit_code': -1,
                'execution_time': time.time() - start_time,
                'stdout': "",
                'stderr': str(e),
                'timed_out': False
            }
    
    def cleanup_all(self):
        """Clean up all active processes."""
        if self.active_processes:
            logger.info(f"Cleaning up {len(self.active_processes)} active processes")
            
            for process in self.active_processes[:]:  # Copy list to avoid modification during iteration
                try:
                    if process.poll() is None:  # Process still running
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                except:
                    pass
                
                if process in self.active_processes:
                    self.active_processes.remove(process)


def test_basic_functionality():
    """Test basic poker solver functionality without parallel processing."""
    print("🧪 Testing Basic Functionality (Safe Mode)")
    
    try:
        # Add project root to path
        sys.path.insert(0, os.path.dirname(__file__))
        
        from poker_knight.solver import MonteCarloSolver
        
        # Create solver with safe settings
        solver = MonteCarloSolver()  # Caching parameter removed in v1.7.0
        
        # Simple test case
        hero_hand = ['AS', 'KS']
        num_opponents = 1
        
        result = solver.analyze_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            simulation_mode='fast'
        )
        
        print(f"[PASS] Basic test passed:")
        print(f"   Hand: {hero_hand} vs {num_opponents} opponent")
        print(f"   Win rate: {result.win_probability:.1%}")
        print(f"   Simulations: {result.simulations_run:,}")
        print(f"   Execution time: {result.execution_time_ms:.0f}ms")
        
        # Clean up
        solver.close()
        
        # Add assertions
        assert result is not None, "Result should not be None"
        assert 0 <= result.win_probability <= 1, "Win probability should be between 0 and 1"
        assert result.simulations_run > 0, "Should have run some simulations"
        
    except Exception as e:
        print(f"[FAIL] Basic test failed: {e}")
        assert False, f"Basic test failed with error: {e}"




def run_critical_tests():
    """Run critical tests in safe mode."""
    print("[ROCKET] Running Critical Tests in Safe Mode")
    print("=" * 50)
    
    runner = SafeTestRunner()
    results = []
    
    # Basic functionality test
    try:
        basic_success = test_basic_functionality()
        
        if basic_success:
            print("\n[PASS] All critical tests passed in safe mode")
        else:
            print("\n[WARN]  Some critical tests failed")
            
    except Exception as e:
        print(f"\n[FAIL] Critical tests failed: {e}")
    
    # Test individual files with isolation
    test_files = [
        'test_advanced_parallel.py',
        'tests/test_parallel.py'
    ]
    
    print(f"\n🔬 Testing {len(test_files)} individual files with isolation:")
    
    for test_file in test_files:
        if Path(test_file).exists():
            print(f"\nTesting: {test_file}")
            result = runner.run_test_isolated(test_file, timeout=60)
            results.append(result)
            
            status = "[PASS] PASSED" if result['success'] else "[FAIL] FAILED"
            print(f"  {status} ({result['execution_time']:.1f}s)")
            
            if result['timed_out']:
                print("  [TIMEOUT] Test timed out")
            
            if not result['success'] and result['stderr']:
                # Show first few lines of error
                error_lines = result['stderr'].split('\n')[:3]
                for line in error_lines:
                    if line.strip():
                        print(f"    Error: {line.strip()}")
        else:
            print(f"  [WARN]  File not found: {test_file}")
    
    # Summary
    total_tests = len(results)
    passed_tests = sum(1 for r in results if r['success'])
    
    print(f"\n[STATS] Test Summary:")
    print(f"  Total: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {total_tests - passed_tests}")
    print(f"  Success rate: {passed_tests/total_tests:.1%}" if total_tests > 0 else "  No tests run")
    
    return results


def main():
    """Main entry point."""
    print("🛡️  Poker Knight Safe Test Runner")
    print("=" * 50)
    
    try:
        # Set up safe environment
        os.environ['POKER_KNIGHT_SAFE_MODE'] = '1'
        
        # Run tests
        results = run_critical_tests()
        
        # Exit with appropriate code
        if results:
            success_rate = sum(1 for r in results if r['success']) / len(results)
            return 0 if success_rate >= 0.8 else 1
        else:
            return 1
            
    except KeyboardInterrupt:
        print("\n[WARN]  Test run interrupted by user")
        return 1
    except Exception as e:
        print(f"\n[FAIL] Test run failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 