#!/usr/bin/env python3
"""
Poker Knight Test Runner

Enhanced test runner with detailed progress feedback and timing information.
"""

import sys
import subprocess
import argparse
import os
import time
import threading
from datetime import datetime, timedelta

class ProgressIndicator:
    """Shows progress for long-running tests."""
    
    def __init__(self, description):
        self.description = description
        self.start_time = time.time()
        self.running = True
        self.thread = None
    
    def _show_progress(self):
        """Show spinning progress indicator."""
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧']
        i = 0
        while self.running:
            elapsed = time.time() - self.start_time
            mins, secs = divmod(elapsed, 60)
            sys.stdout.write(f'\r{spinner[i % len(spinner)]} {self.description} - Running for {int(mins):02d}:{int(secs):02d}')
            sys.stdout.flush()
            time.sleep(0.1)
            i += 1
    
    def start(self):
        """Start the progress indicator."""
        self.thread = threading.Thread(target=self._show_progress, daemon=True)
        self.thread.start()
    
    def stop(self):
        """Stop the progress indicator."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        elapsed = time.time() - self.start_time
        mins, secs = divmod(elapsed, 60)
        sys.stdout.write(f'\r✓ {self.description} - Completed in {int(mins):02d}:{int(secs):02d}\n')
        sys.stdout.flush()

def estimate_test_duration(test_type):
    """Provide duration estimates for different test types."""
    estimates = {
        "unit": "~1-2 minutes",
        "statistical": "~25-30 minutes (long statistical validation)",
        "performance": "~2-3 minutes (benchmarking hand evaluation)",
        "regression": "~5-7 minutes (performance consistency checks)",
        "validation": "~30 seconds (quick sanity checks)"
    }
    return estimates.get(test_type, "~unknown duration")

def run_command_with_progress(cmd, description, test_type=None):
    """Run a command with enhanced progress feedback."""
    print(f"\n{'='*80}")
    print(f"🧪 {description}")
    if test_type:
        estimate = estimate_test_duration(test_type)
        print(f"📊 Estimated duration: {estimate}")
    print(f"🕒 Started at: {datetime.now().strftime('%H:%M:%S')}")
    print('='*80)
    
    start_time = time.time()
    
    # Start progress indicator for long-running tests
    progress = None
    show_real_time = False
    
    if test_type in ["statistical", "regression"]:
        # Long tests: use spinner, no real-time output
        progress = ProgressIndicator(f"Executing {description}")
        progress.start()
        print()  # New line after header
    else:
        # Short tests: show real-time output, no spinner
        show_real_time = True
    
    try:
        # For pytest commands, add more verbose output
        if "pytest" in cmd:
            cmd += " --tb=short -q"
        
        # Run with real-time output for better feedback
        process = subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        output_lines = []
        shown_lines = set()  # Track what we've already shown to prevent duplicates
        
        # Read output in real-time
        while True:
            line = process.stdout.readline()
            if line:
                output_lines.append(line)
                
                # Show real-time output for short tests only
                if show_real_time:
                    # Show important lines immediately
                    line_key = line.strip()
                    if line_key and line_key not in shown_lines:
                        if any(keyword in line.lower() for keyword in ['passed', 'failed', 'error', 'collecting', 'running', 'example']):
                            print(f"  {line.strip()}")
                            shown_lines.add(line_key)
            elif process.poll() is not None:
                break
        
        # Stop progress indicator
        if progress:
            progress.stop()
        
        # Get return code
        return_code = process.poll()
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        mins, secs = divmod(elapsed_time, 60)
        
        # Show output summary ONLY if we didn't show real-time output
        if not show_real_time:
            full_output = ''.join(output_lines)
            
            if "pytest" in cmd:
                # Show only pytest summary lines
                lines = full_output.strip().split('\n')
                for line in lines[-15:]:  # Show last 15 lines for better context
                    if any(keyword in line for keyword in ['passed', 'failed', 'PASSED', 'FAILED', '=====']):
                        print(f"  {line}")
            else:
                # For non-pytest long commands, show condensed summary
                lines = full_output.strip().split('\n')
                # Show first few and last few lines to give context without full duplication
                if len(lines) > 20:
                    print("  ... (output condensed) ...")
                    for line in lines[-10:]:  # Show last 10 lines
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(full_output)
        
        # Final status
        print(f"\n⏱️  Execution time: {int(mins):02d}:{int(secs):02d}")
        print(f"🕒 Completed at: {datetime.now().strftime('%H:%M:%S')}")
        
        if return_code == 0:
            print(f"✅ {description} - PASSED")
        else:
            print(f"❌ {description} - FAILED (exit code: {return_code})")
            
        return return_code == 0
        
    except Exception as e:
        if progress:
            progress.stop()
        print(f"❌ Error running {description}: {e}")
        return False

def show_test_plan(args):
    """Show what tests will be run and estimated total time."""
    tests_to_run = []
    total_estimate_mins = 0
    
    if args.all or args.unit:
        tests_to_run.append(("Unit Tests", "unit", 2))
        total_estimate_mins += 2
    
    if args.all or args.statistical:
        tests_to_run.append(("Statistical Validation", "statistical", 27))
        total_estimate_mins += 27
    
    if args.all or args.performance:
        tests_to_run.append(("Performance Benchmarks", "performance", 3))
        total_estimate_mins += 3
    
    if args.all or args.regression:
        tests_to_run.append(("Performance Regression", "regression", 6))
        total_estimate_mins += 6
    
    if args.all or args.quick:
        tests_to_run.append(("Quick Validation", "validation", 1))
        total_estimate_mins += 1
    
    if args.example:
        tests_to_run.append(("Example Usage", "example", 5))
        total_estimate_mins += 5
    
    print(f"\n📋 Test Execution Plan:")
    print("="*60)
    for i, (name, test_type, mins) in enumerate(tests_to_run, 1):
        print(f"  {i}. {name} (~{mins} min)")
    
    hrs, mins = divmod(total_estimate_mins, 60)
    if hrs > 0:
        time_str = f"{hrs}h {mins}m"
    else:
        time_str = f"{mins}m"
    
    print(f"\n⏱️  Total estimated time: {time_str}")
    print(f"🕒 Expected completion: {(datetime.now() + timedelta(minutes=total_estimate_mins)).strftime('%H:%M:%S')}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Run Poker Knight tests with enhanced progress feedback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py --all           # Run all tests (45+ minutes)
  python run_tests.py --unit          # Quick unit tests only (2 min)
  python run_tests.py --performance   # Performance tests only (3 min)
  python run_tests.py --quick         # Quick validation only (30 sec)
        """
    )
    parser.add_argument("--all", action="store_true", help="Run all tests (~45 minutes)")
    parser.add_argument("--unit", action="store_true", help="Run unit tests (~2 minutes)")
    parser.add_argument("--statistical", action="store_true", help="Run statistical validation (~27 minutes)")
    parser.add_argument("--performance", action="store_true", help="Run performance benchmarks (~3 minutes)")
    parser.add_argument("--regression", action="store_true", help="Run regression tests (~6 minutes)")
    parser.add_argument("--quick", action="store_true", help="Run quick validation (~30 seconds)")
    parser.add_argument("--example", action="store_true", help="Run example usage (~5 minutes)")
    
    args = parser.parse_args()
    
    # If no specific tests requested, run all
    if not any([args.unit, args.statistical, args.performance, args.regression, args.quick, args.example]):
        args.all = True
    
    print("♞ Poker Knight Enhanced Test Runner v1.4.0")
    print("="*80)
    print("🚀 High-performance Monte Carlo poker solver testing suite")
    print(f"🕒 Test session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Show test plan
    show_test_plan(args)
    
    # Confirmation for long test runs
    if args.all or args.statistical:
        print(f"\n⚠️  Note: Full test suite includes long-running statistical validation")
        print(f"   Consider running individual test categories for faster feedback")
    
    print(f"\n🎬 Starting test execution...")
    
    success_count = 0
    total_count = 0
    
    session_start = time.time()
    
    if args.all or args.unit:
        total_count += 1
        if run_command_with_progress("python -m pytest tests/test_poker_solver.py -v", "Unit Tests", "unit"):
            success_count += 1
    
    if args.all or args.statistical:
        total_count += 1
        if run_command_with_progress("python -m pytest tests/test_statistical_validation.py -v", "Statistical Validation Tests", "statistical"):
            success_count += 1
    
    if args.all or args.performance:
        total_count += 1
        if run_command_with_progress("python tests/test_performance.py", "Performance Benchmarks", "performance"):
            success_count += 1
    
    if args.all or args.regression:
        total_count += 1
        if run_command_with_progress("python -m pytest tests/test_performance_regression.py -v", "Performance Regression Tests", "regression"):
            success_count += 1
    
    if args.all or args.quick:
        total_count += 1
        if run_command_with_progress("python tests/test_validation.py", "Quick Validation Tests", "validation"):
            success_count += 1
    
    if args.example:
        total_count += 1
        if run_command_with_progress("python examples/example_usage.py", "Example Usage", "example"):
            success_count += 1
    
    # Final summary
    session_elapsed = time.time() - session_start
    session_mins, session_secs = divmod(session_elapsed, 60)
    
    print(f"\n{'='*80}")
    print(f"🎯 Final Test Session Summary")
    print('='*80)
    print(f"📊 Tests Passed: {success_count}/{total_count}")
    print(f"⏱️  Total session time: {int(session_mins):02d}:{int(session_secs):02d}")
    print(f"🕒 Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if success_count == total_count:
        print("🎉 All tests passed! Poker Knight is ready for production.")
        print("🚀 Monte Carlo solver is performing optimally.")
    else:
        print("⚠️  Some tests failed! Please review the output above.")
        failed_count = total_count - success_count
        print(f"❌ {failed_count} test suite(s) need attention.")
    
    print('='*80)
    
    return 0 if success_count == total_count else 1

if __name__ == "__main__":
    sys.exit(main()) 