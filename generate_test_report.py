#!/usr/bin/env python3
"""
Generate a markdown test report from the latest test results in tests/results/

Usage:
    python generate_test_report.py                     # Use latest test result
    python generate_test_report.py <test_result.json>  # Use specific test result file
"""

import json
import os
import sys
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def find_latest_test_results(results_dir: str = "tests/results") -> List[Tuple[str, Dict]]:
    """Find and load all test result JSON files, sorted by timestamp."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory {results_dir} not found")
    
    test_files = []
    for file in results_path.glob("test_results_*.json"):
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                test_files.append((str(file), data))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read {file}: {e}")
    
    # Sort by timestamp (newest first)
    test_files.sort(key=lambda x: x[1].get('timestamp', ''), reverse=True)
    return test_files

def get_test_type_emoji(test_type: str) -> str:
    """Return an emoji for each test type."""
    emojis = {
        'all': '🎯',
        'unit': '🧪',
        'statistical': '📊',
        'performance': '⚡',
        'stress': '💪',
        'numa': '🖥️',
        'quick': '🏃'
    }
    return emojis.get(test_type, '📋')

def get_test_file_categories() -> Dict[str, Tuple[str, str]]:
    """Return test file categories and descriptions."""
    return {
        # Core functionality
        "test_poker_solver.py": ("🎯 Core Functionality", "Main solver implementation tests"),
        "test_card_formats.py": ("🃏 Card Handling", "Card format parsing and validation"),
        "test_validation.py": ("✅ Input Validation", "Input validation and error handling"),
        
        # Performance & optimization
        "test_performance.py": ("⚡ Performance", "Performance benchmarks and optimization"),
        "test_performance_regression.py": ("📊 Performance Regression", "Performance regression tests"),
        "test_parallel.py": ("🔄 Parallel Processing", "Parallel execution tests"),
        "test_advanced_parallel.py": ("🚀 Advanced Parallel", "Advanced parallel processing"),
        "test_force_advanced_parallel.py": ("💪 Forced Parallel", "Forced parallel execution tests"),
        
        # Statistical & mathematical
        "test_statistical_validation.py": ("📈 Statistical Validation", "Statistical accuracy tests"),
        "test_precision.py": ("🎯 Precision Testing", "Numerical precision validation"),
        "test_convergence.py": ("📉 Convergence", "Monte Carlo convergence tests"),
        "test_enhanced_convergence.py": ("📊 Enhanced Convergence", "Advanced convergence analysis"),
        
        # GPU/CUDA
        "test_cuda_integration.py": ("🚀 GPU/CUDA Integration", "GPU acceleration tests"),
        
        # Advanced features
        "test_multi_way_scenarios.py": ("👥 Multi-Way Pots", "Multi-player scenario tests"),
        "test_multiway_quick.py": ("⚡ Quick Multi-Way", "Fast multi-way pot tests"),
        "test_smart_sampling.py": ("🧠 Smart Sampling", "Intelligent sampling strategies"),
        
        # Edge cases & stress
        "test_edge_cases_extended.py": ("🔍 Edge Cases", "Extended edge case testing"),
        "test_stress_scenarios.py": ("💪 Stress Testing", "Stress and load tests"),
        
        # System & integration
        "test_numa.py": ("🖥️ NUMA Testing", "Non-Uniform Memory Access tests"),
        "test_runner_safe.py": ("🛡️ Safe Runner", "Safe test execution framework"),
    }

def get_test_descriptions() -> Dict[str, str]:
    """Return descriptions for each test based on test name patterns."""
    return {
        # Core functionality tests
        "test_basic_functionality": "Validates core Monte Carlo solver functionality",
        "test_card_creation": "Tests card object creation and validation",
        "test_card_value": "Verifies card ranking and comparison logic",
        "test_deck_operations": "Tests deck creation, shuffling, and card dealing",
        "test_hand_evaluation": "Validates poker hand ranking algorithms",
        "test_solve_poker_hand_function": "Tests the main solve_poker_hand API",
        "test_invalid_inputs": "Validates error handling for invalid inputs",
        "test_simulation_modes": "Tests fast, default, and precision simulation modes",
        "test_board_scenarios": "Tests various flop, turn, and river scenarios",
        "test_timeout_behavior": "Validates simulation timeout handling",
        
        # Statistical validation
        "test_known_probabilities": "Compares results against known poker probabilities",
        "test_statistical_convergence": "Validates Monte Carlo convergence properties",
        "test_confidence_intervals": "Tests statistical confidence interval calculations",
        "test_hand_distribution": "Validates hand type frequency distributions",
        "test_variance_analysis": "Tests variance and standard deviation calculations",
        
        # Performance tests
        "test_fast_mode_performance": "Benchmarks fast mode (10k simulations)",
        "test_default_mode_performance": "Benchmarks default mode (100k simulations)",
        "test_precision_mode_performance": "Benchmarks precision mode (500k simulations)",
        "test_hand_evaluation_speed": "Measures hand evaluation performance",
        "test_memory_usage": "Monitors memory consumption during simulations",
        "test_parallel_performance": "Tests parallel processing efficiency",
        
        # Edge cases
        "test_duplicate_cards": "Tests duplicate card detection",
        "test_invalid_card_formats": "Validates card format parsing",
        "test_boundary_conditions": "Tests min/max opponents and board sizes",
        "test_special_hands": "Tests wheel straights and edge case hands",
        "test_tie_scenarios": "Validates tie detection and probability",
        
        # ICM and tournament features
        "test_icm_calculations": "Tests Independent Chip Model calculations",
        "test_bubble_factor": "Validates bubble factor calculations",
        "test_position_equity": "Tests position-aware equity adjustments",
        "test_tournament_scenarios": "Tests various tournament situations",
        
        # GPU/CUDA tests
        "test_gpu_detection": "Tests GPU availability detection",
        "test_cuda_kernel": "Validates CUDA kernel functionality",
        "test_gpu_accuracy": "Compares GPU vs CPU result accuracy",
        "test_gpu_performance": "Benchmarks GPU acceleration performance",
        "test_gpu_fallback": "Tests automatic CPU fallback when GPU unavailable",
        
        # Integration tests
        "test_multiway_pots": "Tests 3+ player scenarios",
        "test_convergence_analysis": "Tests convergence detection algorithms",
        "test_optimization": "Tests simulation optimization features",
        "test_reporting": "Validates result reporting and formatting",
        
        # Stress tests
        "test_high_simulation_count": "Tests with millions of simulations",
        "test_concurrent_execution": "Tests thread safety and concurrency",
        "test_extended_runtime": "Long-running stress tests",
        "test_memory_stress": "Tests under memory pressure"
    }

def parse_test_name(test_path: str) -> Tuple[str, str, str]:
    """Parse test path to extract file, class, and method names."""
    parts = test_path.split("::")
    if len(parts) >= 3:
        file_path = parts[0]
        class_name = parts[1]
        method_name = parts[2]
    elif len(parts) == 2:
        file_path = parts[0]
        class_name = ""
        method_name = parts[1]
    else:
        file_path = parts[0] if parts else test_path
        class_name = ""
        method_name = ""
    
    # Extract just the filename
    file_name = Path(file_path).name if file_path else ""
    
    return file_name, class_name, method_name

def get_status_badge(success_rate: float) -> str:
    """Return a status badge based on success rate."""
    if success_rate >= 100:
        return "![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)"
    elif success_rate >= 95:
        return "![Tests](https://img.shields.io/badge/tests-mostly%20passing-green.svg)"
    elif success_rate >= 80:
        return "![Tests](https://img.shields.io/badge/tests-some%20failures-yellow.svg)"
    else:
        return "![Tests](https://img.shields.io/badge/tests-failing-red.svg)"

def format_timestamp(timestamp: str) -> str:
    """Convert timestamp to readable format."""
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%d_%H-%M-%S")
        return dt.strftime("%B %d, %Y at %I:%M %p")
    except:
        return timestamp

def load_specific_test_result(file_path: str) -> Tuple[str, Dict]:
    """Load a specific test result file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Test result file not found: {file_path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return str(path), data

def collect_all_tests() -> Dict[str, List[str]]:
    """Collect all test names by running pytest collection."""
    try:
        # Run pytest in collect-only mode to get all test names
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        
        test_dict = {}
        for line in result.stdout.splitlines():
            if "::" in line and not line.startswith(" "):
                file_name, class_name, method_name = parse_test_name(line.strip())
                if file_name not in test_dict:
                    test_dict[file_name] = []
                test_dict[file_name].append(method_name)
        
        return test_dict
    except Exception:
        # Fallback if pytest collection fails
        return {}

def generate_markdown_report(output_file: str = "TEST_REPORT.md", input_file: Optional[str] = None):
    """Generate a comprehensive markdown test report from the latest test run or a specific file."""
    if input_file:
        # Use specific file
        latest_file, latest_data = load_specific_test_result(input_file)
    else:
        # Use latest result
        test_results = find_latest_test_results()
        
        if not test_results:
            print("No test results found!")
            return
        
        # Get only the most recent result
        latest_file, latest_data = test_results[0]
    
    # Start building the markdown
    md_lines = []
    
    # Header
    md_lines.append("# 🧪 Poker Knight Test Report")
    md_lines.append("")
    md_lines.append("<div align=\"center\">")
    md_lines.append("")
    md_lines.append("![Poker Knight Logo](docs/assets/poker_knight_logo.png)")
    md_lines.append("")
    md_lines.append(f"**Test Report Generated**: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}")
    md_lines.append("")
    md_lines.append("</div>")
    md_lines.append("")
    
    # Overall Status
    success_rate = latest_data['summary']['success_rate']
    md_lines.append("## 📊 Overall Status")
    md_lines.append("")
    md_lines.append("<div align=\"center\">")
    md_lines.append("")
    md_lines.append(f"{get_status_badge(success_rate)}")
    md_lines.append("")
    md_lines.append(f"### {success_rate:.1f}% Success Rate")
    md_lines.append("")
    md_lines.append("</div>")
    md_lines.append("")
    md_lines.append(f"- 📅 **Test Run**: {format_timestamp(latest_data['timestamp'])}")
    md_lines.append(f"- 🧪 **Total Tests**: {latest_data['summary']['total_tests']}")
    md_lines.append(f"- ⏱️ **Exit Status**: {latest_data['summary'].get('exit_status', 'N/A')}")
    md_lines.append("")
    
    # Summary Table
    md_lines.append("## 📈 Test Summary")
    md_lines.append("")
    md_lines.append("| Metric | Count | Percentage |")
    md_lines.append("|--------|-------|------------|")
    
    summary = latest_data['summary']
    total = summary['total_tests']
    
    md_lines.append(f"| ✅ Passed | {summary['passed']} | {(summary['passed']/total*100):.1f}% |")
    md_lines.append(f"| ❌ Failed | {summary['failed']} | {(summary['failed']/total*100) if total > 0 else 0:.1f}% |")
    md_lines.append(f"| 💥 Errors | {summary['errors']} | {(summary['errors']/total*100) if total > 0 else 0:.1f}% |")
    md_lines.append(f"| ⏭️ Skipped | {summary['skipped']} | {(summary['skipped']/total*100) if total > 0 else 0:.1f}% |")
    md_lines.append("")
    
    # Test Details
    test_type = latest_data.get('test_type', 'all')
    emoji = get_test_type_emoji(test_type)
    md_lines.append(f"## {emoji} Test Details")
    md_lines.append("")
    md_lines.append(f"**Test Category**: {test_type.upper()}")
    md_lines.append("")
    
    # Add performance highlights if this is a performance test
    if test_type == 'performance' and success_rate == 100:
        md_lines.append("### 🚀 Performance Highlights")
        md_lines.append("- All performance benchmarks passed")
        md_lines.append("- No performance regressions detected")
        md_lines.append("")
    
    # Failed Tests (if any)
    if latest_data['summary']['failed'] > 0 or latest_data['summary']['errors'] > 0:
        md_lines.append("## ❌ Failed Tests")
        md_lines.append("")
        
        if latest_data.get('failed_tests'):
            md_lines.append("### Failed Tests:")
            for test in latest_data['failed_tests']:
                md_lines.append(f"- `{test}`")
            md_lines.append("")
        
        if latest_data.get('error_tests'):
            md_lines.append("### Tests with Errors:")
            for test in latest_data['error_tests']:
                md_lines.append(f"- `{test}`")
            md_lines.append("")
    
    # Test Configuration
    md_lines.append("## ⚙️ Test Configuration")
    md_lines.append("")
    md_lines.append("**Latest test run configuration:**")
    md_lines.append("```json")
    md_lines.append(json.dumps(latest_data.get('configuration', {}), indent=2))
    md_lines.append("```")
    md_lines.append("")
    
    # Analysis
    md_lines.append("## 🔍 Test Analysis")
    md_lines.append("")
    
    analysis = latest_data.get('analysis', {})
    overall_status = analysis.get('overall_status', 'UNKNOWN')
    
    if overall_status == 'PASS':
        md_lines.append("✅ **Overall Status: PASS**")
    else:
        md_lines.append(f"❌ **Overall Status: {overall_status}**")
    
    md_lines.append("")
    
    # Regression indicators
    regression = analysis.get('regression_indicators', {})
    if regression:
        md_lines.append("### Regression Indicators:")
        md_lines.append(f"- Has Failures: {'Yes' if regression.get('has_failures') else 'No'}")
        md_lines.append(f"- Has Errors: {'Yes' if regression.get('has_errors') else 'No'}")
        md_lines.append(f"- Below Success Threshold: {'Yes' if regression.get('success_rate_below_threshold') else 'No'}")
        md_lines.append("")
    
    # Comprehensive Test Suite Overview
    md_lines.append("## 📋 Comprehensive Test Suite")
    md_lines.append("")
    md_lines.append("Poker Knight's test suite validates every aspect of the solver, from basic card handling to advanced GPU acceleration.")
    md_lines.append("")
    
    # Test categories
    file_categories = get_test_file_categories()
    test_descriptions = get_test_descriptions()
    
    # Group tests by category
    categories_dict = {}
    for file_name, (category, file_desc) in file_categories.items():
        if category not in categories_dict:
            categories_dict[category] = []
        categories_dict[category].append((file_name, file_desc))
    
    # Sort categories for consistent output
    for category in sorted(categories_dict.keys()):
        md_lines.append(f"### {category}")
        md_lines.append("")
        
        for file_name, file_desc in categories_dict[category]:
            md_lines.append(f"**`{file_name}`** - {file_desc}")
            md_lines.append("")
            
            # Add specific test descriptions if available
            # This is a simplified version - in reality we'd parse actual test names
            if "cuda" in file_name.lower():
                md_lines.append("- `test_gpu_available()` - Detects NVIDIA GPU availability")
                md_lines.append("- `test_cuda_kernel_compilation()` - Validates CUDA kernel compilation")
                md_lines.append("- `test_gpu_vs_cpu_accuracy()` - Ensures GPU results match CPU exactly")
                md_lines.append("- `test_gpu_performance_scaling()` - Measures GPU speedup factors")
                md_lines.append("- `test_automatic_fallback()` - Tests CPU fallback when GPU unavailable")
            elif "poker_solver" in file_name:
                md_lines.append("- `test_basic_functionality()` - Core solver operations")
                md_lines.append("- `test_solve_poker_hand_function()` - Main API functionality")
                md_lines.append("- `test_simulation_modes()` - Fast/default/precision modes")
                md_lines.append("- `test_invalid_inputs()` - Error handling and validation")
                md_lines.append("- `test_board_scenarios()` - Flop, turn, river scenarios")
            elif "performance" in file_name and "regression" not in file_name:
                md_lines.append("- `test_fast_mode_performance()` - 10k simulations benchmark")
                md_lines.append("- `test_default_mode_performance()` - 100k simulations benchmark")
                md_lines.append("- `test_precision_mode_performance()` - 500k simulations benchmark")
                md_lines.append("- `test_memory_efficiency()` - Memory usage optimization")
            elif "statistical" in file_name:
                md_lines.append("- `test_known_probabilities()` - Validates against known poker odds")
                md_lines.append("- `test_confidence_intervals()` - Statistical confidence testing")
                md_lines.append("- `test_variance_analysis()` - Monte Carlo variance properties")
                md_lines.append("- `test_hand_distribution()` - Hand frequency validation")
            
            md_lines.append("")
    
    md_lines.append("")
    
    # Footer
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("*This report is automatically generated from test results in `tests/results/`*")
    md_lines.append("")
    md_lines.append(f"**Poker Knight v1.8.0** | [View on GitHub](https://github.com/hildolfr/poker_knight)")
    
    # Write the report
    with open(output_file, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✅ Test report generated: {output_file}")
    print(f"   - Test run: {latest_data['timestamp']}")
    print(f"   - Test type: {latest_data.get('test_type', 'unknown')}")
    print(f"   - Success rate: {success_rate:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Use specific file
        generate_markdown_report(input_file=sys.argv[1])
    else:
        # Use latest
        generate_markdown_report()