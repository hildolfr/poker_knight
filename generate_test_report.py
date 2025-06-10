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