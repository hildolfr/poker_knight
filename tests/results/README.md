# Test Results Directory

This directory contains automatically generated, timestamped test result files for AI analysis and regression tracking.

## File Naming Convention

```
test_results_YYYY-MM-DD_HH-MM-SS_<test_type>.json
```

Examples:
- `test_results_2024-06-01_14-30-25_all.json`
- `test_results_2024-06-01_14-35-12_unit.json`
- `test_results_2024-06-01_14-40-18_statistical.json`

## Test Types

- `all` - Complete comprehensive test suite
- `unit` - Core functionality tests only
- `statistical` - Statistical validation tests
- `performance` - Performance and benchmark tests
- `stress` - Stress and load testing
- `quick` - Fast validation subset

## File Format

Each result file is a JSON document with the following structure:

```json
{
  "timestamp": "2024-06-01_14-30-25",
  "test_type": "all",
  "test_suite": "Poker Knight Monte Carlo Solver",
  "summary": {
    "total_tests": 150,
    "passed": 148,
    "failed": 2,
    "errors": 0,
    "skipped": 0,
    "success_rate": 98.7,
    "exit_status": 1
  },
  "configuration": {
    "quick": false,
    "statistical": false,
    "performance": false,
    "stress": false,
    "unit": false,
    "all": true,
    "with_coverage": false
  },
  "failed_tests": [
    {
      "name": "tests/test_example.py::test_function",
      "outcome": "failed",
      "duration": 0.25,
      "longrepr": "AssertionError: Expected 0.85 but got 0.82"
    }
  ],
  "error_tests": [],
  "analysis": {
    "overall_status": "FAIL",
    "critical_failures": 2,
    "regression_indicators": {
      "has_failures": true,
      "has_errors": false,
      "success_rate_below_threshold": false
    }
  }
}
```

## AI Analysis Usage

These files are designed for AI analysis to:

1. **Track Regression**: Compare success rates and failure patterns over time
2. **Identify Trends**: Monitor which test types are failing more frequently
3. **Performance Monitoring**: Track test execution patterns and durations
4. **Quality Metrics**: Analyze overall codebase health through test success rates

## Retention

Files are never overwritten due to timestamps. Consider implementing a cleanup policy for old results if disk space becomes a concern.

## Integration

Results are automatically generated every time tests are run via:
- `pytest` (any options)
- `python tests/run_tests.py` (any options)

No manual intervention required - the system captures all test runs automatically. 