# Testing Guide

Comprehensive guide to running tests and validating Poker Knight's accuracy and performance.

## Quick Start

```bash
# Run comprehensive test suite
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --quick        # Quick validation
python tests/run_tests.py --statistical  # Statistical validation
python tests/run_tests.py --performance  # Performance benchmarks
```

## Test Structure

### Test Categories

| Test File | Purpose | Execution Time |
|-----------|---------|----------------|
| `test_poker_solver.py` | Core functionality tests | ~30s |
| `test_statistical_validation.py` | Statistical accuracy validation | ~2-3min |
| `test_performance.py` | Performance benchmarks | ~1min |
| `test_performance_regression.py` | Performance regression detection | ~2min |
| `test_edge_cases_extended.py` | Edge case handling | ~45s |
| `test_multi_way_scenarios.py` | Multi-opponent scenarios | ~1min |
| `test_stress_scenarios.py` | Stress testing | ~3min |

### Using pytest Directly

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_poker_solver.py -v

# Run tests with specific markers
python -m pytest -m "unit" -v          # Unit tests only
python -m pytest -m "statistical" -v   # Statistical tests only
python -m pytest -m "performance" -v   # Performance tests only

# Run tests with coverage
python -m pytest tests/ --cov=poker_knight --cov-report=html
```

## Statistical Validation

Poker Knight includes comprehensive statistical validation to ensure Monte Carlo simulation accuracy.

### Validation Test Suite

The statistical validation suite performs rigorous testing against established poker mathematics:

#### 🧮 **Chi-Square Goodness-of-Fit Testing**
- Tests hand category distributions against expected poker probabilities
- Validates that observed frequencies match theoretical distributions
- **Result**: χ² = 0.050 (df = 6) - **Excellent fit to expected distributions**

#### 📈 **Monte Carlo Convergence Validation**  
- Confirms error decreases as 1/√n (theoretical Monte Carlo property)
- Tests convergence rates across different simulation counts
- **Result**: Proper convergence with 2x error reduction for 4x simulation increase

#### 📊 **Confidence Interval Coverage**
- Validates that 95% confidence intervals contain true values 95% of the time
- Tests statistical confidence calculation accuracy
- **Result**: 100% coverage rate across test scenarios

#### 🎯 **Known Poker Probability Validation**
- Cross-validates simulation results against established poker mathematics
- Tests pre-flop matchups and post-flop scenarios

| Scenario | Expected | Observed | Status |
|----------|----------|----------|---------|
| AA vs Random (preflop) | 85.0% | 84.9% | ✅ Validated |
| AKs vs Random (preflop) | 66.0% | 66.1% | ✅ Validated |
| 72o vs Random (preflop) | 32.0% | 31.6% | ✅ Validated |
| AA with Top Set | 95.0% | 93.2% | ✅ Validated |

#### ⚖️ **Symmetry Testing**
- Verifies equivalent hands produce equivalent results
- Tests suit symmetry (same hand, different suits)
- **Result**: All equivalent hands within 0.004 difference

#### 📉 **Variance Stability**
- Ensures consistent simulation variance across multiple runs
- Monitors for implementation stability issues
- **Result**: Standard deviation = 0.004 (excellent stability)

### Running Statistical Tests

```bash
# Run complete statistical validation
python tests/run_tests.py --statistical

# Run specific statistical test categories
python -m pytest tests/test_statistical_validation.py::test_chi_square_hand_categories -v
python -m pytest tests/test_statistical_validation.py::test_confidence_interval_coverage -v
python -m pytest tests/test_statistical_validation.py::test_monte_carlo_convergence -v
```

## Performance Testing

### Performance Benchmarks

```bash
# Run performance benchmarks
python tests/run_tests.py --performance

# Run specific performance tests
python -m pytest tests/test_performance.py -v
python -m pytest tests/test_performance_regression.py -v
```

### Expected Performance Metrics

| Scenario | Fast Mode | Default Mode | Precision Mode |
|----------|-----------|--------------|----------------|
| Pre-flop 2 players | <100ms | <500ms | <2000ms |
| Flop 4 players | <150ms | <750ms | <3000ms |
| River 6 players | <200ms | <1000ms | <4000ms |

### Performance Regression Detection

The performance regression test suite monitors for performance degradation:

```python
# Example performance test
def test_preflop_performance():
    start_time = time.time()
    result = solve_poker_hand(['A♠️', 'A♥️'], 2, simulation_mode="fast")
    execution_time = (time.time() - start_time) * 1000
    
    assert execution_time < 100  # Should complete in under 100ms
    assert result.simulations_run >= 8000  # Should run sufficient simulations
```

## Edge Case Testing

### Comprehensive Edge Case Coverage

```bash
# Run edge case tests
python -m pytest tests/test_edge_cases_extended.py -v
```

Edge cases tested include:
- **Duplicate cards**: Proper error handling for invalid input
- **Invalid card formats**: Robust input validation
- **Extreme opponent counts**: Boundary condition handling
- **Empty board scenarios**: Pre-flop analysis validation
- **Complete board scenarios**: River analysis accuracy
- **Timeout scenarios**: Graceful handling of time limits

### Multi-Way Scenario Testing

```bash
# Run multi-opponent tests
python -m pytest tests/test_multi_way_scenarios.py -v
```

Tests complex scenarios with 3-6 opponents:
- **Equity distribution**: Proper probability distribution across opponents
- **Hand strength scaling**: Accurate strength assessment with more opponents
- **Performance scaling**: Reasonable execution times with increased complexity

## Stress Testing

### High-Load Scenarios

```bash
# Run stress tests
python -m pytest tests/test_stress_scenarios.py -v
```

Stress tests include:
- **Rapid successive calls**: Memory leak detection
- **Large simulation counts**: Stability with high precision mode
- **Parallel processing stress**: Thread safety validation
- **Memory usage monitoring**: Resource consumption tracking

### Memory and Resource Testing

```python
# Example stress test
def test_memory_stability():
    initial_memory = get_memory_usage()
    
    # Run 100 analyses
    for _ in range(100):
        result = solve_poker_hand(['A♠️', 'K♠️'], 2)
    
    final_memory = get_memory_usage()
    memory_increase = final_memory - initial_memory
    
    assert memory_increase < 50  # Memory increase should be minimal
```

## Custom Test Configuration

### Test Configuration File

Create `tests/test_config.json` for custom test settings:

```json
{
  "statistical_tests": {
    "chi_square_significance": 0.05,
    "confidence_level": 0.95,
    "monte_carlo_samples": 10000
  },
  "performance_tests": {
    "max_execution_time_ms": 5000,
    "min_simulations_fast": 8000,
    "min_simulations_default": 80000
  },
  "stress_tests": {
    "max_memory_increase_mb": 50,
    "rapid_call_count": 100,
    "parallel_thread_count": 8
  }
}
```

### Running Tests with Custom Configuration

```python
# Custom test runner
import json
from poker_knight import solve_poker_hand

def run_custom_validation():
    with open('tests/test_config.json') as f:
        config = json.load(f)
    
    # Run tests with custom parameters
    max_time = config['performance_tests']['max_execution_time_ms']
    # ... test implementation
```

## Continuous Integration

### GitHub Actions Configuration

```yaml
# .github/workflows/test.yml
name: Test Suite

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11]
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install pytest pytest-cov
    
    - name: Run tests
      run: |
        python tests/run_tests.py --quick
        python -m pytest tests/ --cov=poker_knight
```

## Test Development Guidelines

### Writing New Tests

1. **Follow naming conventions**: `test_<functionality>_<scenario>()`
2. **Use descriptive assertions**: Include meaningful error messages
3. **Test edge cases**: Include boundary conditions and error cases
4. **Performance considerations**: Mark slow tests appropriately
5. **Statistical rigor**: Use appropriate sample sizes for statistical tests

### Example Test Structure

```python
import pytest
from poker_knight import solve_poker_hand

class TestPokerSolver:
    def test_preflop_pocket_aces(self):
        """Test pocket aces pre-flop equity"""
        result = solve_poker_hand(['A♠️', 'A♥️'], 1)
        
        # Pocket aces should have ~85% equity heads-up
        assert 0.80 <= result.win_probability <= 0.90
        assert result.simulations_run > 0
        assert result.execution_time_ms > 0
    
    @pytest.mark.statistical
    def test_statistical_accuracy(self):
        """Test statistical accuracy against known probabilities"""
        # Implementation with statistical validation
        pass
    
    @pytest.mark.performance
    def test_execution_speed(self):
        """Test execution speed requirements"""
        import time
        start = time.time()
        result = solve_poker_hand(['K♠️', 'K♥️'], 2, simulation_mode="fast")
        execution_time = (time.time() - start) * 1000
        
        assert execution_time < 200  # Should complete quickly
```

## Troubleshooting Tests

### Common Test Issues

**Slow test execution:**
- Use `--quick` flag for faster validation
- Check if precision mode is being used unnecessarily
- Verify parallel processing is enabled

**Statistical test failures:**
- Increase sample sizes for more stable results
- Check for proper random seed handling
- Verify confidence interval calculations

**Performance test failures:**
- Check system load during test execution
- Verify configuration settings
- Monitor memory usage patterns

### Test Debugging

```bash
# Run tests with verbose output
python -m pytest tests/ -v -s

# Run specific test with debugging
python -m pytest tests/test_poker_solver.py::test_specific_function -v -s --pdb

# Run tests with coverage report
python -m pytest tests/ --cov=poker_knight --cov-report=term-missing
``` 