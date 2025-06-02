# Cache Pre-Population Testing Guide

This guide covers the comprehensive test suite for the Poker Knight Cache Pre-Population System, including test organization, execution, and integration with the existing test infrastructure.

## 📋 Overview

The cache pre-population test suite validates the complete cache pre-population system including:

- **Configuration Management**: PopulationConfig validation and defaults
- **Scenario Generation**: Comprehensive poker scenario creation (169 hands)
- **Population Engine**: One-time cache population logic and execution
- **Integration**: Seamless integration with existing cache and solver systems
- **Performance**: Timing characteristics and resource usage
- **Error Handling**: Graceful handling of failures and edge cases

## 🏗️ Test Architecture

### Test Organization

```
tests/
├── test_cache_population.py         # Main test suite
├── run_cache_population_tests.py    # Dedicated test runner
├── conftest.py                      # Updated with cache population markers
├── pytest.ini                      # Updated with cache_population marker
└── CACHE_POPULATION_TESTING_GUIDE.md # This documentation
```

### Test Classes

#### **TestPopulationConfig** 
- Configuration validation and defaults
- Custom configuration values
- Range validation and option checking
- **Markers**: `@pytest.mark.cache`, `@pytest.mark.cache_population`

#### **TestScenarioGenerator**
- 169 preflop hand generation
- Premium/common/all hands subsets
- Scenario structure validation
- Hand notation conversion
- **Markers**: `@pytest.mark.cache`, `@pytest.mark.cache_population`

#### **TestCachePrePopulator**
- Population engine initialization
- Population decision logic
- Scenario simulation
- Statistics tracking
- **Markers**: `@pytest.mark.cache`, `@pytest.mark.cache_population`

#### **TestCachePopulationIntegration**
- Integration with cache managers
- Solver integration decorator testing
- Cross-system compatibility
- **Markers**: `@pytest.mark.cache`, `@pytest.mark.cache_population`, `@pytest.mark.integration`

#### **TestCachePopulationPerformance**
- Scenario generation performance
- Simulation timing characteristics
- Timeout enforcement
- **Markers**: `@pytest.mark.cache`, `@pytest.mark.cache_population`, `@pytest.mark.performance`

#### **TestCachePopulationErrorHandling**
- Invalid configuration handling
- Missing dependency graceful degradation
- Simulation error recovery
- Cache operation failures
- **Markers**: `@pytest.mark.cache`, `@pytest.mark.cache_population`, `@pytest.mark.unit`

## 🚀 Running Tests

### Quick Start

```bash
# Run all cache population tests
pytest --cache-population

# Using the dedicated runner
python tests/run_cache_population_tests.py
```

### Command Line Options

#### **Pytest Integration**
```bash
# All cache population tests
pytest --cache-population

# Cache population with verbose output
pytest --cache-population -v

# Cache population with coverage
pytest --cache-population --cov=poker_knight.storage.cache_prepopulation

# Combined with other cache tests
pytest --cache -k "cache_population or cache_unit"
```

#### **Dedicated Test Runner**
```bash
# Basic usage
python tests/run_cache_population_tests.py

# Test type selection
python tests/run_cache_population_tests.py --unit
python tests/run_cache_population_tests.py --integration  
python tests/run_cache_population_tests.py --performance

# Specific tests
python tests/run_cache_population_tests.py --class TestPopulationConfig
python tests/run_cache_population_tests.py --test "test_scenario_generation"

# Output options
python tests/run_cache_population_tests.py --verbose
python tests/run_cache_population_tests.py --quiet
python tests/run_cache_population_tests.py --coverage

# Execution options
python tests/run_cache_population_tests.py --fail-fast
python tests/run_cache_population_tests.py --parallel 4
python tests/run_cache_population_tests.py --durations
```

### Integration with Existing Test Suite

```bash
# Run all cache tests (including population)
pytest --cache

# Run cache tests with population focus
pytest --cache -k cache_population

# Run performance tests (including cache population performance)
pytest --performance

# Run integration tests (including cache population integration)
pytest --integration
```

## 🎯 Test Markers and Categories

### Primary Markers

- **`@pytest.mark.cache`**: All cache-related tests
- **`@pytest.mark.cache_population`**: Specific to cache pre-population

### Secondary Markers

- **`@pytest.mark.unit`**: Fast unit tests, no external dependencies
- **`@pytest.mark.integration`**: Integration with other systems
- **`@pytest.mark.performance`**: Performance and timing tests

### Usage Examples

```python
# Run only cache population unit tests
pytest -m "cache_population and unit"

# Run cache population but exclude performance tests
pytest -m "cache_population and not performance"

# Run all cache tests except population
pytest -m "cache and not cache_population"
```

## 📊 Test Coverage

### Code Coverage Areas

1. **Configuration Management**
   - PopulationConfig validation
   - Default value handling
   - Custom configuration processing

2. **Scenario Generation**
   - 169 preflop hand generation
   - Subset filtering (premium, common, all)
   - Board texture scenario creation
   - Hand notation conversion

3. **Population Engine**
   - Population decision logic
   - Scenario simulation
   - Cache storage operations
   - Progress tracking and statistics

4. **Integration Points**
   - Cache manager integration
   - Solver decorator functionality
   - Cross-system error handling

5. **Performance Characteristics**
   - Generation speed validation
   - Simulation timing verification
   - Timeout enforcement testing

### Coverage Reports

```bash
# Generate coverage report
python tests/run_cache_population_tests.py --coverage

# View HTML coverage report
open tests/results/cache_population_coverage/index.html
```

## 🔧 Test Configuration

### Mock Usage

The test suite uses extensive mocking to:

- **Isolate units**: Test components independently
- **Avoid dependencies**: Skip Redis/SQLite in unit tests
- **Control behavior**: Simulate various scenarios and errors
- **Speed up tests**: Avoid actual cache operations in unit tests

### Temporary Resources

- **Temporary directories**: Each test gets isolated temp storage
- **Test databases**: SQLite databases created per test
- **Cache isolation**: Caches cleared between tests

### Test Data

- **Premium hands**: 23 top poker hands for fast testing
- **Common hands**: ~50 hands for medium coverage
- **All hands**: 169 complete preflop combinations
- **Test scenarios**: Realistic poker situations

## 🏃‍♂️ Performance Expectations

### Test Speed Targets

| Test Category | Target Time | Description |
|---------------|-------------|-------------|
| Unit Tests | < 5 seconds | Configuration, scenario generation |
| Integration Tests | < 15 seconds | Cache manager integration |
| Performance Tests | < 30 seconds | Timing validation |
| Full Suite | < 60 seconds | All cache population tests |

### Performance Validation

The tests validate that:

- **Scenario generation**: >1000 scenarios/second
- **Individual simulation**: <0.01 seconds per scenario
- **Full population**: Respects timeout limits
- **Memory usage**: Reasonable resource consumption

## 🔍 Debugging and Troubleshooting

### Common Issues

#### **Import Errors**
```bash
# Check if cache population module is available
python -c "from poker_knight.storage.cache_prepopulation import PopulationConfig; print('✅ Available')"
```

#### **Test Failures**
```bash
# Run with extra verbose output
pytest --cache-population -vv

# Run specific failing test
pytest tests/test_cache_population.py::TestPopulationConfig::test_default_config -vv

# Run with debugger on failure
pytest --cache-population --pdb
```

#### **Performance Issues**
```bash
# Check test durations
python tests/run_cache_population_tests.py --durations

# Profile specific tests
pytest --cache-population --profile-svg
```

### Debug Options

```bash
# Enable debug logging
POKER_KNIGHT_LOG_LEVEL=DEBUG pytest --cache-population

# Skip slow tests
pytest --cache-population -m "not performance"

# Run only fast unit tests
pytest --cache-population -m unit
```

## 📈 Continuous Integration

### CI Pipeline Integration

```yaml
# Example GitHub Actions integration
- name: Cache Population Tests
  run: |
    python tests/run_cache_population_tests.py --coverage
    
- name: Upload Coverage
  uses: codecov/codecov-action@v1
  with:
    file: tests/results/cache_population_coverage/coverage.xml
```

### Test Matrix

| Python Version | Cache Backend | Test Suite |
|----------------|---------------|------------|
| 3.8+ | Memory only | Unit tests |
| 3.8+ | SQLite | Integration tests |
| 3.8+ | Redis (if available) | Full integration |

## 🎛️ Configuration Options

### Environment Variables

```bash
# Skip cache population tests entirely
export SKIP_CACHE_POPULATION_TESTS=1

# Use different cache path for tests
export POKER_KNIGHT_TEST_CACHE_PATH=/tmp/test_cache.db

# Enable extra debugging
export POKER_KNIGHT_DEBUG_CACHE_POPULATION=1
```

### Test Configuration Files

```python
# Custom test configuration
# tests/cache_population_test_config.py
TEST_CONFIG = {
    "preflop_hands": "premium_only",  # Faster tests
    "max_population_time_minutes": 1,  # Quick timeout
    "opponent_counts": [1, 2],  # Fewer combinations
    "positions": ["button"]  # Single position
}
```

## 📚 Best Practices

### Writing New Tests

1. **Use appropriate markers**: `@pytest.mark.cache_population`
2. **Mock external dependencies**: Avoid actual cache operations in unit tests
3. **Clean up resources**: Use setUp/tearDown for temp files
4. **Test edge cases**: Invalid configurations, missing dependencies
5. **Validate performance**: Include timing assertions where relevant

### Test Organization

1. **Group related tests**: Use test classes for logical grouping
2. **Clear naming**: Test names should describe what they validate
3. **Comprehensive coverage**: Test both success and failure paths
4. **Independent tests**: Each test should be runnable in isolation

### Example Test Template

```python
@pytest.mark.cache
@pytest.mark.cache_population
class TestNewFeature(unittest.TestCase):
    """Test new cache population feature."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        clear_all_caches()
    
    def tearDown(self):
        """Clean up test fixtures."""
        clear_all_caches()
        shutil.rmtree(self.temp_dir)
    
    def test_feature_functionality(self):
        """Test that the feature works correctly."""
        # Test implementation
        pass
    
    def test_feature_error_handling(self):
        """Test feature error handling."""
        # Error case testing
        pass
```

## 🔄 Maintenance

### Regular Tasks

1. **Update test data**: Keep poker scenarios current
2. **Performance baseline**: Update timing expectations
3. **Dependency updates**: Test with new cache/solver versions
4. **Coverage monitoring**: Maintain high test coverage

### Version Compatibility

- **Backward compatibility**: Tests should work with older cache versions
- **Forward compatibility**: Prepare for future cache enhancements
- **Cross-platform**: Ensure tests work on Windows/Linux/macOS

---

## 📞 Support

For questions about the cache population test suite:

1. **Check this documentation** for common scenarios
2. **Run with verbose output** to see detailed test execution
3. **Use the dedicated test runner** for easier debugging
4. **Check existing test patterns** in similar test files

The test suite is designed to be comprehensive, fast, and maintainable while providing excellent coverage of the cache pre-population system. 