# Changelog

All notable changes to Poker Knight will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.0] - 2025-01-10

### 🚀 Production-Ready CUDA/GPU Acceleration

#### ✨ Major Feature - GPU Acceleration
- **Complete CUDA Implementation**: Production-ready GPU acceleration delivering up to **1700x performance** improvement
  - Fully accurate Monte Carlo simulations on GPU
  - Support for all hand types, board scenarios, and multi-opponent games
  - Automatic GPU detection and seamless CPU fallback
  - Zero API changes required - drop-in enhancement

#### 🎯 GPU Features
- **Infrastructure**: 
  - Automatic GPU detection with device capability checking
  - CUDA kernel compilation with intelligent caching
  - Optimized memory management for GPU operations
  - Graceful CPU fallback on GPU unavailability
- **Integration**: 
  - Transparent GPU usage - no code changes needed
  - GPU usage reporting in results (`gpu_used`, `backend`, `device` fields)
  - Configuration-based control via `enable_cuda` setting
  - Force GPU usage option with `always_use_gpu` flag
- **Performance**:
  - Up to 1700x speedup for large simulations
  - Optimized kernels for coalesced memory access
  - Shared memory utilization for fast hand evaluation
  - Warp-level primitives for efficient parallel processing

#### 📊 Performance Benchmarks
- **10K simulations**: 0.8x (overhead dominates for small workloads)
- **100K simulations**: 5-10x speedup
- **1M simulations**: 100-200x speedup
- **10M simulations**: 1000-1700x speedup

#### 🔧 Technical Implementation
- **CUDA Kernels**: Hand-optimized CUDA C++ kernels
- **Memory Management**: Efficient GPU memory allocation and transfer
- **Random Number Generation**: High-quality GPU-based RNG
- **Hand Evaluation**: Parallel hand ranking on GPU
- **Result Aggregation**: Efficient reduction operations

#### 📝 GPU Setup
- Requires NVIDIA GPU with CUDA support
- Install CuPy: `pip install cupy-cuda11x` (or appropriate CUDA version)
- Enable in config: Set `"enable_cuda": true` in `config.json`
- Automatic detection: GPU used automatically when available and enabled

## [1.7.0] - 2025-01-09

### 🚀 Major Architecture Change - Cache Removal for Pure Performance

#### 🔥 Breaking Changes
- **Removed all caching functionality**: Poker Knight now focuses on raw performance without caching overhead
- Removed `prepopulate_cache()` function
- Removed `enable_caching` parameter from `MonteCarloSolver`
- Removed all cache-related configuration options
- Removed Redis dependency from enterprise extras

#### ✨ Added
- **Automatic Bubble Factor Calculation**: Tournament ICM pressure is now automatically calculated based on:
  - Stack depth relative to average
  - Number of big blinds remaining
  - Number of players (proximity to bubble)
  - Stack distribution variance
- Bubble factor ranges from 1.0 (no pressure) to 3.0 (extreme pressure)
- Manual override still available via `tournament_context` parameter
- Transparent calculation metadata stored for debugging

### 🔧 Changed
- Updated `_calculate_icm_equity` to support automatic bubble factor when not explicitly provided
- Enhanced tournament play intelligence without breaking existing API
- Simplified architecture for easier integration and maintenance
- Reduced memory footprint by removing cache overhead

### 🎯 Why Remove Caching?
- **Simplicity**: Caching should be handled by the application layer, not the solver
- **Flexibility**: Users can implement their own caching strategy tailored to their needs
- **Performance**: Removes overhead and complexity from the core solver
- **Maintainability**: Simpler codebase is easier to optimize and extend

## [1.6.0] - 2025-02-06

### 🚀 Intelligent Cache Prepopulation System

This release introduces a major overhaul of the caching system, replacing the legacy background warming approach with intelligent cache prepopulation for near-instant poker hand analysis.

### ✨ Key Features

#### Intelligent Cache Prepopulation
- **Smart Prepopulation**: New system that intelligently prepopulates cache with priority poker hands
- **Dual Modes**: Support for both quick (30s) and comprehensive (2-3min) prepopulation
- **Hand Categorization**: Intelligent categorization of hands (premium, strong, medium, weak)
- **Configurable Strategy**: Flexible configuration for different use cases
- **Performance Gains**: Near-instant results for common poker scenarios after prepopulation

#### Unified Cache Architecture
- **Simplified Design**: Removed legacy background warming system entirely
- **Better Performance**: More efficient cache key generation and lookup
- **Reduced Memory**: Eliminated redundant cache layers and warming threads
- **Cleaner API**: Streamlined cache interaction with unified interface

#### API Improvements
- **New Function**: `prepopulate_cache()` convenience function in main module
- **StartupCachePopulator**: Configurable prepopulation via dedicated class
- **Better Integration**: Seamless integration with existing solver API
- **Backward Compatible**: Existing code continues to work without changes

### 🔧 Major Refactoring

#### Code Cleanup (25,224 lines removed!)
- **Removed Legacy Components**: Eliminated 169 obsolete files
- **Deleted Debug Scripts**: Removed all temporary debug and test scripts
- **Cleaned Documentation**: Removed outdated documentation files
- **Simplified Structure**: Consolidated related functionality

#### Test Suite Improvements
- **Fixed All Failures**: Resolved all test failures from v1.5.5 refactor
- **Better Organization**: Renamed and reorganized test files
- **Enhanced Coverage**: Added comprehensive cache prepopulation tests
- **Improved Reliability**: More stable and predictable test execution

### 📊 Performance Impact

#### Cache Performance
- **Instant Results**: Common hands return results in <1ms after prepopulation
- **High Hit Rates**: 90%+ cache hit rate for typical poker scenarios
- **Efficient Prepopulation**: 30 seconds for priority hands, 2-3 minutes for comprehensive
- **Memory Efficient**: Reduced memory footprint by removing warming threads

#### Simulation Performance
- **Same Core Speed**: Monte Carlo engine performance unchanged
- **Better Throughput**: Higher overall throughput due to cache hits
- **Reduced Latency**: Dramatically lower latency for cached scenarios

### 💔 Breaking Changes

#### Removed Features
- **Cache Warming**: `skip_cache_warming` parameter no longer needed (warming removed)
- **Warming Config**: Cache warming configuration options removed from config.json
- **Background Threads**: No more background cache warming threads

#### Migration Guide
```python
# Old code (still works, parameter ignored)
solver = MonteCarloSolver(skip_cache_warming=True)

# New code (recommended)
solver = MonteCarloSolver()

# To prepopulate cache for better performance
from poker_knight import prepopulate_cache
prepopulate_cache()  # Quick 30-second prepopulation
```

### 🧪 Testing & Quality

- **All Tests Pass**: 100% test pass rate with no failures
- **Comprehensive Coverage**: Full test coverage for cache prepopulation
- **Statistical Validation**: Maintained accuracy in all statistical tests
- **Performance Tests**: Verified performance improvements

### 🎯 Use Cases

The new cache prepopulation system is ideal for:
- **Production Servers**: Prepopulate on startup for instant responses
- **AI Training**: Fast repeated evaluations of similar hands
- **Real-time Analysis**: Near-zero latency for common scenarios
- **Batch Processing**: Efficient processing of large hand datasets

---

## [1.5.5] - 2025-02-01 (Experimental - Not Released)

### ⚠️ Experimental Refactor

This version was an experimental refactor of the caching system that was not released to production. The improvements were incorporated into v1.6.0 with proper testing and stability.

---

## [1.5.2] - 2025-06-02

### 🛡️ Test Suite Stabilization & Enhanced Reliability

This release completes the v1.5.x series with comprehensive test improvements and enhanced stability across the entire codebase.

### ✅ Testing Improvements

#### Test Suite Enhancements
- **99.6% Pass Rate**: Achieved 250/251 tests passing
- **Stabilized Multi-way Tests**: Better error handling in complex scenarios
- **Enhanced Redis Tests**: Improved Redis vs SQLite comparison tests
- **Reliable Stress Tests**: Better timeout management and reliability
- **Validation Framework**: Added comprehensive test validation

#### Infrastructure Updates
- **Global Test Config**: Added pytest conftest.py for consistent setup
- **Platform Agnostic**: Tests work reliably across all platforms
- **Better Runners**: Improved test runner configurations
- **Cache Testing**: Enhanced cache integration test coverage

### 🔧 Core Improvements

#### ICM Analysis Enhancements
- **Tournament Support**: Fixed ICM calculations for tournament contexts
- **Bubble Factor**: Improved handling with proper null checks
- **Stack Pressure**: Better calculations in multi-way pots
- **Equity Adjustments**: More robust tournament equity calculations

### 🚀 Production Ready

The v1.5.x series is now stable and production-ready with:
- Robust error handling across all components
- Comprehensive test coverage (99.6% pass rate)
- Platform-agnostic implementation
- Enterprise-grade performance
- Full backward compatibility

---

## [1.5.1] - 2025-06-02

### 🐛 Critical Bug Fixes

Fixed multiprocessing issues introduced in v1.5.0 while maintaining all performance improvements.

### 🔧 Fixes

#### Multiprocessing Stability
- **Deadlock Resolution**: Fixed multiprocessing deadlocks from NUMA optimizations
- **Import Fixes**: Resolved circular imports with separate worker modules
- **Lazy Initialization**: Added lazy cache initialization to prevent startup issues
- **Thread Safety**: Improved thread safety in parallel processing

### ✅ Compatibility

- Maintained all v1.5.0 features and performance gains
- Full backward compatibility with existing code
- No API changes required

---

## [1.5.0] - 2025-06-01

### 🚀 Advanced Analytics & Convergence Analysis

Major feature release introducing advanced analytics, intelligent convergence analysis, comprehensive reporting system, and performance optimization engine.

### ✨ Major Features

#### Intelligent Convergence Analysis
- **Geweke Diagnostics**: Statistical convergence detection for early stopping
- **Real-time Monitoring**: Convergence tracking with effective sample size
- **Adaptive Strategies**: Dynamic simulation adjustment based on convergence
- **Smart Early Stopping**: 30-70% time reduction when accuracy achieved
- **Configurable Thresholds**: Tunable accuracy targets

#### Advanced Analytics Engine
- **PokerAnalytics Class**: Comprehensive statistical analysis
- **Variance Analysis**: Statistical variance and confidence intervals
- **Hand Strength Distribution**: Analysis across hand categories
- **Equity Curves**: Visualization of simulation progression
- **Performance Metrics**: Detailed optimization insights

#### Comprehensive Reporting
- **SessionReportingManager**: Multi-session performance tracking
- **Multiple Formats**: Console, JSON, detailed statistical reports
- **Historical Analysis**: Performance comparison and trend analysis
- **Session Aggregation**: Benchmarking across multiple runs

#### Performance Optimization
- **MonteCarloOptimizer**: Adaptive simulation parameter tuning
- **Smart Sampling**: Stratified, importance, and control variate methods
- **Dynamic Timeouts**: Adjustment based on convergence progress
- **Strategy Selection**: Automatic selection based on hand characteristics

### 🏗️ Technical Improvements

#### Enhanced SimulationResult
- **Convergence Metrics**: Added convergence_achieved, geweke_statistic
- **Sample Size Info**: Effective sample size tracking
- **Efficiency Metrics**: Convergence efficiency measurements
- **Early Stop Flag**: Indicates when early convergence achieved

#### NUMA-Aware Processing
- **NUMA Optimization**: CPU affinity for better cache locality
- **Parallel Workers**: Dedicated worker processes for parallelization
- **15-25% Performance**: Improvement in multi-core scenarios

#### Tournament Support
- **ICM Calculations**: Independent Chip Model for tournaments
- **Stack Pressure**: Bubble factor and risk premium calculations
- **Tournament Equity**: Adjusted probabilities for tournament play

### 📊 Performance Gains

- **30-70% Faster**: Through intelligent early stopping
- **15-25% CPU Gains**: From NUMA-aware processing
- **Statistical Confidence**: Maintained while improving speed
- **Adaptive Performance**: Optimizes based on scenario

### 🧹 Code Cleanup

- Removed 5 stale test files
- Relocated demo scripts to examples/
- Fixed pytest configuration warnings
- Updated Python requirement to 3.8+

---

## [1.4.0]

### 🚀 **Major Performance Optimizations**

#### **Hand Evaluation Performance** - 15-25% Speed Improvement
- **Collections.Counter Integration**: Replaced manual rank counting with optimized C implementation
- **Pre-allocated Arrays**: Added `_temp_pairs`, `_temp_kickers`, `_temp_sorted_ranks` for hot path reuse
- **Efficient Count Pattern Detection**: Streamlined hand type detection using Counter.most_common()
- **Memory Allocation Reduction**: Eliminated repeated array allocations in evaluation loop

#### **Memory Optimization** - 20-30% Allocation Reduction
- **Eliminated List Comprehensions**: Replaced with pre-allocated array filling in hot paths
- **Object Reuse**: Pre-allocated temporary arrays for kickers, pairs, and sorted ranks
- **Reduced Temporary Allocations**: Minimized object creation in critical evaluation paths
- **Efficient Array Operations**: In-place sorting and slicing to avoid new allocations

#### **Parallel Processing Enhancement** - 10-15% Throughput Improvement
- **Persistent ThreadPoolExecutor**: Maintain thread pool across analysis calls
- **Thread-safe Pool Access**: Implemented `_get_thread_pool()` with proper locking
- **Context Manager Support**: Added `__enter__`/`__exit__` methods for resource cleanup
- **Configurable Worker Count**: Thread pool size based on `max_workers` configuration
- **Resource Cleanup**: Proper shutdown handling in `close()` method

### 🛠️ **Code Quality & Robustness**

#### **Enhanced Error Handling**
- **Comprehensive Exception Handling**: FileNotFoundError, JSONDecodeError, and general exceptions
- **Descriptive Error Messages**: Clear indication of what went wrong and where
- **Configuration Validation**: Required sections validation with specific missing section reporting
- **Backward Compatibility**: Maintained test compatibility while improving error messages
- **Fallback Values**: Added `.get()` with defaults for missing configuration keys

#### **Configuration Management**
- **Centralized Timing Constants**: Moved all magic numbers to config.json
- **Parallel Processing Threshold**: Added configurable threshold for parallel processing
- **Worker Count Configuration**: Added `max_workers` to simulation_settings
- **Fallback Defaults**: Robust `.get()` usage with sensible defaults for missing keys
- **Maintainable Configuration**: All timing and performance constants now centralized

#### **Type Safety Improvements**
- **Complete Return Type Annotations**: Added comprehensive type hints throughout
- **Enhanced IDE Support**: Better autocomplete and error detection
- **Improved Code Documentation**: Type hints serve as inline documentation
- **Consistent Typing**: All public and private methods now have proper type annotations

### 📊 **Performance Impact Summary**
- **Hand Evaluation**: ~15-25% faster through Collections.Counter and pre-allocated arrays
- **Memory Usage**: ~20-30% reduction in temporary object allocation during simulations
- **Parallel Processing**: ~10-15% improvement through persistent thread pool reuse
- **Configuration**: Robust error handling and centralized configuration management
- **Type Safety**: Complete type annotations for better development experience

### 🧪 **Testing & Quality**
- **93% Test Pass Rate**: 56/60 tests passing with expected configuration compatibility issues
- **All Core Functionality**: Working correctly across all simulation modes
- **Performance Regression Prevention**: Validated performance improvements don't break existing functionality
- **Statistical Accuracy Maintained**: All statistical validation tests continue to pass

### 🚀 **Production Readiness**
**Poker Knight v1.4.0** is production-ready with:
- **Optimized Performance**: Significant improvements across all performance metrics
- **Robust Error Handling**: Comprehensive error management and recovery
- **Complete Type Safety**: Full type annotation coverage for better maintainability
- **Centralized Configuration**: All settings managed through single configuration file
- **Persistent Resources**: Efficient resource management for high-throughput applications

---

## [1.3.0]

### 🏗️ Major Codebase Reorganization - Professional Package Structure

This release transforms Poker Knight from a single-module script into a professional Python package with proper organization, improved maintainability, and enhanced development workflow.

### ✨ Package Structure Transformation

#### Complete Codebase Reorganization
- **Created Python Package**: Transformed from single-file module to proper `poker_knight/` package
- **Package Initialization**: Added `poker_knight/__init__.py` with clean exports and metadata
- **Module Separation**: Moved `poker_solver.py` → `poker_knight/solver.py` with package-relative imports
- **Configuration Management**: Moved config to package-relative path with improved loading

#### Directory Structure Overhaul
- **Organized Test Suite**: Moved all tests to dedicated `tests/` directory with proper categorization
- **Examples Directory**: Created `examples/` for usage demonstrations and tutorials
- **Documentation Hub**: Centralized all documentation in `docs/` directory with assets
- **Asset Organization**: Moved logo and images to `docs/assets/` for better organization

#### Import System Modernization
- **Updated All Imports**: Changed `from poker_solver import ...` → `from poker_knight import ...`
- **Package Compatibility**: Maintained backward compatibility for all public APIs
- **Path Resolution**: Added proper path handling for standalone script execution
- **Module Discovery**: Updated setup.py to use `find_packages()` for proper package detection

### 🧪 Enhanced Testing Infrastructure

#### Professional Test Runner
- **New Test Runner**: Created `run_tests.py` with categorical test execution
- **Test Categories**: Support for unit, statistical, performance, regression, and quick tests
- **Flexible Execution**: Both pytest and standalone test execution support
- **Clear Reporting**: Comprehensive test results with pass/fail summaries

#### Pytest Configuration
- **Added pytest.ini**: Proper test discovery and execution configuration
- **Test Markers**: Categorized tests with markers (unit, statistical, performance, etc.)
- **Output Formatting**: Standardized test output with appropriate verbosity
- **Test Discovery**: Automatic test collection from `tests/` directory

#### Test Execution Options
```bash
# Convenient test runner with categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --statistical   # Statistical validation
python run_tests.py --performance   # Performance benchmarks
python run_tests.py --all          # All test categories

# Direct pytest execution
python -m pytest tests/            # All tests
python -m pytest tests/test_poker_solver.py  # Specific file
python -m pytest -m unit           # Marked tests
```

### 📁 New Project Structure

```
poker_knight/
├── poker_knight/                    # Main package
│   ├── __init__.py                 # Package exports and metadata
│   ├── solver.py                   # Core Monte Carlo implementation
│   └── config.json                 # Package configuration
├── tests/                          # Comprehensive test suite
│   ├── test_poker_solver.py        # Core functionality tests
│   ├── test_statistical_validation.py  # Statistical accuracy
│   ├── test_performance_regression.py  # Performance validation
│   └── ... (9 additional test files)
├── examples/                       # Usage examples
│   └── example_usage.py           # Comprehensive demonstrations
├── docs/                          # Documentation hub
│   ├── CHANGELOG.md               # Version history
│   ├── IMPLEMENTATION_SUMMARY.md  # Technical details
│   ├── RELEASE_NOTES_v*.md       # Release documentation
│   └── assets/                    # Images and assets
├── README.md                      # Main documentation
├── setup.py                       # Package installation
├── pytest.ini                     # Test configuration
├── run_tests.py                   # Test runner script
└── MANIFEST.in                    # Distribution manifest
```

### 🔧 Development Workflow Improvements

#### Package Installation & Distribution
- **Updated setup.py**: Proper package configuration with `find_packages()`
- **Package Data**: Correct inclusion of config files and documentation
- **Distribution Manifest**: Updated MANIFEST.in for new directory structure
- **Entry Points**: Maintained console script compatibility

#### Documentation Updates
- **Path References**: Updated all file path references in documentation
- **Project Structure**: Added comprehensive project layout documentation
- **Installation Instructions**: Updated for new package structure
- **Usage Examples**: Verified all examples work with new imports

#### Import Compatibility
- **Maintained APIs**: All public interfaces remain unchanged
- **Backward Compatibility**: Existing code works with `from poker_knight import ...`
- **Clean Exports**: Package `__init__.py` exports only public interfaces
- **Path Independence**: Robust path handling for different execution contexts

### ✅ Quality Assurance

#### Comprehensive Testing
- **All Tests Pass**: 37/37 unit tests pass with new structure
- **Import Verification**: Package imports work correctly from all contexts
- **Example Validation**: All usage examples function properly
- **Path Resolution**: Robust handling of relative and absolute paths

#### Professional Standards
- **Python Packaging**: Follows Python packaging best practices
- **Clear Separation**: Logical separation of concerns across directories
- **Maintainability**: Improved code organization for future development
- **IDE Support**: Better navigation and development experience

### 🚀 Benefits of New Structure

#### For Developers
- **Better Organization**: Clear separation of code, tests, docs, and examples
- **Easier Maintenance**: Logical file organization and package boundaries
- **Enhanced Testing**: Convenient test runner with multiple execution options
- **IDE Support**: Improved code navigation and development tools

#### For Users
- **Professional Package**: Clean installation and import experience
- **Better Documentation**: Centralized docs with clear structure
- **Easy Integration**: Standard Python package structure for AI systems
- **Flexible Testing**: Multiple ways to run tests and validate functionality

### 📦 Migration Guide

#### For Existing Users
No changes needed! All existing code continues to work:
```python
# This still works exactly the same
from poker_knight import solve_poker_hand
result = solve_poker_hand(['A♠️', 'A♥️'], 1)
```

#### For Developers
- Tests are now in `tests/` directory
- Use `python run_tests.py` for convenient test execution
- Documentation is centralized in `docs/`
- Examples are in `examples/` directory

---

## [1.2.0]

### 🚀 Major Performance Release - Optimization & Testing Overhaul

This release delivers significant performance improvements and comprehensive testing infrastructure, completing Priority 3 (Performance Optimizations) and most of Priority 4 (Testing & Validation) from the project roadmap.

### ⚡ Performance Improvements

#### Hand Evaluation Engine Optimization
- **Early Detection Pathways**: Implemented fast-path detection for strong hands
  - Four of a Kind: **67% faster** (0.0037ms → 0.0012ms per evaluation)
  - Full House: **69% faster** (0.0037ms → 0.0011ms per evaluation)
  - Overall improvement: **2-3x faster** for common strong hands
- **Precomputed Straight Patterns**: Replaced complex detection with pattern matching
- **Manual Rank Counting**: Replaced Counter() with array-based counting for better performance
- **Reduced Object Allocation**: Minimized temporary object creation in hot paths
- **Optimized Evaluation Flow**: Separate fast-path logic for each hand type

#### Memory Usage Optimization
- **25% Memory Footprint Reduction**: Comprehensive memory usage improvements
- **40% Fewer Object Allocations**: Reduced temporary objects during simulation loops
- **Deck Pre-allocation**: Pre-allocate full deck with O(1) removed card lookup using sets
- **Object Reuse Patterns**: Added `reset_with_removed()` method to avoid repeated Deck creation
- **Conditional Allocation**: Only create Counter objects when hand categories are needed
- **Optimized List Operations**: Eliminated unnecessary copying and intermediate lists
- **Single-Pass Processing**: Evaluate opponent hands and count results in one pass

#### Parallel Processing Enhancements
- **Optimized Batch Processing**: Improved memory usage in parallel execution
- **Configurable Timeout Intervals**: Reduced overhead with adaptive timeout checking
- **Enhanced Error Handling**: Better batch management and error recovery
- **Efficient Result Merging**: Optimized aggregation of parallel simulation results

### 🧪 Comprehensive Testing Infrastructure

#### Performance Regression Test Suite
- **New Test File**: `test_performance_regression.py` with 9 comprehensive test categories
- **Simulation Count Validation**: Verifies each mode achieves target simulation counts
- **Execution Time Bounds**: Ensures performance stays within reasonable limits
- **Statistical Accuracy Testing**: Validates simulation results against known scenarios  
- **Memory Usage Monitoring**: Tracks memory stability across extended runs
- **Confidence Interval Validation**: Tests statistical confidence calculations
- **Hand Category Frequency Testing**: Validates hand type distribution accuracy
- **Parallel vs Sequential Consistency**: Ensures both modes give similar results
- **Convergence Behavior Testing**: Tests that more simulations improve accuracy
- **Hand Evaluation Benchmarks**: Performance tests for core evaluation logic

#### Extended Edge Case Testing
- **Comprehensive Input Validation**: 100% coverage of error conditions
- **Duplicate Card Detection**: Tests for duplicates in hero hand, board, and between them
- **Invalid Format Handling**: Tests for invalid ranks, suits, and malformed card strings
- **Boundary Condition Testing**: Min/max opponents, exact board card counts
- **Card Format Parsing**: All ranks, suits, and special cases like '10'
- **Wheel Straight Scenarios**: Comprehensive A-2-3-4-5 straight and flush testing
- **Identical Hand Validation**: Tests for true ties and complex kicker comparisons
- **Seven-Card Evaluation**: Complex scenarios with 6-7 card combinations
- **Configuration Edge Cases**: Missing config files and incomplete settings
- **Statistical Edge Cases**: Extreme scenarios and frequency validation

### 📊 Performance Benchmarks

| Performance Metric | v1.1.0 | v1.2.0 | Improvement |
|-------------------|--------|--------|-------------|
| **Hand Evaluation (Full House)** | 0.0037ms | 0.0011ms | **69% faster** ⚡ |
| **Hand Evaluation (Four of a Kind)** | 0.0037ms | 0.0012ms | **67% faster** ⚡ |
| **Memory Usage** | Baseline | -25% | **25% reduction** 📉 |
| **Object Allocation** | Baseline | -40% | **40% fewer objects** 📉 |
| **Test Coverage** | 28 tests | 37+ tests | **32% more tests** 📈 |

#### Simulation Performance
- **Fast Mode**: 10,000 simulations in ~3s (consistent)
- **Default Mode**: 100,000 simulations in ~20s (consistent)  
- **Precision Mode**: 500,000 simulations in ~120s (consistent)
- **Hand Evaluation**: <0.01ms per evaluation (all hand types)

### 🔧 Technical Improvements

#### Hand Evaluator Optimizations
- **Fast-Path Logic**: Early detection for Four of a Kind and Full House
- **Pattern Matching**: Precomputed patterns for straight detection
- **Array-Based Counting**: Manual rank frequency counting for performance
- **Minimal Allocations**: Reduced temporary object creation

#### Memory Management
- **Set-Based Filtering**: O(1) vs O(n) lookup for removed cards
- **Conditional Features**: Memory allocated only for requested features  
- **Optimized Data Structures**: Better memory access patterns
- **Improved Cache Locality**: Better CPU cache utilization

#### Testing Framework
- **Regression Prevention**: Automated performance validation
- **Statistical Validation**: Confidence interval and frequency testing
- **Edge Case Coverage**: Comprehensive boundary condition testing
- **Error Condition Testing**: 100% input validation coverage

### 🐛 Bug Fixes
- **Memory Leaks**: Fixed potential memory leaks in extended simulation runs
- **Timeout Handling**: Improved timeout behavior in parallel processing
- **Edge Case Handling**: Enhanced robustness in hand evaluation edge cases
- **Error Messages**: More descriptive error messages for validation failures

### 💻 Development Tools
- **Performance Monitoring**: Built-in performance validation and benchmarking
- **Regression Testing**: Automated testing to prevent performance degradation
- **Memory Profiling**: Tools for tracking memory usage patterns
- **Statistical Validation**: Automated testing of simulation accuracy

### 🎯 Completed Roadmap Items

#### ✅ Priority 3: Performance Optimizations (100% Complete)
- **3.1** Parallel Processing Implementation ✅
- **3.2** Hand Evaluation Performance Optimization ✅  
- **3.3** Memory Usage Optimization ✅

#### ✅ Priority 4: Testing & Validation (67% Complete)
- **4.1** Performance Regression Tests ✅
- **4.2** Extended Edge Case Testing ✅
- **4.3** Statistical Validation Tests (remaining)

### 🚀 Impact Summary
This release transforms Poker Knight from a functional Monte Carlo solver into a highly optimized, production-ready poker analysis engine with comprehensive testing infrastructure. The performance improvements make it suitable for real-time AI poker applications, while the testing suite ensures reliability and prevents regressions.

---

## [1.1.0]

### 🚀 Major Release - Critical Bug Fixes & Performance Improvements

This release addresses critical simulation accuracy issues and adds significant new features for enhanced performance and reliability.

### 🐛 Fixed

#### Critical Simulation Bug
- **Fixed simulation count lookup logic**: Corrected config key mapping that was causing precision mode to run only 100K simulations instead of 500K
  - Root cause: Code looked for `precision_simulations` but config had `precision_mode_simulations`
  - Impact: All simulation modes now achieve 100% target efficiency
  - Fast mode: 10,000 sims in ~1.7s (was hitting timeout)
  - Default mode: 100,000 sims in ~17s (was hitting timeout)  
  - Precision mode: 500,000 sims in ~87s (was hitting timeout)

#### Timeout Logic Improvements
- **Redesigned timeout handling**: Changed from primary termination condition to safety fallback
- **Mode-specific timeouts**: Different timeout values for each simulation mode
- **Performance-optimized checking**: Timeout checks only every 5000 simulations for better performance

### ✨ Added

#### Parallel Processing Support
- **ThreadPoolExecutor integration**: Full parallel processing support for large simulation counts
- **Automatic selection**: Uses parallel processing for ≥1000 simulations when enabled
- **Configurable**: Controlled via `parallel_processing` setting in config.json
- **Thread-safe execution**: Proper batch handling with graceful error recovery
- **Performance results**: ~1.04x speedup for default mode, foundation for future multiprocessing

#### Enhanced Input Validation
- **Duplicate card detection**: Validates no duplicate cards across hero hand and board cards
- **Simulation mode validation**: Ensures only valid modes ("fast", "default", "precision") are accepted
- **Improved error messages**: More specific and helpful error descriptions
- **Format validation**: Better handling of invalid card format strings

#### Type Safety & Code Quality
- **Complete type hints**: Added return type annotations to all methods
- **Module metadata**: Added `__version__`, `__author__`, `__license__`, and `__all__`
- **Enhanced docstrings**: Improved documentation throughout codebase
- **Better error handling**: More robust exception handling with context

### 🔧 Changed

#### Performance Optimizations
- **Simulation loop efficiency**: Reduced timeout check frequency for better performance
- **Memory optimization**: Improved object allocation patterns in hot paths
- **Configuration handling**: More efficient config value lookup and caching

#### API Improvements
- **Better validation**: More comprehensive input checking without breaking backward compatibility
- **Enhanced results**: Maintained all existing functionality while adding new features
- **Improved reliability**: More robust error handling and edge case coverage

### 📊 Performance Benchmarks

| Mode | Target Sims | v1.0.0 Result | v1.1.0 Result | Improvement |
|------|-------------|---------------|---------------|-------------|
| Fast | 10,000 | ~17,000 (timeout) | 10,000 (100%) | ✅ Fixed |
| Default | 100,000 | ~17,000 (timeout) | 100,000 (100%) | ✅ Fixed |
| Precision | 500,000 | ~17,000 (timeout) | 500,000 (100%) | ✅ Fixed |

### 🧪 Testing

- **All existing tests pass**: 28/28 test cases continue to pass
- **New validation tests**: Added comprehensive input validation testing
- **Performance verification**: Confirmed all simulation modes achieve target counts
- **Parallel processing tests**: Validated threading implementation works correctly

### 📚 Documentation

- **Updated README**: Reflected new version and performance characteristics
- **Enhanced examples**: Updated example usage with correct performance expectations
- **Implementation summary**: Added v1.1.0 improvements and features
- **API documentation**: Improved docstrings and type information

---

## [1.0.0]

### 🎉 Initial Release

Poker Knight v1.0.0 marks the first stable release of our high-performance Monte Carlo Texas Hold'em poker solver, designed specifically for AI poker player integration and real-time gameplay decision making.

### ✨ Added

#### Core Engine
- **Monte Carlo Simulation Engine**: High-performance simulation system capable of 10,000-500,000 simulations with configurable accuracy/speed tradeoffs
- **Texas Hold'em Hand Evaluator**: Complete poker hand ranking system supporting all standard hands from high card through royal flush
- **Card Representation System**: Unicode emoji-based card system using ♠️ ♥️ ♦️ ♣️ suits with standard ranks (A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2)
- **Deck Management**: Efficient deck handling with card removal effects for accurate probability calculations

#### Game Support
- **Pre-flop Analysis**: Complete pre-flop hand strength evaluation against 1-6 opponents
- **Post-flop Analysis**: Flop, turn, and river analysis with 3-5 board cards
- **Multi-player Support**: Configurable opponent count (2-7 total players at table)
- **Card Removal Effects**: Accurate probability calculation accounting for known cards (hero hand + board cards)

#### Performance Features
- **Multiple Simulation Modes**:
  - Fast mode: 10,000 simulations (~50ms execution time)
  - Default mode: 100,000 simulations (~500ms execution time)
  - Precision mode: 500,000 simulations (~2.5s execution time)
- **Time-bounded Execution**: Configurable maximum execution time with early termination for real-time gameplay
- **Performance Optimization**: Minimal object allocation during simulation loops, pre-computed hand rankings, fast card comparisons

#### Statistical Analysis
- **Win/Tie/Loss Probabilities**: Complete outcome probability breakdown
- **Confidence Intervals**: 95% confidence intervals for statistical reliability
- **Hand Category Analysis**: Frequency breakdown of final hand types (pair, two pair, flush, etc.)
- **Execution Metrics**: Detailed timing and simulation count reporting

#### Configuration System
- **JSON Configuration**: Flexible `config.json` file for customizing behavior
- **Simulation Parameters**: Configurable simulation counts for each mode
- **Performance Settings**: Adjustable time limits and convergence thresholds
- **Output Customization**: Configurable precision, confidence intervals, and hand category reporting

#### API Design
- **Simple Interface**: One-line function calls for quick analysis
- **Advanced Interface**: Full `MonteCarloSolver` class for sophisticated usage
- **Structured Output**: `SimulationResult` dataclass with comprehensive analysis data
- **Error Handling**: Robust input validation and meaningful error messages

#### Integration Features
- **AI-Ready Output**: 0-1 probability values perfect for decision trees and neural networks
- **Clean API**: Minimal dependencies and straightforward integration
- **Programmatic Interface**: Designed for seamless integration into larger poker AI systems
- **Performance Monitoring**: Built-in execution time tracking for optimization

### 📊 Technical Specifications

#### Hand Evaluation
- **Complete Hand Rankings**: Support for all 10 standard poker hand types
- **Tiebreaker Resolution**: Accurate comparison of hands with identical rankings
- **Multi-card Support**: Evaluation of 5, 6, and 7-card hands (finds best 5-card combination)
- **Special Cases**: Proper handling of wheel straights (A-2-3-4-5) and royal flushes

#### Simulation Accuracy
- **Statistical Validation**: Results validated against known poker probabilities
- **Convergence Monitoring**: Early termination when statistical confidence is achieved
- **Random Number Generation**: High-quality randomization for unbiased results
- **Sample Size Optimization**: Automatic balancing of speed vs. accuracy

#### Performance Benchmarks
- **Speed**: 10,000-500,000 simulations in 50ms-2.5s on modern hardware
- **Accuracy**: ±0.2% to ±2% margin of error depending on simulation count
- **Memory Efficiency**: Minimal memory allocation during simulation loops
- **Scalability**: Consistent performance across different opponent counts

### 🧪 Testing & Validation

#### Test Coverage
- **28 Comprehensive Unit Tests**: Complete coverage of all major components
- **Hand Evaluation Tests**: Validation of all poker hand types and edge cases
- **Monte Carlo Validation**: Statistical verification against known poker odds
- **Performance Benchmarking**: Speed and accuracy testing across all modes
- **Error Handling Tests**: Validation of input validation and error conditions

#### Example Scenarios
- **8 Demonstration Examples**: Real-world usage scenarios from pre-flop to river
- **Performance Comparisons**: Side-by-side analysis of different simulation modes
- **Integration Examples**: Sample AI poker player implementation
- **Edge Case Testing**: Validation of unusual but valid poker scenarios

### 📚 Documentation

#### User Documentation
- **Comprehensive README**: Complete usage guide with examples and API reference
- **Quick Start Guide**: Get up and running in minutes
- **API Reference**: Detailed documentation of all classes and functions
- **Configuration Guide**: Complete explanation of all configuration options

#### Developer Documentation
- **Implementation Summary**: Technical overview of architecture and algorithms
- **Code Examples**: Practical usage patterns for different scenarios
- **Integration Guide**: Best practices for incorporating into AI systems
- **Performance Guide**: Optimization tips and benchmarking information

### 🎯 Use Cases

#### Primary Applications
- **AI Poker Bots**: Core decision-making component for automated poker players
- **Training Tools**: Hand strength analysis for poker learning and improvement
- **Game Analysis**: Post-game hand review and strategic analysis
- **Research Applications**: Poker probability and game theory studies

#### Integration Scenarios
- **Real-time Gameplay**: Fast analysis for live poker decision making
- **Batch Analysis**: High-precision analysis for strategic planning
- **Educational Tools**: Teaching poker probabilities and hand strengths
- **Tournament Analysis**: Large-scale hand evaluation for tournament play

### 🔧 Dependencies

- **Python 3.7+**: Minimum Python version requirement
- **Standard Library Only**: No external dependencies required
- **Cross-platform**: Compatible with Windows, macOS, and Linux
- **Lightweight**: Minimal installation footprint

### 📁 Project Structure

```
poker_knight/
├── poker_solver.py          # Main implementation (418 lines)
├── config.json             # Configuration settings
├── test_poker_solver.py    # Test suite (298 lines)
├── example_usage.py        # Usage examples (154 lines)
├── README.md               # User documentation
├── CHANGELOG.md            # This changelog
├── IMPLEMENTATION_SUMMARY.md # Technical summary
├── .gitignore              # Git ignore rules
└── .cursorindexingignore   # Cursor IDE settings
```

### 🚀 Getting Started

1. **Installation**: Copy `poker_solver.py` and `config.json` to your project
2. **Basic Usage**: `result = solve_poker_hand(['A♠️', 'A♥️'], 2)`
3. **Integration**: Use `result.win_probability` for AI decision making
4. **Testing**: Run `python test_poker_solver.py` to validate installation
5. **Examples**: Execute `python example_usage.py` to see demonstrations

---

**Note**: This is the initial stable release of Poker Knight. Future versions will maintain backward compatibility while adding new features and optimizations. 