# Poker Knight v1.5.x Release Notes

## Version 1.5.2 (2025-06-02)
### Stability and Test Suite Improvements
- Enhanced test suite reliability and coverage
- Fixed ICM analysis edge cases in tournament contexts
- Improved test runner configurations and validation
- Better handling of Redis/SQLite demo scenarios
- Comprehensive test integration improvements
- 99.6% test pass rate (250/251 tests passing)

### Key Changes
- Updated ICM calculations to handle tournament contexts more robustly
- Enhanced multi-way scenario testing
- Improved stress test reliability
- Better cache integration test coverage
- Fixed minor issues in demo scripts

## Version 1.5.1 (2025-06-02)
### Critical Bug Fixes - Multiprocessing and Import Issues
- **Fixed circular imports** by moving multiprocessing workers to separate module (`core/parallel_workers.py`)
- **Resolved deadlocks** with lazy cache initialization preventing issues during module import
- **Fixed suit format compatibility** - now supports both 'S' and '♠' formats seamlessly
- **Set multiprocessing start method** to 'spawn' for better process safety
- **Reorganized project structure**: 
  - Moved tests to `tests/` directory
  - Moved documentation to `docs/` directory
  - Cleaned up temporary files and debugging utilities
- **Maintained all v1.5.0 features** while restoring stability

### Technical Details
- Separated worker processes from main parallel module to avoid circular dependencies
- Implemented lazy initialization for cache backends
- Added proper multiprocessing context management
- All NUMA and SMP optimizations preserved

## Version 1.5.0 (2025-06-01)
### Major Feature Release - Advanced Analytics & Performance

#### New Features
1. **Advanced Analytics System**
   - Real-time convergence monitoring with statistical analysis
   - Confidence intervals and standard error tracking
   - Convergence rate analysis and visualization
   - Detailed simulation metrics and insights

2. **NUMA-Aware Processing**
   - CPU affinity optimization for NUMA systems
   - Intelligent work distribution across NUMA nodes
   - Memory locality improvements
   - Automatic fallback for non-NUMA systems

3. **Enhanced Parallel Processing**
   - Optimized ThreadPoolExecutor configuration
   - Smart batch sizing based on hand complexity
   - Improved work stealing algorithms
   - Better CPU utilization across all cores

4. **Tournament ICM Support**
   - Independent Chip Model calculations
   - Bubble factor analysis
   - Stack pressure considerations
   - Tournament-specific equity adjustments

#### Performance Improvements
- **15-25% faster** hand evaluation
- **20-30% reduced** memory usage
- **10-15% improved** parallel processing efficiency
- **Near-instant results** with intelligent cache pre-population
- **93%+ test coverage** maintained

#### Technical Enhancements
- Complete type safety with comprehensive type hints
- Enhanced error handling and recovery
- Centralized configuration management
- Improved logging and debugging capabilities
- Platform-agnostic implementation

### Migration Notes
- v1.5.x maintains full backward compatibility with v1.4.x
- New features are opt-in via configuration
- Default behavior unchanged for existing integrations