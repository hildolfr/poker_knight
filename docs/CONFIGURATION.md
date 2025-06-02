# Configuration Guide

Comprehensive guide to configuring Poker Knight for optimal performance and customizing its behavior for your specific use case.

## Configuration File Structure

Edit `poker_knight/config.json` to customize Poker Knight's behavior. The configuration system uses hierarchical defaults with user override capabilities:

```json
{
  "simulation_settings": {
    "default_simulations": 100000,
    "fast_mode_simulations": 10000,
    "precision_mode_simulations": 500000,
    "parallel_processing": true,
    "random_seed": null,
    "max_workers": 4,
    "parallel_processing_threshold": 50000,
    "stratified_sampling": true,
    "importance_sampling": true,
    "variance_reduction": true
  },
  "performance_settings": {
    "max_simulation_time_ms": 5000,
    "timeout_fast_mode_ms": 1000,
    "timeout_default_mode_ms": 5000,
    "timeout_precision_mode_ms": 15000,
    "early_convergence_threshold": 0.001,
    "min_simulations_for_convergence": 1000,
    "adaptive_timeout": true,
    "convergence_check_interval": 10000
  },
  "tournament_settings": {
    "enable_icm_calculations": true,
    "default_bubble_factor": 1.0,
    "position_awareness": true,
    "multi_way_analysis": true
  },
  "output_settings": {
    "include_confidence_interval": true,
    "include_hand_categories": true,
    "include_convergence_stats": true,
    "decimal_precision": 4,
    "verbose_logging": false
  }
}
```

## Configuration Sections

### Simulation Settings

#### Core Simulation Parameters

**`default_simulations` (default: 100000)**
Number of Monte Carlo simulations for "default" mode analysis.
- **Lower values (10k-50k)**: Faster execution, reduced accuracy
- **Higher values (200k-1M)**: Slower execution, increased accuracy
- **Recommended range**: 50,000 - 200,000
- **Impact on accuracy**: ±0.5% at 100k simulations vs theoretical values

**`fast_mode_simulations` (default: 10000)**
Number of simulations for "fast" mode (real-time AI decision making).
- **Typical range**: 5,000 - 25,000
- **Use case**: Live gameplay, AI bot decisions requiring <100ms response
- **Accuracy trade-off**: ±1-2% vs theoretical, sufficient for most real-time decisions

**`precision_mode_simulations` (default: 500000)**
Number of simulations for "precision" mode (critical tournament decisions).
- **Typical range**: 200,000 - 1,000,000
- **Use case**: Tournament final tables, ICM-critical spots, research applications
- **Accuracy**: ±0.2% vs theoretical values, suitable for professional analysis

#### Advanced Sampling Techniques

**`stratified_sampling` (default: true)**
Enable intelligent variance reduction using board texture stratification.
- **true**: Reduces simulation variance by 15-30% with minimal overhead
- **false**: Standard random sampling (faster initialization, higher variance)
- **Recommended**: Keep enabled unless memory is extremely constrained

**`importance_sampling` (default: true)**
Enable weighted sampling focusing on critical decision scenarios.
- **true**: Improves accuracy for close equity spots by 20-40%
- **false**: Uniform sampling across all possible outcomes
- **Best for**: Marginal hands, bubble situations, close tournament decisions

**`variance_reduction` (default: true)**
Enable control variate techniques for mathematical variance reduction.
- **true**: Uses analytical baselines to reduce simulation noise
- **false**: Pure Monte Carlo without variance reduction
- **Performance impact**: <5% overhead for 20-30% variance improvement

#### Parallel Processing Configuration

**`parallel_processing` (default: true)**
Enable multi-threaded execution using ThreadPoolExecutor.
- **true**: Utilize multiple CPU cores for faster execution
- **false**: Single-threaded execution (useful for debugging, limited environments)
- **Performance gain**: 2-4x speedup on modern multi-core systems

**`max_workers` (default: 4)**
Maximum number of worker threads for parallel processing.
- **Recommended**: Number of CPU cores (2-8 for typical systems)
- **Higher values**: Diminishing returns due to Python GIL limitations
- **Lower values**: More conservative resource usage

**`parallel_processing_threshold` (default: 50000)**
Minimum simulation count required to enable parallel processing.
- **Lower values (10k-30k)**: More aggressive parallelization
- **Higher values (50k-100k)**: Reduced thread overhead for small analyses
- **Tuning**: Balance thread creation overhead vs. parallel benefit

**`random_seed` (default: null)**
Fixed random seed for reproducible results.
- **null**: Different results each run (recommended for production)
- **Integer value**: Fixed seed for testing, debugging, and reproducible research
- **Use cases**: Unit testing, algorithm validation, academic research

### Performance Settings

#### Timeout Configuration

**`max_simulation_time_ms` (default: 5000)**
Global timeout for any analysis operation to prevent runaway simulations.
- **Lower values (1-3 seconds)**: Faster maximum response, potential incomplete analysis
- **Higher values (10-30 seconds)**: More complete analysis, longer wait times
- **Real-time apps**: Set to 500-1000ms for responsive user experience

**Mode-Specific Timeouts:**

- **`timeout_fast_mode_ms` (1000ms)**: Quick decision timeout for AI bots
- **`timeout_default_mode_ms` (5000ms)**: Balanced analysis timeout for general use
- **`timeout_precision_mode_ms` (15000ms)**: Extended timeout for high-precision analysis

#### Convergence Detection

**`early_convergence_threshold` (default: 0.001)**
Statistical threshold for early simulation termination when convergence is detected.
- **Lower values (0.0005)**: More precise convergence detection, longer runtime
- **Higher values (0.005)**: Earlier termination, faster results, less precision
- **Adaptive**: System automatically adjusts based on scenario complexity

**`min_simulations_for_convergence` (default: 1000)**
Minimum simulations before enabling convergence checking.
- **Purpose**: Prevents premature termination with insufficient sample sizes
- **Range**: 500-5000 depending on required statistical confidence
- **Statistical basis**: Ensures Central Limit Theorem applicability

**`adaptive_timeout` (default: true)**
Enable intelligent timeout adjustment based on scenario complexity.
- **true**: Automatically extends timeouts for complex multi-way scenarios
- **false**: Fixed timeouts regardless of scenario difficulty
- **Benefit**: Prevents timeout in genuinely complex situations requiring more computation

**`convergence_check_interval` (default: 10000)**
Frequency of convergence checking during simulation runs.
- **Lower values (5k)**: More frequent checks, slight performance overhead
- **Higher values (20k)**: Less frequent checks, potential delayed termination
- **Optimization**: Balance convergence detection speed vs. computational overhead

### Tournament Settings

#### ICM and Advanced Features

**`enable_icm_calculations` (default: true)**
Enable Independent Chip Model calculations for tournament equity.
- **true**: Calculate tournament equity, bubble factors, stack pressure
- **false**: Disable ICM features for cash game focus
- **Features enabled**: Bubble pressure adjustment, stack-to-pot ratio analysis

**`default_bubble_factor` (default: 1.0)**
Default bubble pressure multiplier when none specified.
- **1.0**: No bubble pressure (early tournament or cash game)
- **1.2-1.5**: Moderate bubble pressure (approaching money bubble)
- **1.5-2.0**: High bubble pressure (final table, pay jumps)

**`position_awareness` (default: true)**
Enable position-based equity adjustments for multi-way analysis.
- **true**: Calculate position advantages/disadvantages
- **false**: Position-neutral analysis only
- **Impact**: Early position penalty, late position bonus in equity calculations

**`multi_way_analysis` (default: true)**
Enable advanced statistics for 3+ opponent scenarios.
- **true**: Calculate coordination effects, defense frequencies, range interactions
- **false**: Basic equity calculation only
- **Advanced features**: Bluff-catching optimization, range coordination modeling

### Output Settings

#### Result Formatting and Detail Level

**`include_confidence_interval` (default: true)**
Calculate and include 95% confidence intervals in results.
- **true**: Provide statistical confidence bounds (slight performance cost)
- **false**: Skip confidence intervals for faster execution
- **Statistical basis**: Normal approximation with finite population correction

**`include_hand_categories` (default: true)**
Track frequency distribution of hand types achieved.
- **true**: Detailed breakdown by pairs, straights, flushes, etc.
- **false**: Skip hand categories for minimal memory usage
- **Use cases**: Training data generation, detailed post-mortem analysis

**`include_convergence_stats` (default: true)**
Include convergence analysis and sampling efficiency metrics.
- **true**: Report convergence statistics, effective sample size
- **false**: Basic results only
- **Professional use**: Quality assessment of simulation results

**`decimal_precision` (default: 4)**
Number of decimal places for probability calculations and display.
- **Range**: 2-6 decimal places
- **Higher values**: More precision display, slightly increased memory usage
- **Professional analysis**: 4-5 decimals recommended for tournament play

**`verbose_logging` (default: false)**
Enable detailed logging of simulation progress and internal statistics.
- **true**: Comprehensive logging for debugging and analysis
- **false**: Minimal logging for production use
- **Use cases**: Algorithm development, performance optimization, troubleshooting

## Performance Optimization Profiles

### Real-Time AI Applications

Optimized for poker bots requiring fast decisions with acceptable accuracy:

```json
{
  "simulation_settings": {
    "fast_mode_simulations": 15000,
    "parallel_processing": true,
    "max_workers": 4,
    "stratified_sampling": true,
    "importance_sampling": false,
    "variance_reduction": false
  },
  "performance_settings": {
    "timeout_fast_mode_ms": 500,
    "early_convergence_threshold": 0.005,
    "adaptive_timeout": false,
    "convergence_check_interval": 5000
  },
  "tournament_settings": {
    "enable_icm_calculations": true,
    "position_awareness": true,
    "multi_way_analysis": false
  },
  "output_settings": {
    "include_confidence_interval": false,
    "include_hand_categories": false,
    "include_convergence_stats": false,
    "decimal_precision": 3
  }
}
```

**Expected Performance:** <100ms for most scenarios, ±1-2% accuracy

### High-Precision Tournament Analysis

Optimized for critical tournament decisions requiring maximum accuracy:

```json
{
  "simulation_settings": {
    "precision_mode_simulations": 1000000,
    "parallel_processing": true,
    "max_workers": 8,
    "stratified_sampling": true,
    "importance_sampling": true,
    "variance_reduction": true
  },
  "performance_settings": {
    "timeout_precision_mode_ms": 30000,
    "early_convergence_threshold": 0.0005,
    "adaptive_timeout": true,
    "convergence_check_interval": 25000
  },
  "tournament_settings": {
    "enable_icm_calculations": true,
    "position_awareness": true,
    "multi_way_analysis": true
  },
  "output_settings": {
    "include_confidence_interval": true,
    "include_hand_categories": true,
    "include_convergence_stats": true,
    "decimal_precision": 5,
    "verbose_logging": true
  }
}
```

**Expected Performance:** 5-30 seconds, ±0.1% accuracy with full ICM analysis

### Memory-Constrained Environments

Optimized for systems with limited RAM or embedded applications:

```json
{
  "simulation_settings": {
    "default_simulations": 50000,
    "parallel_processing": false,
    "stratified_sampling": false,
    "importance_sampling": false,
    "variance_reduction": false
  },
  "performance_settings": {
    "early_convergence_threshold": 0.002,
    "convergence_check_interval": 25000
  },
  "tournament_settings": {
    "multi_way_analysis": false
  },
  "output_settings": {
    "include_hand_categories": false,
    "include_convergence_stats": false,
    "decimal_precision": 3,
    "verbose_logging": false
  }
}
```

**Memory Usage:** <50MB, suitable for embedded systems and minimal environments

### Research and Development

Optimized for algorithm development and validation:

```json
{
  "simulation_settings": {
    "random_seed": 42,
    "stratified_sampling": true,
    "importance_sampling": true,
    "variance_reduction": true
  },
  "performance_settings": {
    "adaptive_timeout": true
  },
  "output_settings": {
    "include_confidence_interval": true,
    "include_hand_categories": true,
    "include_convergence_stats": true,
    "decimal_precision": 6,
    "verbose_logging": true
  }
}
```

**Features:** Reproducible results, comprehensive statistics, detailed logging

## Custom Configuration Usage

### Loading Custom Configuration

```python
from poker_knight import MonteCarloSolver

# Load custom configuration file
solver = MonteCarloSolver(config_path="tournament_config.json")

# Use with custom settings
result = solver.analyze_hand(['A♠️', 'K♠️'], 2, simulation_mode="precision")
print(f"High-precision result: {result.win_probability:.4f}")
```

### Runtime Configuration Validation

Poker Knight validates configuration on startup with clear error messages:

```python
# Invalid configuration will raise descriptive errors
try:
    solver = MonteCarloSolver(config_path="invalid_config.json")
except ValueError as e:
    print(f"Configuration error: {e}")
    # Example: "invalid_config.json: max_workers must be positive integer (got -1)"
```

### Configuration Inheritance

Custom configurations inherit from defaults, allowing partial overrides:

```json
{
  "simulation_settings": {
    "fast_mode_simulations": 20000
  },
  "output_settings": {
    "decimal_precision": 5
  }
}
```

This configuration only overrides specific settings while maintaining all other defaults.

## Performance Tuning Guidelines

### CPU Core Utilization

**Optimal Worker Count:**
```python
import os
optimal_workers = min(8, os.cpu_count())  # Cap at 8 due to GIL limitations
```

**Memory vs. Speed Trade-offs:**
- **More simulations**: Better accuracy, higher memory usage
- **Parallel processing**: Faster execution, more memory per thread
- **Variance reduction**: Better accuracy, slight computational overhead

### Scenario-Specific Optimization

**Pre-flop scenarios:** Lower simulation counts sufficient (accuracy naturally higher)
**Complex board textures:** Higher simulation counts recommended
**Multi-way pots (4+ players):** Enable full multi-way analysis
**Heads-up play:** Disable multi-way features for faster execution

### Monitoring Performance

```python
result = solver.analyze_hand(['A♠️', 'K♠️'], 2)
print(f"Simulations: {result.simulations_run:,}")
print(f"Execution time: {result.execution_time_ms:.1f}ms")
print(f"Convergence achieved: {result.convergence_achieved}")
if result.convergence_details:
    print(f"Effective sample size: {result.convergence_details['effective_sample_size']:.0f}")
``` 