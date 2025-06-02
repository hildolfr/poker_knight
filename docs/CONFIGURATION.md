# Configuration Guide

Learn how to configure Poker Knight for optimal performance and customize its behavior for your specific use case.

## Configuration File

Edit `poker_knight/config.json` to customize Poker Knight's behavior:

```json
{
  "simulation_settings": {
    "default_simulations": 100000,
    "fast_mode_simulations": 10000,
    "precision_mode_simulations": 500000,
    "parallel_processing": true,
    "random_seed": null,
    "max_workers": 4,
    "parallel_processing_threshold": 50000
  },
  "performance_settings": {
    "max_simulation_time_ms": 5000,
    "timeout_fast_mode_ms": 1000,
    "timeout_default_mode_ms": 5000,
    "timeout_precision_mode_ms": 15000,
    "early_convergence_threshold": 0.001,
    "min_simulations_for_convergence": 1000
  },
  "output_settings": {
    "include_confidence_interval": true,
    "include_hand_categories": true,
    "decimal_precision": 4
  }
}
```

## Configuration Sections

### Simulation Settings

#### `default_simulations` (default: 100000)
Number of simulations for "default" mode.
- **Lower values**: Faster execution, less accuracy
- **Higher values**: Slower execution, more accuracy
- **Recommended range**: 50,000 - 200,000

#### `fast_mode_simulations` (default: 10000)
Number of simulations for "fast" mode (real-time decisions).
- **Typical range**: 5,000 - 25,000
- **Use case**: Real-time AI decision making

#### `precision_mode_simulations` (default: 500000)
Number of simulations for "precision" mode (critical decisions).
- **Typical range**: 200,000 - 1,000,000
- **Use case**: Tournament final tables, critical spots

#### `parallel_processing` (default: true)
Enable/disable parallel processing using multiple CPU cores.
- **true**: Use ThreadPoolExecutor for faster execution
- **false**: Single-threaded execution

#### `max_workers` (default: 4)
Maximum number of worker threads for parallel processing.
- **Recommended**: Number of CPU cores available
- **Higher values**: Diminishing returns due to GIL limitations

#### `parallel_processing_threshold` (default: 50000)
Minimum simulations required to use parallel processing.
- **Lower values**: More parallel processing usage
- **Higher values**: Less overhead for small analyses

#### `random_seed` (default: null)
Fixed random seed for reproducible results.
- **null**: Different results each run (recommended for production)
- **Integer**: Fixed seed for testing and debugging

### Performance Settings

#### `max_simulation_time_ms` (default: 5000)
Global timeout for any analysis operation.
- **Lower values**: Faster timeouts, potential incomplete analysis
- **Higher values**: More complete analysis, longer waits

#### Mode-Specific Timeouts

- **`timeout_fast_mode_ms`**: 1000ms - Quick decision timeout
- **`timeout_default_mode_ms`**: 5000ms - Balanced analysis timeout  
- **`timeout_precision_mode_ms`**: 15000ms - High precision timeout

#### `early_convergence_threshold` (default: 0.001)
Threshold for early simulation termination when convergence is detected.
- **Lower values**: More precise convergence detection
- **Higher values**: Earlier termination, less precision

#### `min_simulations_for_convergence` (default: 1000)
Minimum simulations before checking for convergence.
- **Purpose**: Prevents premature termination with small samples

### Output Settings

#### `include_confidence_interval` (default: true)
Whether to calculate and include confidence intervals in results.
- **true**: Include confidence intervals (slight performance cost)
- **false**: Skip confidence intervals for faster execution

#### `include_hand_categories` (default: true)
Whether to track hand category frequencies.
- **true**: Include detailed hand type breakdown
- **false**: Skip hand categories for faster execution

#### `decimal_precision` (default: 4)
Number of decimal places for probability calculations.
- **Range**: 2-6 decimal places
- **Higher values**: More precision, slightly slower

## Performance Tuning

### Real-Time Applications

For AI poker bots requiring fast decisions:

```json
{
  "simulation_settings": {
    "fast_mode_simulations": 15000,
    "parallel_processing": true,
    "max_workers": 4
  },
  "performance_settings": {
    "timeout_fast_mode_ms": 500,
    "early_convergence_threshold": 0.005
  },
  "output_settings": {
    "include_confidence_interval": false,
    "include_hand_categories": false
  }
}
```

### High-Precision Analysis

For tournament play or critical decisions:

```json
{
  "simulation_settings": {
    "precision_mode_simulations": 1000000,
    "parallel_processing": true,
    "max_workers": 8
  },
  "performance_settings": {
    "timeout_precision_mode_ms": 30000,
    "early_convergence_threshold": 0.0005
  },
  "output_settings": {
    "include_confidence_interval": true,
    "include_hand_categories": true,
    "decimal_precision": 5
  }
}
```

### Memory-Constrained Environments

For limited memory environments:

```json
{
  "simulation_settings": {
    "default_simulations": 50000,
    "parallel_processing": false
  },
  "output_settings": {
    "include_hand_categories": false,
    "decimal_precision": 3
  }
}
```

## Custom Configuration

### Loading Custom Configuration

```python
from poker_knight import MonteCarloSolver

# Load custom configuration file
solver = MonteCarloSolver(config_path="my_custom_config.json")

# Use with custom settings
result = solver.analyze_hand(['A♠️', 'K♠️'], 2)
```

### Configuration Validation

Poker Knight validates configuration on startup:

```python
# Invalid configuration will raise clear errors
try:
    solver = MonteCarloSolver(config_path="invalid_config.json")
except ValueError as e:
    print(f"Configuration error: {e}")
```

## Simulation Mode Comparison

| Mode | Simulations | Timeout | Use Case | Accuracy |
|------|-------------|---------|----------|----------|
| Fast | 10,000 | 1s | Real-time AI | ±2-3% |
| Default | 100,000 | 5s | General analysis | ±1% |
| Precision | 500,000 | 15s | Critical decisions | ±0.5% |

## Performance Benchmarks

### Typical Execution Times

| Scenario | Fast Mode | Default Mode | Precision Mode |
|----------|-----------|--------------|----------------|
| Pre-flop 2 players | 50ms | 200ms | 800ms |
| Flop 4 players | 80ms | 350ms | 1400ms |
| River 6 players | 120ms | 500ms | 2000ms |

*Benchmarks on 4-core CPU with parallel processing enabled*

### Memory Usage

| Component | Memory Impact |
|-----------|---------------|
| Base solver | ~5MB |
| Simulation data | ~10MB per 100k simulations |
| Hand categories | ~1MB additional |
| Confidence intervals | ~0.5MB additional |

## Troubleshooting

### Performance Issues

**Slow execution:**
1. Enable parallel processing
2. Increase `max_workers` to match CPU cores
3. Reduce simulation counts for real-time use
4. Disable hand categories and confidence intervals

**High memory usage:**
1. Reduce simulation counts
2. Disable parallel processing
3. Disable hand categories tracking

### Configuration Errors

**File not found:**
```python
# Fallback to default configuration
solver = MonteCarloSolver()  # Uses default config
```

**Invalid JSON:**
- Validate JSON syntax using online tools
- Check for trailing commas and proper quoting

**Missing configuration sections:**
- Poker Knight provides sensible defaults for missing values
- Check error messages for specific missing configuration

## Best Practices

1. **Start with defaults** - The default configuration works well for most use cases
2. **Profile your use case** - Measure actual performance in your specific environment
3. **Test configuration changes** - Use the test suite to verify configuration changes
4. **Monitor memory usage** - Especially important for long-running applications
5. **Use appropriate simulation modes** - Don't use precision mode for real-time decisions

## Environment Variables

You can override configuration using environment variables:

```bash
# Override simulation counts
export POKER_KNIGHT_FAST_SIMULATIONS=20000
export POKER_KNIGHT_DEFAULT_SIMULATIONS=150000

# Override timeouts
export POKER_KNIGHT_FAST_TIMEOUT=2000
export POKER_KNIGHT_DEFAULT_TIMEOUT=8000
```

*Note: Environment variables take precedence over config file values* 