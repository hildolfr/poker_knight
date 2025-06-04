# Cache Behavior Analysis Summary

## Issue Description
Tests are expecting identical results from cache but seeing differences of 0.0013 to 0.0091 between cached and non-cached results.

## Root Cause Analysis

### What's Actually Happening
1. **Cache is working correctly**: When a result is cached and retrieved, it returns the exact same values
2. **Monte Carlo variance is normal**: Independent simulation runs produce slightly different results due to randomness
3. **No random seed is set**: The config has `"random_seed": null`, so each simulation uses different random numbers

### Test Results
- Cache hit/miss behavior: ✅ Working correctly
- Cached result retrieval: ✅ Returns exact stored values
- Independent run variance: 
  - Preflop scenarios: 0.0000 (lucky identical results due to rounding)
  - Postflop scenarios: 0.0014-0.0045 (normal variance)

### Code Analysis
1. When caching (solver.py lines 702-743):
   - Results are rounded to 4 decimal places before storing
   - All data is stored in cache exactly as computed

2. When retrieving from cache (solver.py lines 387-431):
   - Cached values are returned with same rounding
   - No new simulation is run

3. Monte Carlo simulations:
   - No fixed random seed
   - Each run produces slightly different results
   - Variance is expected and normal

## The Real Problem
**Tests are incorrectly expecting Monte Carlo simulations to be deterministic**. They should:
1. Test that cache returns exact stored values (this works)
2. Allow for statistical variance when comparing independent runs

## Recommendations
1. **Do NOT change the caching implementation** - it's working correctly
2. **Fix the tests** to properly handle Monte Carlo variance:
   - When testing cache hits, verify exact match with stored value
   - When comparing independent runs, use tolerance (e.g., within 0.01 or 1%)
   - Consider the confidence intervals already calculated by the solver

3. **Optional**: Add a test-only mode with fixed random seed for deterministic testing

## Example Test Fix
```python
# Instead of:
assert result1.win_probability == result2.win_probability

# Use:
if is_cached:
    # Cached results should be exact
    assert result1.win_probability == result2.win_probability
else:
    # Independent runs can vary within tolerance
    assert abs(result1.win_probability - result2.win_probability) < 0.01
```