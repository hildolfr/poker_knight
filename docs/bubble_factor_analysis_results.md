# Bubble Factor Analysis Results

## Executive Summary

The bubble_factor parameter in Poker Knight's tournament ICM calculations is currently working but has significant limitations. Testing confirms that bubble_factor does impact ICM equity calculations, but the current implementation has a hard floor at 0.7x base equity and requires manual specification of the bubble_factor value.

## Key Findings

### 1. Bubble Factor Impact on ICM Equity

Testing with AA vs 3 opponents shows clear impact:

| Bubble Factor | Win Probability | ICM Equity | ICM/Win Ratio |
|---------------|-----------------|------------|---------------|
| 1.0           | 0.6282         | 0.6282     | 1.000         |
| 1.5           | 0.6282         | 0.5340     | 0.850         |
| 2.0           | 0.6282         | 0.4397     | 0.700         |
| 3.0           | 0.6282         | 0.4397     | 0.700         |

**Key Observations:**
- Win probability remains constant (correct behavior - bubble doesn't change hand equity)
- ICM equity decreases with higher bubble_factor values
- There's a hard floor at 0.7x base equity (bubble_factor 2.0 and 3.0 produce identical results)

### 2. Stack Position Effects

With bubble_factor = 2.0, different stack positions show different ICM adjustments:

| Stack Position | Hero Stack | Win Prob | ICM Equity | Stack Pressure |
|----------------|------------|----------|------------|----------------|
| Chip Leader    | 10,000     | 0.6282   | 0.4397     | 0.375          |
| Medium Stack   | 3,000      | 0.6282   | 0.4645     | 0.812          |
| Short Stack    | 1,000      | 0.6282   | 0.4920     | 0.937          |

**Key Observations:**
- Short stacks get a boost to ICM equity (risk premium for survival)
- Stack pressure correctly influences final ICM calculations
- The system accounts for both bubble_factor and stack position

### 3. Current Implementation Details

From `poker_knight/simulation/multiway.py`:

```python
# Current bubble adjustment formula
if bubble_factor > 1.0:
    bubble_adjustment = max(0.7, 1.0 - (bubble_factor - 1.0) * 0.3)
    base_icm_equity *= bubble_adjustment
```

**Limitations:**
1. Hard floor at 0.7x equity (max reduction of 30%)
2. Linear adjustment that doesn't scale well
3. Requires manual bubble_factor specification
4. No automatic calculation based on tournament state

### 4. SimulationResult Attributes

The SimulationResult class includes these ICM-related fields:
- `icm_equity`: Tournament chip equity (adjusted win probability)
- `bubble_factor`: Bubble pressure adjustment value
- `stack_to_pot_ratio`: SPR for decision making
- `tournament_pressure`: Dictionary with stack pressure metrics

Note: There is no `expected_value` attribute - ICM calculations use `icm_equity`.

## Proposed Improvements

The `proposed_bubble_factor_redesign.py` file contains an automatic bubble_factor calculation that would:

1. **Automatically calculate bubble_factor** when not explicitly provided
2. **Use multiple factors:**
   - Stack depth relative to average
   - BB count estimation
   - Number of remaining players
   - Stack distribution variance
3. **Provide more nuanced adjustments** without hard floors
4. **Scale from 1.0 to 3.0** based on tournament conditions

## Recommendations

1. **Implement automatic bubble_factor calculation** to make ICM more accessible
2. **Remove or adjust the 0.7x floor** for more realistic bubble pressure modeling
3. **Add logging/transparency** for bubble_factor calculations
4. **Consider caching** bubble_factor calculations for similar tournament states
5. **Add unit tests** specifically for bubble_factor edge cases

## Test Scripts

Two test scripts were created:
1. `test_bubble_factor_behavior_fixed.py` - Detailed testing with debug output
2. `test_bubble_factor_summary.py` - Clean summary of findings

Both scripts demonstrate that the bubble_factor system is functional but could benefit from the proposed enhancements for more realistic tournament modeling.