# API Reference

Complete API documentation for Poker Knight's classes and functions.

## Main Functions

### `solve_poker_hand(hero_hand, num_opponents, board_cards=None, simulation_mode="default")`

Convenience function for quick analysis.

**Parameters:**
- `hero_hand`: List of 2 card strings (e.g., `['A♠️', 'K♥️']`)
- `num_opponents`: Number of opponents (1-6)
- `board_cards`: Optional list of 3-5 board cards
- `simulation_mode`: "fast", "default", or "precision"

**Returns:** `SimulationResult` object

**Example:**
```python
from poker_knight import solve_poker_hand

# Basic pre-flop analysis
result = solve_poker_hand(['A♠️', 'A♥️'], 2)

# Post-flop analysis with board cards
result = solve_poker_hand(
    ['K♠️', 'Q♠️'],           # Hero hand
    3,                        # Number of opponents  
    ['A♠️', 'J♠️', '10♥️']    # Board cards (flop)
)
```

## Classes

### `MonteCarloSolver`

Main Poker Knight solver class for advanced usage and configuration.

```python
from poker_knight import MonteCarloSolver

solver = MonteCarloSolver()
result = solver.analyze_hand(['A♠️', 'A♥️'], 2)
```

#### Methods

#### `__init__(config_path: Optional[str] = None)`

Initialize the solver with optional custom configuration.

**Parameters:**
- `config_path`: Optional path to custom config.json file

#### `analyze_hand(hero_hand, num_opponents, board_cards=None, simulation_mode="default")`

Analyze a poker hand with detailed configuration options.

**Parameters:**
- `hero_hand`: List of 2 card strings
- `num_opponents`: Number of opponents (1-6)
- `board_cards`: Optional list of 3-5 board cards
- `simulation_mode`: "fast", "default", or "precision"

**Returns:** `SimulationResult` object

#### `close()`

Clean up resources (important when using parallel processing).

### `SimulationResult`

Result object containing comprehensive analysis data.

```python
@dataclass
class SimulationResult:
    win_probability: float              # Probability of winning (0-1)
    tie_probability: float              # Probability of tying (0-1)  
    loss_probability: float             # Probability of losing (0-1)
    simulations_run: int                # Number of simulations executed
    execution_time_ms: float            # Execution time in milliseconds
    confidence_interval: Tuple[float, float]  # 95% confidence interval
    hand_category_frequencies: Dict[str, float]  # Hand type frequencies
```

#### Properties

- **`win_probability`**: Probability of winning the hand (0.0 to 1.0)
- **`tie_probability`**: Probability of tying with other players
- **`loss_probability`**: Probability of losing the hand
- **`simulations_run`**: Actual number of simulations executed
- **`execution_time_ms`**: Time taken for analysis in milliseconds
- **`confidence_interval`**: 95% confidence interval as (lower, upper) tuple
- **`hand_category_frequencies`**: Dictionary mapping hand types to their frequencies

## Card Format

Cards use Unicode emoji suits with standard ranks:

- **Suits**: ♠️ (spades), ♥️ (hearts), ♦️ (diamonds), ♣️ (clubs)
- **Ranks**: A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2

**Examples:**
- `'A♠️'` - Ace of spades
- `'K♥️'` - King of hearts  
- `'10♦️'` - Ten of diamonds
- `'2♣️'` - Two of clubs

## Usage Examples

### Pre-flop Analysis

```python
# Premium hand
result = solve_poker_hand(['A♠️', 'A♥️'], 1)
print(f"Pocket Aces: {result.win_probability:.1%}")

# Marginal hand
result = solve_poker_hand(['2♠️', '7♥️'], 5)
print(f"2-7 offsuit: {result.win_probability:.1%}")

# Suited connectors
result = solve_poker_hand(['9♠️', '8♠️'], 3)
print(f"9-8 suited: {result.win_probability:.1%}")
```

### Post-flop Analysis

```python
# Strong made hand
result = solve_poker_hand(
    ['A♠️', 'A♥️'],                    # Pocket aces
    2,                                 # 2 opponents
    ['A♦️', '7♠️', '2♣️']             # Flop (trip aces)
)
print(f"Trip aces: {result.win_probability:.1%}")

# Drawing hand
result = solve_poker_hand(
    ['A♠️', 'K♠️'],                    # Suited ace-king
    1,                                 # 1 opponent
    ['Q♠️', 'J♦️', '7♠️']             # Flop (flush + straight draws)
)
print(f"Nut flush draw: {result.win_probability:.1%}")

# Weak hand
result = solve_poker_hand(
    ['7♣️', '2♥️'],                    # Weak hand
    4,                                 # 4 opponents
    ['A♠️', 'K♦️', 'Q♠️']             # High flop
)
print(f"7-2 on AKQ: {result.win_probability:.1%}")
```

### Performance Modes

```python
# Fast analysis for real-time decisions
result = solve_poker_hand(['K♠️', 'K♥️'], 3, simulation_mode="fast")
print(f"Fast mode: {result.simulations_run} simulations, {result.execution_time_ms:.1f}ms")

# Default balanced analysis
result = solve_poker_hand(['K♠️', 'K♥️'], 3, simulation_mode="default")
print(f"Default mode: {result.simulations_run} simulations, {result.execution_time_ms:.1f}ms")

# High precision for critical decisions
result = solve_poker_hand(['K♠️', 'K♥️'], 3, simulation_mode="precision")
print(f"Precision mode: {result.simulations_run} simulations, {result.execution_time_ms:.1f}ms")
```

### Advanced Analysis

```python
# Access detailed hand categories
result = solve_poker_hand(['A♠️', 'K♠️'], 2, ['Q♠️', 'J♠️', '10♥️'])

print("Hand category frequencies:")
for category, frequency in result.hand_category_frequencies.items():
    print(f"  {category}: {frequency:.3f}")

# Confidence intervals
lower, upper = result.confidence_interval
print(f"Win probability: {result.win_probability:.3f} ({lower:.3f}-{upper:.3f})")
```

### Error Handling

```python
from poker_knight import solve_poker_hand

try:
    # This will raise an error - duplicate cards
    result = solve_poker_hand(['A♠️', 'A♠️'], 2)
except ValueError as e:
    print(f"Error: {e}")

try:
    # This will raise an error - invalid card format
    result = solve_poker_hand(['AS', 'KH'], 2)
except ValueError as e:
    print(f"Error: {e}")

try:
    # This will raise an error - too many opponents
    result = solve_poker_hand(['A♠️', 'K♠️'], 10)
except ValueError as e:
    print(f"Error: {e}")
```

## Hand Categories

The following hand categories are tracked in `hand_category_frequencies`:

1. **"High Card"** - No pair, straight, or flush
2. **"One Pair"** - Single pair
3. **"Two Pair"** - Two different pairs
4. **"Three of a Kind"** - Three cards of same rank
5. **"Straight"** - Five consecutive ranks (including wheel: A-2-3-4-5)
6. **"Flush"** - Five cards of same suit
7. **"Full House"** - Three of a kind plus a pair
8. **"Four of a Kind"** - Four cards of same rank
9. **"Straight Flush"** - Both straight and flush
10. **"Royal Flush"** - A-K-Q-J-10 all same suit 