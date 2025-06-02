# API Reference

Complete API documentation for Poker Knight's classes and functions with advanced features.

## Main Functions

### `solve_poker_hand(hero_hand, num_opponents, board_cards=None, simulation_mode="default", **kwargs)`

Convenience function for comprehensive poker hand analysis including advanced tournament features.

**Parameters:**
- `hero_hand`: List of 2 card strings (e.g., `['A♠️', 'K♥️']`)
- `num_opponents`: Number of opponents (1-6)
- `board_cards`: Optional list of 3-5 board cards
- `simulation_mode`: "fast" (10k sims), "default" (100k sims), or "precision" (500k sims)

**Advanced Parameters:**
- `hero_position`: Optional position string ("early", "middle", "late", "button", "sb", "bb")
- `stack_sizes`: Optional list of stack sizes [hero, opp1, opp2, ...] for ICM analysis
- `pot_size`: Current pot size for stack-to-pot ratio calculations
- `tournament_context`: Dictionary with ICM settings (e.g., `{'bubble_factor': 1.2}`)

**Returns:** `SimulationResult` object with comprehensive analysis

**Basic Examples:**
```python
from poker_knight import solve_poker_hand

# Pre-flop analysis
result = solve_poker_hand(['A♠️', 'A♥️'], 2)
print(f"Pocket Aces: {result.win_probability:.1%}")

# Post-flop analysis with board cards
result = solve_poker_hand(
    ['K♠️', 'Q♠️'],           # Hero hand
    3,                        # Number of opponents  
    ['A♠️', 'J♠️', '10♥️']    # Board cards (flop)
)
print(f"AKQJ board: {result.win_probability:.1%}")
```

**Advanced Tournament Example:**
```python
# ICM-aware tournament analysis
result = solve_poker_hand(
    ['A♠️', 'K♠️'],                    # Hero hand
    2,                                 # Number of opponents
    ['Q♠️', 'J♠️', '10♥️'],            # Board cards (royal draw)
    simulation_mode="precision",        # High accuracy for critical decision
    hero_position="button",            # Position advantage
    stack_sizes=[15000, 8000, 12000],  # Stack sizes for ICM
    pot_size=2000,                     # Current pot
    tournament_context={               # ICM context
        'bubble_factor': 1.3,          # Bubble pressure
        'payout_structure': [0.5, 0.3, 0.2]  # Prize distribution
    }
)

print(f"Win probability: {result.win_probability:.1%}")
print(f"ICM equity: {result.icm_equity:.1%}")
print(f"Position advantage: {result.position_aware_equity['position_advantage']:.3f}")
```

## Classes

### `MonteCarloSolver`

Main Poker Knight solver class providing advanced configuration and analysis capabilities.

```python
from poker_knight import MonteCarloSolver

# Initialize with default configuration
solver = MonteCarloSolver()

# Initialize with custom configuration
solver = MonteCarloSolver(config_path="custom_config.json")

# Use context manager for automatic resource cleanup
with MonteCarloSolver() as solver:
    result = solver.analyze_hand(['A♠️', 'A♥️'], 2)
```

#### Methods

#### `__init__(config_path: Optional[str] = None)`

Initialize the solver with optional custom configuration.

**Parameters:**
- `config_path`: Optional path to custom config.json file

**Configuration Options:**
- Simulation counts for each mode (fast/default/precision)
- Parallel processing settings
- Performance timeouts
- Statistical confidence parameters

#### `analyze_hand(hero_hand, num_opponents, board_cards=None, simulation_mode="default", **kwargs)`

Comprehensive poker hand analysis with advanced multi-way and tournament features.

**Parameters:**
- `hero_hand`: List of 2 card strings
- `num_opponents`: Number of opponents (1-6)
- `board_cards`: Optional list of 3-5 board cards
- `simulation_mode`: "fast", "default", or "precision"
- `hero_position`: Position for equity adjustments
- `stack_sizes`: Stack sizes for ICM calculations
- `pot_size`: Current pot size
- `tournament_context`: ICM and tournament settings

**Returns:** `SimulationResult` object with comprehensive statistics

**Advanced Analysis Example:**
```python
# Multi-way pot with position and ICM analysis
result = solver.analyze_hand(
    ['K♠️', 'K♥️'],                    # Pocket kings
    4,                                 # 4 opponents (5-way pot)
    ['7♠️', '2♥️', '9♦️'],             # Dry flop
    simulation_mode="default",
    hero_position="early",             # Early position disadvantage
    stack_sizes=[20000, 15000, 18000, 12000, 25000],  # ICM stack info
    pot_size=3000,
    tournament_context={'bubble_factor': 1.5}  # High bubble pressure
)

# Access multi-way statistics
print(f"Multi-way equity: {result.win_probability:.1%}")
print(f"Defense frequency: {result.defense_frequencies['optimal_defense_frequency']:.1%}")
print(f"Coordination effects: {result.coordination_effects['total_coordination_effect']:.3f}")
```

#### `close()`

Clean up resources including thread pools (important when using parallel processing).

```python
solver = MonteCarloSolver()
try:
    result = solver.analyze_hand(['A♠️', 'K♠️'], 2)
finally:
    solver.close()  # Ensure proper cleanup
```

### `SimulationResult`

Comprehensive result object containing detailed analysis data and advanced tournament statistics.

```python
@dataclass
class SimulationResult:
    # Core probabilities
    win_probability: float              # Probability of winning (0-1)
    tie_probability: float              # Probability of tying (0-1)  
    loss_probability: float             # Probability of losing (0-1)
    
    # Execution statistics
    simulations_run: int                # Number of simulations executed
    execution_time_ms: float            # Execution time in milliseconds
    confidence_interval: Tuple[float, float]  # 95% confidence interval
    hand_category_frequencies: Dict[str, float]  # Hand type frequencies
    
    # Advanced tournament features
    position_aware_equity: Optional[Dict[str, float]]  # Position-based equity
    icm_equity: Optional[float]         # Tournament chip equity
    bubble_factor: Optional[float]      # Bubble pressure factor
    stack_to_pot_ratio: Optional[float] # SPR for decision making
    
    # Multi-way pot analysis
    multi_way_statistics: Optional[Dict[str, Any]]     # 3+ opponent stats
    defense_frequencies: Optional[Dict[str, float]]    # Optimal defense rates
    coordination_effects: Optional[Dict[str, float]]   # Range coordination
    bluff_catching_frequency: Optional[float]          # Bluff-catch optimization
```

#### Core Properties

- **`win_probability`**: Probability of winning the hand (0.0 to 1.0)
- **`tie_probability`**: Probability of tying with other players
- **`loss_probability`**: Probability of losing the hand
- **`simulations_run`**: Actual number of simulations executed
- **`execution_time_ms`**: Time taken for analysis in milliseconds
- **`confidence_interval`**: 95% confidence interval as (lower, upper) tuple
- **`hand_category_frequencies`**: Dictionary mapping hand types to frequencies

#### Advanced Properties

- **`icm_equity`**: Tournament equity considering ICM calculations
- **`position_aware_equity`**: Position-specific equity adjustments
- **`multi_way_statistics`**: Comprehensive multi-opponent analysis
- **`defense_frequencies`**: Optimal defense rates for multi-way scenarios
- **`coordination_effects`**: Impact of opponent range coordination

## Card Format

Cards use Unicode emoji suits with standard poker ranks for clear visualization:

- **Suits**: ♠️ (spades), ♥️ (hearts), ♦️ (diamonds), ♣️ (clubs)
- **Ranks**: A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2

**Valid Card Examples:**
- `'A♠️'` - Ace of spades
- `'K♥️'` - King of hearts  
- `'10♦️'` - Ten of diamonds (note: "10", not "T")
- `'2♣️'` - Two of clubs

**Important Notes:**
- Use "10" for tens, not "T"
- All suits must use Unicode emoji characters
- Cards are case-sensitive

## Usage Examples

### Pre-flop Analysis

```python
# Premium hands
result = solve_poker_hand(['A♠️', 'A♥️'], 1)
print(f"Pocket Aces vs 1: {result.win_probability:.1%}")

result = solve_poker_hand(['A♠️', 'A♥️'], 5)
print(f"Pocket Aces vs 5: {result.win_probability:.1%}")

# Marginal hands
result = solve_poker_hand(['2♠️', '7♥️'], 5)
print(f"2-7 offsuit vs 5: {result.win_probability:.1%}")

# Suited connectors
result = solve_poker_hand(['9♠️', '8♠️'], 3)
print(f"9-8 suited vs 3: {result.win_probability:.1%}")

# Pocket pairs
result = solve_poker_hand(['5♠️', '5♥️'], 2)
print(f"Pocket fives vs 2: {result.win_probability:.1%}")
```

### Post-flop Analysis

```python
# Strong made hands
result = solve_poker_hand(
    ['A♠️', 'A♥️'],                    # Pocket aces
    2,                                 # 2 opponents
    ['A♦️', '7♠️', '2♣️']             # Flop (top set)
)
print(f"Top set of aces: {result.win_probability:.1%}")

# Drawing hands
result = solve_poker_hand(
    ['A♠️', 'K♠️'],                    # Suited ace-king
    1,                                 # 1 opponent
    ['Q♠️', 'J♦️', '7♠️']             # Flop (nut flush + straight draws)
)
print(f"Nut flush + straight draw: {result.win_probability:.1%}")

# Bluffs and weak hands
result = solve_poker_hand(
    ['7♣️', '2♥️'],                    # Weak hand
    4,                                 # 4 opponents
    ['A♠️', 'K♦️', 'Q♠️']             # High flop
)
print(f"7-2 on AKQ flop: {result.win_probability:.1%}")

# Turn and river analysis
result = solve_poker_hand(
    ['J♠️', '10♠️'],                   # Jack-ten suited
    2,                                 # 2 opponents
    ['9♠️', '8♦️', '7♠️', 'Q♥️']      # Turn (straight + flush draws)
)
print(f"Open-ended straight flush draw: {result.win_probability:.1%}")
```

### Performance Mode Optimization

```python
# Fast mode for real-time decisions (AI bots)
result = solve_poker_hand(['K♠️', 'K♥️'], 3, simulation_mode="fast")
print(f"Fast mode: {result.simulations_run:,} sims in {result.execution_time_ms:.1f}ms")

# Default mode for balanced analysis
result = solve_poker_hand(['K♠️', 'K♥️'], 3, simulation_mode="default")
print(f"Default mode: {result.simulations_run:,} sims in {result.execution_time_ms:.1f}ms")

# Precision mode for critical tournament decisions
result = solve_poker_hand(['K♠️', 'K♥️'], 3, simulation_mode="precision")
print(f"Precision mode: {result.simulations_run:,} sims in {result.execution_time_ms:.1f}ms")
```

### Advanced Statistical Analysis

```python
# Detailed hand category breakdown
result = solve_poker_hand(['A♠️', 'K♠️'], 2, ['Q♠️', 'J♠️', '10♥️'])

print("Hand category frequencies:")
for category, frequency in result.hand_category_frequencies.items():
    print(f"  {category}: {frequency:.1%}")

# Statistical confidence analysis
lower, upper = result.confidence_interval
margin = (upper - lower) / 2
print(f"Win probability: {result.win_probability:.3f} ± {margin:.3f}")
print(f"95% confidence: ({lower:.3f}, {upper:.3f})")

# Multi-way pot statistics (when 3+ opponents)
if result.multi_way_statistics:
    stats = result.multi_way_statistics
    print(f"Expected finish position: {stats['expected_position_finish']:.1f}")
    print(f"Conditional win probability: {stats['conditional_win_probability']:.1%}")
```

### Error Handling and Validation

```python
from poker_knight import solve_poker_hand

try:
    # This will raise ValueError - duplicate cards
    result = solve_poker_hand(['A♠️', 'A♠️'], 2)
except ValueError as e:
    print(f"Duplicate card error: {e}")

try:
    # This will raise ValueError - invalid card format
    result = solve_poker_hand(['AH', 'KS'], 2)  # Missing emoji suits
except ValueError as e:
    print(f"Invalid format error: {e}")

try:
    # This will raise ValueError - too many opponents
    result = solve_poker_hand(['A♠️', 'K♥️'], 10)
except ValueError as e:
    print(f"Too many opponents error: {e}")

try:
    # This will raise ValueError - invalid board size
    result = solve_poker_hand(['A♠️', 'K♥️'], 2, ['Q♠️', 'J♠️'])  # Only 2 board cards
except ValueError as e:
    print(f"Invalid board error: {e}")
```

### Batch Analysis for Training Data

```python
# Generate training data for poker AI
hands_to_analyze = [
    (['A♠️', 'A♥️'], 2, None),
    (['K♠️', 'K♥️'], 3, ['7♠️', '2♥️', '9♦️']),
    (['Q♠️', 'J♠️'], 4, ['10♠️', '9♥️', '8♦️']),
    # ... more scenarios
]

training_data = []
for hero_hand, opponents, board in hands_to_analyze:
    result = solve_poker_hand(hero_hand, opponents, board, simulation_mode="fast")
    training_data.append({
        'hand': hero_hand,
        'opponents': opponents,
        'board': board,
        'equity': result.win_probability,
        'confidence': result.confidence_interval
    })

print(f"Generated {len(training_data)} training samples")
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