# Technical Details

Implementation details and architecture of Poker Knight's Monte Carlo poker solver.

## Architecture Overview

Poker Knight is built around a core Monte Carlo simulation engine with optimized hand evaluation and statistical analysis capabilities.

### Core Components

```
poker_knight/
├── __init__.py          # Public API exports
├── solver.py            # Main MonteCarloSolver class
└── config.json          # Default configuration
```

### Class Hierarchy

```python
MonteCarloSolver
├── _load_config()       # Configuration management
├── analyze_hand()       # Main analysis entry point
├── _run_simulation()    # Core simulation loop
├── _evaluate_hand()     # Hand strength evaluation
└── _calculate_stats()   # Statistical analysis
```

## Hand Evaluation Engine

### Card Representation

Cards are represented as strings with Unicode emoji suits:
- **Format**: `"<rank><suit>"`
- **Ranks**: A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3, 2
- **Suits**: ♠️ (spades), ♥️ (hearts), ♦️ (diamonds), ♣️ (clubs)

### Hand Evaluation Algorithm

The hand evaluation engine supports all standard poker hands:

```python
def _evaluate_five_cards(self, cards):
    """Evaluate a 5-card poker hand"""
    ranks = [self._card_rank(card) for card in cards]
    suits = [self._card_suit(card) for card in cards]
    
    # Check for flush
    is_flush = len(set(suits)) == 1
    
    # Check for straight
    is_straight = self._is_straight(ranks)
    
    # Count rank frequencies
    rank_counts = Counter(ranks)
    counts = sorted(rank_counts.values(), reverse=True)
    
    # Determine hand type
    if is_straight and is_flush:
        return self._evaluate_straight_flush(ranks)
    elif counts == [4, 1]:
        return self._evaluate_four_of_a_kind(rank_counts)
    # ... additional hand types
```

### Hand Ranking System

Hands are ranked using a numerical system for fast comparison:

| Hand Type | Base Score | Tiebreaker Resolution |
|-----------|------------|----------------------|
| Royal Flush | 9000 | None (all equal) |
| Straight Flush | 8000 | High card |
| Four of a Kind | 7000 | Quad rank + kicker |
| Full House | 6000 | Trips rank + pair rank |
| Flush | 5000 | All five cards (high to low) |
| Straight | 4000 | High card |
| Three of a Kind | 3000 | Trips rank + kickers |
| Two Pair | 2000 | High pair + low pair + kicker |
| One Pair | 1000 | Pair rank + kickers |
| High Card | 0 | All five cards (high to low) |

### Special Cases

#### Wheel Straight (A-2-3-4-5)
```python
def _is_wheel_straight(self, ranks):
    """Check for A-2-3-4-5 straight"""
    wheel_ranks = [14, 2, 3, 4, 5]  # Ace high converted to low
    return sorted(ranks) == sorted(wheel_ranks)
```

#### Ace-High vs Ace-Low
- Aces are high (value 14) except in wheel straights
- Wheel straights rank below all other straights

## Monte Carlo Simulation

### Simulation Algorithm

```python
def _run_simulation(self, hero_hand, num_opponents, board_cards, num_simulations):
    """Core Monte Carlo simulation loop"""
    wins = ties = losses = 0
    hand_categories = Counter()
    
    for _ in range(num_simulations):
        # Create deck with known cards removed
        deck = self._create_deck_with_removed_cards(hero_hand + board_cards)
        
        # Deal opponent hands
        opponent_hands = self._deal_opponent_hands(deck, num_opponents)
        
        # Complete the board if needed
        final_board = self._complete_board(deck, board_cards)
        
        # Evaluate all hands
        hero_strength = self._evaluate_hand(hero_hand + final_board)
        opponent_strengths = [
            self._evaluate_hand(hand + final_board) 
            for hand in opponent_hands
        ]
        
        # Determine winner
        if hero_strength > max(opponent_strengths):
            wins += 1
        elif hero_strength == max(opponent_strengths):
            ties += 1
        else:
            losses += 1
            
        # Track hand categories
        hand_categories[self._get_hand_category(hero_strength)] += 1
    
    return wins, ties, losses, hand_categories
```

### Deck Management

Efficient deck management with card removal:

```python
def _create_deck_with_removed_cards(self, removed_cards):
    """Create deck with specific cards removed"""
    full_deck = [
        f"{rank}{suit}" 
        for rank in ['A', 'K', 'Q', 'J', '10', '9', '8', '7', '6', '5', '4', '3', '2']
        for suit in ['♠️', '♥️', '♦️', '♣️']
    ]
    
    removed_set = set(removed_cards)
    return [card for card in full_deck if card not in removed_set]
```

### Random Sampling

Uses Python's `random.sample()` for unbiased card selection:

```python
def _deal_opponent_hands(self, deck, num_opponents):
    """Deal random hands to opponents"""
    cards_needed = num_opponents * 2
    dealt_cards = random.sample(deck, cards_needed)
    
    opponent_hands = []
    for i in range(num_opponents):
        hand = dealt_cards[i*2:(i+1)*2]
        opponent_hands.append(hand)
    
    return opponent_hands
```

## Performance Optimizations

### Memory Optimization

1. **Object Reuse**: Pre-allocated arrays for hot paths
2. **Minimal Allocation**: Reduced object creation during simulation
3. **Efficient Data Structures**: Counter for rank frequency analysis

```python
class MonteCarloSolver:
    def __init__(self):
        # Pre-allocate arrays for hot path reuse
        self._temp_ranks = [0] * 5
        self._temp_suits = [None] * 5
        self._temp_kickers = [0] * 5
```

### Parallel Processing

ThreadPoolExecutor for CPU-bound simulation work:

```python
def _run_parallel_simulation(self, hero_hand, num_opponents, board_cards, num_simulations):
    """Run simulation using multiple threads"""
    batch_size = num_simulations // self.max_workers
    
    with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
        futures = []
        for _ in range(self.max_workers):
            future = executor.submit(
                self._run_simulation_batch,
                hero_hand, num_opponents, board_cards, batch_size
            )
            futures.append(future)
        
        # Aggregate results
        total_wins = total_ties = total_losses = 0
        combined_categories = Counter()
        
        for future in futures:
            wins, ties, losses, categories = future.result()
            total_wins += wins
            total_ties += ties
            total_losses += losses
            combined_categories.update(categories)
    
    return total_wins, total_ties, total_losses, combined_categories
```

### Early Termination

Time-based and convergence-based early termination:

```python
def _should_terminate_early(self, simulations_run, start_time, current_results):
    """Check if simulation should terminate early"""
    # Time-based termination
    elapsed_ms = (time.time() - start_time) * 1000
    if elapsed_ms > self.max_simulation_time_ms:
        return True
    
    # Convergence-based termination
    if simulations_run >= self.min_simulations_for_convergence:
        if self._has_converged(current_results):
            return True
    
    return False
```

## Statistical Analysis

### Confidence Intervals

95% confidence intervals using normal approximation:

```python
def _calculate_confidence_interval(self, win_probability, num_simulations):
    """Calculate 95% confidence interval for win probability"""
    if num_simulations < 30:
        return (win_probability, win_probability)  # Insufficient data
    
    # Standard error for binomial proportion
    std_error = math.sqrt(win_probability * (1 - win_probability) / num_simulations)
    
    # 95% confidence interval (z = 1.96)
    margin_of_error = 1.96 * std_error
    
    lower_bound = max(0.0, win_probability - margin_of_error)
    upper_bound = min(1.0, win_probability + margin_of_error)
    
    return (lower_bound, upper_bound)
```

### Hand Category Tracking

Frequency analysis of hand types achieved:

```python
def _get_hand_category(self, hand_strength):
    """Determine hand category from strength score"""
    if hand_strength >= 9000:
        return "Royal Flush"
    elif hand_strength >= 8000:
        return "Straight Flush"
    elif hand_strength >= 7000:
        return "Four of a Kind"
    elif hand_strength >= 6000:
        return "Full House"
    elif hand_strength >= 5000:
        return "Flush"
    elif hand_strength >= 4000:
        return "Straight"
    elif hand_strength >= 3000:
        return "Three of a Kind"
    elif hand_strength >= 2000:
        return "Two Pair"
    elif hand_strength >= 1000:
        return "One Pair"
    else:
        return "High Card"
```

## Configuration System

### Configuration Loading

Hierarchical configuration with defaults:

```python
def _load_config(self, config_path=None):
    """Load configuration with fallback to defaults"""
    default_config = {
        "simulation_settings": {
            "default_simulations": 100000,
            "fast_mode_simulations": 10000,
            "precision_mode_simulations": 500000,
            "parallel_processing": True,
            "random_seed": None
        },
        # ... additional sections
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            # Merge user config with defaults
            return self._merge_configs(default_config, user_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
    
    return default_config
```

### Simulation Mode Selection

Dynamic configuration based on simulation mode:

```python
def _get_simulation_count(self, simulation_mode):
    """Get simulation count based on mode"""
    mode_mapping = {
        "fast": self.config["simulation_settings"]["fast_mode_simulations"],
        "default": self.config["simulation_settings"]["default_simulations"],
        "precision": self.config["simulation_settings"]["precision_mode_simulations"]
    }
    return mode_mapping.get(simulation_mode, mode_mapping["default"])
```

## Error Handling

### Input Validation

Comprehensive input validation with descriptive errors:

```python
def _validate_input(self, hero_hand, num_opponents, board_cards):
    """Validate all input parameters"""
    # Validate hero hand
    if not isinstance(hero_hand, list) or len(hero_hand) != 2:
        raise ValueError("Hero hand must be a list of exactly 2 cards")
    
    # Validate card format
    for card in hero_hand:
        if not self._is_valid_card(card):
            raise ValueError(f"Invalid card format: {card}")
    
    # Check for duplicates
    all_cards = hero_hand + (board_cards or [])
    if len(all_cards) != len(set(all_cards)):
        raise ValueError("Duplicate cards detected")
    
    # Validate opponent count
    if not 1 <= num_opponents <= 6:
        raise ValueError("Number of opponents must be between 1 and 6")
```

### Graceful Degradation

Fallback mechanisms for edge cases:

```python
def _handle_timeout(self, partial_results, simulations_run):
    """Handle simulation timeout gracefully"""
    if simulations_run < 1000:
        # Insufficient data - return conservative estimate
        return SimulationResult(
            win_probability=0.5,
            tie_probability=0.0,
            loss_probability=0.5,
            simulations_run=simulations_run,
            execution_time_ms=self.max_simulation_time_ms,
            confidence_interval=(0.4, 0.6),
            hand_category_frequencies={}
        )
    
    # Use partial results
    return self._calculate_final_results(partial_results, simulations_run)
```

## Testing Architecture

### Test Categories

1. **Unit Tests**: Individual function testing
2. **Integration Tests**: End-to-end workflow testing
3. **Statistical Tests**: Mathematical accuracy validation
4. **Performance Tests**: Speed and memory benchmarks
5. **Stress Tests**: High-load scenario testing

### Statistical Validation

Chi-square testing for hand frequency validation:

```python
def test_hand_frequency_distribution():
    """Test that hand frequencies match expected poker probabilities"""
    results = []
    for _ in range(1000):
        result = solve_poker_hand(['7♠️', '2♥️'], 1)  # Weak hand
        results.append(result.hand_category_frequencies)
    
    # Aggregate frequencies
    total_frequencies = Counter()
    for result in results:
        total_frequencies.update(result)
    
    # Expected frequencies (approximate)
    expected = {
        "High Card": 0.501,
        "One Pair": 0.422,
        "Two Pair": 0.047,
        # ... etc
    }
    
    # Chi-square test
    chi_square_stat = calculate_chi_square(total_frequencies, expected)
    assert chi_square_stat < critical_value  # Statistical significance
```

## Future Architecture Considerations

### Extensibility Points

1. **Hand Evaluator Interface**: Pluggable evaluation algorithms
2. **Simulation Strategy**: Alternative simulation approaches
3. **Statistical Analysis**: Additional statistical measures
4. **Configuration Sources**: Database or remote configuration

### Performance Scaling

1. **GPU Acceleration**: CUDA-based simulation for massive parallelism
2. **Distributed Computing**: Multi-machine simulation clusters
3. **Caching Layer**: Result caching for repeated scenarios
4. **Approximation Algorithms**: Fast heuristic-based estimates 