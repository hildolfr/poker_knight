# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poker Knight is a high-performance Monte Carlo Texas Hold'em poker solver. It's a pure Python implementation (no external dependencies for core functionality) designed for AI poker players and real-time gameplay decision making.

**Platform Philosophy**: This project aims to remain platform-agnostic. Avoid platform-specific code, dependencies, or assumptions. The codebase should work equally well on Linux, macOS, and Windows.

THE CACHING SYSTEM SHALL STORE PLAYER CARDS, FLOP, RIVER, TURN. IT SHALL NOT USE BACKGROUND WARMING, BUT THE CACHE SHALL HAVE AN ARGUMENT AVAILABLE THAT WILL PREPOPULATE THE CACHE WILL ALL POSSIBLE DECISIONS FOR FASTER OPERATION.

## Common Development Commands
remember that we are likely working in a venv; when running commands make sure you remember that.

DO NOT RUN TESTS YOURSELF UNLESS EXPLICITLY ASKED, DEFER TO USER IF POSSIBLE -- THEY ARE LENGTHY
### Installation
```bash
# Development installation
pip install -e .

# With all extras (enterprise, performance, dev)
pip install -e ".[all]"
```

### Running Tests
```bash
# Quick validation (recommended for development)
python tests/run_tests.py --quick
# or
pytest -m quick

# Full test suite
pytest

# Specific test categories
pytest -m unit              # Unit tests
pytest -m statistical       # Statistical validation
pytest -m performance       # Performance tests
pytest -m cache            # Cache-related tests

# Single test file
pytest tests/test_poker_solver.py -v

# With coverage
pytest --cov=poker_knight --cov-report=html
```

### Code Quality
```bash
# Format code
black poker_knight tests

# Lint
flake8 poker_knight tests

# Type checking
mypy poker_knight
```

## Architecture Overview

The codebase follows a modular design with the main package in `poker_knight/`:

- **solver.py**: Core Monte Carlo simulation engine and hand evaluation logic
- **analysis.py**: Convergence analysis and statistical monitoring
- **core/parallel.py**: Advanced parallel processing with ThreadPoolExecutor
- **storage/**: Caching system with Redis/SQLite backends
  - cache.py: Main caching implementation
  - cache_prepopulation.py: Intelligent cache pre-population
  - cache_warming.py: Cache warming functionality

Key classes:
- `MonteCarloSolver`: Main solver class with `analyze_hand()` method
- `SimulationResult`: Structured output with probabilities and analysis
- `Card`, `Deck`, `HandEvaluator`: Core poker logic

## Important Configuration

- **config.json**: Runtime settings for simulation modes, caching, performance
- **pytest.ini**: Test markers and configuration
- **VERSION**: Version tracking (check before releases)

## Card Suit Representation

**IMPORTANT**: Always express card suits as Unicode symbols when working with this codebase:
- ♠ (spades)
- ♥ (hearts)
- ♦ (diamonds)
- ♣ (clubs)

Do NOT use letter representations (S, H, D, C) - the script expects Unicode symbols.

## Recent Development Focus

The v1.6.0 development focuses on intelligent cache pre-population for near-instant results. When working on caching:
- Cache population tests are in `tests/test_cache_population.py`
- Redis integration tests require Redis running locally
- SQLite is the default fallback when Redis is unavailable

## Testing Guidelines

- Always run quick tests before committing: `pytest -m quick`
- Performance-sensitive changes should include benchmark comparisons
- Statistical tests may fail occasionally due to randomness - run them multiple times if needed
- Cache tests can be run with `pytest -m cache`
- Don't bother running test suites, tell the user the command and defer to them. Ensure all test suites we write output results to the relevant folder with date and test-type within the filename.

## Quick API Usage

The main API entry point is `solve_poker_hand()`:

```python
from poker_knight import solve_poker_hand

# Basic usage - analyze pre-flop hand
result = solve_poker_hand(['A♠', 'K♠'], 2)  # AK suited vs 2 opponents
print(f"Win: {result.win_probability:.1%}")

# With board cards (flop/turn/river)
result = solve_poker_hand(
    ['K♠', 'Q♠'],           # Hero hand
    3,                       # Number of opponents  
    ['A♠', 'J♠', '10♥']     # Board cards
)

# Advanced usage with tournament context
result = solve_poker_hand(
    ['A♠', 'K♠'], 
    2,
    ['Q♠', 'J♠', '10♥'],
    simulation_mode="precision",    # More simulations for accuracy
    hero_position="button",         # Position info
    stack_sizes=[5000, 3000, 2000], # ICM calculations
    pot_size=1500                   # For SPR calculations
)
```

Key parameters:
- `hero_hand`: List of 2 cards using Unicode suits (♠♥♦♣)
- `num_opponents`: 1-6 opponents
- `board_cards`: Optional list of 3-5 community cards
- `simulation_mode`: "fast" (10k), "default" (100k), or "precision" (500k) simulations

Returns `SimulationResult` with:
- `win_probability`, `tie_probability`, `loss_probability`
- `confidence_interval`, `simulations_run`
- Advanced metrics for tournament play

## Documentation

Major documentation files are in `docs/`:
- API_REFERENCE.md: Complete API documentation
- TECHNICAL_DETAILS.md: Implementation details
- CONFIGURATION.md: Configuration guide
- TESTING.md: Comprehensive testing guide
