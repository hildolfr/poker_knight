# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Poker Knight is a high-performance Monte Carlo Texas Hold'em poker solver. It's a pure Python implementation (no external dependencies for core functionality) designed for AI poker players and real-time gameplay decision making.

**Platform Philosophy**: This project aims to remain platform-agnostic. Avoid platform-specific code, dependencies, or assumptions. The codebase should work equally well on Linux, macOS, and Windows.

## Common Development Commands

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

## Documentation

Major documentation files are in `docs/`:
- API_REFERENCE.md: Complete API documentation
- TECHNICAL_DETAILS.md: Implementation details
- CONFIGURATION.md: Configuration guide
- TESTING.md: Comprehensive testing guide