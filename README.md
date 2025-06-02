# ♞ Poker Knight

<div align="center">
  <img src="docs/assets/poker_knight_logo.png" alt="Poker Knight Logo" width="400">
  
  **A high-performance Monte Carlo Texas Hold'em poker solver with advanced tournament ICM integration, designed for AI poker players and real-time gameplay decision making.**

  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
  [![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](tests/)
  [![Version: 1.5.0](https://img.shields.io/badge/version-1.5.0-green.svg)](docs/CHANGELOG.md)
  [![Performance](https://img.shields.io/badge/performance-optimized-orange.svg)](docs/TECHNICAL_DETAILS.md)
</div>

## 🎯 What is Poker Knight?

Poker Knight is a specialized Monte Carlo simulation engine built specifically for Texas Hold'em poker analysis. It provides lightning-fast, statistically accurate probability calculations for any poker situation, making it an essential tool for AI poker bot development, training applications, and advanced game analysis.

**What sets Poker Knight apart:** Unlike basic poker calculators, Poker Knight includes advanced tournament features like **ICM (Independent Chip Model) integration**, position-aware equity calculations, and sophisticated multi-way pot analysis - features typically found only in professional poker software.

## ✨ Core Capabilities & Technical Innovations

**Advanced Monte Carlo Simulation Engine**
- **Stratified Sampling**: Intelligent variance reduction using board texture analysis and hand strength stratification
- **Importance Sampling**: Weighted simulations focusing on critical decision scenarios
- **Control Variates**: Mathematical variance reduction techniques for faster convergence
- **Adaptive Early Termination**: Intelligent stopping based on statistical confidence intervals
- Supports all game stages: pre-flop, flop, turn, and river analysis with sub-100ms performance

**Tournament-Grade ICM Integration** 🏆
- **Independent Chip Model (ICM)** calculations for tournament equity analysis
- **Bubble factor adjustments** that automatically modify hand values during bubble play
- **Stack pressure dynamics** - short stack vs big stack ICM considerations
- **Multi-table tournament (MTT) optimization** with automated equity calculations
- Position-aware equity that factors ICM pressure into decision making

**Professional Multi-Way Pot Analysis**
- **Range coordination modeling** - how opponent ranges interact against hero in 3+ way pots
- **Position-based equity adjustments** for early/middle/late/button positions
- **Defense frequency calculations** optimized for multi-opponent scenarios  
- **Bluff-catching frequency optimization** against multiple opponents
- **Coordination effects analysis** - statistical modeling of opponent cooperation

**Sophisticated Hand Evaluation**
- **Unicode card representation** using emoji suits (♠️ ♥️ ♦️ ♣️) for clear visualization
- **Optimized hand ranking system** with pre-computed lookup tables for maximum speed
- **Special case handling** for wheel straights (A-2-3-4-5) and ace-high vs ace-low scenarios
- **Hand category frequency analysis** with statistical confidence reporting

**Real-time Decision Support**
- **Sub-100ms analysis** for live gameplay integration
- **Configurable simulation modes**: Fast (10k sims), Default (100k sims), Precision (500k sims)
- **Parallel processing** using ThreadPoolExecutor for CPU-bound optimization
- **Memory-optimized algorithms** with object reuse and minimal allocation patterns

**Zero-Dependency Architecture**
- Pure Python implementation using only standard library
- **Thread-safe design** for concurrent usage in multi-threaded applications
- **Clean single-function API**: `solve_poker_hand(hole_cards, opponents, board_cards)`
- Seamless integration into existing poker AI systems and applications

## 🔗 Resources & Documentation

| Category | Resource | Description |
|----------|----------|-------------|
| **Getting Started** | [Quick Installation Guide](#installation) | Copy-paste installation and basic usage |
| | [API Reference](docs/API_REFERENCE.md) | Complete function documentation with examples |
| | [Quick Test](#quick-test) | Verify your installation works correctly |
| **Advanced Features** | [ICM & Tournament Play](docs/INTEGRATION_EXAMPLES.md#advanced-tournament-bot) | ICM calculations and tournament-specific analysis |
| | [Multi-Way Pot Analysis](docs/INTEGRATION_EXAMPLES.md#hand-history-analyzer) | Position-aware equity and range coordination |
| | [Technical Details](docs/TECHNICAL_DETAILS.md) | Implementation details and advanced algorithms |
| **Configuration** | [Configuration Guide](docs/CONFIGURATION.md) | Performance tuning and advanced settings |
| | [Performance Optimization](docs/TECHNICAL_DETAILS.md#performance-optimizations) | Parallel processing and memory optimization |
| **Integration** | [Integration Examples](docs/INTEGRATION_EXAMPLES.md) | Real-world usage patterns and code samples |
| | [Use Cases](#use-cases) | Common applications and implementation patterns |
| **Development** | [Testing Guide](docs/TESTING.md) | Running tests and statistical validation |
| | [Contributing Guidelines](docs/CONTRIBUTING.md) | How to contribute to the project |
| | [Changelog](docs/CHANGELOG.md) | Version history and feature updates |
| **Support** | [License](LICENSE) | MIT License details |
| | [Requirements](#requirements) | System requirements and dependencies |

## 🚀 Quick Installation {#installation}

**Requirements**: Python 3.8+ (no external dependencies)

Install by copying the `poker_knight/` directory to your project:

```
your_project/
├── poker_knight/          # Copy this entire directory
│   ├── __init__.py
│   ├── solver.py
│   └── config.json
└── your_code.py
```

**Basic Usage:**
```python
from poker_knight import solve_poker_hand

# Analyze pocket aces pre-flop against 2 opponents
result = solve_poker_hand(['A♠️', 'A♥️'], 2)
print(f"Win probability: {result.win_probability:.1%}")

# Advanced tournament analysis with ICM
result = solve_poker_hand(
    ['K♠️', 'Q♠️'],                    # Hero hand
    3,                                 # Number of opponents  
    ['A♠️', 'J♠️', '10♥️'],            # Board cards (flop)
    hero_position="button",            # Position-aware equity
    stack_sizes=[15000, 8000, 12000, 6000],  # Stack sizes for ICM
    pot_size=2000,                     # Current pot for SPR calculation
    tournament_context={'bubble_factor': 1.3}  # ICM bubble pressure
)
print(f"Win probability: {result.win_probability:.1%}")
print(f"ICM equity: {result.icm_equity:.1%}")
print(f"Position advantage: {result.position_aware_equity['position_advantage']:.3f}")
```

## 🧪 Quick Test {#quick-test}

Verify your installation:
```bash
python tests/run_tests.py --quick        # Quick validation
python tests/run_tests.py --statistical  # Full statistical validation
```

## 🎯 Use Cases {#use-cases}

- **🤖 AI Poker Bots**: Core decision-making component with ICM-aware tournament strategy
- **🏆 Tournament Training**: Advanced ICM analysis for bubble play and final table decisions
- **📚 Coaching Software**: Position-aware hand analysis and multi-way pot education
- **📊 Hand History Review**: Post-game analysis with tournament equity calculations
- **🔬 Poker Research**: Academic studies of ICM theory, position dynamics, and game theory optimal play

## 📄 Requirements {#requirements}

- Python 3.8 or higher
- No external dependencies (uses only Python standard library)
- Compatible with Windows, macOS, and Linux
- Multi-core CPU recommended for parallel processing optimization

---

**Poker Knight v1.5.0** - Empowering AI poker players with precise, fast hand analysis and professional tournament features.

*Built with ♠️♥️♦️♣️ by [hildolfr](https://github.com/hildolfr)* 