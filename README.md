# ♞ Poker Knight

<div align="center">
  <img src="docs/assets/poker_knight_logo.png" alt="Poker Knight Logo" width="400">
  
  **A high-performance Monte Carlo Texas Hold'em poker solver designed for AI poker players and real-time gameplay decision making.**

  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
  [![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](tests/)
  [![Version: 1.5.0](https://img.shields.io/badge/version-1.5.0-green.svg)](docs/CHANGELOG.md)
  [![Performance](https://img.shields.io/badge/performance-optimized-orange.svg)](docs/PERFORMANCE.md)
</div>

## ✨ Key Features

- **⚡ Fast Monte Carlo Simulations**: Optimized for speed while maintaining accuracy
- **🃏 Unicode Card Representation**: Uses emoji suits (♠️ ♥️ ♦️ ♣️) for clear visualization
- **🎯 Comprehensive Analysis**: Pre-flop, flop, turn, and river support
- **📊 Statistical Confidence**: Confidence intervals and detailed hand category analysis
- **🔌 Clean API**: Easy integration into larger poker AI systems

## 🚀 Quick Start

```python
from poker_knight import solve_poker_hand

# Analyze pocket aces pre-flop against 2 opponents
result = solve_poker_hand(['A♠️', 'A♥️'], 2)
print(f"Win probability: {result.win_probability:.1%}")

# Analyze with board cards (flop scenario)
result = solve_poker_hand(
    ['K♠️', 'Q♠️'],           # Hero hand
    3,                        # Number of opponents  
    ['A♠️', 'J♠️', '10♥️']    # Board cards (flop)
)
print(f"Win probability: {result.win_probability:.1%}")
```

## 📋 Requirements

- Python 3.8+
- No external dependencies (uses only standard library)

## 🔧 Installation

Install the Poker Knight package by copying the `poker_knight/` directory to your project:

```
your_project/
├── poker_knight/          # Copy this entire directory
│   ├── __init__.py
│   ├── solver.py
│   └── config.json
└── your_code.py
```

### Usage

```python
from poker_knight import solve_poker_hand

# Now you can use Poker Knight
result = solve_poker_hand(['A♠️', 'A♥️'], 2)
```

## 📚 Documentation

| Document | Description |
|----------|-------------|
| [API Reference](docs/API_REFERENCE.md) | Complete API documentation and examples |
| [Configuration Guide](docs/CONFIGURATION.md) | Configuration options and performance tuning |
| [Integration Examples](docs/INTEGRATION_EXAMPLES.md) | How to integrate Poker Knight into larger systems |
| [Testing Guide](docs/TESTING.md) | Running tests and statistical validation |
| [Technical Details](docs/TECHNICAL_DETAILS.md) | Implementation details and architecture |
| [Changelog](docs/CHANGELOG.md) | Version history and changes |

## 🧪 Quick Test

```bash
# Run comprehensive test suite
python tests/run_tests.py

# Run specific test categories
python tests/run_tests.py --quick        # Quick validation
python tests/run_tests.py --statistical  # Statistical validation
```

## 🎯 Use Cases

- **🤖 AI Poker Bots**: Core decision-making component
- **📚 Training Tools**: Hand strength analysis for learning  
- **📊 Game Analysis**: Post-game hand review and analysis
- **🔬 Research**: Poker probability and game theory studies

## 🤝 Contributing

See our [contributing guidelines](docs/CONTRIBUTING.md) for information on how to contribute to Poker Knight.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Poker Knight v1.5.0** - Empowering AI poker players with precise, fast hand analysis.

*Built with ♠️♥️♦️♣️ by [hildolfr](https://github.com/hildolfr)* 