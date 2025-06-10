# ♞ Poker Knight

<div align="center">
  
  ![Poker Knight Logo](docs/assets/poker_knight_logo.png)
  
  **A high-performance Monte Carlo Texas Hold'em poker solver with advanced tournament ICM integration, designed for AI poker players and real-time gameplay decision making.**

  [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Tests](https://img.shields.io/badge/tests-passing-green.svg)](tests/)
  [![Coverage](https://img.shields.io/badge/coverage-93%25-brightgreen.svg)](tests/)
  [![Version: 1.8.0](https://img.shields.io/badge/version-1.8.0-green.svg)](CHANGELOG.md)
  [![Performance](https://img.shields.io/badge/performance-optimized-orange.svg)](.)
  [![GPU](https://img.shields.io/badge/GPU-CUDA%20support-76B900.svg)](docs/cuda/)
</div>

## 🎯 What is Poker Knight?

Poker Knight is a specialized Monte Carlo simulation engine built specifically for Texas Hold'em poker analysis. It provides lightning-fast, statistically accurate probability calculations for any poker situation, making it an essential tool for AI poker bot development, training applications, and advanced game analysis.

**What sets Poker Knight apart:** Unlike basic poker calculators, Poker Knight includes advanced tournament features like **ICM (Independent Chip Model) integration**, position-aware equity calculations, sophisticated multi-way pot analysis, and **GPU acceleration** for massive performance gains - features typically found only in professional poker software.

## ✨ Core Capabilities & Technical Innovations

**High-Performance Monte Carlo Simulation Engine** 🚀
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
- **Unicode card representation** using emoji suits (♠ ♥ ♦ ♣) for clear visualization
- **Optimized hand ranking system** with pre-computed lookup tables for maximum speed
- **Special case handling** for wheel straights (A-2-3-4-5) and ace-high vs ace-low scenarios
- **Hand category frequency analysis** with statistical confidence reporting

**Real-time Decision Support**
- **Sub-100ms analysis** for live gameplay integration
- **Configurable simulation modes**: Fast (10k sims), Default (100k sims), Precision (500k sims)
- **Parallel processing** using ThreadPoolExecutor for CPU-bound optimization
- **Memory-optimized algorithms** with object reuse and minimal allocation patterns

**GPU Acceleration (v1.8.0)** 🚀
- **CUDA Integration**: Production-ready GPU acceleration for Monte Carlo simulations
- **1700x Performance**: Massive speedups for large-scale simulations
- **Automatic Detection**: Intelligently uses GPU when available
- **Seamless Fallback**: Automatically falls back to CPU if GPU unavailable
- **Full Accuracy**: GPU implementation matches CPU results exactly

**Zero-Dependency Architecture**
- Pure Python implementation using only standard library (GPU acceleration requires CuPy)
- **Thread-safe design** for concurrent usage in multi-threaded applications
- **Clean single-function API**: `solve_poker_hand(hole_cards, opponents, board_cards)`
- Seamless integration into existing poker AI systems and applications

## 🔗 Resources & Documentation

| Category | Resource | Description |
|----------|----------|-------------|
| **Getting Started** | [Quick Installation Guide](#installation) | Copy-paste installation and basic usage |
| | [API Reference](API_REFERENCE.md) | Complete function documentation with examples |
| | [Quick Test](#quick-test) | Verify your installation works correctly |
| **Advanced Features** | [ICM & Tournament Play](#use-cases) | ICM calculations and tournament-specific analysis |
| | [Multi-Way Pot Analysis](#use-cases) | Position-aware equity and range coordination |
| **Configuration** | [Configuration Guide](poker_knight/config.json) | Performance tuning and advanced settings |
| **Integration** | [Use Cases](#use-cases) | Common applications and implementation patterns |
| **GPU Support** | [CUDA Documentation](docs/cuda/) | GPU acceleration setup and usage |
| | [GPU Performance Guide](docs/cuda/CUDA_FINAL_SUMMARY.md) | GPU performance benchmarks |
| **Development** | [Testing Guide](#quick-test) | Running tests and statistical validation |
| | [Changelog](CHANGELOG.md) | Version history and feature updates |
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

# Simplest usage - just your hand (assumes 1 opponent, pre-flop)
result = solve_poker_hand(['A♠', 'A♥'])
print(f"Win probability: {result.win_probability:.1%}")

# Analyze pocket aces pre-flop against 2 opponents
result = solve_poker_hand(['A♠', 'A♥'], 2)
print(f"Win probability: {result.win_probability:.1%}")
```

**Complete API Reference:**
The `solve_poker_hand` function accepts the following arguments:

```python
solve_poker_hand(
    hero_hand,              # Required: List of 2 cards (e.g., ['A♠', 'K♥'])
    num_opponents=1,        # Optional: Number of opponents (1-6, default: 1)
    board_cards=None,       # Optional: List of 3-5 board cards (default: None for pre-flop)
    simulation_mode="default",  # Optional: "fast", "default", or "precision"
    hero_position=None,     # Optional: "early", "middle", "late", "button", "sb", "bb"
    stack_sizes=None,       # Optional: List [hero_stack, opp1, opp2, ...] for ICM
    pot_size=None,          # Optional: Current pot size for SPR calculations
    tournament_context=None # Optional: Dict with tournament settings (e.g., {'bubble_factor': 1.3})
)
```

**Note:** You can provide as few arguments as just your hand - `solve_poker_hand(['A♠', 'A♥'])` - which will analyze your hand pre-flop against one opponent using default settings.

**Advanced tournament analysis with ICM:**
```python
result = solve_poker_hand(
    ['K♠', 'Q♠'],                      # Hero hand
    3,                                 # Number of opponents  
    ['A♠', 'J♠', '10♥'],              # Board cards (flop)
    hero_position="button",            # Position-aware equity
    stack_sizes=[15000, 8000, 12000, 6000],  # Stack sizes for ICM
    pot_size=2000,                     # Current pot for SPR calculation
    tournament_context={'bubble_factor': 1.3}  # ICM bubble pressure
)
print(f"Win probability: {result.win_probability:.1%}")
print(f"ICM equity: {result.icm_equity:.1%}")
print(f"Position advantage: {result.position_aware_equity['position_advantage']:.3f}")
```

**Advanced Usage with MonteCarloSolver Class:**
For access to additional optimization features, use the solver class directly:

```python
from poker_knight import MonteCarloSolver

# Create solver instance
solver = MonteCarloSolver()

result = solver.analyze_hand(
    ['A♠', 'A♥'],                      # Hero hand
    2,                                 # Number of opponents
    ['K♠️', 'Q♠️', 'J♠️'],             # Board cards
    simulation_mode="precision",        # Simulation mode
    intelligent_optimization=True,      # Enable auto-optimization
    stack_depth=45.0                   # Stack depth in big blinds for analysis
)

# Access optimization data
if result.optimization_data:
    print(f"Complexity level: {result.optimization_data['complexity_level']}")
    print(f"Optimized simulations: {result.optimization_data['recommended_simulations']}")
    
solver.close()  # Clean up resources
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

## 🚀 GPU Acceleration {#gpu-acceleration}

Poker Knight includes production-ready GPU acceleration for massive performance gains:

**Setup:**
```bash
# Install GPU support (requires NVIDIA GPU)
pip install cupy-cuda11x  # or cupy-cuda12x for CUDA 12

# Enable GPU in config.json
{
  "enable_cuda": true
}
```

**Performance Gains:**
- Small simulations (10K): Use CPU (GPU overhead not worth it)
- Medium simulations (100K): 5-10x faster on GPU
- Large simulations (1M+): 100-1700x faster on GPU

**Automatic GPU Usage:**
```python
# GPU is used automatically when available and enabled
result = solve_poker_hand(['A♠', 'K♠'], 3, simulation_mode="precision")
print(f"GPU used: {result.gpu_used}")  # True if GPU was used
print(f"Backend: {result.backend}")     # 'cuda' or 'cpu'
```

## 📄 Requirements {#requirements}

- Python 3.8 or higher
- No external dependencies for CPU mode (uses only Python standard library)
- For GPU acceleration: NVIDIA GPU + CuPy (`pip install cupy-cuda11x`)
- Compatible with Windows, macOS, and Linux
- Multi-core CPU recommended for parallel processing optimization

---

**Poker Knight v1.8.0** - Empowering AI poker players with precise, fast hand analysis, professional tournament features, and GPU acceleration.

*Built with ♠♥♦♣ by [hildolfr](https://github.com/hildolfr)* 