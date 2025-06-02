# Contributing to Poker Knight

Thank you for your interest in contributing to Poker Knight! This guide will help you get started with contributing to this high-performance Monte Carlo poker solver.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git for version control
- Basic understanding of poker rules and probability theory
- Familiarity with Monte Carlo simulation concepts (helpful but not required)

### Development Setup

1. **Fork and Clone the Repository**
   ```bash
   git clone https://github.com/your-username/poker_knight.git
   cd poker_knight
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install development dependencies
   pip install pytest pytest-cov
   ```

3. **Verify Installation**
   ```bash
   # Run quick tests to ensure everything works
   python tests/run_tests.py --quick
   ```

## Project Structure

```
poker_knight/
├── poker_knight/           # Main package
│   ├── __init__.py        # Public API exports
│   ├── solver.py          # Core Monte Carlo solver
│   └── config.json        # Default configuration
├── tests/                 # Test suite
│   ├── run_tests.py       # Master test runner
│   ├── test_*.py          # Individual test modules
│   └── __init__.py
├── docs/                  # Documentation
│   ├── *.md               # Documentation files
│   └── assets/            # Images and other assets
├── examples/              # Usage examples
├── archived_documentation/ # Historical documentation
└── README.md              # Main project documentation
```

## Development Workflow

### 1. Choose an Issue or Feature

- Check the [Issues](https://github.com/hildolfr/poker_knight/issues) page for open issues
- Look for issues labeled `good first issue` for beginner-friendly tasks
- For new features, open an issue first to discuss the proposal

### 2. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/issue-description
```

### 3. Make Your Changes

Follow these guidelines:

- **Code Style**: Follow PEP 8 Python style guidelines
- **Type Hints**: Add type hints to all new functions and methods
- **Documentation**: Update docstrings and documentation as needed
- **Tests**: Add tests for new functionality

### 4. Test Your Changes

```bash
# Run relevant test categories
python tests/run_tests.py --unit          # For core changes
python tests/run_tests.py --statistical   # For algorithm changes
python tests/run_tests.py --performance   # For optimization changes

# Run all tests before submitting
python tests/run_tests.py
```

### 5. Commit and Push

```bash
git add .
git commit -m "feat: add new feature description"
git push origin feature/your-feature-name
```

### 6. Create Pull Request

- Open a pull request against the `main` branch
- Provide a clear description of your changes
- Reference any related issues
- Ensure all tests pass

## Contribution Guidelines

### Code Quality Standards

1. **Type Safety**
   ```python
   def analyze_hand(self, hero_hand: List[str], num_opponents: int) -> SimulationResult:
       """All functions should have proper type hints"""
   ```

2. **Error Handling**
   ```python
   def validate_input(self, cards: List[str]) -> None:
       """Provide clear, descriptive error messages"""
       if not cards:
           raise ValueError("Card list cannot be empty")
   ```

3. **Documentation**
   ```python
   def calculate_equity(self, hand: List[str]) -> float:
       """
       Calculate hand equity using Monte Carlo simulation.
       
       Args:
           hand: List of 2 card strings in format ['A♠️', 'K♥️']
           
       Returns:
           Equity value between 0.0 and 1.0
           
       Raises:
           ValueError: If hand format is invalid
       """
   ```

### Testing Requirements

All contributions must include appropriate tests:

1. **Unit Tests** for new functions
2. **Integration Tests** for new features
3. **Performance Tests** for optimizations
4. **Statistical Tests** for algorithm changes

Example test structure:
```python
import pytest
from poker_knight import solve_poker_hand

class TestNewFeature:
    def test_basic_functionality(self):
        """Test basic feature operation"""
        result = solve_poker_hand(['A♠️', 'A♥️'], 1)
        assert 0.8 <= result.win_probability <= 0.9
    
    def test_edge_cases(self):
        """Test edge cases and error conditions"""
        with pytest.raises(ValueError):
            solve_poker_hand(['invalid'], 1)
    
    @pytest.mark.performance
    def test_performance(self):
        """Test performance requirements"""
        import time
        start = time.time()
        solve_poker_hand(['K♠️', 'K♥️'], 2, simulation_mode="fast")
        assert time.time() - start < 0.2  # Should complete quickly
```

### Performance Considerations

When contributing performance improvements:

1. **Benchmark Before and After**
   ```bash
   python tests/run_tests.py --performance
   ```

2. **Profile Your Changes**
   ```python
   import cProfile
   cProfile.run('solve_poker_hand(["A♠️", "A♥️"], 2)')
   ```

3. **Memory Usage**
   - Avoid unnecessary object creation in hot paths
   - Reuse objects where possible
   - Consider memory implications of data structures

### Statistical Accuracy

For changes affecting simulation accuracy:

1. **Validate Against Known Probabilities**
   ```python
   # Pocket aces should win ~85% heads-up
   result = solve_poker_hand(['A♠️', 'A♥️'], 1)
   assert 0.84 <= result.win_probability <= 0.86
   ```

2. **Run Statistical Tests**
   ```bash
   python tests/run_tests.py --statistical
   ```

3. **Document Mathematical Basis**
   - Explain the theory behind algorithm changes
   - Provide references to poker literature if applicable

## Areas for Contribution

### High Priority

1. **Performance Optimizations**
   - Hand evaluation speed improvements
   - Memory usage reduction
   - Parallel processing enhancements

2. **Statistical Enhancements**
   - Advanced convergence detection
   - Confidence interval improvements
   - Additional statistical measures

3. **Feature Additions**
   - Opponent range modeling
   - Position-aware analysis
   - Tournament ICM calculations

### Medium Priority

1. **Code Quality**
   - Additional type hints
   - Improved error messages
   - Code documentation

2. **Testing**
   - Edge case coverage
   - Performance regression tests
   - Statistical validation

3. **Documentation**
   - API examples
   - Integration guides
   - Performance tuning tips

### Low Priority

1. **Tooling**
   - Development scripts
   - Continuous integration
   - Code formatting tools

2. **Examples**
   - Real-world use cases
   - Integration examples
   - Tutorial content

## Reporting Issues

When reporting bugs or requesting features:

1. **Use the Issue Templates**
   - Bug reports should include reproduction steps
   - Feature requests should include use cases

2. **Provide Context**
   - Python version
   - Operating system
   - Poker Knight version
   - Relevant configuration

3. **Include Examples**
   ```python
   # Minimal example that demonstrates the issue
   from poker_knight import solve_poker_hand
   result = solve_poker_hand(['A♠️', 'A♥️'], 2)
   # Expected: ~0.85, Actual: 0.42
   ```

## Code Review Process

1. **Automated Checks**
   - All tests must pass
   - Code style checks
   - Performance regression tests

2. **Manual Review**
   - Code quality and readability
   - Test coverage and quality
   - Documentation completeness

3. **Statistical Validation**
   - For algorithm changes, statistical tests must pass
   - Performance benchmarks must meet requirements

## Release Process

Poker Knight follows semantic versioning:

- **Major** (1.0.0): Breaking API changes
- **Minor** (1.1.0): New features, backward compatible
- **Patch** (1.0.1): Bug fixes, backward compatible

## Getting Help

- **Documentation**: Check the [docs/](docs/) directory
- **Issues**: Search existing issues before creating new ones
- **Discussions**: Use GitHub Discussions for questions
- **Email**: Contact the maintainer for sensitive issues

## Recognition

Contributors are recognized in:
- CHANGELOG.md for their contributions
- GitHub contributors list
- Release notes for significant contributions

Thank you for contributing to Poker Knight! Your efforts help make this the best Monte Carlo poker solver available. 