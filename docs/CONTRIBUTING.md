# Contributing to Poker Knight

Thank you for your interest in contributing to Poker Knight! This comprehensive guide will help you get started with contributing to this high-performance Monte Carlo poker solver with advanced ICM integration.

## Getting Started

### Prerequisites

- **Python 3.8 or higher** with pip package manager
- **Git** for version control and collaboration
- **Basic poker knowledge**: Understanding of Texas Hold'em rules and probability theory
- **Monte Carlo concepts**: Familiarity with simulation-based statistical methods (helpful but not required)
- **Mathematical background**: Basic statistics and probability (beneficial for algorithm contributions)

### Development Environment Setup

1. **Fork and Clone the Repository**
   ```bash
   # Fork the repository on GitHub first, then clone your fork
   git clone https://github.com/your-username/poker_knight.git
   cd poker_knight
   
   # Add upstream remote for keeping your fork synchronized
   git remote add upstream https://github.com/hildolfr/poker_knight.git
   ```

2. **Set Up Development Environment**
   ```bash
   # Create isolated virtual environment (strongly recommended)
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   
   # Install development dependencies
   pip install pytest pytest-cov pytest-xdist pytest-benchmark
   pip install black flake8 mypy  # Code formatting and linting tools
   ```

3. **Verify Installation and Environment**
   ```bash
   # Run quick validation to ensure everything works
   python tests/run_tests.py --quick
   
   # Verify code quality tools
   black --check poker_knight/
   flake8 poker_knight/
   mypy poker_knight/
   ```

## Project Architecture

### Codebase Structure

```
poker_knight/
├── poker_knight/           # Main package directory
│   ├── __init__.py        # Public API exports and version info
│   ├── solver.py          # Core Monte Carlo solver implementation
│   ├── hand_evaluator.py  # Hand evaluation and ranking system  
│   ├── icm_calculator.py  # ICM and tournament equity calculations
│   └── config.json        # Default configuration parameters
├── tests/                 # Comprehensive test suite
│   ├── run_tests.py       # Master test runner with categories
│   ├── test_*.py          # Individual test modules by feature
│   ├── conftest.py        # pytest configuration and fixtures
│   └── fixtures/          # Test data and scenarios
├── docs/                  # Documentation and guides
│   ├── *.md               # Comprehensive documentation files
│   ├── assets/            # Images, diagrams, and visual aids
│   └── examples/          # Usage examples and tutorials
├── examples/              # Practical usage examples
├── archived_documentation/ # Historical development documentation
├── .github/               # GitHub workflows and templates
│   └── workflows/         # CI/CD pipeline definitions
└── README.md              # Main project documentation
```

### Key Components and Responsibilities

**Core Solver (`solver.py`)**
- Monte Carlo simulation engine with variance reduction techniques
- Parallel processing coordination and thread management
- Statistical analysis and convergence detection
- Configuration management and performance optimization

**Hand Evaluator (`hand_evaluator.py`)** 
- Fast hand ranking and comparison algorithms
- Card parsing and validation systems
- Hand category classification and frequency analysis
- Optimized lookup tables for performance-critical paths

**ICM Calculator (`icm_calculator.py`)**
- Independent Chip Model calculations for tournament equity
- Bubble factor analysis and stack pressure modeling
- Position-aware equity adjustments for multi-way scenarios
- Advanced tournament features and payout structure analysis

## Development Workflow

### 1. Choose Your Contribution

**For New Contributors:**
- Check the [Issues](https://github.com/hildolfr/poker_knight/issues) page for open tasks
- Look for issues labeled `good first issue` or `help wanted` for beginner-friendly opportunities
- Review `documentation` labeled issues for non-coding contributions

**For Experienced Contributors:**
- Explore `enhancement` and `feature request` issues for substantial contributions
- Consider performance optimization opportunities in `performance` labeled issues
- Investigate algorithmic improvements and advanced features

**For New Features:**
- Open an issue first to discuss the proposal with maintainers
- Provide detailed use cases and implementation approach
- Consider backward compatibility and performance implications

### 2. Create a Development Branch

```bash
# Ensure your main branch is up to date
git checkout main
git pull upstream main

# Create feature branch with descriptive name
git checkout -b feature/icm-payout-structure-support
# or
git checkout -b fix/convergence-detection-edge-case
# or  
git checkout -b docs/improve-api-documentation
```

### 3. Development Best Practices

Follow these essential guidelines during development:

#### Code Style and Quality

**Python Style Guidelines:**
- Follow **PEP 8** Python style guidelines strictly
- Use **Black** for automated code formatting: `black poker_knight/`
- Ensure **flake8** compliance: `flake8 poker_knight/`
- Maintain **mypy** type checking: `mypy poker_knight/`

**Type Annotations:**
```python
from typing import List, Optional, Dict, Tuple, Any

def analyze_tournament_scenario(
    hero_hand: List[str], 
    opponents: int,
    stack_sizes: List[int],
    payout_structure: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Analyze tournament scenario with ICM considerations.
    
    Args:
        hero_hand: Two-card hand in format ['A♠️', 'K♥️']
        opponents: Number of opponents (1-6)
        stack_sizes: Chip stacks [hero, opp1, opp2, ...]
        payout_structure: Prize distribution [1st, 2nd, 3rd, ...]
        
    Returns:
        Dictionary with ICM analysis results
        
    Raises:
        ValueError: If input parameters are invalid
    """
```

**Documentation Standards:**
```python
class TournamentAnalyzer:
    """
    Advanced tournament analysis with ICM integration.
    
    This class provides comprehensive tournament equity calculations
    using Independent Chip Model (ICM) methodology with bubble
    factor adjustments and position-aware equity modifications.
    
    Attributes:
        config: Configuration parameters for analysis
        icm_cache: Cached ICM calculations for performance
        
    Example:
        >>> analyzer = TournamentAnalyzer()
        >>> result = analyzer.calculate_icm_equity(
        ...     stack_sizes=[15000, 8000, 12000],
        ...     payout_structure=[0.5, 0.3, 0.2]
        ... )
        >>> print(f"ICM equity: {result.icm_equity:.1%}")
    """
```

**Error Handling Philosophy:**
```python
def validate_tournament_input(
    stack_sizes: List[int], 
    payout_structure: List[float]
) -> None:
    """
    Comprehensive input validation with descriptive error messages.
    
    Raises:
        ValueError: With specific, actionable error descriptions
    """
    if not stack_sizes:
        raise ValueError("Stack sizes list cannot be empty")
    
    if any(stack <= 0 for stack in stack_sizes):
        raise ValueError(f"All stack sizes must be positive (got {stack_sizes})")
    
    if len(payout_structure) > len(stack_sizes):
        raise ValueError(
            f"Payout structure length ({len(payout_structure)}) "
            f"cannot exceed number of players ({len(stack_sizes)})"
        )
    
    if not math.isclose(sum(payout_structure), 1.0, abs_tol=1e-6):
        raise ValueError(
            f"Payout structure must sum to 1.0 (got {sum(payout_structure):.6f})"
        )
```

#### Testing Requirements

All contributions must include comprehensive testing:

**1. Unit Tests** for new functions and methods
```python
class TestICMCalculation:
    def test_basic_icm_equity(self):
        """Test basic ICM equity calculation"""
        result = calculate_icm_equity([1000, 1000, 1000], [0.5, 0.3, 0.2])
        expected_equity = 1.0 / 3.0  # Equal stacks = equal equity
        assert math.isclose(result, expected_equity, abs_tol=1e-3)
    
    def test_bubble_factor_adjustment(self):
        """Test bubble pressure increases with factor"""
        base_equity = calculate_icm_equity([1000, 1000], [0.6, 0.4])
        bubble_equity = calculate_icm_equity(
            [1000, 1000], [0.6, 0.4], bubble_factor=1.5
        )
        assert bubble_equity < base_equity  # Bubble should reduce equity
```

**2. Integration Tests** for new features
```python
def test_icm_integration_with_solver():
    """Test ICM integration with main solver"""
    result = solve_poker_hand(
        ['A♠️', 'K♠️'], 
        2,
        stack_sizes=[15000, 8000, 12000],
        tournament_context={'bubble_factor': 1.3}
    )
    
    assert result.icm_equity is not None
    assert result.bubble_factor == 1.3
    assert 0 <= result.icm_equity <= 1
```

**3. Performance Tests** for optimizations
```python
@pytest.mark.performance
def test_icm_calculation_performance():
    """Test ICM calculation meets performance standards"""
    import time
    
    start_time = time.time()
    for _ in range(1000):
        calculate_icm_equity([10000, 8000, 6000, 4000], [0.4, 0.3, 0.2, 0.1])
    execution_time = time.time() - start_time
    
    # Should complete 1000 calculations in under 100ms
    assert execution_time < 0.1
```

**4. Statistical Tests** for algorithm changes
```python
@pytest.mark.statistical  
def test_icm_mathematical_accuracy():
    """Test ICM calculations against known mathematical results"""
    # Known scenario: heads-up with 60/40 chip distribution
    result = calculate_icm_equity([600, 400], [0.6, 0.4])
    
    # Mathematical expectation for this scenario
    expected = 0.6 * 0.6 + 0.4 * (600/1000)  # Simplified ICM formula
    assert math.isclose(result, expected, abs_tol=1e-4)
```

### 4. Commit and Documentation Standards

#### Commit Message Guidelines

Use conventional commit format for clear project history:

```bash
# Feature additions
git commit -m "feat(icm): add payout structure support for MTT analysis"

# Bug fixes  
git commit -m "fix(solver): resolve convergence detection in edge cases"

# Performance improvements
git commit -m "perf(hand-eval): optimize hand ranking with lookup tables"

# Documentation updates
git commit -m "docs(api): add comprehensive ICM examples and use cases"

# Testing improvements
git commit -m "test(statistical): add chi-square validation for hand categories"

# Refactoring
git commit -m "refactor(solver): extract ICM logic into separate module"
```

#### Documentation Updates

Update relevant documentation files when contributing:

```bash
# Always update these files for new features:
docs/API_REFERENCE.md      # Function signatures and examples
docs/CONFIGURATION.md      # New configuration options  
docs/INTEGRATION_EXAMPLES.md  # Practical usage examples

# Update these files when appropriate:
docs/TECHNICAL_DETAILS.md  # Algorithm descriptions and architecture
docs/TESTING.md           # New testing procedures or requirements
README.md                 # Major feature additions or changes
```

### 5. Testing Your Changes

```bash
# Run comprehensive test suite
python tests/run_tests.py

# Run specific test categories relevant to your changes
python tests/run_tests.py --unit          # Basic functionality
python tests/run_tests.py --statistical   # Algorithm accuracy  
python tests/run_tests.py --performance   # Speed and efficiency
python tests/run_tests.py --icm           # ICM and tournament features

# Run tests with coverage analysis
python -m pytest tests/ --cov=poker_knight --cov-report=html --cov-report=term

# Run code quality checks
black --check poker_knight/
flake8 poker_knight/
mypy poker_knight/
```

### 6. Submit Your Contribution

```bash
# Push your changes to your fork
git add .
git commit -m "feat(icm): add bubble factor analysis for tournament play"
git push origin feature/icm-bubble-factor-analysis

# Create Pull Request on GitHub with:
# - Clear title describing the change
# - Detailed description of implementation approach
# - Reference to related issues
# - Screenshots or examples if applicable
```

## Contribution Categories

### Algorithm and Performance Improvements

**Monte Carlo Enhancements:**
- Variance reduction techniques (stratified sampling, control variates)
- Convergence detection and early termination algorithms
- Parallel processing optimizations and thread safety improvements
- Memory usage optimization and object pooling strategies

**ICM and Tournament Features:**
- Advanced ICM calculations (satellite tournaments, progressive knockouts)
- Multi-table tournament (MTT) equity calculations
- Position-aware analysis improvements for multi-way scenarios
- Bubble factor modeling and stack pressure dynamics

**Hand Evaluation Optimizations:**
- Hand ranking algorithm improvements and lookup table optimizations
- Card parsing and validation enhancements
- Hand category classification refinements

### Feature Development

**New Analysis Capabilities:**
- Range vs range analysis for advanced scenarios
- Pot odds and implied odds calculations
- Drawing hand analysis and out counting
- Board texture analysis and categorization

**Enhanced Tournament Support:**
- Progressive knockout (PKO) tournament analysis
- Satellite tournament qualification probabilities
- Multi-table tournament position modeling
- Advanced payout structure support (proportional, winner-take-all)

**Integration and API Improvements:**
- Additional output formats (JSON, CSV, XML)
- REST API for web service integration
- Command-line interface enhancements
- Configuration management improvements

### Documentation and Examples

**Technical Documentation:**
- Algorithm explanations with mathematical foundations
- Performance analysis and optimization guides
- Architecture documentation and design decisions
- Troubleshooting guides and FAQ sections

**Usage Examples:**
- Real-world integration scenarios and case studies
- Tournament analysis workflows and best practices
- Performance optimization tutorials
- Advanced configuration examples

**Educational Content:**
- Poker mathematics and probability explanations
- ICM theory and practical applications
- Monte Carlo simulation methodology
- Statistical analysis and interpretation guides

### Testing and Quality Assurance

**Test Coverage Expansion:**
- Edge case identification and testing
- Performance regression test development
- Statistical validation test enhancement
- Integration test scenario expansion

**Quality Improvements:**
- Code review process enhancement
- Automated testing pipeline improvements
- Performance monitoring and alerting
- Documentation accuracy verification

## Code Review Process

### Pull Request Requirements

**Before Submitting:**
- [ ] All tests pass locally (`python tests/run_tests.py`)
- [ ] Code style compliance (`black`, `flake8`, `mypy`)
- [ ] Documentation updated for new features
- [ ] Performance impact assessed and acceptable
- [ ] Backward compatibility maintained (unless breaking change is justified)

**Pull Request Description Should Include:**
1. **Problem statement**: What issue does this solve?
2. **Solution approach**: How does your implementation address the problem?
3. **Testing strategy**: What tests were added/modified and why?
4. **Performance impact**: Any performance improvements or regressions?
5. **Breaking changes**: Any API changes that affect existing users?

### Review Criteria

**Functionality:**
- ✅ Code solves the stated problem correctly
- ✅ Edge cases are handled appropriately  
- ✅ Error handling is comprehensive and user-friendly
- ✅ Integration with existing features works seamlessly

**Code Quality:**
- ✅ Code follows established patterns and conventions
- ✅ Functions and classes have clear, single responsibilities
- ✅ Variable and function names are descriptive and consistent
- ✅ Comments explain complex algorithms and business logic

**Performance:**
- ✅ No significant performance regressions introduced
- ✅ Memory usage is reasonable and stable
- ✅ Algorithms are efficient for their use case
- ✅ Parallel processing is utilized appropriately

**Testing:**
- ✅ Adequate test coverage for new functionality
- ✅ Tests cover both happy path and error scenarios
- ✅ Performance tests included for optimization changes
- ✅ Statistical tests included for algorithm modifications

### Feedback and Iteration

**Responding to Review Feedback:**
- Address all reviewer comments thoroughly
- Ask questions if feedback is unclear
- Make requested changes in separate commits for easy tracking
- Update documentation if implementation approach changes

**Common Review Feedback:**
- **Performance concerns**: Profile code and provide benchmarks
- **Test coverage gaps**: Add tests for uncovered edge cases
- **Documentation needs**: Expand docstrings and usage examples
- **Code clarity**: Refactor complex functions into smaller, focused units

## Development Resources

### Mathematical References

**Poker Mathematics:**
- Chen, Bill. "The Mathematics of Poker" - Fundamental probability theory
- Sklansky, David. "The Theory of Poker" - Strategic foundations
- Gordon, Phil. "Phil Gordon's Little Green Book" - ICM applications

**Statistical Methods:**
- Ross, Sheldon. "Introduction to Probability Models" - Monte Carlo methods
- Robert, Christian. "Monte Carlo Statistical Methods" - Advanced techniques
- Kroese, Dirk. "Handbook of Monte Carlo Methods" - Implementation strategies

### Technical Tools and Resources

**Development Tools:**
```bash
# Code formatting and quality
black poker_knight/           # Automatic code formatting
flake8 poker_knight/          # Linting and style checking  
mypy poker_knight/            # Static type checking
pytest tests/ -v              # Comprehensive testing

# Performance analysis
python -m cProfile solver.py  # Performance profiling
memory_profiler               # Memory usage analysis
line_profiler                 # Line-by-line performance
```

**Debugging and Analysis:**
```python
# Performance profiling example
import cProfile
import pstats

def profile_solver_performance():
    """Profile solver performance for optimization"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run performance-critical code
    for _ in range(100):
        result = solve_poker_hand(['A♠️', 'K♠️'], 2, simulation_mode="default")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(20)

# Memory usage monitoring
from memory_profiler import profile

@profile
def memory_intensive_operation():
    """Monitor memory usage during operation"""
    results = []
    for _ in range(1000):
        result = solve_poker_hand(['K♠️', 'Q♠️'], 4)
        results.append(result)
    return results
```

### Community and Communication

**Getting Help:**
- Open an issue for questions about implementation approaches
- Join discussions in existing issues for collaborative problem-solving
- Review archived documentation for historical context and decisions

**Staying Updated:**
```bash
# Keep your fork synchronized with upstream changes
git fetch upstream
git checkout main  
git merge upstream/main
git push origin main

# Rebase feature branch on latest main
git checkout feature/your-feature
git rebase main
```

**Code of Conduct:**
- Be respectful and constructive in all interactions
- Provide helpful feedback and suggestions
- Credit others' contributions and ideas appropriately  
- Focus on technical merit and project improvement

## Advanced Contribution Opportunities

### Research and Algorithm Development

**Statistical Analysis Improvements:**
- Implement advanced convergence detection algorithms
- Develop new variance reduction techniques for specific poker scenarios
- Research optimal simulation strategies for different game situations

**ICM Algorithm Enhancements:**
- Extend ICM calculations for complex tournament structures
- Implement bubble factor optimization for different tournament phases
- Develop position-aware ICM adjustments for multi-way scenarios

### Performance Optimization Projects

**Parallel Processing Enhancements:**
- GPU acceleration for Monte Carlo simulations
- Distributed computing support for large-scale analysis
- Memory-mapped file systems for large dataset processing

**Algorithmic Optimizations:**
- Hand evaluation speedup using bit manipulation techniques
- Lookup table generation and optimization strategies
- Cache-friendly data structures for hot path optimization

### Integration and Tooling

**Development Infrastructure:**
- Continuous integration pipeline improvements
- Automated performance regression detection
- Documentation generation and maintenance tools

**External Integration Support:**
- Database integration for large-scale analysis
- Web service API development
- Integration with popular poker software platforms

Thank you for contributing to Poker Knight! Your contributions help make this the most advanced open-source poker analysis tool available, benefiting poker players, researchers, and developers worldwide. 