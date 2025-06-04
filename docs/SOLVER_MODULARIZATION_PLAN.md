# Solver Modularization Plan

## Overview

The `solver.py` file currently contains 1,926 lines of code with multiple responsibilities. This plan outlines how to refactor it into a clean, modular architecture while maintaining backward compatibility.

## Current State Analysis

### File Statistics
- **Total Lines**: 1,926
- **Classes**: 5 (Card, HandEvaluator, Deck, SimulationResult, MonteCarloSolver)
- **Main Problem**: MonteCarloSolver class alone is 1,513 lines (78% of the file)

### Component Breakdown

| Component | Lines | Responsibilities | Dependencies |
|-----------|-------|------------------|--------------|
| Card | 28 | Card representation | None |
| HandEvaluator | 147 | Hand evaluation logic | Card |
| Deck | 36 | Deck management | Card |
| SimulationResult | 44 | Result data structure | None |
| MonteCarloSolver | 1,513 | Everything else | All above + external |

## Proposed Module Structure

```
poker_knight/
├── __init__.py          # Public API exports
├── solver.py            # Slim MonteCarloSolver (300-400 lines)
├── api.py              # solve_poker_hand() convenience function
├── constants.py        # SUITS, RANKS, and other constants
│
├── core/
│   ├── __init__.py
│   ├── cards.py        # Card, Deck classes
│   ├── evaluation.py   # HandEvaluator class
│   └── results.py      # SimulationResult dataclass
│
├── simulation/
│   ├── __init__.py
│   ├── runner.py       # Simulation execution logic
│   ├── strategies.py   # Smart sampling, variance reduction
│   └── multiway.py     # Multi-way pot analysis, ICM
│
├── analysis/           # (existing, to be consolidated)
│   ├── __init__.py
│   ├── convergence.py  # Convergence monitoring
│   ├── statistics.py   # Statistical analysis
│   └── performance.py  # Performance metrics
│
└── storage/           # (existing, already well-organized)
    ├── __init__.py
    ├── cache.py
    ├── cache_prepopulation.py
    └── cache_warming.py
```

## Detailed Module Responsibilities

### 1. **poker_knight/constants.py**
```python
# Module-level constants
SUITS = ['♠', '♥', '♦', '♣']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
SUIT_MAPPING = {
    'spades': '♠', 's': '♠', '♠': '♠',
    'hearts': '♥', 'h': '♥', '♥': '♥',
    'diamonds': '♦', 'd': '♦', '♦': '♦',
    'clubs': '♣', 'c': '♣', '♣': '♣'
}
# Hand ranking constants
STRAIGHT_FLUSH = 8
FOUR_OF_A_KIND = 7
# ... etc
```

### 2. **poker_knight/core/cards.py**
```python
from ..constants import SUITS, RANKS, SUIT_MAPPING

class Card:
    """Represents a playing card with suit and rank."""
    # Lines 118-146 from current solver.py

class Deck:
    """Efficient deck of cards with optimized dealing."""
    # Lines 297-333 from current solver.py
```

### 3. **poker_knight/core/evaluation.py**
```python
from .cards import Card
from ..constants import STRAIGHT_FLUSH, FOUR_OF_A_KIND  # etc

class HandEvaluator:
    """Fast Texas Hold'em hand evaluation."""
    # Lines 148-295 from current solver.py
```

### 4. **poker_knight/core/results.py**
```python
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    # Lines 337-381 from current solver.py
```

### 5. **poker_knight/simulation/runner.py**
Extract from MonteCarloSolver:
- `_run_sequential_simulations()` 
- `_run_parallel_simulations()`
- `_simulate_hand()`
- `_batch_simulate_hands()`
- Thread pool management logic

### 6. **poker_knight/simulation/strategies.py**
Extract from MonteCarloSolver:
- `_initialize_smart_sampling()`
- `_compute_stratification_levels()`
- `_adaptive_sampling()`
- `_apply_variance_reduction()`
- All smart sampling related methods

### 7. **poker_knight/simulation/multiway.py**
Extract from MonteCarloSolver:
- `_calculate_multi_way_statistics()`
- `_calculate_position_aware_equity()`
- `_calculate_icm_equity()`
- `_calculate_tournament_pressure()`
- All tournament and multi-way analysis

### 8. **poker_knight/solver.py** (Refactored)
The new slim MonteCarloSolver will:
- Orchestrate the simulation process
- Delegate to specialized modules
- Maintain the public `analyze_hand()` API
- Handle configuration and initialization

```python
from .core import Card, Deck, HandEvaluator, SimulationResult
from .simulation import SimulationRunner, SmartSampler, MultiwayAnalyzer
from .analysis import ConvergenceMonitor
from .storage import HandCache

class MonteCarloSolver:
    """Main Monte Carlo poker solver."""
    
    def __init__(self, config_path=None):
        self.config = self._load_config(config_path)
        self.runner = SimulationRunner(self.config)
        self.sampler = SmartSampler(self.config)
        self.multiway = MultiwayAnalyzer()
        self.cache = self._initialize_cache()
    
    def analyze_hand(self, hero_hand, num_opponents, board_cards=None, **kwargs):
        """Main public API - delegates to specialized components."""
        # Orchestration logic only
```

## Implementation Strategy

### Phase 1: Create New Module Structure (Day 1-2)
1. Create directory structure
2. Move constants to `constants.py`
3. Move Card and Deck to `core/cards.py`
4. Move HandEvaluator to `core/evaluation.py`
5. Move SimulationResult to `core/results.py`

### Phase 2: Extract Simulation Logic (Day 3-4)
1. Create `simulation/runner.py` with execution logic
2. Create `simulation/strategies.py` with sampling methods
3. Create `simulation/multiway.py` with tournament analysis
4. Update MonteCarloSolver to use new modules

### Phase 3: Ensure Backward Compatibility (Day 5)
1. Update `__init__.py` to export all public APIs
2. Create compatibility imports in `solver.py`
3. Run full test suite to verify
4. Add deprecation warnings if needed

### Phase 4: Update Documentation (Day 6)
1. Update import examples in documentation
2. Update API reference
3. Create migration guide
4. Update CLAUDE.md

## Backward Compatibility Plan

### 1. **Maintain Current Imports**
```python
# poker_knight/__init__.py
from .solver import MonteCarloSolver, solve_poker_hand
from .core.cards import Card, Deck
from .core.evaluation import HandEvaluator
from .core.results import SimulationResult

# Maintain existing import paths
__all__ = [
    'Card', 'Deck', 'HandEvaluator', 'SimulationResult',
    'MonteCarloSolver', 'solve_poker_hand'
]
```

### 2. **Compatibility Layer in solver.py**
```python
# poker_knight/solver.py
# For backward compatibility, import moved classes
from .core.cards import Card, Deck
from .core.evaluation import HandEvaluator
from .core.results import SimulationResult
from .api import solve_poker_hand

# This allows existing code like:
# from poker_knight.solver import Card
# to continue working
```

### 3. **Gradual Migration Path**
- Version 1.6.0: Introduce new structure with full backward compatibility
- Version 1.7.0: Add deprecation warnings for old import paths
- Version 2.0.0: Remove compatibility layer

## Testing Strategy

### 1. **Before Refactoring**
```bash
# Run full test suite and save results
pytest --json-report --json-report-file=before-refactor.json
```

### 2. **After Each Phase**
```bash
# Verify no tests break
pytest --json-report --json-report-file=after-phase-X.json
# Compare results
```

### 3. **Import Tests**
Create specific tests for backward compatibility:
```python
def test_old_imports_still_work():
    from poker_knight.solver import Card, Deck, HandEvaluator
    from poker_knight import MonteCarloSolver
    # Verify all work as expected
```

## Benefits of Modularization

1. **Maintainability**: Each module has a single, clear responsibility
2. **Testability**: Smaller modules are easier to unit test
3. **Readability**: Developers can find functionality quickly
4. **Extensibility**: New features can be added to appropriate modules
5. **Performance**: Easier to optimize individual components
6. **Collaboration**: Multiple developers can work on different modules

## Risk Mitigation

1. **Extensive Testing**: Run full test suite after each change
2. **Gradual Approach**: One module at a time
3. **Backward Compatibility**: Maintain all existing imports
4. **Documentation**: Clear migration guide for users
5. **Version Control**: Create feature branch for refactoring

## Success Metrics

- [ ] All existing tests pass without modification
- [ ] No breaking changes to public API
- [ ] solver.py reduced from 1,926 to <400 lines
- [ ] Each new module is <300 lines
- [ ] Import time not significantly increased
- [ ] Documentation updated

## Timeline

- **Day 1-2**: Phase 1 - Basic module extraction
- **Day 3-4**: Phase 2 - Simulation logic extraction  
- **Day 5**: Phase 3 - Compatibility testing
- **Day 6**: Phase 4 - Documentation
- **Day 7**: Buffer for issues and final testing

Total estimated time: 1 week