"""
Result data structures for Poker Knight.

This module contains the SimulationResult dataclass and related types.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation with convergence analysis and multi-way statistics."""
    win_probability: float
    tie_probability: float
    loss_probability: float
    simulations_run: int
    execution_time_ms: float
    confidence_interval: Optional[Tuple[float, float]] = None
    hand_category_frequencies: Optional[Dict[str, float]] = None
    
    # Convergence analysis fields (v1.5.0)
    convergence_achieved: Optional[bool] = None
    geweke_statistic: Optional[float] = None
    effective_sample_size: Optional[float] = None
    convergence_efficiency: Optional[float] = None
    stopped_early: Optional[bool] = None
    convergence_details: Optional[Dict[str, Any]] = None
    
    # Enhanced early confidence stopping fields (Task 3.2)
    adaptive_timeout_used: Optional[bool] = None
    final_timeout_ms: Optional[float] = None
    target_accuracy_achieved: Optional[bool] = None
    final_margin_of_error: Optional[float] = None
    
    # Multi-way pot statistics (Task 7.2)
    position_aware_equity: Optional[Dict[str, float]] = None  # Early/Middle/Late position equity
    multi_way_statistics: Optional[Dict[str, Any]] = None     # 3+ opponent advanced stats
    fold_equity_estimates: Optional[Dict[str, float]] = None  # Position-based fold equity
    coordination_effects: Optional[Dict[str, float]] = None   # Multi-opponent coordination impact
    
    # ICM integration (Task 7.2.b)
    icm_equity: Optional[float] = None                        # Tournament chip equity
    bubble_factor: Optional[float] = None                     # Bubble pressure adjustment
    stack_to_pot_ratio: Optional[float] = None                # SPR for decision making
    tournament_pressure: Optional[Dict[str, float]] = None    # Stack pressure metrics
    
    # Multi-way range analysis (Task 7.2.c) 
    defense_frequencies: Optional[Dict[str, float]] = None    # Multi-way defense requirements
    bluff_catching_frequency: Optional[float] = None         # Optimal bluff catching vs multiple opponents
    range_coordination_score: Optional[float] = None         # How ranges interact in multi-way
    
    # Intelligent optimization data (Task 8.1)
    optimization_data: Optional[Dict[str, Any]] = None        # Scenario complexity analysis and recommendations
    
    # GPU acceleration info (v1.8.0)
    gpu_used: Optional[bool] = None                          # Whether GPU acceleration was used
    backend: Optional[str] = None                            # Backend used: 'cpu', 'cuda', etc.
    device: Optional[str] = None                             # Device name if GPU was used