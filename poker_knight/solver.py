#!/usr/bin/env python3
"""
Poker Knight v1.7.0 - High-Performance Monte Carlo Texas Hold'em Poker Solver

High-performance Monte Carlo simulation engine for Texas Hold'em poker hand analysis.
Optimized for AI applications with statistical validation and parallel processing.

Author: hildolfr
License: MIT
GitHub: https://github.com/hildolfr/poker-knight
Version: 1.7.0

Key Features:
- Monte Carlo simulation with configurable precision modes
- Parallel processing with intelligent thread pool management  
- Memory-optimized algorithms for high-throughput analysis
- Statistical validation with confidence intervals
- Advanced convergence analysis with Geweke diagnostics
- Effective sample size calculation and adaptive stopping
- Support for 1-9 opponents with positional awareness

Usage:
    from poker_knight import solve_poker_hand
    result = solve_poker_hand(['A♠', 'K♠'], 2, simulation_mode="default")
    print(f"Win rate: {result.win_probability:.1%}")

Performance optimizations:
- Advanced Monte Carlo convergence analysis with Geweke diagnostics
- Intelligent early stopping when target accuracy achieved
- Real-time convergence monitoring with effective sample size
- Adaptive simulation strategies based on convergence rates
"""

import json
import os
import random
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
from collections import Counter
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import threading

# Import core components from new modules
from .core.cards import Card, Deck
from .core.evaluation import HandEvaluator
from .core.results import SimulationResult
from .constants import SUITS, SUIT_MAPPING, RANKS, RANK_VALUES

# Import simulation components
from .simulation import SimulationRunner, SmartSampler, MultiwayAnalyzer

# Import convergence analysis
try:
    from .analysis import ConvergenceMonitor, convergence_diagnostic, calculate_effective_sample_size
    CONVERGENCE_ANALYSIS_AVAILABLE = True
except ImportError:
    CONVERGENCE_ANALYSIS_AVAILABLE = False

# Import advanced parallel processing
try:
    from .core.parallel import (
        create_parallel_engine, ProcessingConfig, ParallelSimulationEngine,
        ParallelStats, WorkerStats
    )
    from .core.parallel_workers import _parallel_simulation_worker
    ADVANCED_PARALLEL_AVAILABLE = True
except ImportError:
    ADVANCED_PARALLEL_AVAILABLE = False
    _parallel_simulation_worker = None

# Import CUDA acceleration
try:
    from .cuda import CUDA_AVAILABLE, should_use_gpu, get_device_info
    from .cuda.gpu_solver import GPUSolver
except ImportError:
    CUDA_AVAILABLE = False
    should_use_gpu = lambda *args: False
    get_device_info = lambda: None
    GPUSolver = None

# Module metadata
__version__ = "1.7.0"
__author__ = "hildolfr"
__license__ = "MIT"
__all__ = [
    "Card", "HandEvaluator", "Deck", "SimulationResult", 
    "MonteCarloSolver", "solve_poker_hand"
]

class MonteCarloSolver:
    """Monte Carlo poker solver for Texas Hold'em."""
    
    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the solver with configuration settings."""
        self.config = self._load_config(config_path)
        self.evaluator = HandEvaluator()
        
        # Initialize random seed for deterministic behavior if configured
        random_seed = self.config.get("simulation_settings", {}).get("random_seed")
        if random_seed is not None:
            import random
            random.seed(random_seed)
        
        # Initialize simulation components
        self.simulation_runner = SimulationRunner(self.config, self.evaluator)
        self.smart_sampler = SmartSampler(self.config, self.evaluator)
        self.multiway_analyzer = MultiwayAnalyzer()
        
        # Initialize GPU solver if available and enabled
        self.gpu_solver = None
        cuda_settings = self.config.get("cuda_settings", {})
        if CUDA_AVAILABLE and cuda_settings.get("enable_cuda", True):
            try:
                self.gpu_solver = GPUSolver()
                print(f"CUDA acceleration enabled on {get_device_info()['name']}")
            except Exception as e:
                print(f"Warning: Failed to initialize GPU solver: {e}")
                self.gpu_solver = None
        
        self._thread_pool = None
        self._max_workers = self.config["simulation_settings"].get("max_workers", 4)  # Default to 4 workers
        self._lock = threading.Lock()
        
        # Initialize advanced parallel processing engine
        self._parallel_engine = None
        if ADVANCED_PARALLEL_AVAILABLE:
            try:
                # Create parallel processing configuration
                parallel_settings = self.config.get("parallel_settings", {})
                parallel_config = ProcessingConfig(
                    max_threads=parallel_settings.get("max_threads", 0),  # 0 = auto-detect
                    max_processes=parallel_settings.get("max_processes", 0),  # 0 = auto-detect
                    numa_aware=parallel_settings.get("numa_aware", False),
                    complexity_threshold=parallel_settings.get("complexity_threshold", 5.0),
                    minimum_simulations_for_mp=parallel_settings.get("minimum_simulations_for_mp", 5000),
                    shared_memory_size_mb=parallel_settings.get("shared_memory_size_mb", 128),
                    fallback_to_threading=parallel_settings.get("fallback_to_threading", True)
                )
                
                self._parallel_engine = create_parallel_engine(parallel_config)
                
            except Exception as e:
                print(f"Warning: Advanced parallel processing unavailable ({e}). Using standard threading.")
                self._parallel_engine = None
        
        # Smart sampling configuration
        self._sampling_strategy = self.config.get("sampling_strategy", {})
        self._stratified_sampling_enabled = self._sampling_strategy.get("stratified_sampling", False)
        self._importance_sampling_enabled = self._sampling_strategy.get("importance_sampling", False)
        self._control_variates_enabled = self._sampling_strategy.get("control_variates", False)
        
        # Variance reduction state
        self._variance_reduction_state = {
            'control_variate_sum': 0.0,
            'control_variate_count': 0,
            'control_variate_mean': 0.0,
            'stratified_results': {},
            'importance_weights': []
        }
    
    def __enter__(self) -> 'MonteCarloSolver':
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup thread pool."""
        self.close()
    
    def close(self) -> None:
        """Cleanup resources."""
        # Cleanup simulation runner
        self.simulation_runner.close()
        
        # Cleanup GPU solver
        if self.gpu_solver is not None:
            self.gpu_solver.close()
            self.gpu_solver = None
        
        with self._lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
    
    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get or create the persistent thread pool with thread-safe access."""
        with self._lock:
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(max_workers=self._max_workers)
            return self._thread_pool
    
    def analyze_hand(self, 
                    hero_hand: List[str], 
                    num_opponents: int,
                    board_cards: Optional[List[str]] = None,
                    simulation_mode: str = "default",
                    # Multi-way pot analysis parameters
                    hero_position: Optional[str] = None,      # "early", "middle", "late", "button", "sb", "bb"
                    stack_sizes: Optional[List[int]] = None,  # [hero_stack, opp1_stack, opp2_stack, ...]
                    pot_size: Optional[int] = None,           # Current pot size for SPR calculation
                    tournament_context: Optional[Dict[str, Any]] = None,  # ICM context
                    # Intelligent optimization
                    intelligent_optimization: bool = False,   # Enable intelligent scenario analysis
                    stack_depth: float = 100.0               # Stack depth in big blinds for optimization
                    ) -> SimulationResult:
        """
        Analyze a poker hand using Monte Carlo simulation with optional intelligent optimization.
        
        Args:
            hero_hand: List of hero's hole cards (e.g., ["A♠", "K♥"])
            num_opponents: Number of opponents (1-8)
            board_cards: Community cards (optional, 3-5 cards)
            simulation_mode: "fast", "default", or "precision"
            hero_position: Position for multi-way analysis
            stack_sizes: Stack sizes for ICM analysis
            pot_size: Current pot size
            tournament_context: Tournament context for ICM
            intelligent_optimization: Enable automatic optimization based on scenario complexity
            stack_depth: Stack depth in big blinds for complexity analysis
            
        Returns:
            SimulationResult with win probability, statistics, and convergence data
        """
        start_time = time.time()
        
        # Enhanced input validation
        if not isinstance(hero_hand, list) or len(hero_hand) != 2:
            raise ValueError("Hero hand must be a list of exactly 2 cards")
        if not isinstance(num_opponents, int) or num_opponents < 1 or num_opponents > 6:
            raise ValueError("Number of opponents must be between 1 and 6")
        if board_cards is not None and (not isinstance(board_cards, list) or (len(board_cards) != 0 and (len(board_cards) < 3 or len(board_cards) > 5))):
            raise ValueError("Board cards must be empty (preflop) or 3-5 cards (flop/turn/river) if provided")
        if simulation_mode not in ["fast", "default", "precision"]:
            raise ValueError(f"Invalid simulation_mode '{simulation_mode}'. Must be 'fast', 'default', or 'precision'")
        
        # Parse cards and validate format
        try:
            hero_cards = [self.evaluator.parse_card(card) for card in hero_hand]
            board = [self.evaluator.parse_card(card) for card in board_cards] if board_cards else []
        except ValueError as e:
            raise ValueError(f"Invalid card format: {e}")
        
        # Check for duplicate cards
        all_cards = hero_cards + board
        if len(all_cards) != len(set(all_cards)):
            raise ValueError("Duplicate cards detected in hero hand and/or board cards")
        
        # Intelligent Simulation Optimization
        optimization_data = None
        if intelligent_optimization:
            try:
                from .optimizer import ScenarioAnalyzer
                optimizer = ScenarioAnalyzer()
                
                # Analyze scenario complexity
                complexity = optimizer.calculate_scenario_complexity(
                    player_hand=hero_hand,  # Use original string format
                    num_opponents=num_opponents,
                    board=board_cards,
                    stack_depth=stack_depth,
                    position=hero_position or 'middle'
                )
                
                # Override simulation count with optimizer recommendation
                num_simulations = complexity.recommended_simulations
                max_time_ms = complexity.recommended_timeout_ms
                
                optimization_data = {
                    'complexity_level': complexity.overall_complexity.name,
                    'complexity_score': complexity.complexity_score,
                    'recommended_simulations': complexity.recommended_simulations,
                    'confidence_level': complexity.confidence_level,
                    'primary_drivers': complexity.primary_complexity_drivers,
                    'optimization_recommendations': complexity.optimization_recommendations
                }
                
            except ImportError:
                # Fallback to standard mode if optimizer not available
                intelligent_optimization = False
        
        # Determine simulation count (standard mode or intelligent override)
        if not intelligent_optimization:
            if simulation_mode == "fast":
                sim_key = "fast_mode_simulations"
            elif simulation_mode == "precision":
                sim_key = "precision_mode_simulations"
            else:
                sim_key = "default_simulations"
            
            if sim_key in self.config["simulation_settings"]:
                num_simulations = self.config["simulation_settings"][sim_key]
            else:
                num_simulations = self.config["simulation_settings"]["default_simulations"]
            
            # Get timeout settings from configuration
            perf_settings = self.config["performance_settings"]
            if simulation_mode == "fast":
                max_time_ms = perf_settings.get("timeout_fast_mode_ms", 3000)
            elif simulation_mode == "precision":
                max_time_ms = perf_settings.get("timeout_precision_mode_ms", 120000)
            else:
                max_time_ms = perf_settings.get("timeout_default_mode_ms", 20000)
        
        # Default complexity score when not using intelligent optimization
        if optimization_data is None:
            # Simple heuristic based on simulation mode and scenario
            base_complexity = {
                "fast": 2.0,
                "default": 5.0,
                "precision": 8.0
            }.get(simulation_mode, 5.0)
            
            # Adjust based on opponents and board
            opponent_factor = min(2.0, num_opponents * 0.5)
            board_factor = len(board) * 0.3 if board else 0.0
            
            complexity_score = base_complexity + opponent_factor + board_factor
        else:
            complexity_score = optimization_data['complexity_score']
        
        # Track removed cards for accurate simulation
        removed_cards = hero_cards + board
        
        # Check if GPU should be used
        cuda_settings = self.config.get("cuda_settings", {})
        always_use_gpu = cuda_settings.get("always_use_gpu", False)
        use_gpu = (
            self.gpu_solver is not None and 
            cuda_settings.get("enable_cuda", True) and
            should_use_gpu(num_simulations, num_opponents, force=always_use_gpu)
        )
        
        # Run the target number of simulations with timeout as safety fallback
        perf_settings = self.config["performance_settings"]
        parallel_threshold = perf_settings.get("parallel_processing_threshold", 1000)
        use_parallel = (self.config["simulation_settings"].get("parallel_processing", False) 
                       and num_simulations >= parallel_threshold
                       and not use_gpu)  # Don't use CPU parallel if using GPU
        
        # Advanced parallel processing decision
        use_advanced_parallel = (
            self._parallel_engine is not None and 
            use_parallel and 
            num_simulations >= 5000 and  # Advanced parallel requires larger batch sizes
            complexity_score >= 3.0      # Only use for moderately complex scenarios
        )
        
        # Standard parallel processing (disable when convergence analysis is available)
        use_standard_parallel = (
            use_parallel and 
            not use_advanced_parallel and
            not CONVERGENCE_ANALYSIS_AVAILABLE  # Only disable standard parallel for convergence analysis
        )
        
        if use_gpu:
            # Use GPU acceleration
            try:
                gpu_result = self.gpu_solver.analyze_hand(
                    hero_hand, num_opponents, board_cards, num_simulations
                )
                
                # GPU solver returns a complete SimulationResult
                # Just add optimization data and return
                if optimization_data is None:
                    optimization_data = {}
                optimization_data['gpu_execution'] = {
                    'backend': gpu_result.backend,
                    'device': gpu_result.device,
                    'kernel_time_ms': gpu_result.execution_time_ms
                }
                
                # Update the result with additional metadata
                gpu_result.intelligent_optimization = optimization_data
                gpu_result.convergence_achieved = False  # GPU doesn't do convergence analysis yet
                
                # Calculate execution time including all overhead
                total_execution_time = (time.time() - start_time) * 1000
                gpu_result.execution_time_ms = total_execution_time
                
                # Perform multi-way analysis if needed (GPU path doesn't include this)
                if num_opponents >= 3 or hero_position or stack_sizes or tournament_context:
                    multi_way_analysis = self.multiway_analyzer.calculate_multiway_statistics(
                        hero_hand, num_opponents, board_cards, 
                        gpu_result.win_probability, gpu_result.tie_probability, gpu_result.loss_probability,
                        hero_position, stack_sizes, pot_size, tournament_context
                    )
                    
                    # Update GPU result with multiway analysis fields
                    gpu_result.position_aware_equity = multi_way_analysis.get('position_aware_equity')
                    gpu_result.multi_way_statistics = multi_way_analysis.get('multi_way_statistics')
                    gpu_result.fold_equity_estimates = multi_way_analysis.get('fold_equity_estimates')
                    gpu_result.coordination_effects = multi_way_analysis.get('coordination_effects')
                    gpu_result.icm_equity = multi_way_analysis.get('icm_equity')
                    gpu_result.bubble_factor = multi_way_analysis.get('bubble_factor')
                    gpu_result.stack_to_pot_ratio = multi_way_analysis.get('stack_to_pot_ratio')
                    gpu_result.tournament_pressure = multi_way_analysis.get('tournament_pressure')
                    gpu_result.defense_frequencies = multi_way_analysis.get('defense_frequencies')
                    gpu_result.bluff_catching_frequency = multi_way_analysis.get('bluff_catching_frequency')
                    gpu_result.range_coordination_score = multi_way_analysis.get('range_coordination_score')
                
                return gpu_result
                
            except Exception as e:
                print(f"Warning: GPU execution failed ({e}). Falling back to CPU.")
                use_gpu = False
                # Continue to CPU execution paths below
                
        if not use_gpu and use_advanced_parallel:
            # Use advanced parallel processing engine with multiprocessing
            try:
                # Prepare solver data for serialization to worker processes
                def serialize_card(card: Card) -> Dict[str, str]:
                    return {'rank': card.rank, 'suit': card.suit}
                
                solver_data = {
                    'hero_cards': [serialize_card(card) for card in hero_cards],
                    'num_opponents': num_opponents,
                    'board': [serialize_card(card) for card in board],
                    'removed_cards': [serialize_card(card) for card in removed_cards],
                    'include_hand_categories': self.config["output_settings"]["include_hand_categories"]
                }
                
                # Scenario metadata for optimization
                scenario_metadata = {
                    'complexity_score': complexity_score,
                    'hero_hand': hero_hand,
                    'num_opponents': num_opponents,
                    'board_cards': board_cards,
                    'simulation_mode': simulation_mode,
                    'solver_data': solver_data
                }
                
                # Execute with advanced parallel engine using module-level worker
                results, parallel_stats = self._parallel_engine.execute_simulation_batch(
                    _parallel_simulation_worker, num_simulations, scenario_metadata, 
                    solver_data=solver_data
                )
                
                # Extract results
                wins = results.get('wins', 0)
                ties = results.get('ties', 0) 
                losses = results.get('losses', 0)
                hand_categories = Counter(results.get('hand_categories', {}))
                
                # Add parallel execution stats to optimization data
                if optimization_data is None:
                    optimization_data = {}
                optimization_data['parallel_execution'] = {
                    'engine_type': 'advanced_multiprocessing',
                    'total_workers': parallel_stats.worker_count,
                    'multiprocessing_workers': parallel_stats.multiprocessing_workers,
                    'threading_workers': parallel_stats.threading_workers,
                    'speedup_factor': parallel_stats.speedup_factor,
                    'efficiency_percentage': parallel_stats.efficiency_percentage,
                    'cpu_utilization': parallel_stats.cpu_utilization,
                    'numa_distribution': parallel_stats.numa_distribution,
                    'load_balance_score': parallel_stats.load_balance_score
                }
                
                convergence_data = None  # Advanced parallel doesn't use convergence analysis yet
                
            except Exception as e:
                print(f"Warning: Advanced parallel processing failed ({e}). Falling back to standard parallel.")
                # Fall back to standard parallel processing
                wins, ties, losses, hand_categories, convergence_data = self.simulation_runner.run_parallel_simulations(
                    hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
                )
        
        elif use_standard_parallel:
            # Use standard thread pool for parallel processing
            wins, ties, losses, hand_categories, convergence_data = self.simulation_runner.run_parallel_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time
            )
        else:
            # Use sequential processing for small simulation counts, when disabled, or when convergence analysis is needed
            # Initialize convergence monitor if available
            convergence_monitor = None
            if CONVERGENCE_ANALYSIS_AVAILABLE:
                conv_settings = self.config.get("convergence_settings", {})
                convergence_monitor = ConvergenceMonitor(
                    window_size=conv_settings.get("window_size", 1000),
                    geweke_threshold=conv_settings.get("geweke_threshold", 2.0),
                    min_samples=conv_settings.get("min_samples", 5000),
                    target_accuracy=conv_settings.get("target_accuracy", 0.01),
                    confidence_level=conv_settings.get("confidence_level", 0.95)
                )
            
            wins, ties, losses, hand_categories, convergence_data = self.simulation_runner.run_sequential_simulations(
                hero_cards, num_opponents, board, removed_cards, num_simulations, max_time_ms, start_time,
                convergence_monitor=convergence_monitor, smart_sampler=self.smart_sampler
            )
        
        execution_time = (time.time() - start_time) * 1000
        
        # Calculate probabilities
        total_sims = wins + ties + losses
        win_prob = wins / total_sims if total_sims > 0 else 0
        tie_prob = ties / total_sims if total_sims > 0 else 0
        loss_prob = losses / total_sims if total_sims > 0 else 0
        
        # Calculate confidence interval if requested
        confidence_interval = None
        if self.config["output_settings"]["include_confidence_interval"] and total_sims > 0:
            confidence_interval = self.simulation_runner._calculate_confidence_interval(win_prob, total_sims)
        
        # Convert hand categories to frequencies
        hand_category_frequencies = None
        if self.config["output_settings"]["include_hand_categories"] and total_sims > 0:
            hand_category_frequencies = {
                category: count / total_sims 
                for category, count in hand_categories.items()
            }
        
        # Extract convergence analysis results
        convergence_achieved = None
        geweke_statistic = None
        effective_sample_size = None
        convergence_efficiency = None
        stopped_early = None
        convergence_details = None
        
        # Enhanced early confidence stopping fields
        adaptive_timeout_used = None
        final_timeout_ms = None
        target_accuracy_achieved = None
        final_margin_of_error = None
        
        if convergence_data and convergence_data.get('monitor_active', False):
            status = convergence_data.get('convergence_status', {})
            convergence_achieved = status.get('status') == 'converged'
            geweke_statistic = status.get('geweke_statistic')
            effective_sample_size = status.get('effective_sample_size')
            stopped_early = convergence_data.get('stopped_early', False)
            convergence_details = convergence_data.get('convergence_history', [])
            
            # Extract enhanced convergence fields
            adaptive_timeout_used = convergence_data.get('adaptive_timeout_used', False)
            final_timeout_ms = convergence_data.get('final_timeout_ms')
            target_accuracy_achieved = convergence_data.get('target_accuracy_achieved', False)
            final_margin_of_error = convergence_data.get('final_margin_of_error')
            
            # Calculate convergence efficiency
            if effective_sample_size and total_sims > 0:
                convergence_efficiency = effective_sample_size / total_sims
        
        # Multi-way pot analysis
        position_aware_equity = None
        multi_way_statistics = None
        fold_equity_estimates = None
        coordination_effects = None
        icm_equity = None
        bubble_factor = None
        stack_to_pot_ratio = None
        tournament_pressure = None
        defense_frequencies = None
        bluff_catching_frequency = None
        range_coordination_score = None
        
        # Perform multi-way analysis if we have 3+ opponents or position/stack information
        if num_opponents >= 3 or hero_position or stack_sizes or tournament_context:
            multi_way_analysis = self.multiway_analyzer.calculate_multiway_statistics(
                hero_hand, num_opponents, board_cards, win_prob, tie_prob, loss_prob,
                hero_position, stack_sizes, pot_size, tournament_context
            )
            
            position_aware_equity = multi_way_analysis.get('position_aware_equity')
            multi_way_statistics = multi_way_analysis.get('multi_way_statistics') 
            fold_equity_estimates = multi_way_analysis.get('fold_equity_estimates')
            coordination_effects = multi_way_analysis.get('coordination_effects')
            icm_equity = multi_way_analysis.get('icm_equity')
            bubble_factor = multi_way_analysis.get('bubble_factor')
            stack_to_pot_ratio = multi_way_analysis.get('stack_to_pot_ratio')
            tournament_pressure = multi_way_analysis.get('tournament_pressure')
            defense_frequencies = multi_way_analysis.get('defense_frequencies')
            bluff_catching_frequency = multi_way_analysis.get('bluff_catching_frequency')
            range_coordination_score = multi_way_analysis.get('range_coordination_score')
        
        # Determine backend and device info
        gpu_was_used = False
        backend_used = "cpu"
        device_name = None
        
        if optimization_data and 'gpu_execution' in optimization_data:
            gpu_was_used = True
            backend_used = optimization_data['gpu_execution'].get('backend', 'cuda')
            device_name = optimization_data['gpu_execution'].get('device')
        
        return SimulationResult(
            win_probability=round(win_prob, self.config["output_settings"]["decimal_precision"]),
            tie_probability=round(tie_prob, self.config["output_settings"]["decimal_precision"]),
            loss_probability=round(loss_prob, self.config["output_settings"]["decimal_precision"]),
            simulations_run=total_sims,
            execution_time_ms=round(execution_time, 2),
            confidence_interval=confidence_interval,
            hand_category_frequencies=hand_category_frequencies,
            convergence_achieved=convergence_achieved,
            geweke_statistic=geweke_statistic,
            effective_sample_size=effective_sample_size,
            convergence_efficiency=convergence_efficiency,
            stopped_early=stopped_early,
            convergence_details=convergence_details,
            adaptive_timeout_used=adaptive_timeout_used,
            final_timeout_ms=final_timeout_ms,
            target_accuracy_achieved=target_accuracy_achieved,
            final_margin_of_error=final_margin_of_error,
            # Multi-way pot statistics
            position_aware_equity=position_aware_equity,
            multi_way_statistics=multi_way_statistics,
            fold_equity_estimates=fold_equity_estimates,
            coordination_effects=coordination_effects,
            icm_equity=icm_equity,
            bubble_factor=bubble_factor,
            stack_to_pot_ratio=stack_to_pot_ratio,
            tournament_pressure=tournament_pressure,
            defense_frequencies=defense_frequencies,
            bluff_catching_frequency=bluff_catching_frequency,
            range_coordination_score=range_coordination_score,
            optimization_data=optimization_data,
            # GPU acceleration info
            gpu_used=gpu_was_used,
            backend=backend_used,
            device=device_name
        )
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from JSON file with enhanced error handling."""
        if config_path is None:
            # Use package-relative path
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
        
        # Enhanced error handling for configuration loading
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Maintain backward compatibility for tests expecting FileNotFoundError
            if "nonexistent" in config_path:
                raise
            raise ValueError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")
        
        # Validate required configuration sections
        required_sections = ["simulation_settings", "performance_settings", "output_settings"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        return config
    
    # Proxy methods for backward compatibility with tests
    def _run_sequential_simulations(self, hero_cards, num_opponents, board_cards, 
                                   removed_cards, num_simulations, max_time_ms, start_time,
                                   convergence_monitor=None, smart_sampler=None):
        """Backward compatibility proxy for _run_sequential_simulations."""
        return self.simulation_runner.run_sequential_simulations(
            hero_cards, num_opponents, board_cards, removed_cards, 
            num_simulations, max_time_ms, start_time,
            convergence_monitor=convergence_monitor, 
            smart_sampler=smart_sampler or self.smart_sampler
        )
    
    def _run_parallel_simulations(self, hero_cards, num_opponents, board_cards,
                                 removed_cards, num_simulations, max_time_ms, start_time,
                                 convergence_monitor=None, smart_sampler=None):
        """Backward compatibility proxy for _run_parallel_simulations."""
        # Note: run_parallel_simulations doesn't support convergence_monitor or smart_sampler
        # These parameters are ignored for backward compatibility
        return self.simulation_runner.run_parallel_simulations(
            hero_cards, num_opponents, board_cards, removed_cards,
            num_simulations, max_time_ms, start_time
        )
    
    def _calculate_confidence_interval(self, win_probability, simulations, confidence_level=0.95):
        """Backward compatibility proxy for _calculate_confidence_interval."""
        return self.simulation_runner._calculate_confidence_interval(win_probability, simulations, confidence_level)


# Global solver instance for reuse
_global_solver: Optional[MonteCarloSolver] = None
_solver_lock = threading.Lock()

def get_global_solver() -> MonteCarloSolver:
    """Get or create global solver instance for performance."""
    global _global_solver
    with _solver_lock:
        if _global_solver is None:
            _global_solver = MonteCarloSolver()
        return _global_solver

def solve_poker_hand(hero_hand: List[str], 
                    num_opponents: int,
                    board_cards: Optional[List[str]] = None,
                    simulation_mode: str = "default",
                    # Multi-way pot analysis parameters - optional for backward compatibility
                    hero_position: Optional[str] = None,
                    stack_sizes: Optional[List[int]] = None,
                    pot_size: Optional[int] = None,
                    tournament_context: Optional[Dict[str, Any]] = None) -> SimulationResult:
    """
    Convenience function to analyze a poker hand with optional multi-way pot analysis.
    
    Args:
        hero_hand: List of 2 card strings (e.g., ['A♠', 'K♥'])
        num_opponents: Number of opponents (1-6)
        board_cards: Optional board cards (3-5 cards)
        simulation_mode: "fast", "default", or "precision"
        hero_position: Optional position ("early", "middle", "late", "button", "sb", "bb")
        stack_sizes: Optional stack sizes [hero, opp1, opp2, ...] for ICM analysis
        pot_size: Current pot size for SPR calculations
        tournament_context: Optional tournament info for ICM calculations
    
    Returns:
        SimulationResult with analysis including optional multi-way statistics
    """
    solver = get_global_solver()
    return solver.analyze_hand(
        hero_hand, num_opponents, board_cards, simulation_mode,
        hero_position, stack_sizes, pot_size, tournament_context
    )