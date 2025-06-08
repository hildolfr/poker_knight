"""
♞ Poker Knight Startup Cache Pre-Population

One-time cache pre-population system designed for application startup.
No background workers or daemons - just fast, intelligent cache population
when the solver is initialized for maximum performance during use.

Author: hildolfr  
License: MIT
"""

import time
import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

# Import cache components
from .unified_cache import ThreadSafeMonteCarloCache, CacheResult, create_cache_key
from .preflop_cache import PreflopCache, PreflopCacheConfig, PreflopHandGenerator

logger = logging.getLogger(__name__)


@dataclass
class StartupPopulationConfig:
    """Configuration for startup cache pre-population."""
    # Core settings
    enabled: bool = True
    force_repopulation: bool = False  # Force re-population even if cache exists
    skip_if_coverage_above: float = 0.80  # Skip if coverage > 80%
    
    # Population scope
    priority_hands_only: bool = True  # Only populate premium/strong hands
    max_opponents: int = 6
    simulation_modes: List[str] = None  # ["fast", "default"] - skip precision for startup
    hand_categories: List[str] = None  # ["premium", "strong"] by default
    
    # Performance settings
    max_population_time_seconds: int = 30  # Hard limit for startup
    progress_reporting_interval: int = 50  # Report every N scenarios
    batch_delay_ms: int = 10  # Small delay between batches
    
    # Persistence settings
    cache_file_path: str = "preflop_startup_cache.json"
    save_populated_data: bool = False  # Disabled by default to avoid issues
    load_existing_data: bool = False   # Disabled by default to avoid issues
    
    def __post_init__(self):
        if self.simulation_modes is None:
            self.simulation_modes = ["fast", "default"]  # Skip precision for startup
        if self.hand_categories is None:
            self.hand_categories = ["premium", "strong"] if self.priority_hands_only else ["premium", "strong", "medium"]


@dataclass 
class PopulationResult:
    """Result of startup population process."""
    success: bool = False
    scenarios_populated: int = 0
    scenarios_skipped: int = 0  # Already cached
    scenarios_failed: int = 0
    population_time_seconds: float = 0.0
    initial_coverage: float = 0.0
    final_coverage: float = 0.0
    performance_improvement: float = 0.0  # Estimated speedup factor
    
    @property
    def total_scenarios(self) -> int:
        return self.scenarios_populated + self.scenarios_skipped + self.scenarios_failed
    
    @property
    def success_rate(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return (self.scenarios_populated + self.scenarios_skipped) / self.total_scenarios


class StartupCachePopulator:
    """
    Startup cache pre-population system optimized for application initialization.
    
    Designed for one-time execution during solver startup to provide maximum
    performance for subsequent queries. No background processing or daemon behavior.
    """
    
    def __init__(self, 
                 config: Optional[StartupPopulationConfig] = None,
                 unified_cache: Optional[ThreadSafeMonteCarloCache] = None,
                 preflop_cache: Optional[PreflopCache] = None):
        
        self.config = config or StartupPopulationConfig()
        self.unified_cache = unified_cache
        self.preflop_cache = preflop_cache
        
        # Pre-generate hand definitions for efficiency
        self.all_hands = PreflopHandGenerator.generate_all_hands()
        self.priority_hands = [h for h in self.all_hands if h.category in self.config.hand_categories]
        
        logger.info(f"Startup populator initialized: {len(self.priority_hands)} priority hands")
    
    def should_populate(self) -> Tuple[bool, str]:
        """Check if cache population is needed."""
        if not self.config.enabled:
            return False, "Population disabled in config"
        
        if not self.preflop_cache:
            return False, "No preflop cache available"
        
        if self.config.force_repopulation:
            return True, "Force repopulation requested"
        
        # Check current coverage
        stats = self.preflop_cache.get_stats()
        current_coverage = stats.coverage_percentage
        
        if current_coverage >= self.config.skip_if_coverage_above:
            return False, f"Coverage {current_coverage:.1f}% >= threshold {self.config.skip_if_coverage_above:.1f}%"
        
        return True, f"Coverage {current_coverage:.1f}% below threshold"
    
    def populate_startup_cache(self, simulation_callback: Callable) -> PopulationResult:
        """
        Populate cache during startup with essential preflop scenarios.
        
        Args:
            simulation_callback: Function to call for simulation (hand_notation, opponents, mode) -> CacheResult
            
        Returns:
            PopulationResult with statistics and performance data
        """
        result = PopulationResult()
        start_time = time.time()
        
        # Check if population is needed
        should_populate, reason = self.should_populate()
        if not should_populate:
            logger.info(f"Skipping startup population: {reason}")
            return result
        
        # Get initial coverage
        if self.preflop_cache:
            initial_stats = self.preflop_cache.get_stats()
            result.initial_coverage = initial_stats.coverage_percentage
        
        logger.info(f"Starting startup cache population: {reason}")
        
        # Load existing cache data if enabled
        if self.config.load_existing_data:
            self._load_existing_cache_data()
        
        # Generate population tasks
        tasks = self._generate_startup_tasks()
        
        if not tasks:
            logger.info("No tasks generated for population")
            return result
        
        logger.info(f"Populating {len(tasks)} essential scenarios")
        
        # Populate tasks with time limit
        timeout_time = start_time + self.config.max_population_time_seconds
        
        for i, task in enumerate(tasks):
            # Check timeout
            if time.time() > timeout_time:
                logger.warning(f"Population timeout reached after {self.config.max_population_time_seconds}s")
                break
            
            # Check if already cached
            existing = self.preflop_cache.get_preflop_result(
                task["hand_notation"], 
                task["num_opponents"], 
                task["simulation_mode"]
            )
            
            if existing:
                result.scenarios_skipped += 1
                continue
            
            # Execute simulation
            try:
                sim_result = simulation_callback(
                    task["hand_notation"],
                    task["num_opponents"], 
                    task["simulation_mode"]
                )
                
                if sim_result:
                    # Convert SimulationResult to CacheResult
                    cache_result = CacheResult(
                        win_probability=sim_result.win_probability,
                        tie_probability=sim_result.tie_probability,
                        loss_probability=sim_result.loss_probability,
                        simulations_run=sim_result.simulations_run,
                        execution_time_ms=sim_result.execution_time_ms,
                        hand_categories=sim_result.hand_category_frequencies or {},
                        metadata=None,
                        confidence_interval={'low': sim_result.confidence_interval[0], 'high': sim_result.confidence_interval[1]} if sim_result.confidence_interval else None,
                        convergence_achieved=sim_result.convergence_achieved,
                        timestamp=time.time()
                    )
                    
                    # Store in cache
                    success = self.preflop_cache.store_preflop_result(
                        task["hand_notation"],
                        task["num_opponents"],
                        cache_result,
                        task["simulation_mode"]
                    )
                    
                    if success:
                        result.scenarios_populated += 1
                    else:
                        result.scenarios_failed += 1
                else:
                    result.scenarios_failed += 1
                    
            except Exception as e:
                logger.error(f"Failed to populate {task['hand_notation']}: {e}")
                result.scenarios_failed += 1
            
            # Progress reporting
            if (i + 1) % self.config.progress_reporting_interval == 0:
                progress_pct = ((i + 1) / len(tasks)) * 100
                logger.info(f"Population progress: {progress_pct:.1f}% ({i + 1}/{len(tasks)})")
            
            # Small delay to prevent overwhelming system during startup
            if self.config.batch_delay_ms > 0:
                time.sleep(self.config.batch_delay_ms / 1000.0)
        
        # Calculate final results
        result.population_time_seconds = time.time() - start_time
        
        # Get final coverage
        if self.preflop_cache:
            final_stats = self.preflop_cache.get_stats()
            result.final_coverage = final_stats.coverage_percentage
        
        # Estimate performance improvement
        if result.scenarios_populated > 0:
            # Assume 1000x speedup for cache hits vs simulations
            hit_probability = result.final_coverage / 100.0
            result.performance_improvement = 1 + (hit_probability * 999)  # 1x to 1000x range
        
        result.success = (result.success_rate > 0.8)  # 80% success threshold
        
        # Save populated data if enabled
        if self.config.save_populated_data:
            self._save_cache_data()
        
        self._log_population_results(result)
        
        return result
    
    def _generate_startup_tasks(self) -> List[Dict[str, Any]]:
        """Generate essential startup population tasks."""
        tasks = []
        
        # Use priority hands for startup
        hands_to_populate = self.priority_hands
        
        # Priority order: most common scenarios first
        priority_opponents = [2, 3, 1, 4, 5, 6]  # 2-3 opponents most common
        
        for hand_def in hands_to_populate:
            for num_opponents in priority_opponents[:self.config.max_opponents]:
                for simulation_mode in self.config.simulation_modes:
                    # Calculate priority (lower = higher priority)
                    priority = self._calculate_startup_priority(hand_def, num_opponents, simulation_mode)
                    
                    tasks.append({
                        "hand_notation": hand_def.notation,
                        "num_opponents": num_opponents,
                        "simulation_mode": simulation_mode,
                        "priority": priority,
                        "category": hand_def.category
                    })
        
        # Sort by priority for startup
        tasks.sort(key=lambda t: t["priority"])
        
        return tasks
    
    def _calculate_startup_priority(self, hand_def, num_opponents: int, simulation_mode: str) -> int:
        """Calculate priority for startup population (lower = higher priority)."""
        priority = 0
        
        # Hand category priority (premium hands first)
        category_priorities = {"premium": 0, "strong": 10, "medium": 20, "weak": 30, "trash": 40}
        priority += category_priorities.get(hand_def.category, 50)
        
        # Opponent count priority (2-3 opponents most common)
        opponent_priorities = {2: 0, 3: 1, 1: 5, 4: 10, 5: 15, 6: 20}
        priority += opponent_priorities.get(num_opponents, 25)
        
        # Simulation mode priority (fast first for startup)
        mode_priorities = {"fast": 0, "default": 5, "precision": 15}
        priority += mode_priorities.get(simulation_mode, 20)
        
        return priority
    
    def _load_existing_cache_data(self) -> bool:
        """Load existing cache data from file."""
        cache_file = Path(self.config.cache_file_path)
        
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Load preflop results into cache
            preflop_data = data.get('preflop_results', {})
            loaded_count = 0
            
            for key, result_data in preflop_data.items():
                try:
                    # Parse key: "hand_notation:opponents:mode"
                    parts = key.split(':')
                    if len(parts) == 3:
                        hand_notation, num_opponents, simulation_mode = parts
                        
                        # Create CacheResult
                        confidence_interval = result_data.get('confidence_interval')
                        if confidence_interval and isinstance(confidence_interval, list) and len(confidence_interval) >= 2:
                            conf_interval_dict = {'low': confidence_interval[0], 'high': confidence_interval[1]}
                        elif confidence_interval and isinstance(confidence_interval, dict):
                            conf_interval_dict = confidence_interval
                        else:
                            conf_interval_dict = None
                            
                        cache_result = CacheResult(
                            win_probability=result_data['win_probability'],
                            tie_probability=result_data['tie_probability'],
                            loss_probability=result_data['loss_probability'],
                            confidence_interval=conf_interval_dict,
                            simulations_run=result_data['simulations_run'],
                            execution_time_ms=result_data['execution_time_ms'],
                            hand_categories=result_data.get('hand_categories', {}),
                            metadata=result_data.get('metadata'),
                            timestamp=result_data.get('timestamp', time.time())
                        )
                        
                        # Store in cache
                        success = self.preflop_cache.store_preflop_result(
                            hand_notation, int(num_opponents), cache_result, simulation_mode
                        )
                        
                        if success:
                            loaded_count += 1
                
                except Exception as e:
                    # Don't spam warnings for cache loading issues
                    logger.debug(f"Failed to load cached result {key}: {e}")
            
            logger.info(f"Loaded {loaded_count} cached results from {cache_file}")
            return loaded_count > 0
            
        except Exception as e:
            # Silently fail - corrupted cache files can be regenerated
            logger.debug(f"Failed to load cache data from {cache_file}: {e}")
            return False
    
    def _save_cache_data(self) -> bool:
        """Save cache data to file for future startup loads."""
        if not self.preflop_cache:
            return False
        
        try:
            # Extract current cache data
            cache_data = {}
            
            for hand_def in self.priority_hands:
                for num_opponents in range(1, self.config.max_opponents + 1):
                    for simulation_mode in self.config.simulation_modes:
                        result = self.preflop_cache.get_preflop_result(
                            hand_def.notation, num_opponents, simulation_mode
                        )
                        
                        if result:
                            key = f"{hand_def.notation}:{num_opponents}:{simulation_mode}"
                            cache_data[key] = {
                                'win_probability': result.win_probability,
                                'tie_probability': result.tie_probability,
                                'loss_probability': result.loss_probability,
                                'confidence_interval': list(result.confidence_interval) if result.confidence_interval else None,
                                'simulations_run': result.simulations_run,
                                'execution_time_ms': result.execution_time_ms,
                                'hand_categories': result.hand_categories or {},
                                'timestamp': time.time()
                            }
            
            # Save to file
            save_data = {
                'version': '1.0',
                'created_at': time.time(),
                'total_results': len(cache_data),
                'preflop_results': cache_data
            }
            
            with open(self.config.cache_file_path, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            logger.info(f"Saved {len(cache_data)} cache results to {self.config.cache_file_path}")
            return True
            
        except Exception as e:
            # Silently fail - cache save is optional optimization
            logger.debug(f"Failed to save cache data: {e}")
            return False
    
    def _log_population_results(self, result: PopulationResult) -> None:
        """Log population results."""
        logger.info("Startup cache population completed!")
        logger.info(f"  Time: {result.population_time_seconds:.1f}s")
        logger.info(f"  Populated: {result.scenarios_populated}")
        logger.info(f"  Skipped: {result.scenarios_skipped}")
        logger.info(f"  Failed: {result.scenarios_failed}")
        logger.info(f"  Success rate: {result.success_rate:.1%}")
        logger.info(f"  Coverage: {result.initial_coverage:.1f}% → {result.final_coverage:.1f}%")
        
        if result.performance_improvement > 1:
            logger.info(f"  Estimated speedup: {result.performance_improvement:.0f}x for cached scenarios")
    
    def quick_populate_essentials(self, simulation_callback: Callable) -> PopulationResult:
        """
        Quick population of only the most essential scenarios.
        
        For very fast startup times, populate only AA-QQ vs 2-3 opponents.
        """
        original_config = self.config
        
        # Temporarily use minimal config
        self.config = StartupPopulationConfig(
            enabled=True,
            priority_hands_only=True,
            max_opponents=3,  # Only 1-3 opponents
            simulation_modes=["fast"],  # Only fast mode
            hand_categories=["premium"],  # Only premium hands
            max_population_time_seconds=10,  # 10 second limit
            progress_reporting_interval=20
        )
        
        # Regenerate hands with new config
        self.priority_hands = [h for h in self.all_hands if h.category in self.config.hand_categories]
        
        try:
            result = self.populate_startup_cache(simulation_callback)
            return result
        finally:
            # Restore original config
            self.config = original_config
            self.priority_hands = [h for h in self.all_hands if h.category in original_config.hand_categories]


# Integration helper functions
def populate_preflop_on_startup(simulation_callback: Callable,
                               unified_cache: Optional[ThreadSafeMonteCarloCache] = None,
                               quick_mode: bool = False) -> PopulationResult:
    """
    Populate preflop cache during application startup.
    
    Args:
        simulation_callback: Function to run simulations
        unified_cache: Cache instance to use
        quick_mode: If True, only populate most essential scenarios
        
    Returns:
        PopulationResult with population statistics
    """
    from .preflop_cache import get_preflop_cache
    
    # Get or create preflop cache
    preflop_cache = get_preflop_cache(unified_cache=unified_cache)
    
    # Create populator
    config = StartupPopulationConfig(
        priority_hands_only=True,
        max_population_time_seconds=15 if quick_mode else 30,
        hand_categories=["premium"] if quick_mode else ["premium", "strong"]
    )
    
    populator = StartupCachePopulator(config, unified_cache, preflop_cache)
    
    # Run population
    if quick_mode:
        return populator.quick_populate_essentials(simulation_callback)
    else:
        return populator.populate_startup_cache(simulation_callback)


def should_skip_startup_population(preflop_cache: Optional[PreflopCache] = None,
                                 threshold: float = 0.75) -> Tuple[bool, str]:
    """
    Check if startup population can be skipped.
    
    Returns:
        (should_skip, reason)
    """
    if not preflop_cache:
        return False, "No preflop cache available"
    
    stats = preflop_cache.get_stats()
    coverage = stats.coverage_percentage
    
    if coverage >= threshold:
        return True, f"Coverage {coverage:.1f}% >= threshold {threshold:.1f}%"
    
    return False, f"Coverage {coverage:.1f}% below threshold {threshold:.1f}%"