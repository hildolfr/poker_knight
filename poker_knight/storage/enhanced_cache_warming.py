"""
♞ Poker Knight Advanced Cache Warming System

Intelligent cache warming with configurable policies, priority-based population,
and adaptive strategies. Replaces the legacy cache warming with a more
sophisticated approach optimized for the unified cache architecture.

Author: hildolfr
License: MIT
"""

import time
import threading
import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import psutil

# Import cache components
from .unified_cache import ThreadSafeMonteCarloCache, CacheResult, create_cache_key
from .preflop_cache import PreflopCache, PreflopCacheConfig, PreflopHandGenerator

logger = logging.getLogger(__name__)


class WarmingStrategy(Enum):
    """Cache warming strategies."""
    PRIORITY_FIRST = "priority_first"      # Premium hands first
    BREADTH_FIRST = "breadth_first"        # All opponents for each hand
    DEPTH_FIRST = "depth_first"            # All hands for each opponent count
    ADAPTIVE = "adaptive"                  # Adjust based on system resources
    INCREMENTAL = "incremental"            # Small batches over time


@dataclass
class WarmingConfig:
    """Configuration for cache warming system."""
    # Core settings
    enabled: bool = True
    strategy: WarmingStrategy = WarmingStrategy.PRIORITY_FIRST
    max_warming_time_minutes: int = 10
    background_warming: bool = True
    
    # Performance settings
    max_worker_threads: int = 2
    batch_size: int = 20
    delay_between_batches_ms: int = 100
    cpu_usage_threshold: float = 0.8  # Pause if CPU > 80%
    memory_usage_threshold: float = 0.85  # Pause if memory > 85%
    
    # Coverage settings
    target_coverage: float = 0.90  # Target 90% coverage
    priority_coverage_first: bool = True
    min_coverage_for_background: float = 0.50  # Start background at 50%
    
    # Scope settings
    max_opponents: int = 6
    simulation_modes: List[str] = None
    hand_categories: List[str] = None  # ["premium", "strong"] or None for all
    
    # Persistence settings
    save_progress: bool = True
    progress_file: str = "cache_warming_progress.json"
    
    def __post_init__(self):
        if self.simulation_modes is None:
            self.simulation_modes = ["fast", "default"]  # Skip precision for warming
        if self.hand_categories is None:
            self.hand_categories = ["premium", "strong", "medium"]


@dataclass
class WarmingProgress:
    """Tracks cache warming progress."""
    total_scenarios: int = 0
    completed_scenarios: int = 0
    skipped_scenarios: int = 0  # Already cached
    failed_scenarios: int = 0
    current_strategy: str = ""
    current_phase: str = ""
    start_time: Optional[float] = None
    estimated_completion: Optional[float] = None
    coverage_percentage: float = 0.0
    scenarios_per_second: float = 0.0
    
    @property
    def completion_percentage(self) -> float:
        if self.total_scenarios == 0:
            return 0.0
        return (self.completed_scenarios / self.total_scenarios) * 100


@dataclass
class WarmingTask:
    """Individual warming task."""
    hand_notation: str
    num_opponents: int
    simulation_mode: str
    priority: int  # Lower = higher priority
    category: str
    estimated_time_ms: float = 100.0


class SystemMonitor:
    """Monitors system resources during cache warming."""
    
    def __init__(self, cpu_threshold: float = 0.8, memory_threshold: float = 0.85):
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self._last_check = 0
        self._check_interval = 1.0  # Check every second
    
    def should_pause_warming(self) -> Tuple[bool, str]:
        """Check if warming should be paused due to resource constraints."""
        current_time = time.time()
        
        # Don't check too frequently
        if current_time - self._last_check < self._check_interval:
            return False, ""
        
        self._last_check = current_time
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.cpu_threshold * 100:
                return True, f"High CPU usage: {cpu_percent:.1f}%"
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold * 100:
                return True, f"High memory usage: {memory.percent:.1f}%"
            
            return False, ""
            
        except Exception as e:
            logger.warning(f"Failed to check system resources: {e}")
            return False, ""


class CacheWarmer:
    """
    Advanced cache warming system with intelligent strategies.
    """
    
    def __init__(self, 
                 config: Optional[WarmingConfig] = None,
                 unified_cache: Optional[ThreadSafeMonteCarloCache] = None,
                 preflop_cache: Optional[PreflopCache] = None,
                 simulation_callback: Optional[Callable] = None):
        
        self.config = config or WarmingConfig()
        self.unified_cache = unified_cache
        self.preflop_cache = preflop_cache
        self.simulation_callback = simulation_callback
        
        # Progress tracking
        self.progress = WarmingProgress()
        self.progress.current_strategy = self.config.strategy.value
        
        # System monitoring
        self.system_monitor = SystemMonitor(
            self.config.cpu_usage_threshold,
            self.config.memory_usage_threshold
        )
        
        # Thread management
        self._warming_thread = None
        self._stop_warming = threading.Event()
        self._warming_lock = threading.Lock()
        self._task_queue = queue.PriorityQueue()
        
        # Performance tracking
        self._timing_history = []
        self._avg_scenario_time = 100.0  # Initial estimate in ms
        
        logger.info(f"Cache warmer initialized with {self.config.strategy.value} strategy")
    
    def set_simulation_callback(self, callback: Callable) -> None:
        """Set the simulation callback function."""
        self.simulation_callback = callback
    
    def generate_warming_tasks(self) -> List[WarmingTask]:
        """Generate warming tasks based on strategy."""
        if not self.preflop_cache:
            logger.error("No preflop cache available for warming")
            return []
        
        all_hands = PreflopHandGenerator.generate_all_hands()
        tasks = []
        
        # Filter hands by category if specified
        if self.config.hand_categories:
            hands_to_warm = [h for h in all_hands if h.category in self.config.hand_categories]
        else:
            hands_to_warm = all_hands
        
        # Generate tasks
        for hand_def in hands_to_warm:
            for num_opponents in range(1, self.config.max_opponents + 1):
                for simulation_mode in self.config.simulation_modes:
                    priority = self._calculate_priority(hand_def, num_opponents, simulation_mode)
                    
                    task = WarmingTask(
                        hand_notation=hand_def.notation,
                        num_opponents=num_opponents,
                        simulation_mode=simulation_mode,
                        priority=priority,
                        category=hand_def.category,
                        estimated_time_ms=self._estimate_scenario_time(hand_def, simulation_mode)
                    )
                    tasks.append(task)
        
        # Sort tasks by strategy
        tasks = self._sort_tasks_by_strategy(tasks)
        
        self.progress.total_scenarios = len(tasks)
        logger.info(f"Generated {len(tasks)} warming tasks")
        
        return tasks
    
    def _calculate_priority(self, hand_def, num_opponents: int, simulation_mode: str) -> int:
        """Calculate task priority (lower = higher priority)."""
        priority = 0
        
        # Hand strength priority
        category_priorities = {"premium": 10, "strong": 20, "medium": 30, "weak": 40, "trash": 50}
        priority += category_priorities.get(hand_def.category, 50)
        
        # Opponent count priority (common counts first)
        opponent_priorities = {2: 0, 3: 1, 1: 2, 4: 3, 5: 4, 6: 5}
        priority += opponent_priorities.get(num_opponents, 10)
        
        # Simulation mode priority
        mode_priorities = {"fast": 0, "default": 1, "precision": 2}
        priority += mode_priorities.get(simulation_mode, 5)
        
        return priority
    
    def _estimate_scenario_time(self, hand_def, simulation_mode: str) -> float:
        """Estimate time for scenario based on mode and hand."""
        base_times = {"fast": 50, "default": 100, "precision": 300}  # ms
        base_time = base_times.get(simulation_mode, 100)
        
        # Adjust for hand complexity
        if hand_def.category == "premium":
            return base_time * 0.8  # Faster convergence
        elif hand_def.category == "trash":
            return base_time * 1.2  # Slower convergence
        
        return base_time
    
    def _sort_tasks_by_strategy(self, tasks: List[WarmingTask]) -> List[WarmingTask]:
        """Sort tasks according to warming strategy."""
        if self.config.strategy == WarmingStrategy.PRIORITY_FIRST:
            return sorted(tasks, key=lambda t: t.priority)
        
        elif self.config.strategy == WarmingStrategy.BREADTH_FIRST:
            # Group by hand, then by opponents
            return sorted(tasks, key=lambda t: (t.hand_notation, t.num_opponents, t.priority))
        
        elif self.config.strategy == WarmingStrategy.DEPTH_FIRST:
            # Group by opponents, then by hand
            return sorted(tasks, key=lambda t: (t.num_opponents, t.priority))
        
        elif self.config.strategy == WarmingStrategy.ADAPTIVE:
            # Start with priority, but adapt based on performance
            return sorted(tasks, key=lambda t: t.priority)
        
        elif self.config.strategy == WarmingStrategy.INCREMENTAL:
            # Small batches over time
            return sorted(tasks, key=lambda t: (t.priority, t.hand_notation))
        
        return tasks
    
    def warm_cache_synchronous(self) -> WarmingProgress:
        """Perform synchronous cache warming."""
        if not self.simulation_callback:
            logger.error("No simulation callback provided for cache warming")
            return self.progress
        
        self.progress.start_time = time.time()
        self.progress.current_phase = "Generating tasks"
        
        # Generate warming tasks
        tasks = self.generate_warming_tasks()
        if not tasks:
            return self.progress
        
        self.progress.current_phase = "Warming cache"
        
        # Process tasks
        for i, task in enumerate(tasks):
            if self._stop_warming.is_set():
                break
            
            # Check system resources
            should_pause, reason = self.system_monitor.should_pause_warming()
            if should_pause:
                logger.info(f"Pausing warming: {reason}")
                time.sleep(2.0)  # Brief pause
                continue
            
            # Check if already cached
            if self.preflop_cache:
                existing = self.preflop_cache.get_preflop_result(
                    task.hand_notation, task.num_opponents, task.simulation_mode
                )
                if existing:
                    self.progress.skipped_scenarios += 1
                    continue
            
            # Execute warming task
            success = self._execute_warming_task(task)
            
            if success:
                self.progress.completed_scenarios += 1
            else:
                self.progress.failed_scenarios += 1
            
            # Update progress
            self._update_progress()
            
            # Progress reporting
            if (i + 1) % self.config.batch_size == 0:
                self._report_progress()
            
            # Brief delay to avoid overwhelming system
            if self.config.delay_between_batches_ms > 0:
                time.sleep(self.config.delay_between_batches_ms / 1000.0)
        
        self.progress.current_phase = "Complete"
        self._report_final_progress()
        
        return self.progress
    
    def start_background_warming(self) -> bool:
        """Start background cache warming in separate thread."""
        if self._warming_thread and self._warming_thread.is_alive():
            logger.warning("Background warming already running")
            return False
        
        if not self.simulation_callback:
            logger.error("No simulation callback provided for background warming")
            return False
        
        def background_worker():
            try:
                logger.info("Starting background cache warming")
                self.warm_cache_synchronous()
                logger.info("Background cache warming completed")
            except Exception as e:
                logger.error(f"Background warming failed: {e}")
        
        self._stop_warming.clear()
        self._warming_thread = threading.Thread(target=background_worker, daemon=True)
        self._warming_thread.start()
        
        return True
    
    def stop_background_warming(self) -> None:
        """Stop background cache warming."""
        self._stop_warming.set()
        
        if self._warming_thread and self._warming_thread.is_alive():
            logger.info("Stopping background warming...")
            self._warming_thread.join(timeout=10.0)
        
        logger.info("Background warming stopped")
    
    def _execute_warming_task(self, task: WarmingTask) -> bool:
        """Execute a single warming task."""
        try:
            start_time = time.time()
            
            # Call simulation
            result = self.simulation_callback(
                task.hand_notation, 
                task.num_opponents, 
                task.simulation_mode
            )
            
            if result and self.preflop_cache:
                # Store result
                success = self.preflop_cache.store_preflop_result(
                    task.hand_notation,
                    task.num_opponents,
                    result,
                    task.simulation_mode
                )
                
                # Track timing
                execution_time = (time.time() - start_time) * 1000
                self._timing_history.append(execution_time)
                
                # Update average timing (rolling average)
                if len(self._timing_history) > 10:
                    self._timing_history = self._timing_history[-10:]
                self._avg_scenario_time = sum(self._timing_history) / len(self._timing_history)
                
                return success
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing warming task {task.hand_notation}: {e}")
            return False
    
    def _update_progress(self) -> None:
        """Update progress calculations."""
        total_processed = self.progress.completed_scenarios + self.progress.skipped_scenarios + self.progress.failed_scenarios
        
        if total_processed > 0:
            # Calculate scenarios per second
            if self.progress.start_time:
                elapsed_time = time.time() - self.progress.start_time
                self.progress.scenarios_per_second = total_processed / elapsed_time if elapsed_time > 0 else 0
            
            # Estimate completion time
            remaining_scenarios = self.progress.total_scenarios - total_processed
            if self.progress.scenarios_per_second > 0:
                remaining_seconds = remaining_scenarios / self.progress.scenarios_per_second
                self.progress.estimated_completion = time.time() + remaining_seconds
        
        # Update coverage if preflop cache available
        if self.preflop_cache:
            stats = self.preflop_cache.get_stats()
            self.progress.coverage_percentage = stats.coverage_percentage
    
    def _report_progress(self) -> None:
        """Report current progress."""
        self._update_progress()
        
        completion_pct = self.progress.completion_percentage
        coverage_pct = self.progress.coverage_percentage
        rate = self.progress.scenarios_per_second
        
        logger.info(f"Warming progress: {completion_pct:.1f}% complete, "
                   f"{coverage_pct:.1f}% coverage, {rate:.1f} scenarios/sec")
        
        if self.progress.estimated_completion:
            eta_minutes = (self.progress.estimated_completion - time.time()) / 60
            if eta_minutes > 0:
                logger.info(f"Estimated completion: {eta_minutes:.1f} minutes")
    
    def _report_final_progress(self) -> None:
        """Report final warming results."""
        total_time = time.time() - self.progress.start_time if self.progress.start_time else 0
        
        logger.info("Cache warming completed!")
        logger.info(f"  Total time: {total_time:.1f} seconds")
        logger.info(f"  Completed scenarios: {self.progress.completed_scenarios}")
        logger.info(f"  Skipped scenarios: {self.progress.skipped_scenarios}")
        logger.info(f"  Failed scenarios: {self.progress.failed_scenarios}")
        logger.info(f"  Final coverage: {self.progress.coverage_percentage:.1f}%")
        logger.info(f"  Average rate: {self.progress.scenarios_per_second:.1f} scenarios/sec")
    
    def get_warming_status(self) -> Dict[str, Any]:
        """Get current warming status."""
        is_running = self._warming_thread and self._warming_thread.is_alive()
        
        return {
            "is_running": is_running,
            "strategy": self.config.strategy.value,
            "progress": asdict(self.progress),
            "config": {
                "enabled": self.config.enabled,
                "background_warming": self.config.background_warming,
                "target_coverage": self.config.target_coverage,
                "max_warming_time_minutes": self.config.max_warming_time_minutes
            },
            "system_resources": {
                "cpu_threshold": self.config.cpu_usage_threshold,
                "memory_threshold": self.config.memory_usage_threshold
            }
        }


# Factory functions for easy integration
def create_cache_warmer(config: Optional[WarmingConfig] = None,
                       unified_cache: Optional[ThreadSafeMonteCarloCache] = None,
                       preflop_cache: Optional[PreflopCache] = None) -> CacheWarmer:
    """Create cache warmer with specified configuration."""
    return CacheWarmer(config, unified_cache, preflop_cache)


def quick_warm_preflop_cache(simulation_callback: Callable,
                            priority_only: bool = True,
                            max_time_minutes: int = 5) -> WarmingProgress:
    """Quick warming of preflop cache with priority hands."""
    from .preflop_cache import get_preflop_cache
    
    config = WarmingConfig(
        strategy=WarmingStrategy.PRIORITY_FIRST,
        max_warming_time_minutes=max_time_minutes,
        background_warming=False,
        hand_categories=["premium", "strong"] if priority_only else None
    )
    
    preflop_cache = get_preflop_cache()
    warmer = create_cache_warmer(config, preflop_cache=preflop_cache)
    warmer.set_simulation_callback(simulation_callback)
    
    return warmer.warm_cache_synchronous()