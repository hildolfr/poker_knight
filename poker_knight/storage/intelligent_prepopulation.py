"""
♞ Poker Knight Intelligent Cache Prepopulation

Smart prepopulation system that learns from usage patterns to optimize
startup cache population. Focuses on application startup rather than
background warming, prioritizing scenarios most likely to be requested.

Author: hildolfr
License: MIT
"""

import time
import json
import os
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, Counter
from pathlib import Path
from enum import Enum
import statistics
import hashlib

from .unified_cache import CacheKey, CacheResult
from .preflop_cache import PreflopHandGenerator, PreflopCache
from .startup_prepopulation import StartupCachePopulator, StartupPopulationConfig, PopulationResult

logger = logging.getLogger(__name__)


class UsagePattern(Enum):
    """Types of usage patterns."""
    FREQUENT = "frequent"           # High frequency, consistent usage
    BURST = "burst"                # Periodic high activity bursts  
    TRENDING = "trending"           # Increasing usage over time
    SEASONAL = "seasonal"           # Time-based patterns
    RARE = "rare"                  # Infrequent but important scenarios


@dataclass
class ScenarioUsageStats:
    """Usage statistics for a specific poker scenario."""
    scenario_id: str                # Unique identifier for scenario
    hand_notation: str              # Hand notation (e.g., "AKs", "QQ")
    num_opponents: int              # Number of opponents
    simulation_mode: str            # Simulation mode
    
    # Usage metrics
    total_requests: int = 0
    requests_last_24h: int = 0
    requests_last_week: int = 0
    last_requested: Optional[float] = None
    first_requested: Optional[float] = None
    
    # Pattern analysis
    usage_pattern: UsagePattern = UsagePattern.RARE
    request_frequency: float = 0.0      # Requests per hour
    recency_score: float = 0.0          # How recently used (0-1)
    consistency_score: float = 0.0      # How consistent the usage (0-1)
    trend_score: float = 0.0           # Usage trend (-1 to 1)
    
    # Performance impact
    avg_computation_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    importance_score: float = 0.0       # Overall importance (0-1)
    priority_rank: int = 0              # Prepopulation priority rank


@dataclass
class UsageContext:
    """Context information for usage pattern analysis."""
    session_id: str
    user_type: str = "default"          # "novice", "intermediate", "expert"
    game_type: str = "holdem"           # "holdem", "tournament", etc.
    stack_depth_category: str = "medium" # "shallow", "medium", "deep"
    time_of_day: int = 12               # 0-23 hour
    day_of_week: int = 1                # 1-7 (Monday=1)


@dataclass
class PrepopulationStrategy:
    """Strategy configuration for intelligent prepopulation."""
    name: str
    description: str
    
    # Scenario selection criteria
    min_importance_score: float = 0.1
    max_scenarios: int = 500
    prefer_frequent: bool = True
    prefer_recent: bool = True
    include_trending: bool = True
    
    # Time allocation
    max_prepopulation_time_seconds: int = 30
    priority_time_allocation: Dict[UsagePattern, float] = field(default_factory=lambda: {
        UsagePattern.FREQUENT: 0.5,    # 50% of time on frequent scenarios
        UsagePattern.TRENDING: 0.3,    # 30% on trending scenarios
        UsagePattern.BURST: 0.15,      # 15% on burst patterns
        UsagePattern.SEASONAL: 0.05    # 5% on seasonal patterns
    })


class UsagePatternAnalyzer:
    """Analyzes poker scenario usage patterns for intelligent prepopulation."""
    
    def __init__(self, usage_history_file: str = "poker_usage_patterns.json"):
        self.usage_history_file = usage_history_file
        self._scenario_stats: Dict[str, ScenarioUsageStats] = {}
        self._context_patterns: Dict[str, List[UsageContext]] = defaultdict(list)
        self._load_usage_history()
    
    def record_scenario_usage(self, 
                            hand_notation: str,
                            num_opponents: int,
                            simulation_mode: str,
                            computation_time_ms: float,
                            was_cached: bool,
                            context: Optional[UsageContext] = None):
        """Record usage of a poker scenario for pattern analysis."""
        
        scenario_id = self._create_scenario_id(hand_notation, num_opponents, simulation_mode)
        current_time = time.time()
        
        # Get or create scenario stats
        if scenario_id not in self._scenario_stats:
            self._scenario_stats[scenario_id] = ScenarioUsageStats(
                scenario_id=scenario_id,
                hand_notation=hand_notation,
                num_opponents=num_opponents,
                simulation_mode=simulation_mode,
                first_requested=current_time
            )
        
        stats = self._scenario_stats[scenario_id]
        
        # Update usage counts
        stats.total_requests += 1
        stats.last_requested = current_time
        
        # Update recent usage (last 24h and week)
        day_ago = current_time - 86400
        week_ago = current_time - 604800
        
        # Recalculate recent usage (simplified - in production would use time buckets)
        stats.requests_last_24h = max(1, int(stats.total_requests * 0.1))  # Rough estimate
        stats.requests_last_week = max(1, int(stats.total_requests * 0.5))  # Rough estimate
        
        # Update performance metrics
        if stats.avg_computation_time_ms == 0:
            stats.avg_computation_time_ms = computation_time_ms
        else:
            # Exponential moving average
            stats.avg_computation_time_ms = (0.9 * stats.avg_computation_time_ms + 
                                           0.1 * computation_time_ms)
        
        # Update cache hit rate
        if was_cached:
            stats.cache_hit_rate = min(1.0, stats.cache_hit_rate + 0.1)
        else:
            stats.cache_hit_rate = max(0.0, stats.cache_hit_rate - 0.05)
        
        # Record context if provided
        if context:
            self._context_patterns[scenario_id].append(context)
            # Keep only recent contexts
            if len(self._context_patterns[scenario_id]) > 100:
                self._context_patterns[scenario_id] = self._context_patterns[scenario_id][-50:]
        
        # Analyze and update patterns
        self._analyze_scenario_patterns(scenario_id)
        
        # Periodic save (every 10 requests)
        if stats.total_requests % 10 == 0:
            self._save_usage_history()
    
    def _create_scenario_id(self, hand_notation: str, num_opponents: int, simulation_mode: str) -> str:
        """Create unique scenario identifier."""
        base_string = f"{hand_notation}:{num_opponents}:{simulation_mode}"
        return hashlib.md5(base_string.encode()).hexdigest()[:16]
    
    def _analyze_scenario_patterns(self, scenario_id: str):
        """Analyze usage patterns for a specific scenario."""
        stats = self._scenario_stats[scenario_id]
        current_time = time.time()
        
        if not stats.first_requested or not stats.last_requested:
            return
        
        # Calculate time-based metrics
        total_duration = current_time - stats.first_requested
        time_since_last = current_time - stats.last_requested
        
        # Request frequency (requests per hour)
        if total_duration > 0:
            stats.request_frequency = (stats.total_requests / total_duration) * 3600
        
        # Recency score (how recently used)
        max_recency_hours = 168  # 1 week
        hours_since_last = time_since_last / 3600
        stats.recency_score = max(0, 1.0 - (hours_since_last / max_recency_hours))
        
        # Consistency score (how regular the usage)
        if stats.total_requests > 2:
            # Calculate coefficient of variation for consistency
            # For simplicity, use request frequency vs total requests ratio
            expected_frequency = stats.total_requests / (total_duration / 3600) if total_duration > 0 else 0
            if expected_frequency > 0:
                consistency_ratio = stats.request_frequency / expected_frequency
                stats.consistency_score = 1.0 - abs(1.0 - consistency_ratio)
            else:
                stats.consistency_score = 0.0
        
        # Trend score (increasing/decreasing usage)
        if stats.requests_last_week > 0 and stats.total_requests > stats.requests_last_week:
            recent_ratio = stats.requests_last_week / stats.total_requests
            if recent_ratio > 0.7:  # 70% of requests in last week
                stats.trend_score = 0.5  # Trending up
            elif recent_ratio < 0.3:  # 30% of requests in last week
                stats.trend_score = -0.5  # Trending down
            else:
                stats.trend_score = 0.0  # Stable
        
        # Determine usage pattern
        if stats.request_frequency > 1.0:  # More than 1 request per hour
            stats.usage_pattern = UsagePattern.FREQUENT
        elif stats.trend_score > 0.3:
            stats.usage_pattern = UsagePattern.TRENDING
        elif stats.consistency_score < 0.3 and stats.request_frequency > 0.1:
            stats.usage_pattern = UsagePattern.BURST
        elif self._has_seasonal_pattern(scenario_id):
            stats.usage_pattern = UsagePattern.SEASONAL
        else:
            stats.usage_pattern = UsagePattern.RARE
        
        # Calculate overall importance score
        stats.importance_score = self._calculate_importance_score(stats)
    
    def _has_seasonal_pattern(self, scenario_id: str) -> bool:
        """Check if scenario has seasonal/time-based patterns."""
        contexts = self._context_patterns.get(scenario_id, [])
        if len(contexts) < 10:
            return False
        
        # Check for time-of-day patterns
        hours = [ctx.time_of_day for ctx in contexts]
        hour_variance = statistics.variance(hours) if len(set(hours)) > 1 else 0
        
        # Check for day-of-week patterns  
        days = [ctx.day_of_week for ctx in contexts]
        day_variance = statistics.variance(days) if len(set(days)) > 1 else 0
        
        # Low variance indicates seasonal pattern
        return hour_variance < 20 or day_variance < 2
    
    def _calculate_importance_score(self, stats: ScenarioUsageStats) -> float:
        """Calculate overall importance score for prepopulation priority."""
        
        # Base score from usage frequency
        frequency_score = min(1.0, stats.request_frequency / 2.0)  # Cap at 2 requests/hour
        
        # Recency bonus
        recency_score = stats.recency_score * 0.3
        
        # Consistency bonus
        consistency_score = stats.consistency_score * 0.2
        
        # Trend bonus/penalty
        trend_score = abs(stats.trend_score) * 0.2
        
        # Performance impact factor
        if stats.avg_computation_time_ms > 100:  # Expensive computations
            performance_score = 0.3
        elif stats.avg_computation_time_ms > 50:
            performance_score = 0.2
        else:
            performance_score = 0.1
        
        # Pattern-specific weights
        pattern_weights = {
            UsagePattern.FREQUENT: 1.0,
            UsagePattern.TRENDING: 0.8,
            UsagePattern.BURST: 0.6,
            UsagePattern.SEASONAL: 0.4,
            UsagePattern.RARE: 0.1
        }
        
        pattern_score = pattern_weights.get(stats.usage_pattern, 0.1)
        
        # Combine scores
        importance = (frequency_score * 0.4 + 
                     recency_score + 
                     consistency_score + 
                     trend_score + 
                     performance_score) * pattern_score
        
        return min(1.0, importance)
    
    def get_prepopulation_priorities(self, strategy: PrepopulationStrategy) -> List[ScenarioUsageStats]:
        """Get prioritized list of scenarios for prepopulation."""
        
        # Filter scenarios by minimum importance
        candidates = [
            stats for stats in self._scenario_stats.values()
            if stats.importance_score >= strategy.min_importance_score
        ]
        
        # Apply strategy preferences
        if strategy.prefer_frequent:
            candidates = [s for s in candidates if s.usage_pattern == UsagePattern.FREQUENT] + \
                        [s for s in candidates if s.usage_pattern != UsagePattern.FREQUENT]
        
        if strategy.prefer_recent:
            candidates.sort(key=lambda s: s.recency_score, reverse=True)
        
        # Calculate priority ranks based on importance score
        candidates.sort(key=lambda s: s.importance_score, reverse=True)
        
        for i, stats in enumerate(candidates):
            stats.priority_rank = i + 1
        
        # Limit to max scenarios
        return candidates[:strategy.max_scenarios]
    
    def _load_usage_history(self):
        """Load usage history from file."""
        try:
            if os.path.exists(self.usage_history_file):
                with open(self.usage_history_file, 'r') as f:
                    data = json.load(f)
                
                # Load scenario stats
                for scenario_data in data.get('scenarios', []):
                    stats = ScenarioUsageStats(**scenario_data)
                    self._scenario_stats[stats.scenario_id] = stats
                
                # Load context patterns
                for scenario_id, contexts_data in data.get('contexts', {}).items():
                    self._context_patterns[scenario_id] = [
                        UsageContext(**ctx_data) for ctx_data in contexts_data
                    ]
                
                logger.info(f"Loaded {len(self._scenario_stats)} scenario usage patterns")
        
        except Exception as e:
            logger.warning(f"Failed to load usage history: {e}")
    
    def _save_usage_history(self):
        """Save usage history to file."""
        try:
            # Prepare data for serialization
            data = {
                'version': '1.0',
                'last_updated': time.time(),
                'scenarios': [asdict(stats) for stats in self._scenario_stats.values()],
                'contexts': {
                    scenario_id: [asdict(ctx) for ctx in contexts]
                    for scenario_id, contexts in self._context_patterns.items()
                }
            }
            
            # Save to file
            with open(self.usage_history_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved usage history with {len(self._scenario_stats)} scenarios")
        
        except Exception as e:
            logger.error(f"Failed to save usage history: {e}")
    
    def get_usage_analytics(self) -> Dict[str, Any]:
        """Get usage analytics summary."""
        if not self._scenario_stats:
            return {'error': 'No usage data available'}
        
        all_stats = list(self._scenario_stats.values())
        
        # Pattern distribution
        pattern_counts = Counter(stats.usage_pattern for stats in all_stats)
        
        # Top scenarios by importance
        top_scenarios = sorted(all_stats, key=lambda s: s.importance_score, reverse=True)[:20]
        
        # Usage metrics
        total_requests = sum(stats.total_requests for stats in all_stats)
        avg_frequency = statistics.mean([stats.request_frequency for stats in all_stats])
        
        return {
            'total_scenarios': len(all_stats),
            'total_requests': total_requests,
            'avg_request_frequency': avg_frequency,
            'pattern_distribution': dict(pattern_counts),
            'top_scenarios': [
                {
                    'hand': stats.hand_notation,
                    'opponents': stats.num_opponents,
                    'mode': stats.simulation_mode,
                    'importance': stats.importance_score,
                    'pattern': stats.usage_pattern.value,
                    'frequency': stats.request_frequency,
                    'requests': stats.total_requests
                }
                for stats in top_scenarios
            ]
        }


class IntelligentPrepopulator:
    """
    Intelligent cache prepopulation system that learns from usage patterns.
    
    Uses historical usage data to prioritize scenarios most likely to be
    requested, optimizing startup prepopulation for maximum impact.
    """
    
    def __init__(self, 
                 analyzer: UsagePatternAnalyzer,
                 base_populator: StartupCachePopulator,
                 strategy: Optional[PrepopulationStrategy] = None):
        
        self.analyzer = analyzer
        self.base_populator = base_populator
        self.strategy = strategy or self._create_default_strategy()
        
        logger.info(f"Intelligent prepopulator initialized with {self.strategy.name} strategy")
    
    def _create_default_strategy(self) -> PrepopulationStrategy:
        """Create default prepopulation strategy."""
        return PrepopulationStrategy(
            name="balanced_intelligent",
            description="Balanced strategy prioritizing frequent and trending scenarios",
            min_importance_score=0.1,
            max_scenarios=300,
            prefer_frequent=True,
            prefer_recent=True,
            include_trending=True,
            max_prepopulation_time_seconds=30
        )
    
    def populate_intelligent_cache(self, simulation_callback) -> PopulationResult:
        """
        Perform intelligent cache prepopulation based on usage patterns.
        
        Args:
            simulation_callback: Function to call for simulations
            
        Returns:
            PopulationResult with population statistics
        """
        start_time = time.time()
        
        # Get prioritized scenarios from usage analysis
        priority_scenarios = self.analyzer.get_prepopulation_priorities(self.strategy)
        
        if not priority_scenarios:
            logger.info("No priority scenarios identified, falling back to standard prepopulation")
            return self.base_populator.populate_startup_cache(simulation_callback)
        
        logger.info(f"Identified {len(priority_scenarios)} priority scenarios for intelligent prepopulation")
        
        # Group scenarios by usage pattern for time allocation
        pattern_groups = defaultdict(list)
        for scenario in priority_scenarios:
            pattern_groups[scenario.usage_pattern].append(scenario)
        
        # Allocate time based on strategy
        total_time = self.strategy.max_prepopulation_time_seconds
        time_allocations = {}
        
        for pattern, allocation_ratio in self.strategy.priority_time_allocation.items():
            if pattern in pattern_groups:
                time_allocations[pattern] = total_time * allocation_ratio
        
        # Execute prepopulation with intelligent prioritization
        result = PopulationResult()
        result.start_time = start_time
        
        for pattern, scenarios in pattern_groups.items():
            if pattern not in time_allocations:
                continue
            
            pattern_time_limit = time_allocations[pattern]
            pattern_start_time = time.time()
            
            logger.info(f"Prepopulating {len(scenarios)} {pattern.value} scenarios "
                       f"(time limit: {pattern_time_limit:.1f}s)")
            
            for scenario in scenarios:
                # Check time limit
                if time.time() - pattern_start_time > pattern_time_limit:
                    logger.info(f"Time limit reached for {pattern.value} scenarios")
                    break
                
                # Check if already cached
                existing = self.base_populator.preflop_cache.get_preflop_result(
                    scenario.hand_notation,
                    scenario.num_opponents,
                    scenario.simulation_mode
                )
                
                if existing:
                    result.scenarios_skipped += 1
                    continue
                
                # Execute simulation
                try:
                    sim_result = simulation_callback(
                        scenario.hand_notation,
                        scenario.num_opponents,
                        scenario.simulation_mode
                    )
                    
                    if sim_result:
                        # Store in cache
                        success = self.base_populator.preflop_cache.store_preflop_result(
                            scenario.hand_notation,
                            scenario.num_opponents,
                            sim_result,
                            scenario.simulation_mode
                        )
                        
                        if success:
                            result.scenarios_populated += 1
                        else:
                            result.scenarios_failed += 1
                    else:
                        result.scenarios_failed += 1
                
                except Exception as e:
                    logger.error(f"Failed to populate {scenario.hand_notation}: {e}")
                    result.scenarios_failed += 1
        
        # Complete result
        result.population_time_seconds = time.time() - start_time
        result.success = (result.success_rate > 0.8)
        
        # Update importance scores based on successful prepopulation
        self._update_prepopulation_feedback(priority_scenarios, result)
        
        logger.info(f"Intelligent prepopulation completed: {result.scenarios_populated} populated, "
                   f"{result.scenarios_skipped} skipped, {result.scenarios_failed} failed "
                   f"in {result.population_time_seconds:.1f}s")
        
        return result
    
    def _update_prepopulation_feedback(self, scenarios: List[ScenarioUsageStats], result: PopulationResult):
        """Update scenario importance based on prepopulation success."""
        # Boost importance of successfully prepopulated scenarios
        successful_scenarios = result.scenarios_populated
        if successful_scenarios > 0:
            for scenario in scenarios[:successful_scenarios]:
                scenario.importance_score = min(1.0, scenario.importance_score * 1.1)
        
        # Save updated patterns
        self.analyzer._save_usage_history()
    
    def get_prepopulation_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for optimizing prepopulation strategy."""
        priority_scenarios = self.analyzer.get_prepopulation_priorities(self.strategy)
        
        # Analyze pattern distribution
        pattern_counts = Counter(s.usage_pattern for s in priority_scenarios)
        
        # Calculate potential time savings
        total_computation_time = sum(s.avg_computation_time_ms for s in priority_scenarios)
        estimated_cache_hits = sum(s.request_frequency for s in priority_scenarios)
        
        potential_time_savings = (total_computation_time * estimated_cache_hits) / 1000  # Convert to seconds
        
        return {
            'strategy': self.strategy.name,
            'priority_scenarios': len(priority_scenarios),
            'pattern_distribution': dict(pattern_counts),
            'estimated_time_savings_per_hour': potential_time_savings,
            'top_priority_hands': [
                {
                    'hand': s.hand_notation,
                    'importance': s.importance_score,
                    'pattern': s.usage_pattern.value,
                    'frequency': s.request_frequency
                }
                for s in priority_scenarios[:10]
            ],
            'recommendations': self._generate_strategy_recommendations(priority_scenarios)
        }
    
    def _generate_strategy_recommendations(self, scenarios: List[ScenarioUsageStats]) -> List[str]:
        """Generate recommendations for improving prepopulation strategy."""
        recommendations = []
        
        if not scenarios:
            recommendations.append("No usage patterns detected - consider using default prepopulation")
            return recommendations
        
        # Analyze patterns
        pattern_counts = Counter(s.usage_pattern for s in scenarios)
        frequent_count = pattern_counts.get(UsagePattern.FREQUENT, 0)
        trending_count = pattern_counts.get(UsagePattern.TRENDING, 0)
        
        if frequent_count > trending_count * 2:
            recommendations.append("High frequent usage detected - consider increasing time allocation for frequent patterns")
        
        if trending_count > len(scenarios) * 0.3:
            recommendations.append("Many trending scenarios detected - consider monitoring for pattern changes")
        
        # High importance scenarios
        high_importance = [s for s in scenarios if s.importance_score > 0.7]
        if len(high_importance) > 50:
            recommendations.append(f"Many high-importance scenarios ({len(high_importance)}) - consider increasing max_scenarios limit")
        
        return recommendations


# Factory functions for easy integration
def create_intelligent_prepopulator(usage_history_file: str = "poker_usage_patterns.json",
                                  base_populator: Optional[StartupCachePopulator] = None,
                                  strategy_name: str = "balanced") -> IntelligentPrepopulator:
    """Create intelligent prepopulator with specified configuration."""
    
    analyzer = UsagePatternAnalyzer(usage_history_file)
    
    if base_populator is None:
        # Create default base populator
        from .startup_prepopulation import StartupPopulationConfig
        config = StartupPopulationConfig()
        base_populator = StartupCachePopulator(config)
    
    # Create strategy
    strategies = {
        "balanced": PrepopulationStrategy(
            name="balanced",
            description="Balanced approach for general usage",
            max_scenarios=300,
            max_prepopulation_time_seconds=30
        ),
        "aggressive": PrepopulationStrategy(
            name="aggressive", 
            description="Aggressive prepopulation for high-usage applications",
            max_scenarios=500,
            max_prepopulation_time_seconds=60,
            min_importance_score=0.05
        ),
        "conservative": PrepopulationStrategy(
            name="conservative",
            description="Conservative prepopulation for resource-constrained environments", 
            max_scenarios=150,
            max_prepopulation_time_seconds=15,
            min_importance_score=0.2
        )
    }
    
    strategy = strategies.get(strategy_name, strategies["balanced"])
    
    return IntelligentPrepopulator(analyzer, base_populator, strategy)