"""
♞ Poker Knight Phase 4 Cache Integration & Validation

Complete integration system for Phase 4 performance & memory optimization.
Provides unified interface for hierarchical caching with intelligent management,
monitoring, and validation of performance improvements.

Author: hildolfr
License: MIT
"""

import time
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from contextlib import contextmanager

from .unified_cache import CacheKey, CacheResult
from .hierarchical_cache import (
    HierarchicalCache, HierarchicalCacheConfig, CacheLayer
)
from .adaptive_cache_manager import (
    AdaptiveCacheManager, AdaptiveCacheConfig, OptimizationStrategy
)
from .cache_performance_monitor import (
    PerformanceMonitor, MonitoringConfig, AlertLevel
)
from .intelligent_prepopulation import (
    IntelligentPrepopulator, UsagePatternAnalyzer, PrepopulationStrategy
)
from .optimized_persistence import (
    OptimizedCachePersistenceManager, PersistenceConfig, PersistenceFormat
)

logger = logging.getLogger(__name__)


@dataclass
class Phase4Config:
    """Complete configuration for Phase 4 cache system."""
    # Core settings
    enabled: bool = True
    optimization_level: str = "balanced"  # "conservative", "balanced", "aggressive"
    auto_start_services: bool = True
    
    # Component configurations
    hierarchical_config: Optional[HierarchicalCacheConfig] = None
    adaptive_config: Optional[AdaptiveCacheConfig] = None
    monitoring_config: Optional[MonitoringConfig] = None
    persistence_config: Optional[PersistenceConfig] = None
    
    # Integration settings
    enable_adaptive_management: bool = True
    enable_performance_monitoring: bool = True
    enable_intelligent_prepopulation: bool = True
    enable_optimized_persistence: bool = True
    
    # Performance targets
    target_overall_hit_rate: float = 0.85
    target_l1_hit_rate: float = 0.60
    max_response_time_ms: float = 10.0
    max_memory_usage_mb: int = 1024
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.hierarchical_config is None:
            self.hierarchical_config = self._create_hierarchical_config()
        
        if self.adaptive_config is None:
            self.adaptive_config = self._create_adaptive_config()
        
        if self.monitoring_config is None:
            self.monitoring_config = self._create_monitoring_config()
        
        if self.persistence_config is None:
            self.persistence_config = self._create_persistence_config()
    
    def _create_hierarchical_config(self) -> HierarchicalCacheConfig:
        """Create hierarchical cache config based on optimization level."""
        if self.optimization_level == "conservative":
            return HierarchicalCacheConfig(
                enable_auto_promotion=False,
                enable_background_demotion=False,
                memory_pressure_threshold=0.75
            )
        elif self.optimization_level == "aggressive":
            return HierarchicalCacheConfig(
                enable_auto_promotion=True,
                enable_background_demotion=True,
                memory_pressure_threshold=0.90,
                cache_warming_enabled=True
            )
        else:  # balanced
            return HierarchicalCacheConfig()
    
    def _create_adaptive_config(self) -> AdaptiveCacheConfig:
        """Create adaptive management config based on optimization level."""
        strategy_map = {
            "conservative": OptimizationStrategy.CONSERVATIVE,
            "balanced": OptimizationStrategy.BALANCED,
            "aggressive": OptimizationStrategy.AGGRESSIVE
        }
        
        return AdaptiveCacheConfig(
            optimization_strategy=strategy_map.get(
                self.optimization_level, OptimizationStrategy.BALANCED
            ),
            adjustment_interval_seconds=300 if self.optimization_level == "aggressive" else 600
        )
    
    def _create_monitoring_config(self) -> MonitoringConfig:
        """Create monitoring config based on optimization level."""
        if self.optimization_level == "conservative":
            return MonitoringConfig(
                collection_interval_seconds=60,
                export_enabled=False
            )
        else:
            return MonitoringConfig(
                collection_interval_seconds=30,
                export_enabled=True
            )
    
    def _create_persistence_config(self) -> PersistenceConfig:
        """Create persistence config based on optimization level."""
        format_map = {
            "conservative": PersistenceFormat.JSON,
            "balanced": PersistenceFormat.COMPRESSED,
            "aggressive": PersistenceFormat.COMPRESSED
        }
        
        return PersistenceConfig(
            format=format_map.get(self.optimization_level, PersistenceFormat.COMPRESSED),
            async_persistence=(self.optimization_level != "conservative"),
            parallel_loading=(self.optimization_level == "aggressive")
        )


@dataclass
class Phase4PerformanceMetrics:
    """Comprehensive performance metrics for Phase 4 validation."""
    # Overall cache performance
    overall_hit_rate: float = 0.0
    overall_response_time_ms: float = 0.0
    total_memory_usage_mb: float = 0.0
    
    # Layer-specific performance
    l1_hit_rate: float = 0.0
    l1_response_time_ms: float = 0.0
    l2_hit_rate: float = 0.0
    l2_response_time_ms: float = 0.0
    l3_hit_rate: float = 0.0
    l3_response_time_ms: float = 0.0
    
    # Adaptive management metrics
    cache_optimizations_applied: int = 0
    memory_pressure_events: int = 0
    auto_promotions: int = 0
    auto_demotions: int = 0
    
    # Performance improvements vs baseline
    hit_rate_improvement: float = 0.0
    response_time_improvement: float = 0.0
    memory_efficiency_improvement: float = 0.0
    
    # System health indicators
    active_alerts: int = 0
    cache_efficiency_score: float = 0.0
    system_stability_score: float = 0.0


class Phase4ValidationSuite:
    """Validation suite for Phase 4 performance improvements."""
    
    def __init__(self, cache_system: 'Phase4CacheSystem'):
        self.cache_system = cache_system
        self._baseline_metrics = None
        self._validation_results = []
    
    def establish_baseline(self, duration_seconds: int = 300) -> Phase4PerformanceMetrics:
        """Establish performance baseline before Phase 4 optimizations."""
        logger.info(f"Establishing performance baseline ({duration_seconds}s)")
        
        start_time = time.time()
        baseline_data = []
        
        while time.time() - start_time < duration_seconds:
            # Collect current metrics
            metrics = self._collect_current_metrics()
            baseline_data.append(metrics)
            
            time.sleep(10)  # Sample every 10 seconds
        
        # Calculate baseline averages
        if baseline_data:
            self._baseline_metrics = self._calculate_average_metrics(baseline_data)
            logger.info("Performance baseline established")
            logger.info(f"Baseline hit rate: {self._baseline_metrics.overall_hit_rate:.2%}")
            logger.info(f"Baseline response time: {self._baseline_metrics.overall_response_time_ms:.1f}ms")
        
        return self._baseline_metrics
    
    def validate_performance_improvements(self, duration_seconds: int = 300) -> Dict[str, Any]:
        """Validate Phase 4 performance improvements against baseline."""
        logger.info(f"Validating Phase 4 performance improvements ({duration_seconds}s)")
        
        if not self._baseline_metrics:
            logger.warning("No baseline metrics available - establishing baseline first")
            self.establish_baseline(60)  # Quick baseline
        
        start_time = time.time()
        validation_data = []
        
        while time.time() - start_time < duration_seconds:
            metrics = self._collect_current_metrics()
            validation_data.append(metrics)
            time.sleep(10)
        
        # Calculate validation averages
        current_metrics = self._calculate_average_metrics(validation_data)
        
        # Compare against baseline
        improvements = self._calculate_improvements(current_metrics)
        
        # Validate against targets
        target_validation = self._validate_against_targets(current_metrics)
        
        # Generate validation report
        validation_report = {
            'timestamp': time.time(),
            'baseline_metrics': asdict(self._baseline_metrics) if self._baseline_metrics else None,
            'current_metrics': asdict(current_metrics),
            'improvements': asdict(improvements),
            'target_validation': target_validation,
            'overall_success': self._determine_validation_success(improvements, target_validation),
            'recommendations': self._generate_recommendations(improvements, target_validation)
        }
        
        self._validation_results.append(validation_report)
        
        logger.info("Phase 4 validation completed")
        if validation_report['overall_success']:
            logger.info("✅ Phase 4 performance improvements validated successfully")
        else:
            logger.warning("⚠️ Phase 4 validation shows areas for improvement")
        
        return validation_report
    
    def _collect_current_metrics(self) -> Phase4PerformanceMetrics:
        """Collect current performance metrics from all components."""
        metrics = Phase4PerformanceMetrics()
        
        try:
            # Hierarchical cache metrics
            if self.cache_system.hierarchical_cache:
                stats = self.cache_system.hierarchical_cache.get_stats()
                metrics.overall_hit_rate = stats.overall_hit_rate
                metrics.l1_hit_rate = stats.l1_hit_rate
                metrics.l1_response_time_ms = stats.avg_l1_response_time_ms
                metrics.l2_response_time_ms = stats.avg_l2_response_time_ms
                metrics.l3_response_time_ms = stats.avg_l3_response_time_ms
                
                # Calculate overall response time
                total_hits = stats.l1_hits + stats.l2_hits + stats.l3_hits
                if total_hits > 0:
                    metrics.overall_response_time_ms = (
                        (stats.l1_hits * stats.avg_l1_response_time_ms +
                         stats.l2_hits * stats.avg_l2_response_time_ms +
                         stats.l3_hits * stats.avg_l3_response_time_ms) / total_hits
                    )
                
                metrics.total_memory_usage_mb = (
                    stats.l1_stats.memory_usage_mb +
                    stats.l2_stats.memory_usage_mb +
                    stats.l3_stats.memory_usage_mb
                )
                
                metrics.auto_promotions = stats.promotions
                metrics.memory_pressure_events = stats.memory_pressure_events
            
            # Adaptive management metrics
            if self.cache_system.adaptive_manager:
                status = self.cache_system.adaptive_manager.get_optimization_status()
                metrics.cache_optimizations_applied = status.get('total_optimizations', 0)
            
            # Performance monitoring metrics
            if self.cache_system.performance_monitor:
                dashboard = self.cache_system.performance_monitor.get_performance_dashboard()
                metrics.active_alerts = len(dashboard.get('active_alerts', []))
                
                current = dashboard.get('current', {})
                metrics.cache_efficiency_score = current.get('cache_efficiency', 0.0)
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
        
        return metrics
    
    def _calculate_average_metrics(self, metrics_list: List[Phase4PerformanceMetrics]) -> Phase4PerformanceMetrics:
        """Calculate average metrics from a list of metric samples."""
        if not metrics_list:
            return Phase4PerformanceMetrics()
        
        avg_metrics = Phase4PerformanceMetrics()
        count = len(metrics_list)
        
        # Sum all metrics
        for metrics in metrics_list:
            avg_metrics.overall_hit_rate += metrics.overall_hit_rate
            avg_metrics.overall_response_time_ms += metrics.overall_response_time_ms
            avg_metrics.total_memory_usage_mb += metrics.total_memory_usage_mb
            avg_metrics.l1_hit_rate += metrics.l1_hit_rate
            avg_metrics.l1_response_time_ms += metrics.l1_response_time_ms
            avg_metrics.l2_hit_rate += metrics.l2_hit_rate
            avg_metrics.l2_response_time_ms += metrics.l2_response_time_ms
            avg_metrics.l3_hit_rate += metrics.l3_hit_rate
            avg_metrics.l3_response_time_ms += metrics.l3_response_time_ms
            avg_metrics.cache_efficiency_score += metrics.cache_efficiency_score
        
        # Calculate averages
        avg_metrics.overall_hit_rate /= count
        avg_metrics.overall_response_time_ms /= count
        avg_metrics.total_memory_usage_mb /= count
        avg_metrics.l1_hit_rate /= count
        avg_metrics.l1_response_time_ms /= count
        avg_metrics.l2_hit_rate /= count
        avg_metrics.l2_response_time_ms /= count
        avg_metrics.l3_hit_rate /= count
        avg_metrics.l3_response_time_ms /= count
        avg_metrics.cache_efficiency_score /= count
        
        # Use last values for counters
        if metrics_list:
            last_metrics = metrics_list[-1]
            avg_metrics.cache_optimizations_applied = last_metrics.cache_optimizations_applied
            avg_metrics.auto_promotions = last_metrics.auto_promotions
            avg_metrics.memory_pressure_events = last_metrics.memory_pressure_events
            avg_metrics.active_alerts = last_metrics.active_alerts
        
        return avg_metrics
    
    def _calculate_improvements(self, current: Phase4PerformanceMetrics) -> Phase4PerformanceMetrics:
        """Calculate improvements vs baseline."""
        improvements = Phase4PerformanceMetrics()
        
        if not self._baseline_metrics:
            return improvements
        
        baseline = self._baseline_metrics
        
        # Calculate percentage improvements
        if baseline.overall_hit_rate > 0:
            improvements.hit_rate_improvement = (
                (current.overall_hit_rate - baseline.overall_hit_rate) / baseline.overall_hit_rate
            )
        
        if baseline.overall_response_time_ms > 0:
            improvements.response_time_improvement = (
                (baseline.overall_response_time_ms - current.overall_response_time_ms) / 
                baseline.overall_response_time_ms
            )
        
        if baseline.total_memory_usage_mb > 0:
            improvements.memory_efficiency_improvement = (
                (baseline.total_memory_usage_mb - current.total_memory_usage_mb) / 
                baseline.total_memory_usage_mb
            )
        
        return improvements
    
    def _validate_against_targets(self, metrics: Phase4PerformanceMetrics) -> Dict[str, Any]:
        """Validate metrics against configured targets."""
        config = self.cache_system.config
        
        validation = {
            'hit_rate_target_met': metrics.overall_hit_rate >= config.target_overall_hit_rate,
            'l1_hit_rate_target_met': metrics.l1_hit_rate >= config.target_l1_hit_rate,
            'response_time_target_met': metrics.overall_response_time_ms <= config.max_response_time_ms,
            'memory_usage_target_met': metrics.total_memory_usage_mb <= config.max_memory_usage_mb,
            'target_score': 0.0
        }
        
        # Calculate overall target achievement score
        targets_met = sum(1 for met in validation.values() if isinstance(met, bool) and met)
        total_targets = sum(1 for val in validation.values() if isinstance(val, bool))
        validation['target_score'] = targets_met / total_targets if total_targets > 0 else 0.0
        
        return validation
    
    def _determine_validation_success(self, improvements: Phase4PerformanceMetrics, 
                                    target_validation: Dict[str, Any]) -> bool:
        """Determine if Phase 4 validation is successful."""
        # Success criteria:
        # 1. At least 80% of targets met
        # 2. Positive improvements in key metrics
        # 3. No critical performance regressions
        
        target_score = target_validation.get('target_score', 0.0)
        hit_rate_improved = improvements.hit_rate_improvement >= 0
        response_time_improved = improvements.response_time_improvement >= 0
        
        return (target_score >= 0.8 and 
                hit_rate_improved and 
                response_time_improved)
    
    def _generate_recommendations(self, improvements: Phase4PerformanceMetrics,
                                target_validation: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        if improvements.hit_rate_improvement < 0.05:  # Less than 5% improvement
            recommendations.append("Consider more aggressive cache pre-population strategies")
        
        if improvements.response_time_improvement < 0.1:  # Less than 10% improvement
            recommendations.append("Review cache layer allocation and promotion thresholds")
        
        if not target_validation.get('hit_rate_target_met', True):
            recommendations.append("Increase cache sizes or improve eviction policies")
        
        if not target_validation.get('response_time_target_met', True):
            recommendations.append("Optimize cache persistence and loading mechanisms")
        
        if not target_validation.get('memory_usage_target_met', True):
            recommendations.append("Enable more aggressive memory pressure handling")
        
        if not recommendations:
            recommendations.append("Phase 4 performance targets achieved successfully")
        
        return recommendations


class Phase4CacheSystem:
    """
    Complete Phase 4 cache system integration.
    
    Provides unified interface for all Phase 4 components:
    - Hierarchical caching (L1/L2/L3)
    - Adaptive cache management
    - Performance monitoring
    - Intelligent prepopulation
    - Optimized persistence
    """
    
    def __init__(self, config: Optional[Phase4Config] = None):
        self.config = config or Phase4Config()
        
        # Core components
        self.hierarchical_cache: Optional[HierarchicalCache] = None
        self.adaptive_manager: Optional[AdaptiveCacheManager] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.prepopulator: Optional[IntelligentPrepopulator] = None
        self.persistence_manager: Optional[OptimizedCachePersistenceManager] = None
        
        # Validation suite
        self.validation_suite = Phase4ValidationSuite(self)
        
        # State tracking
        self._initialized = False
        self._started = False
        
        logger.info(f"Phase 4 cache system initialized ({self.config.optimization_level} mode)")
    
    def initialize(self) -> bool:
        """Initialize all Phase 4 components."""
        if self._initialized:
            logger.warning("Phase 4 cache system already initialized")
            return True
        
        try:
            logger.info("Initializing Phase 4 cache system components")
            
            # Initialize hierarchical cache
            self.hierarchical_cache = HierarchicalCache(self.config.hierarchical_config)
            
            # Initialize adaptive manager
            if self.config.enable_adaptive_management:
                from .adaptive_cache_manager import create_adaptive_cache_manager
                self.adaptive_manager = create_adaptive_cache_manager(
                    self.hierarchical_cache,
                    self.config.adaptive_config.optimization_strategy,
                    auto_start=False  # Will start manually
                )
            
            # Initialize performance monitor
            if self.config.enable_performance_monitoring:
                from .cache_performance_monitor import create_performance_monitor
                self.performance_monitor = create_performance_monitor(
                    self.hierarchical_cache,
                    self.adaptive_manager,
                    auto_start=False  # Will start manually
                )
            
            # Initialize intelligent prepopulation
            if self.config.enable_intelligent_prepopulation:
                from .intelligent_prepopulation import create_intelligent_prepopulator
                self.prepopulator = create_intelligent_prepopulator(
                    strategy_name=self.config.optimization_level
                )
            
            # Initialize optimized persistence
            if self.config.enable_optimized_persistence:
                from .optimized_persistence import create_optimized_persistence
                self.persistence_manager = create_optimized_persistence(
                    self.config.persistence_config
                )
            
            self._initialized = True
            logger.info("Phase 4 cache system components initialized successfully")
            
            # Auto-start services if configured
            if self.config.auto_start_services:
                self.start_services()
            
            return True
            
        except Exception as e:
            logger.error(f"Phase 4 cache system initialization failed: {e}")
            return False
    
    def start_services(self) -> bool:
        """Start all Phase 4 services."""
        if not self._initialized:
            logger.error("Cannot start services - system not initialized")
            return False
        
        if self._started:
            logger.warning("Phase 4 services already started")
            return True
        
        try:
            logger.info("Starting Phase 4 cache services")
            
            # Start adaptive management
            if self.adaptive_manager:
                self.adaptive_manager.start_adaptive_optimization()
            
            # Start performance monitoring
            if self.performance_monitor:
                self.performance_monitor.start_monitoring()
            
            # Load persisted cache data
            if self.persistence_manager and self.hierarchical_cache:
                self.persistence_manager.load_hierarchical_cache(self.hierarchical_cache)
            
            self._started = True
            logger.info("Phase 4 cache services started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start Phase 4 services: {e}")
            return False
    
    def stop_services(self) -> bool:
        """Stop all Phase 4 services."""
        if not self._started:
            return True
        
        try:
            logger.info("Stopping Phase 4 cache services")
            
            # Save cache data before stopping
            if self.persistence_manager and self.hierarchical_cache:
                self.persistence_manager.save_hierarchical_cache(self.hierarchical_cache)
            
            # Stop adaptive management
            if self.adaptive_manager:
                self.adaptive_manager.stop_adaptive_optimization()
            
            # Stop performance monitoring
            if self.performance_monitor:
                self.performance_monitor.stop_monitoring()
            
            # Shutdown hierarchical cache
            if self.hierarchical_cache:
                self.hierarchical_cache.shutdown()
            
            # Shutdown persistence manager
            if self.persistence_manager:
                self.persistence_manager.shutdown()
            
            self._started = False
            logger.info("Phase 4 cache services stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop Phase 4 services: {e}")
            return False
    
    @contextmanager
    def performance_validation(self, baseline_duration: int = 60, validation_duration: int = 300):
        """Context manager for performance validation."""
        # Establish baseline
        self.validation_suite.establish_baseline(baseline_duration)
        
        try:
            yield self
        finally:
            # Validate improvements
            validation_report = self.validation_suite.validate_performance_improvements(validation_duration)
            
            # Log summary
            if validation_report['overall_success']:
                logger.info("✅ Phase 4 performance validation PASSED")
            else:
                logger.warning("⚠️ Phase 4 performance validation needs attention")
                
                for recommendation in validation_report['recommendations']:
                    logger.info(f"💡 Recommendation: {recommendation}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            'initialized': self._initialized,
            'started': self._started,
            'optimization_level': self.config.optimization_level,
            'components': {
                'hierarchical_cache': self.hierarchical_cache is not None,
                'adaptive_manager': self.adaptive_manager is not None,
                'performance_monitor': self.performance_monitor is not None,
                'prepopulator': self.prepopulator is not None,
                'persistence_manager': self.persistence_manager is not None
            }
        }
        
        # Add component status if available
        if self.hierarchical_cache:
            status['cache_stats'] = asdict(self.hierarchical_cache.get_stats())
        
        if self.adaptive_manager:
            status['adaptive_status'] = self.adaptive_manager.get_optimization_status()
        
        if self.performance_monitor:
            status['performance_dashboard'] = self.performance_monitor.get_performance_dashboard()
        
        if self.persistence_manager:
            status['persistence_status'] = self.persistence_manager.get_persistence_status()
        
        return status
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        if not self._started:
            self.start_services()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_services()


# Factory functions for different optimization levels
def create_conservative_cache_system() -> Phase4CacheSystem:
    """Create Phase 4 cache system with conservative optimization."""
    config = Phase4Config(optimization_level="conservative")
    return Phase4CacheSystem(config)


def create_balanced_cache_system() -> Phase4CacheSystem:
    """Create Phase 4 cache system with balanced optimization."""
    config = Phase4Config(optimization_level="balanced")
    return Phase4CacheSystem(config)


def create_aggressive_cache_system() -> Phase4CacheSystem:
    """Create Phase 4 cache system with aggressive optimization."""
    config = Phase4Config(optimization_level="aggressive")
    return Phase4CacheSystem(config)


# Example usage and validation
def validate_phase4_performance() -> Dict[str, Any]:
    """
    Complete Phase 4 performance validation example.
    
    This function demonstrates how to use the Phase 4 system
    and validate its performance improvements.
    """
    logger.info("Starting Phase 4 performance validation")
    
    # Create balanced cache system
    cache_system = create_balanced_cache_system()
    
    try:
        # Use performance validation context manager
        with cache_system.performance_validation(baseline_duration=60, validation_duration=300):
            # Simulate cache usage during validation
            logger.info("Phase 4 cache system ready for validation")
            
            # The validation will happen automatically when exiting the context
            
        # Get final system status
        final_status = cache_system.get_system_status()
        
        logger.info("Phase 4 performance validation completed")
        return final_status
        
    except Exception as e:
        logger.error(f"Phase 4 validation failed: {e}")
        return {'error': str(e)}
    
    finally:
        # Ensure cleanup
        cache_system.stop_services()