"""
♞ Poker Knight Adaptive Cache Size Management

Dynamic cache size optimization based on usage patterns, hit rates,
and system resources. Automatically adjusts cache layer sizes for
optimal performance and memory utilization.

Author: hildolfr
License: MIT
"""

import time
import threading
import logging
import psutil
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import statistics
from enum import Enum

from .hierarchical_cache import HierarchicalCache, CacheLayer, HierarchicalCacheConfig

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Cache size optimization strategies."""
    CONSERVATIVE = "conservative"  # Small adjustments, stable performance
    AGGRESSIVE = "aggressive"     # Large adjustments, maximum performance
    BALANCED = "balanced"         # Moderate adjustments, good balance
    MEMORY_AWARE = "memory_aware" # Optimize based on available memory


@dataclass
class CacheSizeMetrics:
    """Metrics for cache size optimization."""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    memory_utilization: float = 0.0
    eviction_rate: float = 0.0
    response_time_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    
    # Performance trends
    hit_rate_trend: float = 0.0      # Positive = improving
    memory_trend: float = 0.0        # Positive = increasing usage
    response_time_trend: float = 0.0 # Positive = slower
    
    # Efficiency metrics
    efficiency_score: float = 0.0    # Combined efficiency metric
    size_recommendation: str = "maintain"  # "increase", "decrease", "maintain"


@dataclass
class AdaptiveCacheConfig:
    """Configuration for adaptive cache management."""
    # Core settings
    enabled: bool = True
    optimization_strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    adjustment_interval_seconds: int = 300  # 5 minutes
    
    # Size adjustment limits
    min_size_mb: int = 32
    max_size_mb: int = 1024
    adjustment_step_percent: float = 0.1  # 10% adjustments
    max_adjustment_percent: float = 0.5   # 50% max change per adjustment
    
    # Performance thresholds
    target_hit_rate: float = 0.85
    min_hit_rate: float = 0.70
    max_memory_utilization: float = 0.90
    target_response_time_ms: float = 5.0
    
    # Trend analysis
    trend_window_size: int = 10  # Number of samples for trend analysis
    significance_threshold: float = 0.05  # Minimum change to be significant
    
    # System constraints
    respect_system_memory: bool = True
    max_system_memory_percent: float = 0.25  # Use at most 25% of system memory


@dataclass
class LayerOptimizationResult:
    """Result of cache layer optimization."""
    layer: CacheLayer
    old_size_mb: int
    new_size_mb: int
    adjustment_reason: str
    expected_improvement: float
    confidence: float


class CacheMetricsCollector:
    """Collects and analyzes cache performance metrics."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self._metrics_history = {
            CacheLayer.L1_MEMORY: deque(maxlen=window_size),
            CacheLayer.L2_REDIS: deque(maxlen=window_size),
            CacheLayer.L3_SQLITE: deque(maxlen=window_size)
        }
        self._lock = threading.RLock()
    
    def collect_metrics(self, cache: HierarchicalCache) -> Dict[CacheLayer, CacheSizeMetrics]:
        """Collect current metrics from cache layers."""
        stats = cache.get_stats()
        current_time = time.time()
        
        metrics = {}
        
        # L1 Metrics
        l1_metrics = CacheSizeMetrics(
            hit_rate=stats.l1_hit_rate,
            miss_rate=1.0 - stats.l1_hit_rate,
            memory_utilization=stats.l1_stats.memory_usage_mb / cache.config.l1_config.max_memory_mb,
            eviction_rate=self._calculate_eviction_rate(stats.l1_stats),
            response_time_ms=stats.avg_l1_response_time_ms,
            throughput_ops_per_sec=self._calculate_throughput(stats.l1_stats)
        )
        metrics[CacheLayer.L1_MEMORY] = l1_metrics
        
        # L2 Metrics
        l2_hit_rate = stats.l2_hits / stats.total_requests if stats.total_requests > 0 else 0.0
        l2_metrics = CacheSizeMetrics(
            hit_rate=l2_hit_rate,
            miss_rate=1.0 - l2_hit_rate,
            memory_utilization=stats.l2_stats.memory_usage_mb / cache.config.l2_config.max_memory_mb,
            eviction_rate=self._calculate_eviction_rate(stats.l2_stats),
            response_time_ms=stats.avg_l2_response_time_ms,
            throughput_ops_per_sec=self._calculate_throughput(stats.l2_stats)
        )
        metrics[CacheLayer.L2_REDIS] = l2_metrics
        
        # L3 Metrics
        l3_hit_rate = stats.l3_hits / stats.total_requests if stats.total_requests > 0 else 0.0
        l3_metrics = CacheSizeMetrics(
            hit_rate=l3_hit_rate,
            miss_rate=1.0 - l3_hit_rate,
            memory_utilization=stats.l3_stats.memory_usage_mb / cache.config.l3_config.max_memory_mb,
            eviction_rate=self._calculate_eviction_rate(stats.l3_stats),
            response_time_ms=stats.avg_l3_response_time_ms,
            throughput_ops_per_sec=self._calculate_throughput(stats.l3_stats)
        )
        metrics[CacheLayer.L3_SQLITE] = l3_metrics
        
        # Store metrics history and calculate trends
        with self._lock:
            for layer, layer_metrics in metrics.items():
                self._metrics_history[layer].append((current_time, layer_metrics))
                layer_metrics.hit_rate_trend = self._calculate_trend(layer, 'hit_rate')
                layer_metrics.memory_trend = self._calculate_trend(layer, 'memory_utilization')
                layer_metrics.response_time_trend = self._calculate_trend(layer, 'response_time_ms')
                layer_metrics.efficiency_score = self._calculate_efficiency_score(layer_metrics)
        
        return metrics
    
    def _calculate_eviction_rate(self, layer_stats) -> float:
        """Calculate eviction rate for cache layer."""
        if hasattr(layer_stats, 'evictions') and layer_stats.total_requests > 0:
            return layer_stats.evictions / layer_stats.total_requests
        return 0.0
    
    def _calculate_throughput(self, layer_stats) -> float:
        """Calculate throughput for cache layer."""
        if hasattr(layer_stats, 'total_requests'):
            # Estimate based on recent activity
            return layer_stats.total_requests / 60.0  # Ops per second (rough estimate)
        return 0.0
    
    def _calculate_trend(self, layer: CacheLayer, metric_name: str) -> float:
        """Calculate trend for specific metric using linear regression."""
        history = self._metrics_history[layer]
        if len(history) < 3:
            return 0.0
        
        try:
            values = []
            times = []
            
            for timestamp, metrics in history:
                if hasattr(metrics, metric_name):
                    values.append(getattr(metrics, metric_name))
                    times.append(timestamp)
            
            if len(values) < 3:
                return 0.0
            
            # Simple linear regression slope
            n = len(values)
            sum_x = sum(times)
            sum_y = sum(values)
            sum_xy = sum(t * v for t, v in zip(times, values))
            sum_x2 = sum(t * t for t in times)
            
            if n * sum_x2 - sum_x * sum_x == 0:
                return 0.0
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            return slope
            
        except Exception as e:
            logger.warning(f"Trend calculation failed for {layer.value}.{metric_name}: {e}")
            return 0.0
    
    def _calculate_efficiency_score(self, metrics: CacheSizeMetrics) -> float:
        """Calculate overall efficiency score for cache layer."""
        # Weighted combination of key metrics
        hit_rate_score = metrics.hit_rate * 0.4
        memory_efficiency = (1.0 - metrics.memory_utilization) * 0.3  # Less memory = better
        response_time_score = max(0, 1.0 - (metrics.response_time_ms / 100.0)) * 0.3  # Under 100ms ideal
        
        return hit_rate_score + memory_efficiency + response_time_score


class AdaptiveCacheManager:
    """
    Adaptive cache size management system.
    
    Continuously monitors cache performance and automatically adjusts
    cache sizes to optimize for hit rates, memory usage, and response times.
    """
    
    def __init__(self, 
                 cache: HierarchicalCache,
                 config: Optional[AdaptiveCacheConfig] = None):
        
        self.cache = cache
        self.config = config or AdaptiveCacheConfig()
        self.metrics_collector = CacheMetricsCollector(self.config.trend_window_size)
        
        # State tracking
        self._last_optimization = 0
        self._optimization_history = []
        self._is_running = False
        self._optimization_thread = None
        self._stop_event = threading.Event()
        
        # System monitoring
        self._system_memory_gb = psutil.virtual_memory().total / (1024**3)
        
        logger.info(f"Adaptive cache manager initialized with {self.config.optimization_strategy.value} strategy")
    
    def start_adaptive_optimization(self):
        """Start adaptive optimization in background thread."""
        if self._is_running:
            logger.warning("Adaptive optimization already running")
            return
        
        self._is_running = True
        self._stop_event.clear()
        
        def optimization_worker():
            logger.info("Starting adaptive cache optimization")
            
            while not self._stop_event.is_set():
                try:
                    if time.time() - self._last_optimization >= self.config.adjustment_interval_seconds:
                        self._perform_optimization_cycle()
                        self._last_optimization = time.time()
                    
                    # Wait for next cycle or stop signal
                    self._stop_event.wait(min(60, self.config.adjustment_interval_seconds))
                    
                except Exception as e:
                    logger.error(f"Optimization cycle failed: {e}")
                    self._stop_event.wait(60)  # Wait 1 minute on error
            
            logger.info("Adaptive cache optimization stopped")
        
        self._optimization_thread = threading.Thread(target=optimization_worker, daemon=True)
        self._optimization_thread.start()
    
    def stop_adaptive_optimization(self):
        """Stop adaptive optimization."""
        if not self._is_running:
            return
        
        logger.info("Stopping adaptive cache optimization")
        self._stop_event.set()
        
        if self._optimization_thread:
            self._optimization_thread.join(timeout=10)
        
        self._is_running = False
    
    def _perform_optimization_cycle(self):
        """Perform one optimization cycle."""
        logger.debug("Performing cache optimization cycle")
        
        # Collect current metrics
        current_metrics = self.metrics_collector.collect_metrics(self.cache)
        
        # Analyze performance and recommend adjustments
        optimizations = []
        
        for layer, metrics in current_metrics.items():
            result = self._analyze_layer_performance(layer, metrics)
            if result:
                optimizations.append(result)
        
        # Apply optimizations
        if optimizations:
            self._apply_optimizations(optimizations)
            logger.info(f"Applied {len(optimizations)} cache optimizations")
        else:
            logger.debug("No optimizations needed this cycle")
    
    def _analyze_layer_performance(self, 
                                 layer: CacheLayer, 
                                 metrics: CacheSizeMetrics) -> Optional[LayerOptimizationResult]:
        """Analyze single layer performance and recommend size adjustment."""
        
        current_config = self._get_layer_config(layer)
        current_size_mb = current_config.max_memory_mb
        
        # Determine if adjustment is needed
        adjustment_needed = False
        adjustment_reason = ""
        size_change_factor = 1.0
        confidence = 0.0
        
        # Low hit rate analysis
        if metrics.hit_rate < self.config.min_hit_rate:
            if metrics.memory_utilization > 0.8:  # Cache is full, need more space
                adjustment_needed = True
                adjustment_reason = f"Low hit rate ({metrics.hit_rate:.2%}) with high memory utilization"
                size_change_factor = 1.0 + self.config.adjustment_step_percent
                confidence = 0.8
        
        # High eviction rate analysis
        elif metrics.eviction_rate > 0.1:  # More than 10% eviction rate
            adjustment_needed = True
            adjustment_reason = f"High eviction rate ({metrics.eviction_rate:.2%})"
            size_change_factor = 1.0 + self.config.adjustment_step_percent
            confidence = 0.7
        
        # Memory waste analysis
        elif metrics.memory_utilization < 0.3 and metrics.hit_rate > self.config.target_hit_rate:
            # Using very little memory but good hit rate - can reduce size
            adjustment_needed = True
            adjustment_reason = f"Low memory utilization ({metrics.memory_utilization:.2%}) with good hit rate"
            size_change_factor = 1.0 - (self.config.adjustment_step_percent * 0.5)  # Smaller reduction
            confidence = 0.6
        
        # Trend-based analysis
        elif metrics.hit_rate_trend < -0.01:  # Hit rate declining
            if metrics.memory_trend > 0.01:  # Memory usage increasing
                adjustment_needed = True
                adjustment_reason = "Declining hit rate with increasing memory pressure"
                size_change_factor = 1.0 + self.config.adjustment_step_percent
                confidence = 0.5
        
        if not adjustment_needed:
            return None
        
        # Calculate new size
        new_size_mb = int(current_size_mb * size_change_factor)
        
        # Apply constraints
        new_size_mb = max(self.config.min_size_mb, min(self.config.max_size_mb, new_size_mb))
        
        # Check system memory constraints
        if self.config.respect_system_memory:
            max_allowed_mb = int(self._system_memory_gb * 1024 * self.config.max_system_memory_percent)
            new_size_mb = min(new_size_mb, max_allowed_mb)
        
        # Don't make tiny adjustments
        size_diff_percent = abs(new_size_mb - current_size_mb) / current_size_mb
        if size_diff_percent < self.config.significance_threshold:
            return None
        
        # Limit maximum adjustment per cycle
        max_change = int(current_size_mb * self.config.max_adjustment_percent)
        if new_size_mb > current_size_mb:
            new_size_mb = min(new_size_mb, current_size_mb + max_change)
        else:
            new_size_mb = max(new_size_mb, current_size_mb - max_change)
        
        # Calculate expected improvement
        expected_improvement = self._estimate_improvement(metrics, size_change_factor)
        
        return LayerOptimizationResult(
            layer=layer,
            old_size_mb=current_size_mb,
            new_size_mb=new_size_mb,
            adjustment_reason=adjustment_reason,
            expected_improvement=expected_improvement,
            confidence=confidence
        )
    
    def _get_layer_config(self, layer: CacheLayer):
        """Get configuration for specific cache layer."""
        if layer == CacheLayer.L1_MEMORY:
            return self.cache.config.l1_config
        elif layer == CacheLayer.L2_REDIS:
            return self.cache.config.l2_config
        elif layer == CacheLayer.L3_SQLITE:
            return self.cache.config.l3_config
        else:
            raise ValueError(f"Unknown cache layer: {layer}")
    
    def _estimate_improvement(self, metrics: CacheSizeMetrics, size_change_factor: float) -> float:
        """Estimate performance improvement from size change."""
        if size_change_factor > 1.0:
            # Increasing size - estimate hit rate improvement
            current_miss_rate = metrics.miss_rate
            # Assume 50% of misses could be converted to hits with more space
            potential_improvement = current_miss_rate * 0.5 * (size_change_factor - 1.0)
            return min(potential_improvement, 0.2)  # Cap at 20% improvement
        else:
            # Decreasing size - estimate minimal impact if memory was underutilized
            if metrics.memory_utilization < 0.5:
                return 0.01  # Minimal impact expected
            else:
                # Some performance degradation expected
                return -(1.0 - size_change_factor) * 0.1
    
    def _apply_optimizations(self, optimizations: List[LayerOptimizationResult]):
        """Apply cache size optimizations."""
        for opt in optimizations:
            try:
                # Update cache configuration
                layer_config = self._get_layer_config(opt.layer)
                old_size = layer_config.max_memory_mb
                layer_config.max_memory_mb = opt.new_size_mb
                
                # Log the change
                change_pct = ((opt.new_size_mb - old_size) / old_size) * 100
                logger.info(f"{opt.layer.value}: {old_size}MB -> {opt.new_size_mb}MB "
                           f"({change_pct:+.1f}%) - {opt.adjustment_reason}")
                
                # Store optimization history
                self._optimization_history.append({
                    'timestamp': time.time(),
                    'layer': opt.layer.value,
                    'old_size_mb': opt.old_size_mb,
                    'new_size_mb': opt.new_size_mb,
                    'reason': opt.adjustment_reason,
                    'expected_improvement': opt.expected_improvement,
                    'confidence': opt.confidence
                })
                
                # Keep only recent history
                if len(self._optimization_history) > 100:
                    self._optimization_history = self._optimization_history[-50:]
                
            except Exception as e:
                logger.error(f"Failed to apply optimization for {opt.layer.value}: {e}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status and statistics."""
        return {
            'is_running': self._is_running,
            'strategy': self.config.optimization_strategy.value,
            'last_optimization': self._last_optimization,
            'optimization_interval': self.config.adjustment_interval_seconds,
            'total_optimizations': len(self._optimization_history),
            'recent_optimizations': self._optimization_history[-10:] if self._optimization_history else [],
            'system_memory_gb': self._system_memory_gb,
            'constraints': {
                'min_size_mb': self.config.min_size_mb,
                'max_size_mb': self.config.max_size_mb,
                'max_system_memory_percent': self.config.max_system_memory_percent,
                'target_hit_rate': self.config.target_hit_rate
            }
        }


# Factory function for easy integration
def create_adaptive_cache_manager(cache: HierarchicalCache,
                                strategy: OptimizationStrategy = OptimizationStrategy.BALANCED,
                                auto_start: bool = True) -> AdaptiveCacheManager:
    """Create and optionally start adaptive cache manager."""
    config = AdaptiveCacheConfig(optimization_strategy=strategy)
    manager = AdaptiveCacheManager(cache, config)
    
    if auto_start:
        manager.start_adaptive_optimization()
    
    return manager