"""
♞ Poker Knight Cache Performance Monitoring System

Comprehensive performance monitoring, alerting, and analytics for the
hierarchical cache system. Provides real-time insights, performance
dashboards, and automated alerts for cache optimization.

Author: hildolfr
License: MIT
"""

import time
import threading
import logging
import json
import os
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict, field
from collections import deque, defaultdict
from enum import Enum
import statistics
from pathlib import Path

from .hierarchical_cache import HierarchicalCache, CacheLayer
from .adaptive_cache_manager import AdaptiveCacheManager

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of performance metrics."""
    HIT_RATE = "hit_rate"
    RESPONSE_TIME = "response_time"
    MEMORY_USAGE = "memory_usage"
    THROUGHPUT = "throughput"
    EVICTION_RATE = "eviction_rate"
    ERROR_RATE = "error_rate"


@dataclass
class PerformanceAlert:
    """Performance alert definition."""
    id: str
    level: AlertLevel
    metric: MetricType
    threshold: float
    message: str
    timestamp: float
    layer: Optional[CacheLayer] = None
    current_value: Optional[float] = None
    resolved: bool = False


@dataclass
class MetricDataPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    layer: CacheLayer
    metric_type: MetricType
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceSnapshot:
    """Performance snapshot at a point in time."""
    timestamp: float
    overall_hit_rate: float
    overall_response_time_ms: float
    total_memory_usage_mb: float
    total_requests: int
    
    # Layer-specific metrics
    l1_hit_rate: float = 0.0
    l1_response_time_ms: float = 0.0
    l1_memory_usage_mb: float = 0.0
    
    l2_hit_rate: float = 0.0
    l2_response_time_ms: float = 0.0
    l2_memory_usage_mb: float = 0.0
    
    l3_hit_rate: float = 0.0
    l3_response_time_ms: float = 0.0
    l3_memory_usage_mb: float = 0.0
    
    # Performance indicators
    cache_efficiency: float = 0.0
    memory_efficiency: float = 0.0
    response_time_trend: float = 0.0
    hit_rate_trend: float = 0.0


@dataclass
class MonitoringConfig:
    """Configuration for performance monitoring."""
    # Core settings
    enabled: bool = True
    collection_interval_seconds: int = 30
    snapshot_retention_hours: int = 24
    metrics_retention_hours: int = 168  # 1 week
    
    # Alert thresholds
    min_hit_rate_warning: float = 0.7
    min_hit_rate_critical: float = 0.5
    max_response_time_warning_ms: float = 50.0
    max_response_time_critical_ms: float = 100.0
    max_memory_usage_warning: float = 0.85
    max_memory_usage_critical: float = 0.95
    
    # Performance analysis
    trend_analysis_window: int = 20  # Number of samples
    performance_baseline_hours: int = 24  # Hours to establish baseline
    
    # Export settings
    export_enabled: bool = True
    export_interval_seconds: int = 300  # 5 minutes
    export_directory: str = "cache_metrics"
    
    # Dashboard settings
    dashboard_enabled: bool = True
    dashboard_refresh_seconds: int = 15


class PerformanceAnalyzer:
    """Analyzes cache performance patterns and trends."""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self._metrics_buffer = {
            layer: {
                metric: deque(maxlen=window_size)
                for metric in MetricType
            }
            for layer in CacheLayer
        }
        self._lock = threading.RLock()
    
    def add_metric(self, layer: CacheLayer, metric_type: MetricType, value: float):
        """Add metric data point for analysis."""
        with self._lock:
            timestamp = time.time()
            data_point = MetricDataPoint(timestamp, value, layer, metric_type)
            self._metrics_buffer[layer][metric_type].append(data_point)
    
    def calculate_trend(self, layer: CacheLayer, metric_type: MetricType) -> float:
        """Calculate trend for specific metric (positive = improving)."""
        with self._lock:
            data = self._metrics_buffer[layer][metric_type]
            if len(data) < 3:
                return 0.0
            
            try:
                values = [point.value for point in data]
                timestamps = [point.timestamp for point in data]
                
                # Linear regression for trend
                n = len(values)
                sum_x = sum(timestamps)
                sum_y = sum(values)
                sum_xy = sum(t * v for t, v in zip(timestamps, values))
                sum_x2 = sum(t * t for t in timestamps)
                
                if n * sum_x2 - sum_x * sum_x == 0:
                    return 0.0
                
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                
                # Normalize slope based on metric type
                if metric_type in [MetricType.HIT_RATE, MetricType.THROUGHPUT]:
                    return slope  # Positive trend is good
                elif metric_type in [MetricType.RESPONSE_TIME, MetricType.MEMORY_USAGE, MetricType.EVICTION_RATE]:
                    return -slope  # Negative trend is good (decreasing is better)
                else:
                    return slope
                    
            except Exception as e:
                logger.warning(f"Trend calculation failed for {layer.value}.{metric_type.value}: {e}")
                return 0.0
    
    def calculate_volatility(self, layer: CacheLayer, metric_type: MetricType) -> float:
        """Calculate volatility (standard deviation) for metric."""
        with self._lock:
            data = self._metrics_buffer[layer][metric_type]
            if len(data) < 2:
                return 0.0
            
            values = [point.value for point in data]
            return statistics.stdev(values) if len(values) > 1 else 0.0
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance analysis summary."""
        summary = {
            'trends': {},
            'volatility': {},
            'overall_health': 'unknown'
        }
        
        try:
            trend_scores = []
            volatility_scores = []
            
            for layer in CacheLayer:
                layer_name = layer.value
                summary['trends'][layer_name] = {}
                summary['volatility'][layer_name] = {}
                
                for metric_type in MetricType:
                    metric_name = metric_type.value
                    trend = self.calculate_trend(layer, metric_type)
                    volatility = self.calculate_volatility(layer, metric_type)
                    
                    summary['trends'][layer_name][metric_name] = trend
                    summary['volatility'][layer_name][metric_name] = volatility
                    
                    # Weight important metrics for overall health
                    if metric_type in [MetricType.HIT_RATE, MetricType.RESPONSE_TIME]:
                        trend_scores.append(trend)
                        volatility_scores.append(volatility)
            
            # Calculate overall health score
            if trend_scores and volatility_scores:
                avg_trend = statistics.mean(trend_scores)
                avg_volatility = statistics.mean(volatility_scores)
                
                # Good health = positive trends, low volatility
                health_score = avg_trend - (avg_volatility * 0.5)
                
                if health_score > 0.1:
                    summary['overall_health'] = 'excellent'
                elif health_score > 0.0:
                    summary['overall_health'] = 'good'
                elif health_score > -0.1:
                    summary['overall_health'] = 'fair'
                else:
                    summary['overall_health'] = 'poor'
        
        except Exception as e:
            logger.error(f"Performance summary calculation failed: {e}")
        
        return summary


class AlertManager:
    """Manages performance alerts and notifications."""
    
    def __init__(self, config: MonitoringConfig):
        self.config = config
        self._active_alerts = {}
        self._alert_history = deque(maxlen=1000)
        self._alert_callbacks = []
        self._lock = threading.RLock()
    
    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """Add callback for alert notifications."""
        self._alert_callbacks.append(callback)
    
    def check_alerts(self, snapshot: PerformanceSnapshot) -> List[PerformanceAlert]:
        """Check for alert conditions and generate alerts."""
        new_alerts = []
        current_time = time.time()
        
        # Check overall hit rate
        if snapshot.overall_hit_rate < self.config.min_hit_rate_critical:
            alert = self._create_alert(
                "overall_hit_rate_critical",
                AlertLevel.CRITICAL,
                MetricType.HIT_RATE,
                snapshot.overall_hit_rate,
                f"Critical: Overall hit rate {snapshot.overall_hit_rate:.1%} below {self.config.min_hit_rate_critical:.1%}"
            )
            new_alerts.append(alert)
        elif snapshot.overall_hit_rate < self.config.min_hit_rate_warning:
            alert = self._create_alert(
                "overall_hit_rate_warning",
                AlertLevel.WARNING,
                MetricType.HIT_RATE,
                snapshot.overall_hit_rate,
                f"Warning: Overall hit rate {snapshot.overall_hit_rate:.1%} below {self.config.min_hit_rate_warning:.1%}"
            )
            new_alerts.append(alert)
        
        # Check response time
        if snapshot.overall_response_time_ms > self.config.max_response_time_critical_ms:
            alert = self._create_alert(
                "response_time_critical",
                AlertLevel.CRITICAL,
                MetricType.RESPONSE_TIME,
                snapshot.overall_response_time_ms,
                f"Critical: Response time {snapshot.overall_response_time_ms:.1f}ms above {self.config.max_response_time_critical_ms:.1f}ms"
            )
            new_alerts.append(alert)
        elif snapshot.overall_response_time_ms > self.config.max_response_time_warning_ms:
            alert = self._create_alert(
                "response_time_warning",
                AlertLevel.WARNING,
                MetricType.RESPONSE_TIME,
                snapshot.overall_response_time_ms,
                f"Warning: Response time {snapshot.overall_response_time_ms:.1f}ms above {self.config.max_response_time_warning_ms:.1f}ms"
            )
            new_alerts.append(alert)
        
        # Check memory usage
        memory_usage_ratio = snapshot.total_memory_usage_mb / 1024  # Assuming 1GB total for ratio
        if memory_usage_ratio > self.config.max_memory_usage_critical:
            alert = self._create_alert(
                "memory_usage_critical",
                AlertLevel.CRITICAL,
                MetricType.MEMORY_USAGE,
                memory_usage_ratio,
                f"Critical: Memory usage {memory_usage_ratio:.1%} above {self.config.max_memory_usage_critical:.1%}"
            )
            new_alerts.append(alert)
        
        # Process new alerts
        for alert in new_alerts:
            self._process_alert(alert)
        
        return new_alerts
    
    def _create_alert(self, alert_id: str, level: AlertLevel, metric: MetricType, 
                     current_value: float, message: str) -> PerformanceAlert:
        """Create new performance alert."""
        return PerformanceAlert(
            id=alert_id,
            level=level,
            metric=metric,
            threshold=0.0,  # Will be set based on config
            message=message,
            timestamp=time.time(),
            current_value=current_value
        )
    
    def _process_alert(self, alert: PerformanceAlert):
        """Process and store alert."""
        with self._lock:
            # Check if this is a duplicate of active alert
            if alert.id in self._active_alerts:
                existing = self._active_alerts[alert.id]
                existing.current_value = alert.current_value
                existing.timestamp = alert.timestamp
            else:
                # New alert
                self._active_alerts[alert.id] = alert
                self._alert_history.append(alert)
                
                # Notify callbacks
                for callback in self._alert_callbacks:
                    try:
                        callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback failed: {e}")
                
                logger.log(
                    logging.CRITICAL if alert.level == AlertLevel.CRITICAL else logging.WARNING,
                    f"Cache Alert [{alert.level.value.upper()}]: {alert.message}"
                )
    
    def resolve_alert(self, alert_id: str):
        """Mark alert as resolved."""
        with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id].resolved = True
                del self._active_alerts[alert_id]
    
    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get all active alerts."""
        with self._lock:
            return list(self._active_alerts.values())


class PerformanceMonitor:
    """
    Comprehensive cache performance monitoring system.
    
    Collects metrics, analyzes performance, generates alerts, and provides
    insights for cache optimization and troubleshooting.
    """
    
    def __init__(self, 
                 cache: HierarchicalCache,
                 config: Optional[MonitoringConfig] = None,
                 adaptive_manager: Optional[AdaptiveCacheManager] = None):
        
        self.cache = cache
        self.config = config or MonitoringConfig()
        self.adaptive_manager = adaptive_manager
        
        # Components
        self.analyzer = PerformanceAnalyzer(self.config.trend_analysis_window)
        self.alert_manager = AlertManager(self.config)
        
        # Data storage
        self._snapshots = deque(maxlen=self._calculate_snapshot_retention())
        self._metrics_export_buffer = []
        
        # State tracking
        self._is_monitoring = False
        self._monitoring_thread = None
        self._stop_event = threading.Event()
        self._last_export = 0
        
        # Performance baseline
        self._baseline_snapshots = deque(maxlen=self._calculate_baseline_retention())
        self._baseline_established = False
        
        # Setup export directory
        if self.config.export_enabled:
            self._setup_export_directory()
        
        logger.info("Performance monitoring system initialized")
    
    def _calculate_snapshot_retention(self) -> int:
        """Calculate number of snapshots to retain."""
        snapshots_per_hour = 3600 / self.config.collection_interval_seconds
        return int(snapshots_per_hour * self.config.snapshot_retention_hours)
    
    def _calculate_baseline_retention(self) -> int:
        """Calculate number of snapshots for baseline."""
        snapshots_per_hour = 3600 / self.config.collection_interval_seconds
        return int(snapshots_per_hour * self.config.performance_baseline_hours)
    
    def _setup_export_directory(self):
        """Setup directory for metrics export."""
        try:
            export_path = Path(self.config.export_directory)
            export_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create export directory: {e}")
            self.config.export_enabled = False
    
    def start_monitoring(self):
        """Start performance monitoring."""
        if self._is_monitoring:
            logger.warning("Performance monitoring already running")
            return
        
        self._is_monitoring = True
        self._stop_event.clear()
        
        def monitoring_worker():
            logger.info("Starting cache performance monitoring")
            
            while not self._stop_event.is_set():
                try:
                    # Collect performance snapshot
                    snapshot = self._collect_performance_snapshot()
                    self._process_snapshot(snapshot)
                    
                    # Export metrics if needed
                    if (self.config.export_enabled and 
                        time.time() - self._last_export >= self.config.export_interval_seconds):
                        self._export_metrics()
                        self._last_export = time.time()
                    
                    # Wait for next collection
                    self._stop_event.wait(self.config.collection_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Monitoring cycle failed: {e}")
                    self._stop_event.wait(60)  # Wait 1 minute on error
            
            logger.info("Cache performance monitoring stopped")
        
        self._monitoring_thread = threading.Thread(target=monitoring_worker, daemon=True)
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        if not self._is_monitoring:
            return
        
        logger.info("Stopping cache performance monitoring")
        self._stop_event.set()
        
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=10)
        
        self._is_monitoring = False
    
    def _collect_performance_snapshot(self) -> PerformanceSnapshot:
        """Collect current performance snapshot."""
        stats = self.cache.get_stats()
        current_time = time.time()
        
        # Calculate overall metrics
        total_memory = (stats.l1_stats.memory_usage_mb + 
                       stats.l2_stats.memory_usage_mb + 
                       stats.l3_stats.memory_usage_mb)
        
        # Calculate weighted average response time
        total_hits = stats.l1_hits + stats.l2_hits + stats.l3_hits
        if total_hits > 0:
            overall_response_time = (
                (stats.l1_hits * stats.avg_l1_response_time_ms +
                 stats.l2_hits * stats.avg_l2_response_time_ms +
                 stats.l3_hits * stats.avg_l3_response_time_ms) / total_hits
            )
        else:
            overall_response_time = 0.0
        
        snapshot = PerformanceSnapshot(
            timestamp=current_time,
            overall_hit_rate=stats.overall_hit_rate,
            overall_response_time_ms=overall_response_time,
            total_memory_usage_mb=total_memory,
            total_requests=stats.total_requests,
            
            l1_hit_rate=stats.l1_hit_rate,
            l1_response_time_ms=stats.avg_l1_response_time_ms,
            l1_memory_usage_mb=stats.l1_stats.memory_usage_mb,
            
            l2_hit_rate=stats.l2_hits / stats.total_requests if stats.total_requests > 0 else 0.0,
            l2_response_time_ms=stats.avg_l2_response_time_ms,
            l2_memory_usage_mb=stats.l2_stats.memory_usage_mb,
            
            l3_hit_rate=stats.l3_hits / stats.total_requests if stats.total_requests > 0 else 0.0,
            l3_response_time_ms=stats.avg_l3_response_time_ms,
            l3_memory_usage_mb=stats.l3_stats.memory_usage_mb
        )
        
        # Calculate efficiency metrics
        snapshot.cache_efficiency = self._calculate_cache_efficiency(snapshot)
        snapshot.memory_efficiency = self._calculate_memory_efficiency(snapshot)
        
        # Calculate trends
        snapshot.hit_rate_trend = self.analyzer.calculate_trend(CacheLayer.L1_MEMORY, MetricType.HIT_RATE)
        snapshot.response_time_trend = self.analyzer.calculate_trend(CacheLayer.L1_MEMORY, MetricType.RESPONSE_TIME)
        
        return snapshot
    
    def _calculate_cache_efficiency(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate overall cache efficiency score."""
        # Weighted combination of hit rate and response time
        hit_rate_score = snapshot.overall_hit_rate * 0.7
        response_time_score = max(0, 1.0 - (snapshot.overall_response_time_ms / 100.0)) * 0.3
        return hit_rate_score + response_time_score
    
    def _calculate_memory_efficiency(self, snapshot: PerformanceSnapshot) -> float:
        """Calculate memory efficiency score."""
        if snapshot.total_memory_usage_mb == 0:
            return 0.0
        
        # Efficiency = performance per MB of memory used
        performance_score = snapshot.overall_hit_rate
        memory_ratio = min(1.0, snapshot.total_memory_usage_mb / 1024)  # Normalize to 1GB
        
        return performance_score / (memory_ratio + 0.1)  # Avoid division by zero
    
    def _process_snapshot(self, snapshot: PerformanceSnapshot):
        """Process performance snapshot."""
        # Store snapshot
        self._snapshots.append(snapshot)
        
        # Update analyzer with new metrics
        self.analyzer.add_metric(CacheLayer.L1_MEMORY, MetricType.HIT_RATE, snapshot.l1_hit_rate)
        self.analyzer.add_metric(CacheLayer.L1_MEMORY, MetricType.RESPONSE_TIME, snapshot.l1_response_time_ms)
        self.analyzer.add_metric(CacheLayer.L1_MEMORY, MetricType.MEMORY_USAGE, snapshot.l1_memory_usage_mb)
        
        # Similar for L2 and L3...
        
        # Check for alerts
        alerts = self.alert_manager.check_alerts(snapshot)
        
        # Build baseline if not established
        if not self._baseline_established:
            self._baseline_snapshots.append(snapshot)
            if len(self._baseline_snapshots) >= self._calculate_baseline_retention():
                self._baseline_established = True
                logger.info("Performance baseline established")
        
        # Add to export buffer
        if self.config.export_enabled:
            self._metrics_export_buffer.append({
                'timestamp': snapshot.timestamp,
                'hit_rate': snapshot.overall_hit_rate,
                'response_time_ms': snapshot.overall_response_time_ms,
                'memory_usage_mb': snapshot.total_memory_usage_mb,
                'cache_efficiency': snapshot.cache_efficiency,
                'memory_efficiency': snapshot.memory_efficiency
            })
    
    def _export_metrics(self):
        """Export metrics to file."""
        if not self._metrics_export_buffer:
            return
        
        try:
            timestamp = int(time.time())
            filename = f"cache_metrics_{timestamp}.json"
            filepath = Path(self.config.export_directory) / filename
            
            export_data = {
                'export_timestamp': timestamp,
                'collection_interval_seconds': self.config.collection_interval_seconds,
                'metrics': self._metrics_export_buffer.copy()
            }
            
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            logger.debug(f"Exported {len(self._metrics_export_buffer)} metrics to {filename}")
            self._metrics_export_buffer.clear()
            
        except Exception as e:
            logger.error(f"Metrics export failed: {e}")
    
    def get_performance_dashboard(self) -> Dict[str, Any]:
        """Get performance dashboard data."""
        if not self._snapshots:
            return {'error': 'No performance data available'}
        
        latest = self._snapshots[-1]
        
        # Recent performance (last hour)
        hour_ago = time.time() - 3600
        recent_snapshots = [s for s in self._snapshots if s.timestamp > hour_ago]
        
        dashboard = {
            'timestamp': latest.timestamp,
            'status': 'monitoring' if self._is_monitoring else 'stopped',
            
            # Current metrics
            'current': {
                'hit_rate': latest.overall_hit_rate,
                'response_time_ms': latest.overall_response_time_ms,
                'memory_usage_mb': latest.total_memory_usage_mb,
                'cache_efficiency': latest.cache_efficiency,
                'memory_efficiency': latest.memory_efficiency,
                'total_requests': latest.total_requests
            },
            
            # Layer breakdown
            'layers': {
                'l1': {
                    'hit_rate': latest.l1_hit_rate,
                    'response_time_ms': latest.l1_response_time_ms,
                    'memory_usage_mb': latest.l1_memory_usage_mb
                },
                'l2': {
                    'hit_rate': latest.l2_hit_rate,
                    'response_time_ms': latest.l2_response_time_ms,
                    'memory_usage_mb': latest.l2_memory_usage_mb
                },
                'l3': {
                    'hit_rate': latest.l3_hit_rate,
                    'response_time_ms': latest.l3_response_time_ms,
                    'memory_usage_mb': latest.l3_memory_usage_mb
                }
            },
            
            # Trends
            'trends': {
                'hit_rate_trend': latest.hit_rate_trend,
                'response_time_trend': latest.response_time_trend
            },
            
            # Recent performance
            'recent_hour': {
                'avg_hit_rate': statistics.mean([s.overall_hit_rate for s in recent_snapshots]) if recent_snapshots else 0,
                'avg_response_time_ms': statistics.mean([s.overall_response_time_ms for s in recent_snapshots]) if recent_snapshots else 0,
                'peak_memory_mb': max([s.total_memory_usage_mb for s in recent_snapshots]) if recent_snapshots else 0
            },
            
            # Alerts
            'active_alerts': [asdict(alert) for alert in self.alert_manager.get_active_alerts()],
            
            # Performance analysis
            'analysis': self.analyzer.get_performance_summary(),
            
            # Adaptive management
            'adaptive_management': (
                self.adaptive_manager.get_optimization_status() 
                if self.adaptive_manager else None
            )
        }
        
        return dashboard
    
    def get_historical_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get historical performance data."""
        cutoff_time = time.time() - (hours * 3600)
        return [
            {
                'timestamp': s.timestamp,
                'hit_rate': s.overall_hit_rate,
                'response_time_ms': s.overall_response_time_ms,
                'memory_usage_mb': s.total_memory_usage_mb,
                'cache_efficiency': s.cache_efficiency
            }
            for s in self._snapshots
            if s.timestamp > cutoff_time
        ]


# Factory function for easy integration
def create_performance_monitor(cache: HierarchicalCache,
                             adaptive_manager: Optional[AdaptiveCacheManager] = None,
                             auto_start: bool = True) -> PerformanceMonitor:
    """Create and optionally start performance monitor."""
    monitor = PerformanceMonitor(cache, adaptive_manager=adaptive_manager)
    
    if auto_start:
        monitor.start_monitoring()
    
    return monitor