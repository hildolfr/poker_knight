#!/usr/bin/env python3
"""
Poker Knight v1.5.0 - Advanced Monte Carlo Convergence Analysis

Advanced statistical analysis and convergence diagnostics for Monte Carlo simulations.
Provides real-time convergence monitoring, Geweke diagnostics, and effective sample size calculations.

Author: hildolfr
License: MIT
GitHub: https://github.com/hildolfr/poker-knight
Version: 1.5.0

Key Features:
- Geweke diagnostic for convergence detection
- Effective sample size (ESS) calculation
- Real-time convergence monitoring
- Adaptive stopping criteria
- Statistical visualization utilities
- Advanced cross-validation framework (Task 7.1)
- Split-chain R-hat diagnostics (Task 7.1)
- Batch convergence analysis (Task 7.1)

Usage:
    from poker_knight.analysis import ConvergenceMonitor
    monitor = ConvergenceMonitor()
    monitor.update(win_rate, simulation_count)
    if monitor.has_converged():
        # Stop simulation early
"""

import math
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Provide basic numpy-like functionality for core features
    class np:
        @staticmethod
        def array(data):
            return list(data)
        
        @staticmethod
        def mean(data):
            return sum(data) / len(data) if data else 0.0
        
        @staticmethod
        def var(data, ddof=0):
            if len(data) <= ddof:
                return 0.0
            mean_val = np.mean(data)
            variance = sum((x - mean_val) ** 2 for x in data) / (len(data) - ddof)
            return variance
        
        @staticmethod
        def corrcoef(x, y):
            if len(x) != len(y) or len(x) < 2:
                return [[float('nan')]]
            
            mean_x = np.mean(x)
            mean_y = np.mean(y)
            
            num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
            den_x = sum((xi - mean_x) ** 2 for xi in x)
            den_y = sum((yi - mean_y) ** 2 for yi in y)
            
            if den_x == 0 or den_y == 0:
                return [[float('nan')]]
            
            corr = num / (den_x * den_y) ** 0.5
            return [[1.0, corr], [corr, 1.0]]
        
        @staticmethod
        def zeros(size):
            if isinstance(size, int):
                return [0.0] * size
            return [0.0] * size[0]
        
        @staticmethod
        def isnan(value):
            return value != value  # NaN is not equal to itself

from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import deque
import time

__version__ = "1.5.0"
__author__ = "hildolfr"
__license__ = "MIT"
__all__ = [
    "ConvergenceMonitor", "GewekeStatistic", "EffectiveSampleSize", 
    "convergence_diagnostic", "calculate_effective_sample_size",
    "BatchConvergenceAnalyzer", "split_chain_diagnostic", "export_convergence_data"
]

@dataclass
class ConvergenceMonitor:
    """Real-time convergence monitoring for Monte Carlo simulations."""
    
    # Configuration parameters
    window_size: int = 1000
    geweke_threshold: float = 2.0
    min_samples: int = 5000
    target_accuracy: float = 0.01
    confidence_level: float = 0.95
    
    # Internal state
    win_rates: deque = field(default_factory=lambda: deque(maxlen=10000))
    sample_counts: deque = field(default_factory=lambda: deque(maxlen=10000))
    running_wins: int = 0
    running_total: int = 0
    last_update_time: float = field(default_factory=time.time)
    convergence_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def update(self, win_rate: float, sample_count: int) -> None:
        """Update convergence monitor with new simulation results."""
        self.win_rates.append(win_rate)
        self.sample_counts.append(sample_count)
        self.running_total = sample_count
        self.running_wins = int(win_rate * sample_count)
        self.last_update_time = time.time()
        
        # Store convergence metrics periodically
        if sample_count > 0 and sample_count % self.window_size == 0:
            metrics = self._calculate_convergence_metrics()
            self.convergence_history.append({
                'sample_count': sample_count,
                'win_rate': win_rate,
                'geweke_stat': metrics.get('geweke_statistic', None),
                'ess': metrics.get('effective_sample_size', None),
                'margin_of_error': metrics.get('margin_of_error', None),
                'timestamp': self.last_update_time
            })
    
    def has_converged(self) -> bool:
        """Check if simulation has converged based on multiple criteria."""
        if self.running_total < self.min_samples:
            return False
        
        # Calculate current metrics
        metrics = self._calculate_convergence_metrics()
        
        # Geweke convergence test
        geweke_stat = metrics.get('geweke_statistic', float('inf'))
        geweke_converged = abs(geweke_stat) < self.geweke_threshold
        
        # Accuracy-based stopping
        margin_of_error = metrics.get('margin_of_error', float('inf'))
        accuracy_converged = margin_of_error < self.target_accuracy
        
        # Both criteria must be met
        return geweke_converged and accuracy_converged
    
    def get_convergence_status(self) -> Dict[str, Any]:
        """Get detailed convergence status information."""
        if self.running_total == 0:
            return {'status': 'insufficient_data', 'samples': 0}
        
        metrics = self._calculate_convergence_metrics()
        current_win_rate = self.running_wins / self.running_total
        
        return {
            'status': 'converged' if self.has_converged() else 'running',
            'samples': self.running_total,
            'current_win_rate': current_win_rate,
            'geweke_statistic': metrics.get('geweke_statistic'),
            'effective_sample_size': metrics.get('effective_sample_size'),
            'margin_of_error': metrics.get('margin_of_error'),
            'convergence_criteria': {
                'geweke_threshold': self.geweke_threshold,
                'target_accuracy': self.target_accuracy,
                'min_samples': self.min_samples
            }
        }
    
    def _calculate_convergence_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive convergence metrics."""
        if len(self.win_rates) < 100:
            return {}
        
        win_rates_array = list(self.win_rates) if not NUMPY_AVAILABLE else np.array(list(self.win_rates))
        
        # Geweke diagnostic
        geweke_stat = self._calculate_geweke_statistic(win_rates_array)
        
        # Effective sample size
        ess = self._calculate_effective_sample_size(win_rates_array)
        
        # Margin of error for current estimate
        current_win_rate = self.running_wins / self.running_total
        margin_of_error = self._calculate_margin_of_error(current_win_rate, self.running_total)
        
        return {
            'geweke_statistic': geweke_stat,
            'effective_sample_size': ess,
            'margin_of_error': margin_of_error
        }
    
    def _calculate_geweke_statistic(self, series) -> float:
        """Calculate Geweke convergence diagnostic."""
        if len(series) < 100:
            return float('inf')
        
        # Split series into first 10% and last 50% (Geweke convention)
        n = len(series)
        first_part = series[:max(1, n // 10)]
        last_part = series[max(1, n // 2):]
        
        if len(first_part) < 2 or len(last_part) < 2:
            return float('inf')
        
        # Calculate means
        mean_first = np.mean(first_part)
        mean_last = np.mean(last_part)
        
        # Calculate variances with spectral density estimation
        var_first = self._spectral_variance_estimate(first_part)
        var_last = self._spectral_variance_estimate(last_part)
        
        if var_first <= 0 or var_last <= 0:
            return float('inf')
        
        # Geweke statistic
        geweke_stat = (mean_first - mean_last) / math.sqrt(var_first + var_last)
        return geweke_stat
    
    def _spectral_variance_estimate(self, series) -> float:
        """Estimate variance using spectral density (simplified approach)."""
        if len(series) < 2:
            return 0.0
        
        # Simple spectral variance estimate using autocorrelation
        var_estimate = np.var(series, ddof=1)
        
        # Adjust for autocorrelation (basic approach)
        if len(series) > 10:
            # Calculate lag-1 autocorrelation
            x_vals = series[:-1]
            y_vals = series[1:]
            corr_matrix = np.corrcoef(x_vals, y_vals)
            lag1_corr = corr_matrix[0][1] if not np.isnan(corr_matrix[0][1]) else 0.0
            
            if abs(lag1_corr) < 0.99:
                # Adjust variance for autocorrelation
                var_estimate *= (1 + 2 * lag1_corr) / (1 - lag1_corr)
        
        return max(var_estimate / len(series), 1e-10)
    
    def _calculate_effective_sample_size(self, series) -> float:
        """Calculate effective sample size accounting for autocorrelation."""
        if len(series) < 10:
            return float(len(series))
        
        # Calculate autocorrelation function
        n = len(series)
        max_lag = min(50, n // 4)
        
        autocorr = self._calculate_autocorrelation(series, max_lag)
        
        # Find cutoff where autocorrelation becomes negligible
        cutoff = 1
        for lag in range(1, len(autocorr)):
            if autocorr[lag] <= 0.1:  # 10% threshold
                cutoff = lag
                break
        
        # Calculate integrated autocorrelation time
        tau_int = 1 + 2 * sum(autocorr[1:cutoff])
        tau_int = max(tau_int, 1.0)
        
        # Effective sample size
        ess = n / (2 * tau_int)
        return max(ess, 1.0)
    
    def _calculate_autocorrelation(self, series, max_lag: int):
        """Calculate autocorrelation function up to max_lag."""
        n = len(series)
        series_mean = np.mean(series)
        series_centered = [x - series_mean for x in series]
        
        autocorr = [0.0] * (max_lag + 1)
        variance = np.var(series, ddof=1)
        
        if variance <= 0:
            autocorr[0] = 1.0
            return autocorr
        
        for lag in range(max_lag + 1):
            if lag == 0:
                autocorr[0] = 1.0
            elif lag < n:
                covariance = sum(series_centered[i] * series_centered[i + lag] 
                               for i in range(n - lag)) / (n - lag)
                autocorr[lag] = covariance / variance
            else:
                autocorr[lag] = 0.0
        
        return autocorr
    
    def _calculate_margin_of_error(self, win_rate: float, sample_size: int) -> float:
        """Calculate margin of error for confidence interval."""
        if sample_size <= 0:
            return float('inf')
        
        # Z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z_score = z_scores.get(self.confidence_level, 1.96)
        
        # Standard error for binomial proportion
        standard_error = math.sqrt((win_rate * (1 - win_rate)) / sample_size)
        margin_of_error = z_score * standard_error
        
        return margin_of_error

@dataclass
class GewekeStatistic:
    """Container for Geweke convergence diagnostic results."""
    statistic: float
    threshold: float
    converged: bool
    first_segment_mean: float
    last_segment_mean: float
    first_segment_variance: float
    last_segment_variance: float

@dataclass 
class EffectiveSampleSize:
    """Container for effective sample size calculation results."""
    effective_size: float
    actual_size: int
    efficiency: float
    autocorrelation_time: float
    autocorrelation_cutoff: int

def convergence_diagnostic(win_rates: List[float], threshold: float = 2.0) -> GewekeStatistic:
    """Standalone function to calculate Geweke convergence diagnostic."""
    if len(win_rates) < 100:
        return GewekeStatistic(
            statistic=float('inf'), threshold=threshold, converged=False,
            first_segment_mean=0.0, last_segment_mean=0.0,
            first_segment_variance=0.0, last_segment_variance=0.0
        )
    
    monitor = ConvergenceMonitor(geweke_threshold=threshold)
    series = list(win_rates) if not NUMPY_AVAILABLE else np.array(win_rates)
    
    # Calculate Geweke statistic using monitor's method
    geweke_stat = monitor._calculate_geweke_statistic(series)
    
    # Calculate segment statistics for detailed output
    n = len(series)
    first_part = series[:max(1, n // 10)]
    last_part = series[max(1, n // 2):]
    
    return GewekeStatistic(
        statistic=geweke_stat,
        threshold=threshold,
        converged=abs(geweke_stat) < threshold,
        first_segment_mean=float(np.mean(first_part)),
        last_segment_mean=float(np.mean(last_part)),
        first_segment_variance=float(np.var(first_part, ddof=1)),
        last_segment_variance=float(np.var(last_part, ddof=1))
    )

def calculate_effective_sample_size(win_rates: List[float]) -> EffectiveSampleSize:
    """Standalone function to calculate effective sample size."""
    if len(win_rates) < 10:
        return EffectiveSampleSize(
            effective_size=float(len(win_rates)),
            actual_size=len(win_rates),
            efficiency=1.0,
            autocorrelation_time=1.0,
            autocorrelation_cutoff=1
        )
    
    monitor = ConvergenceMonitor()
    series = list(win_rates) if not NUMPY_AVAILABLE else np.array(win_rates)
    
    # Calculate effective sample size using monitor's method
    ess = monitor._calculate_effective_sample_size(series)
    actual_size = len(win_rates)
    efficiency = ess / actual_size
    
    # Calculate autocorrelation details
    max_lag = min(50, actual_size // 4)
    autocorr = monitor._calculate_autocorrelation(series, max_lag)
    
    cutoff = 1
    for lag in range(1, len(autocorr)):
        if autocorr[lag] <= 0.1:
            cutoff = lag
            break
    
    tau_int = 1 + 2 * sum(autocorr[1:cutoff])
    
    return EffectiveSampleSize(
        effective_size=ess,
        actual_size=actual_size,
        efficiency=efficiency,
        autocorrelation_time=tau_int,
        autocorrelation_cutoff=cutoff
    )

@dataclass
class BatchConvergenceAnalyzer:
    """
    Advanced batch analysis for convergence diagnostics.
    Implements Task 7.1.a: Enhanced convergence detection with batch analysis.
    """
    batch_size: int = 1000
    min_batches: int = 5
    r_hat_threshold: float = 1.1
    
    def analyze_batches(self, win_rates: List[float]) -> Dict[str, Any]:
        """Analyze convergence using batch-based R-hat and within/between variance."""
        if len(win_rates) < self.batch_size * self.min_batches:
            return {
                'r_hat': float('inf'),
                'converged': False,
                'insufficient_data': True,
                'batches_analyzed': 0
            }
        
        # Split data into batches
        num_batches = len(win_rates) // self.batch_size
        batches = []
        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = start_idx + self.batch_size
            batch = win_rates[start_idx:end_idx]
            batches.append(batch)
        
        # Calculate R-hat statistic (Gelman-Rubin diagnostic)
        r_hat = self._calculate_r_hat(batches)
        
        # Calculate within-batch and between-batch variance
        within_var, between_var = self._calculate_batch_variances(batches)
        
        return {
            'r_hat': r_hat,
            'converged': r_hat < self.r_hat_threshold,
            'within_batch_variance': within_var,
            'between_batch_variance': between_var,
            'batches_analyzed': num_batches,
            'batch_size': self.batch_size,
            'insufficient_data': False
        }
    
    def _calculate_r_hat(self, batches: List[List[float]]) -> float:
        """Calculate R-hat convergence diagnostic (Gelman-Rubin statistic)."""
        if len(batches) < 2:
            return float('inf')
        
        n = len(batches[0])  # batch size
        m = len(batches)     # number of batches
        
        # Calculate batch means
        batch_means = [np.mean(batch) for batch in batches]
        overall_mean = np.mean(batch_means)
        
        # Calculate within-batch variance (W)
        within_variances = [np.var(batch, ddof=1) for batch in batches]
        W = np.mean(within_variances)
        
        # Calculate between-batch variance (B)
        B = sum((mean - overall_mean) ** 2 for mean in batch_means) * n / (m - 1)
        
        if W <= 0:
            return float('inf')
        
        # Calculate R-hat
        var_hat = ((n - 1) / n) * W + (1 / n) * B
        r_hat = math.sqrt(var_hat / W)
        
        return r_hat
    
    def _calculate_batch_variances(self, batches: List[List[float]]) -> Tuple[float, float]:
        """Calculate within-batch and between-batch variances."""
        if len(batches) < 2:
            return 0.0, 0.0
        
        # Within-batch variance
        within_variances = [np.var(batch, ddof=1) for batch in batches]
        within_var = np.mean(within_variances)
        
        # Between-batch variance
        batch_means = [np.mean(batch) for batch in batches]
        overall_mean = np.mean(batch_means)
        between_var = np.var(batch_means, ddof=1)
        
        return within_var, between_var

@dataclass
class SplitChainDiagnostic:
    """
    Split-chain convergence diagnostic results.
    Implements Task 7.1.b: Cross-validation with split-chain analysis.
    """
    r_hat: float
    effective_sample_size: float
    converged: bool
    chain_length: int
    split_point: int
    first_half_mean: float
    second_half_mean: float
    pooled_variance: float

def split_chain_diagnostic(win_rates: List[float], threshold: float = 1.1) -> SplitChainDiagnostic:
    """
    Perform split-chain convergence diagnostic.
    Splits the chain in half and calculates R-hat between halves.
    """
    chain_length = len(win_rates)
    
    if chain_length < 100:
        return SplitChainDiagnostic(
            r_hat=float('inf'),
            effective_sample_size=float(chain_length),
            converged=False,
            chain_length=chain_length,
            split_point=chain_length // 2,
            first_half_mean=0.0,
            second_half_mean=0.0,
            pooled_variance=0.0
        )
    
    # Split chain in half
    split_point = chain_length // 2
    first_half = win_rates[:split_point]
    second_half = win_rates[split_point:]
    
    # Calculate means and variances
    first_mean = np.mean(first_half)
    second_mean = np.mean(second_half)
    first_var = np.var(first_half, ddof=1)
    second_var = np.var(second_half, ddof=1)
    
    # Calculate pooled variance
    pooled_var = (first_var + second_var) / 2
    
    # Calculate R-hat between the two halves
    if pooled_var <= 0:
        r_hat = float('inf')
    else:
        between_var = ((first_mean - second_mean) ** 2) / 2
        r_hat = math.sqrt(1 + between_var / pooled_var)
    
    # Calculate effective sample size for split chain
    ess_calc = calculate_effective_sample_size(win_rates)
    
    return SplitChainDiagnostic(
        r_hat=r_hat,
        effective_sample_size=ess_calc.effective_size,
        converged=r_hat < threshold,
        chain_length=chain_length,
        split_point=split_point,
        first_half_mean=first_mean,
        second_half_mean=second_mean,
        pooled_variance=pooled_var
    )

def export_convergence_data(convergence_history: List[Dict[str, Any]], 
                          scenario_info: Dict[str, Any],
                          filename: Optional[str] = None) -> Dict[str, Any]:
    """
    Export convergence data for visualization and analysis.
    Implements Task 7.1.c: Convergence rate visualization and export.
    """
    import json
    from datetime import datetime
    
    # Prepare export data structure
    export_data = {
        'metadata': {
            'export_timestamp': datetime.now().isoformat(),
            'poker_knight_version': __version__,
            'total_history_points': len(convergence_history),
            'scenario': scenario_info
        },
        'convergence_timeline': convergence_history,
        'summary_statistics': {}
    }
    
    # Calculate summary statistics if we have data
    if convergence_history:
        # Extract time series
        sample_counts = [entry.get('sample_count', 0) for entry in convergence_history]
        win_rates = [entry.get('win_rate', 0) for entry in convergence_history]
        geweke_stats = [entry.get('geweke_stat') for entry in convergence_history if entry.get('geweke_stat') is not None]
        margins_of_error = [entry.get('margin_of_error') for entry in convergence_history if entry.get('margin_of_error') is not None]
        
        export_data['summary_statistics'] = {
            'total_simulations': max(sample_counts) if sample_counts else 0,
            'final_win_rate': win_rates[-1] if win_rates else 0,
            'win_rate_variance': np.var(win_rates) if len(win_rates) > 1 else 0,
            'geweke_convergence_points': len([g for g in geweke_stats if abs(g) < 2.0]),
            'total_geweke_points': len(geweke_stats),
            'final_margin_of_error': margins_of_error[-1] if margins_of_error else None,
            'convergence_efficiency': len(convergence_history) / max(sample_counts) if sample_counts and max(sample_counts) > 0 else 0
        }
        
        # Calculate convergence rate (how quickly margin of error decreases)
        if len(margins_of_error) >= 2:
            convergence_rates = []
            for i in range(1, len(margins_of_error)):
                if margins_of_error[i] > 0 and margins_of_error[i-1] > 0:
                    rate = margins_of_error[i-1] / margins_of_error[i]
                    convergence_rates.append(rate)
            
            if convergence_rates:
                export_data['summary_statistics']['average_convergence_rate'] = np.mean(convergence_rates)
                export_data['summary_statistics']['convergence_rate_variance'] = np.var(convergence_rates)
    
    # Export to file if requested
    if filename:
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
    
    return export_data 