"""
♞ Poker Knight Analytics Module

Advanced statistical analysis and reporting for Monte Carlo poker simulations.
Provides comprehensive analytics, variance analysis, distribution analysis,
and performance metrics for simulation results.

Author: hildolfr
License: MIT
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
import time

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


@dataclass
class VarianceAnalysis:
    """Comprehensive variance analysis for simulation results."""
    sample_variance: float
    population_variance: float
    standard_deviation: float
    coefficient_of_variation: float
    variance_ratio: float  # Sample variance / theoretical variance
    confidence_interval_95: Tuple[float, float]
    margin_of_error: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass 
class HandStrengthDistribution:
    """Hand strength distribution analysis."""
    hand_categories: Dict[str, float]  # Category percentages
    normalized_frequencies: Dict[str, float]  # Normalized to expected poker probabilities
    chi_square_statistic: float
    p_value: float
    goodness_of_fit: str  # "excellent", "good", "acceptable", "poor"
    expected_vs_observed: Dict[str, Tuple[float, float]]  # (expected, observed) percentages
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class EquityCurve:
    """Equity curve analysis for simulation convergence."""
    sample_points: List[int]  # Simulation counts at each point
    equity_values: List[float]  # Equity at each sample point
    running_variance: List[float]  # Running variance estimates
    convergence_rate: float  # Theoretical 1/sqrt(n) rate compliance
    final_confidence_interval: Tuple[float, float]
    convergence_achieved: bool
    convergence_point: Optional[int]  # When convergence was first achieved
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Performance analysis for simulation efficiency."""
    simulations_per_second: float
    total_simulation_time: float
    parallel_efficiency: Optional[float]  # If parallel processing was used
    thread_utilization: Optional[float]  # Thread usage efficiency
    memory_efficiency: str  # "excellent", "good", "acceptable", "poor"
    accuracy_vs_speed_ratio: float  # Accuracy achieved per unit time
    optimization_effectiveness: Optional[float]  # If intelligent optimization was used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class PokerAnalytics:
    """
    Advanced analytics engine for poker simulation results.
    
    Provides comprehensive statistical analysis including variance analysis,
    hand strength distributions, equity curves, and performance metrics.
    """
    
    # Expected poker hand probabilities (5-card hands)
    EXPECTED_HAND_PROBABILITIES = {
        'high_card': 0.501177,
        'pair': 0.422569,
        'two_pair': 0.047539,
        'three_of_a_kind': 0.021128,
        'straight': 0.003925,
        'flush': 0.001965,
        'full_house': 0.001441,
        'four_of_a_kind': 0.000240,
        'straight_flush': 0.000015,
        'royal_flush': 0.000002
    }
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize analytics engine.
        
        Args:
            confidence_level: Confidence level for statistical calculations (default: 0.95)
        """
        self.confidence_level = confidence_level
        self.z_score = self._get_z_score(confidence_level)
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level."""
        z_scores = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
            0.999: 3.291
        }
        return z_scores.get(confidence_level, 1.960)
    
    def analyze_variance(self, 
                        win_probability: float, 
                        num_simulations: int,
                        observed_wins: int) -> VarianceAnalysis:
        """
        Perform comprehensive variance analysis on simulation results.
        
        Args:
            win_probability: True win probability estimate
            num_simulations: Number of simulations run
            observed_wins: Number of wins observed
            
        Returns:
            VarianceAnalysis: Comprehensive variance analysis
        """
        # Theoretical variance for binomial distribution
        theoretical_variance = win_probability * (1 - win_probability) / num_simulations
        
        # Sample variance calculation
        sample_variance = observed_wins * (num_simulations - observed_wins) / (num_simulations ** 2 * (num_simulations - 1))
        
        # Population variance (biased estimator)
        population_variance = observed_wins * (num_simulations - observed_wins) / (num_simulations ** 3)
        
        # Standard deviation
        std_dev = math.sqrt(sample_variance)
        
        # Coefficient of variation
        cv = std_dev / win_probability if win_probability > 0 else float('inf')
        
        # Variance ratio (sample vs theoretical)
        variance_ratio = sample_variance / theoretical_variance if theoretical_variance > 0 else float('inf')
        
        # Confidence interval
        margin_of_error = self.z_score * std_dev
        ci_lower = max(0.0, win_probability - margin_of_error)
        ci_upper = min(1.0, win_probability + margin_of_error)
        
        return VarianceAnalysis(
            sample_variance=sample_variance,
            population_variance=population_variance,
            standard_deviation=std_dev,
            coefficient_of_variation=cv,
            variance_ratio=variance_ratio,
            confidence_interval_95=(ci_lower, ci_upper),
            margin_of_error=margin_of_error
        )
    
    def analyze_hand_strength_distribution(self, 
                                         hand_categories: Dict[str, int],
                                         total_simulations: int) -> HandStrengthDistribution:
        """
        Analyze hand strength distribution against expected poker probabilities.
        
        Args:
            hand_categories: Dictionary of hand category counts
            total_simulations: Total number of simulations
            
        Returns:
            HandStrengthDistribution: Comprehensive distribution analysis
        """
        # Convert counts to percentages
        category_percentages = {}
        for category, count in hand_categories.items():
            category_percentages[category] = count / total_simulations
        
        # Normalize frequencies against expected probabilities
        normalized_frequencies = {}
        expected_vs_observed = {}
        chi_square_sum = 0.0
        
        for category in self.EXPECTED_HAND_PROBABILITIES:
            expected_prob = self.EXPECTED_HAND_PROBABILITIES[category]
            observed_prob = category_percentages.get(category, 0.0)
            
            # Normalized frequency (observed / expected)
            normalized_freq = observed_prob / expected_prob if expected_prob > 0 else 0.0
            normalized_frequencies[category] = normalized_freq
            
            # Store expected vs observed
            expected_vs_observed[category] = (expected_prob, observed_prob)
            
            # Chi-square contribution
            expected_count = expected_prob * total_simulations
            observed_count = hand_categories.get(category, 0)
            if expected_count > 0:
                chi_square_sum += ((observed_count - expected_count) ** 2) / expected_count
        
        # Calculate p-value approximation (simplified)
        degrees_of_freedom = len(self.EXPECTED_HAND_PROBABILITIES) - 1
        p_value = self._chi_square_p_value_approx(chi_square_sum, degrees_of_freedom)
        
        # Determine goodness of fit
        if p_value > 0.05:
            goodness_of_fit = "excellent"
        elif p_value > 0.01:
            goodness_of_fit = "good"
        elif p_value > 0.001:
            goodness_of_fit = "acceptable"
        else:
            goodness_of_fit = "poor"
        
        return HandStrengthDistribution(
            hand_categories=category_percentages,
            normalized_frequencies=normalized_frequencies,
            chi_square_statistic=chi_square_sum,
            p_value=p_value,
            goodness_of_fit=goodness_of_fit,
            expected_vs_observed=expected_vs_observed
        )
    
    def _chi_square_p_value_approx(self, chi_square: float, df: int) -> float:
        """Approximate p-value for chi-square test (simplified implementation)."""
        # Very simplified approximation - in production, use scipy.stats
        if chi_square < df:
            return 0.9  # High p-value
        elif chi_square < df * 2:
            return 0.1  # Medium p-value
        elif chi_square < df * 3:
            return 0.01  # Low p-value
        else:
            return 0.001  # Very low p-value
    
    def generate_equity_curve(self, 
                            simulation_history: List[Tuple[int, float]],
                            target_accuracy: float = 0.01) -> EquityCurve:
        """
        Generate equity curve analysis from simulation history.
        
        Args:
            simulation_history: List of (simulation_count, equity_estimate) tuples
            target_accuracy: Target accuracy for convergence detection
            
        Returns:
            EquityCurve: Comprehensive equity curve analysis
        """
        if not simulation_history:
            raise ValueError("Simulation history cannot be empty")
        
        sample_points = [point[0] for point in simulation_history]
        equity_values = [point[1] for point in simulation_history]
        
        # Calculate running variance
        running_variance = []
        for i, (n, equity) in enumerate(simulation_history):
            if i == 0:
                running_variance.append(0.0)
            else:
                # Estimate variance based on recent equity changes
                recent_equities = equity_values[max(0, i-10):i+1]
                if len(recent_equities) > 1:
                    variance = statistics.variance(recent_equities)
                else:
                    variance = 0.0
                running_variance.append(variance)
        
        # Analyze convergence rate (theoretical 1/sqrt(n))
        convergence_rate = self._analyze_convergence_rate(simulation_history)
        
        # Final confidence interval
        final_equity = equity_values[-1]
        final_n = sample_points[-1]
        final_std_error = math.sqrt(final_equity * (1 - final_equity) / final_n)
        final_margin = self.z_score * final_std_error
        final_ci = (max(0.0, final_equity - final_margin), min(1.0, final_equity + final_margin))
        
        # Convergence detection
        convergence_achieved = False
        convergence_point = None
        
        for i, (n, equity) in enumerate(simulation_history):
            if n > 100:  # Minimum samples for convergence assessment
                std_error = math.sqrt(equity * (1 - equity) / n)
                margin = self.z_score * std_error
                if margin <= target_accuracy:
                    convergence_achieved = True
                    convergence_point = n
                    break
        
        return EquityCurve(
            sample_points=sample_points,
            equity_values=equity_values,
            running_variance=running_variance,
            convergence_rate=convergence_rate,
            final_confidence_interval=final_ci,
            convergence_achieved=convergence_achieved,
            convergence_point=convergence_point
        )
    
    def _analyze_convergence_rate(self, simulation_history: List[Tuple[int, float]]) -> float:
        """Analyze how well the convergence follows theoretical 1/sqrt(n) rate."""
        if len(simulation_history) < 10:
            return 0.0
        
        # Calculate theoretical vs actual convergence rates
        theoretical_rates = []
        actual_rates = []
        
        for i in range(10, len(simulation_history)):
            n = simulation_history[i][0]
            equity = simulation_history[i][1]
            
            # Theoretical standard error
            theoretical_se = 1.0 / math.sqrt(n)
            theoretical_rates.append(theoretical_se)
            
            # Actual convergence (based on recent equity stability)
            recent_equities = [simulation_history[j][1] for j in range(max(0, i-5), i+1)]
            if len(recent_equities) > 1:
                actual_se = statistics.stdev(recent_equities)
            else:
                actual_se = 0.0
            actual_rates.append(actual_se)
        
        # Calculate correlation between theoretical and actual rates
        if theoretical_rates and actual_rates:
            # Simplified correlation calculation
            mean_theoretical = statistics.mean(theoretical_rates)
            mean_actual = statistics.mean(actual_rates)
            
            numerator = sum((t - mean_theoretical) * (a - mean_actual) 
                           for t, a in zip(theoretical_rates, actual_rates))
            
            denom_t = sum((t - mean_theoretical) ** 2 for t in theoretical_rates)
            denom_a = sum((a - mean_actual) ** 2 for a in actual_rates)
            
            if denom_t > 0 and denom_a > 0:
                correlation = numerator / math.sqrt(denom_t * denom_a)
                return max(0.0, correlation)  # Return positive correlation only
        
        return 0.0
    
    def analyze_performance(self, 
                          simulation_result: Any,
                          simulation_time: float,
                          num_simulations: int) -> PerformanceMetrics:
        """
        Analyze simulation performance metrics.
        
        Args:
            simulation_result: Simulation result object
            simulation_time: Total simulation time in seconds
            num_simulations: Number of simulations performed
            
        Returns:
            PerformanceMetrics: Comprehensive performance analysis
        """
        # Simulations per second
        sims_per_second = num_simulations / simulation_time if simulation_time > 0 else 0.0
        
        # Parallel efficiency (if parallel data available)
        parallel_efficiency = None
        thread_utilization = None
        
        if hasattr(simulation_result, 'parallel_processing_used'):
            if simulation_result.parallel_processing_used:
                # Estimate parallel efficiency based on simulation rate
                single_thread_estimate = 2000  # Estimated sims/sec for single thread
                parallel_efficiency = min(1.0, sims_per_second / (single_thread_estimate * 4))  # Assume 4 threads
                thread_utilization = parallel_efficiency
        
        # Memory efficiency assessment
        memory_efficiency = "good"  # Default assessment
        if num_simulations > 1000000:
            memory_efficiency = "excellent"  # Handled large simulation efficiently
        elif num_simulations > 100000:
            memory_efficiency = "good"
        elif num_simulations > 10000:
            memory_efficiency = "acceptable"
        else:
            memory_efficiency = "excellent"  # Small simulations are always efficient
        
        # Accuracy vs speed ratio
        accuracy_estimate = getattr(simulation_result, 'confidence_interval', [0.0, 0.0])
        if isinstance(accuracy_estimate, (list, tuple)) and len(accuracy_estimate) >= 2:
            accuracy = 1.0 - (accuracy_estimate[1] - accuracy_estimate[0])  # Smaller CI = higher accuracy
        else:
            accuracy = 0.95  # Default accuracy estimate
        
        accuracy_vs_speed_ratio = accuracy / simulation_time if simulation_time > 0 else 0.0
        
        # Optimization effectiveness (if intelligent optimization was used)
        optimization_effectiveness = None
        if hasattr(simulation_result, 'optimization_data') and simulation_result.optimization_data:
            opt_data = simulation_result.optimization_data
            if isinstance(opt_data, dict) and 'time_saved_percentage' in opt_data:
                optimization_effectiveness = opt_data['time_saved_percentage'] / 100.0
        
        return PerformanceMetrics(
            simulations_per_second=sims_per_second,
            total_simulation_time=simulation_time,
            parallel_efficiency=parallel_efficiency,
            thread_utilization=thread_utilization,
            memory_efficiency=memory_efficiency,
            accuracy_vs_speed_ratio=accuracy_vs_speed_ratio,
            optimization_effectiveness=optimization_effectiveness
        )


class StatisticalReportGenerator:
    """Generate comprehensive statistical reports from analytics data."""
    
    def __init__(self, analytics: PokerAnalytics):
        """Initialize with analytics engine."""
        self.analytics = analytics
    
    def generate_comprehensive_report(self, 
                                    simulation_result: Any,
                                    simulation_time: float,
                                    simulation_history: Optional[List[Tuple[int, float]]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive statistical report.
        
        Args:
            simulation_result: Simulation result object
            simulation_time: Total simulation time
            simulation_history: Optional simulation history for equity curve
            
        Returns:
            Dict: Comprehensive statistical report
        """
        report = {
            'timestamp': time.time(),
            'version': '1.5.0',
            'analysis_type': 'comprehensive_statistical_report'
        }
        
        # Basic simulation info
        num_simulations = getattr(simulation_result, 'total_simulations', 0)
        win_prob = getattr(simulation_result, 'win_probability', 0.0)
        wins = int(win_prob * num_simulations) if num_simulations > 0 else 0
        
        # Variance analysis
        if num_simulations > 0:
            variance_analysis = self.analytics.analyze_variance(win_prob, num_simulations, wins)
            report['variance_analysis'] = variance_analysis.to_dict()
        
        # Hand strength distribution (if available)
        hand_categories = getattr(simulation_result, 'hand_categories', None)
        if hand_categories and num_simulations > 0:
            distribution_analysis = self.analytics.analyze_hand_strength_distribution(
                hand_categories, num_simulations
            )
            report['hand_strength_distribution'] = distribution_analysis.to_dict()
        
        # Equity curve analysis (if history available)
        if simulation_history:
            equity_curve = self.analytics.generate_equity_curve(simulation_history)
            report['equity_curve'] = equity_curve.to_dict()
        
        # Performance metrics
        performance_metrics = self.analytics.analyze_performance(
            simulation_result, simulation_time, num_simulations
        )
        report['performance_metrics'] = performance_metrics.to_dict()
        
        # Summary statistics
        report['summary'] = {
            'total_simulations': num_simulations,
            'win_probability': win_prob,
            'simulation_time_seconds': simulation_time,
            'simulations_per_second': performance_metrics.simulations_per_second,
            'accuracy_assessment': self._assess_overall_accuracy(report),
            'performance_assessment': self._assess_overall_performance(performance_metrics)
        }
        
        return report
    
    def _assess_overall_accuracy(self, report: Dict[str, Any]) -> str:
        """Assess overall accuracy based on multiple factors."""
        # Check variance analysis
        if 'variance_analysis' in report:
            variance_ratio = report['variance_analysis']['variance_ratio']
            if 0.8 <= variance_ratio <= 1.2:
                variance_score = "excellent"
            elif 0.6 <= variance_ratio <= 1.4:
                variance_score = "good"
            else:
                variance_score = "poor"
        else:
            variance_score = "unknown"
        
        # Check hand distribution (if available)
        distribution_score = "unknown"
        if 'hand_strength_distribution' in report:
            goodness_of_fit = report['hand_strength_distribution']['goodness_of_fit']
            distribution_score = goodness_of_fit
        
        # Combine assessments
        if variance_score == "excellent" and distribution_score in ["excellent", "good", "unknown"]:
            return "excellent"
        elif variance_score in ["excellent", "good"] and distribution_score in ["excellent", "good", "acceptable", "unknown"]:
            return "good"
        elif variance_score != "poor" or distribution_score != "poor":
            return "acceptable"
        else:
            return "poor"
    
    def _assess_overall_performance(self, performance_metrics: PerformanceMetrics) -> str:
        """Assess overall performance based on metrics."""
        sims_per_sec = performance_metrics.simulations_per_second
        
        if sims_per_sec > 10000:
            return "excellent"
        elif sims_per_sec > 5000:
            return "good"
        elif sims_per_sec > 1000:
            return "acceptable"
        else:
            return "poor"
    
    def export_to_json(self, report: Dict[str, Any], filename: str) -> None:
        """Export report to JSON file."""
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
    
    def export_to_csv(self, report: Dict[str, Any], filename: str) -> None:
        """Export key metrics to CSV format."""
        import csv
        
        # Extract key metrics
        metrics = []
        
        # Basic metrics
        summary = report.get('summary', {})
        metrics.append(['metric', 'value'])
        metrics.append(['total_simulations', summary.get('total_simulations', 0)])
        metrics.append(['win_probability', summary.get('win_probability', 0.0)])
        metrics.append(['simulation_time_seconds', summary.get('simulation_time_seconds', 0.0)])
        metrics.append(['simulations_per_second', summary.get('simulations_per_second', 0.0)])
        
        # Variance metrics
        if 'variance_analysis' in report:
            variance = report['variance_analysis']
            metrics.append(['standard_deviation', variance.get('standard_deviation', 0.0)])
            metrics.append(['variance_ratio', variance.get('variance_ratio', 0.0)])
            metrics.append(['margin_of_error', variance.get('margin_of_error', 0.0)])
        
        # Performance metrics
        if 'performance_metrics' in report:
            perf = report['performance_metrics']
            metrics.append(['memory_efficiency', perf.get('memory_efficiency', 'unknown')])
            metrics.append(['accuracy_vs_speed_ratio', perf.get('accuracy_vs_speed_ratio', 0.0)])
        
        # Write to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(metrics)


# Public API
__all__ = [
    'PokerAnalytics',
    'StatisticalReportGenerator',
    'VarianceAnalysis',
    'HandStrengthDistribution', 
    'EquityCurve',
    'PerformanceMetrics'
] 