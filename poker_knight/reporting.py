"""
♞ Poker Knight Reporting Module

Comprehensive statistical reporting and performance metrics dashboard
for Monte Carlo poker simulations. Provides detailed reports, visualizations,
and export capabilities for analysis results.

Author: hildolfr
License: MIT
"""

from typing import Dict, List, Any, Optional, Tuple, Union
import json
import time
import os
from dataclasses import dataclass, asdict
from .analytics import PokerAnalytics, StatisticalReportGenerator, VarianceAnalysis, HandStrengthDistribution, EquityCurve, PerformanceMetrics

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


@dataclass
class ReportConfiguration:
    """Configuration settings for report generation."""
    include_variance_analysis: bool = True
    include_distribution_analysis: bool = True
    include_equity_curve: bool = True
    include_performance_metrics: bool = True
    include_visualizations: bool = True
    export_formats: List[str] = None  # ['json', 'csv', 'html']
    confidence_level: float = 0.95
    chart_style: str = 'seaborn'  # matplotlib style
    
    def __post_init__(self):
        if self.export_formats is None:
            self.export_formats = ['json', 'csv']


class PerformanceDashboard:
    """
    Performance metrics dashboard for simulation analysis.
    
    Provides comprehensive performance analysis including simulation efficiency,
    thread utilization, accuracy vs speed trade-offs, and optimization effectiveness.
    """
    
    def __init__(self, config: Optional[ReportConfiguration] = None):
        """
        Initialize performance dashboard.
        
        Args:
            config: Report configuration settings
        """
        self.config = config or ReportConfiguration()
        self.analytics = PokerAnalytics(confidence_level=self.config.confidence_level)
        self.report_generator = StatisticalReportGenerator(self.analytics)
    
    def generate_performance_summary(self, 
                                   simulation_results: List[Any],
                                   simulation_times: List[float],
                                   simulation_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate performance summary across multiple simulation runs.
        
        Args:
            simulation_results: List of simulation result objects
            simulation_times: List of corresponding simulation times
            simulation_names: Optional names for each simulation
            
        Returns:
            Dict: Comprehensive performance summary
        """
        if len(simulation_results) != len(simulation_times):
            raise ValueError("Simulation results and times must have same length")
        
        if simulation_names and len(simulation_names) != len(simulation_results):
            raise ValueError("Simulation names must match number of results")
        
        if not simulation_names:
            simulation_names = [f"Simulation_{i+1}" for i in range(len(simulation_results))]
        
        summary = {
            'timestamp': time.time(),
            'version': '1.5.0',
            'report_type': 'performance_dashboard_summary',
            'total_simulations_analyzed': len(simulation_results),
            'individual_analyses': [],
            'comparative_analysis': {},
            'optimization_insights': {},
            'recommendations': []
        }
        
        # Analyze each simulation individually
        individual_metrics = []
        for i, (result, sim_time, name) in enumerate(zip(simulation_results, simulation_times, simulation_names)):
            num_sims = getattr(result, 'total_simulations', 0)
            
            perf_metrics = self.analytics.analyze_performance(result, sim_time, num_sims)
            
            individual_analysis = {
                'name': name,
                'index': i,
                'performance_metrics': perf_metrics.to_dict(),
                'efficiency_score': self._calculate_efficiency_score(perf_metrics),
                'optimization_used': hasattr(result, 'optimization_data') and result.optimization_data is not None
            }
            
            summary['individual_analyses'].append(individual_analysis)
            individual_metrics.append(perf_metrics)
        
        # Comparative analysis
        summary['comparative_analysis'] = self._generate_comparative_analysis(
            individual_metrics, simulation_names
        )
        
        # Optimization insights
        summary['optimization_insights'] = self._analyze_optimization_effectiveness(
            simulation_results, individual_metrics
        )
        
        # Generate recommendations
        summary['recommendations'] = self._generate_performance_recommendations(
            individual_metrics, summary['comparative_analysis']
        )
        
        return summary
    
    def _calculate_efficiency_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall efficiency score (0.0 to 1.0)."""
        score = 0.0
        components = 0
        
        # Simulation speed component (0-0.4)
        if metrics.simulations_per_second > 0:
            # Normalize to typical ranges
            speed_score = min(0.4, metrics.simulations_per_second / 25000)  # Max at 25K sims/sec
            score += speed_score
            components += 1
        
        # Accuracy vs speed ratio component (0-0.3)
        if metrics.accuracy_vs_speed_ratio > 0:
            # Normalize to typical ranges
            accuracy_score = min(0.3, metrics.accuracy_vs_speed_ratio * 0.3)
            score += accuracy_score
            components += 1
        
        # Memory efficiency component (0-0.2)
        memory_scores = {"excellent": 0.2, "good": 0.15, "acceptable": 0.1, "poor": 0.0}
        score += memory_scores.get(metrics.memory_efficiency, 0.0)
        components += 1
        
        # Optimization effectiveness component (0-0.1)
        if metrics.optimization_effectiveness is not None:
            opt_score = min(0.1, metrics.optimization_effectiveness * 0.1)
            score += opt_score
            components += 1
        
        return score if components == 0 else score
    
    def _generate_comparative_analysis(self, 
                                     metrics_list: List[PerformanceMetrics],
                                     names: List[str]) -> Dict[str, Any]:
        """Generate comparative analysis across simulations."""
        if not metrics_list:
            return {}
        
        # Extract metrics for comparison
        speeds = [m.simulations_per_second for m in metrics_list]
        times = [m.total_simulation_time for m in metrics_list]
        accuracy_ratios = [m.accuracy_vs_speed_ratio for m in metrics_list]
        
        # Find best/worst performers
        fastest_idx = speeds.index(max(speeds))
        slowest_idx = speeds.index(min(speeds))
        most_efficient_idx = accuracy_ratios.index(max(accuracy_ratios))
        
        # Calculate statistics
        avg_speed = sum(speeds) / len(speeds)
        avg_time = sum(times) / len(times)
        speed_variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
        
        return {
            'performance_statistics': {
                'average_simulations_per_second': avg_speed,
                'average_execution_time': avg_time,
                'speed_variance': speed_variance,
                'performance_consistency': 1.0 - (speed_variance / (avg_speed ** 2)) if avg_speed > 0 else 0.0
            },
            'best_performers': {
                'fastest_simulation': {
                    'name': names[fastest_idx],
                    'simulations_per_second': speeds[fastest_idx]
                },
                'most_efficient': {
                    'name': names[most_efficient_idx],
                    'accuracy_vs_speed_ratio': accuracy_ratios[most_efficient_idx]
                }
            },
            'worst_performers': {
                'slowest_simulation': {
                    'name': names[slowest_idx],
                    'simulations_per_second': speeds[slowest_idx]
                }
            },
            'optimization_opportunities': self._identify_optimization_opportunities(metrics_list, names)
        }
    
    def _analyze_optimization_effectiveness(self, 
                                          results: List[Any],
                                          metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze effectiveness of intelligent optimization."""
        optimized_results = []
        standard_results = []
        
        for result, metric in zip(results, metrics):
            if hasattr(result, 'optimization_data') and result.optimization_data:
                optimized_results.append((result, metric))
            else:
                standard_results.append((result, metric))
        
        insights = {
            'optimization_usage': {
                'total_simulations': len(results),
                'optimized_simulations': len(optimized_results),
                'standard_simulations': len(standard_results),
                'optimization_adoption_rate': len(optimized_results) / len(results) if results else 0.0
            }
        }
        
        if optimized_results and standard_results:
            # Compare optimized vs standard performance
            opt_speeds = [m.simulations_per_second for _, m in optimized_results]
            std_speeds = [m.simulations_per_second for _, m in standard_results]
            
            avg_opt_speed = sum(opt_speeds) / len(opt_speeds)
            avg_std_speed = sum(std_speeds) / len(std_speeds)
            
            insights['performance_comparison'] = {
                'average_optimized_speed': avg_opt_speed,
                'average_standard_speed': avg_std_speed,
                'optimization_speedup': avg_opt_speed / avg_std_speed if avg_std_speed > 0 else 1.0,
                'optimization_benefit': (avg_opt_speed - avg_std_speed) / avg_std_speed if avg_std_speed > 0 else 0.0
            }
            
            # Analyze optimization effectiveness distribution
            effectiveness_scores = []
            for result, _ in optimized_results:
                if hasattr(result, 'optimization_data') and result.optimization_data:
                    opt_data = result.optimization_data
                    if isinstance(opt_data, dict) and 'time_saved_percentage' in opt_data:
                        effectiveness_scores.append(opt_data['time_saved_percentage'])
            
            if effectiveness_scores:
                insights['optimization_effectiveness'] = {
                    'average_time_savings_percentage': sum(effectiveness_scores) / len(effectiveness_scores),
                    'min_time_savings': min(effectiveness_scores),
                    'max_time_savings': max(effectiveness_scores),
                    'optimization_consistency': 1.0 - (max(effectiveness_scores) - min(effectiveness_scores)) / 100.0
                }
        
        return insights
    
    def _identify_optimization_opportunities(self, 
                                           metrics: List[PerformanceMetrics],
                                           names: List[str]) -> List[Dict[str, Any]]:
        """Identify optimization opportunities for each simulation."""
        opportunities = []
        
        for i, (metric, name) in enumerate(zip(metrics, names)):
            simulation_opportunities = {
                'simulation_name': name,
                'index': i,
                'opportunities': []
            }
            
            # Check simulation speed
            if metric.simulations_per_second < 5000:
                simulation_opportunities['opportunities'].append({
                    'type': 'performance',
                    'issue': 'low_simulation_speed',
                    'description': f"Simulation speed ({metric.simulations_per_second:.0f} sims/sec) is below optimal",
                    'recommendation': "Consider enabling intelligent optimization or parallel processing"
                })
            
            # Check memory efficiency
            if metric.memory_efficiency in ['acceptable', 'poor']:
                simulation_opportunities['opportunities'].append({
                    'type': 'memory',
                    'issue': 'suboptimal_memory_usage',
                    'description': f"Memory efficiency is {metric.memory_efficiency}",
                    'recommendation': "Review simulation parameters for memory optimization opportunities"
                })
            
            # Check optimization usage
            if metric.optimization_effectiveness is None:
                simulation_opportunities['opportunities'].append({
                    'type': 'optimization',
                    'issue': 'optimization_not_used',
                    'description': "Intelligent optimization not utilized",
                    'recommendation': "Enable intelligent_optimization=True for automatic performance tuning"
                })
            
            # Check parallel efficiency
            if metric.parallel_efficiency is not None and metric.parallel_efficiency < 0.7:
                simulation_opportunities['opportunities'].append({
                    'type': 'parallel',
                    'issue': 'poor_parallel_efficiency',
                    'description': f"Parallel efficiency ({metric.parallel_efficiency:.1%}) is suboptimal",
                    'recommendation': "Review thread count and simulation batch sizes"
                })
            
            if simulation_opportunities['opportunities']:
                opportunities.append(simulation_opportunities)
        
        return opportunities
    
    def _generate_performance_recommendations(self, 
                                            metrics: List[PerformanceMetrics],
                                            comparative_analysis: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable performance recommendations."""
        recommendations = []
        
        # Overall performance assessment
        avg_speed = comparative_analysis.get('performance_statistics', {}).get('average_simulations_per_second', 0)
        
        if avg_speed < 5000:
            recommendations.append({
                'priority': 'high',
                'category': 'performance',
                'title': 'Enable Intelligent Optimization',
                'description': "Average simulation speed is below optimal. Enable intelligent_optimization=True to automatically tune simulation parameters for better performance."
            })
        
        # Memory efficiency recommendations
        memory_issues = sum(1 for m in metrics if m.memory_efficiency in ['acceptable', 'poor'])
        if memory_issues > len(metrics) * 0.3:  # More than 30% have memory issues
            recommendations.append({
                'priority': 'medium',
                'category': 'memory',
                'title': 'Optimize Memory Usage',
                'description': "Multiple simulations show suboptimal memory efficiency. Consider reducing simulation counts or enabling memory optimizations."
            })
        
        # Parallel processing recommendations
        parallel_metrics = [m for m in metrics if m.parallel_efficiency is not None]
        if parallel_metrics:
            avg_parallel_efficiency = sum(m.parallel_efficiency for m in parallel_metrics) / len(parallel_metrics)
            if avg_parallel_efficiency < 0.7:
                recommendations.append({
                    'priority': 'medium',
                    'category': 'parallel',
                    'title': 'Improve Parallel Processing',
                    'description': f"Average parallel efficiency ({avg_parallel_efficiency:.1%}) is suboptimal. Consider adjusting thread count or batch sizes."
                })
        
        # Optimization adoption recommendations
        optimization_usage = sum(1 for m in metrics if m.optimization_effectiveness is not None)
        if optimization_usage < len(metrics) * 0.5:  # Less than 50% using optimization
            recommendations.append({
                'priority': 'high',
                'category': 'optimization',
                'title': 'Increase Optimization Adoption',
                'description': "Less than half of simulations use intelligent optimization. Enable optimization across all simulations for better performance."
            })
        
        return recommendations


class StatisticalReport:
    """
    Comprehensive statistical report generator.
    
    Generates detailed reports combining analytics, performance metrics,
    and visualizations for poker simulation analysis.
    """
    
    def __init__(self, config: Optional[ReportConfiguration] = None):
        """Initialize statistical report generator."""
        self.config = config or ReportConfiguration()
        self.analytics = PokerAnalytics(confidence_level=self.config.confidence_level)
        self.report_generator = StatisticalReportGenerator(self.analytics)
        self.dashboard = PerformanceDashboard(config)
    
    def generate_full_analysis_report(self, 
                                    simulation_result: Any,
                                    simulation_time: float,
                                    simulation_history: Optional[List[Tuple[int, float]]] = None,
                                    output_directory: str = "reports") -> Dict[str, Any]:
        """
        Generate comprehensive analysis report with all features.
        
        Args:
            simulation_result: Simulation result object
            simulation_time: Total simulation time
            simulation_history: Optional simulation history for equity curve
            output_directory: Directory to save report files
            
        Returns:
            Dict: Comprehensive analysis report
        """
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        # Generate base report
        report = self.report_generator.generate_comprehensive_report(
            simulation_result, simulation_time, simulation_history
        )
        
        # Add report metadata
        report.update({
            'report_configuration': asdict(self.config),
            'generation_timestamp': time.time(),
            'report_version': '1.5.0',
            'report_type': 'full_analysis_report'
        })
        
        # Generate visualizations if enabled and matplotlib available
        if self.config.include_visualizations and MATPLOTLIB_AVAILABLE:
            try:
                visualization_files = self._generate_visualizations(
                    report, simulation_history, output_directory
                )
                report['visualizations'] = visualization_files
            except Exception as e:
                report['visualization_error'] = str(e)
        
        # Export in requested formats
        exported_files = {}
        for format_type in self.config.export_formats:
            try:
                filename = self._export_report(report, format_type, output_directory)
                exported_files[format_type] = filename
            except Exception as e:
                exported_files[f"{format_type}_error"] = str(e)
        
        report['exported_files'] = exported_files
        
        return report
    
    def _generate_visualizations(self, 
                               report: Dict[str, Any],
                               simulation_history: Optional[List[Tuple[int, float]]],
                               output_dir: str) -> Dict[str, str]:
        """Generate visualization files."""
        viz_files = {}
        
        # Set matplotlib style
        if self.config.chart_style in plt.style.available:
            plt.style.use(self.config.chart_style)
        
        # Variance analysis visualization
        if 'variance_analysis' in report:
            filename = self._plot_variance_analysis(report['variance_analysis'], output_dir)
            viz_files['variance_plot'] = filename
        
        # Hand distribution visualization
        if 'hand_strength_distribution' in report:
            filename = self._plot_hand_distribution(report['hand_strength_distribution'], output_dir)
            viz_files['distribution_plot'] = filename
        
        # Equity curve visualization
        if simulation_history and 'equity_curve' in report:
            filename = self._plot_equity_curve(simulation_history, report['equity_curve'], output_dir)
            viz_files['equity_curve_plot'] = filename
        
        # Performance metrics visualization
        if 'performance_metrics' in report:
            filename = self._plot_performance_metrics(report['performance_metrics'], output_dir)
            viz_files['performance_plot'] = filename
        
        return viz_files
    
    def _plot_variance_analysis(self, variance_data: Dict[str, Any], output_dir: str) -> str:
        """Generate variance analysis plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Confidence interval visualization
        ci = variance_data['confidence_interval_95']
        mean_val = (ci[0] + ci[1]) / 2
        margin = variance_data['margin_of_error']
        
        ax1.errorbar([0], [mean_val], yerr=[margin], fmt='o', capsize=10, capthick=2)
        ax1.set_xlim(-0.5, 0.5)
        ax1.set_ylim(max(0, mean_val - margin * 2), min(1, mean_val + margin * 2))
        ax1.set_title('95% Confidence Interval')
        ax1.set_ylabel('Win Probability')
        ax1.grid(True, alpha=0.3)
        
        # Variance metrics bar chart
        metrics = ['Standard Deviation', 'Coefficient of Variation', 'Variance Ratio']
        values = [
            variance_data['standard_deviation'],
            variance_data['coefficient_of_variation'],
            variance_data['variance_ratio']
        ]
        
        bars = ax2.bar(metrics, values, color=['skyblue', 'lightcoral', 'lightgreen'])
        ax2.set_title('Variance Metrics')
        ax2.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, 'variance_analysis.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _plot_hand_distribution(self, distribution_data: Dict[str, Any], output_dir: str) -> str:
        """Generate hand distribution comparison plot."""
        expected_vs_observed = distribution_data['expected_vs_observed']
        
        categories = list(expected_vs_observed.keys())
        expected_values = [expected_vs_observed[cat][0] * 100 for cat in categories]
        observed_values = [expected_vs_observed[cat][1] * 100 for cat in categories]
        
        x = range(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        bars1 = ax.bar([i - width/2 for i in x], expected_values, width, 
                      label='Expected', color='lightblue', alpha=0.8)
        bars2 = ax.bar([i + width/2 for i in x], observed_values, width,
                      label='Observed', color='lightcoral', alpha=0.8)
        
        ax.set_xlabel('Hand Categories')
        ax.set_ylabel('Percentage (%)')
        ax.set_title('Hand Strength Distribution: Expected vs Observed')
        ax.set_xticks(x)
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add goodness of fit indicator
        goodness = distribution_data['goodness_of_fit']
        colors = {'excellent': 'green', 'good': 'blue', 'acceptable': 'orange', 'poor': 'red'}
        ax.text(0.02, 0.98, f'Goodness of Fit: {goodness.title()}', 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                color=colors.get(goodness, 'black'), verticalalignment='top')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, 'hand_distribution.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _plot_equity_curve(self, 
                          simulation_history: List[Tuple[int, float]],
                          equity_data: Dict[str, Any], 
                          output_dir: str) -> str:
        """Generate equity curve plot."""
        sample_points = [point[0] for point in simulation_history]
        equity_values = [point[1] for point in simulation_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Main equity curve
        ax1.plot(sample_points, equity_values, 'b-', linewidth=2, alpha=0.8)
        
        # Add confidence interval if available
        final_ci = equity_data.get('final_confidence_interval')
        if final_ci:
            ax1.axhline(y=final_ci[0], color='red', linestyle='--', alpha=0.5, label='95% CI Lower')
            ax1.axhline(y=final_ci[1], color='red', linestyle='--', alpha=0.5, label='95% CI Upper')
            ax1.fill_between(sample_points, final_ci[0], final_ci[1], alpha=0.1, color='red')
        
        # Mark convergence point if achieved
        if equity_data.get('convergence_achieved') and equity_data.get('convergence_point'):
            conv_point = equity_data['convergence_point']
            ax1.axvline(x=conv_point, color='green', linestyle=':', linewidth=2, label='Convergence Point')
        
        ax1.set_xlabel('Number of Simulations')
        ax1.set_ylabel('Equity')
        ax1.set_title('Equity Convergence Curve')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Running variance plot
        running_variance = equity_data.get('running_variance', [])
        if running_variance:
            ax2.plot(sample_points, running_variance, 'r-', linewidth=1.5, alpha=0.8)
            ax2.set_xlabel('Number of Simulations')
            ax2.set_ylabel('Running Variance')
            ax2.set_title('Convergence Variance')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = os.path.join(output_dir, 'equity_curve.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _plot_performance_metrics(self, performance_data: Dict[str, Any], output_dir: str) -> str:
        """Generate performance metrics visualization."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Simulation speed gauge
        speed = performance_data['simulations_per_second']
        speed_ranges = [0, 1000, 5000, 10000, 25000]
        speed_colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        
        for i in range(len(speed_ranges)-1):
            if speed_ranges[i] <= speed <= speed_ranges[i+1]:
                color = speed_colors[i]
                break
        else:
            color = 'green' if speed > speed_ranges[-1] else 'red'
        
        ax1.bar(['Simulations/Second'], [speed], color=color, alpha=0.7)
        ax1.set_title('Simulation Speed')
        ax1.set_ylabel('Simulations per Second')
        ax1.text(0, speed + speed*0.05, f'{speed:.0f}', ha='center', fontweight='bold')
        
        # Memory efficiency pie chart
        memory_efficiency = performance_data['memory_efficiency']
        memory_colors = {'excellent': 'green', 'good': 'lightgreen', 'acceptable': 'orange', 'poor': 'red'}
        
        ax2.pie([1], labels=[f'Memory: {memory_efficiency.title()}'], 
               colors=[memory_colors.get(memory_efficiency, 'gray')],
               autopct='', startangle=90)
        ax2.set_title('Memory Efficiency')
        
        # Accuracy vs Speed ratio
        accuracy_ratio = performance_data['accuracy_vs_speed_ratio']
        ax3.bar(['Accuracy/Speed Ratio'], [accuracy_ratio], color='skyblue', alpha=0.7)
        ax3.set_title('Accuracy vs Speed Trade-off')
        ax3.set_ylabel('Accuracy per Unit Time')
        ax3.text(0, accuracy_ratio + accuracy_ratio*0.05, f'{accuracy_ratio:.3f}', ha='center', fontweight='bold')
        
        # Optimization effectiveness (if available)
        opt_effectiveness = performance_data.get('optimization_effectiveness')
        if opt_effectiveness is not None:
            ax4.bar(['Optimization Effectiveness'], [opt_effectiveness * 100], color='lightcoral', alpha=0.7)
            ax4.set_title('Intelligent Optimization')
            ax4.set_ylabel('Effectiveness (%)')
            ax4.text(0, opt_effectiveness * 100 + 2, f'{opt_effectiveness:.1%}', ha='center', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Optimization\nNot Used', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14, color='gray')
            ax4.set_title('Intelligent Optimization')
        
        plt.tight_layout()
        filename = os.path.join(output_dir, 'performance_metrics.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def _export_report(self, report: Dict[str, Any], format_type: str, output_dir: str) -> str:
        """Export report in specified format."""
        timestamp = int(time.time())
        
        if format_type == 'json':
            filename = os.path.join(output_dir, f'poker_analysis_report_{timestamp}.json')
            self.report_generator.export_to_json(report, filename)
            return filename
        
        elif format_type == 'csv':
            filename = os.path.join(output_dir, f'poker_analysis_metrics_{timestamp}.csv')
            self.report_generator.export_to_csv(report, filename)
            return filename
        
        elif format_type == 'html':
            filename = os.path.join(output_dir, f'poker_analysis_report_{timestamp}.html')
            self._export_to_html(report, filename)
            return filename
        
        else:
            raise ValueError(f"Unsupported export format: {format_type}")
    
    def _export_to_html(self, report: Dict[str, Any], filename: str) -> None:
        """Export report as HTML."""
        html_content = self._generate_html_report(report)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_html_report(self, report: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Poker Knight Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; }
        h3 { color: #7f8c8d; }
        .metric { background: #ecf0f1; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .metric strong { color: #2c3e50; }
        .summary-box { background: #e8f4fd; border: 1px solid #3498db; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .recommendation { background: #fff3cd; border: 1px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .performance-good { color: #27ae60; font-weight: bold; }
        .performance-poor { color: #e74c3c; font-weight: bold; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #bdc3c7; padding: 12px; text-align: left; }
        th { background-color: #3498db; color: white; }
        .timestamp { color: #7f8c8d; font-size: 0.9em; }
    </style>
</head>
<body>
"""
        
        # Header
        html += f"""
    <h1>♞ Poker Knight Analysis Report</h1>
    <div class="timestamp">Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report.get('timestamp', time.time())))}</div>
    <div class="timestamp">Version: {report.get('version', '1.5.0')}</div>
"""
        
        # Summary section
        summary = report.get('summary', {})
        html += f"""
    <div class="summary-box">
        <h2>📊 Executive Summary</h2>
        <div class="metric"><strong>Total Simulations:</strong> {summary.get('total_simulations', 'N/A'):,}</div>
        <div class="metric"><strong>Win Probability:</strong> {summary.get('win_probability', 'N/A'):.4f}</div>
        <div class="metric"><strong>Simulation Time:</strong> {summary.get('simulation_time_seconds', 'N/A'):.2f} seconds</div>
        <div class="metric"><strong>Performance:</strong> {summary.get('simulations_per_second', 'N/A'):,.0f} simulations/second</div>
        <div class="metric"><strong>Accuracy Assessment:</strong> <span class="performance-good">{summary.get('accuracy_assessment', 'N/A').title()}</span></div>
        <div class="metric"><strong>Performance Assessment:</strong> <span class="performance-good">{summary.get('performance_assessment', 'N/A').title()}</span></div>
    </div>
"""
        
        # Add sections for each analysis type
        if 'variance_analysis' in report:
            html += self._add_variance_section_html(report['variance_analysis'])
        
        if 'hand_strength_distribution' in report:
            html += self._add_distribution_section_html(report['hand_strength_distribution'])
        
        if 'performance_metrics' in report:
            html += self._add_performance_section_html(report['performance_metrics'])
        
        if 'equity_curve' in report:
            html += self._add_equity_curve_section_html(report['equity_curve'])
        
        html += """
</body>
</html>
"""
        return html
    
    def _add_variance_section_html(self, variance_data: Dict[str, Any]) -> str:
        """Add variance analysis section to HTML."""
        ci = variance_data['confidence_interval_95']
        return f"""
    <h2>📈 Variance Analysis</h2>
    <div class="metric"><strong>Standard Deviation:</strong> {variance_data['standard_deviation']:.6f}</div>
    <div class="metric"><strong>Variance Ratio:</strong> {variance_data['variance_ratio']:.4f}</div>
    <div class="metric"><strong>95% Confidence Interval:</strong> ({ci[0]:.4f}, {ci[1]:.4f})</div>
    <div class="metric"><strong>Margin of Error:</strong> ±{variance_data['margin_of_error']:.4f}</div>
"""
    
    def _add_distribution_section_html(self, distribution_data: Dict[str, Any]) -> str:
        """Add hand distribution section to HTML."""
        goodness = distribution_data['goodness_of_fit']
        goodness_class = 'performance-good' if goodness in ['excellent', 'good'] else 'performance-poor'
        
        html = f"""
    <h2>🃏 Hand Strength Distribution</h2>
    <div class="metric"><strong>Chi-Square Statistic:</strong> {distribution_data['chi_square_statistic']:.4f}</div>
    <div class="metric"><strong>P-Value:</strong> {distribution_data['p_value']:.6f}</div>
    <div class="metric"><strong>Goodness of Fit:</strong> <span class="{goodness_class}">{goodness.title()}</span></div>
    
    <h3>Category Breakdown:</h3>
    <table>
        <tr><th>Hand Category</th><th>Observed %</th><th>Expected %</th><th>Difference</th></tr>
"""
        
        for category, (expected, observed) in distribution_data['expected_vs_observed'].items():
            difference = (observed - expected) * 100
            diff_sign = '+' if difference >= 0 else ''
            html += f"""
        <tr>
            <td>{category.replace('_', ' ').title()}</td>
            <td>{observed * 100:.3f}%</td>
            <td>{expected * 100:.3f}%</td>
            <td>{diff_sign}{difference:.3f}pp</td>
        </tr>
"""
        
        html += "    </table>"
        return html
    
    def _add_performance_section_html(self, performance_data: Dict[str, Any]) -> str:
        """Add performance metrics section to HTML."""
        return f"""
    <h2>⚡ Performance Metrics</h2>
    <div class="metric"><strong>Simulations per Second:</strong> {performance_data['simulations_per_second']:,.0f}</div>
    <div class="metric"><strong>Total Simulation Time:</strong> {performance_data['total_simulation_time']:.2f} seconds</div>
    <div class="metric"><strong>Memory Efficiency:</strong> {performance_data['memory_efficiency'].title()}</div>
    <div class="metric"><strong>Accuracy vs Speed Ratio:</strong> {performance_data['accuracy_vs_speed_ratio']:.6f}</div>
    {f'<div class="metric"><strong>Optimization Effectiveness:</strong> {performance_data["optimization_effectiveness"]:.1%}</div>' if performance_data.get('optimization_effectiveness') else '<div class="metric"><strong>Optimization:</strong> Not Used</div>'}
"""
    
    def _add_equity_curve_section_html(self, equity_data: Dict[str, Any]) -> str:
        """Add equity curve section to HTML."""
        convergence_status = "✅ Achieved" if equity_data['convergence_achieved'] else "❌ Not Achieved"
        ci = equity_data['final_confidence_interval']
        
        return f"""
    <h2>📊 Equity Curve Analysis</h2>
    <div class="metric"><strong>Convergence Status:</strong> {convergence_status}</div>
    {f'<div class="metric"><strong>Convergence Point:</strong> {equity_data["convergence_point"]:,} simulations</div>' if equity_data.get('convergence_point') else ''}
    <div class="metric"><strong>Final Confidence Interval:</strong> ({ci[0]:.4f}, {ci[1]:.4f})</div>
    <div class="metric"><strong>Convergence Rate Compliance:</strong> {equity_data['convergence_rate']:.3f}</div>
"""


# Public API
__all__ = [
    'PerformanceDashboard',
    'StatisticalReport',
    'ReportConfiguration'
] 