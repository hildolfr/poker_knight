#!/usr/bin/env python3
"""
♞ Poker Knight Analytics Dashboard Example

Comprehensive demonstration of the analytics and reporting capabilities
introduced in v1.5.0. Shows how to generate detailed statistical reports,
performance dashboards, and visualizations.

This example demonstrates:
- Advanced statistical analysis with PokerAnalytics
- Performance dashboard generation
- Comprehensive report creation with visualizations
- Multiple export formats (JSON, CSV, HTML)
- Optimization effectiveness analysis

Author: hildolfr
License: MIT
"""

import time
import os
import sys
from typing import List, Dict, Any, Tuple

# Add the parent directory to the path so we can import poker_knight
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from poker_knight.solver import MonteCarloSolver
from poker_knight.analytics import PokerAnalytics, StatisticalReportGenerator
from poker_knight.reporting import PerformanceDashboard, StatisticalReport, ReportConfiguration


def run_sample_simulations() -> List[Tuple[Any, float, str, List[Tuple[int, float]]]]:
    """
    Run a variety of sample simulations to demonstrate analytics capabilities.
    
    Returns:
        List of (result, simulation_time, name, history) tuples
    """
    print("🎯 Running sample simulations for analytics demonstration...")
    
    # Initialize solver with different configurations
    solver_standard = MonteCarloSolver("config.json")
    solver_optimized = MonteCarloSolver("config.json")
    
    # Sample hands to analyze
    sample_scenarios = [
        {
            'name': 'Premium_Pocket_Aces',
            'hole_cards': ['A♠️', 'A♥️'],
            'community_cards': ['K♦️', '7♣️', '2♠️'],
            'num_opponents': 3,
            'mode': 'fast',
            'optimization': False
        },
        {
            'name': 'Drawing_Hand_Flush_Draw',
            'hole_cards': ['Q♠️', 'J♠️'],
            'community_cards': ['10♠️', '9♦️', '3♠️'],
            'num_opponents': 2,
            'mode': 'default',
            'optimization': True
        },
        {
            'name': 'Marginal_Hand_Small_Pair',
            'hole_cards': ['6♥️', '6♣️'],
            'community_cards': ['A♦️', 'K♠️', 'Q♥️'],
            'num_opponents': 4,
            'mode': 'default',
            'optimization': True
        },
        {
            'name': 'Bluff_Catcher_High_Card',
            'hole_cards': ['A♦️', 'Q♣️'],
            'community_cards': ['J♠️', '8♥️', '4♣️'],
            'num_opponents': 1,
            'mode': 'precision',
            'optimization': False
        },
        {
            'name': 'Multiway_Medium_Pair',
            'hole_cards': ['9♠️', '9♥️'],
            'community_cards': ['A♣️', '9♦️', '5♠️'],
            'num_opponents': 5,
            'mode': 'fast',
            'optimization': True
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(sample_scenarios):
        print(f"  📊 Running simulation {i+1}/5: {scenario['name']}")
        
        # Choose solver based on optimization setting
        solver = solver_optimized if scenario['optimization'] else solver_standard
        
        # Track simulation history for equity curve analysis
        simulation_history = []
        
        # Custom simulation with history tracking (simplified)
        start_time = time.time()
        
        try:
            result = solver.analyze_hand(
                hole_cards=scenario['hole_cards'],
                community_cards=scenario['community_cards'],
                num_opponents=scenario['num_opponents'],
                simulation_mode=scenario['mode'],
                intelligent_optimization=scenario['optimization']
            )
            
            simulation_time = time.time() - start_time
            
            # Generate simulated history for demonstration
            # In a real implementation, this would be captured during simulation
            total_sims = getattr(result, 'total_simulations', 10000)
            win_prob = getattr(result, 'win_probability', 0.5)
            
            # Create realistic simulation history
            for j in range(1, min(100, total_sims // 100) + 1):
                sim_count = (total_sims * j) // 100
                # Add some realistic variance to the equity progression
                noise = (0.5 - j/100) * 0.02 * ((-1) ** j)  # Decreasing noise over time
                equity_estimate = win_prob + noise
                equity_estimate = max(0.0, min(1.0, equity_estimate))  # Clamp to [0, 1]
                simulation_history.append((sim_count, equity_estimate))
            
            results.append((result, simulation_time, scenario['name'], simulation_history))
            
            print(f"    ✅ Completed: {result.win_probability:.3f} equity in {simulation_time:.2f}s")
            
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            continue
    
    print(f"✅ Completed {len(results)} simulations\n")
    return results


def demonstrate_analytics_engine():
    """Demonstrate the core analytics engine capabilities."""
    print("🔬 ANALYTICS ENGINE DEMONSTRATION")
    print("=" * 50)
    
    # Initialize analytics engine
    analytics = PokerAnalytics(confidence_level=0.95)
    
    # Sample simulation data
    print("📊 Analyzing variance for sample simulation...")
    win_probability = 0.634
    num_simulations = 50000
    observed_wins = 31845
    
    # Perform variance analysis
    variance_analysis = analytics.analyze_variance(win_probability, num_simulations, observed_wins)
    
    print(f"  Sample Variance: {variance_analysis.sample_variance:.8f}")
    print(f"  Standard Deviation: {variance_analysis.standard_deviation:.6f}")
    print(f"  Variance Ratio: {variance_analysis.variance_ratio:.4f}")
    print(f"  95% CI: ({variance_analysis.confidence_interval_95[0]:.4f}, {variance_analysis.confidence_interval_95[1]:.4f})")
    print(f"  Margin of Error: ±{variance_analysis.margin_of_error:.4f}")
    
    # Sample hand distribution analysis
    print("\n🃏 Analyzing hand strength distribution...")
    sample_hand_categories = {
        'high_card': 25089,
        'pair': 21128,
        'two_pair': 2377,
        'three_of_a_kind': 1056,
        'straight': 196,
        'flush': 98,
        'full_house': 72,
        'four_of_a_kind': 12,
        'straight_flush': 1,
        'royal_flush': 0
    }
    
    distribution_analysis = analytics.analyze_hand_strength_distribution(
        sample_hand_categories, num_simulations
    )
    
    print(f"  Chi-Square Statistic: {distribution_analysis.chi_square_statistic:.4f}")
    print(f"  P-Value: {distribution_analysis.p_value:.6f}")
    print(f"  Goodness of Fit: {distribution_analysis.goodness_of_fit}")
    
    print("  Top 3 Categories:")
    for category, (expected, observed) in list(distribution_analysis.expected_vs_observed.items())[:3]:
        print(f"    {category.replace('_', ' ').title()}: {observed:.1%} observed vs {expected:.1%} expected")
    
    # Sample equity curve analysis
    print("\n📈 Analyzing equity curve convergence...")
    sample_history = [
        (1000, 0.645), (2000, 0.638), (5000, 0.642), (10000, 0.635),
        (15000, 0.637), (20000, 0.634), (30000, 0.635), (40000, 0.634),
        (50000, 0.634)
    ]
    
    equity_curve = analytics.generate_equity_curve(sample_history, target_accuracy=0.01)
    
    print(f"  Convergence Achieved: {'Yes' if equity_curve.convergence_achieved else 'No'}")
    if equity_curve.convergence_point:
        print(f"  Convergence Point: {equity_curve.convergence_point:,} simulations")
    print(f"  Convergence Rate: {equity_curve.convergence_rate:.3f}")
    print(f"  Final CI: ({equity_curve.final_confidence_interval[0]:.4f}, {equity_curve.final_confidence_interval[1]:.4f})")
    
    print("\n" + "=" * 50 + "\n")


def demonstrate_performance_dashboard(simulation_results: List[Tuple[Any, float, str, List[Tuple[int, float]]]]):
    """Demonstrate the performance dashboard capabilities."""
    print("📈 PERFORMANCE DASHBOARD DEMONSTRATION")
    print("=" * 50)
    
    # Extract data for dashboard
    results = [item[0] for item in simulation_results]
    times = [item[1] for item in simulation_results]
    names = [item[2] for item in simulation_results]
    
    # Initialize dashboard
    dashboard = PerformanceDashboard(ReportConfiguration(confidence_level=0.95))
    
    # Generate performance summary
    print("🔍 Generating performance summary...")
    performance_summary = dashboard.generate_performance_summary(results, times, names)
    
    # Display key insights
    print("\n📊 Performance Summary:")
    print(f"  Total Simulations Analyzed: {performance_summary['total_simulations_analyzed']}")
    
    # Best performers
    best_performers = performance_summary['comparative_analysis']['best_performers']
    print(f"\n🏆 Best Performers:")
    print(f"  Fastest: {best_performers['fastest_simulation']['name']} ({best_performers['fastest_simulation']['simulations_per_second']:,.0f} sims/sec)")
    print(f"  Most Efficient: {best_performers['most_efficient']['name']} (ratio: {best_performers['most_efficient']['accuracy_vs_speed_ratio']:.6f})")
    
    # Performance statistics
    perf_stats = performance_summary['comparative_analysis']['performance_statistics']
    print(f"\n📈 Performance Statistics:")
    print(f"  Average Speed: {perf_stats['average_simulations_per_second']:,.0f} sims/sec")
    print(f"  Average Time: {perf_stats['average_execution_time']:.2f} seconds")
    print(f"  Performance Consistency: {perf_stats['performance_consistency']:.1%}")
    
    # Optimization insights
    if 'optimization_insights' in performance_summary:
        opt_insights = performance_summary['optimization_insights']
        usage = opt_insights['optimization_usage']
        print(f"\n🚀 Optimization Insights:")
        print(f"  Optimization Adoption Rate: {usage['optimization_adoption_rate']:.1%}")
        print(f"  Optimized Simulations: {usage['optimized_simulations']}/{usage['total_simulations']}")
        
        if 'performance_comparison' in opt_insights:
            perf_comp = opt_insights['performance_comparison']
            print(f"  Optimization Speedup: {perf_comp['optimization_speedup']:.1f}x")
            print(f"  Performance Benefit: {perf_comp['optimization_benefit']:.1%}")
    
    # Recommendations
    print(f"\n💡 Recommendations ({len(performance_summary['recommendations'])} items):")
    for rec in performance_summary['recommendations'][:3]:  # Show top 3
        priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
        print(f"  {priority_emoji} {rec['title']}: {rec['description'][:80]}...")
    
    print("\n" + "=" * 50 + "\n")
    
    return performance_summary


def demonstrate_comprehensive_reporting(simulation_results: List[Tuple[Any, float, str, List[Tuple[int, float]]]]):
    """Demonstrate comprehensive reporting with visualizations."""
    print("📋 COMPREHENSIVE REPORTING DEMONSTRATION")
    print("=" * 50)
    
    # Configure report settings
    config = ReportConfiguration(
        include_variance_analysis=True,
        include_distribution_analysis=True,
        include_equity_curve=True,
        include_performance_metrics=True,
        include_visualizations=True,
        export_formats=['json', 'csv', 'html'],
        confidence_level=0.95
    )
    
    # Initialize report generator
    report_generator = StatisticalReport(config)
    
    # Create output directory
    output_dir = "analytics_demo_reports"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"📁 Creating reports in: {output_dir}")
    
    # Generate reports for each simulation
    for i, (result, sim_time, name, history) in enumerate(simulation_results[:2]):  # Limit to 2 for demo
        print(f"\n🔍 Generating comprehensive report for: {name}")
        
        try:
            # Generate full analysis report
            report = report_generator.generate_full_analysis_report(
                simulation_result=result,
                simulation_time=sim_time,
                simulation_history=history,
                output_directory=os.path.join(output_dir, name.lower())
            )
            
            # Display key metrics from report
            summary = report.get('summary', {})
            print(f"  📊 Total Simulations: {summary.get('total_simulations', 'N/A'):,}")
            print(f"  📊 Win Probability: {summary.get('win_probability', 'N/A'):.4f}")
            print(f"  📊 Accuracy Assessment: {summary.get('accuracy_assessment', 'N/A').title()}")
            print(f"  📊 Performance Assessment: {summary.get('performance_assessment', 'N/A').title()}")
            
            # Show exported files
            exported_files = report.get('exported_files', {})
            print(f"  📄 Exported Files:")
            for format_type, filename in exported_files.items():
                if not format_type.endswith('_error'):
                    print(f"    - {format_type.upper()}: {filename}")
            
            # Show visualizations
            visualizations = report.get('visualizations', {})
            if visualizations:
                print(f"  📈 Visualizations Generated:")
                for viz_type, filename in visualizations.items():
                    print(f"    - {viz_type.replace('_', ' ').title()}: {filename}")
            
        except Exception as e:
            print(f"  ❌ Report generation failed: {str(e)}")
    
    print(f"\n✅ Reports generated successfully in: {output_dir}")
    print("\n" + "=" * 50 + "\n")


def demonstrate_statistical_report_generator():
    """Demonstrate the statistical report generator directly."""
    print("📊 STATISTICAL REPORT GENERATOR DEMONSTRATION")
    print("=" * 50)
    
    # Initialize components
    analytics = PokerAnalytics()
    report_generator = StatisticalReportGenerator(analytics)
    
    # Mock simulation result for demonstration
    class MockSimulationResult:
        def __init__(self):
            self.total_simulations = 25000
            self.win_probability = 0.742
            self.tie_probability = 0.023
            self.confidence_interval = [0.735, 0.749]
            self.hand_categories = {
                'high_card': 12540,
                'pair': 10565,
                'two_pair': 1188,
                'three_of_a_kind': 527,
                'straight': 98,
                'flush': 49,
                'full_house': 36,
                'four_of_a_kind': 6,
                'straight_flush': 1,
                'royal_flush': 0
            }
    
    mock_result = MockSimulationResult()
    mock_time = 2.45
    mock_history = [
        (1000, 0.755), (5000, 0.748), (10000, 0.744), (15000, 0.743),
        (20000, 0.742), (25000, 0.742)
    ]
    
    print("🔍 Generating comprehensive statistical report...")
    
    # Generate report
    report = report_generator.generate_comprehensive_report(
        simulation_result=mock_result,
        simulation_time=mock_time,
        simulation_history=mock_history
    )
    
    # Display report summary
    print("\n📊 Report Summary:")
    summary = report['summary']
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key.replace('_', ' ').title()}: {value:.4f}")
        else:
            print(f"  {key.replace('_', ' ').title()}: {value}")
    
    # Show variance analysis highlights
    if 'variance_analysis' in report:
        var_analysis = report['variance_analysis']
        print(f"\n📈 Variance Analysis Highlights:")
        print(f"  Standard Deviation: {var_analysis['standard_deviation']:.6f}")
        print(f"  Variance Ratio: {var_analysis['variance_ratio']:.4f}")
        ci = var_analysis['confidence_interval_95']
        print(f"  95% Confidence Interval: ({ci[0]:.4f}, {ci[1]:.4f})")
    
    # Show distribution analysis highlights
    if 'hand_strength_distribution' in report:
        dist_analysis = report['hand_strength_distribution']
        print(f"\n🃏 Distribution Analysis Highlights:")
        print(f"  Chi-Square Statistic: {dist_analysis['chi_square_statistic']:.4f}")
        print(f"  Goodness of Fit: {dist_analysis['goodness_of_fit'].title()}")
    
    # Show equity curve highlights
    if 'equity_curve' in report:
        equity_analysis = report['equity_curve']
        print(f"\n📊 Equity Curve Highlights:")
        print(f"  Convergence Achieved: {'Yes' if equity_analysis['convergence_achieved'] else 'No'}")
        if equity_analysis.get('convergence_point'):
            print(f"  Convergence Point: {equity_analysis['convergence_point']:,} simulations")
    
    # Export sample reports
    print(f"\n💾 Exporting sample reports...")
    os.makedirs("sample_reports", exist_ok=True)
    
    try:
        json_file = "sample_reports/sample_analysis.json"
        report_generator.export_to_json(report, json_file)
        print(f"  ✅ JSON exported: {json_file}")
        
        csv_file = "sample_reports/sample_metrics.csv"
        report_generator.export_to_csv(report, csv_file)
        print(f"  ✅ CSV exported: {csv_file}")
        
    except Exception as e:
        print(f"  ❌ Export failed: {str(e)}")
    
    print("\n" + "=" * 50 + "\n")


def main():
    """Main demonstration function."""
    print("♞ POKER KNIGHT ANALYTICS & REPORTING DASHBOARD")
    print("🚀 v1.5.0 Feature Demonstration")
    print("=" * 60)
    print()
    
    try:
        # 1. Demonstrate core analytics engine
        demonstrate_analytics_engine()
        
        # 2. Run sample simulations
        simulation_results = run_sample_simulations()
        
        if not simulation_results:
            print("❌ No simulation results available for dashboard demonstration")
            return
        
        # 3. Demonstrate performance dashboard
        performance_summary = demonstrate_performance_dashboard(simulation_results)
        
        # 4. Demonstrate comprehensive reporting
        demonstrate_comprehensive_reporting(simulation_results)
        
        # 5. Demonstrate statistical report generator
        demonstrate_statistical_report_generator()
        
        # Summary
        print("🎉 DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("✅ All analytics and reporting features demonstrated successfully!")
        print("\n📁 Generated Files:")
        print("  - analytics_demo_reports/ (comprehensive reports with visualizations)")
        print("  - sample_reports/ (JSON and CSV exports)")
        print("\n🔗 Key Features Demonstrated:")
        print("  ✅ Advanced statistical analysis (variance, distribution, equity curves)")
        print("  ✅ Performance dashboard with optimization insights")
        print("  ✅ Comprehensive reporting with visualizations")
        print("  ✅ Multiple export formats (JSON, CSV, HTML)")
        print("  ✅ Real-time performance monitoring and recommendations")
        print("\n💡 Next Steps:")
        print("  - Integrate analytics into your poker analysis workflow")
        print("  - Customize report configurations for your needs")
        print("  - Use performance dashboard to optimize simulation settings")
        print("  - Export data for external analysis tools")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 