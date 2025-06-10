"""
Benchmarking framework for Poker Knight CUDA acceleration.

Provides comprehensive performance testing and comparison between
CPU and GPU implementations.
"""

import time
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from .gpu_solver import GPUSolver
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    GPUSolver = None

from ..solver import MonteCarloSolver
from ..constants import SUITS

class BenchmarkSuite:
    """Comprehensive benchmarking suite for CPU vs GPU performance."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark suite.
        
        Args:
            output_dir: Directory to save results (default: ./benchmarks)
        """
        self.output_dir = output_dir or Path("./benchmarks")
        self.output_dir.mkdir(exist_ok=True)
        
        self.cpu_solver = MonteCarloSolver()
        self.gpu_solver = GPUSolver() if CUDA_AVAILABLE else None
        
        # Test scenarios
        self.scenarios = self._define_scenarios()
        
    def _define_scenarios(self) -> List[Dict]:
        """Define benchmark scenarios."""
        scenarios = []
        
        # Pre-flop scenarios
        for num_opponents in [1, 2, 3, 4, 5, 6]:
            scenarios.append({
                'name': f'preflop_{num_opponents}opp',
                'hero_hand': ['A♠', 'K♠'],
                'num_opponents': num_opponents,
                'board_cards': [],
                'simulations': [10000, 50000, 100000, 500000]
            })
        
        # Post-flop scenarios
        boards = [
            ['Q♠', 'J♠', '10♥'],  # Royal draw
            ['7♠', '7♥', '2♦'],   # Paired board
            ['A♥', 'K♦', 'Q♣'],   # High cards
            ['9♠', '8♠', '7♦'],   # Straight draw
            ['K♠', '5♠', '2♠'],   # Flush draw
        ]
        
        for i, board in enumerate(boards):
            for num_opponents in [1, 3, 5]:
                scenarios.append({
                    'name': f'flop_{i}_{num_opponents}opp',
                    'hero_hand': ['A♠', 'A♥'],
                    'num_opponents': num_opponents,
                    'board_cards': board,
                    'simulations': [10000, 100000, 500000]
                })
        
        return scenarios
    
    def run_benchmark(self, scenario: Dict, backend: str) -> Dict:
        """
        Run a single benchmark scenario.
        
        Args:
            scenario: Benchmark scenario
            backend: 'cpu' or 'gpu'
            
        Returns:
            Benchmark results
        """
        solver = self.cpu_solver if backend == 'cpu' else self.gpu_solver
        
        if solver is None:
            return {'error': f'{backend} solver not available'}
        
        results = []
        
        for num_sims in scenario['simulations']:
            # Warm up
            solver.analyze_hand(
                scenario['hero_hand'],
                scenario['num_opponents'],
                scenario['board_cards'],
                num_simulations=1000
            )
            
            # Time multiple runs
            times = []
            for _ in range(5):
                start = time.perf_counter()
                
                result = solver.analyze_hand(
                    scenario['hero_hand'],
                    scenario['num_opponents'],
                    scenario['board_cards'],
                    num_simulations=num_sims
                )
                
                end = time.perf_counter()
                times.append(end - start)
            
            # Calculate statistics
            times = np.array(times)
            
            results.append({
                'simulations': num_sims,
                'mean_time': float(np.mean(times)),
                'std_time': float(np.std(times)),
                'min_time': float(np.min(times)),
                'max_time': float(np.max(times)),
                'throughput': num_sims / np.mean(times),
                'win_probability': result.win_probability
            })
        
        return {
            'backend': backend,
            'scenario': scenario['name'],
            'results': results
        }
    
    def run_all_benchmarks(self) -> Dict:
        """Run all benchmark scenarios on available backends."""
        all_results = {
            'timestamp': datetime.now().isoformat(),
            'system_info': self._get_system_info(),
            'benchmarks': []
        }
        
        total_scenarios = len(self.scenarios)
        backends = ['cpu']
        if CUDA_AVAILABLE:
            backends.append('gpu')
        
        for i, scenario in enumerate(self.scenarios):
            logger.info(f"Running scenario {i+1}/{total_scenarios}: {scenario['name']}")
            
            for backend in backends:
                logger.info(f"  Backend: {backend}")
                result = self.run_benchmark(scenario, backend)
                all_results['benchmarks'].append(result)
        
        # Calculate speedups
        self._calculate_speedups(all_results)
        
        # Save results
        output_file = self.output_dir / f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        return all_results
    
    def _calculate_speedups(self, results: Dict):
        """Calculate GPU speedup factors."""
        if not CUDA_AVAILABLE:
            return
        
        # Group by scenario
        scenarios = {}
        for bench in results['benchmarks']:
            key = bench['scenario']
            if key not in scenarios:
                scenarios[key] = {}
            scenarios[key][bench['backend']] = bench
        
        # Calculate speedups
        speedups = []
        for scenario_name, backends in scenarios.items():
            if 'cpu' in backends and 'gpu' in backends:
                cpu_results = backends['cpu']['results']
                gpu_results = backends['gpu']['results']
                
                for cpu_res, gpu_res in zip(cpu_results, gpu_results):
                    if cpu_res['simulations'] == gpu_res['simulations']:
                        speedup = cpu_res['mean_time'] / gpu_res['mean_time']
                        speedups.append({
                            'scenario': scenario_name,
                            'simulations': cpu_res['simulations'],
                            'speedup': speedup,
                            'cpu_throughput': cpu_res['throughput'],
                            'gpu_throughput': gpu_res['throughput']
                        })
        
        results['speedups'] = speedups
        
        # Summary statistics
        if speedups:
            all_speedups = [s['speedup'] for s in speedups]
            results['summary'] = {
                'mean_speedup': float(np.mean(all_speedups)),
                'median_speedup': float(np.median(all_speedups)),
                'min_speedup': float(np.min(all_speedups)),
                'max_speedup': float(np.max(all_speedups))
            }
    
    def _get_system_info(self) -> Dict:
        """Get system information for benchmark context."""
        import platform
        
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cuda_available': CUDA_AVAILABLE
        }
        
        if CUDA_AVAILABLE:
            device = cp.cuda.Device()
            props = device.attributes
            info['gpu'] = {
                'name': device.name,
                'compute_capability': f"{props['ComputeCapabilityMajor']}.{props['ComputeCapabilityMinor']}",
                'memory': device.mem_info[1],  # Total memory
                'multiprocessors': props['MultiProcessorCount']
            }
        
        return info
    
    def generate_report(self, results: Dict) -> str:
        """Generate a human-readable benchmark report."""
        report = []
        report.append("# Poker Knight CUDA Benchmark Report")
        report.append(f"\nGenerated: {results['timestamp']}")
        
        # System info
        report.append("\n## System Information")
        for key, value in results['system_info'].items():
            if isinstance(value, dict):
                report.append(f"\n### {key.upper()}")
                for k, v in value.items():
                    report.append(f"- {k}: {v}")
            else:
                report.append(f"- {key}: {value}")
        
        # Summary
        if 'summary' in results:
            report.append("\n## Performance Summary")
            report.append(f"- Mean speedup: {results['summary']['mean_speedup']:.2f}x")
            report.append(f"- Median speedup: {results['summary']['median_speedup']:.2f}x")
            report.append(f"- Range: {results['summary']['min_speedup']:.2f}x - {results['summary']['max_speedup']:.2f}x")
        
        # Detailed results
        report.append("\n## Detailed Results")
        
        # Group by scenario
        scenarios = {}
        for bench in results['benchmarks']:
            key = bench['scenario']
            if key not in scenarios:
                scenarios[key] = {}
            scenarios[key][bench['backend']] = bench
        
        for scenario_name, backends in scenarios.items():
            report.append(f"\n### {scenario_name}")
            
            # Create comparison table
            if 'cpu' in backends and 'gpu' in backends:
                report.append("\n| Simulations | CPU Time (s) | GPU Time (s) | Speedup |")
                report.append("|-------------|--------------|--------------|---------|")
                
                cpu_results = backends['cpu']['results']
                gpu_results = backends['gpu']['results']
                
                for cpu_res, gpu_res in zip(cpu_results, gpu_results):
                    speedup = cpu_res['mean_time'] / gpu_res['mean_time']
                    report.append(
                        f"| {cpu_res['simulations']:,} | "
                        f"{cpu_res['mean_time']:.3f} | "
                        f"{gpu_res['mean_time']:.3f} | "
                        f"{speedup:.2f}x |"
                    )
        
        return "\n".join(report)

def run_quick_benchmark() -> Dict:
    """Run a quick benchmark for testing."""
    suite = BenchmarkSuite()
    
    # Quick scenario
    scenario = {
        'name': 'quick_test',
        'hero_hand': ['A♠', 'K♠'],
        'num_opponents': 2,
        'board_cards': ['Q♠', 'J♠', '10♥'],
        'simulations': [10000, 100000]
    }
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': []
    }
    
    for backend in ['cpu', 'gpu']:
        if backend == 'gpu' and not CUDA_AVAILABLE:
            continue
            
        result = suite.run_benchmark(scenario, backend)
        results['benchmarks'].append(result)
    
    return results

if __name__ == "__main__":
    # Run benchmarks when executed directly
    logging.basicConfig(level=logging.INFO)
    
    if CUDA_AVAILABLE:
        logger.info("Running full benchmark suite...")
        suite = BenchmarkSuite()
        results = suite.run_all_benchmarks()
        
        # Generate and save report
        report = suite.generate_report(results)
        report_file = suite.output_dir / f"report_{datetime.now():%Y%m%d_%H%M%S}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_file}")
    else:
        logger.warning("CUDA not available. Install CuPy to run GPU benchmarks.")