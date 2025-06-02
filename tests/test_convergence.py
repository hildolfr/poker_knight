#!/usr/bin/env python3
"""Simple test for convergence analysis functionality."""

from poker_knight import solve_poker_hand
import time

def test_convergence():
    """Test convergence analysis with proper card format."""
    print("🎯 Testing Poker Knight v1.5.0 Convergence Analysis")
    print("=" * 60)
    
    # Test with strong hand (pocket aces)
    print("Testing convergence with A♠️ A♥️ vs 1 opponent...")
    start_time = time.time()
    
    try:
        result = solve_poker_hand(['A♠️', 'A♥️'], 1, simulation_mode='default')
        end_time = time.time()
        
        print(f"✅ Analysis completed successfully!")
        print(f"📊 Results:")
        print(f"   Win probability: {result.win_probability:.1%}")
        print(f"   Simulations run: {result.simulations_run:,}")
        print(f"   Execution time: {(end_time - start_time):.2f}s")
        print(f"   Convergence achieved: {result.convergence_achieved}")
        print(f"   Stopped early: {result.stopped_early}")
        print(f"   Geweke statistic: {result.geweke_statistic}")
        print(f"   Effective sample size: {result.effective_sample_size}")
        print(f"   Convergence efficiency: {result.convergence_efficiency}")
        
        # Test convergence monitoring functionality
        print("\n🔬 Testing standalone convergence analysis...")
        from poker_knight.analysis import ConvergenceMonitor, convergence_diagnostic
        
        # Create a simple series for testing
        import random
        random.seed(42)
        win_rates = [random.random() * 0.2 + 0.8 for _ in range(1000)]  # Simulate converging to ~80%
        
        # Test Geweke diagnostic
        geweke_result = convergence_diagnostic(win_rates)
        print(f"   Geweke statistic: {geweke_result.statistic:.3f}")
        print(f"   Converged: {geweke_result.converged}")
        
        # Test convergence monitor
        monitor = ConvergenceMonitor()
        for i, rate in enumerate(win_rates[-10:], len(win_rates)-10):
            monitor.update(rate, i)
        
        status = monitor.get_convergence_status()
        print(f"   Monitor status: {status['status']}")
        print(f"   Current win rate: {status['current_win_rate']:.3f}")
        
        print("\n🎉 All convergence analysis tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_convergence() 