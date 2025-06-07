#!/usr/bin/env python3
"""Debug confidence interval test issue."""

from poker_knight import solve_poker_hand, MonteCarloSolver

def main():
    # Create solver without caching to ensure fresh simulations
    solver = MonteCarloSolver(enable_caching=False)
    
    hero_hand = ['A♠', 'A♥']  # Pocket aces
    num_opponents = 1
    true_win_rate = 0.849
    
    print("Testing confidence interval generation...")
    
    # Test a few runs
    for i in range(5):
        result = solver.analyze_hand(hero_hand, num_opponents, simulation_mode="default")
        print(f"\nRun {i+1}:")
        print(f"  Win probability: {result.win_probability:.3f}")
        print(f"  Confidence interval: {result.confidence_interval}")
        print(f"  Simulations run: {result.simulations_run}")
        
        if result.confidence_interval:
            lower, upper = result.confidence_interval
            contains_true = lower <= true_win_rate <= upper
            print(f"  Contains true value ({true_win_rate:.3f})? {contains_true}")
    
    # Also test with the solve_poker_hand function
    print("\n\nTesting with solve_poker_hand function:")
    for i in range(3):
        result = solve_poker_hand(hero_hand, num_opponents, simulation_mode="default")
        print(f"\nRun {i+1}:")
        print(f"  Win probability: {result.win_probability:.3f}")
        print(f"  Confidence interval: {result.confidence_interval}")
        
if __name__ == "__main__":
    main()