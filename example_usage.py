#!/usr/bin/env python3
"""
Poker Knight - Example Usage
Demonstrates various scenarios and features of the Poker Knight Monte Carlo solver.
"""

from poker_solver import solve_poker_hand, MonteCarloSolver
import time

def print_result(description, result):
    """Pretty print a simulation result."""
    print(f"\n{description}")
    print("=" * len(description))
    print(f"Win Probability: {result.win_probability:.1%}")
    print(f"Tie Probability: {result.tie_probability:.1%}")
    print(f"Loss Probability: {result.loss_probability:.1%}")
    print(f"Simulations Run: {result.simulations_run:,}")
    print(f"Execution Time: {result.execution_time_ms:.1f}ms")
    
    if result.confidence_interval:
        lower, upper = result.confidence_interval
        print(f"95% Confidence Interval: [{lower:.1%}, {upper:.1%}]")
    
    if result.hand_category_frequencies:
        print("\nHand Categories:")
        for category, frequency in sorted(result.hand_category_frequencies.items(), 
                                        key=lambda x: x[1], reverse=True):
            if frequency > 0.01:  # Only show categories > 1%
                print(f"  {category.replace('_', ' ').title()}: {frequency:.1%}")

def main():
    """Run example scenarios."""
    print("♞ Poker Knight v1.2.1 - Example Usage")
    print("=" * 50)
    
    # Example 1: Premium hand pre-flop
    print("\n🔥 Example 1: Pocket Aces vs 1 Opponent (Pre-flop)")
    result = solve_poker_hand(['A♠️', 'A♥️'], 1, simulation_mode="default")
    print_result("Pocket Aces vs 1 Opponent", result)
    
    # Example 2: Premium hand vs multiple opponents
    print("\n🔥 Example 2: Pocket Aces vs 5 Opponents (Pre-flop)")
    result = solve_poker_hand(['A♠️', 'A♥️'], 5, simulation_mode="default")
    print_result("Pocket Aces vs 5 Opponents", result)
    
    # Example 3: Medium strength hand
    print("\n🔥 Example 3: Ace-King Suited vs 2 Opponents (Pre-flop)")
    result = solve_poker_hand(['A♠️', 'K♠️'], 2, simulation_mode="default")
    print_result("Ace-King Suited vs 2 Opponents", result)
    
    # Example 4: Weak hand
    print("\n🔥 Example 4: 2-7 Offsuit vs 1 Opponent (Pre-flop)")
    result = solve_poker_hand(['2♠️', '7♥️'], 1, simulation_mode="default")
    print_result("2-7 Offsuit vs 1 Opponent", result)
    
    # Example 5: Post-flop with strong made hand
    print("\n🔥 Example 5: Pocket Aces with Set on Flop")
    result = solve_poker_hand(
        ['A♠️', 'A♥️'], 
        2, 
        ['A♦️', '7♠️', '2♣️'],  # Flop gives us trip aces
        simulation_mode="default"
    )
    print_result("Trip Aces on Flop vs 2 Opponents", result)
    
    # Example 6: Post-flop with draw
    print("\n🔥 Example 6: Flush Draw on Flop")
    result = solve_poker_hand(
        ['A♠️', 'K♠️'], 
        1, 
        ['Q♠️', 'J♦️', '7♠️'],  # Flop gives us nut flush draw + straight draw
        simulation_mode="default"
    )
    print_result("Nut Flush Draw + Straight Draw vs 1 Opponent", result)
    
    # Example 7: Turn situation
    print("\n🔥 Example 7: Made Straight on Turn")
    result = solve_poker_hand(
        ['10♠️', 'J♥️'], 
        3, 
        ['Q♦️', 'K♠️', 'A♣️', '7♥️'],  # Turn completes our straight
        simulation_mode="default"
    )
    print_result("Broadway Straight on Turn vs 3 Opponents", result)
    
    # Example 8: River situation
    print("\n🔥 Example 8: Two Pair on River")
    result = solve_poker_hand(
        ['A♠️', 'K♥️'], 
        2, 
        ['A♦️', 'K♠️', '7♣️', '2♥️', '9♠️'],  # River gives us two pair
        simulation_mode="default"
    )
    print_result("Aces and Kings on River vs 2 Opponents", result)
    
    # Performance comparison
    print("\n🔥 Performance Comparison: Different Simulation Modes")
    print("=" * 55)
    
    hand = ['Q♠️', 'Q♥️']
    opponents = 3
    
    # Fast mode
    start_time = time.time()
    fast_result = solve_poker_hand(hand, opponents, simulation_mode="fast")
    fast_time = time.time() - start_time
    
    # Default mode
    start_time = time.time()
    default_result = solve_poker_hand(hand, opponents, simulation_mode="default")
    default_time = time.time() - start_time
    
    # Precision mode
    start_time = time.time()
    precision_result = solve_poker_hand(hand, opponents, simulation_mode="precision")
    precision_time = time.time() - start_time
    
    print(f"\nPocket Queens vs {opponents} Opponents:")
    print(f"Fast Mode:      {fast_result.win_probability:.1%} ({fast_result.simulations_run:,} sims, {fast_time*1000:.1f}ms)")
    print(f"Default Mode:   {default_result.win_probability:.1%} ({default_result.simulations_run:,} sims, {default_time*1000:.1f}ms)")
    print(f"Precision Mode: {precision_result.win_probability:.1%} ({precision_result.simulations_run:,} sims, {precision_time*1000:.1f}ms)")
    
    # Advanced usage with custom solver
    print("\n🔥 Advanced Usage: Custom Solver Configuration")
    print("=" * 50)
    
    solver = MonteCarloSolver()
    
    # Analyze the same hand multiple times to show consistency
    results = []
    for i in range(3):
        result = solver.analyze_hand(['K♠️', 'K♥️'], 2, simulation_mode="default")
        results.append(result.win_probability)
    
    avg_win_rate = sum(results) / len(results)
    variance = sum((x - avg_win_rate) ** 2 for x in results) / len(results)
    
    print(f"\nPocket Kings vs 2 Opponents (3 runs):")
    print(f"Win rates: {[f'{r:.1%}' for r in results]}")
    print(f"Average: {avg_win_rate:.1%}")
    print(f"Standard deviation: {variance**0.5:.1%}")
    
    print("\n🎯 Summary")
    print("=" * 20)
    print("Poker Knight provides:")
    print("• Fast, accurate Monte Carlo simulations")
    print("• Support for pre-flop, flop, turn, and river analysis")
    print("• Configurable simulation depth and performance settings")
    print("• Detailed statistics including confidence intervals")
    print("• Card removal effects for accurate probability calculation")
    print("• Clean API for integration into larger poker AI systems")
    
    print("\n♞ Poker Knight v1.2.1 - Empowering AI poker players with precise, fast hand analysis.")

if __name__ == "__main__":
    main() 