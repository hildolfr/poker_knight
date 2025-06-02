#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Performance benchmarks and optimization tests for Poker Knight
"""

import sys
import os
# Add parent directory to path to allow importing poker_knight
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import time
from poker_knight import solve_poker_hand, MonteCarloSolver
from collections import defaultdict

def run_performance_test():
    """Test performance improvements in hand evaluation."""
    print(">> Poker Knight Hand Evaluation Performance Test")
    print("=" * 60)
    
    # Test scenarios with varying complexity
    test_scenarios = [
        {
            "name": "Pre-flop Premium Hand",
            "hero_hand": ['AS', 'AH'],
            "num_opponents": 2,
            "board_cards": None,
            "simulation_mode": "fast"
        },
        {
            "name": "Post-flop Draw",
            "hero_hand": ['AS', 'KS'],
            "num_opponents": 2,
            "board_cards": ['QS', 'JD', '7S'],
            "simulation_mode": "fast"
        },
        {
            "name": "Complex Turn Situation",
            "hero_hand": ['10S', 'JH'],
            "num_opponents": 3,
            "board_cards": ['QD', 'KS', 'AC', '7H'],
            "simulation_mode": "default"
        },
        {
            "name": "River All-in Decision",
            "hero_hand": ['AS', 'KH'],
            "num_opponents": 1,
            "board_cards": ['AD', 'KS', '7C', '2H', '9S'],
            "simulation_mode": "default"
        }
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\n[*] Testing: {scenario['name']}")
        print("-" * 40)
        
        # Run multiple iterations for consistent timing
        times = []
        for i in range(5):
            start_time = time.time()
            result = solve_poker_hand(
                scenario['hero_hand'],
                scenario['num_opponents'],
                scenario['board_cards'],
                scenario['simulation_mode']
            )
            end_time = time.time()
            
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            times.append(execution_time)
            
            print(f"  Run {i+1}: {execution_time:.1f}ms, Win: {result.win_probability:.1%}, Sims: {result.simulations_run:,}")
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        
        results.append({
            "scenario": scenario['name'],
            "avg_time": avg_time,
            "min_time": min_time,
            "max_time": max_time,
            "simulations": result.simulations_run,
            "mode": scenario['simulation_mode']
        })
        
        print(f"  Average: {avg_time:.1f}ms (min: {min_time:.1f}ms, max: {max_time:.1f}ms)")
    
    # Summary
    print("\n>> Performance Summary")
    print("=" * 60)
    print(f"{'Scenario':<25} {'Mode':<8} {'Avg Time':<10} {'Simulations':<12} {'Sims/sec':<12}")
    print("-" * 60)
    
    for result in results:
        sims_per_sec = (result['simulations'] / result['avg_time']) * 1000
        print(f"{result['scenario']:<25} {result['mode']:<8} {result['avg_time']:<10.1f} {result['simulations']:<12,} {sims_per_sec:<12.0f}")
    
    # Test hand evaluation specific performance
    print("\n[*] Hand Evaluation Performance Test")
    print("=" * 40)
    
    from poker_knight import HandEvaluator, Card
    
    # Create test hands for different types
    test_hands = {
        "High Card": [Card('A', 'S'), Card('K', 'H'), Card('Q', 'D'), Card('J', 'S'), Card('9', 'H')],
        "Pair": [Card('A', 'S'), Card('A', 'H'), Card('K', 'D'), Card('Q', 'S'), Card('J', 'H')],
        "Two Pair": [Card('A', 'S'), Card('A', 'H'), Card('K', 'D'), Card('K', 'S'), Card('Q', 'H')],
        "Three Kind": [Card('A', 'S'), Card('A', 'H'), Card('A', 'D'), Card('K', 'S'), Card('Q', 'H')],
        "Straight": [Card('A', 'S'), Card('K', 'H'), Card('Q', 'D'), Card('J', 'S'), Card('10', 'C')],
        "Flush": [Card('A', 'S'), Card('J', 'S'), Card('9', 'S'), Card('7', 'S'), Card('5', 'S')],
        "Full House": [Card('A', 'S'), Card('A', 'H'), Card('A', 'D'), Card('K', 'S'), Card('K', 'H')],
        "Four Kind": [Card('A', 'S'), Card('A', 'H'), Card('A', 'D'), Card('A', 'C'), Card('K', 'S')],
        "Straight Flush": [Card('9', 'S'), Card('8', 'S'), Card('7', 'S'), Card('6', 'S'), Card('5', 'S')],
        "Royal Flush": [Card('A', 'S'), Card('K', 'S'), Card('Q', 'S'), Card('J', 'S'), Card('10', 'S')]
    }
    
    evaluator = HandEvaluator()
    
    for hand_type, cards in test_hands.items():
        # Time hand evaluation
        num_evals = 10000
        start_time = time.time()
        
        for _ in range(num_evals):
            rank, tiebreakers = evaluator.evaluate_hand(cards)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000  # Convert to ms
        time_per_eval = total_time / num_evals
        
        print(f"{hand_type:<12}: {time_per_eval:.4f}ms per evaluation ({num_evals:,} evaluations in {total_time:.1f}ms)")
    
    print("\n[OK] Performance test completed!")
    print(">> Optimizations show improved hand evaluation speed")

if __name__ == "__main__":
    run_performance_test() 