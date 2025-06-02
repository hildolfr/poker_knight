#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration of the New Cache Pre-Population System

This demonstrates the improved cache system that replaces background warming
with one-time pre-population for better script usage patterns.

Key improvements:
- One-time cache population instead of background threads
- Predictable startup cost, then instant performance
- Script-friendly with user control options
- Targets 95-100% cache hit rate for common scenarios

Author: hildolfr
License: MIT
"""

import os
import sys
import time
from pathlib import Path

# Add poker_knight to path
if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).parent))

try:
    from poker_knight import MonteCarloSolver, solve_poker_hand
    from poker_knight.storage.cache_prepopulation import (
        PopulationConfig, ScenarioGenerator, CachePrePopulator,
        ensure_cache_populated
    )
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"[FAIL] Cache pre-population dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False


def demonstrate_configuration_options():
    """Demonstrate different cache pre-population configuration options."""
    print("[FIX] Cache Pre-Population Configuration Options")
    print("=" * 60)
    
    # Default configuration (all 169 hands)
    config_all = PopulationConfig()
    print(f"Default config - Preflop hands: {config_all.preflop_hands}")
    print(f"Opponent counts: {config_all.opponent_counts}")
    print(f"Positions: {config_all.positions}")
    print(f"Board patterns: {config_all.board_patterns}")
    
    # Premium hands only (faster population)
    config_premium = PopulationConfig(
        preflop_hands="premium_only",
        max_population_time_minutes=2
    )
    print(f"\nPremium-only config - Hands: {config_premium.preflop_hands}")
    print(f"Max population time: {config_premium.max_population_time_minutes} minutes")
    
    # User control options
    config_skip = PopulationConfig(skip_cache_warming=True)
    config_force = PopulationConfig(force_cache_regeneration=True)
    
    print(f"\nUser control options:")
    print(f"  Skip warming: {config_skip.skip_cache_warming}")
    print(f"  Force regeneration: {config_force.force_cache_regeneration}")
    
    print("[PASS] Configuration options demonstration completed!")


def demonstrate_scenario_generation():
    """Demonstrate the comprehensive scenario generation."""
    print("\n📝 Scenario Generation for Cache Population")
    print("=" * 60)
    
    # Test different coverage levels
    configs = [
        ("Premium Only", PopulationConfig(preflop_hands="premium_only")),
        ("Common Hands", PopulationConfig(preflop_hands="common_only")),
        ("All 169 Hands", PopulationConfig(preflop_hands="all_169"))
    ]
    
    for name, config in configs:
        generator = ScenarioGenerator(config)
        scenarios = generator.generate_all_scenarios()
        
        preflop_scenarios = [s for s in scenarios if s['type'] == 'preflop']
        board_scenarios = [s for s in scenarios if s['type'] == 'board_texture']
        
        print(f"\n{name}:")
        print(f"  Total scenarios: {len(scenarios)}")
        print(f"  Preflop scenarios: {len(preflop_scenarios)}")
        print(f"  Board texture scenarios: {len(board_scenarios)}")
        
        # Show example scenarios
        print("  Example preflop scenarios:")
        for scenario in preflop_scenarios[:3]:
            print(f"    {scenario['scenario_id']}")
        
        if board_scenarios:
            print("  Example board scenarios:")
            for scenario in board_scenarios[:2]:
                print(f"    {scenario['scenario_id']}")
    
    print("\n[PASS] Scenario generation demonstration completed!")


def demonstrate_solver_integration():
    """Demonstrate the new solver integration with cache pre-population."""
    print("\n🔗 Solver Integration with Cache Pre-Population")
    print("=" * 60)
    
    print("Creating solver with different cache options...")
    
    # Option 1: Default behavior (auto-populate if needed)
    print("\n1. Default solver (auto-populate cache if needed):")
    try:
        solver1 = MonteCarloSolver(enable_caching=True)
        print("   [PASS] Solver created with auto cache population")
        
        # Test a query
        print("   Testing query: AA vs 2 opponents...")
        start_time = time.time()
        result1 = solver1.analyze_hand(["AS", "AH"], 2, simulation_mode="fast")
        query_time = time.time() - start_time
        
        print(f"   Result: {result1.win_probability:.1%} win rate in {query_time:.3f}s")
        
        # Get population stats if available
        if hasattr(solver1, '_population_stats') and solver1._population_stats:
            stats = solver1._population_stats
            print(f"   Population stats: {stats.populated_scenarios} scenarios populated")
            print(f"   Coverage: {stats.coverage_after:.1%}")
        
        solver1.close()
        
    except Exception as e:
        print(f"   [WARN] Default solver test failed: {e}")
    
    # Option 2: Skip cache warming entirely
    print("\n2. Solver with cache warming disabled:")
    try:
        solver2 = MonteCarloSolver(enable_caching=True, skip_cache_warming=True)
        print("   [PASS] Solver created with cache warming disabled")
        
        # Test the same query
        print("   Testing same query without cache warming...")
        start_time = time.time()
        result2 = solver2.analyze_hand(["AS", "AH"], 2, simulation_mode="fast")
        query_time = time.time() - start_time
        
        print(f"   Result: {result2.win_probability:.1%} win rate in {query_time:.3f}s")
        
        solver2.close()
        
    except Exception as e:
        print(f"   [WARN] No-warming solver test failed: {e}")
    
    # Option 3: No caching at all
    print("\n3. Solver with caching disabled:")
    try:
        solver3 = MonteCarloSolver(enable_caching=False)
        print("   [PASS] Solver created with caching disabled")
        
        # Test the same query
        print("   Testing same query without caching...")
        start_time = time.time()
        result3 = solver3.analyze_hand(["AS", "AH"], 2, simulation_mode="fast")
        query_time = time.time() - start_time
        
        print(f"   Result: {result3.win_probability:.1%} win rate in {query_time:.3f}s")
        
        solver3.close()
        
    except Exception as e:
        print(f"   [WARN] No-caching solver test failed: {e}")
    
    print("\n[PASS] Solver integration demonstration completed!")


def demonstrate_script_usage_pattern():
    """Demonstrate typical script usage patterns."""
    print("\n📜 Script Usage Pattern Demonstration")
    print("=" * 60)
    
    # Simulate script-like usage
    scenarios = [
        (["AS", "AH"], 2, "Pocket Aces vs 2"),
        (["KS", "KH"], 3, "Pocket Kings vs 3"),
        (["AS", "KS"], 2, "AK suited vs 2"),
        (["QS", "QH"], 4, "Pocket Queens vs 4"),
        (["JS", "JH"], 1, "Pocket Jacks vs 1")
    ]
    
    print("Simulating multiple script runs...")
    
    for i, (hero_hand, opponents, description) in enumerate(scenarios):
        print(f"\nScript run {i+1}: {description}")
        
        # Each script run creates a new solver
        start_time = time.time()
        solver = MonteCarloSolver(enable_caching=True, skip_cache_warming=(i > 0))  # Skip after first
        creation_time = time.time() - start_time
        
        # Analyze hand
        start_time = time.time()
        result = solver.analyze_hand(hero_hand, opponents, simulation_mode="fast")
        analysis_time = time.time() - start_time
        
        print(f"  Solver creation: {creation_time:.3f}s")
        print(f"  Analysis time: {analysis_time:.3f}s")
        print(f"  Win probability: {result.win_probability:.1%}")
        
        # Check if result was cached
        cached = analysis_time < 0.01  # Very fast = likely cached
        print(f"  Likely cached: {'Yes' if cached else 'No'}")
        
        solver.close()
    
    print("\n[PASS] Script usage pattern demonstration completed!")


def run_comprehensive_demo():
    """Run comprehensive demonstration of cache pre-population."""
    print("♞ Poker Knight Cache Pre-Population System Demo")
    print("=" * 80)
    print("Demonstrating improved cache system with one-time pre-population")
    print("Replaces background warming with script-friendly approach")
    print("=" * 80)
    
    if not DEPENDENCIES_AVAILABLE:
        print("[FAIL] Required dependencies not available. Cannot run demo.")
        return
    
    try:
        # Run all demonstrations
        demonstrate_configuration_options()
        demonstrate_scenario_generation()
        demonstrate_solver_integration()
        demonstrate_script_usage_pattern()
        
        print("\n🎉 All cache pre-population demonstrations completed successfully!")
        
        print("\n📋 Summary of Improvements:")
        print("[PASS] One-time pre-population instead of background threading")
        print("[PASS] Script-friendly: predictable startup cost, then instant queries")
        print("[PASS] User control: skip warming, force regeneration, disable caching")
        print("[PASS] Comprehensive coverage: all 169 preflop hands + board textures")
        print("[PASS] Configurable scope: premium-only, common, or full coverage")
        print("[PASS] Memory efficient: persistent storage with intelligent caching")
        
        print("\n🎯 Perfect for:")
        print("  • One-shot analysis scripts")
        print("  • Batch processing tools")
        print("  • Interactive sessions")
        print("  • Web applications")
        print("  • AI poker bots")
        
        print("\n[ROCKET] Expected performance:")
        print("  • First run: 2-3 minute population + instant query")
        print("  • Subsequent runs: <0.001s for cached scenarios")
        print("  • Cache hit rate: 95-100% for common scenarios")
        print("  • Storage: 10-20MB persistent cache")
        
    except Exception as e:
        print(f"[FAIL] Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_comprehensive_demo() 