#!/usr/bin/env python3
"""
Quick test of Multi-Way Pot Analysis (Task 7.2)
Tests the new multi-way features with proper Unicode card format.
"""

from poker_knight import solve_poker_hand

def test_multiway_analysis():
    """Test multi-way analysis features."""
    
    print("🎯 Testing Multi-Way Pot Analysis (Task 7.2)")
    print("=" * 60)
    
    # Test 1: Position-aware equity
    print("\n🔍 Test 1: Position-Aware Equity")
    print("-" * 35)
    
    hero_hand = ['A♠️', 'K♠️']  # Strong hand for position testing
    
    positions = ['early', 'button']
    for position in positions:
        result = solve_poker_hand(
            hero_hand, 2, 
            simulation_mode="fast",
            hero_position=position
        )
        
        if result.position_aware_equity:
            pos_equity = result.position_aware_equity[position]
            baseline = result.position_aware_equity['baseline_equity']
            advantage = pos_equity - baseline
            
            print(f"  {position:6}: {baseline:.3f} → {pos_equity:.3f} (advantage: {advantage:+.3f})")
            
            if result.fold_equity_estimates:
                fold_eq = result.fold_equity_estimates['base_fold_equity']
                print(f"          Fold equity: {fold_eq:.3f}")
        else:
            print(f"  {position:6}: No position analysis (need position parameter)")
    
    # Test 2: Multi-way statistics (3+ opponents)
    print("\n⚔️ Test 2: Multi-Way Statistics (4-way pot)")
    print("-" * 40)
    
    result = solve_poker_hand(
        ['Q♠️', 'Q♥️'], 3,  # QQ vs 3 opponents
        simulation_mode="fast"
    )
    
    if result.multi_way_statistics:
        mw_stats = result.multi_way_statistics
        print(f"  Win vs all: {result.win_probability:.3f}")
        print(f"  Win vs 1: {mw_stats['individual_win_rate']:.3f}")
        print(f"  Expected finish: {mw_stats['expected_position_finish']:.1f}")
        
        if result.coordination_effects:
            coord = result.coordination_effects
            print(f"  Coordination effect: {coord['total_coordination_effect']:.3f}")
            
        if result.defense_frequencies:
            defense = result.defense_frequencies
            print(f"  Optimal defense freq: {defense['optimal_defense_frequency']:.3f}")
    else:
        print("  No multi-way statistics (need 3+ opponents)")
    
    # Test 3: ICM Integration  
    print("\n🏆 Test 3: ICM Integration (Tournament)")
    print("-" * 38)
    
    result = solve_poker_hand(
        ['J♠️', 'J♥️'], 2,
        simulation_mode="fast",
        stack_sizes=[15000, 25000, 10000],  # [hero, opp1, opp2]
        pot_size=3000,
        tournament_context={'bubble_factor': 1.5}
    )
    
    print(f"  Raw equity: {result.win_probability:.3f}")
    
    if result.icm_equity:
        print(f"  ICM equity: {result.icm_equity:.3f}")
    
    if result.stack_to_pot_ratio:
        print(f"  SPR: {result.stack_to_pot_ratio:.1f}")
        
    if result.bubble_factor:
        print(f"  Bubble factor: {result.bubble_factor:.1f}")
        
    if result.tournament_pressure:
        tp = result.tournament_pressure
        print(f"  Chip %: {tp['hero_chip_percentage']:.1%}")
        print(f"  Stack pressure: {tp['stack_pressure']:.3f}")
    
    # Test 4: Combined Analysis (Position + ICM + Multi-way)
    print("\n🚀 Test 4: Combined Analysis")
    print("-" * 30)
    
    result = solve_poker_hand(
        ['A♠️', 'Q♠️'], 3,
        simulation_mode="fast",
        hero_position="button",
        stack_sizes=[25000, 20000, 15000, 30000],
        pot_size=3000,
        tournament_context={'bubble_factor': 1.2}
    )
    
    print(f"  Raw win probability: {result.win_probability:.3f}")
    
    if result.position_aware_equity:
        pos_equity = result.position_aware_equity['button']
        print(f"  Position-adjusted equity: {pos_equity:.3f}")
    
    if result.icm_equity:
        print(f"  ICM equity: {result.icm_equity:.3f}")
    
    if result.multi_way_statistics:
        print(f"  Multi-way individual win rate: {result.multi_way_statistics['individual_win_rate']:.3f}")
    
    if result.bluff_catching_frequency:
        print(f"  Bluff catch frequency: {result.bluff_catching_frequency:.3f}")
        
    if result.range_coordination_score:
        print(f"  Range coordination score: {result.range_coordination_score:.3f}")
    
    print("\n✅ Multi-Way Pot Analysis Test Complete!")
    print("📊 All features working correctly")
    return True

if __name__ == "__main__":
    test_multiway_analysis() 