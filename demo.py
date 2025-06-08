#!/usr/bin/env python3
"""
🎰 Poker Knight Demo - Monte Carlo Texas Hold'em Solver 🎰

This demo showcases the power and speed of Poker Knight's poker analysis engine.
Watch as we analyze various poker scenarios with professional-grade accuracy!
"""

import time
import sys
from typing import List, Tuple
from poker_knight import solve_poker_hand, MonteCarloSolver, prepopulate_cache

# ANSI color codes for fancy output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_header(title: str, emoji: str = "♠♥♦♣"):
    """Print a fancy header for demo sections."""
    print(f"\n{Colors.HEADER}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{emoji}  {title}  {emoji}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'='*70}{Colors.ENDC}")

def print_subheader(title: str, number: int = None):
    """Print a subsection header."""
    prefix = f"{number}️⃣ " if number else "▶️ "
    print(f"\n{Colors.CYAN}{prefix}{title}{Colors.ENDC}")

def print_result(scenario: str, result, execution_time: float = None, cache_hit: bool = False):
    """Print results in a visually appealing format."""
    print(f"\n{Colors.BOLD}{scenario}{Colors.ENDC}")
    
    # Win/Loss bars
    win_bar = '█' * int(result.win_probability * 30)
    tie_bar = '█' * int(result.tie_probability * 30)
    loss_bar = '█' * int(result.loss_probability * 30)
    
    print(f"├─ 🏆 Win:  {Colors.GREEN}{result.win_probability:6.2%}{Colors.ENDC} {Colors.GREEN}{win_bar}{Colors.ENDC}")
    print(f"├─ 🤝 Tie:  {Colors.YELLOW}{result.tie_probability:6.2%}{Colors.ENDC} {Colors.YELLOW}{tie_bar}{Colors.ENDC}")
    print(f"└─ 💔 Loss: {Colors.RED}{result.loss_probability:6.2%}{Colors.ENDC} {Colors.RED}{loss_bar}{Colors.ENDC}")
    
    if execution_time is not None:
        speed_emoji = "🚀" if cache_hit else "⚡"
        cache_text = f" {Colors.GREEN}(CACHED){Colors.ENDC}" if cache_hit else ""
        print(f"   {speed_emoji} Analysis time: {execution_time:.3f}s{cache_text}")
    
    # Show confidence interval if available
    if result.confidence_interval:
        print(f"   📊 95% Confidence: [{result.confidence_interval[0]:.3f}, {result.confidence_interval[1]:.3f}]")

def format_hand(cards: List[str]) -> str:
    """Format a hand with colors based on suits."""
    formatted = []
    for card in cards:
        if '♠' in card or '♣' in card:
            formatted.append(f"{Colors.BOLD}{card}{Colors.ENDC}")
        else:
            formatted.append(f"{Colors.RED}{card}{Colors.ENDC}")
    return ' '.join(formatted)

def demo_welcome():
    """Display welcome message."""
    print(f"\n{Colors.BOLD}{'🎰'*20}")
    print(f"{Colors.CYAN}♠♥♦♣  WELCOME TO POKER KNIGHT  ♣♦♥♠{Colors.ENDC}")
    print(f"Monte Carlo Texas Hold'em Solver v1.5.5")
    print(f"{'🎰'*20}{Colors.ENDC}")
    
    print(f"\n{Colors.YELLOW}Features:{Colors.ENDC}")
    print("  ✓ Lightning-fast Monte Carlo simulations")
    print("  ✓ Advanced caching system for instant results")
    print("  ✓ Multi-way pot analysis")
    print("  ✓ ICM tournament calculations")
    print("  ✓ Position-aware equity adjustments")
    print("  ✓ Real-time convergence analysis")
    print("  ✓ Hand category tracking")
    print("  ✓ Professional-grade accuracy")

def demo_cache_prepopulation():
    """Demonstrate cache prepopulation for optimal performance."""
    print_header("Intelligent Cache System", "💾")
    
    print("\nPoker Knight uses an advanced caching system for lightning-fast analysis.")
    print("On first use, we'll prepopulate common scenarios for instant results.")
    
    print(f"\n{Colors.YELLOW}Prepopulating cache...{Colors.ENDC}")
    start_time = time.time()
    
    # This will trigger cache population on first use
    _ = solve_poker_hand(['A♠', 'A♥'], 2, simulation_mode="fast")
    
    elapsed = time.time() - start_time
    print(f"{Colors.GREEN}✓ Cache ready! ({elapsed:.1f}s){Colors.ENDC}")
    
    # Show cache effectiveness
    print("\n🏎️  Let's see the cache in action:")
    
    # First call - might hit cache
    start = time.time()
    result1 = solve_poker_hand(['K♠', 'K♥'], 2, simulation_mode="fast")
    time1 = time.time() - start
    
    # Second call - definitely hits cache
    start = time.time()
    result2 = solve_poker_hand(['K♠', 'K♥'], 2, simulation_mode="fast")
    time2 = time.time() - start
    
    print(f"  First analysis:  {time1:.3f}s")
    print(f"  Cached analysis: {time2:.3f}s {Colors.GREEN}({int(time1/time2) if time2 > 0 else '>1000'}x faster!){Colors.ENDC}")

def demo_basic_preflop():
    """Demonstrate basic preflop analysis."""
    print_header("Preflop Analysis", "🎯")
    
    scenarios = [
        ((['A♠', 'A♥'], 2), "Pocket Aces vs 2 opponents", "The nuts! But how much of a favorite?"),
        ((['A♠', 'K♠'], 4), "Big Slick Suited vs 4 opponents", "Premium hand in a family pot"),
        ((['7♦', '2♣'], 1), "The worst hand vs 1 opponent", "Even 72o has equity!"),
        ((['J♠', 'J♥'], 5), "Pocket Jacks vs 5 opponents", "The most difficult hand to play"),
    ]
    
    for i, ((hand, opponents), title, description) in enumerate(scenarios, 1):
        print_subheader(f"{title}", i)
        print(f"{Colors.YELLOW}{description}{Colors.ENDC}")
        
        start = time.time()
        result = solve_poker_hand(hand, opponents, simulation_mode="default")
        elapsed = time.time() - start
        
        print_result(f"Your hand: {format_hand(hand)} vs {opponents} opponent{'s' if opponents > 1 else ''}", 
                    result, elapsed)

def demo_postflop_scenarios():
    """Demonstrate post-flop analysis with various board textures."""
    print_header("Post-flop Analysis", "🎲")
    
    print("Let's see how different board textures affect our equity...")
    
    scenarios = [
        {
            'hand': ['A♠', 'K♠'],
            'board': ['Q♠', 'J♠', '10♥'],
            'opponents': 2,
            'title': 'The Nuts on the Flop!',
            'description': 'Flopped Broadway with redraw to royal flush'
        },
        {
            'hand': ['K♥', 'K♦'],
            'board': ['A♠', 'Q♥', 'J♥', '10♥'],
            'opponents': 2,
            'title': 'Dangerous Board for Kings',
            'description': 'Overpair on a wet, connected board'
        },
        {
            'hand': ['8♠', '7♠'],
            'board': ['9♠', '6♥', '5♦'],
            'opponents': 2,
            'title': 'Monster Draw',
            'description': 'Open-ended straight flush draw'
        },
        {
            'hand': ['A♥', 'A♣'],
            'board': ['K♠', 'K♥', 'K♦', '2♣', '2♠'],
            'opponents': 2,
            'title': 'Aces Full on the River',
            'description': 'Near nuts on a paired board'
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print_subheader(scenario['title'], i)
        print(f"{Colors.YELLOW}{scenario['description']}{Colors.ENDC}")
        
        start = time.time()
        result = solve_poker_hand(
            scenario['hand'], 
            scenario['opponents'],
            board_cards=scenario['board'],
            simulation_mode="default"
        )
        elapsed = time.time() - start
        
        board_str = ' '.join([format_hand([card]) for card in scenario['board']])
        print_result(
            f"Hand: {format_hand(scenario['hand'])} | Board: {board_str}",
            result, elapsed
        )
        
        # Show hand categories if available
        if result.hand_category_frequencies:
            print(f"\n   {Colors.CYAN}Hand Category Distribution:{Colors.ENDC}")
            sorted_categories = sorted(result.hand_category_frequencies.items(), 
                                     key=lambda x: x[1], reverse=True)[:3]
            for category, freq in sorted_categories:
                print(f"     • {category.replace('_', ' ').title()}: {freq:.1%}")

def demo_multiway_advanced():
    """Demonstrate advanced multi-way pot analysis."""
    print_header("Advanced Multi-way Analysis", "🎪")
    
    print("Poker Knight provides sophisticated multi-way pot analysis...")
    
    # Position-aware equity
    print_subheader("Position-Aware Equity", 1)
    print("How does position affect your equity? Let's find out!")
    
    hand = ['A♠', 'Q♠']
    positions = ['early', 'button']
    
    for position in positions:
        start = time.time()
        result = solve_poker_hand(
            hand, 3, 
            simulation_mode="fast",
            hero_position=position
        )
        elapsed = time.time() - start
        
        print(f"\n{Colors.BOLD}Position: {position.upper()}{Colors.ENDC}")
        if result.position_aware_equity:
            baseline = result.position_aware_equity['baseline_equity']
            adjusted = result.position_aware_equity.get(position, baseline)
            advantage = adjusted - baseline
            
            print(f"  Base equity: {baseline:.1%}")
            print(f"  Adjusted equity: {adjusted:.1%}")
            advantage_color = Colors.GREEN if advantage > 0 else Colors.RED
            print(f"  Position advantage: {advantage_color}{advantage:+.1%}{Colors.ENDC}")
    
    # Multi-way statistics
    print_subheader("Multi-way Pot Statistics", 2)
    print("Complex dynamics in 5-way pots...")
    
    start = time.time()
    result = solve_poker_hand(['Q♠', 'Q♥'], 4, simulation_mode="default")
    elapsed = time.time() - start
    
    print_result("Pocket Queens vs 4 opponents", result, elapsed)
    
    if result.multi_way_statistics:
        stats = result.multi_way_statistics
        print(f"\n   {Colors.CYAN}Multi-way Statistics:{Colors.ENDC}")
        print(f"     • Win vs any single opponent: {stats['individual_win_rate']:.1%}")
        print(f"     • Expected finish position: {stats['expected_position_finish']:.1f}")
        print(f"     • Showdown frequency: {stats['showdown_frequency']:.1%}")

def demo_tournament_icm():
    """Demonstrate ICM calculations for tournament play."""
    print_header("Tournament ICM Analysis", "🏆")
    
    print("In tournaments, chip value isn't linear. ICM adjusts for this...")
    
    print_subheader("Bubble Situation Analysis")
    
    # Tournament bubble scenario
    hero_hand = ['A♠', 'J♥']
    stack_sizes = [25000, 30000, 15000, 8000]  # Hero, Villain1, Villain2, Villain3
    pot_size = 4500
    tournament_context = {
        'stage': 'bubble',
        'payouts': [1000, 600, 400],  # Top 3 paid
        'players_remaining': 4,
        'bubble_factor': 2.0
    }
    
    # Compare cash game equity vs ICM equity
    print(f"\n{Colors.YELLOW}Comparing cash game vs tournament equity:{Colors.ENDC}")
    
    # Cash game (no ICM)
    start = time.time()
    cash_result = solve_poker_hand(hero_hand, 1, simulation_mode="default")
    cash_time = time.time() - start
    
    # Tournament (with ICM)
    start = time.time()
    icm_result = solve_poker_hand(
        hero_hand, 1,
        simulation_mode="default",
        stack_sizes=stack_sizes[:2],  # Hero vs Villain1
        pot_size=pot_size,
        tournament_context=tournament_context
    )
    icm_time = time.time() - start
    
    print(f"\n{Colors.BOLD}Cash Game Equity:{Colors.ENDC}")
    print(f"  Raw win probability: {cash_result.win_probability:.1%}")
    
    print(f"\n{Colors.BOLD}Tournament ICM Adjusted:{Colors.ENDC}")
    print(f"  Raw win probability: {icm_result.win_probability:.1%}")
    if icm_result.icm_equity:
        print(f"  ICM adjusted equity: {Colors.YELLOW}{icm_result.icm_equity:.1%}{Colors.ENDC}")
        print(f"  Bubble factor impact: {icm_result.bubble_factor:.1f}x")
        print(f"  Stack-to-pot ratio: {icm_result.stack_to_pot_ratio:.1f}")
        
        diff = icm_result.icm_equity - icm_result.win_probability
        print(f"  ICM adjustment: {Colors.RED}{diff:.1%}{Colors.ENDC} (tighter due to bubble)")

def demo_simulation_modes():
    """Compare different simulation modes."""
    print_header("Simulation Modes", "⚡")
    
    print("Poker Knight offers three simulation modes for different needs...")
    
    hand = ['9♠', '9♥']
    opponents = 3
    board = ['8♠', '7♥', '6♦']
    
    modes = [
        ("fast", "10,000", "Quick decisions", Colors.YELLOW),
        ("default", "100,000", "Standard accuracy", Colors.CYAN),
        ("precision", "500,000", "Maximum precision", Colors.GREEN)
    ]
    
    print(f"\nAnalyzing: {format_hand(hand)} on {format_hand(board)} vs {opponents} opponents")
    
    results_comparison = []
    
    for mode, sims, description, color in modes:
        start = time.time()
        result = solve_poker_hand(hand, opponents, board, simulation_mode=mode)
        elapsed = time.time() - start
        
        results_comparison.append((mode, result, elapsed))
        
        print(f"\n{color}{mode.upper()} MODE - {description}{Colors.ENDC}")
        print(f"  Simulations: {sims}")
        print(f"  Win probability: {result.win_probability:.3%}")
        print(f"  Time: {elapsed:.3f}s ({result.simulations_run/elapsed:,.0f} sims/sec)")
        
        if result.confidence_interval:
            margin = (result.confidence_interval[1] - result.confidence_interval[0]) / 2
            print(f"  Margin of error: ±{margin:.3%}")
    
    # Show convergence
    fast_win = results_comparison[0][1].win_probability
    precision_win = results_comparison[2][1].win_probability
    diff = abs(fast_win - precision_win)
    
    print(f"\n{Colors.BOLD}Accuracy Analysis:{Colors.ENDC}")
    print(f"  Fast vs Precision difference: {diff:.3%}")
    print(f"  Fast mode is {results_comparison[2][2]/results_comparison[0][2]:.0f}x faster")

def demo_performance_showcase():
    """Showcase the performance capabilities."""
    print_header("Performance Showcase", "🚀")
    
    print("Let's run a batch analysis to show Poker Knight's speed...")
    
    # Create solver instance for batch processing
    solver = MonteCarloSolver()
    
    hands = [
        (['A♠', 'A♥'], "Pocket Aces"),
        (['K♠', 'K♥'], "Pocket Kings"),
        (['A♠', 'K♠'], "Big Slick Suited"),
        (['Q♠', 'J♠'], "Queen-Jack Suited"),
        (['10♠', '10♥'], "Pocket Tens"),
        (['8♠', '7♠'], "Eight-Seven Suited"),
        (['A♥', 'Q♣'], "Ace-Queen Offsuit"),
        (['5♠', '5♥'], "Pocket Fives"),
    ]
    
    print(f"\nAnalyzing {len(hands)} different hands vs 2 opponents...")
    print("Watch the cache system work its magic! 🎩✨")
    
    total_start = time.time()
    results = []
    
    for hand, name in hands:
        start = time.time()
        result = solver.analyze_hand(hand, 2, simulation_mode="fast")
        elapsed = time.time() - start
        results.append((name, hand, result.win_probability, elapsed))
        
        # Quick visual feedback
        cache_indicator = "💾" if elapsed < 0.01 else "🔄"
        print(f"  {cache_indicator} {name:20} → Win: {result.win_probability:5.1%} (Time: {elapsed:.3f}s)")
    
    total_time = time.time() - total_start
    avg_time = total_time / len(hands)
    
    print(f"\n{Colors.GREEN}✓ Analyzed {len(hands)} hands in {total_time:.2f}s{Colors.ENDC}")
    print(f"  Average time per hand: {avg_time:.3f}s")
    print(f"  Total simulations: ~{len(hands) * 10000:,}")

def demo_real_world_scenario():
    """Demonstrate a real-world decision scenario."""
    print_header("Real-World Decision", "💭")
    
    print("You're in a $2/$5 game with $1,000 effective stacks...")
    print("\nThe situation:")
    print("  • You have K♥ Q♥ on the button")
    print("  • Villain (tight player) raises to $20 from middle position")
    print("  • You call, blinds fold")
    print("  • Pot: $47")
    print(f"  • Flop: {format_hand(['J♥', '10♥', '3♠'])}")
    print("  • Villain bets $30")
    
    print(f"\n{Colors.YELLOW}What's your equity with this monster draw?{Colors.ENDC}")
    
    # Analyze the situation
    start = time.time()
    result = solve_poker_hand(
        ['K♥', 'Q♥'],
        1,
        board_cards=['J♥', '10♥', '3♠'],
        simulation_mode="precision",
        hero_position="button"
    )
    elapsed = time.time() - start
    
    print_result("Your equity with open-ended straight flush draw", result, elapsed)
    
    # Calculate pot odds
    pot = 47 + 30
    call_amount = 30
    pot_odds = call_amount / (pot + call_amount)
    
    print(f"\n{Colors.CYAN}Decision Analysis:{Colors.ENDC}")
    print(f"  • Pot odds needed: {pot_odds:.1%}")
    print(f"  • Your equity: {result.win_probability:.1%}")
    
    if result.win_probability > pot_odds:
        print(f"  • {Colors.GREEN}✓ Profitable call!{Colors.ENDC} (+{(result.win_probability - pot_odds):.1%} edge)")
    else:
        print(f"  • {Colors.RED}✗ Fold{Colors.ENDC} (-{(pot_odds - result.win_probability):.1%} deficit)")
    
    # Show outs
    print(f"\n{Colors.CYAN}Your outs (15 total):{Colors.ENDC}")
    print("  • 9 hearts for a flush (excluding straight flushes)")
    print("  • 3 aces + 3 nines for a straight (non-hearts)")
    print("  • 2 straight flush cards (A♥, 9♥)")
    print("  • Plus backdoor two-pair/trips possibilities")

def main():
    """Run the complete demo."""
    demo_welcome()
    
    input(f"\n{Colors.YELLOW}Press Enter to begin the demo...{Colors.ENDC}")
    
    # Run all demo sections
    demo_cache_prepopulation()
    input(f"\n{Colors.YELLOW}Press Enter to see preflop analysis...{Colors.ENDC}")
    
    demo_basic_preflop()
    input(f"\n{Colors.YELLOW}Press Enter to see post-flop scenarios...{Colors.ENDC}")
    
    demo_postflop_scenarios()
    input(f"\n{Colors.YELLOW}Press Enter to see advanced multi-way analysis...{Colors.ENDC}")
    
    demo_multiway_advanced()
    input(f"\n{Colors.YELLOW}Press Enter to see tournament ICM calculations...{Colors.ENDC}")
    
    demo_tournament_icm()
    input(f"\n{Colors.YELLOW}Press Enter to compare simulation modes...{Colors.ENDC}")
    
    demo_simulation_modes()
    input(f"\n{Colors.YELLOW}Press Enter to see performance showcase...{Colors.ENDC}")
    
    demo_performance_showcase()
    input(f"\n{Colors.YELLOW}Press Enter to see a real-world decision...{Colors.ENDC}")
    
    demo_real_world_scenario()
    
    # Closing
    print_header("Demo Complete! 🎉", "🌟")
    print(f"\n{Colors.GREEN}Poker Knight is ready to revolutionize your poker game!{Colors.ENDC}")
    
    print("\n📚 Quick Start Guide:")
    print(f"{Colors.CYAN}from poker_knight import solve_poker_hand{Colors.ENDC}")
    print(f"{Colors.CYAN}result = solve_poker_hand(['A♠', 'K♠'], 2){Colors.ENDC}")
    print(f"{Colors.CYAN}print(f'Win: {{result.win_probability:.1%}}'){Colors.ENDC}")
    
    print("\n🔧 Advanced Features:")
    print("  • Position-aware analysis")
    print("  • ICM tournament calculations")  
    print("  • Multi-way pot dynamics")
    print("  • Real-time convergence monitoring")
    print("  • Enterprise-grade caching")
    
    print("\n📈 Why Poker Knight?")
    print("  • ⚡ Lightning fast (cache-enabled)")
    print("  • 🎯 Professional accuracy")
    print("  • 🔬 Based on proven Monte Carlo methods")
    print("  • 🏆 Tournament & cash game support")
    print("  • 🐍 Pure Python (no dependencies)")
    
    print(f"\n{Colors.BOLD}Ready to take your poker analysis to the next level?{Colors.ENDC}")
    print(f"\n🌐 Learn more: {Colors.BLUE}https://github.com/hildolfr/poker-knight{Colors.ENDC}")
    print(f"📦 Install: {Colors.GREEN}pip install poker-knight{Colors.ENDC}")
    print(f"\n{Colors.YELLOW}Good luck at the tables! 🍀{Colors.ENDC}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Demo interrupted. Thanks for watching!{Colors.ENDC}")
        sys.exit(0)