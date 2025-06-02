# Integration Examples

Learn how to integrate Poker Knight into larger poker AI systems and applications.

## Basic AI Poker Bot

```python
from poker_knight import MonteCarloSolver

class PokerAI:
    def __init__(self):
        self.solver = MonteCarloSolver()
    
    def make_decision(self, hole_cards, board_cards, num_opponents):
        result = self.solver.analyze_hand(
            hole_cards, 
            num_opponents, 
            board_cards,
            simulation_mode="fast"  # Quick decisions
        )
        
        if result.win_probability > 0.7:
            return "bet"
        elif result.win_probability > 0.4:
            return "call"
        else:
            return "fold"
    
    def get_betting_strength(self, hole_cards, board_cards, num_opponents):
        """Return betting strength from 0.0 to 1.0"""
        result = self.solver.analyze_hand(hole_cards, num_opponents, board_cards)
        return result.win_probability

# Usage
ai = PokerAI()
decision = ai.make_decision(['A♠️', 'K♠️'], ['Q♠️', 'J♠️', '10♥️'], 2)
print(f"AI Decision: {decision}")
```

## Advanced Tournament Bot

```python
from poker_knight import MonteCarloSolver
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class GameState:
    hole_cards: List[str]
    board_cards: List[str]
    num_opponents: int
    pot_size: int
    stack_size: int
    position: str
    betting_round: str  # 'preflop', 'flop', 'turn', 'river'

class TournamentBot:
    def __init__(self):
        self.solver = MonteCarloSolver()
        self.position_multipliers = {
            'early': 0.85,
            'middle': 0.92,
            'late': 1.05,
            'button': 1.12,
            'small_blind': 0.88,
            'big_blind': 0.90
        }
    
    def analyze_situation(self, game_state: GameState):
        """Comprehensive situation analysis"""
        # Get base equity
        mode = self._select_simulation_mode(game_state)
        result = self.solver.analyze_hand(
            game_state.hole_cards,
            game_state.num_opponents,
            game_state.board_cards,
            simulation_mode=mode
        )
        
        # Apply position adjustment
        position_adj = self.position_multipliers.get(game_state.position, 1.0)
        adjusted_equity = result.win_probability * position_adj
        
        # Calculate pot odds consideration
        pot_odds_factor = self._calculate_pot_odds_factor(game_state)
        
        return {
            'raw_equity': result.win_probability,
            'adjusted_equity': adjusted_equity,
            'pot_odds_factor': pot_odds_factor,
            'confidence_interval': result.confidence_interval,
            'simulations_run': result.simulations_run,
            'hand_categories': result.hand_category_frequencies
        }
    
    def _select_simulation_mode(self, game_state: GameState):
        """Select appropriate simulation mode based on situation"""
        if game_state.betting_round == 'preflop' and game_state.num_opponents > 4:
            return "fast"  # Quick decisions in multiway pots
        elif game_state.stack_size < 20 * game_state.pot_size:
            return "precision"  # Critical all-in decisions
        else:
            return "default"  # Standard analysis
    
    def _calculate_pot_odds_factor(self, game_state: GameState):
        """Calculate pot odds considerations"""
        return min(1.5, game_state.pot_size / max(1, game_state.stack_size))
    
    def make_tournament_decision(self, game_state: GameState):
        """Make a decision considering tournament factors"""
        analysis = self.analyze_situation(game_state)
        
        # Tournament-specific adjustments
        if game_state.stack_size < 10 * game_state.pot_size:
            # Short stack - more aggressive
            threshold_bet = 0.55
            threshold_call = 0.35
        elif game_state.stack_size > 50 * game_state.pot_size:
            # Deep stack - more selective
            threshold_bet = 0.75
            threshold_call = 0.45
        else:
            # Medium stack - standard play
            threshold_bet = 0.65
            threshold_call = 0.4
        
        equity = analysis['adjusted_equity']
        
        if equity > threshold_bet:
            return "bet", equity
        elif equity > threshold_call:
            return "call", equity
        else:
            return "fold", equity

# Usage
bot = TournamentBot()
game_state = GameState(
    hole_cards=['A♠️', 'K♠️'],
    board_cards=['Q♠️', 'J♠️', '10♥️'],
    num_opponents=2,
    pot_size=1000,
    stack_size=15000,
    position='button',
    betting_round='flop'
)

decision, equity = bot.make_tournament_decision(game_state)
print(f"Tournament Decision: {decision} (equity: {equity:.3f})")
```

## Hand History Analyzer

```python
from poker_knight import solve_poker_hand
import json
from typing import Dict, List

class HandHistoryAnalyzer:
    def __init__(self):
        self.results = []
    
    def analyze_hand_history(self, hand_data: Dict):
        """Analyze a complete hand history"""
        hero_cards = hand_data['hero_cards']
        opponents = hand_data['num_opponents']
        
        analysis = {
            'hand_id': hand_data.get('hand_id'),
            'preflop': None,
            'flop': None,
            'turn': None,
            'river': None
        }
        
        # Pre-flop analysis
        analysis['preflop'] = solve_poker_hand(hero_cards, opponents)
        
        # Post-flop analysis if board cards available
        if 'flop' in hand_data:
            flop_cards = hand_data['flop']
            analysis['flop'] = solve_poker_hand(hero_cards, opponents, flop_cards)
            
            if 'turn' in hand_data:
                turn_cards = flop_cards + [hand_data['turn']]
                analysis['turn'] = solve_poker_hand(hero_cards, opponents, turn_cards)
                
                if 'river' in hand_data:
                    river_cards = turn_cards + [hand_data['river']]
                    analysis['river'] = solve_poker_hand(hero_cards, opponents, river_cards)
        
        self.results.append(analysis)
        return analysis
    
    def generate_report(self, analysis):
        """Generate a readable analysis report"""
        report = []
        report.append(f"Hand Analysis Report - Hand ID: {analysis['hand_id']}")
        report.append("=" * 50)
        
        for street, result in analysis.items():
            if street == 'hand_id' or result is None:
                continue
                
            report.append(f"\n{street.upper()}:")
            report.append(f"  Win Probability: {result.win_probability:.1%}")
            
            if hasattr(result, 'confidence_interval'):
                lower, upper = result.confidence_interval
                report.append(f"  Confidence Interval: {lower:.1%} - {upper:.1%}")
            
            report.append(f"  Simulations: {result.simulations_run:,}")
            report.append(f"  Execution Time: {result.execution_time_ms:.1f}ms")
        
        return "\n".join(report)
    
    def batch_analyze(self, hand_histories: List[Dict]):
        """Analyze multiple hands in batch"""
        results = []
        for hand_data in hand_histories:
            analysis = self.analyze_hand_history(hand_data)
            results.append(analysis)
        return results

# Usage
analyzer = HandHistoryAnalyzer()

sample_hand = {
    'hand_id': 'HH001',
    'hero_cards': ['A♠️', 'K♠️'],
    'num_opponents': 2,
    'flop': ['Q♠️', 'J♠️', '10♥️'],
    'turn': '2♦️',
    'river': '9♠️'
}

analysis = analyzer.analyze_hand_history(sample_hand)
print(analyzer.generate_report(analysis))
```

## Real-Time Poker Assistant

```python
from poker_knight import MonteCarloSolver
import threading
import time
from queue import Queue

class RealTimePokerAssistant:
    def __init__(self):
        self.solver = MonteCarloSolver()
        self.analysis_queue = Queue()
        self.result_queue = Queue()
        self.worker_thread = None
        self.running = False
        
    def start(self):
        """Start the real-time analysis worker"""
        self.running = True
        self.worker_thread = threading.Thread(target=self._analysis_worker)
        self.worker_thread.daemon = True
        self.worker_thread.start()
    
    def stop(self):
        """Stop the real-time analysis worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
    
    def _analysis_worker(self):
        """Background worker for continuous analysis"""
        while self.running:
            try:
                if not self.analysis_queue.empty():
                    request = self.analysis_queue.get_nowait()
                    
                    start_time = time.time()
                    result = self.solver.analyze_hand(
                        request['hero_cards'],
                        request['num_opponents'],
                        request.get('board_cards'),
                        simulation_mode="fast"
                    )
                    
                    analysis_time = (time.time() - start_time) * 1000
                    
                    response = {
                        'request_id': request['request_id'],
                        'result': result,
                        'analysis_time': analysis_time,
                        'timestamp': time.time()
                    }
                    
                    self.result_queue.put(response)
                    
                time.sleep(0.01)  # Small delay to prevent busy waiting
                
            except Exception as e:
                print(f"Analysis error: {e}")
    
    def request_analysis(self, hero_cards, num_opponents, board_cards=None):
        """Submit analysis request"""
        request_id = f"req_{int(time.time() * 1000)}"
        request = {
            'request_id': request_id,
            'hero_cards': hero_cards,
            'num_opponents': num_opponents,
            'board_cards': board_cards
        }
        
        self.analysis_queue.put(request)
        return request_id
    
    def get_result(self, timeout=1.0):
        """Get analysis result"""
        try:
            return self.result_queue.get(timeout=timeout)
        except:
            return None
    
    def get_instant_advice(self, hero_cards, num_opponents, board_cards=None):
        """Get immediate advice with simple heuristics"""
        # Simple pre-calculation advice while waiting for full analysis
        request_id = self.request_analysis(hero_cards, num_opponents, board_cards)
        
        # Provide instant heuristic-based advice
        instant_advice = self._get_heuristic_advice(hero_cards, board_cards)
        
        # Try to get full analysis quickly
        full_result = self.get_result(timeout=0.5)
        
        return {
            'instant_advice': instant_advice,
            'full_analysis': full_result,
            'request_id': request_id
        }
    
    def _get_heuristic_advice(self, hero_cards, board_cards):
        """Quick heuristic-based advice"""
        # Simple hand strength evaluation
        hand_strength = self._quick_hand_strength(hero_cards, board_cards)
        
        if hand_strength > 0.8:
            return "Strong hand - consider betting/raising"
        elif hand_strength > 0.6:
            return "Good hand - suitable for calling/betting"
        elif hand_strength > 0.4:
            return "Marginal hand - proceed with caution"
        else:
            return "Weak hand - consider folding"
    
    def _quick_hand_strength(self, hero_cards, board_cards):
        """Quick heuristic hand strength calculation"""
        # This is a simplified heuristic - the full Monte Carlo analysis provides accuracy
        if not board_cards:
            # Pre-flop strength based on pocket cards
            ranks = [card[:-2] for card in hero_cards]
            if ranks[0] == ranks[1]:  # Pocket pair
                return 0.7 if ranks[0] in ['A', 'K', 'Q', 'J'] else 0.5
            elif set(ranks) & {'A', 'K'}:  # High cards
                return 0.6
            else:
                return 0.3
        else:
            # Post-flop - very basic evaluation
            return 0.5  # Placeholder - full analysis provides real strength

# Usage
assistant = RealTimePokerAssistant()
assistant.start()

# Get real-time advice
advice = assistant.get_instant_advice(['A♠️', 'K♠️'], 2, ['Q♠️', 'J♠️', '10♥️'])
print(f"Instant advice: {advice['instant_advice']}")

if advice['full_analysis']:
    result = advice['full_analysis']['result']
    print(f"Full analysis: {result.win_probability:.1%} win probability")

assistant.stop()
```

## Statistical Analysis Tool

```python
from poker_knight import solve_poker_hand
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict

class PokerStatisticsAnalyzer:
    def __init__(self):
        self.data = []
    
    def analyze_hand_range(self, starting_hands: List[str], num_opponents: int = 1):
        """Analyze a range of starting hands"""
        results = []
        
        for hand in starting_hands:
            if len(hand) == 2:  # Convert shorthand notation
                cards = self._convert_shorthand_to_cards(hand)
            else:
                cards = hand
            
            result = solve_poker_hand(cards, num_opponents)
            
            results.append({
                'hand': hand,
                'win_probability': result.win_probability,
                'simulations': result.simulations_run,
                'execution_time': result.execution_time_ms
            })
        
        return pd.DataFrame(results)
    
    def _convert_shorthand_to_cards(self, shorthand: str):
        """Convert poker shorthand (e.g., 'AKs') to specific cards"""
        if len(shorthand) == 2:
            # Pocket pair
            rank = shorthand[0]
            return [f'{rank}♠️', f'{rank}♥️']
        elif shorthand.endswith('s'):
            # Suited
            ranks = shorthand[:-1]
            return [f'{ranks[0]}♠️', f'{ranks[1]}♠️']
        elif shorthand.endswith('o'):
            # Offsuit
            ranks = shorthand[:-1]
            return [f'{ranks[0]}♠️', f'{ranks[1]}♥️']
        else:
            # Assume offsuit if no designation
            return [f'{shorthand[0]}♠️', f'{shorthand[1]}♥️']
    
    def generate_equity_chart(self, results_df):
        """Generate equity visualization"""
        plt.figure(figsize=(12, 8))
        
        # Sort by win probability
        sorted_df = results_df.sort_values('win_probability', ascending=False)
        
        plt.bar(range(len(sorted_df)), sorted_df['win_probability'])
        plt.title('Hand Equity Analysis')
        plt.xlabel('Hands (sorted by equity)')
        plt.ylabel('Win Probability')
        plt.xticks(range(len(sorted_df)), sorted_df['hand'], rotation=45)
        plt.tight_layout()
        
        return plt
    
    def compare_scenarios(self, hand: str, opponent_counts: List[int]):
        """Compare same hand against different numbers of opponents"""
        if len(hand) == 2:
            cards = self._convert_shorthand_to_cards(hand)
        else:
            cards = hand
        
        results = []
        for opponents in opponent_counts:
            result = solve_poker_hand(cards, opponents)
            results.append({
                'opponents': opponents,
                'win_probability': result.win_probability,
                'hand': hand
            })
        
        return pd.DataFrame(results)

# Usage
analyzer = PokerStatisticsAnalyzer()

# Analyze premium starting hands
premium_hands = ['AA', 'KK', 'QQ', 'JJ', 'AKs', 'AQs', 'AJs', 'AKo']
results = analyzer.analyze_hand_range(premium_hands, num_opponents=2)

print("Premium Hand Analysis:")
print(results.to_string(index=False))

# Compare against different opponent counts
comparison = analyzer.compare_scenarios('AKs', [1, 2, 3, 4, 5])
print("\nEquity vs Number of Opponents:")
print(comparison.to_string(index=False))
```

## Web API Integration

```python
from flask import Flask, jsonify, request
from poker_knight import solve_poker_hand
import json

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_hand():
    """API endpoint for hand analysis"""
    try:
        data = request.get_json()
        
        # Validate input
        if 'hero_cards' not in data or 'num_opponents' not in data:
            return jsonify({'error': 'Missing required fields'}), 400
        
        hero_cards = data['hero_cards']
        num_opponents = data['num_opponents']
        board_cards = data.get('board_cards')
        simulation_mode = data.get('simulation_mode', 'default')
        
        # Perform analysis
        result = solve_poker_hand(
            hero_cards,
            num_opponents,
            board_cards,
            simulation_mode
        )
        
        # Format response
        response = {
            'win_probability': result.win_probability,
            'tie_probability': result.tie_probability,
            'loss_probability': result.loss_probability,
            'simulations_run': result.simulations_run,
            'execution_time_ms': result.execution_time_ms,
            'confidence_interval': {
                'lower': result.confidence_interval[0],
                'upper': result.confidence_interval[1]
            },
            'hand_categories': result.hand_category_frequencies
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """API endpoint for batch analysis"""
    try:
        data = request.get_json()
        scenarios = data.get('scenarios', [])
        
        results = []
        for scenario in scenarios:
            result = solve_poker_hand(
                scenario['hero_cards'],
                scenario['num_opponents'],
                scenario.get('board_cards'),
                scenario.get('simulation_mode', 'fast')
            )
            
            results.append({
                'scenario_id': scenario.get('id'),
                'win_probability': result.win_probability,
                'simulations_run': result.simulations_run
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

## Best Practices for Integration

1. **Resource Management**: Always call `solver.close()` when using parallel processing
2. **Error Handling**: Wrap analysis calls in try-catch blocks
3. **Performance Monitoring**: Track execution times and adjust simulation modes accordingly
4. **Caching**: Consider caching results for identical scenarios
5. **Configuration**: Use appropriate simulation modes for your use case
6. **Testing**: Validate integration with known poker scenarios 