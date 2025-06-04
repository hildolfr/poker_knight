"""
Multi-way pot and tournament analysis for Poker Knight.

This module contains logic for analyzing multi-way pots, ICM calculations,
and tournament-specific adjustments.
"""

from typing import List, Optional, Dict, Any


class MultiwayAnalyzer:
    """Handles multi-way pot statistics and tournament analysis."""
    
    def __init__(self):
        """Initialize the multiway analyzer."""
        pass
    
    def calculate_multiway_statistics(self, hero_hand: List[str], num_opponents: int,
                                    board_cards: Optional[List[str]], 
                                    win_prob: float, tie_prob: float, loss_prob: float,
                                    hero_position: Optional[str], stack_sizes: Optional[List[int]], 
                                    pot_size: Optional[int], 
                                    tournament_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate comprehensive multi-way pot statistics and ICM analysis.
        """
        analysis = {}
        
        # Position-aware equity calculation
        if hero_position:
            position_analysis = self._calculate_position_aware_equity(
                hero_hand, num_opponents, board_cards, win_prob, hero_position
            )
            analysis['position_aware_equity'] = position_analysis['equity_by_position']
            analysis['fold_equity_estimates'] = position_analysis['fold_equity']
        
        # Multi-way statistics for 3+ opponents
        if num_opponents >= 3:
            multiway_stats = self._calculate_multiway_core_statistics(
                hero_hand, num_opponents, board_cards, win_prob, tie_prob, loss_prob
            )
            analysis['multi_way_statistics'] = multiway_stats
            analysis['coordination_effects'] = self._calculate_coordination_effects(num_opponents, win_prob)
            analysis['defense_frequencies'] = self._calculate_defense_frequencies(num_opponents, win_prob)
            analysis['bluff_catching_frequency'] = self._calculate_bluff_catching_frequency(num_opponents, win_prob)
            analysis['range_coordination_score'] = self._calculate_range_coordination_score(num_opponents, win_prob)
        
        # ICM Integration
        if tournament_context or stack_sizes:
            icm_analysis = self._calculate_icm_equity(
                win_prob, stack_sizes, pot_size, tournament_context
            )
            analysis.update(icm_analysis)
        
        return analysis
    
    def _calculate_position_aware_equity(self, hero_hand: List[str], num_opponents: int,
                                       board_cards: Optional[List[str]], win_prob: float, 
                                       hero_position: str) -> Dict[str, Any]:
        """Calculate position-aware equity adjustments."""
        # Position multipliers based on poker theory
        position_multipliers = {
            'early': 0.85,    # Under the gun - tight ranges, less fold equity
            'middle': 0.92,   # Middle position - moderate ranges
            'late': 1.05,     # Late position - wider ranges, more fold equity
            'button': 1.12,   # Button - maximum positional advantage
            'sb': 0.88,       # Small blind - out of position post-flop
            'bb': 0.90        # Big blind - already invested, but out of position
        }
        
        base_multiplier = position_multipliers.get(hero_position, 1.0)
        
        # Adjust for number of opponents
        opponent_adjustment = 1.0 + (num_opponents - 1) * 0.02
        
        # Calculate position-adjusted equity
        position_equity = win_prob * base_multiplier * opponent_adjustment
        position_equity = max(0.0, min(1.0, position_equity))  # Clamp to valid range
        
        # Calculate fold equity estimates by position
        fold_equity_base = {
            'early': 0.15,
            'middle': 0.25,
            'late': 0.35,
            'button': 0.45,
            'sb': 0.12,
            'bb': 0.08
        }
        
        fold_equity = fold_equity_base.get(hero_position, 0.20)
        
        # Adjust fold equity based on hand strength and opponents
        hand_strength_factor = min(win_prob * 1.5, 1.0)
        opponent_factor = max(0.5, 1.0 - (num_opponents - 2) * 0.1)
        
        adjusted_fold_equity = fold_equity * hand_strength_factor * opponent_factor
        
        return {
            'equity_by_position': {
                hero_position: position_equity,
                'baseline_equity': win_prob,
                'position_advantage': position_equity - win_prob
            },
            'fold_equity': {
                'base_fold_equity': adjusted_fold_equity,
                'position_modifier': base_multiplier,
                'opponent_adjustment': opponent_factor
            }
        }
    
    def _calculate_multiway_core_statistics(self, hero_hand: List[str], num_opponents: int,
                                          board_cards: Optional[List[str]], 
                                          win_prob: float, tie_prob: float, loss_prob: float) -> Dict[str, Any]:
        """Calculate core statistics for multi-way pots."""
        # Calculate probability of winning against specific number of opponents
        prob_win_vs_1 = win_prob ** (1.0 / num_opponents)  # Approximate individual win rate
        prob_win_vs_all = win_prob  # Probability of beating all opponents
        
        # Multi-way variance reduction
        multiway_variance_reduction = 1.0 - (num_opponents - 2) * 0.05
        
        # Conditional win probability
        conditional_win_prob = win_prob / (win_prob + tie_prob) if (win_prob + tie_prob) > 0 else 0
        
        return {
            'total_opponents': num_opponents,
            'individual_win_rate': prob_win_vs_1,
            'conditional_win_probability': conditional_win_prob,
            'multiway_variance_factor': multiway_variance_reduction,
            'expected_position_finish': self._estimate_finish_position(win_prob, num_opponents),
            'pot_equity_vs_individual': prob_win_vs_1,
            'showdown_frequency': win_prob + tie_prob,
        }
    
    def _calculate_coordination_effects(self, num_opponents: int, win_prob: float) -> Dict[str, float]:
        """Calculate how opponent ranges coordinate against hero in multi-way pots."""
        # More opponents means more coordination potential
        coordination_factor = min(0.3, (num_opponents - 2) * 0.08)
        
        # Strong hands face more coordination
        hand_strength_coordination = min(win_prob * 0.4, 0.3)
        
        total_coordination_effect = coordination_factor + hand_strength_coordination
        
        return {
            'coordination_factor': coordination_factor,
            'hand_strength_coordination': hand_strength_coordination,
            'total_coordination_effect': total_coordination_effect,
            'isolation_difficulty': total_coordination_effect * 2.0
        }
    
    def _calculate_defense_frequencies(self, num_opponents: int, win_prob: float) -> Dict[str, float]:
        """Calculate optimal defense frequencies for multi-way scenarios."""
        # Basic defense frequency calculation
        base_defense_frequency = 1.0 / (num_opponents + 1)
        
        # Adjust based on hand strength
        strength_adjustment = win_prob * 0.5
        
        # Position-neutral defense frequency
        optimal_defense_freq = base_defense_frequency + strength_adjustment
        optimal_defense_freq = max(0.1, min(0.8, optimal_defense_freq))
        
        # Minimum defense frequency to prevent exploitation
        min_defense_freq = base_defense_frequency * 0.8
        
        return {
            'optimal_defense_frequency': optimal_defense_freq,
            'minimum_defense_frequency': min_defense_freq,
            'base_mathematical_frequency': base_defense_frequency,
            'strength_adjustment': strength_adjustment
        }
    
    def _calculate_bluff_catching_frequency(self, num_opponents: int, win_prob: float) -> float:
        """Calculate optimal bluff catching frequency against multiple opponents."""
        # With more opponents, need stronger hands to call bluffs profitably
        opponent_adjustment = max(0.3, 1.0 - (num_opponents - 2) * 0.15)
        
        # Base bluff catching frequency based on hand strength
        base_frequency = min(win_prob * 1.2, 0.6)
        
        # Adjust for multi-way dynamics
        multiway_bluff_catch_freq = base_frequency * opponent_adjustment
        
        return max(0.1, min(0.5, multiway_bluff_catch_freq))
    
    def _calculate_range_coordination_score(self, num_opponents: int, win_prob: float) -> float:
        """Calculate how well opponent ranges coordinate in multi-way scenarios."""
        # Base coordination increases with number of opponents
        base_coordination = min(0.7, 0.2 + (num_opponents - 2) * 0.1)
        
        # Strong hands face more coordinated opposition
        strength_penalty = win_prob * 0.3
        
        # Final coordination score
        coordination_score = base_coordination + strength_penalty
        
        return max(0.0, min(1.0, coordination_score))
    
    def _estimate_finish_position(self, win_prob: float, num_opponents: int) -> float:
        """Estimate expected finish position in multi-way scenario."""
        total_players = num_opponents + 1
        
        # Expected position when winning
        win_position = 1.0
        
        # Expected position when losing
        lose_position = (total_players + 2) / 2
        
        # Weighted average
        expected_position = (win_prob * win_position) + ((1 - win_prob) * lose_position)
        
        return expected_position
    
    def _calculate_icm_equity(self, win_prob: float, stack_sizes: Optional[List[int]], 
                            pot_size: Optional[int], 
                            tournament_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate ICM (Independent Chip Model) equity for tournament play."""
        icm_analysis = {}
        
        # Calculate stack-to-pot ratio
        if stack_sizes and pot_size and len(stack_sizes) > 0:
            hero_stack = stack_sizes[0]
            spr = hero_stack / pot_size if pot_size > 0 else float('inf')
            icm_analysis['stack_to_pot_ratio'] = spr
            
            # Calculate tournament pressure
            total_chips = sum(stack_sizes)
            hero_chip_percentage = hero_stack / total_chips if total_chips > 0 else 0
            
            icm_analysis['tournament_pressure'] = {
                'hero_chip_percentage': hero_chip_percentage,
                'average_stack': total_chips / len(stack_sizes),
                'stack_pressure': 1.0 - hero_chip_percentage
            }
        
        # Calculate ICM equity
        if tournament_context or (stack_sizes and pot_size):
            # Get bubble factor from tournament context
            bubble_factor = 1.0
            if tournament_context:
                bubble_factor = tournament_context.get('bubble_factor', 1.0)
                icm_analysis['bubble_factor'] = bubble_factor
            
            # Calculate ICM equity (simplified model)
            base_icm_equity = win_prob
            
            # Adjust for bubble pressure
            if bubble_factor > 1.0:
                # More conservative during bubble
                bubble_adjustment = max(0.7, 1.0 - (bubble_factor - 1.0) * 0.3)
                base_icm_equity *= bubble_adjustment
            
            # Adjust for stack pressure
            if 'tournament_pressure' in icm_analysis:
                stack_pressure = icm_analysis['tournament_pressure']['stack_pressure']
                if stack_pressure > 0.7:  # Short stack
                    # Short stacks need to take more risks
                    base_icm_equity *= min(1.2, 1.0 + (stack_pressure - 0.7) * 0.5)
                elif stack_pressure < 0.3:  # Big stack
                    # Big stacks can afford to be more conservative
                    base_icm_equity *= max(0.9, 1.0 - (0.3 - stack_pressure) * 0.2)
            
            icm_analysis['icm_equity'] = max(0.0, min(1.0, base_icm_equity))
        
        return icm_analysis