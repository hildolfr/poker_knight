"""
Smart sampling strategies for Poker Knight.

This module contains advanced sampling strategies for variance reduction
and improved convergence in Monte Carlo simulations.
"""

from typing import List, Dict, Any, Optional
from collections import Counter
from dataclasses import dataclass

from ..core.cards import Card
from ..core.evaluation import HandEvaluator


@dataclass
class SamplingState:
    """State for smart sampling strategies."""
    strategy: str = 'uniform'  # uniform, stratified, importance
    stratification_levels: List[Dict[str, Any]] = None
    importance_weights: List[float] = None
    control_variate_baseline: float = 0.0
    variance_reduction_efficiency: Optional[float] = None
    control_variate_sum: float = 0.0
    control_variate_count: int = 0
    control_variate_mean: float = 0.0
    stratified_results: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.stratification_levels is None:
            self.stratification_levels = []
        if self.importance_weights is None:
            self.importance_weights = []
        if self.stratified_results is None:
            self.stratified_results = {}


class SmartSampler:
    """Implements smart sampling strategies for variance reduction."""
    
    def __init__(self, config: Dict[str, Any], evaluator: HandEvaluator):
        """Initialize the smart sampler with configuration."""
        self.config = config
        self.evaluator = evaluator
        
        # Get sampling strategy settings
        sampling_strategy = config.get("sampling_strategy", {})
        self.stratified_sampling_enabled = sampling_strategy.get("stratified_sampling", False)
        self.importance_sampling_enabled = sampling_strategy.get("importance_sampling", False)
        self.control_variates_enabled = sampling_strategy.get("control_variates", False)
    
    def initialize_sampling(self, hero_cards: List[Card], board: List[Card], 
                          num_simulations: int) -> SamplingState:
        """Initialize smart sampling strategies based on scenario analysis."""
        state = SamplingState()
        
        # Determine appropriate sampling strategy
        if self.stratified_sampling_enabled and num_simulations >= 10000:
            # Use stratified sampling for large simulations
            state.strategy = 'stratified'
            state.stratification_levels = self._compute_stratification_levels(hero_cards, board)
        elif self.importance_sampling_enabled and self._is_extreme_scenario(hero_cards, board):
            # Use importance sampling for extreme scenarios
            state.strategy = 'importance'
            state.importance_weights = self._compute_importance_weights(hero_cards, board)
        
        # Initialize control variates if enabled
        if self.control_variates_enabled:
            state.control_variate_baseline = self._compute_control_variate_baseline(hero_cards, board)
        
        return state
    
    def simulate_with_strategy(self, runner, hero_cards: List[Card], num_opponents: int,
                             board: List[Card], removed_cards: List[Card],
                             sampling_state: SamplingState, sim_number: int) -> Dict[str, Any]:
        """Run simulation with appropriate sampling strategy."""
        if sampling_state.strategy == 'stratified':
            return self._simulate_stratified(
                runner, hero_cards, num_opponents, board, removed_cards, 
                sampling_state, sim_number
            )
        elif sampling_state.strategy == 'importance':
            return self._simulate_importance(
                runner, hero_cards, num_opponents, board, removed_cards,
                sampling_state, sim_number
            )
        else:
            return runner.simulate_hand(hero_cards, num_opponents, board, removed_cards)
    
    def apply_variance_reduction(self, result: Dict[str, Any], 
                               sampling_state: SamplingState, 
                               sim_number: int) -> Dict[str, Any]:
        """Apply control variates for variance reduction."""
        if not self.control_variates_enabled:
            return result
        
        # Control variate: use analytical approximation to reduce variance
        baseline = sampling_state.control_variate_baseline
        observed_win = 1.0 if result['result'] == 'win' else 0.0
        
        # Update running control variate statistics
        sampling_state.control_variate_sum += observed_win
        sampling_state.control_variate_count += 1
        
        if sampling_state.control_variate_count > 100:
            # Calculate control variate adjustment
            current_mean = (sampling_state.control_variate_sum / 
                          sampling_state.control_variate_count)
            
            # Control variate adjustment (simplified)
            control_variate_correction = 0.5 * (baseline - current_mean)
            
            # Apply correction to result
            result['control_variate_correction'] = control_variate_correction
        
        return result
    
    def _compute_stratification_levels(self, hero_cards: List[Card], 
                                     board: List[Card]) -> List[Dict[str, Any]]:
        """Compute stratification levels for rare hand categories."""
        # Define stratification based on final hand strength categories
        strata = [
            {'name': 'premium', 'min_rank': 8, 'target_proportion': 0.05},  # Four of a kind+
            {'name': 'strong', 'min_rank': 6, 'target_proportion': 0.15},   # Full house, flush
            {'name': 'medium', 'min_rank': 4, 'target_proportion': 0.30},   # Three of a kind, straight
            {'name': 'weak', 'min_rank': 2, 'target_proportion': 0.35},     # Pair, two pair
            {'name': 'high_card', 'min_rank': 1, 'target_proportion': 0.15} # High card
        ]
        
        # Adjust proportions based on board texture
        if len(board) >= 3:
            board_analysis = self._analyze_board_texture(board)
            if board_analysis['flush_possible']:
                strata[1]['target_proportion'] *= 1.2  # Increase flush sampling
            if board_analysis['straight_possible']:
                strata[2]['target_proportion'] *= 1.2  # Increase straight sampling
        
        return strata
    
    def _analyze_board_texture(self, board: List[Card]) -> Dict[str, bool]:
        """Analyze board texture for sampling optimization."""
        if len(board) < 3:
            return {'flush_possible': False, 'straight_possible': False, 'paired': False}
        
        # Check for flush possibilities
        suits = [card.suit for card in board]
        suit_counts = Counter(suits)
        flush_possible = max(suit_counts.values()) >= 2
        
        # Check for straight possibilities
        ranks = sorted([card.value for card in board])
        straight_possible = any(ranks[i+1] - ranks[i] <= 4 for i in range(len(ranks)-1))
        
        # Check for pairs
        rank_counts = Counter([card.value for card in board])
        paired = max(rank_counts.values()) >= 2
        
        return {
            'flush_possible': flush_possible,
            'straight_possible': straight_possible,
            'paired': paired
        }
    
    def _is_extreme_scenario(self, hero_cards: List[Card], board: List[Card]) -> bool:
        """Determine if this is an extreme scenario for importance sampling."""
        total_cards = hero_cards + board
        
        # Pre-flop scenarios
        if len(total_cards) < 5:
            if len(hero_cards) == 2:
                # Pocket pairs
                if hero_cards[0].value == hero_cards[1].value:
                    pair_rank = hero_cards[0].value
                    if pair_rank >= 11 or pair_rank <= 4:  # QQ+ or 22-55
                        return True
                
                # Very strong or weak non-pair hands
                high_card = max(hero_cards[0].value, hero_cards[1].value)
                low_card = min(hero_cards[0].value, hero_cards[1].value)
                
                # AK, AQ (very strong)
                if high_card == 12 and low_card >= 10:
                    return True
                
                # Very weak hands
                if high_card <= 7 and (high_card - low_card) >= 5:
                    return True
            
            return False
        
        # Post-flop analysis
        hero_hand_rank, _ = self.evaluator.evaluate_hand(total_cards)
        
        # Extreme scenarios
        if hero_hand_rank >= 8:  # Four of a kind or better
            return True
        if hero_hand_rank == 1 and len(board) >= 3:  # High card on dangerous board
            return True
        
        # Pocket pairs vs overcards
        if len(hero_cards) == 2 and hero_cards[0].value == hero_cards[1].value:
            if len(board) >= 3:
                board_high_card = max(card.value for card in board)
                if hero_cards[0].value < board_high_card:
                    return True  # Underpair to board
        
        return False
    
    def _compute_importance_weights(self, hero_cards: List[Card], 
                                  board: List[Card]) -> List[float]:
        """Compute importance sampling weights for extreme scenarios."""
        total_cards = hero_cards + board
        
        # Pre-flop scenarios
        if len(total_cards) < 5:
            if len(hero_cards) == 2:
                # Pocket pairs
                if hero_cards[0].value == hero_cards[1].value:
                    pair_rank = hero_cards[0].value
                    if pair_rank >= 11:  # Very strong pairs
                        return [0.8, 0.1, 0.1]  # Focus on wins
                    elif pair_rank <= 4:  # Very weak pairs
                        return [0.2, 0.1, 0.7]  # Focus on losses
                
                # High card hands
                high_card = max(hero_cards[0].value, hero_cards[1].value)
                if high_card == 12:  # Ace high
                    return [0.6, 0.2, 0.2]  # Slightly favor wins
                elif high_card <= 7:  # Low cards
                    return [0.1, 0.1, 0.8]  # Focus on losses
            
            return [0.4, 0.2, 0.4]  # Balanced for non-extreme
        
        # Post-flop analysis
        hero_hand_rank, _ = self.evaluator.evaluate_hand(total_cards)
        
        if hero_hand_rank >= 8:  # Very strong hands
            return [0.7, 0.2, 0.1]  # [win, tie, loss] weights
        elif hero_hand_rank == 1:  # Very weak hands  
            return [0.1, 0.1, 0.8]  # [win, tie, loss] weights
        else:
            return [0.4, 0.2, 0.4]  # Balanced for medium strength
    
    def _compute_control_variate_baseline(self, hero_cards: List[Card], 
                                        board: List[Card]) -> float:
        """Compute control variate baseline for variance reduction."""
        total_cards = hero_cards + board
        
        # Pre-flop scenarios
        if len(total_cards) < 5:
            if len(hero_cards) == 2:
                # Pocket pair analysis
                if hero_cards[0].value == hero_cards[1].value:
                    pair_rank = hero_cards[0].value
                    if pair_rank >= 10:  # JJ+
                        return 0.75
                    elif pair_rank >= 6:  # 77-TT
                        return 0.60
                    else:  # 22-66
                        return 0.45
                
                # High card analysis
                high_card = max(hero_cards[0].value, hero_cards[1].value)
                low_card = min(hero_cards[0].value, hero_cards[1].value)
                
                if high_card >= 12:  # Ace high
                    return 0.55 + (low_card / 26)
                elif high_card >= 10:  # King or Queen high
                    return 0.45 + (low_card / 39)
                else:
                    return 0.30 + (high_card / 52)
            
            return 0.50  # Default
        
        # Post-flop analysis
        hero_hand_rank, _ = self.evaluator.evaluate_hand(total_cards)
        
        # Rough analytical approximation based on hand strength
        baseline_probabilities = {
            1: 0.15,   # High card
            2: 0.35,   # Pair
            3: 0.55,   # Two pair
            4: 0.70,   # Three of a kind
            5: 0.80,   # Straight
            6: 0.85,   # Flush
            7: 0.90,   # Full house
            8: 0.95,   # Four of a kind
            9: 0.98,   # Straight flush
            10: 0.99   # Royal flush
        }
        
        return baseline_probabilities.get(hero_hand_rank, 0.50)
    
    def _simulate_stratified(self, runner, hero_cards: List[Card], num_opponents: int,
                           board: List[Card], removed_cards: List[Card],
                           sampling_state: SamplingState, sim_number: int) -> Dict[str, Any]:
        """Simulate hand using stratified sampling."""
        strata = sampling_state.stratification_levels
        
        # Determine which stratum to target
        stratum_index = sim_number % len(strata)
        target_stratum = strata[stratum_index]
        
        # Run multiple attempts to get a result in the target stratum
        max_attempts = 10
        for attempt in range(max_attempts):
            result = runner.simulate_hand(hero_cards, num_opponents, board, removed_cards)
            
            # Check if result belongs to target stratum
            hero_hand_rank = result.get('hero_hand_rank', 1)
            if hero_hand_rank >= target_stratum['min_rank']:
                # Weight the result to correct for stratified sampling bias
                result['stratified_weight'] = 1.0 / target_stratum['target_proportion']
                result['stratum'] = target_stratum['name']
                return result
        
        # If we can't get target stratum, return regular result
        result = runner.simulate_hand(hero_cards, num_opponents, board, removed_cards)
        result['stratified_weight'] = 1.0
        result['stratum'] = 'fallback'
        return result
    
    def _simulate_importance(self, runner, hero_cards: List[Card], num_opponents: int,
                           board: List[Card], removed_cards: List[Card],
                           sampling_state: SamplingState, sim_number: int) -> Dict[str, Any]:
        """Simulate hand using importance sampling."""
        # Get the regular simulation result
        result = runner.simulate_hand(hero_cards, num_opponents, board, removed_cards)
        
        # Apply importance sampling weight based on outcome
        weights = sampling_state.importance_weights
        if result['result'] == 'win':
            result['importance_weight'] = weights[0]
        elif result['result'] == 'tie':
            result['importance_weight'] = weights[1]
        else:  # loss
            result['importance_weight'] = weights[2]
        
        return result