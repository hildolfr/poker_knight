"""
Intelligent Optimization Module for Poker Knight

Provides advanced simulation optimization capabilities including:
- Scenario complexity analysis with hand strength and board texture scoring
- Opponent count and stack depth factor analysis
- Automatic simulation count recommendations based on scenario complexity
- Adaptive optimization strategies for maximum efficiency
- Smart sampling and variance reduction techniques

Author: hildolfr
Version: 1.5.0
"""

import math
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum

# Import poker-specific utilities
try:
    from .deck import Card, Deck
    from .hand_evaluator import HandEvaluator
except ImportError:
    # Fallback for standalone usage
    pass


class ComplexityLevel(Enum):
    """Scenario complexity levels for optimization decisions."""
    TRIVIAL = 1      # <1000 simulations recommended
    SIMPLE = 2       # 1K-10K simulations recommended  
    MODERATE = 3     # 10K-50K simulations recommended
    COMPLEX = 4      # 50K-200K simulations recommended
    EXTREME = 5      # 200K+ simulations recommended


@dataclass
class ScenarioComplexity:
    """Comprehensive scenario complexity analysis."""
    overall_complexity: ComplexityLevel
    complexity_score: float  # 0.0 to 10.0 scale
    
    # Individual complexity factors
    hand_strength_factor: float
    board_texture_factor: float
    opponent_count_factor: float
    stack_depth_factor: float
    position_factor: float
    
    # Recommended simulation parameters
    recommended_simulations: int
    recommended_timeout_ms: int
    recommended_convergence_threshold: float
    confidence_level: float
    
    # Optimization insights
    primary_complexity_drivers: List[str]
    optimization_recommendations: List[str]


@dataclass
class HandStrengthAnalysis:
    """Hand strength evaluation for complexity analysis."""
    hand_category: str  # 'premium', 'strong', 'marginal', 'weak'
    equity_estimate: float
    playability_score: float  # How well hand plays post-flop
    volatility_factor: float  # Variance in outcomes
    drawing_potential: float  # Potential to improve


@dataclass
class BoardTextureAnalysis:
    """Board texture complexity evaluation."""
    texture_type: str  # 'dry', 'coordinated', 'wet', 'dangerous'
    draw_density: float  # Number of potential draws
    connectivity_score: float  # How connected the board is
    flush_potential: float
    straight_potential: float
    volatility_multiplier: float


class ScenarioAnalyzer:
    """Advanced scenario complexity analysis for optimization."""
    
    def __init__(self):
        """Initialize the scenario analyzer."""
        # Hand strength categories and base equity estimates
        self.hand_categories = {
            'premium': {'min_equity': 0.80, 'playability': 0.95, 'volatility': 0.2},
            'strong': {'min_equity': 0.65, 'playability': 0.80, 'volatility': 0.3},
            'marginal': {'min_equity': 0.45, 'playability': 0.60, 'volatility': 0.5},
            'weak': {'min_equity': 0.25, 'playability': 0.40, 'volatility': 0.7}
        }
        
        # Base simulation recommendations by complexity level
        self.simulation_recommendations = {
            ComplexityLevel.TRIVIAL: {'base': 500, 'max': 1000},
            ComplexityLevel.SIMPLE: {'base': 2000, 'max': 10000},
            ComplexityLevel.MODERATE: {'base': 15000, 'max': 50000},
            ComplexityLevel.COMPLEX: {'base': 75000, 'max': 200000},
            ComplexityLevel.EXTREME: {'base': 250000, 'max': 500000}
        }
    
    def analyze_hand_strength(self, player_hand: Union[str, List[str]], 
                            board: Optional[Union[str, List[str]]] = None,
                            num_opponents: int = 1) -> HandStrengthAnalysis:
        """
        Analyze hand strength complexity factors.
        
        Args:
            player_hand: Player's hole cards (e.g., "As Ah" or ["A♠️", "A♥️"])
            board: Community cards (optional, e.g., "Kh 9c 5d" or ["K♥️", "9♣️", "5♦️"])
            num_opponents: Number of opponents
            
        Returns:
            HandStrengthAnalysis with complexity factors
        """
        # Parse hand for analysis
        hand_cards = self._parse_hand(player_hand)
        
        # Determine hand category and base metrics
        hand_category = self._categorize_hand(hand_cards, board)
        base_metrics = self.hand_categories[hand_category]
        
        # Adjust equity estimate based on opponents
        equity_adjustment = max(0.1, 1.0 - (num_opponents - 1) * 0.15)
        estimated_equity = base_metrics['min_equity'] * equity_adjustment
        
        # Calculate playability (how well hand performs across different boards)
        playability = self._calculate_playability(hand_cards, board)
        
        # Calculate volatility factor (variance in outcomes)
        volatility = self._calculate_hand_volatility(hand_cards, board, num_opponents)
        
        # Calculate drawing potential
        drawing_potential = self._calculate_drawing_potential(hand_cards, board)
        
        return HandStrengthAnalysis(
            hand_category=hand_category,
            equity_estimate=estimated_equity,
            playability_score=playability,
            volatility_factor=volatility,
            drawing_potential=drawing_potential
        )
    
    def analyze_board_texture(self, board: Optional[Union[str, List[str]]] = None) -> BoardTextureAnalysis:
        """
        Analyze board texture complexity factors.
        
        Args:
            board: Community cards (e.g., "Kh 9c 5d" or ["K♥️", "9♣️", "5♦️"])
            
        Returns:
            BoardTextureAnalysis with texture complexity metrics
        """
        if not board:
            # Pre-flop analysis
            return BoardTextureAnalysis(
                texture_type='preflop',
                draw_density=0.0,
                connectivity_score=0.0,
                flush_potential=0.0,
                straight_potential=0.0,
                volatility_multiplier=1.0
            )
        
        board_cards = self._parse_hand(board)
        
        # Analyze board connectivity
        connectivity = self._calculate_board_connectivity(board_cards)
        
        # Analyze draw potential
        flush_potential = self._calculate_flush_potential(board_cards)
        straight_potential = self._calculate_straight_potential(board_cards)
        draw_density = flush_potential + straight_potential
        
        # Determine texture type
        texture_type = self._classify_board_texture(connectivity, draw_density)
        
        # Calculate volatility multiplier
        volatility_multiplier = 1.0 + (draw_density * 0.5) + (connectivity * 0.3)
        
        return BoardTextureAnalysis(
            texture_type=texture_type,
            draw_density=draw_density,
            connectivity_score=connectivity,
            flush_potential=flush_potential,
            straight_potential=straight_potential,
            volatility_multiplier=volatility_multiplier
        )
    
    def calculate_scenario_complexity(self, 
                                    player_hand: Union[str, List[str]],
                                    num_opponents: int = 1,
                                    board: Optional[Union[str, List[str]]] = None,
                                    stack_depth: float = 100.0,
                                    position: str = 'middle') -> ScenarioComplexity:
        """
        Perform comprehensive scenario complexity analysis.
        
        Args:
            player_hand: Player's hole cards (e.g., "As Ah" or ["A♠️", "A♥️"])
            num_opponents: Number of opponents (1-8)
            board: Community cards (optional, e.g., "Kh 9c 5d" or ["K♥️", "9♣️", "5♦️"])
            stack_depth: Stack depth in big blinds
            position: Player position ('early', 'middle', 'late', 'blinds')
            
        Returns:
            ScenarioComplexity with comprehensive analysis and recommendations
        """
        # Analyze individual complexity factors
        hand_analysis = self.analyze_hand_strength(player_hand, board, num_opponents)
        board_analysis = self.analyze_board_texture(board)
        
        # Calculate individual complexity factors (0.0 to 2.0 scale each)
        hand_strength_factor = self._calculate_hand_complexity_factor(hand_analysis)
        board_texture_factor = self._calculate_board_complexity_factor(board_analysis)
        opponent_count_factor = self._calculate_opponent_complexity_factor(num_opponents)
        stack_depth_factor = self._calculate_stack_depth_factor(stack_depth)
        position_factor = self._calculate_position_factor(position)
        
        # Calculate overall complexity score (0.0 to 10.0)
        complexity_score = (
            hand_strength_factor +
            board_texture_factor + 
            opponent_count_factor +
            stack_depth_factor +
            position_factor
        )
        
        # Determine complexity level
        overall_complexity = self._determine_complexity_level(complexity_score)
        
        # Generate simulation recommendations
        recommendations = self._generate_simulation_recommendations(
            overall_complexity, complexity_score, hand_analysis, board_analysis
        )
        
        # Identify primary complexity drivers
        drivers = self._identify_complexity_drivers({
            'hand_strength': hand_strength_factor,
            'board_texture': board_texture_factor,
            'opponent_count': opponent_count_factor,
            'stack_depth': stack_depth_factor,
            'position': position_factor
        })
        
        # Generate optimization recommendations
        optimization_recommendations = self._generate_optimization_recommendations(
            overall_complexity, drivers, hand_analysis, board_analysis
        )
        
        return ScenarioComplexity(
            overall_complexity=overall_complexity,
            complexity_score=complexity_score,
            hand_strength_factor=hand_strength_factor,
            board_texture_factor=board_texture_factor,
            opponent_count_factor=opponent_count_factor,
            stack_depth_factor=stack_depth_factor,
            position_factor=position_factor,
            recommended_simulations=recommendations['simulations'],
            recommended_timeout_ms=recommendations['timeout_ms'],
            recommended_convergence_threshold=recommendations['convergence_threshold'],
            confidence_level=recommendations['confidence_level'],
            primary_complexity_drivers=drivers,
            optimization_recommendations=optimization_recommendations
        )
    
    def _parse_hand(self, hand_input: Union[str, List[str]]) -> List[Tuple[str, str]]:
        """
        Parse hand input into rank-suit tuples.
        
        Accepts both formats for API consistency:
        - Unicode format (like solver): ["K♥️", "Q♥️"] or "K♥️ Q♥️"  
        - Simple format: "Kh Qh"
        
        Args:
            hand_input: Either a string ("Kh Qh" or "K♥️ Q♥️") or list (["K♥️", "Q♥️"])
            
        Returns:
            List of (rank, suit) tuples with normalized simple suits
        """
        # Handle list input (Unicode format from solver)
        if isinstance(hand_input, list):
            cards = hand_input
        else:
            # Handle string input
            cards = hand_input.strip().split()
        
        parsed_cards = []
        
        # Unicode to simple suit mapping
        unicode_suit_map = {
            '♠️': 's', '♠': 's',  # Spades (with and without variation selector)
            '♥️': 'h', '♥': 'h',  # Hearts  
            '♦️': 'd', '♦': 'd',  # Diamonds
            '♣️': 'c', '♣': 'c'   # Clubs
        }
        
        for card in cards:
            if not card:  # Skip empty strings
                continue
                
            # Handle 10 specially since it's two characters
            if card.startswith('10'):
                rank = 'T'  # Normalize 10 to T
                suit_part = card[2:]
            else:
                rank = card[0]
                suit_part = card[1:]
            
            # Convert Unicode suits to simple format if needed
            simple_suit = unicode_suit_map.get(suit_part, suit_part.lower())
            
            # Validate that we have a valid suit
            if simple_suit not in ['s', 'h', 'd', 'c']:
                raise ValueError(f"Invalid suit in card '{card}': {suit_part}")
            
            # Validate rank
            if rank not in ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']:
                raise ValueError(f"Invalid rank in card '{card}': {rank}")
            
            parsed_cards.append((rank, simple_suit))
        
        return parsed_cards
    
    def _categorize_hand(self, hand_cards: List[Tuple[str, str]], 
                        board: Optional[Union[str, List[str]]] = None) -> str:
        """Categorize hand strength."""
        if len(hand_cards) < 2:
            return 'weak'
        
        rank1, suit1 = hand_cards[0]
        rank2, suit2 = hand_cards[1]
        
        # Convert face cards to numeric values for comparison
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        val1 = rank_values.get(rank1, 2)
        val2 = rank_values.get(rank2, 2)
        
        is_pair = (val1 == val2)
        is_suited = (suit1 == suit2)
        high_card = max(val1, val2)
        
        # Premium hands
        if is_pair and val1 >= 10:  # TT+
            return 'premium'
        if high_card == 14:  # Ace high
            if is_pair or (is_suited and min(val1, val2) >= 10):
                return 'premium'
        
        # Strong hands
        if is_pair and val1 >= 7:  # 77+
            return 'strong'
        if high_card >= 12 and (is_suited or min(val1, val2) >= 10):  # KQ+, KT+s
            return 'strong'
        
        # Marginal hands
        if is_pair or (high_card >= 10 and is_suited):
            return 'marginal'
        
        return 'weak'
    
    def _calculate_playability(self, hand_cards: List[Tuple[str, str]], 
                             board: Optional[Union[str, List[str]]] = None) -> float:
        """Calculate hand playability score."""
        if len(hand_cards) < 2:
            return 0.3
        
        rank1, suit1 = hand_cards[0]
        rank2, suit2 = hand_cards[1]
        
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        val1 = rank_values.get(rank1, 2)
        val2 = rank_values.get(rank2, 2)
        
        is_pair = (val1 == val2)
        is_suited = (suit1 == suit2)
        is_connected = abs(val1 - val2) <= 1
        
        playability = 0.5  # Base playability
        
        # Bonus for pairs
        if is_pair:
            playability += 0.3
        
        # Bonus for suited cards
        if is_suited:
            playability += 0.2
        
        # Bonus for connectivity
        if is_connected and not is_pair:
            playability += 0.15
        
        # Bonus for high cards
        high_card = max(val1, val2)
        playability += (high_card - 7) * 0.02
        
        return min(1.0, max(0.1, playability))
    
    def _calculate_hand_volatility(self, hand_cards: List[Tuple[str, str]], 
                                 board: Optional[Union[str, List[str]]], num_opponents: int) -> float:
        """Calculate hand outcome volatility."""
        base_volatility = 0.3
        
        # More opponents increase volatility
        opponent_multiplier = 1.0 + (num_opponents - 1) * 0.1
        
        # Drawing hands have higher volatility
        if len(hand_cards) >= 2:
            rank1, suit1 = hand_cards[0]
            rank2, suit2 = hand_cards[1]
            is_suited = (suit1 == suit2)
            
            if is_suited:
                base_volatility += 0.2
        
        return min(1.0, base_volatility * opponent_multiplier)
    
    def _calculate_drawing_potential(self, hand_cards: List[Tuple[str, str]], 
                                   board: Optional[Union[str, List[str]]]) -> float:
        """Calculate drawing potential."""
        if not board or len(hand_cards) < 2:
            return 0.5  # Pre-flop drawing potential
        
        # Simplified drawing potential calculation
        rank1, suit1 = hand_cards[0]
        rank2, suit2 = hand_cards[1]
        is_suited = (suit1 == suit2)
        
        drawing_potential = 0.2  # Base drawing potential
        
        if is_suited:
            drawing_potential += 0.3  # Flush potential
        
        # Straight potential (simplified)
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        val1 = rank_values.get(rank1, 2)
        val2 = rank_values.get(rank2, 2)
        
        if abs(val1 - val2) <= 4:  # Connected enough for straight potential
            drawing_potential += 0.2
        
        return min(1.0, drawing_potential)
    
    def _calculate_board_connectivity(self, board_cards: List[Tuple[str, str]]) -> float:
        """Calculate board connectivity score."""
        if len(board_cards) < 3:
            return 0.0
        
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        ranks = [rank_values.get(card[0], 2) for card in board_cards]
        ranks.sort()
        
        connectivity = 0.0
        
        # Check for consecutive ranks
        for i in range(len(ranks) - 1):
            if ranks[i+1] - ranks[i] == 1:
                connectivity += 0.3
            elif ranks[i+1] - ranks[i] == 2:
                connectivity += 0.2
        
        return min(1.0, connectivity)
    
    def _calculate_flush_potential(self, board_cards: List[Tuple[str, str]]) -> float:
        """Calculate flush draw potential."""
        if len(board_cards) < 3:
            return 0.0
        
        suit_counts = {}
        for _, suit in board_cards:
            suit_counts[suit] = suit_counts.get(suit, 0) + 1
        
        max_suit_count = max(suit_counts.values())
        
        if max_suit_count >= 3:
            return 0.8
        elif max_suit_count == 2:
            return 0.4
        else:
            return 0.1
    
    def _calculate_straight_potential(self, board_cards: List[Tuple[str, str]]) -> float:
        """Calculate straight draw potential."""
        if len(board_cards) < 3:
            return 0.0
        
        rank_values = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
                      '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        ranks = set(rank_values.get(card[0], 2) for card in board_cards)
        
        # Check for straight potential
        straight_potential = 0.0
        
        for rank in ranks:
            # Count how many cards in a 5-card straight window
            window_count = sum(1 for r in range(rank-2, rank+3) if r in ranks)
            if window_count >= 2:
                straight_potential = max(straight_potential, window_count * 0.2)
        
        return min(1.0, straight_potential)
    
    def _classify_board_texture(self, connectivity: float, draw_density: float) -> str:
        """Classify board texture type."""
        if connectivity + draw_density > 1.5:
            return 'dangerous'
        elif draw_density > 0.8:
            return 'wet'
        elif connectivity > 0.5:
            return 'coordinated'
        else:
            return 'dry'
    
    def _calculate_hand_complexity_factor(self, hand_analysis: HandStrengthAnalysis) -> float:
        """Calculate hand strength complexity factor."""
        # More volatile and marginal hands require more simulations
        if hand_analysis.hand_category == 'premium':
            return 0.5
        elif hand_analysis.hand_category == 'strong':
            return 1.0
        elif hand_analysis.hand_category == 'marginal':
            return 1.8
        else:  # weak
            return 1.5
    
    def _calculate_board_complexity_factor(self, board_analysis: BoardTextureAnalysis) -> float:
        """Calculate board texture complexity factor."""
        base_factor = 0.5
        
        if board_analysis.texture_type == 'dangerous':
            base_factor = 2.0
        elif board_analysis.texture_type == 'wet':
            base_factor = 1.5
        elif board_analysis.texture_type == 'coordinated':
            base_factor = 1.0
        
        return base_factor
    
    def _calculate_opponent_complexity_factor(self, num_opponents: int) -> float:
        """Calculate opponent count complexity factor."""
        # More opponents exponentially increase complexity
        return min(2.0, 0.3 + (num_opponents - 1) * 0.3)
    
    def _calculate_stack_depth_factor(self, stack_depth: float) -> float:
        """Calculate stack depth complexity factor."""
        # Very short and very deep stacks increase complexity
        if stack_depth < 20:  # Short stack
            return 1.5
        elif stack_depth > 200:  # Very deep
            return 1.2
        else:
            return 0.8  # Normal stack depth
    
    def _calculate_position_factor(self, position: str) -> float:
        """Calculate positional complexity factor."""
        position_factors = {
            'early': 1.2,
            'middle': 1.0,
            'late': 0.8,
            'blinds': 1.1
        }
        return position_factors.get(position, 1.0)
    
    def _determine_complexity_level(self, complexity_score: float) -> ComplexityLevel:
        """Determine overall complexity level."""
        if complexity_score <= 2.0:
            return ComplexityLevel.TRIVIAL
        elif complexity_score <= 4.0:
            return ComplexityLevel.SIMPLE
        elif complexity_score <= 6.0:
            return ComplexityLevel.MODERATE
        elif complexity_score <= 8.0:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.EXTREME
    
    def _generate_simulation_recommendations(self, complexity_level: ComplexityLevel,
                                           complexity_score: float,
                                           hand_analysis: HandStrengthAnalysis,
                                           board_analysis: BoardTextureAnalysis) -> Dict[str, Any]:
        """Generate simulation parameter recommendations."""
        base_rec = self.simulation_recommendations[complexity_level]
        
        # Adjust based on precise complexity score
        score_multiplier = 1.0 + (complexity_score - complexity_level.value * 2) * 0.1
        recommended_sims = int(base_rec['base'] * score_multiplier)
        recommended_sims = min(recommended_sims, base_rec['max'])
        
        # Timeout recommendations (ms)
        timeout_ms = recommended_sims // 10  # Rough estimate: 10 sims per ms
        
        # Convergence threshold (lower for more complex scenarios)
        convergence_threshold = max(0.001, 0.01 - complexity_score * 0.001)
        
        # Confidence level (higher for more important decisions)
        confidence_level = 0.95 if complexity_score > 6.0 else 0.90
        
        return {
            'simulations': recommended_sims,
            'timeout_ms': timeout_ms,
            'convergence_threshold': convergence_threshold,
            'confidence_level': confidence_level
        }
    
    def _identify_complexity_drivers(self, factors: Dict[str, float]) -> List[str]:
        """Identify primary complexity drivers."""
        # Sort factors by value and return top contributors
        sorted_factors = sorted(factors.items(), key=lambda x: x[1], reverse=True)
        
        drivers = []
        for factor_name, value in sorted_factors:
            if value > 1.0:  # Significant complexity contributor
                drivers.append(factor_name.replace('_', ' ').title())
        
        return drivers[:3]  # Top 3 drivers
    
    def _generate_optimization_recommendations(self, complexity_level: ComplexityLevel,
                                             drivers: List[str],
                                             hand_analysis: HandStrengthAnalysis,
                                             board_analysis: BoardTextureAnalysis) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if complexity_level in [ComplexityLevel.TRIVIAL, ComplexityLevel.SIMPLE]:
            recommendations.append("Enable fast mode for quick analysis")
        
        if complexity_level in [ComplexityLevel.COMPLEX, ComplexityLevel.EXTREME]:
            recommendations.append("Enable convergence analysis for early stopping")
            recommendations.append("Consider parallel processing for faster results")
        
        if 'Board Texture' in drivers:
            recommendations.append("High board texture complexity - increase simulation count")
        
        if 'Opponent Count' in drivers:
            recommendations.append("Multiple opponents detected - use precision mode")
        
        if hand_analysis.hand_category == 'marginal':
            recommendations.append("Marginal hand strength - requires detailed analysis")
        
        if len(recommendations) == 0:
            recommendations.append("Standard optimization settings appropriate")
        
        return recommendations


def create_scenario_analyzer() -> ScenarioAnalyzer:
    """Create and return a new scenario analyzer instance."""
    return ScenarioAnalyzer() 