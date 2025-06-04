"""
♞ Poker Knight Comprehensive Board Scenario Cache

Advanced caching system for all board scenarios (preflop, flop, turn, river)
with intelligent board texture recognition and hand-board interaction analysis.

Extends beyond preflop to provide comprehensive caching coverage for all
poker scenarios while maintaining high cache hit rates through smart
pattern recognition and normalization.

Author: hildolfr
License: MIT
"""

import time
import hashlib
import logging
import threading
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum
import itertools

# Import unified cache components
from .unified_cache import (
    ThreadSafeMonteCarloCache, CacheKey, CacheResult, 
    CacheKeyNormalizer, create_cache_key
)

logger = logging.getLogger(__name__)


class BoardStage(Enum):
    """Board stages in poker."""
    PREFLOP = "preflop"
    FLOP = "flop"      # 3 cards
    TURN = "turn"      # 4 cards  
    RIVER = "river"    # 5 cards


class BoardTexture(Enum):
    """Board texture classifications."""
    # Flop textures
    HIGH_CARD = "high_card"           # A72 rainbow
    PAIRED = "paired"                 # AA7, KK9
    TWO_PAIR_BOARD = "two_pair"       # AA99, KK77
    TRIPS_BOARD = "trips"             # AAA, KKK
    
    # Draw textures
    FLUSH_DRAW = "flush_draw"         # 2+ of same suit
    STRAIGHT_DRAW = "straight_draw"   # Connected ranks
    COMBO_DRAW = "combo_draw"         # Flush + straight draws
    
    # Wet vs dry
    DRY_BOARD = "dry"                 # A72 rainbow
    WET_BOARD = "wet"                 # 987ss, JT9
    
    # Coordination
    COORDINATED = "coordinated"       # Connected/suited
    UNCOORDINATED = "uncoordinated"   # Rainbow, gapped


@dataclass
class BoardPattern:
    """Normalized board pattern for caching."""
    stage: BoardStage
    texture: BoardTexture
    rank_pattern: str        # "A-K-Q", "A-A-7", "9-8-7"
    suit_pattern: str        # "rainbow", "flush", "two_tone"
    connectivity: str        # "connected", "gapped", "unconnected"
    pair_count: int         # 0, 1, 2, 3
    flush_draw_strength: int # 0=none, 1=weak, 2=strong, 3=flush
    straight_draw_strength: int # 0=none, 1=gutshot, 2=oesd, 3=straight


@dataclass
class BoardCacheConfig:
    """Configuration for board scenario caching."""
    enable_board_cache: bool = True
    max_memory_mb: int = 256
    enable_persistence: bool = True
    
    # Board stage settings
    cache_preflop: bool = True
    cache_flop: bool = True
    cache_turn: bool = True
    cache_river: bool = True
    
    # Pattern recognition settings
    normalize_suit_patterns: bool = True
    normalize_rank_patterns: bool = True
    group_similar_textures: bool = True
    
    # Coverage settings
    max_board_combinations: int = 10000  # Limit combinations for memory
    priority_textures: List[str] = None
    
    def __post_init__(self):
        if self.priority_textures is None:
            self.priority_textures = [
                "high_card", "paired", "flush_draw", "straight_draw", 
                "dry", "wet", "coordinated"
            ]


class BoardAnalyzer:
    """Analyzes board textures and patterns for caching."""
    
    # Rank values for analysis
    RANK_VALUES = {
        'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '10': 10,
        '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2
    }
    
    # Suit mapping
    SUITS = {'♠': 'spades', '♥': 'hearts', '♦': 'diamonds', '♣': 'clubs'}
    
    @classmethod
    def analyze_board(cls, board_cards: List[str]) -> BoardPattern:
        """Analyze board and return normalized pattern."""
        if not board_cards:
            return BoardPattern(
                stage=BoardStage.PREFLOP,
                texture=BoardTexture.HIGH_CARD,
                rank_pattern="preflop",
                suit_pattern="preflop", 
                connectivity="preflop",
                pair_count=0,
                flush_draw_strength=0,
                straight_draw_strength=0
            )
        
        # Determine stage
        stage = cls._get_board_stage(len(board_cards))
        
        # Parse cards
        ranks, suits = cls._parse_board_cards(board_cards)
        
        # Analyze patterns
        rank_pattern = cls._analyze_rank_pattern(ranks)
        suit_pattern = cls._analyze_suit_pattern(suits)
        connectivity = cls._analyze_connectivity(ranks)
        pair_count = cls._count_pairs(ranks)
        flush_draw_strength = cls._analyze_flush_draws(suits)
        straight_draw_strength = cls._analyze_straight_draws(ranks)
        
        # Determine primary texture
        texture = cls._determine_primary_texture(
            ranks, suits, pair_count, flush_draw_strength, straight_draw_strength
        )
        
        return BoardPattern(
            stage=stage,
            texture=texture,
            rank_pattern=rank_pattern,
            suit_pattern=suit_pattern,
            connectivity=connectivity,
            pair_count=pair_count,
            flush_draw_strength=flush_draw_strength,
            straight_draw_strength=straight_draw_strength
        )
    
    @classmethod
    def _get_board_stage(cls, num_cards: int) -> BoardStage:
        """Get board stage from number of cards."""
        if num_cards == 0:
            return BoardStage.PREFLOP
        elif num_cards == 3:
            return BoardStage.FLOP
        elif num_cards == 4:
            return BoardStage.TURN
        elif num_cards == 5:
            return BoardStage.RIVER
        else:
            raise ValueError(f"Invalid board size: {num_cards}")
    
    @classmethod
    def _parse_board_cards(cls, board_cards: List[str]) -> Tuple[List[str], List[str]]:
        """Parse board cards into ranks and suits."""
        ranks = []
        suits = []
        
        for card in board_cards:
            if card.startswith('10'):
                rank = 'T'  # Normalize 10 to T
                suit = card[2:]
            else:
                rank = card[0]
                suit = card[1:]
            
            # Normalize rank
            if rank == '10':
                rank = 'T'
            
            ranks.append(rank)
            suits.append(suit)
        
        return ranks, suits
    
    @classmethod
    def _analyze_rank_pattern(cls, ranks: List[str]) -> str:
        """Analyze rank pattern for normalization."""
        if not ranks:
            return "preflop"
        
        # Count rank frequencies
        rank_counts = defaultdict(int)
        for rank in ranks:
            rank_counts[rank] += 1
        
        # Sort ranks by value (high to low)
        sorted_ranks = sorted(ranks, key=lambda r: cls.RANK_VALUES[r], reverse=True)
        
        # Create pattern based on frequencies
        pattern_parts = []
        for rank in sorted_ranks:
            count = rank_counts[rank]
            if count > 1:
                pattern_parts.append(f"{rank}x{count}")
            else:
                pattern_parts.append(rank)
            # Remove from counts to avoid duplicates
            if rank_counts[rank] > 0:
                rank_counts[rank] = 0
        
        # Filter out empty parts
        pattern_parts = [p for p in pattern_parts if not p.endswith('x0')]
        
        return "-".join(pattern_parts[:5])  # Limit to 5 cards max
    
    @classmethod
    def _analyze_suit_pattern(cls, suits: List[str]) -> str:
        """Analyze suit pattern."""
        if not suits:
            return "preflop"
        
        # Count suits
        suit_counts = defaultdict(int)
        for suit in suits:
            # Normalize suit representation
            normalized_suit = cls.SUITS.get(suit, suit)
            suit_counts[normalized_suit] += 1
        
        # Determine pattern
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        num_suits = len([c for c in suit_counts.values() if c > 0])
        
        if max_suit_count >= 3:
            return "flush_possible"
        elif max_suit_count == 2:
            return "two_tone"
        elif num_suits == len(suits):
            return "rainbow"
        else:
            return "mixed"
    
    @classmethod
    def _analyze_connectivity(cls, ranks: List[str]) -> str:
        """Analyze rank connectivity."""
        if len(ranks) < 2:
            return "single"
        
        # Convert to values and sort
        values = sorted([cls.RANK_VALUES[rank] for rank in ranks], reverse=True)
        
        # Check for straights and draws
        connected_count = 0
        for i in range(len(values) - 1):
            if values[i] - values[i + 1] == 1:
                connected_count += 1
        
        # Classify connectivity
        if connected_count >= len(values) - 1:
            return "straight"
        elif connected_count >= len(values) - 2:
            return "connected"
        elif connected_count > 0:
            return "semi_connected"
        else:
            return "unconnected"
    
    @classmethod
    def _count_pairs(cls, ranks: List[str]) -> int:
        """Count number of pairs on board."""
        rank_counts = defaultdict(int)
        for rank in ranks:
            rank_counts[rank] += 1
        
        pairs = 0
        for count in rank_counts.values():
            if count >= 2:
                pairs += 1
        
        return pairs
    
    @classmethod
    def _analyze_flush_draws(cls, suits: List[str]) -> int:
        """Analyze flush draw strength (0=none, 1=weak, 2=strong, 3=flush)."""
        if not suits:
            return 0
        
        # Count suits
        suit_counts = defaultdict(int)
        for suit in suits:
            normalized_suit = cls.SUITS.get(suit, suit)
            suit_counts[normalized_suit] += 1
        
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        
        if max_suit_count >= 5:
            return 3  # Flush made
        elif max_suit_count == 4:
            return 2  # Strong flush draw
        elif max_suit_count == 3:
            return 1  # Weak flush draw
        else:
            return 0  # No flush draw
    
    @classmethod
    def _analyze_straight_draws(cls, ranks: List[str]) -> int:
        """Analyze straight draw strength (0=none, 1=gutshot, 2=oesd, 3=straight)."""
        if len(ranks) < 3:
            return 0
        
        # Convert to values and check for straights
        values = sorted(set([cls.RANK_VALUES[rank] for rank in ranks]), reverse=True)
        
        # Check for made straight
        for i in range(len(values) - 4):
            if all(values[i] - values[i + j] == j for j in range(5)):
                return 3  # Straight made
        
        # Check for straight draws (simplified)
        connected_sequences = []
        current_seq = [values[0]]
        
        for i in range(1, len(values)):
            if values[i-1] - values[i] == 1:
                current_seq.append(values[i])
            else:
                if len(current_seq) >= 2:
                    connected_sequences.append(current_seq)
                current_seq = [values[i]]
        
        if len(current_seq) >= 2:
            connected_sequences.append(current_seq)
        
        # Analyze sequences for draws
        max_seq_length = max([len(seq) for seq in connected_sequences]) if connected_sequences else 0
        
        if max_seq_length >= 4:
            return 2  # Open-ended straight draw
        elif max_seq_length == 3:
            return 1  # Gutshot straight draw
        else:
            return 0  # No straight draw
    
    @classmethod
    def _determine_primary_texture(cls, ranks: List[str], suits: List[str], 
                                 pair_count: int, flush_strength: int, 
                                 straight_strength: int) -> BoardTexture:
        """Determine primary board texture."""
        # Check for trips/quads first
        rank_counts = defaultdict(int)
        for rank in ranks:
            rank_counts[rank] += 1
        
        max_rank_count = max(rank_counts.values()) if rank_counts else 0
        
        if max_rank_count >= 3:
            return BoardTexture.TRIPS_BOARD
        elif pair_count >= 2:
            return BoardTexture.TWO_PAIR_BOARD
        elif pair_count == 1:
            return BoardTexture.PAIRED
        
        # Check for draws and coordination
        if flush_strength >= 2 and straight_strength >= 2:
            return BoardTexture.COMBO_DRAW
        elif flush_strength >= 2:
            return BoardTexture.FLUSH_DRAW
        elif straight_strength >= 2:
            return BoardTexture.STRAIGHT_DRAW
        
        # Determine wet vs dry
        if (flush_strength >= 1 or straight_strength >= 1 or 
            cls._analyze_connectivity(ranks) in ["connected", "semi_connected"]):
            return BoardTexture.WET_BOARD
        else:
            return BoardTexture.DRY_BOARD


class BoardScenarioCache:
    """
    Comprehensive cache for all board scenarios (preflop, flop, turn, river).
    
    Uses intelligent board pattern recognition to maximize cache hit rates
    across different board textures while maintaining deterministic results.
    """
    
    def __init__(self, 
                 config: Optional[BoardCacheConfig] = None,
                 unified_cache: Optional[ThreadSafeMonteCarloCache] = None):
        
        self.config = config or BoardCacheConfig()
        self.unified_cache = unified_cache or ThreadSafeMonteCarloCache(
            max_memory_mb=self.config.max_memory_mb,
            enable_persistence=self.config.enable_persistence
        )
        
        # Statistics tracking
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'requests_by_stage': defaultdict(int),
            'hits_by_stage': defaultdict(int),
            'pattern_frequencies': defaultdict(int)
        }
        
        logger.info("Board scenario cache initialized")
    
    def get_board_result(self, 
                        hero_hand: List[str],
                        num_opponents: int,
                        board_cards: Optional[List[str]] = None,
                        simulation_mode: str = "default") -> Optional[CacheResult]:
        """Get cached result for board scenario."""
        self._stats['total_requests'] += 1
        
        # Analyze board pattern
        board_pattern = BoardAnalyzer.analyze_board(board_cards or [])
        self._stats['requests_by_stage'][board_pattern.stage.value] += 1
        self._stats['pattern_frequencies'][board_pattern.texture.value] += 1
        
        # Create optimized cache key
        cache_key = self._create_board_cache_key(
            hero_hand, num_opponents, board_pattern, simulation_mode
        )
        
        # Try to get from cache
        result = self.unified_cache.get(cache_key)
        
        if result:
            self._stats['cache_hits'] += 1
            self._stats['hits_by_stage'][board_pattern.stage.value] += 1
            return result
        else:
            self._stats['cache_misses'] += 1
            return None
    
    def store_board_result(self,
                          hero_hand: List[str],
                          num_opponents: int,
                          result: CacheResult,
                          board_cards: Optional[List[str]] = None,
                          simulation_mode: str = "default") -> bool:
        """Store result for board scenario."""
        # Analyze board pattern
        board_pattern = BoardAnalyzer.analyze_board(board_cards or [])
        
        # Create optimized cache key
        cache_key = self._create_board_cache_key(
            hero_hand, num_opponents, board_pattern, simulation_mode
        )
        
        # Store in unified cache
        return self.unified_cache.store(cache_key, result)
    
    def _create_board_cache_key(self, 
                               hero_hand: List[str],
                               num_opponents: int,
                               board_pattern: BoardPattern,
                               simulation_mode: str) -> CacheKey:
        """Create optimized cache key for board scenarios."""
        
        # Normalize hero hand
        normalized_hand = CacheKeyNormalizer.normalize_hand(hero_hand)
        
        # Create board representation
        if board_pattern.stage == BoardStage.PREFLOP:
            board_repr = "preflop"
        else:
            # Create normalized board representation
            board_repr = f"{board_pattern.stage.value}_{board_pattern.texture.value}_{board_pattern.rank_pattern}_{board_pattern.suit_pattern}"
        
        # Create cache key
        return CacheKey(
            hero_hand=normalized_hand,
            num_opponents=num_opponents,
            board_cards=board_repr,
            simulation_mode=simulation_mode
        )
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        hit_rate = (self._stats['cache_hits'] / self._stats['total_requests'] 
                   if self._stats['total_requests'] > 0 else 0.0)
        
        # Calculate hit rates by stage
        stage_hit_rates = {}
        for stage in BoardStage:
            requests = self._stats['requests_by_stage'][stage.value]
            hits = self._stats['hits_by_stage'][stage.value]
            stage_hit_rates[stage.value] = hits / requests if requests > 0 else 0.0
        
        # Get unified cache stats
        unified_stats = self.unified_cache.get_stats()
        
        return {
            'total_requests': self._stats['total_requests'],
            'cache_hits': self._stats['cache_hits'],
            'cache_misses': self._stats['cache_misses'],
            'hit_rate': hit_rate,
            'stage_hit_rates': stage_hit_rates,
            'pattern_frequencies': dict(self._stats['pattern_frequencies']),
            'unified_cache_stats': {
                'memory_usage_mb': unified_stats.memory_usage_mb,
                'persistence_type': unified_stats.persistence_type
            }
        }
    
    def clear_cache(self) -> bool:
        """Clear board scenario cache."""
        success = self.unified_cache.clear()
        
        if success:
            # Reset statistics
            self._stats = {
                'total_requests': 0,
                'cache_hits': 0,
                'cache_misses': 0,
                'requests_by_stage': defaultdict(int),
                'hits_by_stage': defaultdict(int),
                'pattern_frequencies': defaultdict(int)
            }
        
        return success
    
    def analyze_scenario_coverage(self) -> Dict[str, Any]:
        """Analyze cache coverage across different scenarios."""
        coverage_analysis = {
            'by_stage': {},
            'by_texture': {},
            'total_patterns': len(self._stats['pattern_frequencies']),
            'most_common_patterns': sorted(
                self._stats['pattern_frequencies'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }
        
        # Coverage by stage
        for stage in BoardStage:
            requests = self._stats['requests_by_stage'][stage.value]
            hits = self._stats['hits_by_stage'][stage.value]
            coverage_analysis['by_stage'][stage.value] = {
                'requests': requests,
                'hits': hits,
                'hit_rate': hits / requests if requests > 0 else 0.0
            }
        
        # Coverage by texture
        for texture in BoardTexture:
            freq = self._stats['pattern_frequencies'][texture.value]
            coverage_analysis['by_texture'][texture.value] = freq
        
        return coverage_analysis


# Global board cache instance
_board_cache = None
_board_cache_lock = threading.Lock()


def get_board_cache(config: Optional[BoardCacheConfig] = None,
                   unified_cache: Optional[ThreadSafeMonteCarloCache] = None) -> BoardScenarioCache:
    """Get global board cache instance."""
    global _board_cache, _board_cache_lock
    
    with _board_cache_lock:
        if _board_cache is None:
            _board_cache = BoardScenarioCache(config, unified_cache)
        return _board_cache


def clear_board_cache():
    """Clear global board cache."""
    global _board_cache, _board_cache_lock
    
    with _board_cache_lock:
        if _board_cache:
            _board_cache.clear_cache()
            _board_cache = None