"""
♞ Poker Knight Specialized Preflop Cache

Dedicated cache for preflop scenarios with 169 hand combinations.
Optimized for high hit rates and memory efficiency with intelligent
pre-population strategies.

Author: hildolfr
License: MIT
"""

import time
import json
import threading
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import OrderedDict
import itertools

# Import unified cache components
from .unified_cache import (
    CacheKey, CacheResult, ThreadSafeMonteCarloCache,
    CacheKeyNormalizer, create_cache_key
)

logger = logging.getLogger(__name__)


@dataclass
class PreflopHandDefinition:
    """Definition of a preflop hand combination."""
    notation: str          # e.g., "AKs", "QQ", "72o"
    rank1: str            # e.g., "A", "Q", "7"
    rank2: str            # e.g., "K", "Q", "2"
    suited: bool          # True for suited, False for offsuit/pairs
    is_pair: bool         # True for pocket pairs
    strength_rank: int    # 1-169, where 1 is best (AA)
    category: str         # "premium", "strong", "medium", "weak", "trash"


@dataclass
class PreflopCacheConfig:
    """Configuration for preflop cache system."""
    enable_preflop_cache: bool = True
    max_memory_mb: int = 64
    enable_persistence: bool = True
    preload_on_startup: bool = True
    warming_batch_size: int = 10
    warming_delay_ms: int = 100
    target_coverage: float = 0.95  # Target 95% coverage
    priority_hands_first: bool = True
    max_opponents: int = 6
    simulation_modes: List[str] = None
    
    def __post_init__(self):
        if self.simulation_modes is None:
            self.simulation_modes = ["fast", "default", "precision"]


@dataclass
class PreflopCacheStats:
    """Statistics for preflop cache."""
    total_combinations: int = 0
    cached_combinations: int = 0
    coverage_percentage: float = 0.0
    memory_usage_mb: float = 0.0
    warming_time_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    hit_rate: float = 0.0
    last_warming: Optional[float] = None


class PreflopHandGenerator:
    """Generates and manages the 169 preflop hand combinations."""
    
    # Hand strength rankings (1 = best, 169 = worst)
    HAND_RANKINGS = None  # Will be generated on first use
    
    # Hand categories
    PREMIUM_HANDS = [
        "AA", "KK", "QQ", "JJ", "TT", "AKs", "AKo", "AQs", "AQo", "AJs", "AJo"
    ]
    
    STRONG_HANDS = [
        "99", "88", "77", "ATs", "ATo", "A9s", "KQs", "KQo", "KJs", "KJo", 
        "KTs", "QJs", "QJo", "QTs", "JTs"
    ]
    
    @classmethod
    def generate_all_hands(cls) -> List[PreflopHandDefinition]:
        """Generate all 169 preflop hand combinations."""
        hands = []
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        # Generate pocket pairs (13 hands)
        for i, rank in enumerate(ranks):
            notation = f"{rank}{rank}"
            strength_rank = i + 1  # AA=1, KK=2, ..., 22=13
            category = cls._get_hand_category(notation)
            
            hands.append(PreflopHandDefinition(
                notation=notation,
                rank1=rank,
                rank2=rank,
                suited=False,  # Pairs are neither suited nor offsuit
                is_pair=True,
                strength_rank=strength_rank,
                category=category
            ))
        
        strength_rank = 14  # Continue numbering after pairs
        
        # Generate suited hands (78 hands)
        for i, rank1 in enumerate(ranks):
            for rank2 in ranks[i+1:]:
                notation = f"{rank1}{rank2}s"
                category = cls._get_hand_category(notation)
                
                hands.append(PreflopHandDefinition(
                    notation=notation,
                    rank1=rank1,
                    rank2=rank2,
                    suited=True,
                    is_pair=False,
                    strength_rank=strength_rank,
                    category=category
                ))
                strength_rank += 1
        
        # Generate offsuit hands (78 hands)
        for i, rank1 in enumerate(ranks):
            for rank2 in ranks[i+1:]:
                notation = f"{rank1}{rank2}o"
                category = cls._get_hand_category(notation)
                
                hands.append(PreflopHandDefinition(
                    notation=notation,
                    rank1=rank1,
                    rank2=rank2,
                    suited=False,
                    is_pair=False,
                    strength_rank=strength_rank,
                    category=category
                ))
                strength_rank += 1
        
        # Sort by strength rank
        hands.sort(key=lambda h: h.strength_rank)
        
        return hands
    
    @classmethod
    def _get_hand_category(cls, notation: str) -> str:
        """Categorize hand strength."""
        if notation in cls.PREMIUM_HANDS:
            return "premium"
        elif notation in cls.STRONG_HANDS:
            return "strong"
        elif notation.startswith(('A', 'K', 'Q')) or notation in ['JJ', '99', '88']:
            return "medium"
        elif notation[0] in ['J', 'T', '9', '8'] or 'A' in notation:
            return "weak"
        else:
            return "trash"
    
    @classmethod
    def notation_to_cards(cls, notation: str) -> List[str]:
        """Convert hand notation to card list."""
        suits = ['♠', '♥', '♦', '♣']
        
        if len(notation) == 2:  # Pocket pair
            rank = notation[0]
            return [f"{rank}{suits[0]}", f"{rank}{suits[1]}"]
        elif len(notation) == 3:
            rank1, rank2, suited = notation[0], notation[1], notation[2]
            if suited == 's':  # Suited
                suit = suits[0]
                return [f"{rank1}{suit}", f"{rank2}{suit}"]
            elif suited == 'o':  # Offsuit
                return [f"{rank1}{suits[0]}", f"{rank2}{suits[1]}"]
        
        raise ValueError(f"Invalid hand notation: {notation}")
    
    @classmethod
    def get_prioritized_hands(cls) -> List[PreflopHandDefinition]:
        """Get hands ordered by priority (premium first)."""
        all_hands = cls.generate_all_hands()
        
        # Sort by category priority then by strength
        category_order = {"premium": 1, "strong": 2, "medium": 3, "weak": 4, "trash": 5}
        
        return sorted(all_hands, key=lambda h: (category_order[h.category], h.strength_rank))


class PreflopCache:
    """
    Specialized cache for preflop scenarios with intelligent warming.
    
    Built on top of the unified cache but optimized for the specific
    characteristics of preflop play (limited combinations, high reuse).
    """
    
    def __init__(self, 
                 config: Optional[PreflopCacheConfig] = None,
                 unified_cache: Optional[ThreadSafeMonteCarloCache] = None):
        
        self.config = config or PreflopCacheConfig()
        self.stats = PreflopCacheStats()
        
        # Use provided unified cache or create new one
        if unified_cache:
            self.unified_cache = unified_cache
        else:
            self.unified_cache = ThreadSafeMonteCarloCache(
                max_memory_mb=self.config.max_memory_mb,
                enable_persistence=self.config.enable_persistence,
                sqlite_path="poker_knight_preflop_cache.db"
            )
        
        # Generate hand definitions
        self.all_hands = PreflopHandGenerator.generate_all_hands()
        self.stats.total_combinations = len(self.all_hands) * self.config.max_opponents * len(self.config.simulation_modes)
        
        # Thread management for background warming
        self._warming_thread = None
        self._stop_warming = threading.Event()
        self._warming_lock = threading.Lock()
        
        logger.info(f"Preflop cache initialized with {len(self.all_hands)} hand combinations")
    
    def get_preflop_result(self, 
                          hand_notation: str,
                          num_opponents: int,
                          simulation_mode: str = "default") -> Optional[CacheResult]:
        """Get cached preflop result for specific hand/opponents/mode."""
        try:
            # Convert notation to cards
            hero_cards = PreflopHandGenerator.notation_to_cards(hand_notation)
            
            # Create cache key
            cache_key = create_cache_key(
                hero_hand=hero_cards,
                num_opponents=num_opponents,
                board_cards=None,  # Preflop
                simulation_mode=simulation_mode
            )
            
            # Get from unified cache
            result = self.unified_cache.get(cache_key)
            
            # Update statistics
            if result:
                self.stats.cache_hits += 1
            else:
                self.stats.cache_misses += 1
            
            self._update_hit_rate()
            return result
            
        except Exception as e:
            logger.error(f"Error getting preflop result for {hand_notation}: {e}")
            return None
    
    def store_preflop_result(self,
                           hand_notation: str,
                           num_opponents: int,
                           result: CacheResult,
                           simulation_mode: str = "default") -> bool:
        """Store preflop result in cache."""
        try:
            # Convert notation to cards
            hero_cards = PreflopHandGenerator.notation_to_cards(hand_notation)
            
            # Create cache key
            cache_key = create_cache_key(
                hero_hand=hero_cards,
                num_opponents=num_opponents,
                board_cards=None,  # Preflop
                simulation_mode=simulation_mode
            )
            
            # Store in unified cache
            success = self.unified_cache.store(cache_key, result)
            
            if success:
                self.stats.cached_combinations += 1
                self._update_coverage()
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing preflop result for {hand_notation}: {e}")
            return False
    
    def warm_cache(self, 
                   priority_hands_only: bool = False,
                   simulation_callback: Optional[callable] = None) -> None:
        """
        Warm the preflop cache with pre-computed results.
        
        Args:
            priority_hands_only: If True, only warm premium/strong hands
            simulation_callback: Function to call for simulation (hand_notation, opponents, mode) -> CacheResult
        """
        if not simulation_callback:
            logger.warning("No simulation callback provided for cache warming")
            return
        
        start_time = time.time()
        
        # Get hands to warm (prioritized or all)
        if priority_hands_only:
            hands_to_warm = [h for h in self.all_hands if h.category in ["premium", "strong"]]
        else:
            hands_to_warm = PreflopHandGenerator.get_prioritized_hands()
        
        total_scenarios = len(hands_to_warm) * self.config.max_opponents * len(self.config.simulation_modes)
        completed_scenarios = 0
        
        logger.info(f"Starting preflop cache warming: {total_scenarios} scenarios")
        
        for hand_def in hands_to_warm:
            if self._stop_warming.is_set():
                break
            
            for num_opponents in range(1, self.config.max_opponents + 1):
                for simulation_mode in self.config.simulation_modes:
                    if self._stop_warming.is_set():
                        break
                    
                    # Check if already cached
                    existing = self.get_preflop_result(hand_def.notation, num_opponents, simulation_mode)
                    if existing:
                        completed_scenarios += 1
                        continue
                    
                    try:
                        # Run simulation
                        result = simulation_callback(hand_def.notation, num_opponents, simulation_mode)
                        
                        if result:
                            # Store result
                            self.store_preflop_result(hand_def.notation, num_opponents, result, simulation_mode)
                            completed_scenarios += 1
                            
                            # Progress logging
                            if completed_scenarios % self.config.warming_batch_size == 0:
                                progress = (completed_scenarios / total_scenarios) * 100
                                logger.info(f"Cache warming progress: {progress:.1f}% ({completed_scenarios}/{total_scenarios})")
                            
                            # Brief pause to avoid overwhelming system
                            if self.config.warming_delay_ms > 0:
                                time.sleep(self.config.warming_delay_ms / 1000.0)
                        
                    except Exception as e:
                        logger.error(f"Error warming cache for {hand_def.notation} vs {num_opponents}: {e}")
        
        self.stats.warming_time_seconds = time.time() - start_time
        self.stats.last_warming = time.time()
        
        logger.info(f"Cache warming completed in {self.stats.warming_time_seconds:.1f}s. "
                   f"Coverage: {self.stats.coverage_percentage:.1f}%")
    
    def start_background_warming(self, simulation_callback: callable) -> None:
        """Start background cache warming in separate thread."""
        if self._warming_thread and self._warming_thread.is_alive():
            logger.warning("Background warming already running")
            return
        
        def warming_worker():
            try:
                self.warm_cache(
                    priority_hands_only=False,
                    simulation_callback=simulation_callback
                )
            except Exception as e:
                logger.error(f"Background warming failed: {e}")
        
        self._stop_warming.clear()
        self._warming_thread = threading.Thread(target=warming_worker, daemon=True)
        self._warming_thread.start()
        
        logger.info("Background cache warming started")
    
    def stop_background_warming(self) -> None:
        """Stop background cache warming."""
        self._stop_warming.set()
        
        if self._warming_thread and self._warming_thread.is_alive():
            self._warming_thread.join(timeout=5.0)
            
        logger.info("Background cache warming stopped")
    
    def get_coverage_analysis(self) -> Dict[str, Any]:
        """Get detailed coverage analysis by hand category and opponent count."""
        coverage_by_category = {}
        coverage_by_opponents = {}
        
        for hand_def in self.all_hands:
            category = hand_def.category
            if category not in coverage_by_category:
                coverage_by_category[category] = {"total": 0, "cached": 0}
            
            for num_opponents in range(1, self.config.max_opponents + 1):
                if num_opponents not in coverage_by_opponents:
                    coverage_by_opponents[num_opponents] = {"total": 0, "cached": 0}
                
                for simulation_mode in self.config.simulation_modes:
                    coverage_by_category[category]["total"] += 1
                    coverage_by_opponents[num_opponents]["total"] += 1
                    
                    # Check if cached
                    result = self.get_preflop_result(hand_def.notation, num_opponents, simulation_mode)
                    if result:
                        coverage_by_category[category]["cached"] += 1
                        coverage_by_opponents[num_opponents]["cached"] += 1
        
        # Calculate percentages
        for category_data in coverage_by_category.values():
            if category_data["total"] > 0:
                category_data["coverage"] = (category_data["cached"] / category_data["total"]) * 100
        
        for opponent_data in coverage_by_opponents.values():
            if opponent_data["total"] > 0:
                opponent_data["coverage"] = (opponent_data["cached"] / opponent_data["total"]) * 100
        
        return {
            "overall_coverage": self.stats.coverage_percentage,
            "by_category": coverage_by_category,
            "by_opponents": coverage_by_opponents,
            "total_hands": len(self.all_hands),
            "total_combinations": self.stats.total_combinations,
            "cached_combinations": self.stats.cached_combinations
        }
    
    def _update_hit_rate(self) -> None:
        """Update hit rate statistics."""
        total_requests = self.stats.cache_hits + self.stats.cache_misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.cache_hits / total_requests
    
    def _update_coverage(self) -> None:
        """Update coverage percentage."""
        if self.stats.total_combinations > 0:
            self.stats.coverage_percentage = (self.stats.cached_combinations / self.stats.total_combinations) * 100
    
    def get_stats(self) -> PreflopCacheStats:
        """Get current preflop cache statistics."""
        # Update memory usage from unified cache
        unified_stats = self.unified_cache.get_stats()
        self.stats.memory_usage_mb = unified_stats.memory_usage_mb
        
        return self.stats
    
    def clear(self) -> bool:
        """Clear preflop cache."""
        success = self.unified_cache.clear()
        
        if success:
            self.stats.cached_combinations = 0
            self.stats.coverage_percentage = 0.0
            self.stats.cache_hits = 0
            self.stats.cache_misses = 0
            self.stats.hit_rate = 0.0
        
        return success
    
    def export_coverage_report(self, filepath: str) -> bool:
        """Export detailed coverage report to JSON file."""
        try:
            coverage_analysis = self.get_coverage_analysis()
            stats_dict = asdict(self.stats)
            
            report = {
                "timestamp": time.time(),
                "preflop_cache_stats": stats_dict,
                "coverage_analysis": coverage_analysis,
                "hand_definitions": [asdict(hand) for hand in self.all_hands[:10]]  # Sample
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Coverage report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export coverage report: {e}")
            return False


# Global preflop cache instance
_preflop_cache = None
_preflop_cache_lock = threading.Lock()


def get_preflop_cache(config: Optional[PreflopCacheConfig] = None,
                     unified_cache: Optional[ThreadSafeMonteCarloCache] = None) -> PreflopCache:
    """Get global preflop cache instance."""
    global _preflop_cache, _preflop_cache_lock
    
    with _preflop_cache_lock:
        if _preflop_cache is None:
            _preflop_cache = PreflopCache(config, unified_cache)
        return _preflop_cache


def clear_preflop_cache():
    """Clear global preflop cache."""
    global _preflop_cache, _preflop_cache_lock
    
    with _preflop_cache_lock:
        if _preflop_cache:
            _preflop_cache.clear()
            _preflop_cache = None