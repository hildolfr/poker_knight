"""
Generate and manage lookup tables for GPU hand evaluation.

This module creates optimized lookup tables for fast poker hand evaluation
on the GPU, including flush detection, straight detection, and rank combinations.
"""

import numpy as np
import logging
from typing import Tuple, Dict
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = np  # Fallback to numpy for table generation

# Cache directory for lookup tables
CACHE_DIR = Path(__file__).parent / ".lookup_cache"
CACHE_DIR.mkdir(exist_ok=True)

class LookupTableGenerator:
    """Generates optimized lookup tables for GPU hand evaluation."""
    
    def __init__(self):
        self.flush_lookup = None
        self.straight_lookup = None
        self.prime_products = None
        self.rank_values = None
        
    def generate_all_tables(self) -> Dict[str, np.ndarray]:
        """Generate all lookup tables for hand evaluation."""
        logger.info("Generating poker hand evaluation lookup tables...")
        
        tables = {
            'flush_lookup': self.generate_flush_lookup(),
            'straight_lookup': self.generate_straight_lookup(),
            'prime_products': self.generate_prime_products(),
            'rank_values': self.generate_rank_values()
        }
        
        logger.info(f"Generated {len(tables)} lookup tables")
        return tables
    
    def generate_flush_lookup(self) -> np.ndarray:
        """
        Generate lookup table for flush evaluation.
        
        For any 13-bit mask representing cards of one suit,
        returns the rank of the best 5-card flush.
        """
        cache_file = CACHE_DIR / "flush_lookup.npy"
        
        if cache_file.exists():
            logger.info("Loading cached flush lookup table")
            return np.load(cache_file)
            
        logger.info("Generating flush lookup table...")
        
        # 13-bit index (2^13 = 8192 entries)
        flush_lookup = np.zeros(8192, dtype=np.uint32)
        
        # For each possible combination of cards in one suit
        for mask in range(8192):
            bits = []
            for i in range(13):
                if mask & (1 << i):
                    bits.append(i)
            
            if len(bits) >= 5:
                # Find best 5-card combination
                # Take the 5 highest cards (bits are 0=2, 1=3, ..., 12=A)
                best_five = sorted(bits, reverse=True)[:5]
                
                # Encode as rank value
                rank = 0
                for i, bit in enumerate(best_five):
                    rank |= (bit << (4 * (4 - i)))
                
                flush_lookup[mask] = rank
        
        # Cache the result
        np.save(cache_file, flush_lookup)
        logger.info(f"Cached flush lookup table to {cache_file}")
        
        return flush_lookup
    
    def generate_straight_lookup(self) -> np.ndarray:
        """
        Generate lookup table for straight detection.
        
        For any 13-bit mask representing ranks present,
        returns the rank of the best straight (0 if none).
        """
        cache_file = CACHE_DIR / "straight_lookup.npy"
        
        if cache_file.exists():
            logger.info("Loading cached straight lookup table")
            return np.load(cache_file)
            
        logger.info("Generating straight lookup table...")
        
        straight_lookup = np.zeros(8192, dtype=np.uint32)
        
        # All possible straights (including A-5)
        straights = [
            [12, 0, 1, 2, 3],    # A-2-3-4-5 (wheel)
            [0, 1, 2, 3, 4],     # 2-3-4-5-6
            [1, 2, 3, 4, 5],     # 3-4-5-6-7
            [2, 3, 4, 5, 6],     # 4-5-6-7-8
            [3, 4, 5, 6, 7],     # 5-6-7-8-9
            [4, 5, 6, 7, 8],     # 6-7-8-9-10
            [5, 6, 7, 8, 9],     # 7-8-9-10-J
            [6, 7, 8, 9, 10],    # 8-9-10-J-Q
            [7, 8, 9, 10, 11],   # 9-10-J-Q-K
            [8, 9, 10, 11, 12],  # 10-J-Q-K-A
        ]
        
        for mask in range(8192):
            best_straight = 0
            
            # Check each possible straight
            for i, straight in enumerate(straights):
                has_all = True
                for rank in straight:
                    if not (mask & (1 << rank)):
                        has_all = False
                        break
                
                if has_all:
                    # Encode the high card of the straight
                    if i == 0:  # Wheel (A-5)
                        high_card = 3  # 5 is high
                    else:
                        high_card = straight[4]
                    
                    best_straight = max(best_straight, high_card + 1)
            
            straight_lookup[mask] = best_straight
        
        # Cache the result
        np.save(cache_file, straight_lookup)
        logger.info(f"Cached straight lookup table to {cache_file}")
        
        return straight_lookup
    
    def generate_prime_products(self) -> np.ndarray:
        """
        Generate prime numbers for each rank.
        
        Used for fast duplicate detection in hand evaluation.
        """
        # First 13 prime numbers for ranks 2-A
        primes = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], 
                         dtype=np.uint16)
        return primes
    
    def generate_rank_values(self) -> np.ndarray:
        """
        Generate rank value lookup for card strength comparison.
        
        Maps card rank (0-12) to comparison value.
        """
        # Simple linear mapping for now
        # Could be adjusted for game-specific rankings
        return np.arange(13, dtype=np.uint8)
    
    def load_to_gpu(self, tables: Dict[str, np.ndarray]) -> Dict[str, cp.ndarray]:
        """
        Load lookup tables to GPU memory.
        
        Args:
            tables: Dictionary of numpy arrays
            
        Returns:
            Dictionary of cupy arrays on GPU
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
            
        gpu_tables = {}
        for name, table in tables.items():
            gpu_tables[name] = cp.asarray(table)
            logger.info(f"Loaded {name} to GPU ({table.nbytes} bytes)")
            
        return gpu_tables

class OptimizedHandEvaluator:
    """
    Optimized hand evaluator using lookup tables.
    
    This is a CPU reference implementation that mirrors the GPU logic.
    """
    
    def __init__(self, tables: Dict[str, np.ndarray]):
        self.flush_lookup = tables['flush_lookup']
        self.straight_lookup = tables['straight_lookup']
        self.prime_products = tables['prime_products']
        
    def evaluate_5_cards(self, cards: np.ndarray) -> int:
        """
        Evaluate a 5-card poker hand.
        
        Args:
            cards: Array of 5 cards (encoded as rank*4 + suit)
            
        Returns:
            Hand rank (higher is better)
        """
        ranks = cards // 4
        suits = cards % 4
        
        # Count occurrences
        rank_counts = np.bincount(ranks, minlength=13)
        suit_counts = np.bincount(suits, minlength=4)
        
        # Check for flush
        is_flush = np.any(suit_counts == 5)
        
        # Create rank mask
        rank_mask = 0
        for rank in ranks:
            rank_mask |= (1 << rank)
        
        # Check for straight
        straight_high = self.straight_lookup[rank_mask]
        is_straight = straight_high > 0
        
        # Count pairs, trips, quads
        pairs = np.sum(rank_counts == 2)
        trips = np.sum(rank_counts == 3)
        quads = np.sum(rank_counts == 4)
        
        # Determine hand type
        if is_straight and is_flush:
            return 8_000_000 + straight_high
        elif quads > 0:
            quad_rank = np.where(rank_counts == 4)[0][0]
            kicker = np.max(np.where(rank_counts > 0)[0][rank_counts > 0 != quad_rank])
            return 7_000_000 + quad_rank * 100 + kicker
        elif trips > 0 and pairs > 0:
            trip_rank = np.where(rank_counts == 3)[0][0]
            pair_rank = np.where(rank_counts == 2)[0][0]
            return 6_000_000 + trip_rank * 100 + pair_rank
        elif is_flush:
            flush_ranks = sorted(ranks, reverse=True)
            return 5_000_000 + sum(r * (100 ** (4-i)) for i, r in enumerate(flush_ranks))
        elif is_straight:
            return 4_000_000 + straight_high
        elif trips > 0:
            trip_rank = np.where(rank_counts == 3)[0][0]
            kickers = sorted([r for r in range(13) if rank_counts[r] > 0 and r != trip_rank], 
                           reverse=True)[:2]
            return 3_000_000 + trip_rank * 10000 + kickers[0] * 100 + kickers[1]
        elif pairs >= 2:
            pair_ranks = sorted(np.where(rank_counts == 2)[0], reverse=True)[:2]
            kicker = max([r for r in range(13) if rank_counts[r] > 0 and r not in pair_ranks])
            return 2_000_000 + pair_ranks[0] * 10000 + pair_ranks[1] * 100 + kicker
        elif pairs == 1:
            pair_rank = np.where(rank_counts == 2)[0][0]
            kickers = sorted([r for r in range(13) if rank_counts[r] > 0 and r != pair_rank], 
                           reverse=True)[:3]
            return 1_000_000 + pair_rank * 100000 + kickers[0] * 1000 + kickers[1] * 10 + kickers[2]
        else:
            high_cards = sorted(ranks, reverse=True)[:5]
            return sum(r * (100 ** (4-i)) for i, r in enumerate(high_cards))

# Module-level functions
def generate_lookup_tables() -> Dict[str, np.ndarray]:
    """Generate all lookup tables for GPU hand evaluation."""
    generator = LookupTableGenerator()
    return generator.generate_all_tables()

def get_lookup_tables() -> Dict[str, np.ndarray]:
    """Get lookup tables, generating if necessary."""
    # Check if all tables exist in cache
    required_tables = ['flush_lookup', 'straight_lookup']
    all_cached = all((CACHE_DIR / f"{name}.npy").exists() for name in required_tables)
    
    if all_cached:
        # Load from cache
        generator = LookupTableGenerator()
        tables = {}
        for name in required_tables:
            tables[name] = np.load(CACHE_DIR / f"{name}.npy")
        tables['prime_products'] = generator.generate_prime_products()
        tables['rank_values'] = generator.generate_rank_values()
        return tables
    else:
        # Generate all tables
        return generate_lookup_tables()