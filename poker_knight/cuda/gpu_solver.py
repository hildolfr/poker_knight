"""
GPU-accelerated Monte Carlo solver for Poker Knight.

This module provides the Python interface to CUDA kernels for high-performance
poker hand analysis.
"""

import numpy as np
import math
from typing import List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import cupy as cp
import logging

from ..core.results import SimulationResult
from ..constants import RANK_VALUES, SUIT_MAPPING

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

if CUDA_AVAILABLE:
    try:
        from .poker_sim_ng import PokerSimNG
    except ImportError:
        PokerSimNG = None

class GPUSolver:
    """GPU-accelerated poker solver using CUDA."""
    
    def __init__(self, config=None):
        """Initialize the GPU solver."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available. Use CPU solver instead.")
            
        self.device = cp.cuda.Device()
        self.config = config or {}
        
        # Initialize PokerSimNG solver
        if PokerSimNG is not None:
            self.ng_solver = PokerSimNG()
        else:
            raise RuntimeError("PokerSimNG not available")
        
        # Pre-allocate memory for common scenarios
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        
        # Initialize lookup tables in GPU constant memory
        self._init_lookup_tables()
        
        logger.info(f"GPU Solver initialized on {self.device} with kernelPokerSimNG")
    
    def _init_lookup_tables(self):
        """Initialize hand evaluation lookup tables in GPU memory."""
        # Generate lookup tables for flush and straight detection
        # These will be stored in constant memory for fast access
        
        # Placeholder for actual lookup table generation
        self.flush_lookup = cp.zeros(8192, dtype=cp.uint32)
        self.straight_lookup = cp.zeros(8192, dtype=cp.uint32)
        
        # Copy to constant memory (implementation depends on kernel design)
        
    def analyze_hand(
        self,
        hero_hand: List[str],
        num_opponents: int,
        board_cards: Optional[List[str]] = None,
        num_simulations: int = 100000
    ) -> SimulationResult:
        """
        Analyze a poker hand using GPU acceleration.
        
        Args:
            hero_hand: List of 2 card strings
            num_opponents: Number of opponents (1-9)  
            board_cards: Optional list of board cards (0-5)
            num_simulations: Number of Monte Carlo simulations
            
        Returns:
            SimulationResult with analysis
        """
        import time
        start_time = time.time()
        
        # Convert cards to numpy arrays for PokerSimNG
        hero_np = self._cards_to_numpy(hero_hand)
        board_np, board_size = self._board_to_numpy(board_cards)
        
        # Use NG solver
        result_data = self.ng_solver.simulate_single(
            hero_np, board_np, board_size,
            num_opponents, num_simulations,
            track_categories=self.config.get('include_hand_categories', True),
            compute_variance=True  # Enable for confidence intervals
        )
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Create SimulationResult using NG solver's method
        return self.ng_solver.create_simulation_result(
            result_data,
            execution_time_ms=execution_time
        )
    
    def _cards_to_numpy(self, cards: List[str]) -> np.ndarray:
        """Convert card strings to numpy array format."""
        if not cards:
            return np.zeros(0, dtype=np.uint8)
            
        np_cards = np.zeros(len(cards), dtype=np.uint8)
        
        for i, card in enumerate(cards):
            # Handle Unicode and ASCII representations
            if len(card) >= 2:
                rank_char = card[0]
                suit_char = card[1]
            else:
                raise ValueError(f"Invalid card format: {card}")
            
            # Convert rank - handle special case for 10
            if card.startswith('10'):
                rank = RANK_VALUES['10']
                suit_char = card[2]
            elif rank_char in RANK_VALUES:
                rank = RANK_VALUES[rank_char]
            else:
                raise ValueError(f"Invalid rank: {rank_char}")
            
            # Convert suit - handle both Unicode and letter representations
            suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3, 'S': 0, 'H': 1, 'D': 2, 'C': 3}
            if suit_char in suit_map:
                suit = suit_map[suit_char]
            else:
                raise ValueError(f"Invalid suit: {suit_char}")
            
            # Pack into 8-bit format
            np_cards[i] = (rank & 0xF) | ((suit & 0x3) << 4) | 0x80
            
        return np_cards
    
    def _board_to_numpy(self, board_cards: Optional[List[str]]) -> Tuple[np.ndarray, int]:
        """Convert board cards to numpy array and return size."""
        if not board_cards:
            return np.zeros(5, dtype=np.uint8), 0
            
        board_np = np.zeros(5, dtype=np.uint8)
        board_cards_np = self._cards_to_numpy(board_cards)
        board_np[:len(board_cards_np)] = board_cards_np
        
        return board_np, len(board_cards)
    
    def _cards_to_gpu(self, cards: List[str]):
        """Convert card strings to GPU format."""
        if not cards:
            return cp.zeros(0, dtype=cp.uint8)
            
        gpu_cards = cp.zeros(len(cards), dtype=cp.uint8)
        
        for i, card in enumerate(cards):
            # Handle Unicode and ASCII representations
            if len(card) >= 2:
                rank_char = card[0]
                suit_char = card[1]
            else:
                raise ValueError(f"Invalid card format: {card}")
            
            # Convert rank - handle special case for 10
            if card.startswith('10'):
                rank = RANK_VALUES['10']  # Use the RANK_VALUES mapping
                suit_char = card[2]
            elif rank_char in RANK_VALUES:
                rank = RANK_VALUES[rank_char]  # Already 0-indexed
            else:
                raise ValueError(f"Invalid rank: {rank_char}")
            
            # Convert suit - handle both Unicode and letter representations
            suit_map = {'♠': 0, '♥': 1, '♦': 2, '♣': 3, 'S': 0, 'H': 1, 'D': 2, 'C': 3}
            if suit_char in suit_map:
                suit = suit_map[suit_char]
            else:
                raise ValueError(f"Invalid suit: {suit_char}")
            
            # Pack into 8-bit format
            gpu_cards[i] = (rank & 0xF) | ((suit & 0x3) << 4) | 0x80
            
        return gpu_cards
    
    def _init_rng_states(self, num_states: int):
        """Initialize cuRAND states for each thread."""
        # Placeholder - actual implementation would use cuRAND
        # Use float32 and convert since CuPy random doesn't support uint32
        return (cp.random.random((num_states, 4), dtype=cp.float32) * 2**32).astype(cp.uint32)
    
    def close(self):
        """Clean up GPU resources."""
        self.memory_pool.free_all_blocks()
        self.pinned_memory_pool.free_all_blocks()

class GPUSolverFactory:
    """Factory for creating GPU solvers with fallback to CPU."""
    
    @staticmethod
    def create_solver(config: dict):
        """
        Create a solver based on configuration and hardware availability.
        
        Args:
            config: Solver configuration dictionary
            
        Returns:
            GPUSolver if CUDA available and enabled, otherwise None
        """
        if not config.get('enable_cuda', True):
            return None
            
        if not CUDA_AVAILABLE:
            logger.info("CUDA not available, using CPU solver")
            return None
            
        try:
            return GPUSolver(config)
        except Exception as e:
            logger.warning(f"Failed to initialize GPU solver: {e}")
            return None