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
        from .kernels import compile_kernels
    except ImportError:
        compile_kernels = None

class GPUSolver:
    """GPU-accelerated poker solver using CUDA."""
    
    def __init__(self):
        """Initialize the GPU solver."""
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available. Use CPU solver instead.")
            
        self.device = cp.cuda.Device()
        if compile_kernels is not None:
            self.kernels = compile_kernels()
        else:
            self.kernels = {}
        
        # Pre-allocate memory for common scenarios
        self.memory_pool = cp.get_default_memory_pool()
        self.pinned_memory_pool = cp.get_default_pinned_memory_pool()
        
        # Initialize lookup tables in GPU constant memory
        self._init_lookup_tables()
        
        logger.info(f"GPU Solver initialized on {self.device}")
    
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
        
        # Convert cards to GPU format
        hero_gpu = self._cards_to_gpu(hero_hand)
        board_gpu = self._cards_to_gpu(board_cards or [])
        board_size = len(board_cards) if board_cards else 0
        
        # Calculate optimal grid configuration
        threads_per_block = 256
        
        # Adaptive configuration based on workload size
        if num_simulations < 10000:
            # For small workloads, use fewer threads with more work each
            threads_per_block = 64
            simulations_per_thread = max(10, num_simulations // (threads_per_block * 4))
        else:
            # For large workloads, maximize parallelism
            simulations_per_thread = max(100, num_simulations // (10000 * threads_per_block))
        
        # Calculate number of blocks needed
        total_threads_needed = (num_simulations + simulations_per_thread - 1) // simulations_per_thread
        num_blocks = (total_threads_needed + threads_per_block - 1) // threads_per_block
        
        # Allocate output arrays
        win_counts = cp.zeros(num_blocks * threads_per_block, dtype=cp.uint32)
        tie_counts = cp.zeros(num_blocks * threads_per_block, dtype=cp.uint32)
        
        # Use the kernel wrapper for proper execution
        from .kernels import KernelWrapper
        kernel_wrapper = KernelWrapper()
        
        try:
            # Run Monte Carlo simulation
            total_wins, total_ties, total_sims = kernel_wrapper.monte_carlo(
                hero_gpu, board_gpu, board_size, num_opponents, 
                num_simulations, threads_per_block
            )
        except Exception as e:
            logger.error(f"GPU kernel execution failed: {e}")
            raise
        
        # Calculate probabilities
        win_probability = total_wins / total_sims
        tie_probability = total_ties / total_sims
        loss_probability = 1.0 - win_probability - tie_probability
        
        execution_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Calculate proper confidence interval (95% confidence)
        z_score = 1.96
        margin_of_error = z_score * math.sqrt((win_probability * (1 - win_probability)) / total_sims)
        
        # Calculate bounds and ensure they stay within [0, 1]
        lower_bound = max(0.0, win_probability - margin_of_error)
        upper_bound = min(1.0, win_probability + margin_of_error)
        
        # Round to appropriate precision (default 3 decimal places)
        precision = 3
        lower_bound = round(lower_bound, precision)
        upper_bound = round(upper_bound, precision)
        
        # Final clamp after rounding to ensure bounds stay valid
        lower_bound = max(0.0, min(1.0, lower_bound))
        upper_bound = max(0.0, min(1.0, upper_bound))
        
        confidence_interval = (lower_bound, upper_bound)
        
        return SimulationResult(
            win_probability=win_probability,
            tie_probability=tie_probability,
            loss_probability=loss_probability,
            simulations_run=total_sims,
            execution_time_ms=execution_time,
            confidence_interval=confidence_interval,
            gpu_used=True,
            backend='cuda',
            device=str(self.device)
        )
    
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
            return GPUSolver()
        except Exception as e:
            logger.warning(f"Failed to initialize GPU solver: {e}")
            return None