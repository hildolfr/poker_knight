"""
Python wrapper for kernelPokerSimNG - Next Generation Unified Poker Simulation.

This module provides the Python interface to the unified CUDA kernel that
replaces all existing Monte Carlo simulation kernels.
"""

import cupy as cp
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from pathlib import Path
import logging

from ..core.results import SimulationResult
from ..constants import HAND_RANKINGS

logger = logging.getLogger(__name__)

# Configuration flags matching kernel constants
FLAG_TRACK_CATEGORIES = 1 << 0
FLAG_COMPUTE_VARIANCE = 1 << 1
FLAG_ANALYZE_BOARD = 1 << 2
FLAG_USE_SHARED_MEM = 1 << 3
FLAG_BATCH_MODE = 1 << 4

# Category mapping
CATEGORY_NAMES = {
    1: 'high_card',
    2: 'pair',
    3: 'two_pair',
    4: 'three_of_a_kind',
    5: 'straight',
    6: 'flush',
    7: 'full_house',
    8: 'four_of_a_kind',
    9: 'straight_flush',
    10: 'royal_flush'
}


class PokerSimNG:
    """Next Generation GPU Poker Simulator using unified kernel."""
    
    def __init__(self, device_id: int = 0):
        """Initialize the simulator with specified GPU device."""
        self.device = cp.cuda.Device(device_id)
        self.kernel = None
        self.kernel_single = None
        self._compile_kernel()
        
    def _compile_kernel(self):
        """Compile the kernelPokerSimNG CUDA kernel."""
        kernel_path = Path(__file__).parent / "kernel_poker_sim_ng.cu"
        
        if not kernel_path.exists():
            raise RuntimeError(f"Kernel source not found: {kernel_path}")
            
        with open(kernel_path, 'r') as f:
            kernel_source = f.read()
            
        # Compile with optimizations
        try:
            module = cp.RawModule(
                code=kernel_source,
                options=('-std=c++17', '-use_fast_math'),
                name_expressions=['kernelPokerSimNG', 'kernelPokerSimNG_single']
            )
            
            self.kernel = module.get_function('kernelPokerSimNG')
            self.kernel_single = module.get_function('kernelPokerSimNG_single')
            
            logger.info("kernelPokerSimNG compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to compile kernel: {e}")
            raise
    
    def simulate_single(
        self,
        hero_hand: np.ndarray,
        board_cards: np.ndarray,
        board_size: int,
        num_opponents: int,
        num_simulations: int,
        track_categories: bool = True,
        compute_variance: bool = False,
        analyze_board: bool = False,
        seed: Optional[int] = None
    ) -> Dict[str, Union[int, float, Dict]]:
        """
        Run simulation for a single hand.
        
        Args:
            hero_hand: Array of 2 cards (uint8)
            board_cards: Array of up to 5 cards (uint8)
            board_size: Number of valid board cards (0-5)
            num_opponents: Number of opponents (1-6)
            num_simulations: Total simulations to run
            track_categories: Track hand category frequencies
            compute_variance: Calculate variance for confidence intervals
            analyze_board: Analyze board texture
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary with simulation results
        """
        # Convert to GPU arrays
        hero_gpu = cp.asarray(hero_hand, dtype=cp.uint8)
        board_gpu = cp.asarray(board_cards, dtype=cp.uint8)
        
        # Allocate output arrays
        wins = cp.zeros(1, dtype=cp.uint32)
        ties = cp.zeros(1, dtype=cp.uint32)
        losses = cp.zeros(1, dtype=cp.uint32)
        
        # Optional outputs
        categories = cp.zeros(11, dtype=cp.uint32) if track_categories else None
        
        # Calculate grid dimensions
        threads_per_block = 256
        blocks = min(65535, (num_simulations + threads_per_block - 1) // threads_per_block)
        blocks = max(1, blocks)
        
        # Set random seed
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
            
        # Launch kernel
        self.kernel_single(
            (blocks,), (threads_per_block,),
            (
                hero_gpu, board_gpu, board_size, num_opponents,
                num_simulations, wins, ties, losses,
                categories if categories is not None else cp.zeros(1, dtype=cp.uint32),
                seed
            )
        )
        
        # Synchronize
        cp.cuda.Stream.null.synchronize()
        
        # Get results
        wins_val = int(wins[0])
        ties_val = int(ties[0])
        losses_val = int(losses[0])
        total = wins_val + ties_val + losses_val
        
        result = {
            'wins': wins_val,
            'ties': ties_val,
            'losses': losses_val,
            'total_simulations': total,
            'win_probability': wins_val / total if total > 0 else 0,
            'tie_probability': ties_val / total if total > 0 else 0,
            'loss_probability': losses_val / total if total > 0 else 0
        }
        
        # Add hand categories if tracked
        if track_categories and categories is not None:
            cat_dict = {}
            cat_array = categories.get()
            total_hands = sum(cat_array[1:])  # Skip index 0
            
            if total_hands > 0:
                for idx, name in CATEGORY_NAMES.items():
                    frequency = cat_array[idx] / total_hands
                    if frequency > 0:
                        cat_dict[name] = frequency
                        
            result['hand_category_frequencies'] = cat_dict
            
        return result
    
    def simulate_batch(
        self,
        hero_hands: np.ndarray,
        board_cards: np.ndarray,
        board_sizes: np.ndarray,
        num_opponents: np.ndarray,
        num_simulations: int,
        config_flags: int = FLAG_TRACK_CATEGORIES,
        seed: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Run simulation for multiple hands in a single kernel launch.
        
        Args:
            hero_hands: Array of hero hands (num_hands, 2) uint8
            board_cards: Array of board cards (num_hands, 5) uint8
            board_sizes: Number of board cards per hand (num_hands,) uint8
            num_opponents: Opponents per hand (num_hands,) uint8
            num_simulations: Simulations per hand
            config_flags: Configuration flags
            seed: Random seed
            
        Returns:
            Dictionary with batch results
        """
        num_hands = len(hero_hands)
        
        # Convert to GPU arrays
        hero_gpu = cp.asarray(hero_hands.reshape(-1), dtype=cp.uint8)
        board_gpu = cp.asarray(board_cards.reshape(-1), dtype=cp.uint8)
        board_sizes_gpu = cp.asarray(board_sizes, dtype=cp.uint8)
        opponents_gpu = cp.asarray(num_opponents, dtype=cp.uint8)
        
        # Allocate output arrays
        wins = cp.zeros(num_hands, dtype=cp.uint32)
        ties = cp.zeros(num_hands, dtype=cp.uint32)
        losses = cp.zeros(num_hands, dtype=cp.uint32)
        
        # Optional outputs
        categories = None
        variances = None
        textures = None
        
        if config_flags & FLAG_TRACK_CATEGORIES:
            categories = cp.zeros(num_hands * 11, dtype=cp.uint32)
        if config_flags & FLAG_COMPUTE_VARIANCE:
            variances = cp.zeros(num_hands, dtype=cp.float32)
        if config_flags & FLAG_ANALYZE_BOARD:
            textures = cp.zeros(num_hands, dtype=cp.uint16)
            
        # Create config as single uint32
        config_value = config_flags | FLAG_BATCH_MODE
        
        # Calculate grid dimensions for batch processing
        threads_per_block = 256
        blocks = num_hands  # One block per hand in batch mode
        
        # Set random seed
        if seed is None:
            seed = np.random.randint(0, 2**63 - 1)
            
        # Launch kernel
        self.kernel(
            (blocks,), (threads_per_block,),
            (
                hero_gpu, board_gpu, board_sizes_gpu, opponents_gpu,
                config_value, num_hands, num_simulations,
                wins, ties, losses,
                categories if categories is not None else cp.zeros(1, dtype=cp.uint32),
                variances if variances is not None else cp.zeros(1, dtype=cp.float32),
                textures if textures is not None else cp.zeros(1, dtype=cp.uint16),
                cp.zeros(1, dtype=cp.uint8),  # RNG states placeholder
                seed
            )
        )
        
        # Synchronize
        cp.cuda.Stream.null.synchronize()
        
        # Build results
        results = {
            'wins': wins.get(),
            'ties': ties.get(),
            'losses': losses.get(),
            'num_hands': num_hands
        }
        
        if categories is not None:
            results['hand_categories'] = categories.get().reshape(num_hands, 11)
        if variances is not None:
            results['variances'] = variances.get()
        if textures is not None:
            results['board_textures'] = textures.get()
            
        return results
    
    def create_simulation_result(
        self,
        sim_data: Dict,
        execution_time_ms: float,
        config: Optional[Dict] = None
    ) -> SimulationResult:
        """
        Create a SimulationResult object compatible with the CPU solver.
        
        Args:
            sim_data: Raw simulation data from kernel
            execution_time_ms: Total execution time
            config: Optional configuration data
            
        Returns:
            SimulationResult object
        """
        total = sim_data['total_simulations']
        
        # Calculate confidence interval 
        confidence_interval = None
        if total > 0:
            # Use binomial proportion confidence interval
            # Standard error = sqrt(p * (1 - p) / n)
            p = sim_data['win_probability']
            std_error = np.sqrt(p * (1 - p) / total)
            # 95% confidence interval
            margin = 1.96 * std_error
            # Round to 3 decimal places for consistency
            lower = round(max(0.0, p - margin), 3)
            upper = round(min(1.0, p + margin), 3)
            confidence_interval = (lower, upper)
            
        return SimulationResult(
            win_probability=sim_data['win_probability'],
            tie_probability=sim_data['tie_probability'],
            loss_probability=sim_data['loss_probability'],
            simulations_run=total,
            execution_time_ms=execution_time_ms,
            confidence_interval=confidence_interval,
            hand_category_frequencies=sim_data.get('hand_category_frequencies'),
            
            # GPU doesn't do convergence analysis
            convergence_achieved=False,
            
            # GPU-specific fields
            gpu_used=True,
            backend='cuda-ng',  # Next Generation
            device=str(self.device)
        )


def create_ng_solver():
    """Factory function to create a PokerSimNG instance."""
    try:
        return PokerSimNG()
    except Exception as e:
        logger.error(f"Failed to create PokerSimNG: {e}")
        return None