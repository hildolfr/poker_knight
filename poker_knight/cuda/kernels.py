"""
Kernel compilation and management for Poker Knight CUDA acceleration.

This module handles dynamic compilation of CUDA kernels and provides
Python interfaces to the compiled functions.
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupy import RawKernel, RawModule
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    RawKernel = None
    RawModule = None

# Cache directory for compiled kernels
CACHE_DIR = Path(__file__).parent / ".cuda_cache"
CACHE_DIR.mkdir(exist_ok=True)

class KernelManager:
    """Manages CUDA kernel compilation and caching."""
    
    def __init__(self):
        self.kernels: Dict[str, RawKernel] = {}
        self.module = None
        self._kernel_source = None
        
    def get_kernel_source(self) -> str:
        """Load and return the CUDA kernel source code."""
        if self._kernel_source is None:
            # Try improved kernel first
            try:
                self._kernel_source = self._get_improved_kernel_source()
                logger.info("Loaded improved kernel source")
            except Exception as e:
                logger.warning(f"Failed to load improved kernel: {e}")
                self._kernel_source = self._get_embedded_kernel_source()
                    
        return self._kernel_source
    
    def _get_improved_kernel_source(self) -> str:
        """Load the improved kernel source."""
        # Try optimized version first
        optimized_path = Path(__file__).parent / "kernel_optimized.cu"
        if optimized_path.exists():
            return optimized_path.read_text()
        
        # Fall back to v2
        kernel_path = Path(__file__).parent / "kernel_v2.cu"
        if kernel_path.exists():
            return kernel_path.read_text()
        return self._get_embedded_kernel_source()
    
    def _get_embedded_kernel_source(self) -> str:
        """Return embedded minimal kernel source for testing."""
        return '''
#include <cuda_runtime.h>
#include <curand_kernel.h>

typedef unsigned char Card;

// Card rank extraction (0-12 for 2-A)
__device__ inline int get_rank(Card c) {
    return c & 0xF;
}

// Card suit extraction (0-3)
__device__ inline int get_suit(Card c) {
    return (c >> 4) & 0x3;
}

// Simple hand strength evaluation
__device__ unsigned int evaluate_hand_simple(const Card cards[7], int num_cards) {
    // Count ranks
    int rank_counts[13] = {0};
    int suit_counts[4] = {0};
    
    for (int i = 0; i < num_cards; i++) {
        if (cards[i] & 0x80) {  // Valid card
            rank_counts[get_rank(cards[i])]++;
            suit_counts[get_suit(cards[i])]++;
        }
    }
    
    // Check for pairs, trips, quads
    int pairs = 0, trips = 0, quads = 0;
    int highest_rank = 0;
    
    for (int r = 12; r >= 0; r--) {
        if (rank_counts[r] == 4) quads++;
        else if (rank_counts[r] == 3) trips++;
        else if (rank_counts[r] == 2) pairs++;
        if (rank_counts[r] > 0 && highest_rank == 0) highest_rank = r;
    }
    
    // Simple scoring system
    if (quads > 0) return 7000 + highest_rank;
    if (trips > 0 && pairs > 0) return 6000 + highest_rank;  // Full house
    if (trips > 0) return 3000 + highest_rank;
    if (pairs >= 2) return 2000 + highest_rank;
    if (pairs == 1) return 1000 + highest_rank;
    return highest_rank;  // High card
}

// Minimal Monte Carlo kernel
extern "C" __global__ void monte_carlo_kernel_simple(
    const Card* hero_hand,
    int num_opponents,
    int simulations_per_thread,
    unsigned int* wins,
    unsigned int* ties
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Create pseudo-random generator
    unsigned int seed = idx * 1103515245 + 12345;
    
    unsigned int thread_wins = 0;
    unsigned int thread_ties = 0;
    
    // Build 7-card hand for hero (just 2 cards for now)
    Card hero_cards[7];
    hero_cards[0] = hero_hand[0];
    hero_cards[1] = hero_hand[1];
    for (int i = 2; i < 7; i++) hero_cards[i] = 0;
    
    unsigned int hero_strength = evaluate_hand_simple(hero_cards, 2);
    
    for (int sim = 0; sim < simulations_per_thread; sim++) {
        // Simple random opponent strength
        seed = seed * 1103515245 + 12345;
        unsigned int random_strength = (seed >> 16) % 8000;
        
        // Adjust based on number of opponents
        for (int opp = 1; opp < num_opponents; opp++) {
            seed = seed * 1103515245 + 12345;
            unsigned int opp_strength = (seed >> 16) % 8000;
            if (opp_strength > random_strength) {
                random_strength = opp_strength;
            }
        }
        
        if (hero_strength > random_strength) {
            thread_wins++;
        } else if (hero_strength == random_strength) {
            thread_ties++;
        }
    }
    
    atomicAdd(&wins[0], thread_wins);
    atomicAdd(&ties[0], thread_ties);
}

// RNG initialization (simplified)
extern "C" __global__ void init_rng_simple(curandState* states, unsigned long seed) {
    // Not used in simple version
}
'''
    
    def compile_kernels(self, force_recompile: bool = False) -> Dict[str, RawKernel]:
        """
        Compile CUDA kernels with caching support.
        
        Args:
            force_recompile: Force recompilation even if cached
            
        Returns:
            Dictionary of compiled kernels
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
            
        if self.kernels and not force_recompile:
            return self.kernels
            
        # Get kernel source
        source = self.get_kernel_source()
        
        # Generate cache key
        source_hash = hashlib.md5(source.encode()).hexdigest()
        cache_file = CACHE_DIR / f"kernels_{source_hash}.cubin"
        
        # Try to load from cache
        if cache_file.exists() and not force_recompile:
            try:
                logger.info(f"Loading cached kernels from {cache_file}")
                self.module = cp.RawModule(path=str(cache_file))
            except Exception as e:
                logger.warning(f"Failed to load cached kernels: {e}")
                self.module = None
                
        # Compile if needed
        if self.module is None:
            logger.info("Compiling CUDA kernels...")
            
            # Compile options for optimization
            # Detect compute capability
            device = cp.cuda.Device()
            cc_major = device.compute_capability[0]
            cc_minor = device.compute_capability[1]
            arch = f'sm_{cc_major}{cc_minor}'
            
            options = (
                '-std=c++17',
                '-use_fast_math',
            )
            
            try:
                # Try full kernel compilation
                self.module = cp.RawModule(code=source, options=options)
                
                # Skip caching for now - CuPy handles its own caching
                logger.info("Compiled kernels successfully")
                    
            except Exception as e:
                logger.warning(f"Failed to compile full kernels: {e}")
                logger.info("Falling back to simplified kernels")
                
                # Fallback to simplified kernels
                simple_source = self._get_embedded_kernel_source()
                self.module = cp.RawModule(code=simple_source, options=options[:4])
        
        # Extract kernel functions based on what's actually in the source
        kernel_loaded = False
        
        # Try optimized kernel first
        if "monte_carlo_optimized" in source:
            try:
                self.kernels['monte_carlo_optimized'] = self.module.get_function('monte_carlo_optimized')
                self.kernels['monte_carlo_improved'] = self.kernels['monte_carlo_optimized']  # Alias
                logger.info("Loaded monte_carlo_optimized kernel")
                kernel_loaded = True
            except Exception as e:
                logger.error(f"Failed to load monte_carlo_optimized: {e}")
        
        # Try improved kernel
        if not kernel_loaded and "monte_carlo_improved" in source:
            try:
                self.kernels['monte_carlo_improved'] = self.module.get_function('monte_carlo_improved')
                logger.info("Loaded monte_carlo_improved kernel")
                kernel_loaded = True
            except Exception as e:
                logger.error(f"Failed to load monte_carlo_improved: {e}")
        
        # Fallback to simple kernel
        if not kernel_loaded:
            try:
                self.kernels['monte_carlo_kernel_simple'] = self.module.get_function('monte_carlo_kernel_simple')
                logger.info("Loaded monte_carlo_kernel_simple")
            except Exception as e:
                logger.error(f"Failed to load simple kernel: {e}")
                        
        return self.kernels
    
    def get_kernel(self, name: str) -> Optional[RawKernel]:
        """Get a specific compiled kernel."""
        if not self.kernels:
            self.compile_kernels()
        return self.kernels.get(name)

# Global kernel manager instance
_kernel_manager = None

def get_kernel_manager() -> KernelManager:
    """Get the global kernel manager instance."""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = KernelManager()
    return _kernel_manager

def compile_kernels(force_recompile: bool = False) -> Dict[str, RawKernel]:
    """
    Compile and return all CUDA kernels.
    
    Args:
        force_recompile: Force recompilation even if cached
        
    Returns:
        Dictionary mapping kernel names to compiled kernels
    """
    manager = get_kernel_manager()
    return manager.compile_kernels(force_recompile)

def get_kernel(name: str) -> Optional[RawKernel]:
    """
    Get a specific compiled kernel by name.
    
    Args:
        name: Name of the kernel function
        
    Returns:
        Compiled kernel or None if not found
    """
    manager = get_kernel_manager()
    return manager.get_kernel(name)

# Kernel wrapper functions for easier use
class KernelWrapper:
    """Provides high-level interface to CUDA kernels."""
    
    def __init__(self):
        self.kernels = compile_kernels()
        
    def monte_carlo(
        self,
        hero_hand: cp.ndarray,
        board_cards: cp.ndarray,
        board_size: int,
        num_opponents: int,
        num_simulations: int,
        threads_per_block: int = 256
    ) -> tuple:
        """
        Run Monte Carlo simulation on GPU.
        
        Returns:
            Tuple of (wins, ties, total_simulations)
        """
        # Calculate grid dimensions for better GPU utilization
        # Target: Use many more threads for better occupancy
        
        # GTX 1060: 10 SMs, 2048 threads per SM max
        # Aim for at least 50% occupancy = 10,000+ threads
        min_threads = 10240  # 10 SMs * 1024 threads
        
        # Calculate threads needed based on work per thread
        target_sims_per_thread = 100  # Lower for better distribution
        
        if num_simulations < 100000:
            # For small workloads, still use many threads
            total_threads = min_threads
            simulations_per_thread = max(1, num_simulations // total_threads)
        else:
            # For large workloads, scale up
            needed_threads = (num_simulations + target_sims_per_thread - 1) // target_sims_per_thread
            total_threads = max(min_threads, needed_threads)
            
            # Cap at reasonable limit
            max_threads = 81920  # 320 blocks * 256 threads
            total_threads = min(total_threads, max_threads)
        
        # Calculate blocks (ensure multiple of SM count for better distribution)
        num_blocks = (total_threads + threads_per_block - 1) // threads_per_block
        sm_count = 10  # GTX 1060 has 10 SMs
        if num_blocks < sm_count * 2:
            num_blocks = sm_count * 2  # At least 2 blocks per SM
        
        # Recalculate actual threads
        total_threads = num_blocks * threads_per_block
        
        # Distribute work evenly
        simulations_per_thread = num_simulations // total_threads
        if simulations_per_thread == 0:
            simulations_per_thread = 1
            
        # Calculate actual total simulations that will be run
        actual_total_sims = total_threads * simulations_per_thread
        
        # Debug output
        logger.debug(f"Grid config: {num_blocks} blocks x {threads_per_block} threads = {total_threads} threads")
        logger.debug(f"Simulations per thread: {simulations_per_thread}")
        logger.debug(f"Actual total simulations: {actual_total_sims}")
        
        # Allocate output arrays
        wins = cp.zeros(1, dtype=cp.uint32)
        ties = cp.zeros(1, dtype=cp.uint32)
        
        # Try to get improved kernel first
        mc_kernel = self.kernels.get('monte_carlo_improved')
        
        if mc_kernel is not None:
            # Use improved kernel with proper RNG
            seed = np.uint64(42)  # Fixed seed for reproducibility (uint64 for optimized kernel)
            mc_kernel(
                (num_blocks,), (threads_per_block,),
                (hero_hand, board_cards, board_size, num_opponents, 
                 simulations_per_thread, wins, ties, seed)
            )
        else:
            # Fall back to simple kernel
            mc_kernel = self.kernels.get('monte_carlo_kernel_simple',
                                       self.kernels.get('monte_carlo'))
            if mc_kernel is None:
                raise RuntimeError("No Monte Carlo kernel available")
            
            # Run simple kernel (no board cards or RNG seed)
            mc_kernel(
                (num_blocks,), (threads_per_block,),
                (hero_hand, num_opponents, simulations_per_thread, wins, ties)
            )
        
        # Synchronize and return results
        cp.cuda.Stream.null.synchronize()
        
        # Return the actual number of simulations run
        return int(wins[0]), int(ties[0]), actual_total_sims