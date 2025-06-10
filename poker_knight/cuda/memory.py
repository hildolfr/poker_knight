"""
GPU memory management for Poker Knight CUDA acceleration.

This module provides efficient memory allocation, pooling, and transfer
strategies for GPU computations.
"""

import logging
from typing import Dict, List, Optional, Tuple
import weakref
import threading

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cp = None

class GPUMemoryPool:
    """
    Manages GPU memory allocation with pooling and reuse.
    
    This class provides efficient memory management for repeated
    Monte Carlo simulations, minimizing allocation overhead.
    """
    
    def __init__(self, initial_size_mb: int = 512):
        """
        Initialize GPU memory pool.
        
        Args:
            initial_size_mb: Initial pool size in megabytes
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
            
        self.pool = cp.get_default_memory_pool()
        self.pinned_pool = cp.get_default_pinned_memory_pool()
        
        # Pre-allocate memory
        self.initial_size = initial_size_mb * 1024 * 1024
        self._preallocate()
        
        # Track allocations
        self.allocations: Dict[str, weakref.ref] = {}
        self.lock = threading.Lock()
        
        logger.info(f"GPU memory pool initialized with {initial_size_mb}MB")
    
    def _preallocate(self):
        """Pre-allocate memory to avoid fragmentation."""
        try:
            # Allocate and immediately free to establish pool
            temp = cp.zeros(self.initial_size // 4, dtype=cp.float32)
            del temp
            cp.cuda.Stream.null.synchronize()
        except cp.cuda.memory.OutOfMemoryError:
            logger.warning("Could not pre-allocate full pool size")
    
    def allocate(self, 
                 name: str, 
                 shape: Tuple[int, ...], 
                 dtype=cp.float32,
                 reuse: bool = True) -> cp.ndarray:
        """
        Allocate GPU memory with optional reuse.
        
        Args:
            name: Identifier for this allocation
            shape: Array shape
            dtype: Data type
            reuse: Whether to reuse existing allocation if available
            
        Returns:
            CuPy array on GPU
        """
        with self.lock:
            # Check for existing allocation
            if reuse and name in self.allocations:
                ref = self.allocations[name]
                arr = ref()
                if arr is not None and arr.shape == shape and arr.dtype == dtype:
                    return arr
            
            # Allocate new array
            arr = cp.zeros(shape, dtype=dtype)
            self.allocations[name] = weakref.ref(arr)
            
            return arr
    
    def allocate_pinned(self, shape: Tuple[int, ...], dtype=cp.float32) -> cp.ndarray:
        """
        Allocate pinned (page-locked) memory for faster CPU-GPU transfer.
        
        Args:
            shape: Array shape
            dtype: Data type
            
        Returns:
            Pinned memory array
        """
        # Use pinned memory pool
        with cp.cuda.using_allocator(self.pinned_pool.malloc):
            return cp.zeros(shape, dtype=dtype)
    
    def get_info(self) -> Dict[str, int]:
        """Get memory pool statistics."""
        return {
            'used_bytes': self.pool.used_bytes(),
            'total_bytes': self.pool.total_bytes(),
            'pinned_used_bytes': self.pinned_pool.n_free_blocks(),
            'active_allocations': len(self.allocations)
        }
    
    def clear(self):
        """Clear all cached allocations."""
        with self.lock:
            self.allocations.clear()
        self.pool.free_all_blocks()
        self.pinned_pool.free_all_blocks()

class BatchMemoryManager:
    """
    Manages memory for batch processing of multiple poker hands.
    
    Optimizes memory layout for coalesced access patterns.
    """
    
    def __init__(self, max_batch_size: int = 1000):
        """
        Initialize batch memory manager.
        
        Args:
            max_batch_size: Maximum number of hands in a batch
        """
        self.max_batch_size = max_batch_size
        self.memory_pool = GPUMemoryPool()
        
        # Pre-allocate batch buffers
        self._init_batch_buffers()
    
    def _init_batch_buffers(self):
        """Initialize reusable batch processing buffers."""
        # Input buffers
        self.hero_hands_buffer = self.memory_pool.allocate(
            'hero_hands', (self.max_batch_size, 2), dtype=cp.uint8
        )
        self.board_cards_buffer = self.memory_pool.allocate(
            'board_cards', (self.max_batch_size, 5), dtype=cp.uint8
        )
        self.board_sizes_buffer = self.memory_pool.allocate(
            'board_sizes', (self.max_batch_size,), dtype=cp.int32
        )
        self.num_opponents_buffer = self.memory_pool.allocate(
            'num_opponents', (self.max_batch_size,), dtype=cp.int32
        )
        
        # Output buffers
        self.results_buffer = self.memory_pool.allocate(
            'results', (self.max_batch_size, 3), dtype=cp.uint32
        )
        
        # RNG states (48 bytes per thread, assuming 256 threads per hand)
        self.rng_states_buffer = self.memory_pool.allocate(
            'rng_states', (self.max_batch_size * 256 * 48,), dtype=cp.uint8
        )
    
    def prepare_batch(self, 
                     hands_data: List[Dict]) -> Tuple[cp.ndarray, ...]:
        """
        Prepare batch data for GPU processing.
        
        Args:
            hands_data: List of hand dictionaries with keys:
                - hero_hand: List of 2 cards
                - board_cards: List of 0-5 cards
                - num_opponents: Number of opponents
                
        Returns:
            Tuple of GPU arrays ready for kernel execution
        """
        batch_size = len(hands_data)
        if batch_size > self.max_batch_size:
            raise ValueError(f"Batch size {batch_size} exceeds maximum {self.max_batch_size}")
        
        # Fill buffers
        for i, hand in enumerate(hands_data):
            # Hero hand
            self.hero_hands_buffer[i, 0] = hand['hero_hand'][0]
            self.hero_hands_buffer[i, 1] = hand['hero_hand'][1]
            
            # Board cards (pad with zeros)
            board = hand.get('board_cards', [])
            board_size = len(board)
            self.board_sizes_buffer[i] = board_size
            
            for j in range(5):
                if j < board_size:
                    self.board_cards_buffer[i, j] = board[j]
                else:
                    self.board_cards_buffer[i, j] = 0
            
            # Opponents
            self.num_opponents_buffer[i] = hand['num_opponents']
        
        # Return views of the actual batch size
        return (
            self.hero_hands_buffer[:batch_size],
            self.board_cards_buffer[:batch_size],
            self.board_sizes_buffer[:batch_size],
            self.num_opponents_buffer[:batch_size],
            self.results_buffer[:batch_size],
            self.rng_states_buffer[:batch_size * 256 * 48]
        )
    
    def get_results(self, batch_size: int) -> cp.ndarray:
        """Extract results for the processed batch."""
        return self.results_buffer[:batch_size].copy()

class StreamManager:
    """
    Manages CUDA streams for concurrent kernel execution.
    
    Enables overlapping computation and memory transfers.
    """
    
    def __init__(self, num_streams: int = 4):
        """
        Initialize stream manager.
        
        Args:
            num_streams: Number of CUDA streams to create
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA not available")
            
        self.streams = [cp.cuda.Stream() for _ in range(num_streams)]
        self.current_stream = 0
        
    def get_next_stream(self) -> cp.cuda.Stream:
        """Get the next available stream in round-robin fashion."""
        stream = self.streams[self.current_stream]
        self.current_stream = (self.current_stream + 1) % len(self.streams)
        return stream
    
    def synchronize_all(self):
        """Synchronize all streams."""
        for stream in self.streams:
            stream.synchronize()

# Utility functions
def estimate_memory_requirements(
    num_simulations: int,
    num_opponents: int,
    batch_size: int = 1
) -> Dict[str, int]:
    """
    Estimate GPU memory requirements for a simulation.
    
    Args:
        num_simulations: Number of Monte Carlo iterations
        num_opponents: Number of opponents
        batch_size: Number of hands to process
        
    Returns:
        Dictionary with memory estimates in bytes
    """
    # Assume 256 threads per block, 100 simulations per thread
    num_threads = min(10000, num_simulations // 100)
    
    estimates = {
        'input_cards': batch_size * 7 * 1,  # 7 cards max per hand
        'rng_states': num_threads * 48,      # cuRAND state size
        'output_results': batch_size * 3 * 4, # wins, ties, total
        'lookup_tables': 8192 * 4 * 2,       # flush + straight tables
        'shared_memory': 256 * 4 * 2,        # per block
        'total': 0
    }
    
    estimates['total'] = sum(estimates.values())
    
    return estimates

def optimize_batch_size(available_memory: int, 
                       simulation_params: Dict) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        available_memory: Available GPU memory in bytes
        simulation_params: Parameters including num_simulations, num_opponents
        
    Returns:
        Optimal batch size
    """
    # Estimate memory per hand
    per_hand_memory = estimate_memory_requirements(
        simulation_params['num_simulations'],
        simulation_params['num_opponents'],
        batch_size=1
    )['total']
    
    # Leave 20% buffer for other operations
    usable_memory = int(available_memory * 0.8)
    
    # Calculate batch size
    batch_size = max(1, usable_memory // per_hand_memory)
    
    # Cap at reasonable maximum
    return min(batch_size, 1000)