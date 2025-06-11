"""
CUDA acceleration module for Poker Knight.

This module provides GPU-accelerated Monte Carlo simulations for poker hand analysis.
Falls back gracefully to CPU implementation if CUDA is not available.
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Check CUDA availability
CUDA_AVAILABLE = False
CUDA_ERROR_MESSAGE = ""

try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    if CUDA_AVAILABLE:
        logger.info(f"CUDA available with {cp.cuda.runtime.getDeviceCount()} device(s)")
    else:
        CUDA_ERROR_MESSAGE = "No CUDA devices found"
except ImportError:
    CUDA_ERROR_MESSAGE = "CuPy not installed. Install with: pip install cupy-cuda11x"
except Exception as e:
    CUDA_ERROR_MESSAGE = f"CUDA initialization error: {str(e)}"

if not CUDA_AVAILABLE:
    logger.warning(f"CUDA not available: {CUDA_ERROR_MESSAGE}")

# Constants for GPU memory layout
CARD_BITS = 8
DECK_SIZE = 52
MAX_OPPONENTS = 9
THREADS_PER_BLOCK = 256

# Simulation thresholds for GPU usage
MIN_SIMULATIONS_FOR_GPU = 1000  # Reduced threshold - use GPU for most cases
OPTIMAL_SIMULATIONS_PER_THREAD = 1000  # Balance between parallelism and work per thread

def should_use_gpu(num_simulations: int, num_opponents: int, force: bool = False) -> bool:
    """
    Determine if GPU acceleration should be used based on problem size.
    
    Args:
        num_simulations: Number of Monte Carlo simulations
        num_opponents: Number of opponents in the hand
        force: If True, always use GPU when available (ignores thresholds)
        
    Returns:
        True if GPU should be used, False otherwise
    """
    if not CUDA_AVAILABLE:
        return False
    
    # If force flag is set, always use GPU
    if force:
        return True
        
    # Still avoid GPU for extremely small simulations to prevent errors
    if num_simulations < MIN_SIMULATIONS_FOR_GPU:
        return False
        
    # Otherwise, always prefer GPU when available
    return True

def get_device_info() -> Optional[dict]:
    """Get information about available CUDA devices."""
    if not CUDA_AVAILABLE:
        return None
        
    try:
        import cupy as cp
        device = cp.cuda.Device()
        props = device.attributes
        
        # Get device name using runtime API
        device_id = cp.cuda.runtime.getDevice()
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        
        return {
            'name': device_props['name'].decode('utf-8'),
            'compute_capability': f"{device.compute_capability[0]}.{device.compute_capability[1]}",
            'total_memory': device.mem_info[1],  # Total memory
            'multiprocessors': props.get('MultiProcessorCount', 'Unknown'),
            'max_threads_per_block': props.get('MaxThreadsPerBlock', 1024),
            'max_threads_per_mp': props.get('MaxThreadsPerMultiprocessor', 2048)
        }
    except Exception as e:
        logger.error(f"Error getting device info: {e}")
        return None

# Hand category tracking is now built into kernelPokerSimNG

__all__ = [
    'CUDA_AVAILABLE',
    'should_use_gpu',
    'get_device_info'
]