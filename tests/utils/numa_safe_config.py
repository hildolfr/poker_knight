
# NUMA and parallel processing safe configuration
import os
import multiprocessing as mp

def get_safe_parallel_config():
    """Get safe parallel processing configuration."""
    # Disable NUMA for testing if requested
    if os.environ.get('POKER_KNIGHT_DISABLE_NUMA', '0') == '1':
        return {
            'numa_aware': False,
            'numa_node_affinity': False,
            'max_processes': 1,
            'max_threads': 1
        }
    
    # Conservative parallel settings
    cpu_count = mp.cpu_count()
    return {
        'numa_aware': False,  # Disable for safety
        'numa_node_affinity': False,
        'max_processes': min(2, cpu_count // 2),  # Conservative
        'max_threads': 2,
        'complexity_threshold': 10.0  # Higher threshold
    }

def setup_safe_test_environment():
    """Set up environment variables for safe testing."""
    os.environ['POKER_KNIGHT_SAFE_MODE'] = '1'
    os.environ['POKER_KNIGHT_DISABLE_CACHE_WARMING'] = '1'
    os.environ['POKER_KNIGHT_DISABLE_NUMA'] = '1'
    os.environ['POKER_KNIGHT_DISABLE_REDIS'] = '1'
    
    print("Safe test environment configured")

# Apply safe settings
setup_safe_test_environment()
