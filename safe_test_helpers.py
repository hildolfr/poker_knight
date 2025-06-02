
# Safe cache configuration that avoids deadlocks
def create_safe_cache_config():
    """Create a cache configuration that avoids common deadlock issues."""
    try:
        from poker_knight.storage.cache import CacheConfig
        return CacheConfig(
            max_memory_mb=64,
            hand_cache_size=100,
            enable_persistence=False,  # Disable Redis for safety
            enable_compression=False,  # Disable compression
            redis_host='localhost',
            redis_port=6379,
            redis_timeout=5.0,  # Short timeout
            connection_pool_size=1  # Minimal pool
        )
    except ImportError:
        return None

def create_safe_solver(**kwargs):
    """Create a solver with safe settings that avoid deadlocks."""
    try:
        from poker_knight.solver import MonteCarloSolver
        
        # Safe configuration
        safe_config = {
            'simulation_settings': {
                'parallel_processing': False,  # Disable parallel processing
                'fast_mode_simulations': 1000,
                'default_simulations': 5000,
                'precision_mode_simulations': 10000
            },
            'performance_settings': {
                'enable_intelligent_optimization': False,
                'enable_convergence_analysis': False
            }
        }
        
        return MonteCarloSolver(
            enable_caching=False,  # Disable caching
            **kwargs
        )
    except Exception as e:
        print(f"Failed to create safe solver: {e}")
        return None
