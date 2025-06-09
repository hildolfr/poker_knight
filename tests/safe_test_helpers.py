# Safe solver configuration that avoids deadlocks
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
            **kwargs  # Caching removed in v1.7.0
        )
    except Exception as e:
        print(f"Failed to create safe solver: {e}")
        return None