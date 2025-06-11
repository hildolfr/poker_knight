"""
Unit tests for CUDA integration in Poker Knight.

Tests GPU acceleration functionality, fallback mechanisms, and correctness
compared to CPU implementation.
"""

import pytest
import numpy as np
from typing import List, Tuple

# Skip all tests if CUDA not available
cupy = pytest.importorskip("cupy")

from poker_knight import MonteCarloSolver, solve_poker_hand
from poker_knight.cuda import CUDA_AVAILABLE, should_use_gpu, get_device_info

# Only import GPU solver if CUDA is available
if CUDA_AVAILABLE:
    from poker_knight.cuda.gpu_solver import GPUSolver


@pytest.mark.cuda
class TestCUDAAvailability:
    """Test CUDA availability detection and device info."""
    
    def test_cuda_available(self):
        """Test that CUDA availability is properly detected."""
        assert isinstance(CUDA_AVAILABLE, bool)
        
        if CUDA_AVAILABLE:
            device_info = get_device_info()
            assert device_info is not None
            assert 'name' in device_info
            assert 'compute_capability' in device_info
            assert 'total_memory' in device_info
    
    def test_should_use_gpu_logic(self):
        """Test GPU usage decision logic."""
        # Small simulations should use CPU (threshold is 1000)
        assert not should_use_gpu(999, 2)
        assert not should_use_gpu(500, 1)
        
        # Simulations >= 1000 should use GPU (if available)
        if CUDA_AVAILABLE:
            assert should_use_gpu(1000, 1)
            assert should_use_gpu(10000, 3)
            assert should_use_gpu(100000, 1)
            
        # Force flag should override threshold
        if CUDA_AVAILABLE:
            assert should_use_gpu(100, 1, force=True)


@pytest.mark.cuda
class TestGPUSolver:
    """Test GPU solver functionality."""
    
    @pytest.fixture
    def gpu_solver(self):
        """Create a GPU solver instance."""
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
        return GPUSolver()
    
    def test_gpu_solver_initialization(self, gpu_solver):
        """Test GPU solver initializes correctly."""
        assert gpu_solver is not None
        assert gpu_solver.device is not None
        assert gpu_solver.ng_solver is not None
    
    def test_simple_hand_analysis(self, gpu_solver):
        """Test basic hand analysis on GPU."""
        result = gpu_solver.analyze_hand(
            ['A♠', 'K♠'],
            2,
            None,
            10000
        )
        
        assert result is not None
        assert 0 <= result.win_probability <= 1
        assert 0 <= result.tie_probability <= 1
        assert 0 <= result.loss_probability <= 1
        assert abs(result.win_probability + result.tie_probability + result.loss_probability - 1.0) < 0.001
        assert result.backend == 'cuda-ng'
        assert result.device is not None
        assert result.gpu_used == True
    
    def test_postflop_analysis(self, gpu_solver):
        """Test post-flop hand analysis."""
        result = gpu_solver.analyze_hand(
            ['A♠', 'A♥'],
            3,
            ['K♠', 'Q♠', 'J♥'],
            50000
        )
        
        assert result is not None
        assert result.simulations_run > 0  # May not be exactly 50000 due to grid sizing
        assert result.execution_time_ms > 0
    
    @pytest.mark.parametrize("num_opponents", [1, 2, 3, 4, 5, 6])
    def test_various_opponent_counts(self, gpu_solver, num_opponents):
        """Test GPU solver with different opponent counts."""
        result = gpu_solver.analyze_hand(
            ['Q♥', 'Q♦'],
            num_opponents,
            None,
            10000
        )
        
        assert result is not None
        assert result.win_probability > 0
        
        # Win probability should generally decrease with more opponents
        if num_opponents > 1:
            result2 = gpu_solver.analyze_hand(
                ['Q♥', 'Q♦'],
                1,
                None,
                10000
            )
            assert result.win_probability <= result2.win_probability + 0.1  # Allow small variance


@pytest.mark.cuda
class TestCPUGPUConsistency:
    """Test that GPU and CPU produce consistent results."""
    
    @pytest.fixture
    def solvers(self):
        """Create both CPU and GPU solvers."""
        cpu_solver = MonteCarloSolver()
        
        if not CUDA_AVAILABLE:
            pytest.skip("CUDA not available")
            
        gpu_solver = GPUSolver()
        return cpu_solver, gpu_solver
    
    def test_preflop_consistency(self, solvers):
        """Test CPU and GPU produce similar results for pre-flop."""
        cpu_solver, gpu_solver = solvers
        
        # Use same seed for reproducibility
        np.random.seed(42)
        
        # Force CPU by temporarily disabling GPU
        original_gpu_solver = cpu_solver.gpu_solver
        cpu_solver.gpu_solver = None
        
        cpu_result = cpu_solver.analyze_hand(
            ['A♠', 'K♠'],
            2,
            None,
            'fast'  # 10k simulations
        )
        
        # Restore GPU solver
        cpu_solver.gpu_solver = original_gpu_solver
        
        # Run on GPU directly
        gpu_result = gpu_solver.analyze_hand(
            ['A♠', 'K♠'],
            2,
            None,
            10000
        )
        
        # Results should be within reasonable tolerance
        # Note: GPU uses different RNG and grid sizing, so variance can be higher
        assert abs(cpu_result.win_probability - gpu_result.win_probability) < 0.10  # 10% tolerance
        assert abs(cpu_result.tie_probability - gpu_result.tie_probability) < 0.05  # 5% tolerance
    
    def test_postflop_consistency(self, solvers):
        """Test CPU and GPU produce similar results for post-flop."""
        cpu_solver, gpu_solver = solvers
        
        board = ['Q♠', 'J♠', '10♥']
        
        # Force CPU
        original_gpu_solver = cpu_solver.gpu_solver
        cpu_solver.gpu_solver = None
        
        cpu_result = cpu_solver.analyze_hand(
            ['A♠', 'K♠'],
            3,
            board,
            'fast'
        )
        
        cpu_solver.gpu_solver = original_gpu_solver
        
        # GPU result
        gpu_result = gpu_solver.analyze_hand(
            ['A♠', 'K♠'],
            3,
            board,
            10000
        )
        
        # Compare probabilities (higher tolerance for GPU/CPU differences)
        assert abs(cpu_result.win_probability - gpu_result.win_probability) < 0.10
        assert abs(cpu_result.loss_probability - gpu_result.loss_probability) < 0.10


@pytest.mark.cuda
class TestGPUKernels:
    """Test GPU kernel functionality."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_kernel_compilation(self):
        """Test that kernels compile successfully."""
        from poker_knight.cuda.kernels import compile_kernels
        
        kernels = compile_kernels(force_recompile=True)
        assert len(kernels) > 0
        
        # Should have at least one Monte Carlo kernel
        kernel_names = list(kernels.keys())
        assert any('monte_carlo' in name for name in kernel_names)
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_poker_sim_ng_kernel(self):
        """Test kernelPokerSimNG functionality."""
        from poker_knight.cuda.poker_sim_ng import PokerSimNG
        
        solver = PokerSimNG()
        
        # Ensure kernel is loaded
        assert solver.kernel is not None, "kernelPokerSimNG not loaded"
        assert solver.kernel_single is not None, "Single hand kernel not loaded"
        
        # Test single hand simulation
        hero_hand = np.array([140, 156], dtype=np.uint8)  # A♠ A♥
        board = np.zeros(5, dtype=np.uint8)
        
        # Test simulation without category tracking
        result = solver.simulate_single(
            hero_hand, board, 0, 1, 10000,
            track_categories=False
        )
        
        assert result['total_simulations'] > 0
        assert result['wins'] > 0  # AA should win most of the time
        assert result['wins'] + result['ties'] <= result['total_simulations']
        
        # Win rate should be reasonable for AA vs 1
        win_rate = result['win_probability']
        assert 0.8 < win_rate < 0.9
        
        # Test with category tracking
        result_with_cats = solver.simulate_single(
            hero_hand, board, 0, 1, 10000,
            track_categories=True
        )
        
        assert result_with_cats['total_simulations'] > 0
        assert result_with_cats['wins'] > 0
        assert 'hand_category_frequencies' in result_with_cats
        assert isinstance(result_with_cats['hand_category_frequencies'], dict)


@pytest.mark.cuda
class TestIntegration:
    """Test full integration with main solver."""
    
    def test_solver_with_cuda_enabled(self):
        """Test that main solver properly uses GPU when available."""
        # This should automatically use GPU for large simulations
        result = solve_poker_hand(
            ['A♠', 'A♥'],
            3,
            simulation_mode='default'  # 100k simulations
        )
        
        assert result is not None
        assert result.win_probability > 0.5  # Aces should win often
        
        # Check if GPU was used
        if CUDA_AVAILABLE:
            assert result.gpu_used == True
            assert result.backend == 'cuda-ng'
    
    def test_cuda_fallback(self):
        """Test graceful fallback when GPU fails."""
        solver = MonteCarloSolver()
        
        # Temporarily disable GPU if available
        original_gpu_solver = solver.gpu_solver
        solver.gpu_solver = None
        
        # Should still work with CPU
        result = solver.analyze_hand(
            ['K♥', 'K♦'],
            2,
            None,
            'fast'
        )
        
        assert result is not None
        assert result.win_probability > 0
        assert result.gpu_used == False
        assert result.backend == 'cpu'
        
        # Restore GPU solver
        solver.gpu_solver = original_gpu_solver


@pytest.mark.cuda
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_very_small_simulations(self):
        """Test GPU with very small simulation counts."""
        gpu_solver = GPUSolver()
        
        # Should handle even 1 simulation
        result = gpu_solver.analyze_hand(
            ['7♠', '2♥'],
            1,
            None,
            1
        )
        
        assert result is not None
        assert result.simulations_run >= 1
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_full_board(self):
        """Test with complete board (river)."""
        gpu_solver = GPUSolver()
        
        result = gpu_solver.analyze_hand(
            ['A♠', 'K♠'],
            2,
            ['Q♠', 'J♠', '10♠', '9♠', '8♠'],  # Straight flush on board
            10000
        )
        
        assert result is not None
        # A♠K♠ makes a higher straight flush (royal flush), so should win
        assert result.win_probability > 0.9  # Should win most of the time
    
    def test_gpu_config_disable(self):
        """Test that GPU can be disabled via config."""
        # Create solver and manually disable GPU
        solver = MonteCarloSolver()
        
        # Force GPU to be None (simulating disabled config)
        original_gpu_solver = solver.gpu_solver
        solver.gpu_solver = None
        
        # Test should use CPU
        result = solver.analyze_hand(['A♠', 'A♥'], 2, None, 'default')
        
        assert result.gpu_used == False
        assert result.backend == 'cpu'
        
        # Also test that should_use_gpu respects the config
        solver.config['cuda_settings']['enable_cuda'] = False
        use_gpu = should_use_gpu(100000, 2) and solver.config['cuda_settings']['enable_cuda']
        assert use_gpu == False
        
        # Restore
        solver.gpu_solver = original_gpu_solver
        solver.config['cuda_settings']['enable_cuda'] = True


@pytest.mark.cuda
@pytest.mark.benchmark
class TestPerformance:
    """Performance benchmarks for GPU acceleration."""
    
    @pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")
    def test_gpu_faster_than_cpu(self):
        """Test that GPU is faster than CPU for large simulations."""
        import time
        
        # Test scenario
        hero_hand = ['A♠', 'K♠']
        num_opponents = 3
        board = ['Q♠', 'J♠', '10♥']
        num_simulations = 100000
        
        # Time CPU
        solver = MonteCarloSolver()
        original_gpu = solver.gpu_solver
        solver.gpu_solver = None
        
        cpu_start = time.perf_counter()
        cpu_result = solver.analyze_hand(hero_hand, num_opponents, board, 'default')
        cpu_time = time.perf_counter() - cpu_start
        
        solver.gpu_solver = original_gpu
        
        # Time GPU
        if solver.gpu_solver:
            gpu_start = time.perf_counter()
            gpu_result = solver.gpu_solver.analyze_hand(
                hero_hand, num_opponents, board, num_simulations
            )
            gpu_time = time.perf_counter() - gpu_start
            
            # GPU should be significantly faster
            speedup = cpu_time / gpu_time
            assert speedup > 10  # At least 10x faster
            
            # Results should be similar
            assert abs(cpu_result.win_probability - gpu_result.win_probability) < 0.02


if __name__ == "__main__":
    pytest.main([__file__, "-v"])