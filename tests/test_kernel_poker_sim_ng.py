"""Comprehensive test suite for kernelPokerSimNG."""

import pytest
import numpy as np
from poker_knight.cuda.poker_sim_ng import PokerSimNG, FLAG_TRACK_CATEGORIES
from poker_knight import solve_poker_hand, MonteCarloSolver

# Skip all tests if CUDA not available
cupy = pytest.importorskip("cupy")


@pytest.mark.cuda
class TestKernelPokerSimNG:
    """Test suite for the unified kernelPokerSimNG."""
    
    @pytest.fixture
    def solver(self):
        """Create a PokerSimNG instance."""
        return PokerSimNG()
    
    def test_single_hand_simulation(self, solver):
        """Test basic single hand simulation."""
        # AA vs 1 opponent
        hero_hand = np.array([140, 156], dtype=np.uint8)  # A♠ A♥
        board = np.zeros(5, dtype=np.uint8)
        
        result = solver.simulate_single(
            hero_hand, board, 0, 1, 10000,
            track_categories=False
        )
        
        assert result['total_simulations'] > 0
        assert result['wins'] > 0
        assert result['wins'] + result['ties'] + result['losses'] == result['total_simulations']
        assert 0.80 < result['win_probability'] < 0.90  # AA should win ~85%
    
    def test_batch_processing(self, solver):
        """Test batch processing multiple hands."""
        # Test 3 different hands
        hero_hands = np.array([
            [140, 156],  # A♠ A♥
            [139, 155],  # K♠ K♥
            [130, 146],  # 4♠ 4♥
        ], dtype=np.uint8)
        
        board_cards = np.zeros((3, 5), dtype=np.uint8)
        board_sizes = np.zeros(3, dtype=np.uint8)
        num_opponents = np.array([2, 2, 2], dtype=np.uint8)
        
        results = solver.simulate_batch(
            hero_hands, board_cards, board_sizes,
            num_opponents, 10000
        )
        
        wins = results['wins']
        assert len(wins) == 3
        
        # Verify win rates make sense (AA > KK > 44)
        win_rates = wins / (results['wins'] + results['ties'] + results['losses'])
        assert win_rates[0] > win_rates[1]  # AA > KK
        assert win_rates[1] > win_rates[2]  # KK > 44
    
    def test_configuration_flags(self, solver):
        """Test different configuration flags."""
        hero_hand = np.array([140, 156], dtype=np.uint8)
        board = np.zeros(5, dtype=np.uint8)
        
        # Test with categories disabled
        result_no_cats = solver.simulate_single(
            hero_hand, board, 0, 1, 5000,
            track_categories=False
        )
        assert 'hand_category_frequencies' not in result_no_cats
        
        # Test with categories enabled
        result_with_cats = solver.simulate_single(
            hero_hand, board, 0, 1, 5000,
            track_categories=True
        )
        assert 'hand_category_frequencies' in result_with_cats
        assert isinstance(result_with_cats['hand_category_frequencies'], dict)
        assert len(result_with_cats['hand_category_frequencies']) > 0
    
    def test_hand_category_tracking(self, solver):
        """Test hand category frequency tracking."""
        # Mixed hand possibilities
        hero_hand = np.array([140, 139], dtype=np.uint8)  # A♠ K♠
        board = np.array([138, 130, 146, 0, 0], dtype=np.uint8)  # Q♠ 4♣ 4♥
        
        result = solver.simulate_single(
            hero_hand, board, 3, 1, 10000,
            track_categories=True
        )
        
        categories = result['hand_category_frequencies']
        assert len(categories) > 0
        
        # Should have various hand types with this board
        # We have a pair on board, so at minimum we'll see pairs
        assert 'pair' in categories or 'two_pair' in categories or 'three_of_a_kind' in categories
        
        # All frequencies should sum to ~1.0
        total_freq = sum(categories.values())
        assert 0.99 < total_freq < 1.01
    
    def test_accuracy_vs_cpu(self):
        """Compare accuracy against CPU implementation."""
        test_cases = [
            {
                'hero': ['A♠', 'A♥'],
                'opponents': 2,
                'board': [],
                'expected_win': 0.73  # ~73% for AA vs 2
            },
            {
                'hero': ['K♥', 'Q♥'],
                'opponents': 1,
                'board': ['J♥', '10♥', '9♣'],
                'expected_win': 0.7  # Straight with flush draw
            }
        ]
        
        for case in test_cases:
            # Force CPU
            cpu_solver = MonteCarloSolver()
            cpu_solver.gpu_solver = None
            
            cpu_result = cpu_solver.analyze_hand(
                case['hero'],
                case['opponents'],
                case['board'],
                'default'  # 100k simulations
            )
            
            # GPU result (automatic)
            gpu_result = solve_poker_hand(
                case['hero'],
                case['opponents'],
                case['board'],
                'default'
            )
            
            # Compare probabilities (allow 2% difference)
            assert abs(cpu_result.win_probability - gpu_result.win_probability) < 0.02
            assert gpu_result.backend == 'cuda-ng'
            assert gpu_result.gpu_used == True
            
            # Both should have hand categories
            assert cpu_result.hand_category_frequencies is not None
            assert gpu_result.hand_category_frequencies is not None
    
    def test_edge_cases(self, solver):
        """Test edge cases and boundary conditions."""
        hero_hand = np.array([140, 156], dtype=np.uint8)
        
        # Test with full board (river)
        full_board = np.array([139, 138, 137, 136, 135], dtype=np.uint8)
        result = solver.simulate_single(
            hero_hand, full_board, 5, 1, 1000,
            track_categories=True
        )
        assert result['total_simulations'] > 0
        
        # Test with 6 opponents (maximum)
        board = np.zeros(5, dtype=np.uint8)
        result = solver.simulate_single(
            hero_hand, board, 0, 6, 1000,
            track_categories=False
        )
        assert result['total_simulations'] > 0
        assert result['win_probability'] < 0.5  # Should be lower with 6 opponents
    
    def test_performance_characteristics(self, solver):
        """Test performance characteristics of the kernel."""
        import time
        
        hero_hand = np.array([140, 156], dtype=np.uint8)
        board = np.zeros(5, dtype=np.uint8)
        
        # Warm up
        solver.simulate_single(hero_hand, board, 0, 2, 1000)
        
        # Time 100k simulations
        start = time.perf_counter()
        result = solver.simulate_single(
            hero_hand, board, 0, 2, 100000,
            track_categories=True
        )
        elapsed = (time.perf_counter() - start) * 1000  # ms
        
        # Should be very fast (< 50ms for 100k simulations)
        assert elapsed < 50
        assert result['total_simulations'] >= 100000
        
        # Performance should scale well
        # Test batch of 10 hands
        hero_hands = np.tile(hero_hand, (10, 1))
        board_cards = np.zeros((10, 5), dtype=np.uint8)
        board_sizes = np.zeros(10, dtype=np.uint8)
        num_opponents = np.full(10, 2, dtype=np.uint8)
        
        start = time.perf_counter()
        batch_results = solver.simulate_batch(
            hero_hands, board_cards, board_sizes,
            num_opponents, 10000
        )
        batch_elapsed = (time.perf_counter() - start) * 1000
        
        # Batch should be efficient (< 5ms per hand)
        assert batch_elapsed < 50  # 10 hands * 5ms
        assert len(batch_results['wins']) == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])