"""
Simulation execution logic for Poker Knight.

This module contains the SimulationRunner class that handles the actual
Monte Carlo simulation execution, both sequential and parallel.
"""

import time
import random
from typing import List, Tuple, Optional, Dict, Any, Counter
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..core.cards import Card, Deck
from ..core.evaluation import HandEvaluator
from ..constants import HAND_RANKINGS


class SimulationRunner:
    """Handles Monte Carlo simulation execution."""
    
    def __init__(self, config: Dict[str, Any], evaluator: HandEvaluator):
        """Initialize the simulation runner with configuration."""
        self.config = config
        self.evaluator = evaluator
        self._thread_pool = None
        self._max_workers = config["simulation_settings"].get("max_workers", 4)
        self._lock = threading.Lock()
    
    def close(self) -> None:
        """Cleanup thread pool resources."""
        with self._lock:
            if self._thread_pool is not None:
                self._thread_pool.shutdown(wait=True)
                self._thread_pool = None
    
    def _get_thread_pool(self) -> ThreadPoolExecutor:
        """Get or create the persistent thread pool with thread-safe access."""
        with self._lock:
            if self._thread_pool is None:
                self._thread_pool = ThreadPoolExecutor(max_workers=self._max_workers)
            return self._thread_pool
    
    def simulate_hand(self, hero_cards: List[Card], num_opponents: int, 
                     board: List[Card], removed_cards: List[Card]) -> Dict[str, Any]:
        """Simulate a single hand with memory optimizations."""
        # Create deck with removed cards
        deck = Deck(removed_cards)
        
        # Pre-allocate opponent hands list to avoid resize
        opponent_hands = []
        opponent_hands.extend([deck.deal(2) for _ in range(num_opponents)])
        
        # Complete the board if needed (reuse board list)
        cards_needed = 5 - len(board)
        if cards_needed > 0:
            board_cards = board + deck.deal(cards_needed)
        else:
            board_cards = board
        
        # Evaluate hero hand
        hero_rank, hero_tiebreakers = self.evaluator.evaluate_hand(hero_cards + board_cards)
        
        # Evaluate opponent hands and count results in one pass
        hero_better_count = 0
        hero_tied_count = 0
        
        for opp_cards in opponent_hands:
            opp_rank, opp_tiebreakers = self.evaluator.evaluate_hand(opp_cards + board_cards)
            
            if hero_rank > opp_rank:
                hero_better_count += 1
            elif hero_rank == opp_rank:
                if hero_tiebreakers > opp_tiebreakers:
                    hero_better_count += 1
                elif hero_tiebreakers == opp_tiebreakers:
                    hero_tied_count += 1
        
        # Determine result without intermediate variables
        if hero_better_count == num_opponents:
            result = "win"
        elif hero_better_count + hero_tied_count == num_opponents and hero_tied_count > 0:
            result = "tie"
        else:
            result = "loss"
        
        # Return result with cached hand type lookup
        return {
            "result": result,
            "hero_hand_type": self._get_hand_type_name(hero_rank),
            "hero_hand_rank": hero_rank  # For stratified sampling
        }
    
    def _get_hand_type_name(self, hand_rank: int) -> str:
        """Convert hand rank to readable name."""
        for name, rank in HAND_RANKINGS.items():
            if rank == hand_rank:
                return name
        return "unknown"
    
    def run_sequential_simulations(self, hero_cards: List[Card], num_opponents: int, 
                                  board: List[Card], removed_cards: List[Card], 
                                  num_simulations: int, max_time_ms: int, start_time: float,
                                  convergence_monitor=None, smart_sampler=None) -> Tuple[int, int, int, Counter, Optional[Dict[str, Any]]]:
        """Run simulations sequentially with optional convergence monitoring and smart sampling."""
        wins = 0
        ties = 0
        losses = 0
        hand_categories = Counter() if self.config["output_settings"]["include_hand_categories"] else None
        
        # Initialize convergence data
        convergence_data = None
        stopped_early = False
        adaptive_timeout_ms = max_time_ms
        last_convergence_check = 0
        
        # Smart sampling initialization
        sampling_state = None
        if smart_sampler:
            sampling_state = smart_sampler.initialize_sampling(hero_cards, board, num_simulations)
        
        # Enhanced timeout and convergence checking intervals
        base_timeout_interval = min(5000, max(1000, num_simulations // 20))
        base_convergence_interval = min(100, max(50, num_simulations // 100)) if convergence_monitor else num_simulations + 1
        
        # Adaptive intervals
        timeout_check_interval = base_timeout_interval
        convergence_check_interval = base_convergence_interval
        
        for sim in range(num_simulations):
            # Adaptive timeout check
            if sim > 0 and sim % timeout_check_interval == 0:
                current_time = time.time()
                elapsed_ms = (current_time - start_time) * 1000
                
                # Check standard timeout
                if elapsed_ms > adaptive_timeout_ms:
                    break
                
                # Real-time confidence interval monitoring
                if convergence_monitor and sim >= convergence_monitor.min_samples:
                    total_sims = wins + ties + losses
                    current_win_rate = wins / total_sims if total_sims > 0 else 0
                    
                    # Calculate current confidence interval
                    confidence_interval = self._calculate_confidence_interval(
                        current_win_rate, total_sims, convergence_monitor.confidence_level
                    )
                    margin_of_error = (confidence_interval[1] - confidence_interval[0]) / 2
                    
                    # Adaptive timeout based on convergence rate
                    convergence_status = convergence_monitor.get_convergence_status()
                    if convergence_status.get('status') == 'converged':
                        # If converged, reduce remaining timeout
                        remaining_time = adaptive_timeout_ms - elapsed_ms
                        adaptive_timeout_ms = elapsed_ms + min(remaining_time * 0.5, 5000)
                    elif margin_of_error > convergence_monitor.target_accuracy * 2:
                        # If accuracy is poor, extend timeout slightly
                        adaptive_timeout_ms = min(adaptive_timeout_ms * 1.1, max_time_ms * 2)
                    
                    # Adaptive timeout check intervals
                    if margin_of_error < convergence_monitor.target_accuracy * 1.5:
                        timeout_check_interval = max(base_timeout_interval // 4, 100)
                    else:
                        timeout_check_interval = min(base_timeout_interval * 2, 10000)
            
            # Generate sample with appropriate strategy
            if smart_sampler and sampling_state:
                result = smart_sampler.simulate_with_strategy(
                    self, hero_cards, num_opponents, board, removed_cards, 
                    sampling_state, sim
                )
                # Apply variance reduction
                result = smart_sampler.apply_variance_reduction(result, sampling_state, sim)
            else:
                result = self.simulate_hand(hero_cards, num_opponents, board, removed_cards)
            
            # Process result
            result_type = result["result"]
            if result_type == "win":
                wins += 1
            elif result_type == "tie":
                ties += 1
            else:
                losses += 1
            
            # Track hand categories if needed
            if hand_categories is not None:
                hand_categories[result["hero_hand_type"]] += 1
            
            # Enhanced convergence monitoring
            if convergence_monitor and sim > 0 and sim % convergence_check_interval == 0:
                total_sims = wins + ties + losses
                current_win_rate = wins / total_sims if total_sims > 0 else 0
                
                # Update convergence monitor
                convergence_monitor.update(current_win_rate, total_sims)
                
                # Calculate margin of error
                confidence_interval = self._calculate_confidence_interval(
                    current_win_rate, total_sims, convergence_monitor.confidence_level
                )
                margin_of_error = (confidence_interval[1] - confidence_interval[0]) / 2
                
                # Check stopping criteria
                accuracy_achieved = margin_of_error <= convergence_monitor.target_accuracy
                convergence_achieved = convergence_monitor.has_converged()
                min_samples_met = total_sims >= convergence_monitor.min_samples
                
                if min_samples_met and accuracy_achieved and convergence_achieved:
                    stopped_early = True
                    break
                
                # Adaptive convergence checking
                last_convergence_check = sim
                if accuracy_achieved or convergence_achieved:
                    convergence_check_interval = max(base_convergence_interval // 2, 25)
                elif sim > last_convergence_check + base_convergence_interval * 5:
                    convergence_check_interval = min(base_convergence_interval * 2, 500)
                else:
                    convergence_check_interval = base_convergence_interval
        
        # Collect convergence data
        if convergence_monitor:
            convergence_status = convergence_monitor.get_convergence_status()
            final_win_rate = wins / (wins + ties + losses) if (wins + ties + losses) > 0 else 0
            final_confidence = self._calculate_confidence_interval(
                final_win_rate, wins + ties + losses, convergence_monitor.confidence_level
            )
            final_margin_of_error = (final_confidence[1] - final_confidence[0]) / 2
            
            convergence_data = {
                'monitor_active': True,
                'stopped_early': stopped_early,
                'convergence_status': convergence_status,
                'convergence_history': convergence_monitor.convergence_history,
                'adaptive_timeout_used': adaptive_timeout_ms != max_time_ms,
                'final_timeout_ms': adaptive_timeout_ms,
                'final_margin_of_error': final_margin_of_error,
                'target_accuracy_achieved': final_margin_of_error <= convergence_monitor.target_accuracy,
                'confidence_interval_final': final_confidence,
                'smart_sampling_enabled': sampling_state is not None,
                'variance_reduction_efficiency': sampling_state.variance_reduction_efficiency if sampling_state else None
            }
        else:
            convergence_data = {'monitor_active': False}
        
        return wins, ties, losses, hand_categories or Counter(), convergence_data
    
    def run_parallel_simulations(self, hero_cards: List[Card], num_opponents: int, 
                                board: List[Card], removed_cards: List[Card], 
                                num_simulations: int, max_time_ms: int, start_time: float) -> Tuple[int, int, int, Counter, Optional[Dict[str, Any]]]:
        """Run simulations in parallel using persistent ThreadPoolExecutor."""
        wins = 0
        ties = 0
        losses = 0
        hand_categories = Counter()
        
        # Determine batch size and number of workers
        num_workers = min(self._max_workers, max(1, num_simulations // 1000))
        batch_size = num_simulations // num_workers
        remaining = num_simulations % num_workers
        
        # Create batches
        batches = [batch_size] * num_workers
        if remaining > 0:
            batches[-1] += remaining
        
        # Cache config values
        include_hand_categories = self.config["output_settings"]["include_hand_categories"]
        
        def run_batch(batch_size: int) -> Tuple[int, int, int, Dict[str, int]]:
            """Run a batch of simulations."""
            batch_wins = 0
            batch_ties = 0
            batch_losses = 0
            batch_categories = Counter() if include_hand_categories else None
            
            # Local timeout check interval
            batch_timeout_interval = min(1000, max(100, batch_size // 10))
            
            for i in range(batch_size):
                # Timeout check
                if i > 0 and i % batch_timeout_interval == 0:
                    if (time.time() - start_time) * 1000 > max_time_ms:
                        break
                
                result = self.simulate_hand(hero_cards, num_opponents, board, removed_cards)
                
                # Process result
                result_type = result["result"]
                if result_type == "win":
                    batch_wins += 1
                elif result_type == "tie":
                    batch_ties += 1
                else:
                    batch_losses += 1
                
                # Track hand categories
                if batch_categories is not None:
                    batch_categories[result["hero_hand_type"]] += 1
            
            return batch_wins, batch_ties, batch_losses, dict(batch_categories) if batch_categories else {}
        
        # Run batches in parallel
        thread_pool = self._get_thread_pool()
        futures = [thread_pool.submit(run_batch, batch_size) for batch_size in batches]
        
        for future in as_completed(futures):
            try:
                batch_wins, batch_ties, batch_losses, batch_categories = future.result()
                wins += batch_wins
                ties += batch_ties
                losses += batch_losses
                
                # Merge hand categories
                if include_hand_categories and batch_categories:
                    for category, count in batch_categories.items():
                        hand_categories[category] += count
                        
            except Exception as e:
                print(f"Warning: Batch simulation failed: {e}")
        
        return wins, ties, losses, hand_categories, None
    
    def _calculate_confidence_interval(self, win_prob: float, sample_size: int, 
                                      confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for win probability."""
        import math
        
        if sample_size == 0:
            return (0.0, 0.0)
        
        # Use normal approximation for binomial proportion
        z_score = 1.96  # 95% confidence
        if confidence_level == 0.99:
            z_score = 2.576
        elif confidence_level == 0.90:
            z_score = 1.645
        
        margin_of_error = z_score * math.sqrt((win_prob * (1 - win_prob)) / sample_size)
        
        lower_bound = max(0, win_prob - margin_of_error)
        upper_bound = min(1, win_prob + margin_of_error)
        
        precision = self.config["output_settings"]["decimal_precision"]
        return (round(lower_bound, precision), round(upper_bound, precision))