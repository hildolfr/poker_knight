"""
Worker functions for parallel processing.

These functions are separated from the main module to avoid circular imports
when multiprocessing pickles them for execution in separate processes.
"""

from typing import Dict, List, Any
from collections import Counter


def _parallel_simulation_worker(batch_size: int, worker_id: str, solver_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Worker function for parallel simulation batches.
    This function is at module level to be serializable for multiprocessing.
    """
    # Import here to avoid circular imports at module level
    from ..solver import Card, HandEvaluator, Deck
    
    # Extract solver data
    hero_cards_data = solver_data['hero_cards']
    num_opponents = solver_data['num_opponents']
    board_data = solver_data['board']
    removed_cards_data = solver_data['removed_cards']
    include_hand_categories = solver_data['include_hand_categories']
    
    # Recreate Card objects from serialized data
    hero_cards = [Card(rank=card['rank'], suit=card['suit']) for card in hero_cards_data]
    board = [Card(rank=card['rank'], suit=card['suit']) for card in board_data]
    removed_cards = [Card(rank=card['rank'], suit=card['suit']) for card in removed_cards_data]
    
    # Create evaluator for this worker
    evaluator = HandEvaluator()
    
    batch_wins = 0
    batch_ties = 0
    batch_losses = 0
    batch_categories = Counter() if include_hand_categories else {}
    
    # Run simulations
    for _ in range(batch_size):
        result = _simulate_single_hand(
            hero_cards, num_opponents, board, removed_cards, evaluator
        )
        
        if result["result"] == "win":
            batch_wins += 1
        elif result["result"] == "tie":
            batch_ties += 1
        else:
            batch_losses += 1
        
        if isinstance(batch_categories, Counter):
            batch_categories[result["hero_hand_type"]] += 1
    
    return {
        'wins': batch_wins,
        'ties': batch_ties,
        'losses': batch_losses,
        'hand_categories': dict(batch_categories),
        'simulations': batch_size
    }


def _simulate_single_hand(hero_cards: List[Any], num_opponents: int, 
                         board: List[Any], removed_cards: List[Any],
                         evaluator: Any) -> Dict[str, Any]:
    """Simulate a single hand - module level for multiprocessing."""
    # Import here to avoid circular imports
    from ..solver import Deck
    
    # Create deck with removed cards
    deck = Deck(removed_cards)
    
    # Deal opponent hands
    opponent_hands = [deck.deal(2) for _ in range(num_opponents)]
    
    # Complete the board if needed
    cards_needed = 5 - len(board)
    if cards_needed > 0:
        board_cards = board + deck.deal(cards_needed)
    else:
        board_cards = board
    
    # Evaluate hero hand
    hero_rank, hero_tiebreakers = evaluator.evaluate_hand(hero_cards + board_cards)
    
    # Evaluate opponent hands and count results
    hero_better_count = 0
    hero_tied_count = 0
    
    for opp_hand in opponent_hands:
        opp_rank, opp_tiebreakers = evaluator.evaluate_hand(opp_hand + board_cards)
        
        if hero_rank > opp_rank:
            hero_better_count += 1
        elif hero_rank == opp_rank:
            # Compare tiebreakers
            if hero_tiebreakers > opp_tiebreakers:
                hero_better_count += 1
            elif hero_tiebreakers == opp_tiebreakers:
                hero_tied_count += 1
    
    # Determine result
    if hero_better_count == num_opponents:
        result = "win"
    elif hero_better_count + hero_tied_count == num_opponents and hero_tied_count > 0:
        result = "tie"
    else:
        result = "loss"
    
    return {
        "result": result,
        "hero_hand_type": _get_hand_type_name_static(hero_rank),
        "hero_hand_rank": hero_rank
    }


def _get_hand_type_name_static(hand_rank: int) -> str:
    """Convert hand rank to readable name - static version for multiprocessing."""
    hand_rankings = {
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
    return hand_rankings.get(hand_rank, "unknown")