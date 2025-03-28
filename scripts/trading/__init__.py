"""
Trading functionality for the trading bot.
Contains betting logic, money management, and decision making.
"""

from .betting import (
    should_place_bet,
    place_bet,
    check_for_win,
    calculate_bet_amount,
    get_betting_history,
    get_win_rate,
    get_progressive_win_rate,
    initialize_performance_metrics,
    update_performance
)

from .money import (
    calculate_optimal_bet_size,
    calculate_risk_adjusted_bet_size,
    get_adaptive_bet_size
)

__all__ = [
    'should_place_bet',
    'place_bet',
    'check_for_win',
    'calculate_optimal_bet_size',
    'calculate_risk_adjusted_bet_size',
    'get_adaptive_bet_size',
    'calculate_bet_amount',
    'get_betting_history',
    'get_win_rate',
    'get_progressive_win_rate',
    'initialize_performance_metrics',
    'update_performance'
] 