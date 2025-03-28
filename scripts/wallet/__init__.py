"""
Wallet functionality for the trading bot.
Handles blockchain transactions, balance management, and claims processing.
"""

from .balance import get_wallet_balance
from .claims import claim_rewards, check_claimable_rounds
from .betting import calculate_bet_amount, place_bet
from .developer import get_dev_wallet, is_dev_wallet, calculate_dev_fee

__all__ = [
    'get_wallet_balance',
    'claim_rewards',
    'check_claimable_rounds',
    'calculate_bet_amount',
    'place_bet',
    'get_dev_wallet',
    'is_dev_wallet',
    'calculate_dev_fee'
] 