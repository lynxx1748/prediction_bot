"""
Blockchain interaction functionality for the trading bot.
Contains functions for interacting with and monitoring the blockchain.
"""

from .events import (
    setup_event_listeners,
    track_betting_events,
    start_event_monitor,
    check_new_events
)

__all__ = [
    'setup_event_listeners',
    'track_betting_events',
    'start_event_monitor',
    'check_new_events'
] 