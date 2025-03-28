"""
Core functionality for the trading bot.
Contains configuration, constants, and utility functions.
"""

import os
import logging
from pathlib import Path

# Define the core directory path
CORE_DIR = Path(__file__).parent

# Create directories if they don't exist
os.makedirs(CORE_DIR, exist_ok=True)

# Initialize logger
logger = logging.getLogger(__name__)

# Import functionality
from .constants import (
    BASE_DIR, 
    DB_FILE, 
    TABLES, 
    config, 
    web3, 
    contract
)
from .utils import (
    get_price_trend,
    get_recent_outcomes,
    get_historical_data,
    calculate_time_offset,
    sleep_and_check_for_interruption
)

# Export public interface
__all__ = [
    'BASE_DIR',
    'DB_FILE',
    'TABLES',
    'config',
    'web3',
    'contract',
    'get_price_trend',
    'get_recent_outcomes',
    'get_historical_data',
    'calculate_time_offset',
    'sleep_and_check_for_interruption'
] 