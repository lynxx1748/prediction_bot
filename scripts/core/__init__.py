"""
Core functionality for the trading bot.
Contains configuration, constants, and utility functions.
"""

import logging
import os
from pathlib import Path

# Define the core directory path
CORE_DIR = Path(__file__).parent

# Create directories if they don't exist
os.makedirs(CORE_DIR, exist_ok=True)

# Initialize logger
logger = logging.getLogger(__name__)

# Import functionality
from .constants import BASE_DIR, DB_FILE, TABLES, config, contract, web3
from .utils import (calculate_time_offset, get_historical_data,
                    get_price_trend, get_recent_outcomes,
                    sleep_and_check_for_interruption)

# Export public interface
__all__ = [
    "BASE_DIR",
    "DB_FILE",
    "TABLES",
    "config",
    "web3",
    "contract",
    "get_price_trend",
    "get_recent_outcomes",
    "get_historical_data",
    "calculate_time_offset",
    "sleep_and_check_for_interruption",
]
