"""
Data management module for the trading bot.
Provides database operations and data collection functionality.
"""

import os
import logging
from pathlib import Path

# Define the data directory path
DATA_DIR = Path(__file__).parent

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Import key functionality
from .create_db import initialize_databases
from .collect_data import fetch_historical_data, collect_real_time_data
from .blockchain import (
    get_round_data,
    get_current_epoch,
    get_betting_epoch,
    get_time_until_lock,
    get_time_until_round_end,
    get_enriched_round_data,
    get_round_with_market_prices,
    fetch_rounds_range,
    get_round_info
)

# Initialize logger
logger = logging.getLogger(__name__)

def get_db_path(db_name):
    """
    Get the full path to a database file in the data directory.
    
    Args:
        db_name: Name of the database file
        
    Returns:
        Path object to the database file
    """
    return DATA_DIR / db_name

def ensure_data_ready():
    """
    Make sure the data directory and databases are ready for use.
    
    Returns:
        bool: True if successful, False if there was an error
    """
    try:
        # Make sure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Initialize databases
        initialize_databases()
        
        logger.info("Data directory and databases initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error ensuring data is ready: {e}")
        return False

# Export public interface
__all__ = [
    'DATA_DIR', 
    'get_db_path', 
    'ensure_data_ready', 
    'initialize_databases',
    'fetch_historical_data',
    'collect_real_time_data',
    'get_round_data',
    'get_current_epoch',
    'get_betting_epoch',
    'get_time_until_lock',
    'get_time_until_round_end',
    'get_enriched_round_data',
    'get_round_with_market_prices',
    'fetch_rounds_range',
    'get_round_info'
] 