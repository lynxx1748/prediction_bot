"""
Scripts package for the trading bot.
Contains core functionality for data analysis, prediction, and trading.
"""

import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Import public interface from core modules
from .core.constants import (DB_FILE, MODEL_FILE, SCALER_FILE,
                             STRATEGY_WEIGHTS, TABLES, THRESHOLDS, contract,
                             web3)

# Define public interface
__all__ = [
    "DB_FILE",
    "TABLES",
    "MODEL_FILE",
    "SCALER_FILE",
    "web3",
    "contract",
    "THRESHOLDS",
    "STRATEGY_WEIGHTS",
]
