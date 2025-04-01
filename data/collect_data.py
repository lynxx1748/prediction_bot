"""
Data collection functionality for trading bot.
For fetching historical and real-time data.
"""

import logging
import os
import time
from datetime import datetime, timedelta

import requests

from scripts.core.constants import CONTRACT_ADDRESS, MARKET_API, contract
from scripts.data.database import get_db_connection, TABLES

logger = logging.getLogger(__name__)

def fetch_historical_data(days=30, save_to_db=True):
    """
    Fetch historical price and betting data.

    Args:
        days: Number of days of history to fetch
        save_to_db: Whether to save fetched data to database

    Returns:
        dict: Historical data
    """
    logger.info(f"Fetching {days} days of historical data")

    try:
        # Calculate start time
        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        current_time = int(time.time())

        # Initialize data structure
        historical_data = {
            "prices": [],
            "rounds": [],
            "start_timestamp": start_time,
            "end_timestamp": current_time,
            "contract_address": CONTRACT_ADDRESS,
        }

        # Fetch round data from contract
        current_epoch = contract.functions.currentEpoch().call()
        start_epoch = max(1, current_epoch - (days * 24 * 60 * 60 // 300))  # Convert days to epochs
        
        for epoch in range(start_epoch, current_epoch):
            round_data = contract.functions.rounds(epoch).call()
            historical_data["rounds"].append({
                "epoch": epoch,
                "lockPrice": round_data[4],
                "closePrice": round_data[5],
                "bullAmount": round_data[9],
                "bearAmount": round_data[10],
                "totalAmount": round_data[8],
                "outcome": "BULL" if round_data[5] > round_data[4] else "BEAR"
            })

        # Fetch price data from API
        if MARKET_API:
            url = f"{MARKET_API}/historical-prices?days={days}&contract={CONTRACT_ADDRESS}"
            response = requests.get(url)
            if response.status_code == 200:
                price_data = response.json()
                historical_data["prices"] = price_data
                logger.info(f"Fetched {len(price_data)} price points")
            else:
                logger.error(
                    f"Failed to fetch historical prices: {response.status_code}"
                )

        # Save to database if requested
        if save_to_db:
            save_historical_data_to_db(historical_data)

        return historical_data

    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

def save_historical_data_to_db(historical_data):
    """
    Save historical data to database.

    Args:
        historical_data: Dictionary of historical prices and rounds
    """
    try:
        # Ensure data directory exists
        os.makedirs(os.path.dirname("data/trading_bot.db"), exist_ok=True)
        
        conn = get_db_connection()
        cursor = conn.cursor()

        # Save price data
        for price_point in historical_data.get("prices", []):
            try:
                cursor.execute(
                    f"""
                    INSERT OR IGNORE INTO {TABLES['market_data']} 
                    (timestamp, price, volume, market_cap, dominance)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        price_point.get("timestamp", 0),
                        price_point.get("price", 0),
                        price_point.get("volume", 0),
                        price_point.get("market_cap", 0),
                        price_point.get("dominance", 0),
                    ),
                )
            except Exception as e:
                logger.error(f"Error saving price point: {e}")

        # Save round data
        for round_data in historical_data.get("rounds", []):
            try:
                # First check if this round already exists
                cursor.execute(
                    f"""
                    SELECT id FROM {TABLES['trades']} WHERE epoch = ?
                    """,
                    (round_data.get("epoch", 0),),
                )

                if cursor.fetchone() is None:
                    cursor.execute(
                        f"""
                        INSERT INTO {TABLES['trades']} 
                        (epoch, timestamp, lockPrice, closePrice, bullAmount, bearAmount, 
                        totalAmount, bullRatio, bearRatio, outcome)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            round_data.get("epoch", 0),
                            round_data.get("closeTimestamp", 0),
                            round_data.get("lockPrice", 0),
                            round_data.get("closePrice", 0),
                            round_data.get("bullAmount", 0),
                            round_data.get("bearAmount", 0),
                            round_data.get("totalAmount", 0),
                            round_data.get("bullRatio", 0),
                            round_data.get("bearRatio", 0),
                            round_data.get("outcome", "UNKNOWN"),
                        ),
                    )
            except Exception as e:
                logger.error(f"Error saving round data: {e}")

        conn.commit()
        conn.close()
        logger.info("Historical data saved to database")

    except Exception as e:
        logger.error(f"Error saving historical data to database: {e}")
