"""
Utility functions for the trading bot.
Contains general-purpose helper functions used across different modules.
"""

import logging
import sqlite3
import time
import traceback

from .constants import DB_FILE, TABLES, contract

logger = logging.getLogger(__name__)


def get_price_trend(lookback=5):
    """
    Get BNB price trend based on historical data.

    Args:
        lookback: Number of epochs to look back

    Returns:
        tuple: (trend, confidence) where trend is "up", "down", or "neutral"
    """
    try:
        # Use lazy import to avoid circular dependency
        from ..data.blockchain import get_current_epoch, get_round_data

        # Get current and previous epochs
        current_epoch = get_current_epoch()
        if not current_epoch:
            logger.warning("Unable to get current epoch for price trend")
            return "neutral", 0.5

        # Get prices from recent rounds
        prices = []
        epochs = range(current_epoch - lookback, current_epoch)

        for epoch in epochs:
            if epoch <= 0:
                continue

            round_data = get_round_data(epoch)
            if round_data and round_data.get("closePrice", 0) > 0:
                prices.append(round_data.get("closePrice", 0))

        # Insufficient data
        if len(prices) < 3:
            logger.debug(
                f"Insufficient data for price trend (found {len(prices)} points)"
            )
            return "neutral", 0.5

        # Calculate trend
        price_change = ((prices[-1] - prices[0]) / prices[0]) * 100

        if price_change > 1.0:
            confidence = min(0.5 + abs(price_change) / 10, 0.9)
            logger.info(
                f"📈 Upward price trend detected: {price_change:.2f}% (confidence: {confidence:.2f})"
            )
            return "up", confidence
        elif price_change < -1.0:
            confidence = min(0.5 + abs(price_change) / 10, 0.9)
            logger.info(
                f"📉 Downward price trend detected: {price_change:.2f}% (confidence: {confidence:.2f})"
            )
            return "down", confidence
        else:
            logger.info(f"➖ Neutral price trend: {price_change:.2f}%")
            return "neutral", 0.5

    except Exception as e:
        logger.error(f"Error getting price trend: {e}")
        traceback.print_exc()
        return "neutral", 0.5


def get_recent_outcomes(count=8):
    """
    Get outcomes of recent rounds from the database.

    Args:
        count: Number of recent outcomes to retrieve

    Returns:
        list: List of recent outcomes ("bull" or "bear")
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute(
            f"SELECT epoch, outcome FROM {TABLES['trades']} ORDER BY epoch DESC LIMIT {count}"
        )
        rows = cursor.fetchall()

        conn.close()

        if not rows:
            logger.warning("No recent outcomes found in database")
            return []

        outcomes = [outcome.lower() for _, outcome in rows]
        logger.info(
            f"Recent outcomes (last {len(outcomes)} rounds): {', '.join(outcome.upper() for outcome in outcomes)}"
        )
        return outcomes

    except Exception as e:
        logger.error(f"Error fetching recent outcomes: {e}")
        traceback.print_exc()
        return []


def get_historical_data(epoch):
    """
    Get historical data for a specific round from the database.

    Args:
        epoch: Epoch number to retrieve

    Returns:
        dict: Round data or None if not found
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {TABLES['trades']} WHERE epoch = ?", (epoch,))
        data = cursor.fetchone()

        if not data:
            return None

        column_names = [description[0] for description in cursor.description]
        result = dict(zip(column_names, data))

        conn.close()
        return result
    except Exception as e:
        logger.error(f"Error loading historical data for epoch {epoch}: {e}")
        traceback.print_exc()
        return None


def calculate_time_offset():
    """
    Calculate the time offset between local time and blockchain time.

    Returns:
        float: Time offset in seconds (positive if blockchain is ahead)
    """
    try:
        current_epoch = contract.functions.currentEpoch().call()
        current_round = contract.functions.rounds(current_epoch).call()

        startTimestamp = current_round[2]
        lockTimestamp = current_round[3]
        closeTimestamp = current_round[4]

        local_time = int(time.time())

        round_duration = closeTimestamp - startTimestamp
        if round_duration == 0:
            logger.warning("Invalid round duration (0)")
            return 0

        expected_progress = (local_time - startTimestamp) / round_duration

        if 0 <= expected_progress <= 1:
            if expected_progress < 0:
                offset = startTimestamp - local_time
            elif expected_progress > 1:
                offset = closeTimestamp - local_time
            else:
                current_position = local_time - startTimestamp
                expected_position = round_duration * expected_progress
                offset = expected_position - current_position

            logger.debug(f"Time offset: {offset:.2f} seconds")
            return offset
        else:
            logger.warning(f"Expected progress out of range: {expected_progress}")
            return 0

    except Exception as e:
        logger.error(f"Error calculating time offset: {e}")
        traceback.print_exc()
        return 0


def sleep_and_check_for_interruption(seconds):
    """
    Sleep for the specified number of seconds while allowing for keyboard interruption.

    Args:
        seconds (int/float): Number of seconds to sleep

    Returns:
        bool: True if completed without interruption, False if interrupted
    """
    try:
        interval = (
            0.1  # Check every 0.1 seconds for more responsive keyboard interrupts
        )
        iterations = int(seconds / interval)

        for _ in range(iterations):
            time.sleep(interval)

        # Sleep any remainder
        remainder = seconds % interval
        if remainder > 0:
            time.sleep(remainder)

        return True
    except KeyboardInterrupt:
        logger.warning("Sleep interrupted by user")
        return False
