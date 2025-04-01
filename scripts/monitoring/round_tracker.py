"""
Real-time round monitoring for the trading bot.
Tracks price movements during active rounds and detects potential trend reversals.
"""

import logging
import sqlite3
import threading
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import requests

from ..core.constants import DB_FILE, TABLES

logger = logging.getLogger(__name__)

# Global variables to track mid-round data
current_round_data = {
    "epoch": 0,
    "start_time": None,
    "prices": [],
    "volumes": [],
    "timestamps": [],
    "micro_trends": [],
    "initial_prediction": None,
    "swing_detected": False,
    "swing_direction": None,
    "swing_time": None,
}


def initialize_mid_round_tracking(epoch, initial_prediction):
    """
    Initialize tracking for a new round.

    Args:
        epoch: Current round epoch number
        initial_prediction: Initial prediction for the round ("BULL" or "BEAR")

    Returns:
        bool: Success status
    """
    global current_round_data

    try:
        current_round_data = {
            "epoch": epoch,
            "start_time": datetime.now(),
            "prices": [],
            "volumes": [],
            "timestamps": [],
            "micro_trends": [],
            "initial_prediction": initial_prediction,
            "swing_detected": False,
            "swing_direction": None,
            "swing_time": None,
        }

        logger.info(f"🔄 Mid-round tracking initialized for epoch {epoch}")

        # Start the background monitoring thread
        if (
            not hasattr(initialize_mid_round_tracking, "monitor_thread")
            or not initialize_mid_round_tracking.monitor_thread.is_alive()
        ):
            initialize_mid_round_tracking.monitor_thread = threading.Thread(
                target=monitor_mid_round_prices, daemon=True
            )
            initialize_mid_round_tracking.monitor_thread.start()
            logger.info("🔍 Mid-round price monitoring started")

        return True

    except Exception as e:
        logger.error(f"❌ Error initializing mid-round tracking: {e}")
        traceback.print_exc()
        return False


def monitor_mid_round_prices():
    """
    Background thread to monitor prices during a round.
    Continuously polls for price data and detects potential swings.
    """
    global current_round_data

    try:
        while True:
            # Only monitor if we have an active round
            if current_round_data["start_time"] is not None:
                # Check if the round should still be active (within 6 minutes)
                elapsed = (
                    datetime.now() - current_round_data["start_time"]
                ).total_seconds()

                if elapsed < 360:  # 6 minutes in seconds
                    # Get current price
                    current_price = get_current_price()

                    if current_price:
                        # Record the price point
                        current_round_data["prices"].append(current_price)
                        current_round_data["timestamps"].append(datetime.now())

                        # Detect micro-trends and potential swings
                        detect_mid_round_swing()

            # Sleep for a short period before checking again
            # Short enough to catch swings, but not too short to overwhelm the system
            time.sleep(15)  # Check every 15 seconds

    except Exception as e:
        logger.error(f"❌ Error in mid-round price monitoring: {e}")
        traceback.print_exc()


def get_current_price():
    """
    Get the current market price.

    Returns:
        float: Current price or None on failure
    """
    try:
        # Adjust this to your preferred price source
        # Example using Binance API for BNB price
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT"
        )
        if response.status_code == 200:
            data = response.json()
            return float(data["price"])
        return None
    except Exception as e:
        logger.error(f"❌ Error fetching current price: {e}")
        return None


def detect_mid_round_swing():
    """
    Detect if a price swing has occurred mid-round.
    Analyzes recent price movements to identify trend reversals.

    Returns:
        bool: True if a swing was detected, False otherwise
    """
    global current_round_data

    try:
        prices = current_round_data["prices"]
        timestamps = current_round_data["timestamps"]

        # Need at least a few data points to detect a swing
        if len(prices) < 4:
            return False

        # Calculate price changes
        recent_changes = [prices[i] / prices[i - 1] - 1 for i in range(1, len(prices))]

        # Check for significant price movement in short time
        # Focus on the most recent changes (last 2-3 data points)
        recent_movement = (
            recent_changes[-3:] if len(recent_changes) >= 3 else recent_changes
        )

        # Calculate cumulative movement
        cumulative_change = sum(recent_movement)

        # Detect direction change compared to initial prediction
        swing_detected = False

        if (
            current_round_data["initial_prediction"] == "BULL"
            and cumulative_change < -0.01
        ):
            # We predicted BULL but now seeing bearish movement
            if not current_round_data["swing_detected"]:
                current_round_data["swing_detected"] = True
                current_round_data["swing_direction"] = "BEAR"
                current_round_data["swing_time"] = timestamps[-1]

                logger.warning(
                    f"⚠️ MID-ROUND SWING DETECTED: {current_round_data['initial_prediction']} → BEAR"
                )
                logger.warning(
                    f"   Time: {timestamps[-1].strftime('%H:%M:%S')}, Movement: {cumulative_change:.4f}%"
                )

                # Record the swing in database
                record_mid_round_swing(
                    current_round_data["epoch"],
                    current_round_data["initial_prediction"],
                    "BEAR",
                    cumulative_change,
                )
                swing_detected = True

        elif (
            current_round_data["initial_prediction"] == "BEAR"
            and cumulative_change > 0.01
        ):
            # We predicted BEAR but now seeing bullish movement
            if not current_round_data["swing_detected"]:
                current_round_data["swing_detected"] = True
                current_round_data["swing_direction"] = "BULL"
                current_round_data["swing_time"] = timestamps[-1]

                logger.warning(
                    f"⚠️ MID-ROUND SWING DETECTED: {current_round_data['initial_prediction']} → BULL"
                )
                logger.warning(
                    f"   Time: {timestamps[-1].strftime('%H:%M:%S')}, Movement: {cumulative_change:.4f}%"
                )

                # Record the swing in database
                record_mid_round_swing(
                    current_round_data["epoch"],
                    current_round_data["initial_prediction"],
                    "BULL",
                    cumulative_change,
                )
                swing_detected = True

        return swing_detected

    except Exception as e:
        logger.error(f"❌ Error detecting mid-round swing: {e}")
        traceback.print_exc()
        return False


def record_mid_round_swing(epoch, initial_prediction, swing_direction, magnitude):
    """
    Record a detected mid-round swing to database.

    Args:
        epoch: Round epoch number
        initial_prediction: Initial prediction for the round
        swing_direction: Direction of the detected swing
        magnitude: Magnitude of the price movement

    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Use TABLES constant to access table name
        swings_table = TABLES.get("swings", "mid_round_swings")

        # Create table if it doesn't exist
        cursor.execute(
            f"""
        CREATE TABLE IF NOT EXISTS {swings_table} (
            epoch INTEGER PRIMARY KEY,
            timestamp INTEGER,
            initial_prediction TEXT,
            swing_direction TEXT,
            magnitude REAL,
            elapsed_seconds INTEGER
        )
        """
        )

        # Calculate elapsed time since round start
        elapsed = (datetime.now() - current_round_data["start_time"]).total_seconds()

        # Insert the swing record
        cursor.execute(
            f"""
        INSERT OR REPLACE INTO {swings_table} 
        (epoch, timestamp, initial_prediction, swing_direction, magnitude, elapsed_seconds)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                epoch,
                int(time.time()),
                initial_prediction,
                swing_direction,
                magnitude,
                int(elapsed),
            ),
        )

        conn.commit()
        conn.close()
        return True

    except Exception as e:
        logger.error(f"❌ Error recording mid-round swing: {e}")
        traceback.print_exc()
        return False


def get_mid_round_swing_statistics(lookback=20):
    """
    Get statistics about mid-round swings.

    Args:
        lookback: Number of recent rounds to analyze

    Returns:
        dict: Swing statistics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='mid_round_swings'"
        )
        if not cursor.fetchone():
            conn.close()
            return {
                "swings_detected": 0,
                "bull_to_bear": 0,
                "bear_to_bull": 0,
                "avg_timing": 0,
            }

        # Get overall count
        cursor.execute(
            "SELECT COUNT(*) FROM mid_round_swings ORDER BY epoch DESC LIMIT ?",
            (lookback,),
        )
        total_swings = cursor.fetchone()[0]

        # Get direction counts
        cursor.execute(
            """
        SELECT 
            initial_prediction, 
            swing_direction, 
            COUNT(*) 
        FROM mid_round_swings 
        GROUP BY initial_prediction, swing_direction
        ORDER BY epoch DESC LIMIT ?
        """,
            (lookback,),
        )

        direction_counts = cursor.fetchall()

        # Get average timing
        cursor.execute(
            """
        SELECT AVG(elapsed_seconds) 
        FROM mid_round_swings 
        ORDER BY epoch DESC LIMIT ?
        """,
            (lookback,),
        )

        avg_timing = cursor.fetchone()[0] or 0

        conn.close()

        # Process the statistics
        bull_to_bear = 0
        bear_to_bull = 0

        for initial, swing, count in direction_counts:
            if initial == "BULL" and swing == "BEAR":
                bull_to_bear = count
            elif initial == "BEAR" and swing == "BULL":
                bear_to_bull = count

        return {
            "swings_detected": total_swings,
            "bull_to_bear": bull_to_bear,
            "bear_to_bull": bear_to_bull,
            "avg_timing": avg_timing,
        }

    except Exception as e:
        logger.error(f"❌ Error getting mid-round swing statistics: {e}")
        traceback.print_exc()
        return {
            "swings_detected": 0,
            "bull_to_bear": 0,
            "bear_to_bull": 0,
            "avg_timing": 0,
        }


def get_potential_mid_round_swing():
    """
    Check for potential mid-round swing based on real-time monitoring.

    Returns:
        dict: Information about potential swing or None if no swing detected
    """
    global current_round_data

    if current_round_data["swing_detected"]:
        elapsed = (datetime.now() - current_round_data["start_time"]).total_seconds()

        # Return information about the detected swing
        return {
            "detected": True,
            "initial_prediction": current_round_data["initial_prediction"],
            "new_direction": current_round_data["swing_direction"],
            "elapsed_seconds": elapsed,
            "confidence": calculate_swing_confidence(),
        }

    return None


def calculate_swing_confidence():
    """
    Calculate confidence level for the detected swing.

    Returns:
        float: Confidence score (0-1)
    """
    global current_round_data

    try:
        # Basic confidence starts at 0.7
        confidence = 0.7

        # If we have more data points, confidence increases
        prices_count = len(current_round_data["prices"])
        if prices_count > 10:
            confidence += 0.1

        # Calculate recent price changes to gauge strength of the move
        recent_prices = current_round_data["prices"][-5:]
        if len(recent_prices) >= 2:
            # Calculate average percentage change
            changes = [
                abs(recent_prices[i] / recent_prices[i - 1] - 1)
                for i in range(1, len(recent_prices))
            ]
            avg_change = sum(changes) / len(changes)

            # Stronger moves increase confidence
            if avg_change > 0.01:  # More than 1% average change
                confidence += 0.1

        return min(confidence, 0.95)  # Cap at 0.95

    except Exception as e:
        logger.error(f"❌ Error calculating swing confidence: {e}")
        return 0.7  # Default confidence


def analyze_round_progress(current_price, lock_price, elapsed_seconds):
    """
    Analyze the progress of the current round.

    Args:
        current_price: Current market price
        lock_price: Price at round lock time
        elapsed_seconds: Seconds elapsed since round start

    Returns:
        dict: Round analysis data
    """
    try:
        # Calculate price change since lock
        pct_change = (current_price / lock_price - 1) * 100

        # Determine trend strength based on time elapsed and movement
        # As we get closer to the end, the trend becomes more reliable
        time_factor = min(elapsed_seconds / 300, 1.0)  # 5 minutes round duration

        # Calculate momentum
        momentum = pct_change * time_factor

        # Predict final outcome
        if momentum > 0.5:
            prediction = "BULL"
            confidence = min(0.5 + abs(momentum) / 10, 0.95)
        elif momentum < -0.5:
            prediction = "BEAR"
            confidence = min(0.5 + abs(momentum) / 10, 0.95)
        else:
            # If momentum is weak, outcome is uncertain
            prediction = "BULL" if pct_change > 0 else "BEAR"
            confidence = 0.5 + (abs(pct_change) / 10)

        return {
            "elapsed_seconds": elapsed_seconds,
            "price_change_pct": pct_change,
            "momentum": momentum,
            "prediction": prediction,
            "confidence": confidence,
            "time_elapsed_pct": (elapsed_seconds / 360)
            * 100,  # Percentage of round complete
        }

    except Exception as e:
        logger.error(f"❌ Error analyzing round progress: {e}")
        return {"prediction": "UNKNOWN", "confidence": 0.5}


def get_prediction_window_times():
    """
    Calculate optimal prediction window times using timedelta.

    Returns:
        dict: Dictionary of prediction window timestamps
    """
    now = datetime.now()

    return {
        "optimal_start": now - timedelta(minutes=2, seconds=30),
        "optimal_end": now - timedelta(seconds=30),
        "lock_cutoff": now + timedelta(seconds=10),
        "result_time": now + timedelta(minutes=5, seconds=30),
        "next_round_start": now + timedelta(minutes=6),
    }


def analyze_price_trend(prices):
    """
    Analyze price trend using numpy calculations.

    Args:
        prices: List of price data points

    Returns:
        dict: Trend analysis results
    """
    try:
        if not prices or len(prices) < 3:
            return {"trend": "unknown", "strength": 0}

        # Convert to numpy array for calculations
        price_array = np.array(prices)

        # Calculate changes
        changes = np.diff(price_array) / price_array[:-1]

        # Calculate basic statistics
        mean_change = np.mean(changes)
        std_dev = np.std(changes)
        momentum = np.sum(changes[-3:]) if len(changes) >= 3 else mean_change

        # Determine trend direction and strength
        if mean_change > 0.0001:
            trend = "rising"
            strength = min(0.5 + (abs(mean_change) / 0.001), 0.9)
        elif mean_change < -0.0001:
            trend = "falling"
            strength = min(0.5 + (abs(mean_change) / 0.001), 0.9)
        else:
            trend = "neutral"
            strength = 0.5

        return {
            "trend": trend,
            "strength": strength,
            "momentum": float(momentum),
            "volatility": float(std_dev),
            "raw_change": float(mean_change),
        }

    except Exception as e:
        logger.error(f"Error analyzing price trend: {e}")
        return {"trend": "error", "strength": 0}
