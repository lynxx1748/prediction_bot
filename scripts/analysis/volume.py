"""
Volume analysis functions for the trading bot.
Provides volume profiling, OBV, VWAP, and volume trend analysis.
"""

import logging
import sqlite3
import time
import traceback
from functools import lru_cache

import numpy as np

from ..core.constants import DB_FILE, TABLES
from ..data.database import get_recent_trades, get_recent_volume_data

logger = logging.getLogger(__name__)


def analyze_volume_profile(round_data, historical_rounds):
    """
    Analyze volume profile for smart money movements.

    Args:
        round_data: Current round data with volume information
        historical_rounds: List of previous rounds with volume data

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or None
    """
    try:
        current_volume = round_data["totalAmount"]
        bull_ratio = round_data["bullRatio"]
        bear_ratio = round_data["bearRatio"]

        # Calculate volume metrics
        avg_volume = sum(r["totalAmount"] for r in historical_rounds) / len(
            historical_rounds
        )
        volume_increase = current_volume > (avg_volume * 1.2)  # 20% above average

        # Calculate on-balance volume (OBV)
        obv_signal, obv_strength = calculate_obv(historical_rounds)

        # Volume Weighted Average Price (VWAP) analysis
        vwap_signal, vwap_strength = analyze_vwap(round_data, historical_rounds)

        # Detect volume divergence patterns
        divergence_signal, divergence_strength = detect_volume_divergence(
            historical_rounds
        )

        # Combine all signals
        signals = []

        # Smart money patterns based on volume imbalance
        if volume_increase:
            # Large volume with clear direction
            if bull_ratio > 0.7:  # Strong bulls
                signals.append(("BULL", min(0.5 + bull_ratio, 0.95)))
                logger.info(f"📈 Volume surge with strong bull bias: {bull_ratio:.2f}")
            elif bear_ratio > 0.7:  # Strong bears
                signals.append(("BEAR", min(0.5 + bear_ratio, 0.95)))
                logger.info(f"📉 Volume surge with strong bear bias: {bear_ratio:.2f}")

        # Add OBV signal if valid
        if obv_signal:
            signals.append((obv_signal, obv_strength))
            logger.info(f"📊 OBV signal: {obv_signal} ({obv_strength:.2f})")

        # Add VWAP signal if valid
        if vwap_signal:
            signals.append((vwap_signal, vwap_strength))
            logger.info(f"📊 VWAP signal: {vwap_signal} ({vwap_strength:.2f})")

        # Add divergence signal if valid
        if divergence_signal:
            signals.append((divergence_signal, divergence_strength))
            logger.info(
                f"📊 Volume divergence: {divergence_signal} ({divergence_strength:.2f})"
            )

        # Check for accumulation/distribution
        recent_volumes = [r["totalAmount"] for r in historical_rounds[-5:]]
        volume_trend = all(v <= current_volume for v in recent_volumes)

        if volume_trend:
            if bull_ratio > bear_ratio:
                signals.append(("BULL", 0.7))
                logger.info(
                    "📈 Accumulation pattern detected (rising volume with bull bias)"
                )
            else:
                signals.append(("BEAR", 0.7))
                logger.info(
                    "📉 Distribution pattern detected (rising volume with bear bias)"
                )

        # Return most confident signal or none
        if signals:
            # Sort by strength and return strongest signal
            strongest = sorted(signals, key=lambda x: x[1], reverse=True)[0]
            logger.info(
                f"📊 Strongest volume signal: {strongest[0]} ({strongest[1]:.2f})"
            )
            return strongest

        return None, 0

    except Exception as e:
        logger.error(f"❌ Error in volume analysis: {e}")
        traceback.print_exc()
        return None, 0


def calculate_obv(historical_data):
    """
    Calculate On-Balance Volume (OBV) to detect accumulation/distribution.
    OBV adds volume on up days and subtracts on down days.

    Args:
        historical_data: List of historical rounds with volume data

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or None
    """
    try:
        if len(historical_data) < 5:
            return None, 0

        # Extract prices and volumes
        prices = []
        volumes = []

        for round_data in historical_data:
            if "closePrice" in round_data and round_data["closePrice"] > 0:
                prices.append(round_data["closePrice"])
                volumes.append(round_data["totalAmount"])

        if len(prices) < 5:
            return None, 0

        # Calculate OBV
        obv = [0]
        for i in range(1, len(prices)):
            if prices[i] > prices[i - 1]:
                obv.append(obv[-1] + volumes[i])
            elif prices[i] < prices[i - 1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])

        # Check for trend in last 5 values
        if len(obv) >= 5:
            recent_obv = obv[-5:]
            obv_trend = sum(
                1
                for i in range(1, len(recent_obv))
                if recent_obv[i] > recent_obv[i - 1]
            )

            if obv_trend >= 4:  # Strong uptrend
                return "BULL", 0.8
            elif obv_trend <= 1:  # Strong downtrend
                return "BEAR", 0.8

        return None, 0

    except Exception as e:
        logger.error(f"❌ Error calculating OBV: {e}")
        return None, 0


def analyze_vwap(current_round, historical_rounds):
    """
    Analyze Volume Weighted Average Price (VWAP).
    VWAP = Cumulative(Price * Volume) / Cumulative(Volume)

    Args:
        current_round: Current round data
        historical_rounds: List of historical rounds with volume data

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or None
    """
    try:
        if len(historical_rounds) < 8:
            return None, 0

        # Calculate VWAP
        cumulative_pv = 0
        cumulative_volume = 0

        for round_data in historical_rounds[-8:]:
            if "closePrice" in round_data and round_data["totalAmount"] > 0:
                price = round_data["closePrice"]
                volume = round_data["totalAmount"]

                cumulative_pv += price * volume
                cumulative_volume += volume

        if cumulative_volume > 0:
            vwap = cumulative_pv / cumulative_volume

            # Compare current price to VWAP
            current_price = current_round.get("lockPrice", 0)

            if current_price > 0:
                # Calculate distance from VWAP
                distance = (current_price - vwap) / vwap

                # Generate signal based on price vs VWAP
                if distance > 0.02:  # Price is 2% above VWAP
                    return "BEAR", min(0.6 + abs(distance) * 10, 0.9)  # Mean reversion
                elif distance < -0.02:  # Price is 2% below VWAP
                    return "BULL", min(0.6 + abs(distance) * 10, 0.9)  # Mean reversion

        return None, 0

    except Exception as e:
        logger.error(f"❌ Error analyzing VWAP: {e}")
        return None, 0


def detect_volume_divergence(historical_rounds):
    """
    Detect volume divergence patterns.
    Volume divergence occurs when price moves in one direction but volume trend moves in opposite.

    Args:
        historical_rounds: List of historical rounds with volume data

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or None
    """
    try:
        if len(historical_rounds) < 6:
            return None, 0

        # Get recent rounds
        recent_rounds = historical_rounds[-6:]

        # Extract prices and volumes
        prices = []
        volumes = []

        for round_data in recent_rounds:
            if "closePrice" in round_data and round_data["closePrice"] > 0:
                prices.append(round_data["closePrice"])
                volumes.append(round_data["totalAmount"])

        if len(prices) < 5:
            return None, 0

        # Check price direction
        price_direction = 1 if prices[-1] > prices[0] else -1

        # Check volume direction
        volume_direction = 1 if volumes[-1] > volumes[0] else -1

        # Classic divergence
        if price_direction != volume_direction:
            # Price up, volume down = bearish
            if price_direction > 0 and volume_direction < 0:
                return "BEAR", 0.7
            # Price down, volume up = bullish
            elif price_direction < 0 and volume_direction > 0:
                return "BULL", 0.7

        return None, 0

    except Exception as e:
        logger.error(f"❌ Error detecting volume divergence: {e}")
        return None, 0


def get_volume_prediction(lookback=20):
    """
    Get prediction based on volume analysis from the database.

    Args:
        lookback: Number of rounds to analyze

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or "UNKNOWN"
    """
    try:
        # Ensure lookback is an integer
        if isinstance(lookback, dict):
            logger.warning("Lookback parameter is a dictionary, using default value")
            lookback = 20
        elif not isinstance(lookback, (int, float)):
            try:
                lookback = int(lookback)
            except (ValueError, TypeError):
                logger.warning(
                    "Couldn't convert lookback parameter to integer, using default value"
                )
                lookback = 20

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # First check if totalAmount column exists in the trades table
        table_name = TABLES["trades"]
        cursor.execute("PRAGMA table_info('%s')" % table_name)
        columns = [row[1] for row in cursor.fetchall()]

        # Add totalAmount column if it doesn't exist
        if "totalAmount" not in columns:
            try:
                cursor.execute(
                    "ALTER TABLE '%s' ADD COLUMN totalAmount REAL DEFAULT 0"
                    % table_name
                )
                cursor.execute(
                    "ALTER TABLE '%s' ADD COLUMN bullAmount REAL DEFAULT 0" % table_name
                )
                cursor.execute(
                    "ALTER TABLE '%s' ADD COLUMN bearAmount REAL DEFAULT 0" % table_name
                )
                logger.info("Added missing amount columns to trades table")
                conn.commit()
            except Exception as e:
                logger.warning(f"Error adding amount columns: {e}")

        # Get a list of columns that actually exist in the table
        columns_to_select = []
        essential_columns = ["epoch", "outcome"]
        optional_columns = ["lockPrice", "closePrice", "timestamp"]

        # Always include essential columns
        for col in essential_columns:
            if col in columns:
                columns_to_select.append(col)
            else:
                logger.warning(f"Essential column '{col}' missing from trades table")
                return "UNKNOWN", 0  # Can't proceed without essential columns

        # Add optional columns if they exist
        for col in optional_columns:
            if col in columns:
                columns_to_select.append(col)

        # Use a hardcoded LIMIT with the validated lookback value
        query = f"""
        SELECT {', '.join(columns_to_select)}
        FROM trades
        ORDER BY epoch DESC
        LIMIT {int(lookback)}
        """

        cursor.execute(query)  # No parameters, value hardcoded in query

        results = cursor.fetchall()
        conn.close()

        if not results:
            return "UNKNOWN", 0

        # Simple volume-less analysis - only using outcome column (index may vary)
        outcome_index = (
            columns_to_select.index("outcome") if "outcome" in columns_to_select else 1
        )
        bull_count = sum(1 for r in results if r[outcome_index] == "BULL")
        bear_count = sum(1 for r in results if r[outcome_index] == "BEAR")

        total = bull_count + bear_count
        if total == 0:
            return "UNKNOWN", 0

        # Calculate prediction based on trend
        bull_ratio = bull_count / total
        if bull_ratio > 0.51:
            return "BULL", 0.6
        elif bull_ratio < 0.48:
            return "BEAR", 0.6
        else:
            # When close to 50/50, default to BULL
            recent_results = results[:5]
            recent_bull = sum(1 for r in recent_results if r[outcome_index] == "BULL")
            if recent_bull >= 2:
                return "BULL", 0.55
            else:
                return "BEAR", 0.55

    except Exception as e:
        logger.error(f"❌ Error in volume prediction: {e}")
        traceback.print_exc()
        return "UNKNOWN", 0


def get_volume_trend_prediction(round_data):
    """
    Get prediction based on volume trends and price action correlation.

    Args:
        round_data: Current round data

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or None
    """
    try:
        # Get recent volume and price data
        volume_data = get_recent_volume_data(lookback=15)
        if not volume_data or len(volume_data) < 10:
            return None, 0

        # Extract volumes and prices
        volumes = [v["volume"] for v in volume_data]
        prices = [v["price"] for v in volume_data]

        # Calculate moving averages
        volume_ma = np.mean(volumes[-5:]) / np.mean(volumes[-10:-5])
        price_ma = np.mean(prices[-5:]) / np.mean(prices[-10:-5])

        # Volume trend
        volume_increasing = volume_ma > 1.05
        volume_decreasing = volume_ma < 0.95

        # Price trend
        price_increasing = price_ma > 1.01
        price_decreasing = price_ma < 0.99

        # Decision logic - look for divergences and confirmations
        prediction = None
        confidence = 0

        if volume_increasing and price_increasing:
            # Bullish confirmation - strong uptrend
            prediction = "BULL"
            confidence = 0.75
            logger.info("📈 Volume and price both increasing - bullish confirmation")
        elif volume_increasing and price_decreasing:
            # Volume divergence - potential reversal
            prediction = "BULL"
            confidence = 0.65
            logger.info(
                "🔄 Volume increasing but price decreasing - potential bullish reversal"
            )
        elif volume_decreasing and price_decreasing:
            # Bearish confirmation - strong downtrend
            prediction = "BEAR"
            confidence = 0.75
            logger.info("📉 Volume and price both decreasing - bearish confirmation")
        elif volume_decreasing and price_increasing:
            # Volume weakness - potential reversal
            prediction = "BEAR"
            confidence = 0.65
            logger.info(
                "🔄 Volume decreasing but price increasing - potential bearish reversal"
            )

        # Check final volume spike
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[:-1])

        if current_volume > avg_volume * 1.5 and prediction == "BULL":
            # Volume spike confirms bullish move
            confidence += 0.1
            logger.info("📈 Volume spike confirms bullish move")
        elif current_volume > avg_volume * 1.5 and prediction == "BEAR":
            # Climactic volume - potential reversal
            prediction = "BULL"
            confidence = 0.65
            logger.info("🔄 Climactic volume spike suggests bullish reversal")

        return prediction, min(confidence, 0.85)

    except Exception as e:
        logger.error(f"❌ Error in volume trend prediction: {e}")
        traceback.print_exc()
        return None, 0


@lru_cache(maxsize=32)
def get_cached_volume_data(lookback=20):
    """
    Cached version of getting volume data to improve performance.
    Uses lru_cache to store recent results.

    Args:
        lookback: Number of rounds to look back

    Returns:
        dict: Volume data
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Get volume data
        cursor.execute(
            f"""
            SELECT epoch, totalAmount, bullAmount, bearAmount
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {int(lookback)}
        """
        )

        results = cursor.fetchall()
        conn.close()

        # Convert to JSON-serializable dict for caching
        return {
            "data": [
                {"epoch": r[0], "total": r[1], "bull": r[2], "bear": r[3]}
                for r in results
            ],
            "timestamp": int(time.time()),
        }

    except Exception as e:
        logger.error(f"Error in cached volume data: {e}")
        return {"data": [], "timestamp": 0}


def analyze_recent_trade_volume(lookback=10):
    """
    Analyze volume from recent trades to identify volume patterns.
    Uses get_recent_trades to access data.

    Args:
        lookback: Number of trades to analyze

    Returns:
        dict: Volume analysis results
    """
    try:
        # Use the imported get_recent_trades function
        recent_trades = get_recent_trades(lookback)

        if not recent_trades or len(recent_trades) < 3:
            return {"trend": "neutral", "strength": 0}

        # Calculate volume trend
        volumes = [trade.get("totalAmount", 0) for trade in recent_trades]

        # Check for increasing volume pattern
        increasing = all(
            volumes[i] >= volumes[i + 1] for i in range(min(3, len(volumes) - 1))
        )
        decreasing = all(
            volumes[i] <= volumes[i + 1] for i in range(min(3, len(volumes) - 1))
        )

        if increasing:
            return {"trend": "increasing", "strength": 0.7}
        elif decreasing:
            return {"trend": "decreasing", "strength": 0.6}
        else:
            return {"trend": "mixed", "strength": 0.3}

    except Exception as e:
        logger.error(f"Error analyzing recent trade volume: {e}")
        return {"trend": "error", "strength": 0}
