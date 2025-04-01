"""
Short-term market analysis for the trading bot.
Specializes in very short-term price movements (6-minute timeframe).
"""

import logging
import sqlite3
import traceback
from datetime import datetime, timedelta

import numpy as np

from ..core.constants import DB_FILE, TABLES
from ..data.database import get_market_balance_stats, get_recent_trades

logger = logging.getLogger(__name__)


def analyze_short_term_momentum(recent_data=None, lookback=10):
    """
    Analyze very short-term momentum specifically for 6-minute price movements.

    Args:
        recent_data: Optional pre-fetched recent trade data
        lookback: Number of recent rounds to analyze

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or "UNKNOWN"
    """
    try:
        # Get recent trade data if not provided
        if not recent_data:
            recent_data = get_recent_trades(lookback * 2)

        if not recent_data or len(recent_data) < 5:
            logger.warning("⚠️ Not enough data for short-term analysis")
            return "UNKNOWN", 0.0

        # Extract price changes within the 6-minute windows
        price_changes = []
        for trade in recent_data:
            if "lockPrice" in trade and "closePrice" in trade:
                lock_price = trade.get("lockPrice", 0)
                close_price = trade.get("closePrice", 0)

                if lock_price and close_price and lock_price > 0:
                    price_change = (close_price - lock_price) / lock_price
                    price_changes.append(price_change)

        if not price_changes or len(price_changes) < 3:
            return "UNKNOWN", 0.0

        # Calculate short-term momentum indicators

        # 1. Very recent momentum (last 1-3 rounds)
        very_recent = price_changes[:3]
        very_recent_momentum = sum(1 for p in very_recent if p > 0) / len(very_recent)

        # 2. Recent momentum (last 3-7 rounds)
        recent = price_changes[3:7] if len(price_changes) >= 7 else []
        recent_momentum = (
            sum(1 for p in recent if p > 0) / len(recent) if recent else 0.5
        )

        # 3. Medium-term momentum (last 7+ rounds)
        medium = price_changes[7:] if len(price_changes) >= 8 else []
        medium_momentum = (
            sum(1 for p in medium if p > 0) / len(medium) if medium else 0.5
        )

        # 4. Weighted momentum calculation
        weights = [0.5, 0.3, 0.2]  # Very recent, recent, medium-term
        weighted_momentum = (
            very_recent_momentum * weights[0]
            + recent_momentum * weights[1]
            + medium_momentum * weights[2]
        )

        # 5. Detect momentum shifts
        momentum_shift = very_recent_momentum - medium_momentum

        # 6. Calculate pattern recognition
        pattern = detect_price_pattern(price_changes[:8])

        # 7. Calculate microtrends
        micro_trend, micro_strength = calculate_microtrend(price_changes[:8])

        # Decision logic for 6-minute prediction
        if micro_trend == "UP" and micro_strength > 0.6:
            # Strong upward microtrend
            prediction = "BULL"
            confidence = micro_strength
            logger.info(
                f"📈 Strong 6-min UP microtrend: {prediction} ({confidence:.2f})"
            )
        elif micro_trend == "DOWN" and micro_strength > 0.6:
            # Strong downward microtrend
            prediction = "BEAR"
            confidence = micro_strength
            logger.info(
                f"📉 Strong 6-min DOWN microtrend: {prediction} ({confidence:.2f})"
            )
        elif pattern:
            # Use pattern-based prediction
            prediction = pattern["prediction"]
            confidence = pattern["confidence"]
            logger.info(
                f"🔄 6-min Pattern detected: {pattern['name']} → {prediction} ({confidence:.2f})"
            )
        elif momentum_shift > 0.3:
            # Recent positive momentum shift
            prediction = "BULL"
            confidence = 0.6 + min(momentum_shift, 0.3)
            logger.info(
                f"🔼 6-min Positive momentum shift: {prediction} ({confidence:.2f})"
            )
        elif momentum_shift < -0.3:
            # Recent negative momentum shift
            prediction = "BEAR"
            confidence = 0.6 + min(abs(momentum_shift), 0.3)
            logger.info(
                f"🔽 6-min Negative momentum shift: {prediction} ({confidence:.2f})"
            )
        elif weighted_momentum > 0.6:
            # Overall bullish momentum
            prediction = "BULL"
            confidence = weighted_momentum
            logger.info(f"📈 6-min Bullish momentum: {prediction} ({confidence:.2f})")
        elif weighted_momentum < 0.4:
            # Overall bearish momentum
            prediction = "BEAR"
            confidence = 1 - weighted_momentum
            logger.info(f"📉 6-min Bearish momentum: {prediction} ({confidence:.2f})")
        else:
            # No clear signal
            return "UNKNOWN", 0.0

        return prediction, confidence

    except Exception as e:
        logger.error(f"❌ Error in short-term momentum analysis: {e}")
        traceback.print_exc()
        return "UNKNOWN", 0.0


def detect_price_pattern(price_changes):
    """
    Detect specific short-term price patterns.

    Args:
        price_changes: List of recent price changes

    Returns:
        dict: Pattern information or None if no pattern detected
    """
    try:
        if len(price_changes) < 3:
            return None

        # Convert to up/down/flat sequence
        sequence = []
        for change in price_changes:
            if change > 0.005:  # Significant up
                sequence.append(1)
            elif change < -0.005:  # Significant down
                sequence.append(-1)
            else:  # Flat
                sequence.append(0)

        # Pattern: Double top (1,1,-1,1,-1) → bearish
        if len(sequence) >= 5 and sequence[0:5] == [1, 1, -1, 1, -1]:
            return {"name": "Double Top", "prediction": "BEAR", "confidence": 0.75}

        # Pattern: Double bottom (-1,-1,1,-1,1) → bullish
        if len(sequence) >= 5 and sequence[0:5] == [-1, -1, 1, -1, 1]:
            return {"name": "Double Bottom", "prediction": "BULL", "confidence": 0.75}

        # Pattern: Three pushes to high (1,1,1) → bearish reversal
        if len(sequence) >= 3 and sequence[0:3] == [1, 1, 1]:
            return {
                "name": "Three Pushes to High",
                "prediction": "BEAR",
                "confidence": 0.7,
            }

        # Pattern: Three pushes to low (-1,-1,-1) → bullish reversal
        if len(sequence) >= 3 and sequence[0:3] == [-1, -1, -1]:
            return {
                "name": "Three Pushes to Low",
                "prediction": "BULL",
                "confidence": 0.7,
            }

        # Pattern: Bullish engulfing (-1,1) where second move > first
        if (
            len(sequence) >= 2
            and sequence[0:2] == [-1, 1]
            and abs(price_changes[1]) > abs(price_changes[0])
        ):
            return {
                "name": "Bullish Engulfing",
                "prediction": "BULL",
                "confidence": 0.65,
            }

        # Pattern: Bearish engulfing (1,-1) where second move > first
        if (
            len(sequence) >= 2
            and sequence[0:2] == [1, -1]
            and abs(price_changes[1]) > abs(price_changes[0])
        ):
            return {
                "name": "Bearish Engulfing",
                "prediction": "BEAR",
                "confidence": 0.65,
            }

        return None

    except Exception as e:
        logger.error(f"❌ Error detecting price pattern: {e}")
        return None


def calculate_microtrend(price_changes):
    """
    Calculate very short-term microtrend specifically for 6-minute forecasting.

    Args:
        price_changes: List of recent price changes

    Returns:
        tuple: (trend, strength) where trend is "UP", "DOWN", or "NEUTRAL"
    """
    try:
        if not price_changes or len(price_changes) < 3:
            return "NEUTRAL", 0.5

        # Use stronger exponential weighting for 6-minute focus
        # This gives even more emphasis to the most recent 1-2 price moves
        weights = np.exp(
            np.linspace(0, 1.5, len(price_changes))
        )  # Increased exponent from 1 to 1.5
        weights = weights / weights.sum()

        # Calculate weighted price change with increased recency bias
        weighted_change = sum(c * w for c, w in zip(price_changes, weights))

        # Calculate momentum consistency with focus on last few moves
        up_moves = sum(1 for c in price_changes[:3] if c > 0)  # Reduced from 4 to 3
        down_moves = sum(1 for c in price_changes[:3] if c < 0)  # Reduced from 4 to 3
        total_moves = up_moves + down_moves if (up_moves + down_moves) > 0 else 1

        consistency = max(up_moves, down_moves) / total_moves

        # More sensitive thresholds for 6-minute moves (smaller changes matter)
        if weighted_change > 0.0008:  # Reduced from 0.001 for higher sensitivity
            trend = "UP"
            # Stronger confidence formula for 6-minute predictions
            strength = min(
                0.55 + abs(weighted_change) * 60 + consistency * 0.35, 0.95
            )  # Increased multipliers
        elif weighted_change < -0.0008:  # Reduced from -0.001 for higher sensitivity
            trend = "DOWN"
            strength = min(
                0.55 + abs(weighted_change) * 60 + consistency * 0.35, 0.95
            )  # Increased multipliers
        else:
            trend = "NEUTRAL"
            strength = 0.5

        return trend, strength

    except Exception as e:
        logger.error(f"❌ Error calculating microtrend: {e}")
        return "NEUTRAL", 0.5


def get_volume_acceleration(lookback=8):
    """
    Analyze volume acceleration in recent rounds.

    Args:
        lookback: Number of rounds to analyze

    Returns:
        float: Volume acceleration (-1 to 1 range)
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Get volume data for recent rounds
        cursor.execute(
            f"""
            SELECT epoch, totalAmount
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {lookback}
        """
        )

        results = cursor.fetchall()
        conn.close()

        if not results or len(results) < 4:
            return 0

        # Calculate volume changes
        volumes = [r[1] for r in results]
        volume_changes = [
            volumes[i] / volumes[i + 1] if volumes[i + 1] > 0 else 1
            for i in range(len(volumes) - 1)
        ]

        # Calculate acceleration (change in volume changes)
        recent_ratio = sum(volume_changes[:2]) / 2 if volume_changes else 1
        earlier_ratio = sum(volume_changes[2:4]) / 2 if len(volume_changes) >= 4 else 1

        acceleration = recent_ratio / earlier_ratio if earlier_ratio > 0 else 1

        return acceleration - 1  # -1 to 0 = deceleration, 0 to 1 = acceleration

    except Exception as e:
        logger.error(f"❌ Error calculating volume acceleration: {e}")
        return 0


def get_bootstrap_signal():
    """
    Get optimized 6-minute signal during bootstrap phase.
    Uses specialized techniques for limited data scenarios.

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or "UNKNOWN"
    """
    try:
        # Check market balance (even with limited data)
        market_stats = get_market_balance_stats(lookback=20)
        has_bias = market_stats["sample_size"] >= 5

        if has_bias:
            # Use recent market outcome bias if available
            if market_stats["bull_ratio"] > 0.6:
                # Clear bull bias
                bull_confidence = 0.5 + min(market_stats["bull_ratio"] - 0.5, 0.3)
                logger.info(
                    f"🔍 BOOTSTRAP BIAS: Bull trend detected ({market_stats['bull_ratio']:.2f})"
                )
                return "BULL", bull_confidence
            elif market_stats["bear_ratio"] > 0.6:
                # Clear bear bias
                bear_confidence = 0.5 + min(market_stats["bear_ratio"] - 0.5, 0.3)
                logger.info(
                    f"🔍 BOOTSTRAP BIAS: Bear trend detected ({market_stats['bear_ratio']:.2f})"
                )
                return "BEAR", bear_confidence

        # Fallback to pure price action with limited data
        recent_trades = get_recent_trades(5)

        if recent_trades and len(recent_trades) >= 2:
            # Just look at the last two completed rounds
            price_changes = []
            for trade in recent_trades[:2]:
                if "lockPrice" in trade and "closePrice" in trade:
                    lock = trade.get("lockPrice", 0)
                    close = trade.get("closePrice", 0)
                    if lock and close and lock > 0:
                        price_changes.append((close - lock) / lock)

            # Very simple momentum with limited data
            if len(price_changes) >= 2:
                # Both recent moves in same direction
                if price_changes[0] > 0 and price_changes[1] > 0:
                    logger.info("🔍 BOOTSTRAP: Two consecutive up moves")
                    return "BULL", 0.65
                elif price_changes[0] < 0 and price_changes[1] < 0:
                    logger.info("🔍 BOOTSTRAP: Two consecutive down moves")
                    return "BEAR", 0.65
                # Momentum shift
                elif price_changes[0] > 0 and price_changes[1] < 0:
                    logger.info("🔍 BOOTSTRAP: Momentum shift to positive")
                    return "BULL", 0.6
                elif price_changes[0] < 0 and price_changes[1] > 0:
                    logger.info("🔍 BOOTSTRAP: Momentum shift to negative")
                    return "BEAR", 0.6

        # No clear signal
        return "UNKNOWN", 0.0

    except Exception as e:
        logger.error(f"❌ Error in bootstrap signal: {e}")
        return "UNKNOWN", 0.0


def get_recent_time_periods():
    """
    Get time periods for short-term analysis using datetime.

    Returns:
        dict: Different time periods for analysis
    """
    now = datetime.now()

    return {
        "last_hour": {"start": now - timedelta(hours=1), "end": now},
        "last_30min": {"start": now - timedelta(minutes=30), "end": now},
        "last_6min": {"start": now - timedelta(minutes=6), "end": now},
        "next_6min": {"start": now, "end": now + timedelta(minutes=6)},
    }
