"""
Price swing detection for the trading bot.
Specialized in identifying rapid reversals and optimal swing trading entry points.
"""

import logging
import traceback

import numpy as np

from ..data.database import get_recent_price_changes

logger = logging.getLogger(__name__)


def detect_price_swing(price_changes=None, lookback=8):
    """
    Specialized detector for short-term price swings in both directions.
    Identifies rapid reversals and catches both bull and bear swings quickly.

    Args:
        price_changes: Optional pre-loaded price change data
        lookback: Number of periods to analyze

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or "UNKNOWN"
    """
    try:
        # Get price change data if not provided
        if not price_changes:
            price_changes = get_recent_price_changes(lookback)

        if not price_changes or len(price_changes) < 3:
            logger.warning("Not enough data for swing detection")
            return "UNKNOWN", 0.0

        # SWING DETECTION METRICS

        # 1. Direction changes in recent moves (swing detection)
        direction_changes = 0
        prev_direction = None

        for pc in price_changes[:5]:  # Focus on most recent 5 changes
            curr_direction = 1 if pc > 0 else (-1 if pc < 0 else 0)
            if (
                prev_direction is not None
                and curr_direction != prev_direction
                and curr_direction != 0
            ):
                direction_changes += 1
            prev_direction = curr_direction if curr_direction != 0 else prev_direction

        # High number of direction changes indicates choppy market (swings)
        is_swing_market = direction_changes >= 2

        # 2. Rate of change (acceleration)
        if len(price_changes) >= 3:
            rate_change = abs(price_changes[0]) - abs(price_changes[1])
            accel_direction = 1 if rate_change > 0 else -1

            # Use acceleration direction to adjust confidence in cases of strong acceleration
            if abs(rate_change) > 0.002:  # Significant acceleration
                logger.info(
                    f"📈 Detected {'increasing' if accel_direction > 0 else 'decreasing'} momentum"
                )

                # Store for later use in decision logic
                accel_strength = min(
                    abs(rate_change) * 40, 0.3
                )  # Cap at 0.3 additional confidence
            else:
                accel_strength = 0

            # Rate of change of the rate of change (second derivative)
            if len(price_changes) >= 4:
                prev_rate_change = abs(price_changes[1]) - abs(price_changes[2])
                rate_of_rate = rate_change - prev_rate_change

                # Positive acceleration of acceleration indicates potential swing point
                swing_acceleration = rate_of_rate > 0.001

                if swing_acceleration:
                    logger.info(f"📊 Detected swing acceleration: {rate_of_rate:.4f}")

        # 3. Most recent price change and its size
        latest_change = price_changes[0] if price_changes else 0
        prev_change = price_changes[1] if len(price_changes) > 1 else 0

        # 4. Detect counter-trend move (swing)
        counter_trend = (latest_change > 0 and prev_change < 0) or (
            latest_change < 0 and prev_change > 0
        )

        # 5. Detect momentum divergence
        momentum_divergence = False
        if len(price_changes) >= 5:
            # Calculate momentum (sum of recent changes)
            recent_momentum = sum(price_changes[:3])
            earlier_momentum = sum(price_changes[3:5])

            # Divergence occurs when momentum changes direction
            momentum_divergence = (recent_momentum > 0 and earlier_momentum < 0) or (
                recent_momentum < 0 and earlier_momentum > 0
            )

        # DECISION LOGIC FOR SWING DETECTION

        # A. Detect immediate reversal swing
        if counter_trend and abs(latest_change) > abs(prev_change) * 1.5:
            # Strong reversal - the counter move is gaining strength
            if latest_change > 0:
                prediction = "BULL"
                confidence = min(0.6 + abs(latest_change) * 30, 0.9)
                # Add accel_strength to confidence if acceleration aligns with prediction
                if accel_direction > 0:  # Acceleration is positive
                    confidence = min(confidence + accel_strength, 0.95)
                logger.info(
                    f"🔄 Detected BULL SWING: Strong upward reversal ({confidence:.2f})"
                )
            else:
                prediction = "BEAR"
                confidence = min(0.6 + abs(latest_change) * 30, 0.9)
                # Add accel_strength to confidence if acceleration aligns with prediction
                if accel_direction < 0:  # Acceleration is negative
                    confidence = min(confidence + accel_strength, 0.95)
                logger.info(
                    f"🔄 Detected BEAR SWING: Strong downward reversal ({confidence:.2f})"
                )
            return prediction, confidence

        # B. Detect momentum divergence swing
        elif momentum_divergence:
            if recent_momentum > 0:
                prediction = "BULL"
                confidence = 0.65
                logger.info(
                    f"🔀 Momentum divergence: Shifted to BULL ({confidence:.2f})"
                )
            else:
                prediction = "BEAR"
                confidence = 0.65
                logger.info(
                    f"🔀 Momentum divergence: Shifted to BEAR ({confidence:.2f})"
                )
            return prediction, confidence

        # C. Detect continued strong move
        elif latest_change != 0 and abs(latest_change) > 0.005:
            # Non-zero recent move with significant size
            if latest_change > 0:
                prediction = "BULL"
                confidence = min(0.55 + abs(latest_change) * 25, 0.85)
                logger.info(f"📈 Strong continued move: BULL ({confidence:.2f})")
            else:
                prediction = "BEAR"
                confidence = min(0.55 + abs(latest_change) * 25, 0.85)
                logger.info(f"📉 Strong continued move: BEAR ({confidence:.2f})")
            return prediction, confidence

        # D. Look for pattern in recent swings
        elif len(price_changes) >= 4:
            # Try to detect swing patterns
            pattern = detect_swing_pattern(price_changes[:5])
            if pattern:
                logger.info(f"🔍 Detected swing pattern: {pattern['name']}")
                return pattern["prediction"], pattern["confidence"]

        # E. Use swing market status when other signals unclear
        elif is_swing_market:
            # In choppy markets with direction changes but no clear pattern,
            # predict based on most recent price change but with lower confidence
            if latest_change > 0:
                prediction = "BULL"
                confidence = 0.55  # Lower confidence in choppy markets
                logger.info(
                    f"🔄 Choppy market with latest move up: Tentative BULL ({confidence:.2f})"
                )
            elif latest_change < 0:
                prediction = "BEAR"
                confidence = 0.55  # Lower confidence in choppy markets
                logger.info(
                    f"🔄 Choppy market with latest move down: Tentative BEAR ({confidence:.2f})"
                )
            else:
                return "UNKNOWN", 0.0
            return prediction, confidence

        # No clear swing signal
        return "UNKNOWN", 0.0

    except Exception as e:
        logger.error(f"❌ Error in swing detection: {e}")
        traceback.print_exc()
        return "UNKNOWN", 0.0


def detect_swing_pattern(price_changes):
    """
    Detect common swing patterns in price changes.

    Args:
        price_changes: List of recent price changes

    Returns:
        dict: Pattern information with name, prediction, and confidence or None if no pattern
    """
    try:
        if len(price_changes) < 4:
            return None

        # Convert to direction sequence for pattern recognition
        directions = []
        for pc in price_changes:
            if pc > 0.002:  # Significant up
                directions.append(1)
            elif pc < -0.002:  # Significant down
                directions.append(-1)
            else:  # Sideways
                directions.append(0)

        # Pattern: V-Bottom [sequence of downs followed by up]
        if all(d <= 0 for d in directions[1:4]) and directions[0] > 0:
            return {"name": "V-Bottom", "prediction": "BULL", "confidence": 0.75}

        # Pattern: Inverted V-Top [sequence of ups followed by down]
        if all(d >= 0 for d in directions[1:4]) and directions[0] < 0:
            return {"name": "Inverted V-Top", "prediction": "BEAR", "confidence": 0.75}

        # Pattern: Higher lows [down,up,down,up] with second down less severe
        if (
            len(directions) >= 4
            and directions[0:4] == [1, -1, 1, -1]
            and abs(price_changes[1]) < abs(price_changes[3])
        ):
            return {"name": "Higher Lows", "prediction": "BULL", "confidence": 0.7}

        # Pattern: Lower highs [up,down,up,down] with second up less strong
        if (
            len(directions) >= 4
            and directions[0:4] == [-1, 1, -1, 1]
            and price_changes[1] < price_changes[3]
        ):
            return {"name": "Lower Highs", "prediction": "BEAR", "confidence": 0.7}

        # Look for reversal indicators even in flat patterns
        if directions[0] != 0:
            # Count recent moves in opposite direction
            opposites = sum(1 for d in directions[1:3] if d != 0 and d != directions[0])
            if opposites >= 2:
                # Sign of reversal
                return {
                    "name": "Counter Trend",
                    "prediction": "BULL" if directions[0] > 0 else "BEAR",
                    "confidence": 0.65,
                }

        return None

    except Exception as e:
        logger.error(f"❌ Error detecting swing pattern: {e}")
        return None


def optimize_swing_trading(price_changes, threshold=0.001):
    """
    Optimize entry points for swing trading the short-term timeframe.

    Args:
        price_changes: Recent price changes
        threshold: Sensitivity threshold

    Returns:
        dict: Optimization parameters including direction and confidence if
              a swing opportunity is detected
    """
    try:
        if not price_changes or len(price_changes) < 3:
            return {"swing_opportunity": False}

        # Calculate swing metrics
        latest_change = price_changes[0]
        prev_changes = price_changes[1:4]

        # Detect if we're at a potential swing point
        swing_point = False

        # Case 1: Reversal after consecutive moves in same direction
        if all(pc < 0 for pc in prev_changes) and latest_change > threshold:
            # Bullish reversal after downtrend
            swing_point = True
            direction = "BULL"
            confidence = min(0.6 + abs(latest_change) * 30, 0.9)
            logger.info(
                f"⚡ Optimal BULL swing entry point detected! ({confidence:.2f})"
            )

        elif all(pc > 0 for pc in prev_changes) and latest_change < -threshold:
            # Bearish reversal after uptrend
            swing_point = True
            direction = "BEAR"
            confidence = min(0.6 + abs(latest_change) * 30, 0.9)
            logger.info(
                f"⚡ Optimal BEAR swing entry point detected! ({confidence:.2f})"
            )

        # Case 2: Continuation after brief counter move
        elif (
            prev_changes[0] < 0
            and prev_changes[1] > 0
            and latest_change > threshold * 2
        ):
            # Bullish continuation after brief pullback
            swing_point = True
            direction = "BULL"
            confidence = 0.75
            logger.info(
                f"🔄 BULL continuation after pullback: Optimal entry ({confidence:.2f})"
            )

        elif (
            prev_changes[0] > 0
            and prev_changes[1] < 0
            and latest_change < -threshold * 2
        ):
            # Bearish continuation after brief bounce
            swing_point = True
            direction = "BEAR"
            confidence = 0.75
            logger.info(
                f"🔄 BEAR continuation after bounce: Optimal entry ({confidence:.2f})"
            )

        if swing_point:
            return {
                "swing_opportunity": True,
                "direction": direction,
                "confidence": confidence,
                "comment": f"Optimal {direction} swing entry point",
            }
        else:
            return {"swing_opportunity": False}

    except Exception as e:
        logger.error(f"❌ Error optimizing swing entry: {e}")
        return {"swing_opportunity": False}


def calculate_swing_metrics(price_changes):
    """
    Calculate statistical metrics for swing analysis using numpy.

    Args:
        price_changes: List of recent price changes

    Returns:
        dict: Statistical metrics about price swings
    """
    if not price_changes or len(price_changes) < 3:
        return {}

    # Convert to numpy array for calculations
    changes_array = np.array(price_changes)

    # Calculate key metrics using numpy
    metrics = {
        "mean": float(np.mean(changes_array)),
        "volatility": float(np.std(changes_array)),
        "momentum": float(np.sum(changes_array[:3])),  # Momentum of recent changes
        "trend_strength": float(
            np.abs(np.sum(changes_array)) / np.sum(np.abs(changes_array))
        ),
        "direction_changes": float(np.sum(np.diff(np.sign(changes_array)) != 0)),
    }

    return metrics
