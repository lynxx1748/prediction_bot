"""
Signal validation for prediction strategies.
Applies validation rules to improve win rate by rejecting problematic signals.
"""

import logging
import traceback

from ..analysis.pattern import get_pattern_strength
from ..data.database import get_recent_signals

logger = logging.getLogger(__name__)


def validate_trade_signal(prediction, confidence, market_regime, round_data):
    """
    Apply validation rules to improve win rate while allowing reasonable trade volume.

    Args:
        prediction: The prediction direction ("BULL" or "BEAR")
        confidence: Confidence level of the prediction (0-1)
        market_regime: Current market regime information
        round_data: Current round data with market information

    Returns:
        tuple: (is_valid, rejection_reason) where is_valid is a boolean and
               rejection_reason is a string explaining why the signal was rejected
    """
    try:
        # Get key data points
        bull_ratio = round_data.get("bullRatio", 0.5)
        bear_ratio = round_data.get("bearRatio", 0.5)

        # Add market regime validation
        regime = market_regime.get("regime", "unknown")
        if regime == "volatile" and confidence < 0.7:
            logger.warning(f"Rejecting signal - confidence too low for volatile market")
            return False, "volatile_market"

        # Rule 1: Extreme sentiment check (avoid "obvious" traps) - RELAXED
        if prediction == "BULL" and bear_ratio > 0.9:  # INCREASED from 0.75
            logger.warning(
                f"Rejecting BULL signal - market too bearish ({bear_ratio:.2f})"
            )
            return False, "extreme_sentiment"

        if prediction == "BEAR" and bull_ratio > 0.9:  # INCREASED from 0.75
            logger.warning(
                f"Rejecting BEAR signal - market too bullish ({bull_ratio:.2f})"
            )
            return False, "extreme_sentiment"

        # Rule 2: Recent performance consistency check - RELAXED
        try:
            recent_signals = get_recent_signals(5)

            if recent_signals and len(recent_signals) >= 3:
                recent_accuracy = sum(
                    1 for s in recent_signals if s["prediction"] == s["outcome"]
                ) / len(recent_signals)

                # LOWERED from 0.4 to 0.3
                if recent_accuracy < 0.3 and confidence < 0.65:  # REDUCED from 0.7
                    logger.warning(
                        f"Rejecting signal - recent accuracy low ({recent_accuracy:.2f}) and confidence not high enough"
                    )
                    return False, "recent_performance"
        except Exception as e:
            logger.warning(f"Skipping recent performance check: {e}")

        # Rule 3: Pattern consistency with prediction - RELAXED
        try:
            pattern_str, pattern_dir = get_pattern_strength(round_data)

            if pattern_str > 0.6:  # INCREASED from 0.6
                if (pattern_dir == "bullish" and prediction == "BEAR") or (
                    pattern_dir == "bearish" and prediction == "BULL"
                ):
                    # Prediction contradicts strong pattern
                    if confidence < 0.7:  # REDUCED from 0.75
                        logger.warning(
                            f"Rejecting {prediction} signal - contradicts strong {pattern_dir} pattern"
                        )
                        return False, "pattern_contradiction"
        except Exception as e:
            logger.warning(f"Skipping pattern consistency check: {e}")

        # All validation passed
        logger.info(f"✅ Signal validation passed for {prediction} prediction")
        return True, "passed"

    except Exception as e:
        logger.error(f"❌ Error validating trade signal: {e}")
        traceback.print_exc()
        return False, "validation_error"
