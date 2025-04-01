"""
Integration functionality for the prediction system.
Combines different prediction signals and handles special detection cases.
"""

import logging
import traceback

from ..analysis.technical import (detect_combined_technical_reversal,
                                  detect_market_reversal)

logger = logging.getLogger(__name__)


def integrate_reversal_detection(predictions, confidences, weights):
    """
    Integrate reversal detection into the prediction system.

    Args:
        predictions: Dictionary of current predictions
        confidences: Dictionary of confidence levels
        weights: Dictionary of weights for each prediction

    Returns:
        tuple: (updated_predictions, updated_confidences, updated_weights)
    """
    try:
        # Check for reversals with combined detector
        reversal_pred, reversal_conf, reason = detect_market_reversal()

        if reversal_pred != "UNKNOWN" and reversal_conf > 0.65:
            logger.warning(
                f"⚠️ REVERSAL ALERT! {reversal_pred} ({reversal_conf:.2f}): {reason}"
            )

            # Add reversal prediction to the mix
            predictions["reversal"] = reversal_pred
            confidences["reversal"] = reversal_conf

            # Give higher weight to reversal signals (they're important)
            weights["reversal"] = 0.35

            # Normalize other weights
            total_other_weight = sum(v for k, v in weights.items() if k != "reversal")
            factor = (1 - weights["reversal"]) / total_other_weight

            for k in weights:
                if k != "reversal":
                    weights[k] *= factor

        # Also check for combined technical reversals
        combined_reversal = detect_combined_technical_reversal()

        if (
            combined_reversal
            and combined_reversal["detected"]
            and reversal_pred == "UNKNOWN"
        ):
            # Only add this if the main reversal detector didn't fire
            logger.warning(
                f"⚠️ TECHNICAL REVERSAL PATTERN: {combined_reversal['direction']} ({combined_reversal['confidence']:.2f})"
            )

            # Add combined reversal prediction
            predictions["tech_reversal"] = combined_reversal["direction"]
            confidences["tech_reversal"] = combined_reversal["confidence"]
            weights["tech_reversal"] = 0.3

            # Normalize other weights again
            total_other_weight = sum(
                v for k, v in weights.items() if k != "tech_reversal"
            )
            factor = (1 - weights["tech_reversal"]) / total_other_weight

            for k in weights:
                if k != "tech_reversal" and k != "reversal":
                    weights[k] *= factor

        return predictions, confidences, weights

    except Exception as e:
        logger.error(f"❌ Error integrating reversal detection: {e}")
        traceback.print_exc()
        return predictions, confidences, weights


def integrate_signals(base_signals, market_data, round_data):
    """
    Integrate multiple prediction signals into a consolidated prediction.

    Args:
        base_signals: Dictionary of base prediction signals
        market_data: Dictionary with market data
        round_data: Dictionary with round data

    Returns:
        tuple: (final_prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Initialize with any existing signals
        predictions = {}
        confidences = {}
        weights = {}

        # Add base signals
        for signal_name, signal_data in base_signals.items():
            if "prediction" in signal_data and "confidence" in signal_data:
                predictions[signal_name] = signal_data["prediction"]
                confidences[signal_name] = signal_data["confidence"]
                weights[signal_name] = signal_data.get("weight", 1.0)

        # Check for large market moves
        if market_data and "bnb_24h_change" in market_data:
            change = float(market_data["bnb_24h_change"])
            if abs(change) > 5.0:  # Significant 24h move
                # In large moves, trend tends to continue short term
                trend_pred = "BULL" if change > 0 else "BEAR"
                trend_conf = min(
                    0.55 + abs(change) / 20, 0.75
                )  # Scale with move size, cap at 0.75

                logger.info(
                    f"📈 Large 24h price move detected: {change:.2f}% - adding trend signal"
                )

                predictions["large_move"] = trend_pred
                confidences["large_move"] = trend_conf
                weights["large_move"] = 0.25

        # Add reversal detection
        predictions, confidences, weights = integrate_reversal_detection(
            predictions, confidences, weights
        )

        # Add volume analysis
        from .analysis import analyze_volume

        volume_pred, volume_conf = analyze_volume(
            round_data.get("bullAmount", 0), round_data.get("bearAmount", 0)
        )

        if volume_pred and volume_conf > 0.6:  # Only use strong volume signals
            predictions["volume"] = volume_pred
            confidences["volume"] = volume_conf
            weights["volume"] = 0.2

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            for k in weights:
                weights[k] /= total_weight

        # Calculate weighted prediction
        bull_weight = 0
        bear_weight = 0

        for signal, prediction in predictions.items():
            if prediction == "BULL":
                bull_weight += confidences[signal] * weights[signal]
            elif prediction == "BEAR":
                bear_weight += confidences[signal] * weights[signal]

        # Determine final prediction
        if bull_weight > bear_weight:
            final_prediction = "BULL"
            confidence = 0.5 + (bull_weight - bear_weight)
        elif bear_weight > bull_weight:
            final_prediction = "BEAR"
            confidence = 0.5 + (bear_weight - bull_weight)
        else:
            # If tied, use technical analysis as tiebreaker
            from ..analysis.technical import get_technical_prediction

            tech_pred, tech_conf = get_technical_prediction(round_data)
            final_prediction = (
                tech_pred if tech_pred else "BULL"
            )  # Default to BULL if no tech prediction
            confidence = 0.5 + (
                tech_conf * 0.02
            )  # Use tech_conf to slightly adjust confidence

        # Cap confidence at 0.95
        confidence = min(confidence, 0.95)

        logger.info(
            f"🔄 Integrated {len(predictions)} signals: {final_prediction} with {confidence:.2f} confidence"
        )

        return final_prediction, confidence

    except Exception as e:
        logger.error(f"❌ Error integrating signals: {e}")
        traceback.print_exc()
        return "BULL", 0.5  # Default to a low-confidence BULL prediction on error
