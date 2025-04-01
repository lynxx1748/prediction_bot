"""
Feature engineering for the trading bot.
Prepares data features for prediction models.
"""

import logging
import os
import traceback
from datetime import datetime

import joblib
import numpy as np

logger = logging.getLogger(__name__)


def prepare_features(round_data):
    """
    Prepare features for prediction.

    Args:
        round_data: Dictionary with round data

    Returns:
        dict: Dictionary of prepared features
    """
    try:
        features = {}

        # Basic ratios (these should always be available)
        bullAmount = float(round_data.get("bullAmount", 0))
        bearAmount = float(round_data.get("bearAmount", 0))
        totalAmount = bullAmount + bearAmount

        if totalAmount > 0:
            features["bullRatio"] = bullAmount / totalAmount
            features["bearRatio"] = bearAmount / totalAmount
        else:
            features["bullRatio"] = 0.5
            features["bearRatio"] = 0.5

        # Price changes
        features["bnb_change"] = float(round_data.get("bnb_change", 0))
        features["btc_change"] = float(round_data.get("btc_change", 0))

        # Optional timestamp-based features
        timestamp = round_data.get("lockTimestamp") or round_data.get("timestamp")
        if timestamp:
            dt = datetime.fromtimestamp(int(timestamp))
            # Use both naming conventions for compatibility
            features["hour"] = dt.hour
            features["hour_of_day"] = dt.hour
            features["minute"] = dt.minute
            features["day_of_week"] = dt.weekday()

        return features

    except Exception as e:
        logger.error(f"❌ Error preparing features: {e}")
        traceback.print_exc()
        return {"bullRatio": 0.5, "bearRatio": 0.5, "bnb_change": 0, "btc_change": 0}


def encode_features(features, include_additional=True):
    """
    Encode features for model input.

    Args:
        features: Dictionary of features
        include_additional: Whether to include additional engineered features

    Returns:
        numpy.ndarray: Encoded feature array
    """
    try:
        # Core features that should always be available
        core_features = [
            features.get("bullRatio", 0.5),
            features.get("bearRatio", 0.5),
            features.get("totalAmount", 0),
            features.get("bnb_change", 0),
            features.get("btc_change", 0),
        ]

        if not include_additional:
            return np.array(core_features).reshape(1, -1)

        # Additional engineered features
        additional_features = []

        # Imbalance ratio (difference between bull and bear)
        bull_bear_imbalance = features.get("bullRatio", 0.5) - features.get(
            "bearRatio", 0.5
        )
        additional_features.append(bull_bear_imbalance)

        # Detect extreme ratios
        extreme_bull = 1 if features.get("bullRatio", 0.5) > 0.7 else 0
        extreme_bear = 1 if features.get("bearRatio", 0.5) > 0.7 else 0
        additional_features.extend([extreme_bull, extreme_bear])

        # Time-based features
        if "hour_of_day" in features:
            # Encode hour as sin/cos for cyclical nature
            hour = features["hour_of_day"]
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            additional_features.extend([hour_sin, hour_cos])

        if "day_of_week" in features:
            # Encode day as sin/cos for cyclical nature
            day = features["day_of_week"]
            day_sin = np.sin(2 * np.pi * day / 7)
            day_cos = np.cos(2 * np.pi * day / 7)
            additional_features.extend([day_sin, day_cos])

        # Combine all features
        all_features = core_features + additional_features
        return np.array(all_features).reshape(1, -1)

    except Exception as e:
        logger.error(f"❌ Error encoding features: {e}")
        traceback.print_exc()
        # Return basic feature array on error
        return np.array([0.5, 0.5, 0, 0, 0]).reshape(1, -1)


def print_model_features(model_path="random_forest_model.pkl"):
    """
    Print the feature names used by the trained model.

    Args:
        model_path: Path to the model file

    Returns:
        list: Feature names or None on failure
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None

        model = joblib.load(model_path)

        if hasattr(model, "feature_names_in_"):
            logger.info("Model was trained with these features:")
            for i, feature_name in enumerate(model.feature_names_in_):
                logger.info(f"  {i+1}. {feature_name}")
            return list(model.feature_names_in_)
        else:
            logger.warning("Model does not have feature names stored.")
            return None

    except Exception as e:
        logger.error(f"❌ Error printing model features: {e}")
        traceback.print_exc()
        return None


def get_feature_importance(model_path="random_forest_model.pkl"):
    """
    Get feature importance from the model.

    Args:
        model_path: Path to the model file

    Returns:
        dict: Feature names and their importance, or None on failure
    """
    try:
        if not os.path.exists(model_path):
            logger.error(f"❌ Model file not found: {model_path}")
            return None

        model = joblib.load(model_path)

        # Check if model has feature importance
        if not hasattr(model, "feature_importances_"):
            logger.warning("Model does not have feature importances")
            return None

        # Check if model has feature names
        if not hasattr(model, "feature_names_in_"):
            logger.warning("Model does not have feature names")
            return None

        # Create dictionary of feature name to importance
        feature_importance = {}
        for name, importance in zip(
            model.feature_names_in_, model.feature_importances_
        ):
            feature_importance[name] = importance

        # Sort by importance
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )

        # Log top features
        logger.info("Feature importance:")
        for name, importance in sorted_features:
            logger.info(f"  {name}: {importance:.4f}")

        return dict(sorted_features)

    except Exception as e:
        logger.error(f"❌ Error getting feature importance: {e}")
        traceback.print_exc()
        return None
