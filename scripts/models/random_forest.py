"""
Random Forest model implementation for the trading bot.
Handles training, prediction, and model management.
"""

import logging
import os
import traceback

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from ..core.constants import MODEL_FILE, SCALER_FILE, config
from ..data.features import prepare_features

logger = logging.getLogger(__name__)


def get_model_prediction(round_data):
    """
    Get prediction from the trained Random Forest model.

    Args:
        round_data: Dictionary with round data

    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Prepare features
        feature_dict = prepare_features(round_data)

        # Convert dictionary to list in correct order
        features = [
            feature_dict.get("bullRatio", 0.5),
            feature_dict.get("bearRatio", 0.5),
            feature_dict.get("bnb_change", 0),
            feature_dict.get("btc_change", 0),
        ]

        # Load model
        try:
            model = joblib.load("models/prediction_model.joblib")
        except Exception as e:
            logger.debug(f"Could not load model: {e}")
            return None, 0

        # Make prediction
        model_prediction_raw = model.predict([features])[0]
        confidence = max(model.predict_proba([features])[0])

        # Convert to BULL/BEAR
        prediction = "BULL" if model_prediction_raw == 1 else "BEAR"

        return prediction, confidence

    except Exception as e:
        logger.error(f"❌ Error in model prediction: {e}")
        traceback.print_exc()
        return None, 0


def load_model_and_scaler():
    """
    Load the trained model and scaler.

    Returns:
        tuple: (model, scaler) or (None, None) on failure
    """
    try:
        # Check if files exist
        if not os.path.exists(MODEL_FILE):
            logger.error(f"❌ Model file not found at: {MODEL_FILE}")
            return None, None

        if not os.path.exists(SCALER_FILE):
            logger.error(f"❌ Scaler file not found at: {SCALER_FILE}")
            return None, None

        # Load model and scaler
        model = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)

        logger.info(f"✅ Loaded model from {MODEL_FILE}")
        logger.info(f"✅ Loaded scaler from {SCALER_FILE}")

        return model, scaler

    except Exception as e:
        logger.error(f"❌ Error loading model and scaler: {e}")
        traceback.print_exc()
        return None, None


def get_training_data():
    """
    Get historical data for model training with detailed error reporting.

    Returns:
        dict: Training data dictionary with 'X' and 'y' keys, or None on failure
    """
    try:
        import sqlite3

        from ..core.constants import DB_FILE, TABLES

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Count total records first
        cursor.execute(f"SELECT COUNT(*) FROM {TABLES['predictions']}")
        total_records = cursor.fetchone()[0]

        # Count records with actual outcome
        cursor.execute(
            f"SELECT COUNT(*) FROM {TABLES['predictions']} WHERE actual_outcome IS NOT NULL"
        )
        outcome_records = cursor.fetchone()[0]

        logger.info(
            f"📊 Database contains {total_records} total records, {outcome_records} with outcomes"
        )

        # Get historical rounds with outcomes - less restrictive query
        query = f"""
        SELECT 
            bullRatio,
            bearRatio,
            bnb_change,
            btc_change,
            CASE 
                WHEN actual_outcome = 'BULL' THEN 1
                WHEN actual_outcome = 'BEAR' THEN 0
                ELSE NULL
            END as outcome
        FROM {TABLES['predictions']}
        WHERE actual_outcome IS NOT NULL
        ORDER BY epoch DESC
        LIMIT 5000
        """

        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            logger.warning("⚠️ No training data found in database after querying")
            return None

        # Convert to features and labels
        X = []  # Features
        y = []  # Labels (outcomes)

        for row in rows:
            # Only skip rows where outcome is None
            if row[-1] is not None:
                features = []
                valid_row = True

                # Check each feature for None
                for i in range(len(row) - 1):
                    if row[i] is None:
                        valid_row = False
                        break
                    features.append(row[i])

                if valid_row:
                    X.append(features)
                    y.append(row[-1])

        logger.info(
            f"✅ Got {len(X)} usable training samples from {len(rows)} database records"
        )
        return {"X": X, "y": y}

    except Exception as e:
        logger.error(f"❌ Error getting training data: {e}")
        traceback.print_exc()
        return None


def train_model(force=False):
    """
    Train or load the prediction model.

    Args:
        force: Whether to force retraining even if model exists

    Returns:
        tuple: (model, scaler) or (None, None) on failure
    """
    try:
        # Try to load existing model first
        model, scaler = load_model_and_scaler()
        if model and scaler and not force:
            return model, scaler

        logger.info("🔄 Training new model...")

        # Get training data
        training_data = get_training_data()
        if training_data is None or len(training_data["X"]) < config.get(
            "model", {}
        ).get("update_threshold", 50):
            logger.warning(
                f"⚠️ Not enough training data: {len(training_data['X']) if training_data else 0} samples"
            )
            return None, None

        # Train model
        X = training_data["X"]
        y = training_data["y"]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=10, random_state=42
        )
        model.fit(X_scaled, y)

        # Save model and scaler
        os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
        os.makedirs(os.path.dirname(SCALER_FILE), exist_ok=True)

        joblib.dump(model, MODEL_FILE)
        joblib.dump(scaler, SCALER_FILE)

        logger.info(f"✅ Saved model to {MODEL_FILE}")
        logger.info(f"✅ Saved scaler to {SCALER_FILE}")

        return model, scaler

    except Exception as e:
        logger.error(f"❌ Error training model: {e}")
        traceback.print_exc()
        return None, None


def evaluate_model_performance(test_size=0.2):
    """
    Evaluate the model's performance on held-out data.

    Args:
        test_size: Portion of data to use for testing

    Returns:
        dict: Performance metrics
    """
    try:
        from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                                     recall_score)
        from sklearn.model_selection import train_test_split

        # Get training data
        training_data = get_training_data()
        if training_data is None or len(training_data["X"]) < 50:
            logger.warning(
                f"⚠️ Not enough data for evaluation: {len(training_data['X']) if training_data else 0} samples"
            )
            return None

        X = np.array(training_data["X"])
        y = np.array(training_data["y"])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_split=10, random_state=42
        )
        model.fit(X_train_scaled, y_train)

        # Make predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        # Calculate feature importance
        feature_importance = dict(
            zip(
                ["bullRatio", "bearRatio", "bnb_change", "btc_change"],
                model.feature_importances_,
            )
        )

        # Log results
        logger.info(f"📊 Model Evaluation:")
        logger.info(f"   Accuracy: {accuracy:.4f}")
        logger.info(f"   Precision: {precision:.4f}")
        logger.info(f"   Recall: {recall:.4f}")
        logger.info(f"   F1 Score: {f1:.4f}")
        logger.info(f"   Sample Size: {len(y_test)}")

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sample_size": len(y_test),
            "feature_importance": feature_importance,
        }

    except Exception as e:
        logger.error(f"❌ Error evaluating model: {e}")
        traceback.print_exc()
        return None


def optimize_model_hyperparameters():
    """
    Find optimal hyperparameters for the model.

    Returns:
        dict: Best parameters and performance metrics
    """
    try:
        from sklearn.metrics import f1_score
        from sklearn.model_selection import GridSearchCV, train_test_split

        # Get training data
        training_data = get_training_data()
        if training_data is None or len(training_data["X"]) < 100:
            logger.warning(
                f"⚠️ Not enough data for optimization: {len(training_data['X']) if training_data else 0} samples"
            )
            return None

        X = np.array(training_data["X"])
        y = np.array(training_data["y"])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Scale test data too

        # Define parameter grid
        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        }

        # Create model
        model = RandomForestClassifier(random_state=42)

        # Create grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
        )

        # Fit grid search
        logger.info("🔍 Starting hyperparameter optimization...")
        grid_search.fit(X_train_scaled, y_train)

        # Get best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Evaluate on test set
        best_model = grid_search.best_estimator_
        test_pred = best_model.predict(X_test_scaled)
        test_score = f1_score(y_test, test_pred)

        logger.info(f"✅ Best parameters: {best_params}")
        logger.info(f"✅ Best cross-validation score: {best_score:.4f}")
        logger.info(f"✅ Test set score: {test_score:.4f}")

        return {
            "best_params": best_params,
            "best_score": best_score,
            "test_score": test_score,
        }

    except Exception as e:
        logger.error(f"❌ Error optimizing model: {e}")
        traceback.print_exc()
        return None
