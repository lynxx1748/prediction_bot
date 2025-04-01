import logging
import os
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

# Import from configuration module
from configuration import config

from .model_evaluation import ModelEvaluator

# Setup logger
logger = logging.getLogger(__name__)


class AIStrategy:
    """Advanced AI strategy for market prediction with self-learning capabilities."""

    def __init__(self, config_override=None):
        """
        Initialize AI strategy with configuration.

        Args:
            config_override: Optional config dict to override global config
        """
        # Use provided config or get from global config
        self.config = config_override or config.get("ai_strategy")

        # Initialize model and parameters
        self.model = None
        self.scaler = StandardScaler()
        self.min_samples = self.config.get("training", {}).get("min_samples", 100)
        self.retrain_frequency = self.config.get("training", {}).get(
            "retrain_frequency", 10
        )
        self.samples_since_training = 0
        self.feature_weights = self.config.get("training", {}).get(
            "feature_weights", {}
        )

        # Add model_version attribute
        self.model_version = 1

        # Get model file path from configuration
        self.model_path = config.get("paths.model_file", "models/ai_model.joblib")

        # Initialize evaluator for performance tracking
        self.evaluator = ModelEvaluator("ai_strategy")

        # Tracking variables
        self.last_trained = 0
        self.training_interval = 3600 * 6  # Retrain every 6 hours
        self.recent_predictions = {}
        self.rounds_since_training = 0
        self.min_rounds_between_training = 100

        # Define base features that we know exist in historical data
        self.base_features = ["bullRatio", "bearRatio", "bnb_change", "btc_change"]

        # Track failed predictions to enable learning
        self.failed_predictions = []
        self.max_failed_storage = 50  # Store last 50 failed predictions

        # Initialize selected model
        if self.config.get("models", {}).get("random_forest", {}).get("enabled", True):
            self.model_type = "random_forest"
            self.model_params = (
                self.config.get("models", {})
                .get("random_forest", {})
                .get(
                    "parameters",
                    {"n_estimators": 100, "max_depth": 5, "min_samples_split": 10},
                )
            )
            self.model = RandomForestClassifier(**self.model_params)
        elif self.config.get("models", {}).get("xgboost", {}).get("enabled", False):
            import xgboost as xgb

            self.model_type = "xgboost"
            self.model_params = (
                self.config.get("models", {})
                .get("xgboost", {})
                .get(
                    "parameters",
                    {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1},
                )
            )
            self.model = xgb.XGBClassifier(**self.model_params)

        logger.info(f"AI Strategy initialized with {self.model_type}")

        # Load existing model or train new one
        self._initialize_model()

    def _initialize_model(self):
        """Load existing model or prepare for training a new one."""
        try:
            if os.path.exists(self.model_path):
                # Load existing model
                model_data = joblib.load(self.model_path)

                # Handle different possible formats
                if isinstance(model_data, tuple) and len(model_data) >= 3:
                    self.model, self.scaler, self.base_features = model_data[:3]
                    if len(model_data) > 3:
                        # Version was saved with the model
                        self.evaluator.current_version = model_data[3]
                elif hasattr(model_data, "predict_proba"):
                    # Just the model was saved
                    self.model = model_data

                # Initialize the scaler if it wasn't loaded
                if self.scaler is None:
                    self.scaler = StandardScaler()

                logger.info(
                    f"Loaded AI model (version {self.evaluator.current_version}) from {self.model_path}"
                )
                return True
            else:
                logger.info(
                    "No existing AI model found, will train when sufficient data is available"
                )
                return False
        except Exception as e:
            logger.error(f"Error loading AI model: {e}")
            traceback.print_exc()
            return False

    def train(self, force=False, limit=1000):
        """
        Train the model on historical data.

        Args:
            force: Force training even if minimum conditions not met
            limit: Maximum number of samples to use for training

        Returns:
            bool: True if training was successful
        """
        # Track time spent on training
        start_time = time.time()

        try:
            # Check if we have enough new samples since last training
            if (
                not force
                and self.rounds_since_training < self.min_rounds_between_training
            ):
                logger.info(
                    f"Not enough new rounds ({self.rounds_since_training}) since last training"
                )
                return False

            # Get database paths from config
            db_file = config.get("database.file")
            prediction_table = config.get("database.tables.predictions")

            if not db_file or not os.path.exists(db_file):
                logger.error(f"Database file not found: {db_file}")
                return False

            # Connect to database
            conn = sqlite3.connect(db_file)

            # Query for training data
            query = f"""
            SELECT * FROM {prediction_table}
            WHERE final_prediction IS NOT NULL
            AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT {limit}
            """

            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                logger.warning("No training data available")
                return False

            # Calculate outcome (1 for BULL, 0 for BEAR)
            df["outcome_binary"] = (df["actual_outcome"] == "BULL").astype(int)

            # Select features based on availability
            available_features = [f for f in self.base_features if f in df.columns]

            if not available_features:
                logger.error("No valid features found in dataset")
                return False

            # Prepare feature matrix and target vector
            X = df[available_features].fillna(0)
            y = df["outcome_binary"]

            # Scale features
            X_scaled = self.scaler.fit_transform(X)

            # Train the model
            self.model.fit(X_scaled, y)

            # Evaluate on training data
            training_accuracy = self.model.score(X_scaled, y)

            # Record the training event and metrics
            self.evaluator.record_metrics(
                {
                    "accuracy": training_accuracy,
                    "training_samples": len(df),
                    "features_used": len(available_features),
                    "training_time": time.time() - start_time,
                }
            )

            # If there are significant changes or improvements, increment version
            if force or training_accuracy > 0.60:
                reason = (
                    f"Forced training"
                    if force
                    else f"Improved accuracy to {training_accuracy:.2f}"
                )
                new_version = self.evaluator.increment_version(reason)
            else:
                new_version = self.evaluator.current_version

            # Save the model
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(
                (self.model, self.scaler, available_features, new_version),
                self.model_path,
            )

            # Record successful training
            self.last_trained = time.time()
            self.rounds_since_training = 0

            # Record feature importance if available
            if hasattr(self.model, "feature_importances_"):
                feature_importance = dict(
                    zip(available_features, self.model.feature_importances_)
                )
                logger.info(f"Feature importance: {feature_importance}")

                # Update feature weights if configured
                if self.config.get("training", {}).get("update_weights", False):
                    for feature, importance in feature_importance.items():
                        self.feature_weights[feature] = importance

            logger.info(
                f"AI model trained successfully (v{new_version}). "
                f"Accuracy: {training_accuracy:.2f}, Samples: {len(df)}"
            )

            # Calculate and log class distribution
            class_distribution = (
                df["outcome_binary"].value_counts(normalize=True).to_dict()
            )
            logger.info(f"Class distribution: {class_distribution}")

            return True

        except Exception as e:
            logger.error(f"Error training AI model: {e}")
            traceback.print_exc()
            return False

    def predict(self, data):
        """
        Generate a prediction using the trained model.

        Args:
            data: Dictionary containing feature values

        Returns:
            tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
        """
        try:
            # Increment rounds since training
            self.rounds_since_training += 1

            # Check if model exists and is trained
            if self.model is None:
                logger.warning(
                    "No trained model available, returning default prediction"
                )
                return "BULL", 0.51  # Default to slight bull bias

            # Extract features from data, use zero for missing features
            epoch = data.get("epoch", 0)
            features = np.array([[data.get(f, 0) for f in self.base_features]])

            try:
                # Scale features
                features_scaled = self.scaler.transform(features)

                # Get probability prediction
                probs = self.model.predict_proba(features_scaled)[0]

                # BEAR is index 0, BULL is index 1
                bear_prob, bull_prob = probs

                # Calculate prediction and confidence based on probabilities
                prediction = "BULL" if bull_prob > bear_prob else "BEAR"

                # Calculate confidence based on the difference between probabilities
                confidence = abs(bull_prob - bear_prob)

                # Apply additional confidence scaling if needed
                confidence = min(
                    0.5 + confidence, 0.95
                )  # Scale confidence between 0.5-0.95

                # Log both probabilities for analysis
                logger.info(
                    f"AI Model Probabilities: BEAR={bear_prob:.4f}, BULL={bull_prob:.4f}"
                )
                logger.info(
                    f"AI Prediction: {prediction} with confidence {confidence:.4f}"
                )

                # Store prediction in history for learning
                if epoch > 0:
                    self.recent_predictions[epoch] = {
                        "timestamp": datetime.now().isoformat(),
                        "prediction": prediction,
                        "confidence": confidence,
                        "features": {f: data.get(f, 0) for f in self.base_features},
                    }

                # Check if we should retrain based on accumulated data
                self._check_retrain_condition()

                return prediction, confidence

            except NotFittedError:
                logger.warning("Model not fitted yet, returning default prediction")
                return "BULL", 0.51

        except Exception as e:
            logger.error(f"Error in AI prediction: {e}")
            traceback.print_exc()
            # Return a default prediction with low confidence
            return "BULL", 0.51  # Slight bull bias as fallback

    def _check_retrain_condition(self):
        """Check if the model should be retrained based on various conditions."""
        try:
            # Check time-based condition
            time_since_training = time.time() - self.last_trained
            if time_since_training > self.training_interval:
                logger.info(
                    f"Retraining model due to time interval ({time_since_training/3600:.1f} hours since last training)"
                )
                self.train(force=True)
                return

            # Check performance-based condition using evaluator
            should_retrain, reason = self.evaluator.should_retrain()
            if should_retrain:
                logger.info(f"Retraining model due to performance: {reason}")
                self.train(force=True)
                return

            # Check accumulated rounds condition
            if self.rounds_since_training >= self.min_rounds_between_training:
                logger.info(
                    f"Retraining model due to accumulated rounds ({self.rounds_since_training})"
                )
                self.train()
                return

        except Exception as e:
            logger.error(f"Error checking retrain condition: {e}")

    def record_outcome(self, epoch, prediction, outcome, round_data=None):
        """
        Record the outcome of a prediction to improve the model.

        Args:
            epoch: The epoch number
            prediction: Our prediction (BULL or BEAR)
            outcome: The actual outcome (BULL or BEAR)
            round_data: Optional full round data

        Returns:
            bool: Success status
        """
        try:
            # Keep track of outcomes for learning
            if not hasattr(self, "outcome_history"):
                self.outcome_history = []

            # Create a record with the basic information
            record = {
                "epoch": epoch,
                "prediction": prediction,
                "outcome": outcome,
                "correct": prediction == outcome,
                "timestamp": int(time.time()),
            }

            # Add more data if provided
            if round_data:
                record["bull_ratio"] = round_data.get("bullRatio", 0.5)
                record["bear_ratio"] = round_data.get("bearRatio", 0.5)
                record["total_amount"] = round_data.get("totalAmount", 0)

            # Store in history
            self.outcome_history.append(record)

            # If prediction was wrong, add to failed predictions for analysis
            if prediction != outcome:
                if not hasattr(self, "failed_predictions"):
                    self.failed_predictions = []

                # Only store limited number of failures
                if len(self.failed_predictions) >= self.max_failed_storage:
                    self.failed_predictions.pop(0)  # Remove oldest

                self.failed_predictions.append(record)

            # Increment counter for samples since training
            self.samples_since_training += 1

            return True

        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            return False

    def evaluate_performance(self):
        """
        Evaluate the AI strategy's performance.

        Returns:
            dict: Performance metrics
        """
        try:
            # Change from relative import to absolute import
            from scripts.data.database import get_prediction_history

            # Get recent predictions with AI strategy
            predictions = get_prediction_history(limit=100, strategy="ai")

            if not predictions or len(predictions) < 5:
                logger.warning("⚠️ Not enough data to evaluate AI performance")
                return {"win_rate": 0, "sample_size": 0, "valid": False}

            # Calculate metrics
            total = len(predictions)
            wins = sum(1 for p in predictions if p.get("win") == 1)
            win_rate = wins / total if total > 0 else 0

            return {
                "win_rate": win_rate,
                "total_predictions": total,
                "wins": wins,
                "losses": total - wins,
                "valid": total >= 5,
            }

        except Exception as e:
            logger.error(f"Error evaluating AI performance: {e}")
            traceback.print_exc()
            return {"win_rate": 0, "sample_size": 0, "valid": False}

    def get_version_history(self):
        """
        Get the version history of the model.

        Returns:
            list: Version history
        """
        return self.evaluator.metrics_history["versions"]

    def analyze_failed_predictions(self):
        """
        Analyze failed predictions to identify patterns.

        Returns:
            dict: Analysis results
        """
        if not self.failed_predictions:
            return {"error": "No failed predictions recorded"}

        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame(self.failed_predictions)

            # Extract features
            features_df = pd.DataFrame([p["features"] for p in self.failed_predictions])

            # Calculate feature statistics for failed predictions
            feature_stats = {}
            for feature in self.base_features:
                if feature in features_df.columns:
                    feature_stats[feature] = {
                        "mean": features_df[feature].mean(),
                        "median": features_df[feature].median(),
                        "std": features_df[feature].std(),
                        "min": features_df[feature].min(),
                        "max": features_df[feature].max(),
                    }

            # Calculate outcome distribution
            outcome_counts = df["outcome"].value_counts().to_dict()

            # Calculate confidence distribution
            confidence_stats = {
                "mean": df["confidence"].mean(),
                "median": df["confidence"].median(),
                "std": df["confidence"].std(),
            }

            return {
                "count": len(df),
                "feature_stats": feature_stats,
                "outcome_distribution": outcome_counts,
                "confidence_stats": confidence_stats,
            }

        except Exception as e:
            logger.error(f"Error analyzing failed predictions: {e}")
            return {"error": str(e)}

    def get_feature_importance(self):
        """
        Get feature importance from the model.

        Returns:
            dict: Feature importance
        """
        if not self.model or not hasattr(self.model, "feature_importances_"):
            return {}

        return dict(zip(self.base_features, self.model.feature_importances_))

    def learn_from_failures(self):
        """
        Analyze failed predictions and adjust model strategy accordingly.

        Returns:
            bool: True if adjustments were made, False otherwise
        """
        if len(self.failed_predictions) < 5:
            logger.info("Not enough failed predictions to learn from yet")
            return False

        try:
            analysis = self.analyze_failed_predictions()
            if "error" in analysis:
                logger.warning(
                    f"Could not analyze failed predictions: {analysis['error']}"
                )
                return False

            logger.info(f"Learning from {analysis['count']} failed predictions")

            # Extract feature patterns from failures
            feature_stats = analysis.get("feature_stats", {})

            # Check if we have clear patterns in the failures
            pattern_detected = False
            adjusted_weights = {}

            # Analyze each feature's contribution to failure
            for feature, stats in feature_stats.items():
                if feature not in self.feature_weights:
                    continue

                # Check if feature has extreme values in failed predictions
                if stats["std"] < 0.1 and (stats["mean"] > 0.8 or stats["mean"] < 0.2):
                    # Feature has consistent extreme values in failures - adjust its weight
                    new_weight = max(
                        0.1, min(3.0, 1.0 / self.feature_weights.get(feature, 1.0))
                    )
                    adjusted_weights[feature] = new_weight
                    pattern_detected = True
                    logger.info(
                        f"Adjusting weight for {feature} from {self.feature_weights.get(feature, 1.0):.2f} to {new_weight:.2f}"
                    )

            # Apply the adjusted weights
            if pattern_detected:
                for feature, new_weight in adjusted_weights.items():
                    self.feature_weights[feature] = new_weight

                # Create a new version to track this adjustment
                self.evaluator.increment_version(
                    f"Adjusted weights based on {analysis['count']} failed predictions"
                )
                return True

            # If no clear pattern, but have many failures, consider more drastic action
            if analysis["count"] > 20:
                logger.warning(
                    "Many failures with no clear pattern - considering model rebuild"
                )
                if analysis.get("confidence_stats", {}).get("mean", 0) > 0.7:
                    # High confidence but wrong - try inverting prediction strategy
                    logger.warning(
                        "High confidence failures detected - inverting prediction approach"
                    )
                    # Invert all feature weights
                    for feature in self.feature_weights:
                        self.feature_weights[feature] = 1.0 / max(
                            0.1, self.feature_weights[feature]
                        )

                    self.evaluator.increment_version(
                        "Inverted weights due to high-confidence failures"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Error learning from failures: {e}")
            traceback.print_exc()
            return False

    def self_optimize(self, iterations=3):
        """
        Run a complete self-optimization cycle.

        Args:
            iterations: Number of optimization iterations to run

        Returns:
            dict: Results of optimization
        """
        results = {
            "starting_accuracy": None,
            "ending_accuracy": None,
            "iterations": 0,
            "retrained": False,
            "weights_adjusted": False,
            "version_incremented": False,
        }

        try:
            # Get initial performance
            initial_performance = self.evaluate_performance()
            if initial_performance:
                results["starting_accuracy"] = initial_performance.get("accuracy")

            # Run optimization iterations
            for i in range(iterations):
                logger.info(f"Running self-optimization iteration {i+1}/{iterations}")
                results["iterations"] += 1

                # Step 1: Check if we should retrain based on performance
                should_retrain, reason = self.evaluator.should_retrain()
                if should_retrain:
                    logger.info(f"Retraining triggered: {reason}")
                    if self.train(force=True):
                        results["retrained"] = True

                # Step 2: Learn from failures regardless of retraining
                if self.learn_from_failures():
                    results["weights_adjusted"] = True

                # Step 3: Apply the model with current settings to recent data
                # This would typically be done through normal operation

                # Step 4: Check if our changes improved performance
                new_performance = self.evaluate_performance()

                # If we have both measurements, check if we improved
                if initial_performance and new_performance:
                    old_acc = initial_performance.get("accuracy", 0)
                    new_acc = new_performance.get("accuracy", 0)

                    if new_acc > old_acc + 0.05:  # 5% improvement
                        logger.info(
                            f"Optimization successful: Accuracy improved from {old_acc:.2f} to {new_acc:.2f}"
                        )
                        results["ending_accuracy"] = new_acc
                        # Increment version to mark the improvement
                        self.evaluator.increment_version(
                            f"Self-optimization improved accuracy by {new_acc-old_acc:.2f}"
                        )
                        results["version_incremented"] = True
                        # No need for further iterations if we improved significantly
                        break

            # Final performance check if not already set
            if not results["ending_accuracy"]:
                final_performance = self.evaluate_performance()
                if final_performance:
                    results["ending_accuracy"] = final_performance.get("accuracy")

            return results

        except Exception as e:
            logger.error(f"Error in self-optimization: {e}")
            traceback.print_exc()
            return results

    def load_model(self):
        """
        Load the pre-trained AI model for predictions.

        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            # Get model paths
            model_paths = get_model_paths()
            model_file = model_paths["model_file"]
            scaler_file = model_paths["scaler_file"]
            
            # Create models directory if it doesn't exist
            model_dir = Path("models")
            model_dir.mkdir(exist_ok=True)

            # Initialize new model and scaler
            if not os.path.exists(model_file) or not os.path.exists(scaler_file):
                logger.info("Initializing new AI model (first run or missing files)")
                
                # Initialize new model with default parameters
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=10,
                    random_state=42
                )
                
                # Initialize and fit scaler with some default data
                self.scaler = StandardScaler()
                default_data = np.array([[0.5, 0.5, 0, 0]])  # Default feature values
                self.scaler.fit(default_data)
                
                # Save the newly initialized model and scaler
                try:
                    joblib.dump(self.model, model_file)
                    joblib.dump(self.scaler, scaler_file)
                    logger.info(f"✅ New AI model saved to {model_file}")
                    return True
                except Exception as save_error:
                    logger.error(f"❌ Error saving new model files: {save_error}")
                    return True  # Return True since we still have a working model in memory

            # Try to load existing model files
            try:
                self.model = joblib.load(model_file)
                self.scaler = joblib.load(scaler_file)
                logger.info(f"✅ AI model loaded successfully from {model_file}")
                return True
                
            except Exception as load_error:
                logger.error(f"❌ Error loading model files: {load_error}")
                logger.info("Falling back to new model initialization")
                
                # Initialize new model as fallback
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_split=10,
                    random_state=42
                )
                self.scaler = StandardScaler()
                self.scaler.fit(np.array([[0.5, 0.5, 0, 0]]))  # Default feature values
                
                # Try to save the new model
                try:
                    joblib.dump(self.model, model_file)
                    joblib.dump(self.scaler, scaler_file)
                    logger.info("✅ New fallback model saved successfully")
                except Exception as save_error:
                    logger.warning(f"Could not save fallback model: {save_error}")
                
                return True

        except Exception as e:
            logger.error(f"❌ Error in load_model: {e}")
            traceback.print_exc()
            
            # Last resort initialization
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_split=10,
                random_state=42
            )
            self.scaler = StandardScaler()
            self.scaler.fit(np.array([[0.5, 0.5, 0, 0]]))
            logger.info("✅ Initialized new model as last resort")
            return True


def get_model_paths():
    """
    Get paths to model files using Path for cross-platform compatibility.

    Returns:
        dict: Dictionary of model file paths
    """
    base_path = Path("models")

    return {
        "model_file": base_path / "ai_model.joblib",
        "scaler_file": base_path / "scaler.joblib",
        "metadata_file": base_path / "model_metadata.json",
        "history_dir": base_path / "history",
    }
