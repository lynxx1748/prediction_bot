import os
import sys
import unittest
from unittest.mock import MagicMock, mock_open, patch

import joblib
import numpy as np
import pandas as pd

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.func_ai_strategy import AIStrategy


class TestAIStrategy(unittest.TestCase):
    """Test cases for the AI Strategy class."""

    @patch("models.func_ai_strategy.config")
    def setUp(self, mock_config):
        """Set up test fixture."""
        # Mock the configuration
        mock_config.get.return_value = {
            "enabled": True,
            "models": {
                "random_forest": {
                    "enabled": True,
                    "weight": 0.7,
                    "parameters": {
                        "n_estimators": 50,
                        "max_depth": 5,
                        "min_samples_split": 10,
                    },
                },
                "xgboost": {"enabled": False, "weight": 0.3, "parameters": {}},
            },
            "training": {
                "min_samples": 50,
                "retrain_frequency": 5,
                "feature_weights": {
                    "bullRatio": 1.2,
                    "bearRatio": 1.2,
                    "bnb_change": 1.0,
                    "btc_change": 1.0,
                },
            },
        }

        # Create the AI Strategy instance
        self.ai_strategy = AIStrategy()

        # Sample data for prediction
        self.sample_data = {
            "epoch": 123,
            "bullRatio": 0.6,
            "bearRatio": 0.4,
            "bnb_change": 1.5,
            "btc_change": 0.8,
        }

    @patch("os.path.exists")
    @patch("joblib.load")
    @patch("joblib.dump")
    def test_initialize_model(self, mock_dump, mock_load, mock_exists):
        """Test model initialization."""
        # Test loading existing model
        mock_exists.return_value = True
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
        mock_load.return_value = (
            mock_model,
            MagicMock(),
            ["bullRatio", "bearRatio", "bnb_change", "btc_change"],
        )

        # Re-initialize with mocked functions
        ai_strategy = AIStrategy()

        # Test model exists and is returned by AIStrategy._initialize_model
        self.assertIsNotNone(ai_strategy.model)

    @patch("models.func_ai_strategy.config")
    @patch("joblib.dump")
    def test_train(self, mock_dump, mock_config):
        """Test model training."""
        # Mock configuration
        mock_config.get.return_value = self.ai_strategy.config

        # Mock SQLite connection and cursor
        with patch("sqlite3.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_connect.return_value = mock_conn

            # Create a sample DataFrame
            sample_df = pd.DataFrame(
                {
                    "bullRatio": np.random.uniform(0.3, 0.7, 100),
                    "bearRatio": np.random.uniform(0.3, 0.7, 100),
                    "bnb_change": np.random.uniform(-3, 3, 100),
                    "btc_change": np.random.uniform(-2, 2, 100),
                    "outcome": np.random.choice(["BULL", "BEAR"], 100),
                }
            )

            # Mock pd.read_sql to return our sample DataFrame
            pd.read_sql = MagicMock(return_value=sample_df)

            # Test training
            result = self.ai_strategy.train()
            self.assertTrue(result)

            # Check that joblib.dump was called
            mock_dump.assert_called_once()

    @patch("models.func_ai_strategy.config")
    def test_predict(self, mock_config):
        """Test prediction with trained model."""
        # Mock the configuration
        mock_config.get.return_value = self.ai_strategy.config

        # Create a simple mock model
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array(
            [[0.4, 0.6]]
        )  # 60% confidence in BULL
        self.ai_strategy.model = mock_model

        # Test prediction
        prediction, confidence = self.ai_strategy.predict(self.sample_data)

        # Should predict BULL with ~60% confidence
        self.assertEqual(prediction, "BULL")
        self.assertAlmostEqual(confidence, 0.6, places=1)

    @patch("models.func_ai_strategy.get_ai_prediction_performance")
    def test_evaluate_performance(self, mock_get_performance):
        """Test performance evaluation."""
        # Mock performance data
        mock_get_performance.return_value = {
            "accuracy": 0.65,
            "sample_size": 20,
            "bull_accuracy": 0.7,
            "bear_accuracy": 0.6,
        }

        # Test evaluation
        performance = self.ai_strategy.evaluate_performance()

        # Should return the performance data
        self.assertEqual(performance["accuracy"], 0.65)
        self.assertEqual(performance["sample_size"], 20)

    @patch("models.func_ai_strategy.config")
    def test_predict_without_model(self, mock_config):
        """Test prediction behavior when model is None."""
        # Mock the configuration
        mock_config.get.return_value = self.ai_strategy.config

        # Set model to None to simulate uninitialized model
        self.ai_strategy.model = None

        # Test prediction (should return default values)
        prediction, confidence = self.ai_strategy.predict(self.sample_data)

        # Should return a default prediction
        self.assertEqual(prediction, "BULL")
        self.assertAlmostEqual(confidence, 0.51, places=2)


if __name__ == "__main__":
    unittest.main()
