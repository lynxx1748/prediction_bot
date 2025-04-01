import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.func_ai_strategy import AIStrategy


class TestAISelfLearning(unittest.TestCase):
    """Test the self-learning capabilities of the AI Strategy."""

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
                    "parameters": {"n_estimators": 10, "max_depth": 3},
                }
            },
            "training": {
                "min_samples": 10,
                "retrain_frequency": 2,
                "feature_weights": {
                    "bullRatio": 1.0,
                    "bearRatio": 1.0,
                    "bnb_change": 1.0,
                    "btc_change": 1.0,
                },
            },
        }

        # Create the AI Strategy instance
        with patch("models.func_ai_strategy.ModelEvaluator"):
            self.ai_strategy = AIStrategy()

    def test_failed_prediction_tracking(self):
        """Test that failed predictions are tracked correctly."""
        # Mock the predict method to always return BULL
        self.ai_strategy.predict = MagicMock(return_value=("BULL", 0.8))

        # Record a correct prediction
        sample_data1 = {"epoch": 1, "bullRatio": 0.6, "bearRatio": 0.4}
        self.ai_strategy.record_outcome(1, "BULL", sample_data1)

        # Record an incorrect prediction
        sample_data2 = {"epoch": 2, "bullRatio": 0.3, "bearRatio": 0.7}
        self.ai_strategy.record_outcome(2, "BEAR", sample_data2)

        # Check that failed prediction was tracked
        self.assertEqual(len(self.ai_strategy.failed_predictions), 1)
        self.assertEqual(self.ai_strategy.failed_predictions[0]["epoch"], 2)

    @patch("models.func_ai_strategy.ModelEvaluator")
    def test_learn_from_failures(self, mock_evaluator):
        """Test learning from failed predictions."""
        # Add some failed predictions
        for i in range(10):
            self.ai_strategy.failed_predictions.append(
                {
                    "epoch": i,
                    "prediction": "BULL",
                    "actual": "BEAR",
                    "confidence": 0.8,
                    "features": {
                        "bullRatio": 0.9,  # Very high bull ratio in failures
                        "bearRatio": 0.1,
                        "bnb_change": 0.5,
                        "btc_change": 0.5,
                    },
                }
            )

        # Initial weights
        initial_weights = self.ai_strategy.feature_weights.copy()

        # Run learning from failures
        self.ai_strategy.learn_from_failures()

        # Check that weights were adjusted
        self.assertNotEqual(self.ai_strategy.feature_weights, initial_weights)

        # Specifically, the bullRatio weight should be reduced
        self.assertLess(
            self.ai_strategy.feature_weights["bullRatio"], initial_weights["bullRatio"]
        )

    @patch("models.func_ai_strategy.get_performance")
    def test_self_optimization(self, mock_get_performance):
        """Test the self-optimization loop."""
        # Mock initial poor performance
        mock_get_performance.return_value = {"accuracy": 0.4, "sample_size": 20}

        # Mock training success
        self.ai_strategy.train = MagicMock(return_value=True)

        # Mock learning from failures
        self.ai_strategy.learn_from_failures = MagicMock(return_value=True)

        # Add failed predictions data
        for i in range(10):
            self.ai_strategy.failed_predictions.append(
                {
                    "epoch": i,
                    "prediction": "BULL",
                    "actual": "BEAR",
                    "confidence": 0.8,
                    "features": {"bullRatio": 0.7, "bearRatio": 0.3},
                }
            )

        # Mock improved performance after optimization
        mock_get_performance.side_effect = [
            {"accuracy": 0.4, "sample_size": 20},  # Initial
            {"accuracy": 0.6, "sample_size": 25},  # After optimization
        ]

        # Run self-optimization
        results = self.ai_strategy.self_optimize(iterations=1)

        # Check results
        self.assertTrue(results["retrained"])
        self.assertTrue(results["weights_adjusted"])
        self.assertEqual(results["starting_accuracy"], 0.4)
        self.assertEqual(results["ending_accuracy"], 0.6)


if __name__ == "__main__":
    unittest.main()
