import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from models.func_hybrid import hybrid_prediction


class TestHybridPrediction(unittest.TestCase):
    """Test cases for the hybrid prediction function."""

    def setUp(self):
        """Set up test fixture."""
        # Sample round data
        self.round_data = {
            "epoch": 123,
            "bullRatio": 0.6,
            "bearRatio": 0.4,
            "totalAmount": 125.5,
        }

        # Sample technical data (e.g., price history)
        self.technical_data = {
            "prices": [100, 102, 98, 103, 101, 104, 106],
            "volumes": [1000, 1200, 900, 1100, 1050, 1300, 1250],
        }

        # Sample market data
        self.market_data = {
            "bnb_24h_change": 1.5,
            "btc_24h_change": 0.8,
            "market_sentiment": "positive",
        }

        # Sample config
        self.config = {
            "STRATEGY_WEIGHTS": {
                "pattern": 0.2,
                "market": 0.2,
                "technical": 0.3,
                "sentiment": 0.3,
            }
        }

    @patch("models.func_hybrid.detect_pattern")
    @patch("models.func_hybrid.config")
    @patch("models.func_hybrid.record_prediction")
    def test_hybrid_prediction_bullish(
        self, mock_record, mock_config, mock_detect_pattern
    ):
        """Test hybrid prediction with bullish signals."""
        # Mock configuration
        mock_config.get.return_value = self.config

        # Mock pattern detection to return bullish pattern
        mock_detect_pattern.return_value = ("BULL", 0.7)

        # Call the function
        prediction, confidence = hybrid_prediction(
            self.round_data, self.technical_data, self.market_data
        )

        # Should be bullish with decent confidence
        self.assertEqual(prediction, "BULL")
        self.assertTrue(0.5 < confidence < 1.0)

        # Verify record_prediction was called
        mock_record.assert_called_once()

    @patch("models.func_hybrid.detect_pattern")
    @patch("models.func_hybrid.config")
    @patch("models.func_hybrid.record_prediction")
    def test_hybrid_prediction_bearish(
        self, mock_record, mock_config, mock_detect_pattern
    ):
        """Test hybrid prediction with bearish signals."""
        # Modify round data to be more bearish
        bearish_round_data = self.round_data.copy()
        bearish_round_data["bullRatio"] = 0.3
        bearish_round_data["bearRatio"] = 0.7

        # Mock configuration
        mock_config.get.return_value = self.config

        # Mock pattern detection to return bearish pattern
        mock_detect_pattern.return_value = ("BEAR", 0.75)

        # Call the function
        prediction, confidence = hybrid_prediction(
            bearish_round_data, self.technical_data, self.market_data
        )

        # Should be bearish with decent confidence
        self.assertEqual(prediction, "BEAR")
        self.assertTrue(0.5 < confidence < 1.0)

    @patch("models.func_hybrid.detect_pattern")
    @patch("models.func_hybrid.config")
    @patch("models.func_hybrid.record_prediction")
    def test_hybrid_prediction_error_handling(
        self, mock_record, mock_config, mock_detect_pattern
    ):
        """Test error handling in hybrid prediction."""
        # Mock pattern detection to raise an exception
        mock_detect_pattern.side_effect = Exception("Pattern detection error")

        # Mock configuration
        mock_config.get.return_value = self.config

        # Call the function - should not raise an exception
        prediction, confidence = hybrid_prediction(
            self.round_data, self.technical_data, self.market_data
        )

        # Should return "UNKNOWN" with 0.0 confidence on error
        self.assertEqual(prediction, "UNKNOWN")
        self.assertEqual(confidence, 0.0)


if __name__ == "__main__":
    unittest.main()
