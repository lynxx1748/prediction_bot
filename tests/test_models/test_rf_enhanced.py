import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.func_rf_enhanced import AdaptiveRandomForest


class TestAdaptiveRandomForest(unittest.TestCase):
    """Test cases for the Adaptive Random Forest model."""
    
    @patch('models.func_rf_enhanced.config')
    def setUp(self, mock_config):
        """Set up test fixture."""
        # Mock the configuration
        mock_config.get.return_value = {
            'parameters': {
                'n_estimators': 10,
                'max_depth': 3,
                'min_samples_split': 2
            },
            'min_samples': 20,
            'recency_weight': True,
            'recency_halflife': 100,
            'retrain_threshold': 5,
            'retrain_accuracy_threshold': 0.6
        }
        
        # Create model with mocked dependencies
        with patch('models.func_rf_enhanced.ModelEvaluator'), \
             patch('models.func_rf_enhanced.ModelVersionControl'), \
             patch('models.func_rf_enhanced.joblib'):
            self.model = AdaptiveRandomForest(model_name="test_model")
    
    def test_predict_without_training(self):
        """Test prediction behavior without training."""
        # Should return default values
        prediction, confidence = self.model.predict({'bullRatio': 0.6, 'bearRatio': 0.4})
        self.assertEqual(prediction, "BULL")
        self.assertAlmostEqual(confidence, 0.51, places=2)
    
    @patch('models.func_rf_enhanced.joblib')
    def test_train_and_predict(self, mock_joblib):
        """Test training and prediction."""
        # Add some training data
        self.model.training_data = [
            {'bullRatio': 0.7, 'bearRatio': 0.3, 'bnb_change': 0.5, 'btc_change': 1.0, 'outcome': 'BULL'},
            {'bullRatio': 0.3, 'bearRatio': 0.7, 'bnb_change': -0.5, 'btc_change': -1.0, 'outcome': 'BEAR'},
        ]
        
        # Add recent predictions with outcomes
        for i in range(20):
            is_bull = i % 2 == 0
            self.model.recent_predictions.append({
                'epoch': i,
                'prediction': 'BULL' if is_bull else 'BEAR',
                'confidence': 0.7,
                'features': {
                    'bullRatio': 0.7 if is_bull else 0.3,
                    'bearRatio': 0.3 if is_bull else 0.7,
                    'bnb_change': 0.5 if is_bull else -0.5,
                    'btc_change': 1.0 if is_bull else -1.0
                },
                'actual': 'BULL' if is_bull else 'BEAR'  # Correct predictions
            })
        
        # Train the model
        result = self.model.train(force=True)
        self.assertTrue(result)
        
        # Test bullish prediction
        bull_pred, bull_conf = self.model.predict({
            'bullRatio': 0.8, 
            'bearRatio': 0.2, 
            'bnb_change': 1.0, 
            'btc_change': 1.5
        })
        self.assertEqual(bull_pred, "BULL")
        self.assertTrue(bull_conf > 0.5)
        
        # Test bearish prediction
        bear_pred, bear_conf = self.model.predict({
            'bullRatio': 0.2, 
            'bearRatio': 0.8, 
            'bnb_change': -1.0, 
            'btc_change': -1.5
        })
        self.assertEqual(bear_pred, "BEAR")
        self.assertTrue(bear_conf > 0.5)
    
    def test_record_outcome(self):
        """Test recording outcomes."""
        # Add a prediction
        self.model.recent_predictions.append({
            'epoch': 123,
            'timestamp': '2023-01-01T12:00:00',
            'prediction': 'BULL',
            'confidence': 0.8,
            'features': {'bullRatio': 0.7, 'bearRatio': 0.3},
            'actual': None
        })
        
        # Record the outcome
        result = self.model.record_outcome(123, 'BEAR')
        self.assertTrue(result)
        
        # Check that the prediction was updated
        pred = next(p for p in self.model.recent_predictions if p['epoch'] == 123)
        self.assertEqual(pred['actual'], 'BEAR')
    
    def test_get_performance_metrics(self):
        """Test performance metrics calculation."""
        # Add some predictions with outcomes
        self.model.recent_predictions = [
            {'epoch': 1, 'prediction': 'BULL', 'actual': 'BULL'},
            {'epoch': 2, 'prediction': 'BULL', 'actual': 'BEAR'},
            {'epoch': 3, 'prediction': 'BEAR', 'actual': 'BEAR'},
            {'epoch': 4, 'prediction': 'BEAR', 'actual': 'BULL'}
        ]
        
        # Get metrics
        metrics = self.model.get_performance_metrics()
        
        # Check metrics
        self.assertEqual(metrics['sample_count'], 4)
        self.assertEqual(metrics['accuracy'], 0.5)  # 2/4 correct
        self.assertEqual(metrics['bull_accuracy'], 0.5)  # 1/2 bull correct
        self.assertEqual(metrics['bear_accuracy'], 0.5)  # 1/2 bear correct


if __name__ == '__main__':
    unittest.main() 