import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.func_rf import train_model


class TestRandomForestModel(unittest.TestCase):
    """Test cases for the Random Forest model."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample DataFrame that mimics what we'd get from the database
        self.sample_data = pd.DataFrame({
            'epoch': range(1, 101),
            'bullRatio': np.random.uniform(0.3, 0.7, 100),
            'bearRatio': np.random.uniform(0.3, 0.7, 100),
            'bnb_change': np.random.uniform(-3, 3, 100),
            'btc_change': np.random.uniform(-2, 2, 100),
            'lockPrice': np.random.uniform(250, 350, 100),
            'closePrice': np.random.uniform(250, 350, 100)
        })
        
        # Make closePrice > lockPrice for about 60% of rows to simulate market bias
        for i in range(60):
            self.sample_data.loc[i, 'closePrice'] = self.sample_data.loc[i, 'lockPrice'] * 1.01
    
    @patch('sqlite3.connect')
    @patch('joblib.dump')
    def test_train_model(self, mock_dump, mock_connect):
        """Test model training with mocked database."""
        # Set up the mock connection and cursor
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Make read_sql return our sample DataFrame
        mock_conn.execute = MagicMock()
        pd.read_sql = MagicMock(return_value=self.sample_data)
        
        # Call the function
        model, scaler = train_model(
            db_file='test.db', 
            trades_table='test_table',
            model_file='test_model.pkl',
            scaler_file='test_scaler.pkl'
        )
        
        # Check that the function returned a model
        self.assertIsNotNone(model)
        
        # Check that joblib.dump was called to save the model
        mock_dump.assert_called_once()
    
    @patch('sqlite3.connect')
    def test_train_model_empty_data(self, mock_connect):
        """Test model training behavior with empty dataset."""
        # Set up the mock to return empty DataFrame
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        pd.read_sql = MagicMock(return_value=pd.DataFrame())
        
        # Call the function
        model, scaler = train_model(
            db_file='test.db', 
            trades_table='test_table',
            model_file='test_model.pkl',
            scaler_file='test_scaler.pkl'
        )
        
        # Should return None for both model and scaler
        self.assertIsNone(model)
        self.assertIsNone(scaler)
    
    @patch('sqlite3.connect')
    def test_train_model_exception(self, mock_connect):
        """Test model training behavior when an exception occurs."""
        # Make the connection raise an exception
        mock_connect.side_effect = Exception("Database error")
        
        # Call the function
        model, scaler = train_model(
            db_file='test.db', 
            trades_table='test_table',
            model_file='test_model.pkl',
            scaler_file='test_scaler.pkl'
        )
        
        # Should return None for both model and scaler
        self.assertIsNone(model)
        self.assertIsNone(scaler)


if __name__ == '__main__':
    unittest.main() 