import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from models.func_ta import TechnicalAnalysis, get_technical_indicators


class TestTechnicalAnalysis(unittest.TestCase):
    """Test cases for the technical analysis module."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data
        self.price_data = np.array([100, 102, 98, 103, 99, 102, 105, 107, 106, 110, 
                                   112, 111, 113, 114, 116, 115, 117, 120, 119, 121])
        self.ta = TechnicalAnalysis()
    
    def test_rsi(self):
        """Test RSI calculation."""
        rsi = TechnicalAnalysis.rsi(self.price_data)
        # RSI should be between 0 and 100
        self.assertTrue(0 <= rsi <= 100)
        # RSI should be high for our sample with mostly rising prices
        self.assertTrue(rsi > 60)
        
        # Test with smaller period
        rsi_short = TechnicalAnalysis.rsi(self.price_data, period=5)
        self.assertTrue(0 <= rsi_short <= 100)
    
    def test_macd(self):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = TechnicalAnalysis.macd(self.price_data)
        
        # Test if values are calculated
        self.assertIsNotNone(macd_line)
        self.assertIsNotNone(signal_line)
        self.assertIsNotNone(histogram)
        
        # For our sample with uptrend, MACD should be positive
        self.assertTrue(macd_line > 0)
    
    def test_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        upper, middle, lower = TechnicalAnalysis.bollinger_bands(self.price_data)
        
        # Upper should be greater than middle, which should be greater than lower
        self.assertTrue(upper > middle > lower)
        
        # Middle band should be close to the mean of recent prices
        self.assertAlmostEqual(middle, np.mean(self.price_data[-20:]), delta=0.01)
        
        # Upper - lower should be positive and proportional to volatility
        self.assertTrue((upper - lower) > 0)
    
    def test_ema(self):
        """Test EMA calculation."""
        ema_value = TechnicalAnalysis.ema(self.price_data, period=5)
        
        # EMA should be calculable
        self.assertIsNotNone(ema_value)
        
        # EMA should be related to recent prices (close to them)
        self.assertTrue(min(self.price_data[-5:]) <= ema_value <= max(self.price_data[-5:]))
    
    def test_sma(self):
        """Test SMA calculation."""
        sma_value = TechnicalAnalysis.sma(self.price_data, period=5)
        
        # SMA should equal the mean of the last 5 prices
        expected_sma = np.mean(self.price_data[-5:])
        self.assertEqual(sma_value, expected_sma)
    
    @patch('requests.get')
    def test_get_current_price(self, mock_get):
        """Test get_current_price with mocked requests."""
        # Mock the response from Binance API
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'price': '250.75'}
        mock_get.return_value = mock_response
        
        price = TechnicalAnalysis.get_current_price()
        self.assertEqual(price, 250.75)
        
        # Test error handling
        mock_get.side_effect = Exception('API error')
        price = TechnicalAnalysis.get_current_price()
        self.assertIsNone(price)


class TestTechnicalIndicators(unittest.TestCase):
    """Test the get_technical_indicators function."""
    
    def setUp(self):
        """Set up test data."""
        # Create sample price data with uptrend
        self.uptrend_data = np.array([100, 102, 104, 103, 106, 108, 110, 109, 112, 115])
        # Create sample price data with downtrend
        self.downtrend_data = np.array([120, 118, 119, 115, 113, 110, 109, 107, 108, 105])
        
    @patch('models.func_ta.config')
    def test_get_technical_indicators_uptrend(self, mock_config):
        """Test get_technical_indicators with uptrend data."""
        # Mock configuration
        mock_config.get.return_value = {
            'enable': True,
            'lookback_periods': {'short': 3, 'medium': 5, 'long': 8},
            'indicators': {'rsi': True, 'macd': True, 'bollinger_bands': True, 'ema': True}
        }
        
        prediction, confidence = get_technical_indicators(self.uptrend_data)
        
        # In an uptrend, we should get a bullish prediction with decent confidence
        self.assertEqual(prediction, "BULL")
        self.assertTrue(0.5 < confidence < 1.0)
    
    @patch('models.func_ta.config')
    def test_get_technical_indicators_downtrend(self, mock_config):
        """Test get_technical_indicators with downtrend data."""
        # Mock configuration 
        mock_config.get.return_value = {
            'enable': True,
            'lookback_periods': {'short': 3, 'medium': 5, 'long': 8},
            'indicators': {'rsi': True, 'macd': True, 'bollinger_bands': True, 'ema': True}
        }
        
        prediction, confidence = get_technical_indicators(self.downtrend_data)
        
        # In a downtrend, we should get a bearish prediction with decent confidence
        self.assertEqual(prediction, "BEAR")
        self.assertTrue(0.5 < confidence < 1.0)
    
    @patch('models.func_ta.config')
    def test_technical_analysis_disabled(self, mock_config):
        """Test behavior when technical analysis is disabled."""
        # Mock configuration with technical analysis disabled
        mock_config.get.return_value = {'enable': False}
        
        prediction, confidence = get_technical_indicators(self.uptrend_data)
        
        # Should return default values when disabled
        self.assertEqual(prediction, "BULL")
        self.assertAlmostEqual(confidence, 0.51, places=2)


if __name__ == '__main__':
    unittest.main() 