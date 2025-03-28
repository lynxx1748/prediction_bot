"""
Prediction strategies for the trading bot.
Each strategy implements a different prediction approach.
"""

import logging
import traceback
import numpy as np
from abc import ABC, abstractmethod
from ..utils.helpers import get_price_trend
from ..analysis.market import get_market_sentiment, get_market_direction

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """Base class for all prediction strategies."""
    
    def __init__(self, name=None, config=None):
        """
        Initialize strategy.
        
        Args:
            name: Strategy name
            config: Optional configuration
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}
        self.historical_performance = {
            'predictions': 0,
            'correct': 0,
            'accuracy': 0.5
        }
        
    @abstractmethod
    def predict(self, round_data):
        """
        Make a prediction for the current round.
        
        Args:
            round_data: Dictionary with current round data
            
        Returns:
            tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
        """
        pass
    
    def record_outcome(self, prediction, actual):
        """
        Record prediction outcome for performance tracking.
        
        Args:
            prediction: Predicted outcome ("BULL" or "BEAR")
            actual: Actual outcome ("BULL" or "BEAR")
        """
        if prediction and actual:
            self.historical_performance['predictions'] += 1
            if prediction.upper() == actual.upper():
                self.historical_performance['correct'] += 1
                
            # Update accuracy
            if self.historical_performance['predictions'] > 0:
                self.historical_performance['accuracy'] = (
                    self.historical_performance['correct'] / 
                    self.historical_performance['predictions']
                )

    def log_error(self, message, exception=None):
        """
        Log error with detailed traceback information.
        
        Args:
            message: Error message
            exception: Exception object
        """
        logger.error(f"❌ {message}")
        if exception:
            logger.error(f"Exception: {str(exception)}")
            traceback.print_exc()  # Use the traceback module
        return None

    def analyze_performance_stats(self):
        """
        Calculate advanced performance statistics using numpy.
        
        Returns:
            dict: Performance statistics
        """
        try:
            if not hasattr(self, 'prediction_history'):
                self.prediction_history = []
            
            if len(self.prediction_history) < 3:
                return {
                    'confidence_mean': 0,
                    'confidence_std': 0,
                    'consistency': 0
                }
            
            # Convert confidence values to numpy array
            confidence_values = np.array([p.get('confidence', 0) for p in self.prediction_history])
            correct_predictions = np.array([1 if p.get('correct', False) else 0 for p in self.prediction_history])
            
            # Calculate statistics using numpy
            stats = {
                'confidence_mean': float(np.mean(confidence_values)),
                'confidence_std': float(np.std(confidence_values)),
                'consistency': float(np.mean(correct_predictions[-10:]) if len(correct_predictions) >= 10 else np.mean(correct_predictions)),
                'recent_accuracy': float(np.sum(correct_predictions[-20:]) / min(20, len(correct_predictions)) if len(correct_predictions) > 0 else 0)
            }
            
            return stats
        
        except Exception as e:
            self.log_error("Error analyzing performance stats", e)
            return {}

class TrendFollowingStrategy(Strategy):
    """Strategy that follows price trends."""
    
    def predict(self, round_data):
        """Predict based on price trends."""
        try:
            trend, strength = get_price_trend()
            
            if trend == "up" and strength > 0.3:
                return "BULL", strength
            elif trend == "down" and strength > 0.3:
                return "BEAR", strength
            else:
                return None, 0  # No strong trend detected
                
        except Exception as e:
            return self.log_error("Error in trend following strategy", e)

class ContrarianStrategy(Strategy):
    """Strategy that goes against the market sentiment."""
    
    def predict(self, round_data):
        """Predict opposite of market sentiment when sentiment is extreme."""
        try:
            sentiment, strength = get_market_sentiment(round_data)
            
            # Contrarian logic: bet against the crowd when sentiment is very strong
            if sentiment == "bullish" and strength > 0.75:
                contrarian_strength = min(1.0, (strength - 0.75) * 2 + 0.5)
                return "BEAR", contrarian_strength
            elif sentiment == "bearish" and strength > 0.75:
                contrarian_strength = min(1.0, (strength - 0.75) * 2 + 0.5)
                return "BULL", contrarian_strength
            else:
                return None, 0  # No contrarian signal
                
        except Exception as e:
            return self.log_error("Error in contrarian strategy", e)

class VolumeBasedStrategy(Strategy):
    """Strategy based on volume analysis."""
    
    def predict(self, round_data):
        """Predict based on volume patterns."""
        try:
            # Extract volume data
            bull_amount = float(round_data.get('bullAmount', 0))
            bear_amount = float(round_data.get('bearAmount', 0))
            total_amount = bull_amount + bear_amount
            
            # Skip if volume is too low
            if total_amount < 0.1:
                return None, 0
                
            # Calculate volume ratios
            bull_ratio = bull_amount / total_amount if total_amount > 0 else 0.5
            bear_ratio = bear_amount / total_amount if total_amount > 0 else 0.5
            
            # Simple volume-based decision with contrarian approach
            if bull_ratio > 0.65:  # Strong bull bias
                return "BEAR", min(bull_ratio, 0.9)  # Contrarian approach
            elif bear_ratio > 0.65:  # Strong bear bias
                return "BULL", min(bear_ratio, 0.9)  # Contrarian approach
            
            return None, 0
            
        except Exception as e:
            return self.log_error("Error in volume-based strategy", e)

class SwingTradingStrategy(Strategy):
    """Strategy based on price swing detection."""
    
    def predict(self, round_data):
        """Predict based on price swing patterns."""
        try:
            from ..data.database import get_recent_price_changes
            from ..analysis.swing import detect_price_swing, optimize_swing_trading
            
            # Get recent price change data
            price_changes = get_recent_price_changes(10)
            
            # First check for optimal swing trades
            swing_opportunity = optimize_swing_trading(price_changes)
            
            if swing_opportunity.get('swing_opportunity', False):
                return swing_opportunity['direction'], swing_opportunity['confidence']
            
            # Otherwise use general swing detection
            prediction, confidence = detect_price_swing(price_changes)
            
            # Only return if we have a decent confidence
            if prediction != "UNKNOWN" and confidence >= 0.6:
                return prediction, confidence
                
            # No valid signal
            return None, 0
            
        except Exception as e:
            return self.log_error("Error in swing trading strategy", e)

class MarketDirectionStrategy(Strategy):
    """Strategy based on market direction signals."""
    
    def predict(self, round_data):
        """Predict based on market direction indicators."""
        try:
            # Use the imported get_market_direction function
            direction, strength, signals = get_market_direction(lookback=12)
            
            if direction == "bullish" and strength > 0.6:
                return "BULL", strength
            elif direction == "bearish" and strength > 0.6:
                return "BEAR", strength
            
            # Check for signal agreement
            if signals and len(signals) >= 3:
                bull_signals = sum(1 for s in signals if s == "bullish")
                bear_signals = sum(1 for s in signals if s == "bearish")
                
                # Strong agreement in signals
                if bull_signals >= 3 and bull_signals > bear_signals * 2:
                    return "BULL", 0.65
                elif bear_signals >= 3 and bear_signals > bull_signals * 2:
                    return "BEAR", 0.65
            
            # No clear signal
            return None, 0
            
        except Exception as e:
            return self.log_error("Error in market direction strategy", e)

# Factory function to get strategy based on name
def get_strategy(strategy_name, config=None):
    """
    Get an instance of the requested prediction strategy.
    
    Args:
        strategy_name: Name of the strategy
        config: Optional configuration
        
    Returns:
        Strategy: Instance of the strategy
    """
    strategies = {
        'trend_following': TrendFollowingStrategy,
        'contrarian': ContrarianStrategy,
        'volume': VolumeBasedStrategy,
        'swing': SwingTradingStrategy,
        'market_direction': MarketDirectionStrategy,
        # Add more strategies here
    }
    
    strategy_class = strategies.get(strategy_name)
    if strategy_class:
        return strategy_class(strategy_name, config)
    else:
        logger.warning(f"⚠️ Unknown strategy: {strategy_name}, using TrendFollowingStrategy")
        return TrendFollowingStrategy(strategy_name, config)

def get_optimal_strategy(market_regime, historical_performance, round_data):
    """
    Get optimal strategy instance based on market conditions.
    
    Args:
        market_regime: Current market regime information
        historical_performance: Historical performance metrics
        round_data: Current round data
        
    Returns:
        Strategy: Instantiated strategy object
    """
    from .strategy_selector import select_optimal_strategy
    
    # Get optimal strategy selection
    strategy_selection = select_optimal_strategy(market_regime, historical_performance, round_data)
    
    # Get primary strategy
    primary_name = strategy_selection.get('primary', 'trend_following')
    primary_strategy = get_strategy(primary_name)
    
    # Return the instantiated strategy
    return primary_strategy 