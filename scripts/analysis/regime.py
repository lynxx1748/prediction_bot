"""
Market regime detection for the trading bot.
Identifies current market conditions to adapt trading strategy.
"""

import numpy as np
from scipy import stats
import traceback
import logging
from .market import get_historical_prices

logger = logging.getLogger(__name__)

def detect_market_regime(prices=None):
    """
    Detect current market regime based on price data.
    
    Args:
        prices: List of historical prices (fetches if None)
        
    Returns:
        dict: Market regime information
    """
    try:
        if prices is None or len(prices) < 8:
            prices = get_historical_prices(30)
            
        if not prices or len(prices) < 8:
            return {"regime": "unknown", "confidence": 0}
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility (annualized)
        volatility = np.std(returns) * np.sqrt(365)
        
        # Calculate momentum
        lookback = min(len(prices) - 1, 10)
        momentum = (prices[-1] / prices[-lookback] - 1) * 100
        
        # Calculate efficiency ratio (trend strength)
        price_range = np.max(prices) - np.min(prices)
        path_length = np.sum(np.abs(np.diff(prices)))
        efficiency = price_range / path_length if path_length > 0 else 0
        
        # Determine regime
        if volatility > 0.4:  # High volatility
            regime = "volatile"
            confidence = min(volatility / 0.6, 0.95)
        elif efficiency > 0.6:  # Strong trend
            if momentum > 0:
                regime = "uptrend"
            else:
                regime = "downtrend"
            confidence = min(efficiency, 0.95)
        else:  # Range-bound
            regime = "ranging"
            confidence = min((1 - efficiency) * 0.8, 0.95)
            
        logger.info(f"🔍 Market Regime Detection: {regime} (conf: {confidence:.2f})")
        return {
            "regime": regime,
            "confidence": confidence,
            "volatility": volatility,
            "momentum": momentum,
            "efficiency": efficiency
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting market regime: {e}")
        traceback.print_exc()
        return {"regime": "unknown", "confidence": 0}

def detect_market_regime_advanced(prices, volumes=None):
    """
    Enhanced market regime detection with volume confirmation.
    
    Args:
        prices: List of price data
        volumes: Optional list of volume data
        
    Returns:
        dict: Detailed market regime information
    """
    try:
        if not prices or len(prices) < 10:
            return {"regime": "unknown", "confidence": 0}
            
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Volatility measurement
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Trend detection using linear regression
        x = np.arange(len(prices))
        slope, _, r_value, _, _ = stats.linregress(x, prices)
        
        # Normalize r-squared to get trend strength
        trend_strength = r_value ** 2
        
        # Volume confirmation if available
        volume_confirmation = 0.5
        if volumes is not None and len(volumes) == len(prices):
            # Volume trend calculation
            vol_slope, _, vol_r_value, _, _ = stats.linregress(x, volumes)
            volume_trend = vol_r_value ** 2
            
            # Check if volume confirms price movement
            volume_confirms = (np.sign(slope) == np.sign(vol_slope))
            
            # Use volume_trend to adjust confirmation strength
            if volume_confirms:
                # Stronger volume trend gives stronger confirmation
                volume_confirmation = 0.6 + (volume_trend * 0.4)  # Scale between 0.6-1.0
            else:
                # Strong disagreeing volume trend reduces confidence more
                volume_confirmation = 0.4 - (volume_trend * 0.2)  # Scale between 0.2-0.4
        
        # Regime classification with confidence
        if volatility > 0.25:  # High volatility threshold
            regime = "volatile"
            confidence = min(0.5 + volatility, 0.95)
        elif trend_strength > 0.6:  # Strong trend
            regime = "uptrend" if slope > 0 else "downtrend"
            confidence = min(trend_strength * volume_confirmation, 0.95)
        else:  # Ranging market
            regime = "ranging"
            confidence = min(0.7 - trend_strength, 0.8)
        
        # Return detailed information
        return {
            "regime": regime,
            "confidence": confidence,
            "volatility": volatility,
            "trend_strength": trend_strength,
            "slope": slope,
            "volume_confirmation": volume_confirmation
        }
    
    except Exception as e:
        logger.error(f"❌ Error in advanced market regime detection: {e}")
        traceback.print_exc()
        return {"regime": "unknown", "confidence": 0}

def get_regime_based_strategy(regime_data):
    """
    Get recommended trading strategy based on market regime.
    
    Args:
        regime_data: Market regime data from detect_market_regime
        
    Returns:
        dict: Strategy recommendations
    """
    try:
        regime = regime_data.get('regime', 'unknown')
        confidence = regime_data.get('confidence', 0)
        
        # Default strategy weights
        strategy_weights = {
            'trend_following': 0.25,
            'mean_reversion': 0.25,
            'momentum': 0.25,
            'volatility_breakout': 0.25
        }
        
        # Adjust weights based on regime
        if regime == 'uptrend' or regime == 'downtrend':
            # In trending markets, favor trend following and momentum
            strategy_weights['trend_following'] = 0.4
            strategy_weights['momentum'] = 0.3
            strategy_weights['mean_reversion'] = 0.1
            strategy_weights['volatility_breakout'] = 0.2
            
        elif regime == 'ranging':
            # In ranging markets, favor mean reversion
            strategy_weights['mean_reversion'] = 0.5
            strategy_weights['trend_following'] = 0.1
            strategy_weights['momentum'] = 0.2
            strategy_weights['volatility_breakout'] = 0.2
            
        elif regime == 'volatile':
            # In volatile markets, favor breakout strategies
            strategy_weights['volatility_breakout'] = 0.4
            strategy_weights['trend_following'] = 0.2
            strategy_weights['momentum'] = 0.3
            strategy_weights['mean_reversion'] = 0.1
            
        # Scale by confidence
        if confidence > 0:
            for key in strategy_weights:
                # Strengthen the dominant strategy based on confidence
                if strategy_weights[key] == max(strategy_weights.values()):
                    strategy_weights[key] = strategy_weights[key] + (1 - strategy_weights[key]) * confidence * 0.5
                
            # Normalize to ensure weights sum to 1
            total = sum(strategy_weights.values())
            for key in strategy_weights:
                strategy_weights[key] /= total
                
        return {
            'regime': regime,
            'confidence': confidence,
            'strategy_weights': strategy_weights,
            'description': f"Market regime is {regime} with {confidence:.2f} confidence"
        }
        
    except Exception as e:
        logger.error(f"❌ Error determining regime-based strategy: {e}")
        return {
            'regime': 'unknown',
            'confidence': 0,
            'strategy_weights': {
                'trend_following': 0.25,
                'mean_reversion': 0.25,
                'momentum': 0.25,
                'volatility_breakout': 0.25
            },
            'description': "Using default strategy weights due to unknown regime"
        } 