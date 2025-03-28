"""
Strategy weights management for the prediction system.
Handles dynamic weight adjustment, regime-based optimization, and performance tracking.
"""

import logging
import traceback
import numpy as np

from ..analysis.regime import detect_market_regime
from ..data.database import get_prediction_history, get_signal_performance
from ..analysis.market import get_historical_prices

logger = logging.getLogger(__name__)

def update_strategy_weights():
    """
    Dynamically update strategy weights based on recent performance, market regime and strategy optimization.
    
    Returns:
        dict: Dictionary of strategy weights optimized for current conditions
    """
    try:
        # Get recent predictions using centralized function
        recent_predictions = get_prediction_history(50)
        
        # Also get signal performance data for adaptive weighting
        signal_performance = get_signal_performance()
        logger.info("Signal Performance:")
        if signal_performance:
            for signal in signal_performance:
                accuracy = signal.get('accuracy', 0)
                logger.info(f"  {signal['signal']}: {accuracy:.2f} accuracy ({signal['correct']}/{signal['total']})")
        else:
            logger.info("  No signal performance data available yet")
        
        # Detect current market regime
        prices = get_historical_prices(lookback=30)
        market_regime = "unknown"
        regime_confidence = 0
        
        if prices and len(prices) > 15:
            regime_data = detect_market_regime(prices)
            market_regime = regime_data['regime']
            regime_confidence = regime_data['confidence']
            logger.info(f"Current Market Regime: {market_regime.upper()} (confidence: {regime_confidence:.2f})")
        
        if not recent_predictions:
            logger.warning("No recent predictions with outcomes found. Using market regime optimized weights.")
            # Optimize weights based on detected market regime
            return get_regime_optimized_weights(market_regime)
        
        # Calculate success rate for each strategy
        strategy_success = {
            'model': {'correct': 0, 'total': 0},
            'trend_following': {'correct': 0, 'total': 0},
            'contrarian': {'correct': 0, 'total': 0},
            'volume_analysis': {'correct': 0, 'total': 0},
            'market_indicators': {'correct': 0, 'total': 0}
        }
        
        # Track bull/bear accuracy separately
        bull_accuracy = {'correct': 0, 'total': 0}
        bear_accuracy = {'correct': 0, 'total': 0}
        
        # Apply time decay - more recent predictions have higher weight
        time_decay_factor = 0.95
        prediction_weights = [time_decay_factor ** i for i in range(len(recent_predictions))]
        prediction_weights = [w / sum(prediction_weights) for w in prediction_weights]
        
        weighted_accuracy = {}
        
        for idx, pred in enumerate(recent_predictions):
            # Extract prediction data from dict
            model_pred = pred.get('model_prediction')
            trend_pred = pred.get('trend_prediction')
            contrarian_pred = pred.get('contrarian_prediction')
            volume_pred = pred.get('volume_prediction')
            market_pred = pred.get('market_prediction')
            actual = pred.get('actual_outcome')
            
            # Skip if actual outcome is None
            if actual is None:
                continue
                
            # Apply time decay weight to this prediction
            weight = prediction_weights[idx]
            
            # Track bull/bear prediction accuracy
            if pred.get('final_prediction') == 'BULL':
                bull_accuracy['total'] += weight
                if actual == 'BULL':
                    bull_accuracy['correct'] += weight
            elif pred.get('final_prediction') == 'BEAR':
                bear_accuracy['total'] += weight
                if actual == 'BEAR':
                    bear_accuracy['correct'] += weight
                
            # Check model prediction
            if model_pred is not None:
                strategy_success['model']['total'] += weight
                if model_pred == actual:
                    strategy_success['model']['correct'] += weight
            
            # Check trend following prediction
            if trend_pred is not None:
                strategy_success['trend_following']['total'] += weight
                if trend_pred == actual:
                    strategy_success['trend_following']['correct'] += weight
            
            # Check contrarian prediction
            if contrarian_pred is not None:
                strategy_success['contrarian']['total'] += weight
                if contrarian_pred == actual:
                    strategy_success['contrarian']['correct'] += weight
            
            # Check volume analysis prediction
            if volume_pred is not None:
                strategy_success['volume_analysis']['total'] += weight
                if volume_pred == actual:
                    strategy_success['volume_analysis']['correct'] += weight
            
            # Check market indicators prediction
            if market_pred is not None:
                strategy_success['market_indicators']['total'] += weight
                if market_pred == actual:
                    strategy_success['market_indicators']['correct'] += weight
        
        # Calculate success rates with time decay applied
        success_rates = {}
        for strategy, data in strategy_success.items():
            if data['total'] > 0:
                success_rates[strategy] = data['correct'] / data['total']
            else:
                # Get regime-optimized default rates
                default_rates = get_strategy_defaults(market_regime)
                success_rates[strategy] = default_rates.get(strategy, 0.5)
        
        # Log success rates
        logger.info("Strategy Performance:")
        for strategy, rate in success_rates.items():
            correct = strategy_success[strategy]['correct']
            total = strategy_success[strategy]['total']
            logger.info(f"  {strategy}: {rate:.2f} ({correct:.2f}/{total:.2f})")
            
        # Log bull/bear accuracy
        if bull_accuracy['total'] > 0:
            bull_acc = bull_accuracy['correct'] / bull_accuracy['total']
            logger.info(f"  BULL predictions: {bull_acc:.2f} ({bull_accuracy['correct']:.2f}/{bull_accuracy['total']:.2f})")
        if bear_accuracy['total'] > 0:
            bear_acc = bear_accuracy['correct'] / bear_accuracy['total']
            logger.info(f"  BEAR predictions: {bear_acc:.2f} ({bear_accuracy['correct']:.2f}/{bear_accuracy['total']:.2f})")
        
        # Calculate weights based on success rates but with minimum thresholds
        total_success = sum(success_rates.values())
        
        if total_success == 0:
            # If all strategies have 0 success rate, use our regime-optimized weights
            weights = get_regime_optimized_weights(market_regime)
        else:
            # Calculate raw weights based on success rates
            raw_weights = {strategy: rate / total_success for strategy, rate in success_rates.items()}
            
            # Apply minimum thresholds based on market regime
            min_weights = get_minimum_weights(market_regime)
            
            # Ensure minimum weights are applied
            adjusted_weights = {}
            remaining_weight = 1.0
            
            # First, apply minimum weights
            for strategy, min_weight in min_weights.items():
                if raw_weights.get(strategy, 0) < min_weight:
                    adjusted_weights[strategy] = min_weight
                    remaining_weight -= min_weight
                else:
                    # Mark for proportional distribution
                    adjusted_weights[strategy] = None
            
            # Calculate sum of raw weights for strategies that exceeded their minimums
            sum_exceeding_raw = sum(raw_weights[s] for s in adjusted_weights if adjusted_weights[s] is None)
            
            # Distribute remaining weight proportionally
            if sum_exceeding_raw > 0:
                for strategy in adjusted_weights:
                    if adjusted_weights[strategy] is None:
                        adjusted_weights[strategy] = (raw_weights[strategy] / sum_exceeding_raw) * remaining_weight
            
            # Normalize to ensure weights sum to 1
            total_weight = sum(adjusted_weights.values())
            weights = {strategy: weight / total_weight for strategy, weight in adjusted_weights.items()}
            
            # Apply final market regime adjustment
            weights = apply_regime_boost(weights, market_regime, regime_confidence)
        
        # Log the final weights
        logger.info("Final strategy weights:")
        for strategy, weight in weights.items():
            logger.info(f"  {strategy}: {weight:.2f}")
            
        return weights
        
    except Exception as e:
        logger.error(f"Error updating strategy weights: {e}")
        traceback.print_exc()
        # Return default weights optimized for current market conditions
        return get_regime_optimized_weights("unknown")

def get_regime_optimized_weights(regime):
    """
    Get optimal strategy weights for each market regime.
    
    Args:
        regime: Current market regime ('uptrend', 'downtrend', 'ranging', 'volatile', 'unknown')
        
    Returns:
        dict: Regime-optimized weights for each strategy
    """
    # Default weights for unknown regime
    default_weights = {
        'model': 0.10,
        'trend_following': 0.20,
        'contrarian': 0.15,
        'volume_analysis': 0.20,
        'market_indicators': 0.35
    }
    
    # Regime-specific optimizations
    if regime == "uptrend":
        return {
            'model': 0.10,
            'trend_following': 0.30,  # Higher in uptrends
            'contrarian': 0.10,       # Lower in uptrends
            'volume_analysis': 0.20,
            'market_indicators': 0.30
        }
    elif regime == "downtrend":
        return {
            'model': 0.10,
            'trend_following': 0.25,  # Still good in downtrends
            'contrarian': 0.15,       # Slightly higher in downtrends
            'volume_analysis': 0.20,
            'market_indicators': 0.30
        }
    elif regime == "ranging":
        return {
            'model': 0.10,
            'trend_following': 0.15,  # Lower in ranging markets
            'contrarian': 0.25,       # Higher in ranging markets
            'volume_analysis': 0.20,
            'market_indicators': 0.30
        }
    elif regime == "volatile":
        return {
            'model': 0.05,            # Less reliable in volatility
            'trend_following': 0.15,  # Lower in volatile markets
            'contrarian': 0.20,       # Moderate in volatile markets
            'volume_analysis': 0.30,  # Higher in volatile markets (volume is key)
            'market_indicators': 0.30
        }
    
    return default_weights

def get_strategy_defaults(regime):
    """
    Get default success rates for different market regimes.
    
    Args:
        regime: Current market regime
        
    Returns:
        dict: Default success rates for each strategy in the given regime
    """
    if regime == "uptrend":
        return {
            'model': 0.55,
            'trend_following': 0.70,  # Trend following works well in uptrends
            'contrarian': 0.45,       # Contrarian less effective in uptrends
            'volume_analysis': 0.60,
            'market_indicators': 0.65
        }
    elif regime == "downtrend":
        return {
            'model': 0.55,
            'trend_following': 0.65,  # Trend following works in downtrends too
            'contrarian': 0.50,       # Contrarian moderate in downtrends
            'volume_analysis': 0.60,
            'market_indicators': 0.60
        }
    elif regime == "ranging":
        return {
            'model': 0.50,
            'trend_following': 0.45,  # Trend following poor in ranges
            'contrarian': 0.70,       # Contrarian works well in ranges
            'volume_analysis': 0.55,
            'market_indicators': 0.60
        }
    elif regime == "volatile":
        return {
            'model': 0.45,            # Models struggle with volatility
            'trend_following': 0.50,  # Trend following moderate in volatility
            'contrarian': 0.55,       # Contrarian slightly better in volatility
            'volume_analysis': 0.65,  # Volume analysis better in volatility
            'market_indicators': 0.55
        }
    
    # Default/unknown
    return {
        'model': 0.50,
        'trend_following': 0.55,
        'contrarian': 0.55,
        'volume_analysis': 0.55,
        'market_indicators': 0.60
    }

def get_minimum_weights(regime):
    """
    Get minimum weights based on market regime.
    
    Args:
        regime: Current market regime
        
    Returns:
        dict: Minimum weights for each strategy in the given regime
    """
    if regime == "uptrend":
        return {
            'market_indicators': 0.25,
            'trend_following': 0.20,    # Higher minimum in uptrends
            'contrarian': 0.10,         # Lower minimum in uptrends
            'volume_analysis': 0.15,
            'model': 0.05
        }
    elif regime == "downtrend":
        return {
            'market_indicators': 0.25,
            'trend_following': 0.20,    # Still higher minimum in downtrends
            'contrarian': 0.15,         # Moderate minimum in downtrends
            'volume_analysis': 0.15,
            'model': 0.05
        }
    elif regime == "ranging":
        return {
            'market_indicators': 0.20,
            'trend_following': 0.10,    # Lower minimum in ranges
            'contrarian': 0.25,         # Higher minimum in ranges
            'volume_analysis': 0.15,
            'model': 0.05
        }
    elif regime == "volatile":
        return {
            'market_indicators': 0.20,
            'trend_following': 0.10,    # Lower minimum in volatility
            'contrarian': 0.15,         # Moderate minimum in volatility
            'volume_analysis': 0.25,    # Higher minimum for volume in volatility
            'model': 0.05
        }
    
    # Default/unknown
    return {
        'market_indicators': 0.25,
        'trend_following': 0.15,
        'contrarian': 0.10,
        'volume_analysis': 0.15,
        'model': 0.05
    }

def apply_regime_boost(weights, regime, confidence):
    """
    Apply a final boost based on market regime and confidence.
    
    Args:
        weights: Current strategy weights
        regime: Current market regime
        confidence: Confidence in the regime detection (0-1)
        
    Returns:
        dict: Adjusted weights with regime-specific boosts applied
    """
    if confidence < 0.5:
        return weights  # Don't apply boost if regime confidence is low
        
    boost_factor = confidence * 0.2  # Max 20% boost
    
    result = weights.copy()
    
    if regime == "uptrend":
        # Boost trend following in uptrends
        if 'trend_following' in result:
            result['trend_following'] = min(result['trend_following'] * (1 + boost_factor), 0.40)
    elif regime == "downtrend":
        # Moderate boost to trend following and market indicators in downtrends
        if 'trend_following' in result:
            result['trend_following'] = min(result['trend_following'] * (1 + boost_factor*0.5), 0.35)
        if 'market_indicators' in result:
            result['market_indicators'] = min(result['market_indicators'] * (1 + boost_factor*0.5), 0.40)
    elif regime == "ranging":
        # Boost contrarian in ranging markets
        if 'contrarian' in result:
            result['contrarian'] = min(result['contrarian'] * (1 + boost_factor), 0.40)
    elif regime == "volatile":
        # Boost volume analysis in volatile markets
        if 'volume_analysis' in result:
            result['volume_analysis'] = min(result['volume_analysis'] * (1 + boost_factor), 0.40)
    
    # Normalize weights to ensure they sum to 1
    total = sum(result.values())
    return {k: v/total for k, v in result.items()} 