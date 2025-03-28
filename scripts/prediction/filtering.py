"""
Signal filtering for prediction strategies.
Dynamically adjusts and filters prediction signals based on performance and market conditions.
"""

import logging
import traceback

from ..data.database import get_strategy_performance

logger = logging.getLogger(__name__)

def filter_signals(predictions, min_confidence=0.55, market_regime=None):
    """
    Filter prediction signals based on confidence and other criteria.
    
    Args:
        predictions: Dictionary of predictions from various strategies
        min_confidence: Minimum confidence threshold (lowered to 0.55)
        market_regime: Market regime information (string or dict)
        
    Returns:
        list: Filtered high-confidence predictions
    """
    try:
        # Get recent performance for different signal types
        try:
            recent_performance = get_strategy_performance()
            
            # Calculate success rates for each strategy
            strategy_success = {}
            for strategy, data in recent_performance.items():
                if data['total'] > 0:
                    success_rate = data['wins'] / data['total']
                    strategy_success[strategy] = success_rate
        except Exception as e:
            logger.warning(f"Error getting strategy performance: {e}")
            strategy_success = {}
        
        # Get current market conditions - handle both string and dict inputs
        if isinstance(market_regime, dict):
            regime = market_regime.get('regime', 'unknown')
            regime_confidence = market_regime.get('confidence', 0.5)
        else:
            # Handle case where market_regime is a string
            regime = market_regime if market_regime else 'unknown'
            regime_confidence = 0.5  # Default confidence when not provided
        
        # Debug information
        logger.info(f"Filtering signals for {len(predictions)} predictions with min threshold {min_confidence}")
        for name, pred in predictions.items():
            logger.info(f"  - {name}: {pred.get('prediction')} ({pred.get('confidence', 0):.2f})")
        
        # Setup initial quality scores for each prediction
        quality_scores = {}
        for signal_type, pred_data in predictions.items():
            prediction = pred_data.get('prediction')
            confidence = float(pred_data.get('confidence', 0))
            
            # Skip invalid predictions - lowered minimum to 0.52
            if not prediction or prediction == "UNKNOWN" or confidence < 0.52:
                continue
                
            # Base quality is confidence
            quality = confidence
            
            # Simplified quality calculation during initial data collection phase
            # This will allow more signals to pass through for faster data collection
            
            # Only apply slight adjustments based on performance if we have data
            if signal_type in strategy_success and len(strategy_success) > 5:
                performance_multiplier = 1.0
                success_rate = strategy_success[signal_type]
                
                if success_rate >= 0.55:  # Good performance
                    performance_multiplier = 1.15  # Reduced boost
                    logger.info(f"🔥 Boosting {signal_type} due to good performance")
                elif success_rate <= 0.45:  # Poor performance
                    performance_multiplier = 0.9  # Less reduction
                    logger.warning(f"Reducing {signal_type} weight due to poor performance")
                
                quality *= performance_multiplier
                        
            # Store quality score
            quality_scores[signal_type] = quality
            logger.info(f"Signal {signal_type} ({prediction}) received quality score: {quality:.2f}")
        
        # Very permissive threshold during initial data collection
        # Base threshold starts lower (0.55)
        base_threshold = 0.55
        
        # More sophisticated threshold adjustment after we have data
        if len(strategy_success) > 10:
            # Dynamic threshold based on recent overall performance
            overall_success = sum(strategy_success.values()) / len(strategy_success) if strategy_success else 0.5
            
            # Adjust threshold based on performance
            if overall_success < 0.45:
                min_threshold = 0.60  # Slightly stricter when doing poorly
            else:
                min_threshold = 0.55  # Base threshold when doing well
                
            # Slightly higher threshold in volatile markets
            if regime == "volatile":
                min_threshold += 0.05
        else:
            # Very permissive early on to gather data
            min_threshold = base_threshold
            logger.info(f"Using permissive threshold ({min_threshold}) to gather training data")
        
        # Filter out low quality signals
        filtered_predictions = {}
        for signal_type, quality in quality_scores.items():
            if quality >= min_threshold:
                filtered_predictions[signal_type] = predictions[signal_type]
                filtered_predictions[signal_type]['quality'] = quality
                logger.info(f"✅ {signal_type} passed filtering with quality {quality:.2f}")
            else:
                logger.info(f"❌ {signal_type} filtered out with quality {quality:.2f} < {min_threshold}")
        
        # Simplified consensus check - only apply if we have enough data
        if len(filtered_predictions) >= 2 and len(strategy_success) > 10:
            bull_count = sum(1 for k, v in filtered_predictions.items() if v.get('prediction') == 'BULL')
            bear_count = sum(1 for k, v in filtered_predictions.items() if v.get('prediction') == 'BEAR')
            
            # Only apply this with sufficient data and poor performance
            if bull_count > 0 and bear_count > 0 and overall_success < 0.45:
                stricter_predictions = {}
                stricter_threshold = min_threshold + 0.05  # Less strict adjustment
                logger.info(f"🔍 Applying stricter filtering due to mixed signals and poor performance")
                
                for signal_type, pred_data in filtered_predictions.items():
                    if pred_data.get('quality', 0) >= stricter_threshold:
                        stricter_predictions[signal_type] = pred_data
                        logger.info(f"✅ {signal_type} passed stricter filtering")
                    else:
                        logger.info(f"❌ {signal_type} filtered out by stricter threshold")
                
                filtered_predictions = stricter_predictions
        
        # IMPORTANT: Early data collection protection - always let through high confidence signals
        # regardless of other filters, to ensure we collect enough data
        if not filtered_predictions:
            # Find highest confidence prediction if nothing passes filters
            highest_conf = 0
            highest_signal = None
            
            for signal_type, pred_data in predictions.items():
                conf = float(pred_data.get('confidence', 0))
                if conf > highest_conf and conf >= 0.6:  # Must have at least 0.6 confidence
                    highest_conf = conf
                    highest_signal = signal_type
            
            if highest_signal:
                filtered_predictions[highest_signal] = predictions[highest_signal]
                filtered_predictions[highest_signal]['quality'] = highest_conf
                logger.info(f"🔄 No predictions passed filters but allowing {highest_signal} with {highest_conf:.2f} confidence to collect data")
        
        # Log which strategies we're using
        if filtered_predictions:
            used_strategies = list(filtered_predictions.keys())
            logger.info(f"🎯 Used prediction signals: {', '.join(used_strategies)}")
        else:
            logger.warning("No signals passed filtering")
        
        return filtered_predictions
        
    except Exception as e:
        logger.error(f"❌ Error filtering signals: {e}")
        traceback.print_exc()
        return {}  # Return empty dict on error 