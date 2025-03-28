"""
Analysis functions for prediction evaluation and strategy adjustment.
"""

import traceback
import logging

from ..core.constants import config
from ..data.database import get_prediction_accuracy, get_recent_predictions
from ..utils.helpers import get_price_trend

logger = logging.getLogger(__name__)

def analyze_prediction_accuracy(config):
    """
    Analyze recent prediction accuracy to determine if contrarian mode should be activated.
    Uses external data sources when blockchain data is unavailable.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        bool: True if contrarian mode should be activated, False otherwise
    """
    try:
        # Check if contrarian mode is enabled in config
        contrarian_config = config.get("contrarian_mode", {})
        if not contrarian_config.get("enable", True):
            logger.info("📝 Contrarian mode is disabled in config")
            return False
            
        # Get thresholds from config
        accuracy_threshold = contrarian_config.get("accuracy_threshold", 0.35)
        consecutive_wrong_threshold = contrarian_config.get("consecutive_wrong_threshold", 5)
        min_samples = contrarian_config.get("minimum_samples", 5)
        
        # Get recent predictions for detailed analysis
        recent_predictions = get_recent_predictions(10)
        
        # Print recent prediction details
        if recent_predictions:
            logger.info("\n📊 Recent Prediction Analysis:")
            for pred in recent_predictions[:3]:
                outcome = "✅ Correct" if pred.get('final_prediction') == pred.get('actual_outcome') else "❌ Wrong"
                logger.info(f"  Round {pred.get('epoch')}: {pred.get('final_prediction')} vs {pred.get('actual_outcome')} - {outcome}")
        
        # Get accuracy stats from database handler
        accuracy, consecutive_wrong = get_prediction_accuracy(min_samples)
        
        # If we don't have enough data, try using price trend as a proxy
        if accuracy == 0.5 and consecutive_wrong == 0:
            logger.info(f"📝 Not enough samples for contrarian mode analysis")
            
            # Use price trend as alternative data source
            try:
                # Get current trend
                trend, strength = get_price_trend(lookback=8)
                
                # If we have a strong trend, consider going contrarian
                if trend and strength > 0.7:
                    logger.info(f"📈 Strong {trend.upper()} trend detected (strength: {strength:.2f})")
                    logger.info(f"⚠️ Using trend data for contrarian decision - CONTRARIAN MODE ACTIVATED")
                    return True
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to get trend data: {e}")
            
            # Not enough data and no strong trend
            logger.info("📝 Not enough data for contrarian analysis - using regular mode")
            return False
        
        # Print accuracy for debugging
        logger.info(f"📊 Recent prediction accuracy: {accuracy:.2f}")
        logger.info(f"📊 Consecutive wrong predictions: {consecutive_wrong}")
        
        # Determine if contrarian mode should be activated
        if accuracy < accuracy_threshold:
            logger.warning(f"⚠️ Low prediction accuracy ({accuracy:.2f}) - CONTRARIAN MODE ACTIVATED")
            return True
        elif consecutive_wrong >= consecutive_wrong_threshold:
            logger.warning(f"⚠️ {consecutive_wrong} consecutive wrong predictions - CONTRARIAN MODE ACTIVATED")
            return True
            
        logger.info("📝 Regular prediction mode active")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error analyzing prediction accuracy: {e}")
        traceback.print_exc()
        return False

def analyze_volume(bull_amount, bear_amount):
    """
    Analyze volume data to make a prediction.
    
    Args:
        bull_amount: Amount bet on bull
        bear_amount: Amount bet on bear
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Calculate total volume
        total_amount = bull_amount + bear_amount
        
        # Skip if no volume
        if total_amount == 0:
            return None, 0
        
        # Calculate volume ratios
        bull_ratio = bull_amount / total_amount
        bear_ratio = bear_amount / total_amount
        
        # Calculate confidence based on imbalance
        imbalance = abs(bull_ratio - bear_ratio)
        confidence = min(imbalance * 2, 0.95)  # Scale imbalance to confidence
        
        # Simple volume-based prediction
        prediction = "BULL" if bull_amount > bear_amount else "BEAR"
        
        # Log the analysis
        logger.info(f"📊 Volume Analysis: BULL {bull_ratio:.2%} / BEAR {bear_ratio:.2%}")
        logger.info(f"📊 Volume-based prediction: {prediction} with {confidence:.2f} confidence")
        
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"❌ Error analyzing volume: {e}")
        traceback.print_exc()
        return None, 0

def analyze_sentiment(market_data):
    """
    Analyze market sentiment data to make a prediction.
    
    Args:
        market_data: Dictionary containing market sentiment indicators
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Check if we have fear and greed data
        if 'fear_greed_value' not in market_data:
            return None, 0
            
        fear_greed = int(market_data.get('fear_greed_value', 50))
        
        # Analyze fear and greed
        # Extreme fear often signals buying opportunity (contrarian)
        # Extreme greed often signals selling opportunity (contrarian)
        if fear_greed <= 20:  # Extreme fear
            prediction = "BULL"  # Contrarian approach - buy when others are fearful
            confidence = 0.5 + ((20 - fear_greed) / 40)  # Higher confidence with more extreme fear
        elif fear_greed >= 80:  # Extreme greed
            prediction = "BEAR"  # Contrarian approach - sell when others are greedy
            confidence = 0.5 + ((fear_greed - 80) / 40)  # Higher confidence with more extreme greed
        else:
            # In neutral territory, use a sliding scale
            if fear_greed < 45:  # Slight fear
                prediction = "BULL"
                confidence = 0.5 + ((45 - fear_greed) / 50)
            elif fear_greed > 55:  # Slight greed
                prediction = "BEAR"
                confidence = 0.5 + ((fear_greed - 55) / 50)
            else:
                # Very neutral - no strong signal
                prediction = "BULL" if fear_greed >= 50 else "BEAR"
                confidence = 0.51  # Very low confidence
        
        logger.info(f"💭 Sentiment Analysis: Fear & Greed Index = {fear_greed}")
        logger.info(f"💭 Sentiment-based prediction: {prediction} with {confidence:.2f} confidence")
        
        return prediction, confidence
        
    except Exception as e:
        logger.error(f"❌ Error analyzing sentiment: {e}")
        traceback.print_exc()
        return None, 0
    
def get_consolidated_analysis(round_data, market_data):
    """
    Perform a consolidated analysis using multiple methods.
    
    Args:
        round_data: Dictionary with round data
        market_data: Dictionary with market data
        
    Returns:
        dict: Analysis results
    """
    results = {
        'volume': None,
        'sentiment': None,
        'technical': None,
        'overall': None,
        'confidence': 0,
        'signals_agreement': False
    }
    
    try:
        # Volume analysis
        volume_pred, volume_conf = analyze_volume(
            round_data.get('bullAmount', 0), 
            round_data.get('bearAmount', 0)
        )
        results['volume'] = {
            'prediction': volume_pred,
            'confidence': volume_conf
        }
        
        # Sentiment analysis
        sentiment_pred, sentiment_conf = analyze_sentiment(market_data)
        results['sentiment'] = {
            'prediction': sentiment_pred,
            'confidence': sentiment_conf
        }
        
        # Technical analysis
        from ..analysis.technical import get_technical_prediction
        tech_pred, tech_conf = get_technical_prediction(round_data)
        results['technical'] = {
            'prediction': tech_pred,
            'confidence': tech_conf
        }
        
        # Count signals
        signals = []
        confidences = []
        
        if volume_pred:
            signals.append(volume_pred)
            confidences.append(volume_conf)
            
        if sentiment_pred:
            signals.append(sentiment_pred)
            confidences.append(sentiment_conf)
            
        if tech_pred:
            signals.append(tech_pred)
            confidences.append(tech_conf)
        
        # Determine overall prediction
        if signals:
            # Count bull vs bear signals
            bull_count = signals.count("BULL")
            bear_count = signals.count("BEAR")
            
            # Calculate weighted confidence
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
            else:
                avg_confidence = 0.5
                
            # Determine if signals agree
            results['signals_agreement'] = (bull_count == len(signals)) or (bear_count == len(signals))
            
            # Overall prediction based on majority
            if bull_count > bear_count:
                results['overall'] = "BULL"
                # Adjust confidence based on agreement
                results['confidence'] = avg_confidence * (0.8 + (0.2 * results['signals_agreement']))
            elif bear_count > bull_count:
                results['overall'] = "BEAR"
                # Adjust confidence based on agreement
                results['confidence'] = avg_confidence * (0.8 + (0.2 * results['signals_agreement']))
            else:
                # Equal signals - use technical as tiebreaker
                results['overall'] = tech_pred or "BULL"
                results['confidence'] = 0.51
        else:
            # No signals, make a very low confidence prediction
            results['overall'] = "BULL"
            results['confidence'] = 0.5
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error in consolidated analysis: {e}")
        traceback.print_exc()
        return results

def evaluate_strategy_performance(strategy_name, lookback=100):
    """
    Evaluate the performance of a specific prediction strategy.
    
    Args:
        strategy_name: Name of the strategy to evaluate
        lookback: Number of recent predictions to analyze
        
    Returns:
        dict: Performance metrics
    """
    try:
        from ..data.processing import calculate_strategy_performance
        
        # Get overall strategy performance data
        performance_data = calculate_strategy_performance()
        
        # Extract data for the requested strategy
        if 'strategies' in performance_data and strategy_name in performance_data['strategies']:
            strategy_perf = performance_data['strategies'][strategy_name]
            
            # Calculate additional metrics
            win_rate = strategy_perf.get('accuracy', 0.5)
            sample_size = strategy_perf.get('sample_size', 0)
            
            # Determine if strategy is reliable
            reliable = sample_size >= 10 and (win_rate > 0.6 or win_rate < 0.4)
            
            # Determine if strategy should be used in contrarian mode
            contrarian_mode = win_rate < 0.4 and sample_size >= 10
            
            # Log analysis
            logger.info(f"\n📊 Strategy Analysis: {strategy_name}")
            logger.info(f"Win Rate: {win_rate:.2f} over {sample_size} samples")
            logger.info(f"Reliable: {'Yes' if reliable else 'No'}")
            logger.info(f"Recommended Mode: {'Contrarian' if contrarian_mode else 'Normal'}")
            
            return {
                'name': strategy_name,
                'win_rate': win_rate,
                'sample_size': sample_size,
                'reliable': reliable,
                'contrarian_mode': contrarian_mode
            }
        else:
            logger.warning(f"⚠️ No performance data for strategy: {strategy_name}")
            return {
                'name': strategy_name,
                'win_rate': 0.5,
                'sample_size': 0,
                'reliable': False,
                'contrarian_mode': False
            }
            
    except Exception as e:
        logger.error(f"❌ Error evaluating strategy performance: {e}")
        traceback.print_exc()
        return {
            'name': strategy_name,
            'win_rate': 0.5,
            'sample_size': 0,
            'reliable': False,
            'contrarian_mode': False
        }

def get_prediction_config_settings():
    """
    Get prediction configuration settings from the config object.
    
    Returns:
        dict: Prediction configuration settings
    """
    try:
        # Access the config variable that's imported from ..core.constants
        prediction_settings = {
            'min_confidence': config.get('prediction', {}).get('min_confidence', 0.6),
            'contrarian_enabled': config.get('contrarian_mode', {}).get('enable', True),
            'accuracy_threshold': config.get('contrarian_mode', {}).get('accuracy_threshold', 0.35),
            'volume_weight': config.get('prediction', {}).get('volume_weight', 0.3),
            'sentiment_weight': config.get('prediction', {}).get('sentiment_weight', 0.2),
            'technical_weight': config.get('prediction', {}).get('technical_weight', 0.5),
            'high_confidence_threshold': config.get('prediction', {}).get('high_confidence', 0.75)
        }
        
        logger.info(f"Loaded prediction settings from config: min_confidence={prediction_settings['min_confidence']}")
        return prediction_settings
        
    except Exception as e:
        logger.error(f"Error getting prediction config: {e}")
        return {
            'min_confidence': 0.6,
            'contrarian_enabled': True,
            'accuracy_threshold': 0.35,
            'volume_weight': 0.3,
            'sentiment_weight': 0.2,
            'technical_weight': 0.5,
            'high_confidence_threshold': 0.75
        } 