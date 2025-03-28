import numpy as np
import json
import traceback
import logging
from datetime import datetime
from collections import deque

# Import from configuration module
from configuration import config

# Setup logger
logger = logging.getLogger(__name__)

# Import necessary functions from other modules
from scripts.analysis.market import fetch_market_data
from scripts.prediction.analysis import analyze_prediction_accuracy
from scripts.data.database import record_prediction
from scripts.prediction.ensemble import ensemble_prediction
from scripts.analysis.pattern import detect_advanced_patterns as detect_pattern

# Recent predictions cache
prediction_history = deque(maxlen=20)

def hybrid_prediction(round_data, technical_data, market_data, config_override=None):
    """
    Generate prediction using multiple models and strategies.
    
    Args:
        round_data: Dictionary containing the current betting round data
        technical_data: Technical analysis data
        market_data: Market price data
        config_override: Optional config to override global config
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Use provided config or get from global config
        cfg = config_override or config.get()
        
        # Get pattern prediction
        pattern_prediction, pattern_confidence = detect_pattern(technical_data)
        
        # Use bull/bear ratio as market pressure indicator
        bull_ratio = round_data.get('bullRatio', 0.5)
        bear_ratio = round_data.get('bearRatio', 0.5)
        
        market_pressure_score = bull_ratio - bear_ratio  # Range: -1 to 1
        
        # Determine market pressure prediction
        if market_pressure_score > 0.15:  # Strong bull pressure
            market_prediction = "BEAR"  # Contrarian approach
            market_confidence = min(0.9, abs(market_pressure_score) * 1.5)
        elif market_pressure_score < -0.15:  # Strong bear pressure
            market_prediction = "BULL"  # Contrarian approach
            market_confidence = min(0.9, abs(market_pressure_score) * 1.5)
        else:
            # Default case when no strong market pressure
            market_prediction = "BULL" if bull_ratio > 0.5 else "BEAR"
            market_confidence = 0.51
        
        # Weight the predictions according to config
        predictions = {
            'pattern': (pattern_prediction, pattern_confidence, cfg.get('STRATEGY_WEIGHTS', {}).get('pattern', 0.20)),
            'market': (market_prediction, market_confidence, cfg.get('STRATEGY_WEIGHTS', {}).get('market', 0.20)),
        }
        
        # Calculate weighted prediction
        bull_weight = 0
        bear_weight = 0
        total_weight = 0
        
        for model, (prediction, confidence, weight) in predictions.items():
            if prediction and confidence > 0:
                if prediction == "BULL":
                    bull_weight += confidence * weight
                elif prediction == "BEAR":
                    bear_weight += confidence * weight
                total_weight += weight
        
        # Determine final prediction
        if total_weight > 0:
            if bull_weight > bear_weight:
                final_prediction = "BULL"
                confidence = bull_weight / total_weight
            else:
                final_prediction = "BEAR"
                confidence = bear_weight / total_weight
        else:
            # Default to previous round winner if no clear signal
            final_prediction = "BULL" if bull_ratio > bear_ratio else "BEAR"
            confidence = 0.51
        
        # Store the prediction with all metadata
        epoch = round_data.get('epoch')
        
        prediction_data = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'pattern_prediction': pattern_prediction,
            'market_prediction': market_prediction,
            'final_prediction': final_prediction,
            'pattern_confidence': float(pattern_confidence),
            'market_confidence': float(market_confidence),
            'confidence': float(confidence),
            'bull_ratio': float(bull_ratio),
            'bear_ratio': float(bear_ratio)
        }
        
        # Add to local cache
        prediction_history.append(prediction_data)
        
        # Store in database using centralized function
        record_prediction(epoch, prediction_data)
        
        logger.info(f"✅ Hybrid prediction: {final_prediction} (confidence: {confidence:.2f})")
        
        return final_prediction, confidence
        
    except Exception as e:
        logger.error(f"❌ Error generating hybrid prediction: {e}")
        traceback.print_exc()
        return "UNKNOWN", 0.0

def hybrid_prediction_old(round_data, model=None, scaler=None, contract=None):
    """Make prediction using hybrid strategy with model and market analysis"""
    try:
        # Get the epoch/round number from the round_data
        betting_round = round_data.get('epoch')
        if not betting_round and contract:
            # If epoch is not in round_data, try to get current epoch from contract
            betting_round = contract.functions.currentEpoch().call()
        
        # Get market data for additional features
        market_data = fetch_market_data()
        
        if market_data is None:
            # Default values if market data can't be fetched
            bnb_price_change = 0.0
            btc_price_change = 0.0
            logger.warning("⚠️ Using default market values due to fetch error")
        else:
            bnb_price_change = market_data.get("bnb_24h_change", 0.0)
            btc_price_change = market_data.get("btc_24h_change", 0.0)
        
        # Extract features for model prediction
        features = np.array([[
            round_data['bullRatio'],
            round_data['bearRatio'],
            round_data['totalAmount'],
            bnb_price_change,
            btc_price_change
        ]])
        
        logger.info(f"Using features: bullRatio={round_data['bullRatio']:.2f}, bearRatio={round_data['bearRatio']:.2f}, " +
              f"totalAmount={round_data['totalAmount']:.2f}, bnb_change={bnb_price_change:.2f}, btc_change={btc_price_change:.2f}")
        
        # Use the new ensemble prediction method
        final_prediction, final_confidence = ensemble_prediction(round_data, market_data, config)
        
        # Apply contrarian mode if activated
        contrarian_mode = analyze_prediction_accuracy(config)
        if contrarian_mode:
            logger.warning(f"⚠️ CONTRARIAN MODE ACTIVATED - Flipping prediction")
            final_prediction = "BEAR" if final_prediction == "BULL" else "BULL"
            final_confidence = 0.7  # Set a reasonable confidence for contrarian predictions
        
        # Print comprehensive analysis
        logger.info("\n🔍 Comprehensive Market Analysis:")
        logger.info(f"\n📊 Volume Analysis:")
        logger.info(f"Bull Ratio: {round_data['bullRatio']:.2f}")
        logger.info(f"Bear Ratio: {round_data['bearRatio']:.2f}")
        logger.info(f"Total Amount: {round_data['totalAmount']:.2f} BNB")
        
        # Store prediction for learning
        record_prediction(betting_round, {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': betting_round,
            'nn_prediction': None,
            'rf_prediction': None,
            'pattern_prediction': None,
            'market_prediction': None,
            'final_prediction': final_prediction,
            'nn_confidence': 0,
            'rf_confidence': 0,
            'pattern_confidence': 0,
            'market_confidence': 0,
            'confidence': final_confidence,
            'bull_ratio': round_data['bullRatio'],
            'bear_ratio': round_data['bearRatio'],
            'totalAmount': round_data['totalAmount'],
            'bnb_24h_change': bnb_price_change,
            'btc_24h_change': btc_price_change
        })
        
        return final_prediction, final_confidence
        
    except Exception as e:
        logger.error(f"❌ Error in hybrid prediction: {e}")
        traceback.print_exc()
        return "BULL", 0.5  # Default prediction on error 

def hybrid_prediction_with_learning(round_data, technical_data, market_data, prediction_history=None):
    """
    Enhanced hybrid prediction with self-learning capabilities.
    
    Args:
        round_data: Dictionary with current round data
        technical_data: Dictionary with technical analysis data
        market_data: Dictionary with market sentiment data
        prediction_history: Optional dictionary tracking past prediction performance
        
    Returns:
        tuple: (prediction, confidence, updated_history)
    """
    # Initialize prediction history if not provided
    if prediction_history is None:
        prediction_history = {
            'predictions': [],
            'weights': {
                'pattern': 0.2,
                'market': 0.2, 
                'technical': 0.3,
                'sentiment': 0.3
            },
            'performance': {
                'pattern': 0.5,
                'market': 0.5,
                'technical': 0.5,
                'sentiment': 0.5
            }
        }
    
    # Get standard prediction
    prediction, confidence = hybrid_prediction(round_data, technical_data, market_data)
    
    # Store this prediction for future learning
    current_epoch = round_data.get('epoch')
    prediction_history['predictions'].append({
        'epoch': current_epoch,
        'prediction': prediction,
        'confidence': confidence,
        'actual': None,  # Will be filled later when outcome is known
        'components': {
            # Store individual component predictions here
            # This would need to be implemented by extracting component predictions
        }
    })
    
    # Adjust weights based on past performance if we have enough data
    if len(prediction_history['predictions']) > 10:
        # Calculate component accuracy
        component_accuracy = {}
        for component in ['pattern', 'market', 'technical', 'sentiment']:
            if component in prediction_history['performance']:
                component_accuracy[component] = prediction_history['performance'][component]
        
        # Adjust weights proportionally to accuracy
        total_accuracy = sum(component_accuracy.values())
        if total_accuracy > 0:
            new_weights = {}
            for component, accuracy in component_accuracy.items():
                # Normalize accuracy and apply as weight, with some smoothing
                new_weights[component] = 0.1 + (0.9 * accuracy / total_accuracy)
                
            # Update weights
            prediction_history['weights'] = new_weights
            logger.info(f"Updated hybrid prediction weights: {new_weights}")
    
    return prediction, confidence, prediction_history

def record_hybrid_outcome(prediction_history, epoch, actual_outcome):
    """
    Record actual outcome to enable learning.
    
    Args:
        prediction_history: Dictionary tracking prediction history
        epoch: Epoch number of the prediction
        actual_outcome: Actual outcome (BULL or BEAR)
        
    Returns:
        dict: Updated prediction history
    """
    # Find the prediction for this epoch
    for pred in prediction_history['predictions']:
        if pred['epoch'] == epoch and pred['actual'] is None:
            # Record actual outcome
            pred['actual'] = actual_outcome
            
            # Update component performance
            correct = pred['prediction'] == actual_outcome
            
            # Update each component's accuracy
            for component, prediction in pred.get('components', {}).items():
                component_correct = prediction == actual_outcome
                
                # Exponential moving average of performance (75% new, 25% history)
                current = prediction_history['performance'].get(component, 0.5)
                prediction_history['performance'][component] = (0.75 * component_correct) + (0.25 * current)
            
            break
    
    # Trim history to keep only last 100 predictions
    if len(prediction_history['predictions']) > 100:
        prediction_history['predictions'] = prediction_history['predictions'][-100:]
    
    return prediction_history 