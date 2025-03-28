"""
Ensemble prediction for the trading bot.
Combines predictions from multiple models and sources.
"""

import traceback
import logging
import asyncio

from ..analysis.market import (
    get_market_prices_with_fallback,
    get_market_sentiment,
    fetch_market_data
)
from ..utils.helpers import get_price_trend
from ..data.features import prepare_features

logger = logging.getLogger(__name__)

async def ensemble_prediction(round_data, strategies=None):
    """
    Generate prediction using multiple strategies.
    
    Args:
        round_data: Dictionary with current round data
        strategies: Optional dictionary of strategy instances
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Default weights
        weights = {
            "model": 0.15,
            "pattern": 0.20,
            "market": 0.20,
            "technical": 0.20,
            "sentiment": 0.25
        }
        
        # Initialize prediction containers
        predictions = {}
        confidences = {}
        
        # 1. Get prediction from traditional ML model
        features = prepare_features(round_data)
        
        try:
            # Import here to avoid circular imports
            from ..models.model_prediction import get_model_prediction
            model_pred, model_conf = get_model_prediction(features)
            
            if model_pred:
                predictions['model'] = model_pred
                confidences['model'] = model_conf
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting model prediction: {e}")
        
        # 2. Get prediction from technical analysis
        try:
            from ..analysis.technical import get_technical_prediction
            tech_pred, tech_conf = get_technical_prediction(round_data)
            
            if tech_pred:
                predictions['technical'] = tech_pred
                confidences['technical'] = tech_conf
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting technical prediction: {e}")
        
        # 3. Get prediction from pattern analysis
        if 'prices' in round_data and round_data['prices']:
            try:
                from ..analysis.pattern import detect_advanced_patterns
                pattern_pred, pattern_conf = detect_advanced_patterns(round_data['prices'])
                
                if pattern_pred:
                    predictions['pattern'] = pattern_pred
                    confidences['pattern'] = pattern_conf
                    
            except Exception as e:
                logger.warning(f"⚠️ Error getting pattern prediction: {e}")
        
        # 4. Get prediction from market sentiment
        try:
            from ..analysis.market import get_market_sentiment
            sentiment, strength = get_market_sentiment(round_data)
            
            if sentiment == "bullish":
                predictions['sentiment'] = "BULL"
                confidences['sentiment'] = strength
            elif sentiment == "bearish":
                predictions['sentiment'] = "BEAR"
                confidences['sentiment'] = strength
                
        except Exception as e:
            logger.warning(f"⚠️ Error getting sentiment prediction: {e}")
        
        # If no predictions, return default
        if not predictions:
            logger.warning("⚠️ No predictions generated, using default")
            return "BULL", 0.51
        
        # Calculate weighted prediction
        bull_weight = 0
        bear_weight = 0
        total_weight = 0
        
        for strategy, prediction in predictions.items():
            weight = weights.get(strategy, 0.1) * confidences.get(strategy, 0.5)
            total_weight += weight
            
            if prediction == "BULL":
                bull_weight += weight
            elif prediction == "BEAR":
                bear_weight += weight
        
        if total_weight == 0:
            return "BULL", 0.51
            
        # Final prediction
        final_prediction = "BULL" if bull_weight > bear_weight else "BEAR"
        confidence = max(bull_weight, bear_weight) / total_weight
        
        # Log the ensemble results
        logger.info(f"\n🤖 Ensemble Prediction Analysis:")
        logger.info(f"   Strategies used: {list(predictions.keys())}")
        for strategy in predictions:
            logger.info(f"   {strategy.upper()}: {predictions[strategy]} ({confidences[strategy]:.2f})")
        logger.info(f"   Final prediction: {final_prediction}")
        logger.info(f"   Confidence: {confidence:.2f}")
        
        return final_prediction, confidence
        
    except Exception as e:
        logger.error(f"❌ Error in ensemble prediction: {e}")
        traceback.print_exc()
        return "BULL", 0.51

async def get_predictions_concurrently(round_data):
    """
    Get predictions from multiple sources concurrently using asyncio.
    
    Args:
        round_data: Dictionary with current round data
        
    Returns:
        dict: Predictions from various sources
    """
    try:
        # Define async tasks to run concurrently
        tasks = [
            get_model_prediction_async(round_data),
            get_technical_prediction_async(round_data),
            get_pattern_prediction_async(round_data),
            get_sentiment_prediction_async(round_data),
            get_trend_prediction_async(round_data)
        ]
        
        # Use asyncio.gather to run all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        predictions = {}
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Task {i} failed: {result}")
            elif result and isinstance(result, tuple) and len(result) >= 2:
                source, prediction, confidence = result[0], result[1], result[2]
                predictions[source] = {'prediction': prediction, 'confidence': confidence}
                
        return predictions
        
    except Exception as e:
        logger.error(f"Error in concurrent predictions: {e}")
        return {}

# Async wrapper functions for prediction sources
async def get_model_prediction_async(round_data):
    try:
        features = prepare_features(round_data)
        from ..models.model_prediction import get_model_prediction
        pred, conf = get_model_prediction(features)
        return ('model', pred, conf)
    except Exception as e:
        logger.error(f"Model prediction error: {e}")
        raise

async def get_technical_prediction_async(round_data):
    """
    Async wrapper for technical prediction that uses market price data.
    
    Args:
        round_data: Dictionary with round data
        
    Returns:
        tuple: (source, prediction, confidence)
    """
    try:
        # Use the imported get_market_prices_with_fallback function
        prices = get_market_prices_with_fallback(lookback=15)
        
        if not prices or len(prices) < 5:
            logger.warning("Insufficient price data for technical prediction")
            return ('technical', None, 0)
            
        # Add prices to round data if not present
        if 'prices' not in round_data or not round_data['prices']:
            round_data['prices'] = prices
            
        # Get technical prediction
        from ..analysis.technical import get_technical_prediction
        pred, conf = get_technical_prediction(round_data)
        return ('technical', pred, conf)
        
    except Exception as e:
        logger.error(f"Technical prediction error: {e}")
        raise

async def get_sentiment_prediction_async(round_data):
    """
    Async wrapper for sentiment prediction.
    
    Args:
        round_data: Dictionary with round data
        
    Returns:
        tuple: (source, prediction, confidence)
    """
    try:
        # Use the imported get_market_sentiment function
        sentiment, strength = get_market_sentiment(round_data)
        
        if sentiment == "bullish":
            return ('sentiment', "BULL", strength)
        elif sentiment == "bearish":
            return ('sentiment', "BEAR", strength)
        else:
            return ('sentiment', None, 0)
            
    except Exception as e:
        logger.error(f"Sentiment prediction error: {e}")
        raise

async def get_pattern_prediction_async(round_data):
    """
    Async wrapper for pattern prediction that uses market data.
    
    Args:
        round_data: Dictionary with round data
        
    Returns:
        tuple: (source, prediction, confidence)
    """
    try:
        # Use the imported fetch_market_data function to get extra data
        market_data = fetch_market_data()
        
        # Enhance round data with market data
        enhanced_data = {**round_data}
        if market_data:
            enhanced_data['market_data'] = market_data
        
        # Get pattern prediction using enhanced data
        prices = round_data.get('prices', [])
        if not prices and 'market_data' in enhanced_data:
            prices = enhanced_data['market_data'].get('recent_prices', [])
            
        if not prices or len(prices) < 5:
            logger.warning("Insufficient price data for pattern prediction")
            return ('pattern', None, 0)
            
        from ..analysis.pattern import detect_advanced_patterns
        pred, conf = detect_advanced_patterns(prices)
        return ('pattern', pred, conf)
        
    except Exception as e:
        logger.error(f"Pattern prediction error: {e}")
        raise

async def get_trend_prediction_async(round_data):
    """
    Async wrapper for trend-based prediction.
    Uses get_price_trend to determine market direction.
    
    Args:
        round_data: Dictionary with round data
        
    Returns:
        tuple: (source, prediction, confidence)
    """
    try:
        # Use the imported get_price_trend function
        trend, strength = get_price_trend(lookback=10)
        
        prediction = None
        if trend == "uptrend":
            prediction = "BULL"
        elif trend == "downtrend":
            prediction = "BEAR"
        else:
            return ('trend', None, 0)
            
        # Adjust confidence based on trend strength
        confidence = min(0.5 + strength, 0.9)
        
        logger.info(f"Trend prediction: {prediction} ({confidence:.2f}) based on {trend}")
        return ('trend', prediction, confidence)
        
    except Exception as e:
        logger.error(f"Trend prediction error: {e}")
        raise 