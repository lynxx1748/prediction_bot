"""
Prediction storage functionality.
Handles saving prediction signals to the database for tracking and analysis.
"""

import logging
import traceback

from ..data.database import record_prediction

logger = logging.getLogger(__name__)

def store_signal_predictions(epoch, predictions):
    """
    Store predictions from different signals in the database.
    
    Args:
        epoch: The epoch number for the prediction
        predictions: Dictionary of predictions from different sources
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract predictions
        model_pred = predictions.get('model')
        trend_pred = predictions.get('trend')
        market_pred = predictions.get('market')
        pattern_pred = predictions.get('pattern')
        volume_pred = predictions.get('volume')
        ai_pred = predictions.get('ai')
        
        # Prepare data for database handler
        prediction_data = {
            'model_prediction': model_pred,
            'trend_prediction': trend_pred,
            'market_prediction': market_pred,
            'pattern_prediction': pattern_pred,
            'volume_prediction': volume_pred,
            'ai_prediction': ai_pred
        }
        
        # Use the centralized database handler
        record_prediction(epoch, prediction_data)
        
        logger.info(f"✅ Stored signal predictions for epoch {epoch}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error storing signal predictions: {e}")
        traceback.print_exc()
        return False 