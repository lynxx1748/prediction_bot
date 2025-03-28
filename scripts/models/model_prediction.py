"""
Model prediction adapter module.
Routes prediction requests to the appropriate model implementation.
"""

import logging
import traceback

logger = logging.getLogger(__name__)

def get_model_prediction(features_or_round_data):
    """
    Get prediction from the appropriate model based on configuration.
    Acts as an adapter to route to the correct model implementation.
    
    Args:
        features_or_round_data: Either preprocessed features or raw round data
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Import here to avoid circular imports
        from .random_forest import get_model_prediction as rf_prediction
        
        # For now, simply route to random forest model
        return rf_prediction(features_or_round_data)
        
    except Exception as e:
        logger.error(f"❌ Error in model prediction adapter: {e}")
        traceback.print_exc()
        return None, 0 