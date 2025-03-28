"""
Models module for the trading bot.
Contains prediction models, technical analysis, and AI strategies.
"""

import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Import key functionality
from .func_ai_strategy import AIStrategy
from .func_hybrid import hybrid_prediction, hybrid_prediction_with_learning, record_hybrid_outcome
from .func_rf import train_model
from .func_rf_enhanced import AdaptiveRandomForest
from .func_ta import TechnicalAnalysis, get_technical_indicators
from .model_evaluation import ModelEvaluator
from .model_version_control import ModelVersionControl

# Define public interface
__all__ = [
    'AIStrategy',
    'hybrid_prediction',
    'hybrid_prediction_with_learning',
    'record_hybrid_outcome',
    'train_model',
    'AdaptiveRandomForest',
    'TechnicalAnalysis',
    'get_technical_indicators',
    'ModelEvaluator',
    'ModelVersionControl'
]
