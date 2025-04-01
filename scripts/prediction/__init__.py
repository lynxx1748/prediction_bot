"""
Prediction functionality for the trading bot.
Contains prediction handlers, strategies, integration, and ensemble methods.
"""

# Add a reference to the short-term module for use in prediction strategies
from ..analysis.short_term import analyze_short_term_momentum
from .analysis import (analyze_prediction_accuracy, analyze_sentiment,
                       analyze_volume, evaluate_strategy_performance,
                       get_consolidated_analysis)
from .ensemble import ensemble_prediction
from .filtering import filter_signals
from .handler import PredictionHandler
from .integration import integrate_reversal_detection, integrate_signals
from .storage import store_signal_predictions
from .strategies import (ContrarianStrategy, Strategy, SwingTradingStrategy,
                         TrendFollowingStrategy, VolumeBasedStrategy,
                         get_optimal_strategy, get_strategy)
from .strategy_selector import get_fallback_strategy, select_optimal_strategy
from .timing import get_optimal_entry_time
from .validation import validate_trade_signal
from .weights import (apply_regime_boost, get_minimum_weights,
                      get_regime_optimized_weights, get_strategy_defaults,
                      update_strategy_weights)

__all__ = [
    "PredictionHandler",
    "ensemble_prediction",
    "analyze_prediction_accuracy",
    "evaluate_strategy_performance",
    "analyze_volume",
    "analyze_sentiment",
    "get_consolidated_analysis",
    "integrate_reversal_detection",
    "integrate_signals",
    "Strategy",
    "TrendFollowingStrategy",
    "ContrarianStrategy",
    "VolumeBasedStrategy",
    "SwingTradingStrategy",
    "get_strategy",
    "get_optimal_strategy",
    "analyze_short_term_momentum",
    "filter_signals",
    "validate_trade_signal",
    "store_signal_predictions",
    "select_optimal_strategy",
    "get_fallback_strategy",
    "get_optimal_entry_time",
    "update_strategy_weights",
    "get_regime_optimized_weights",
    "get_strategy_defaults",
    "get_minimum_weights",
    "apply_regime_boost",
]
