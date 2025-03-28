"""
Prediction functionality for the trading bot.
Contains prediction handlers, strategies, integration, and ensemble methods.
"""

from .handler import PredictionHandler
from .ensemble import ensemble_prediction
from .analysis import (
    analyze_prediction_accuracy,
    evaluate_strategy_performance,
    analyze_volume,
    analyze_sentiment,
    get_consolidated_analysis
)
from .integration import (
    integrate_reversal_detection,
    integrate_signals
)
from .strategies import (
    Strategy,
    TrendFollowingStrategy,
    ContrarianStrategy,
    VolumeBasedStrategy,
    SwingTradingStrategy,
    get_strategy,
    get_optimal_strategy
)
from .filtering import filter_signals
from .validation import validate_trade_signal
from .storage import store_signal_predictions
from .strategy_selector import select_optimal_strategy, get_fallback_strategy
from .timing import get_optimal_entry_time
from .weights import (
    update_strategy_weights,
    get_regime_optimized_weights,
    get_strategy_defaults,
    get_minimum_weights,
    apply_regime_boost
)
# Add a reference to the short-term module for use in prediction strategies
from ..analysis.short_term import analyze_short_term_momentum

__all__ = [
    'PredictionHandler',
    'ensemble_prediction',
    'analyze_prediction_accuracy',
    'evaluate_strategy_performance',
    'analyze_volume',
    'analyze_sentiment',
    'get_consolidated_analysis',
    'integrate_reversal_detection',
    'integrate_signals',
    'Strategy',
    'TrendFollowingStrategy', 
    'ContrarianStrategy',
    'VolumeBasedStrategy',
    'SwingTradingStrategy',
    'get_strategy',
    'get_optimal_strategy',
    'analyze_short_term_momentum',
    'filter_signals',
    'validate_trade_signal',
    'store_signal_predictions',
    'select_optimal_strategy',
    'get_fallback_strategy',
    'get_optimal_entry_time',
    'update_strategy_weights',
    'get_regime_optimized_weights',
    'get_strategy_defaults',
    'get_minimum_weights',
    'apply_regime_boost'
] 