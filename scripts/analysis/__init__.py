"""
Analysis modules for the trading bot.
Contains technical analysis, market analysis, pattern recognition, 
volume analysis, and market regime detection.
"""

from .market import (get_historical_prices, get_market_direction,
                     get_market_prices_with_fallback, get_market_sentiment)
from .pattern import detect_advanced_patterns
from .regime import detect_market_regime
from .short_term import analyze_short_term_momentum
from .short_term import calculate_microtrend as calculate_short_term_microtrend
from .short_term import (detect_price_pattern, get_bootstrap_signal,
                         get_volume_acceleration)
from .swing import (detect_price_swing, detect_swing_pattern,
                    optimize_swing_trading)
from .technical import (analyze_bollinger, analyze_ema_cross, analyze_macd,
                        analyze_market_range, analyze_rsi,
                        calculate_bollinger_bands, calculate_ema,
                        calculate_macd, calculate_microtrend, calculate_rsi,
                        detect_combined_technical_reversal,
                        detect_market_reversal, get_ranging_market_prediction,
                        get_technical_analysis,
                        get_technical_indicators_with_fallback,
                        get_technical_prediction)
from .volume import (analyze_volume_profile, analyze_vwap, calculate_obv,
                     detect_volume_divergence, get_volume_prediction,
                     get_volume_trend_prediction)

__all__ = [
    "get_market_sentiment",
    "get_market_direction",
    "get_historical_prices",
    "get_market_prices_with_fallback",
    "detect_advanced_patterns",
    "detect_market_regime",
    "calculate_microtrend",
    "get_technical_prediction",
    "detect_market_reversal",
    "detect_combined_technical_reversal",
    "analyze_market_range",
    "get_ranging_market_prediction",
    "analyze_short_term_momentum",
    "detect_price_pattern",
    "calculate_short_term_microtrend",
    "get_volume_acceleration",
    "get_bootstrap_signal",
    "calculate_rsi",
    "calculate_macd",
    "calculate_bollinger_bands",
    "calculate_ema",
    "analyze_rsi",
    "analyze_macd",
    "analyze_bollinger",
    "analyze_ema_cross",
    "get_technical_analysis",
    "get_technical_indicators_with_fallback",
    "analyze_volume_profile",
    "calculate_obv",
    "analyze_vwap",
    "detect_volume_divergence",
    "get_volume_prediction",
    "get_volume_trend_prediction",
    "detect_price_swing",
    "detect_swing_pattern",
    "optimize_swing_trading",
]
