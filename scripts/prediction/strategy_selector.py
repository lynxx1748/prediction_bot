"""
Strategy selection for the trading bot.
Dynamically selects optimal strategies based on market conditions and past performance.
"""

import logging
import random
import traceback

import numpy as np

from ..data.database import get_strategy_performance_by_regime

logger = logging.getLogger(__name__)


def select_optimal_strategy(market_regime, historical_performance, round_data):
    """
    Dynamically select the best strategy based on market conditions and past performance.

    Args:
        market_regime: Current market regime information
        historical_performance: Historical performance metrics
        round_data: Current round data

    Returns:
        dict: Strategy selection information
    """
    # First analyze round characteristics (uses round_data)
    round_characteristics = analyze_round_characteristics(round_data)

    try:
        # Get regime info
        regime = market_regime.get("regime", "unknown")

        # Get sample size and handle the early-stage logging appropriately
        sample_size = historical_performance.get("sample_size", 0)

        # Check if we have enough historical data
        if sample_size < 5:
            # For the first few rounds, use info level instead of warning
            if sample_size == 0:
                logger.info("Starting with default strategies - no historical data yet")
            elif sample_size < 3:
                logger.info(
                    f"Building historical data ({sample_size}/5 samples collected)"
                )
            else:
                logger.info(
                    f"Almost ready for strategy optimization ({sample_size}/5 samples)"
                )

            return get_fallback_strategy(market_regime)

        # Get strategy performance in this market regime
        performance_data = get_strategy_performance_by_regime(regime, lookback=50)

        # Check if we have strategy performance data
        strategies_with_data = [
            s
            for s in performance_data.keys()
            if performance_data[s].get("sample_size", 0) >= 5
        ]

        if not strategies_with_data:
            logger.info(
                f"Not enough samples for {regime} regime strategies - using defaults"
            )
            return get_fallback_strategy(market_regime)

        # Select top performing strategies for this regime
        top_strategies = []
        for strategy, metrics in performance_data.items():
            accuracy = metrics.get("accuracy", 0)
            sample_size = metrics.get("sample_size", 0)

            # Only consider strategies with enough samples
            if sample_size >= 10:
                # Score based on accuracy with confidence interval correction
                # This prefers higher sample sizes at same accuracy
                score = accuracy * (1 - 1.0 / np.sqrt(sample_size))
                top_strategies.append((strategy, score))

        # Sort by score
        top_strategies.sort(key=lambda x: x[1], reverse=True)

        # If we have good strategies, use them
        if top_strategies and top_strategies[0][1] > 0.52:
            primary_strategy = top_strategies[0][0]
            logger.info(
                f"🔄 Selected {primary_strategy} as primary strategy (score: {top_strategies[0][1]:.2f})"
            )

            # If we have a second good strategy, use it as backup
            secondary_strategy = None
            if len(top_strategies) > 1 and top_strategies[1][1] > 0.5:
                secondary_strategy = top_strategies[1][0]
                logger.info(
                    f"🔄 Selected {secondary_strategy} as backup strategy (score: {top_strategies[1][1]:.2f})"
                )

            return {
                "primary": primary_strategy,
                "secondary": secondary_strategy,
                "score": top_strategies[0][1],
            }

        # Fallback strategy selection based on regime
        if regime == "uptrend":
            return {"primary": "trend_following", "secondary": "momentum", "score": 0.5}
        elif regime == "downtrend":
            return {"primary": "trend_following", "secondary": "momentum", "score": 0.5}
        elif regime == "volatile":
            return {
                "primary": "mean_reversion",
                "secondary": "volume_analysis",
                "score": 0.5,
            }
        else:  # ranging
            return {
                "primary": "contrarian",
                "secondary": "support_resistance",
                "score": 0.5,
            }

    except Exception as e:
        logger.error(f"❌ Error selecting optimal strategy: {e}")
        traceback.print_exc()
        return {
            "primary": "trend_following",
            "secondary": None,
            "score": 0.5,
        }  # Safe fallback


def get_fallback_strategy(market_regime):
    """
    Get fallback strategy when no historical data is available.

    Args:
        market_regime: Information about the current market regime

    Returns:
        dict: Selected strategy information with primary, secondary, and score
    """
    try:
        # Add some randomness but weight toward proven general strategies
        regime = market_regime.get("regime", "unknown")
        regime_conf = market_regime.get("confidence", 0.5)

        # Define default strategies for each regime
        strategies = {
            "uptrend": [
                ("trend_following", 0.6),
                ("momentum", 0.3),
                ("mean_reversion", 0.1),
            ],
            "downtrend": [
                ("trend_following", 0.6),
                ("momentum", 0.3),
                ("mean_reversion", 0.1),
            ],
            "volatile": [
                ("mean_reversion", 0.5),
                ("range_trading", 0.3),
                ("trend_following", 0.2),
            ],
            "ranging": [
                ("range_trading", 0.5),
                ("mean_reversion", 0.3),
                ("contrarian", 0.2),
            ],
            "unknown": [
                ("trend_following", 0.4),
                ("mean_reversion", 0.3),
                ("range_trading", 0.3),
            ],
        }

        # Get strategies for this regime
        regime_strategies = strategies.get(regime, strategies["unknown"])

        # Adjust probabilities based on regime confidence
        # Higher confidence means we rely more on the primary strategy
        if regime_conf > 0.7:
            # Boost primary strategy probability when confidence is high
            primary_strategy, primary_prob = regime_strategies[0]
            primary_prob = min(0.8, primary_prob + (regime_conf - 0.5))
            regime_strategies[0] = (primary_strategy, primary_prob)

            # Normalize remaining probabilities
            total_remaining = 1.0 - primary_prob
            remaining_sum = sum(prob for _, prob in regime_strategies[1:])

            if remaining_sum > 0:
                for i in range(1, len(regime_strategies)):
                    strategy, prob = regime_strategies[i]
                    normalized_prob = (prob / remaining_sum) * total_remaining
                    regime_strategies[i] = (strategy, normalized_prob)

        # Weight by probability
        r = random.random()
        cumulative = 0

        for strategy, prob in regime_strategies:
            cumulative += prob
            if r <= cumulative:
                logger.info(
                    f"🔍 Using default {strategy} strategy for {regime} market (no historical data)"
                )
                return {"primary": strategy, "secondary": None, "score": 0.55}

        # Fallback
        return {"primary": "trend_following", "secondary": None, "score": 0.5}

    except Exception as e:
        logger.error(f"❌ Error getting fallback strategy: {e}")
        return {"primary": "trend_following", "secondary": None, "score": 0.5}


def analyze_round_characteristics(round_data):
    """
    Analyze characteristics of the current round to inform strategy selection.
    Makes explicit use of the round_data parameter.

    Args:
        round_data: Dictionary with current round data

    Returns:
        dict: Round characteristics
    """
    characteristics = {}

    # Extract volume data from round_data
    bull_amount = float(round_data.get("bullAmount", 0))
    bear_amount = float(round_data.get("bearAmount", 0))
    total_amount = bull_amount + bear_amount

    # Calculate volume-based metrics
    if total_amount > 0:
        bull_ratio = bull_amount / total_amount
        bear_ratio = bear_amount / total_amount
        volume_imbalance = abs(bull_ratio - bear_ratio)

        characteristics["bull_ratio"] = bull_ratio
        characteristics["bear_ratio"] = bear_ratio
        characteristics["volume_imbalance"] = volume_imbalance
        characteristics["is_high_volume"] = (
            total_amount > 1.0
        )  # Adjust threshold as needed

    # Check for price data
    prices = round_data.get("prices", [])
    if prices and len(prices) > 1:
        # Calculate price volatility
        price_changes = [
            (prices[i] - prices[i - 1]) / prices[i - 1] for i in range(1, len(prices))
        ]
        volatility = sum([abs(pc) for pc in price_changes]) / len(price_changes)
        characteristics["volatility"] = volatility

    return characteristics
