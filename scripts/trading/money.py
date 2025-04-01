"""
Money management for the trading bot.
Handles bet sizing and risk management.
"""

import logging

from ..data.processing import get_overall_performance

logger = logging.getLogger(__name__)


def calculate_optimal_bet_size(
    wallet_balance=1.0,
    win_rate=0.55,
    strategy_confidence=0.6,
    market_regime=None,
    base_amount=None,
):
    """
    Calculate optimal bet size using adaptive Kelly Criterion.

    Args:
        wallet_balance: Current wallet balance
        win_rate: Estimated win rate (0-1)
        strategy_confidence: Confidence in the prediction (0-1)
        market_regime: Dictionary with market regime information
        base_amount: Optional base wager amount (for compatibility)

    Returns:
        float: Optimal bet size in BNB
    """
    # Handle the base_amount parameter if provided
    if base_amount is not None:
        logger.info(f"Using provided base amount: {base_amount}")
        return _calculate_simple_bet_size(
            base_amount, strategy_confidence, wallet_balance
        )

    # Otherwise, use the original complex logic
    return _calculate_kelly_bet_size(
        wallet_balance, win_rate, strategy_confidence, market_regime
    )


def _calculate_simple_bet_size(base_amount, confidence, balance):
    """Simple bet size calculation using base amount and confidence."""
    try:
        # Validate confidence
        if confidence < 0.5:
            confidence = 0.5  # Minimum 50% confidence
        elif confidence > 0.95:
            confidence = 0.95  # Cap at 95%

        # Scale bet by confidence
        confidence_factor = 0.8 + (confidence * 0.4)  # 0.8-1.2 range
        bet_amount = base_amount * confidence_factor

        # Ensure minimum bet size
        min_bet = 0.001  # Minimum BNB
        if bet_amount < min_bet:
            bet_amount = min_bet

        # Ensure maximum bet size
        max_bet = balance * 0.05  # Never more than 5% of balance
        if bet_amount > max_bet:
            bet_amount = max_bet

        logger.info(
            f"💰 Simple bet sizing: {bet_amount:.5f} BNB (from base: {base_amount}, conf: {confidence:.2f})"
        )
        return bet_amount

    except Exception as e:
        logger.error(f"Error calculating simple bet size: {e}")
        return 0.002  # Default fallback


def _calculate_kelly_bet_size(
    wallet_balance, win_rate, strategy_confidence, market_regime
):
    """
    Original Kelly Criterion implementation.
    """
    try:
        # Get actual performance data to make better decisions
        performance = get_overall_performance(lookback=20)
        sample_size = performance.get("sample_size", 0)

        # Force minimum bets for the first few rounds
        if sample_size < 5:
            logger.info(
                f"🔍 Forcing minimal bets to build history (sample size: {sample_size})"
            )
            return 0.005  # Start with 0.005 BNB

        # Get regime-specific adjustments
        regime = market_regime.get("regime", "unknown")
        regime_conf = market_regime.get("confidence", 0.5)

        # Normalize confidence values
        strategy_confidence = min(max(strategy_confidence, 0.5), 0.95)
        regime_conf = min(max(regime_conf, 0.5), 0.95)

        # Calculate base win rate from strategy confidence
        base_win_rate = win_rate * (0.5 + strategy_confidence / 2)

        # Apply regime-specific adjustments
        if regime == "volatile":
            # More conservative in volatile markets
            adjusted_win_rate = base_win_rate * (0.8 + (0.2 * regime_conf))
        elif regime in ["uptrend", "downtrend"]:
            # More aggressive in trending markets
            adjusted_win_rate = base_win_rate * (0.9 + (0.2 * regime_conf))
        else:
            # Balanced approach for other regimes
            adjusted_win_rate = base_win_rate * (0.85 + (0.3 * regime_conf))

        # Ensure win rate stays within reasonable bounds
        adjusted_win_rate = min(max(adjusted_win_rate, 0.5), 0.95)

        # Calculate Kelly criterion
        b = 0.95  # Typical pancakeswap odds
        p = adjusted_win_rate
        q = 1 - p

        kelly_fraction = (p * b - q) / b
        half_kelly = max(kelly_fraction * 0.5, 0)  # Use half Kelly for safety

        # Calculate bet size
        max_bet_pct = 0.05  # Max 5% of wallet
        bet_pct = min(half_kelly, max_bet_pct)

        if bet_pct <= 0:
            logger.info("💰 Kelly criterion suggests not betting")
            return 0

        optimal_bet = wallet_balance * bet_pct

        # Apply minimum/maximum constraints
        min_bet = 0.005  # Minimum 0.005 BNB
        max_bet = wallet_balance * 0.05  # Maximum 5% of wallet

        final_bet = min(max(optimal_bet, min_bet), max_bet)

        logger.info(f"💰 Kelly bet: {final_bet:.5f} BNB ({bet_pct*100:.1f}% of wallet)")
        logger.info(f"   Win Rate: {adjusted_win_rate:.2f} (base: {base_win_rate:.2f})")
        logger.info(f"   Regime: {regime} (conf: {regime_conf:.2f})")

        return final_bet

    except Exception as e:
        logger.error(f"Error in Kelly calculation: {e}")
        return 0.005  # Return minimum bet on error


def calculate_risk_adjusted_bet_size(
    wallet_balance, win_rate, confidence, risk_level="medium"
):
    """
    Calculate bet size based on win rate, confidence and risk level.
    A simplified alternative to full Kelly Criterion.

    Args:
        wallet_balance: Current wallet balance
        win_rate: Estimated win rate (0-1)
        confidence: Confidence in the prediction (0-1)
        risk_level: 'low', 'medium', or 'high'

    Returns:
        float: Bet size
    """
    # Base percentage of wallet based on risk level
    risk_factors = {
        "low": 0.01,  # 1% of wallet
        "medium": 0.02,  # 2% of wallet
        "high": 0.04,  # 4% of wallet
    }

    base_percentage = risk_factors.get(risk_level, risk_factors["medium"])

    # Adjust for win rate - higher win rate means larger bet
    win_factor = 0.5 + win_rate  # ranges from 0.5 to 1.5

    # Adjust for confidence - higher confidence means larger bet
    confidence_factor = 0.5 + confidence  # ranges from 0.5 to 1.5

    # Calculate final bet percentage
    bet_percentage = base_percentage * win_factor * confidence_factor

    # Cap at reasonable maximum
    max_percentage = 0.05  # 5% of wallet
    bet_percentage = min(bet_percentage, max_percentage)

    # Calculate actual amount
    bet_amount = wallet_balance * bet_percentage

    # Ensure minimum bet or return 0
    min_bet = 0.005  # BNB
    if bet_amount < min_bet:
        return 0

    logger.info(
        f"💰 Risk-adjusted bet: {bet_amount:.5f} BNB ({bet_percentage*100:.1f}% of wallet)"
    )
    return bet_amount


def get_adaptive_bet_size(
    wallet_balance, win_rate, confidence, market_regime, losing_streak=0
):
    """
    Get adaptive bet size that adjusts based on recent performance.

    Args:
        wallet_balance: Current wallet balance
        win_rate: Estimated win rate (0-1)
        confidence: Confidence in the prediction (0-1)
        market_regime: Dictionary with market regime information
        losing_streak: Current losing streak count

    Returns:
        float: Bet size
    """
    # Start with optimal bet calculation
    optimal_bet = calculate_optimal_bet_size(
        wallet_balance, win_rate, confidence, market_regime
    )

    # If we're on a losing streak, reduce bet size
    if losing_streak > 0:
        # Apply a progressive reduction based on streak length
        reduction_factor = max(
            0.5, 1 - (losing_streak * 0.1)
        )  # Reduce by 10% per loss, min 50%
        adjusted_bet = optimal_bet * reduction_factor

        logger.info(
            f"⚠️ On a losing streak ({losing_streak} losses) - reducing bet by {(1-reduction_factor)*100:.0f}%"
        )

        # Ensure minimum bet or return 0
        min_bet = 0.005  # BNB
        if adjusted_bet < min_bet:
            return 0

        return adjusted_bet

    return optimal_bet
