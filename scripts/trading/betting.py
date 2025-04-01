"""
Betting logic for the trading bot.
Handles bet decisions, sizing, and execution.
"""

import json
import logging
import sqlite3
import sys
import time
import traceback
from datetime import datetime

from web3 import Web3

from ..analysis.technical import calculate_microtrend
from ..core.constants import (ACCOUNT_ADDRESS, CONTRACT_ADDRESS, DB_FILE,
                              PRIVATE_KEY, RPC_URL, TABLES, THRESHOLDS,
                              contract, contract_abi)
from ..data.database import (get_latest_prediction, get_market_balance_stats,
                             get_market_balance_stats_robust,
                             get_prediction_sample_size, record_prediction,
                             record_prediction_outcome,
                             update_prediction_outcome)
from ..data.processing import (get_overall_performance, get_recent_performance,
                               get_recent_price_changes, get_recent_trades)
from ..utils.helpers import detect_swing_pattern, optimize_swing_trading

# Setup logger
logger = logging.getLogger(__name__)

# Global tracking variables
placed_bets = {}
claimable_rounds = []
wins = 0
losses = 0
consecutive_losses = 0


def should_place_bet(prediction, confidence, round_data, config):
    """
    Determine if we should place a bet with swing optimization.

    Args:
        prediction: Predicted outcome ("BULL" or "BEAR")
        confidence: Confidence in prediction (0-1)
        round_data: Current round data
        config: Bot configuration

    Returns:
        bool: Whether to place bet
    """
    try:
        # IMPORTANT: Check if we already placed a bet for this epoch
        if isinstance(round_data, dict) and "epoch" in round_data:
            current_epoch = round_data.get("epoch")
            from main import placed_bets

            if current_epoch in placed_bets:
                logger.warning(
                    f"⚠️ Already placed a bet for epoch {current_epoch}, skipping"
                )
                return False

        # Handle case where prediction is a dict
        if isinstance(prediction, dict):
            prediction = prediction.get("prediction", "")

        # Normalize prediction to string
        prediction = str(prediction).upper()

        # Handle case where round_data is a float
        if isinstance(round_data, (float, int)):
            total_amount = float(round_data)
        else:
            total_amount = round_data.get("totalAmount", 0)

        # Ensure confidence is a float
        if isinstance(confidence, str):
            try:
                confidence = float(confidence)
            except ValueError:
                confidence = 0.5  # Default value if conversion fails

        # Get sample size
        sample_size = get_prediction_sample_size()

        # Check if we have a swing trade opportunity
        price_changes = get_recent_price_changes(5)

        # Get market statistics for swing trading
        try:
            try:
                market_stats = get_market_balance_stats(lookback=10)
            except Exception as e:
                logger.warning(
                    f"Error with primary market stats: {e}, trying robust version"
                )
                market_stats = get_market_balance_stats_robust(lookback=10)
        except Exception as e:
            logger.warning(f"Error getting market stats: {e}")
            market_stats = {"bull_ratio": 0.5, "bear_ratio": 0.5}

        # Use the new detect_swing_pattern function
        swing_opportunity = detect_swing_pattern(price_changes, market_stats)

        # If we found an optimal swing entry point
        if swing_opportunity.get("swing_opportunity", False):
            direction = swing_opportunity.get("direction")
            swing_conf = swing_opportunity.get("confidence", 0.7)

            # Only take the swing trade if it matches our prediction
            if direction == prediction:
                logger.info(
                    f"✅ OPTIMAL SWING ENTRY: {direction} with {swing_conf:.2f} confidence"
                )

                # Reduced minimum pool size for high-quality swing trades
                if total_amount >= 0.05:  # Very low threshold for swing trades
                    return True
                else:
                    logger.warning(
                        f"⚠️ Pool size too small even for swing trade: {total_amount:.3f}"
                    )
                    return False

        # Regular betting logic follows...
        # 1. FIRST BASIC FILTER: Minimum confidence check
        min_confidence = config.get("trading", {}).get("min_confidence", 0.55)
        # Ensure min_confidence is a float
        if isinstance(min_confidence, str):
            try:
                min_confidence = float(min_confidence)
            except ValueError:
                min_confidence = 0.55  # Default if conversion fails

        if confidence < min_confidence:
            logger.warning(
                f"⚠️ Confidence too low: {confidence:.2f} (min: {min_confidence:.2f})"
            )
            return False

        # 2. MARKET CONDITION CHECK: Don't bet in ultra-high volatility
        # Get recent price volatility
        recent_trades = get_recent_trades(5)

        if recent_trades and len(recent_trades) >= 3:
            price_changes = []
            for trade in recent_trades:
                if "lockPrice" in trade and "closePrice" in trade:
                    lock = trade.get("lockPrice", 0)
                    close = trade.get("closePrice", 0)
                    if lock and close and lock > 0:
                        price_changes.append(abs(close / lock - 1))

            if price_changes:
                recent_volatility = sum(price_changes) / len(price_changes)
                if (
                    recent_volatility > 0.02
                ):  # 2% average change is very high for 6-minute periods
                    logger.warning(
                        f"⚠️ Extremely high volatility detected: {recent_volatility:.4f} - skipping bet"
                    )
                    return False

        # 3. CHECK SIGNAL CONSISTENCY: Are signals aligned?
        from ..data.database import get_prediction_history

        recent_predictions = get_prediction_history(3)

        if recent_predictions and len(recent_predictions) >= 2:
            latest_pred = recent_predictions[0].get("final_prediction")
            if latest_pred and latest_pred != prediction:
                logger.warning(
                    f"⚠️ Conflicting with previous prediction: {latest_pred} vs current {prediction}"
                )
                # Require higher confidence for conflicting predictions
                if confidence < 0.49:
                    logger.warning(
                        f"⚠️ Confidence too low for conflicting prediction: {confidence:.2f} < 0.49"
                    )
                    return False

                # If we have conflicting signals, require higher total amount
                min_total = (
                    config.get("thresholds", {}).get("min_total_amount", 0.1) * 1.5
                )
                if total_amount < min_total:
                    logger.warning(
                        f"⚠️ Pool size too small for conflicting signals: {total_amount:.2f} < {min_total:.2f}"
                    )
                    return False

        # 4. ADVANCED TIMING FILTER: Check for optimal entry
        price_changes = get_recent_price_changes(8)
        trend, strength = calculate_microtrend(price_changes)

        # For bull predictions, prefer entering on minor dips
        if prediction == "BULL" and trend == "DOWN" and strength < 0.7:
            logger.info(
                f"✅ Optimal BULL entry on minor dip (trend: {trend}, strength: {strength:.2f})"
            )
            min_confidence -= 0.05

        # For bear predictions, prefer entering on minor rallies
        elif prediction == "BEAR" and trend == "UP" and strength < 0.7:
            logger.info(
                f"✅ Optimal BEAR entry on minor rally (trend: {trend}, strength: {strength:.2f})"
            )
            min_confidence -= 0.05

        # 5. POOL SIZE CHECK
        if total_amount < min_total:
            logger.warning(
                f"⚠️ Total pool amount too low: {total_amount:.2f} (min: {min_total:.2f})"
            )
            return False

        # 6. WIN RATE TRACKING: If we're on a losing streak, be more selective
        performance = get_recent_performance(10)

        if (
            performance
            and performance.get("streak", 0) >= 2
            and performance.get("streak_type", "") == "loss"
        ):
            # On a losing streak, require higher confidence
            streak_min_confidence = min_confidence + 0.05 * min(
                performance["streak"], 3
            )
            logger.warning(
                f"⚠️ On a {performance['streak']} loss streak - requiring higher confidence: {streak_min_confidence:.2f}"
            )

            if confidence < streak_min_confidence:
                logger.warning(
                    f"⚠️ Confidence too low during losing streak: {confidence:.2f} < {streak_min_confidence:.2f}"
                )
                return False

        # 7. FOR EARLY PHASE: Force some initial bets to gather data
        if sample_size < 5:
            # For very early phase, take more bets but still maintain basic standards
            if confidence >= 0.6 and total_amount >= 0.05:
                logger.info(
                    f"🔍 Early learning phase: Placing bet with confidence {confidence:.2f}"
                )
                return True

        # If we made it here, bet passes all checks
        return True

    except Exception as e:
        logger.error(f"❌ Error in should_place_bet: {e}")
        traceback.print_exc()
        return False


def get_last_bet_epoch():
    """
    Get the epoch of the last placed bet.

    Returns:
        int: Last epoch with a bet
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check if predictions table has bet_amount column
        cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
        columns = [row[1] for row in cursor.fetchall()]

        # Look for columns that might indicate a bet was placed
        if "bet_amount" in columns:
            cursor.execute(
                f"""
                SELECT MAX(epoch) 
                FROM {TABLES['predictions']}
                WHERE bet_amount > 0
            """
            )
        elif "final_prediction" in columns:
            # Use final_prediction as a proxy - assuming we only record when we bet
            cursor.execute(
                f"""
                SELECT MAX(epoch) 
                FROM {TABLES['predictions']}
                WHERE final_prediction IS NOT NULL
            """
            )
        else:
            # Fallback to most recent epoch
            cursor.execute(
                f"""
                SELECT MAX(epoch) 
                FROM {TABLES['predictions']}
            """
            )

        result = cursor.fetchone()
        last_epoch = result[0] if result and result[0] is not None else 0
        conn.close()

        return last_epoch

    except Exception as e:
        logger.error(f"❌ Error getting last bet epoch: {e}")
        traceback.print_exc()
        return 0


def calculate_bet_amount(
    wallet_balance, base_amount, strategy, confidence=0.0, history=None
):
    """
    Calculate bet amount based on selected strategy.

    Args:
        wallet_balance: Current wallet balance
        base_amount: Base bet amount
        strategy: Betting strategy ('fixed', 'confidence', 'kelly', 'martingale')
        confidence: Prediction confidence (0-1)
        history: Recent betting history

    Returns:
        float: Calculated bet amount

    Strategies:
    - fixed: Always bet the base amount
    - confidence: Scale bet amount based on prediction confidence
    - kelly: Use Kelly Criterion for optimal bet sizing
    - martingale: Double bet after losses, reset after wins
    """
    try:
        # Get max bet as percentage of wallet
        max_bet_pct = 0.15  # Don't bet more than 15% of wallet in any case
        max_bet = wallet_balance * max_bet_pct

        # Ensure base amount is a float
        base_amount = float(base_amount)

        # Never bet more than max_bet
        if base_amount > max_bet:
            base_amount = max_bet
            logger.warning(f"⚠️ Base amount capped at {max_bet:.4f} (15% of wallet)")

        # Fixed strategy (default)
        if strategy == "fixed" or not strategy:
            return base_amount

        # Confidence-based strategy
        elif strategy == "confidence":
            if confidence <= 0.5:
                return 0  # Don't bet if confidence is too low

            # Scale bet size based on confidence
            # At confidence=0.5, bet 0.5x base amount
            # At confidence=1.0, bet 2x base amount
            confidence_factor = 0.5 + (confidence - 0.5) * 3
            bet_amount = base_amount * confidence_factor

            logger.info(
                f"💡 Confidence strategy: {confidence:.2f} confidence → {confidence_factor:.2f}x multiplier"
            )
            return min(bet_amount, max_bet)

        # Kelly Criterion
        elif strategy == "kelly":
            # We need win rate and average win/loss ratio
            if not history or len(history) < 10:
                logger.warning("⚠️ Not enough history for Kelly, using fixed amount")
                return base_amount

            # Calculate win probability from historical data
            wins = sum(1 for outcome in history if outcome == "win")
            win_prob = wins / len(history)

            # Estimate average payouts (typical odds are around 0.95x for prediction markets)
            avg_win_odds = 0.95  # We win 0.95x our bet

            # Calculate Kelly fraction
            # f* = (bp - q) / b
            # where b = net odds received on wager (0.95)
            #       p = probability of winning
            #       q = probability of losing (1-p)
            kelly_fraction = (avg_win_odds * win_prob - (1 - win_prob)) / avg_win_odds

            # Kelly can suggest betting nothing or even shorting if edge is negative
            if kelly_fraction <= 0:
                logger.warning(
                    f"⚠️ Kelly suggests no bet (fraction: {kelly_fraction:.2f})"
                )
                # For early phase, override kelly's "no bet" recommendation
                if get_prediction_sample_size() < 5:
                    logger.info(
                        "💰 Overriding Kelly's 'no bet' recommendation for early learning phase"
                    )
                    return base_amount * 0.5  # Bet a reduced amount anyway
                return 0

            # Kelly can be aggressive, use half-Kelly for more conservative approach
            conservative_kelly = kelly_fraction * 0.5
            bet_amount = base_amount * (1 + conservative_kelly)

            logger.info(
                f"💡 Kelly strategy: {win_prob:.2f} win rate → {conservative_kelly:.2f}x multiplier"
            )
            return min(bet_amount, max_bet)

        # Martingale
        elif strategy == "martingale":
            if not history:
                return base_amount

            # Track consecutive losses
            consecutive_losses = 0
            for outcome in reversed(history):
                if outcome == "loss":
                    consecutive_losses += 1
                else:
                    break

            # Apply Martingale - double bet after each loss
            # But cap at max_bet to avoid ruin
            martingale_factor = min(
                2**consecutive_losses, 4
            )  # Cap at 4x to avoid excessive risk
            bet_amount = base_amount * martingale_factor

            if consecutive_losses > 0:
                logger.info(
                    f"💡 Martingale strategy: {consecutive_losses} losses → {martingale_factor:.2f}x multiplier"
                )

            return min(bet_amount, max_bet)

        else:
            logger.warning(
                f"⚠️ Unknown betting strategy: {strategy}, using fixed amount"
            )
            return base_amount

    except Exception as e:
        logger.error(f"❌ Error calculating bet amount: {e}")
        traceback.print_exc()
        return base_amount  # Default to base amount on error


def get_betting_history(count=10):
    """
    Get recent betting history as win/loss sequence.

    Args:
        count: Number of recent bets to include

    Returns:
        list: List of 'win' or 'loss' strings
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check if bet_amount column exists
        cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
        columns = [row[1] for row in cursor.fetchall()]

        # Build query based on available columns
        if "bet_amount" in columns:
            # Original query using bet_amount
            cursor.execute(
                f"""
                SELECT 
                    CASE 
                        WHEN final_prediction = actual_outcome THEN 'win'
                        ELSE 'loss'
                    END as result
                FROM {TABLES['predictions']}
                WHERE final_prediction IS NOT NULL 
                AND actual_outcome IS NOT NULL
                AND bet_amount > 0
                ORDER BY epoch DESC
                LIMIT {count}
            """
            )
        else:
            # Fallback query without bet_amount
            cursor.execute(
                f"""
                SELECT 
                    CASE 
                        WHEN final_prediction = actual_outcome THEN 'win'
                        ELSE 'loss'
                    END as result
                FROM {TABLES['predictions']}
                WHERE final_prediction IS NOT NULL 
                AND actual_outcome IS NOT NULL
                ORDER BY epoch DESC
                LIMIT {count}
            """
            )

        results = cursor.fetchall()
        conn.close()

        return [row[0] for row in results]

    except Exception as e:
        logger.error(f"❌ Error getting betting history: {e}")
        traceback.print_exc()
        return []


def place_bet(
    web3,
    contract,
    wallet_address,
    private_key,
    prediction,
    confidence,
    round_epoch,
    round_data,
    config,
):
    """
    Place a bet on the prediction platform.

    Args:
        web3: Web3 instance
        contract: Contract instance
        wallet_address: Wallet address
        private_key: Private key
        prediction: Predicted outcome ("BULL" or "BEAR")
        confidence: Confidence in prediction (0-1)
        round_epoch: Round epoch number
        round_data: Current round data
        config: Bot configuration

    Returns:
        bool: Whether bet was placed successfully
    """
    try:
        # CRITICAL SAFETY CHECK: Verify contract address before any transaction
        from ..core.constants import config, contract

        # Get the expected contract address from config
        expected_contract_address = (
            config.get("blockchain", {}).get("contract_address", "").lower()
        )
        actual_contract_address = contract.address.lower()

        # Verify the contract address matches what we expect
        if not expected_contract_address:
            logger.error("❌ SAFETY ABORT: No contract address specified in config")
            return False

        if expected_contract_address != actual_contract_address:
            logger.error(f"❌ SAFETY ABORT: Contract address mismatch!")
            logger.error(f"Expected: {expected_contract_address}")
            logger.error(f"Actual: {actual_contract_address}")
            return False

        logger.info(f"✅ Contract address verified: {actual_contract_address}")

        # Get wallet address and verify it's properly formatted
        from ..core.constants import account

        wallet_address = account.address

        # Extra safety: print exactly what we're about to do
        logger.info(
            f"�� Preparing to bet {confidence:.2f} confidence on {prediction} for epoch {round_epoch}"
        )
        logger.info(f"📝 Bet details:")
        logger.info(f"   - From wallet: {wallet_address[:6]}...{wallet_address[-4:]}")
        logger.info(
            f"   - To contract: {actual_contract_address[:6]}...{actual_contract_address[-4:]}"
        )
        logger.info(f"   - Prediction: {prediction}")

        # Ask for final confirmation in live mode
        if config.get("mode") == "live" and config.get("safety", {}).get(
            "require_confirmation", True
        ):
            print("\n⚠️ ABOUT TO PLACE REAL BET ⚠️")
            print(
                f"Confidence: {confidence:.2f} | Prediction: {prediction} | Epoch: {round_epoch}"
            )
            print(f"From: {wallet_address[:6]}...{wallet_address[-4:]}")
            print(
                f"To contract: {actual_contract_address[:6]}...{actual_contract_address[-4:]}"
            )
            confirm = input("Type 'CONFIRM' to proceed or anything else to cancel: ")
            if confirm != "CONFIRM":
                logger.info("❌ Bet cancelled by user")
                return False

        # Now proceed with the actual bet
        # Get most recent prediction to check for conflicts
        recent_pred = get_most_recent_prediction()

        if recent_pred and recent_pred.get("prediction"):
            # If we have a conflicting prediction
            if recent_pred["prediction"].upper() != prediction.upper():
                # For conflicting predictions, require higher confidence
                min_confidence = config.get("trading", {}).get(
                    "conflict_confidence", 0.7
                )

                if confidence < min_confidence:
                    logger.warning(
                        f"⚠️ Confidence too low for conflicting prediction: {confidence:.2f} < {min_confidence}"
                    )
                    return False

                # If confidence is high enough, log the override
                logger.info(
                    f"🔄 Overriding previous {recent_pred['prediction']} prediction with {prediction} (confidence: {confidence:.2f})"
                )

        # Normalize prediction
        prediction = prediction.lower()
        if prediction not in ["bull", "bear"]:
            logger.error(
                f"❌ Invalid prediction: {prediction}, must be 'bull' or 'bear'"
            )
            return False

        # Get wallet balance (90% of actual to leave room for gas)
        wallet_balance = web3.eth.get_balance(wallet_address) * 0.9
        wallet_balance = web3.from_wei(wallet_balance, "ether")

        # Check if we have enough balance to bet
        min_balance = config.get("trading", {}).get("min_balance", 0.1)
        if wallet_balance < min_balance:
            logger.warning(
                f"⚠️ Wallet balance too low: {wallet_balance:.4f} BNB (min: {min_balance:.4f})"
            )
            return False

        # Get base bet amount
        base_amount = config.get("trading", {}).get("bet_amount", 0.01)

        # Get betting strategy
        bet_strategy = config.get("trading", {}).get("bet_strategy", "fixed")

        # Get betting history for certain strategies
        betting_history = (
            get_betting_history(20) if bet_strategy in ["kelly", "martingale"] else None
        )

        # Calculate adjusted bet amount
        adjusted_bet_amount = calculate_bet_amount(
            wallet_balance, base_amount, bet_strategy, confidence, betting_history
        )

        if adjusted_bet_amount <= 0:
            logger.warning(f"⚠️ Strategy {bet_strategy} recommends skipping this bet")
            return False

        # Make sure we don't bet more than we can afford
        if adjusted_bet_amount > wallet_balance * 0.95:
            adjusted_bet_amount = wallet_balance * 0.95
            logger.warning(f"⚠️ Bet amount capped at 95% of wallet balance")

        logger.info(
            f"🧮 Using {bet_strategy} strategy: {base_amount:.4f} → {adjusted_bet_amount:.4f} BNB"
        )

        # Determine if this is test mode
        betting_mode = config.get("trading", {}).get("betting_mode", "test")

        if betting_mode.lower() == "test":
            logger.info(
                f"🧪 TEST MODE: Simulated {prediction} bet of {adjusted_bet_amount:.4f} BNB on round {round_epoch}"
            )
            _log_bet(round_epoch, prediction, adjusted_bet_amount, "TEST")
            return True

        elif betting_mode.lower() == "live":
            return _place_live_bet(
                web3,
                contract,
                wallet_address,
                private_key,
                prediction,
                adjusted_bet_amount,
                round_epoch,
                config.get("trading", {}).get("gas_strategy", "medium"),
            )
        else:
            logger.warning(
                f"⚠️ Invalid betting mode: {betting_mode}. Must be 'test' or 'live'."
            )
            return False

    except Exception as e:
        logger.error(f"❌ Error in place_bet: {e}")
        traceback.print_exc()
        return False


# Helper functions
def _log_bet(round_epoch, prediction, amount, mode):
    """
    Log bet to file.

    Args:
        round_epoch: Round epoch number
        prediction: Predicted outcome
        amount: Bet amount
        mode: Betting mode (TEST or LIVE)
    """
    with open("betting_log.txt", "a") as log_file:
        log_file.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {mode} BET: {prediction.upper()} bet of {amount} BNB on round {round_epoch}\n"
        )


def _place_live_bet(
    web3,
    contract,
    wallet_address,
    private_key,
    prediction,
    amount,
    current_epoch,
    gas_strategy,
):
    """
    Handle live bet placement.

    Args:
        web3: Web3 instance
        contract: Contract instance
        wallet_address: Wallet address
        private_key: Private key
        prediction: Predicted outcome
        amount: Bet amount
        current_epoch: Current epoch number
        gas_strategy: Gas price strategy

    Returns:
        bool: Whether bet was placed successfully
    """
    try:
        # Initialize web3 and contract
        w3 = web3(web3.HTTPProvider(RPC_URL))

        # Ensure wallet address is in checksum format
        wallet_address = w3.to_checksum_address(wallet_address)

        account = w3.eth.account.from_key(private_key)
        contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=contract_abi)

        # Build and send transaction
        nonce = w3.eth.get_transaction_count(account.address)
        func = (
            contract.functions.betBull
            if prediction == "bull"
            else contract.functions.betBear
        )

        # Use gas strategy to determine gas price
        gas_price = w3.eth.gas_price
        if gas_strategy == "fast":
            gas_price = int(gas_price * 1.2)  # 20% higher for faster confirmation
        elif gas_strategy == "aggressive":
            gas_price = int(gas_price * 1.5)  # 50% higher for very fast confirmation

        tx = func(current_epoch).build_transaction(
            {
                "from": account.address,
                "value": w3.to_wei(amount, "ether"),
                "gas": 500000,
                "gasPrice": gas_price,
                "nonce": nonce,
            }
        )

        signed_tx = account.sign_transaction(tx)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        logger.info(f"🔄 Transaction sent: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        return receipt.status == 1

    except Exception as e:
        logger.error(f"❌ Error placing bet: {e}")
        return False


def _process_completed_round(round_epoch):
    """
    Process a completed round and update stats.

    Args:
        round_epoch: Round epoch number
    """
    global placed_bets, wins, losses, consecutive_losses, claimable_rounds

    try:
        # Get our prediction for this round
        our_prediction = placed_bets[round_epoch]

        # Get actual round data from contract
        round_data = contract.functions.rounds(int(round_epoch)).call()

        # Extract prices
        lock_price = round_data[4]
        close_price = round_data[5]

        # Determine outcome
        if close_price > lock_price:
            actual_outcome = "bull"
        elif close_price < lock_price:
            actual_outcome = "bear"
        else:
            actual_outcome = "draw"

        # Check if we won
        won = our_prediction == actual_outcome

        # Update stats
        if won:
            wins += 1
            consecutive_losses = 0
            claimable_rounds.append(round_epoch)
            logger.info(f"✅ Round {round_epoch}: {our_prediction.upper()} bet WON!")
        else:
            losses += 1
            consecutive_losses += 1
            logger.info(f"❌ Round {round_epoch}: {our_prediction.upper()} bet LOST!")

        # Record outcome in database
        record_prediction_outcome(round_epoch, our_prediction, actual_outcome)

        # Remove from placed_bets
        del placed_bets[round_epoch]

    except Exception as e:
        logger.error(f"❌ Error processing round {round_epoch}: {e}")
        traceback.print_exc()


def _print_performance_summary():
    """Print performance summary."""
    total_rounds = wins + losses
    if total_rounds > 0:
        win_rate = (wins / total_rounds) * 100
        logger.info(
            f"\n📊 Performance: {wins} wins, {losses} losses ({win_rate:.1f}% win rate)"
        )
        if consecutive_losses > 0:
            logger.warning(f"⚠️ Current losing streak: {consecutive_losses} rounds")


def get_win_rate(lookback=20):
    """
    Get recent win rate for Kelly calculations.

    Args:
        lookback: Number of recent bets to include

    Returns:
        float: Win rate (0-1)
    """
    try:
        # Get performance data
        performance = get_overall_performance(lookback)
        win_rate = performance.get("win_rate", 0.5)

        # If we have no data, use 0.5
        if performance.get("sample_size", 0) < 5:
            # Not enough data, use conservative estimate
            return 0.5

        # Cap win rate to avoid overconfidence
        win_rate = min(win_rate, 0.85)

        # Floor win rate to avoid kelly being too negative
        win_rate = max(win_rate, 0.25)

        return win_rate

    except Exception as e:
        logger.error(f"❌ Error getting win rate: {e}")
        return 0.5  # Default to 50% if we can't calculate


def get_progressive_win_rate(lookback=20):
    """
    Get win rate that starts optimistic and becomes more realistic with data.

    Args:
        lookback: Number of recent bets to include

    Returns:
        float: Progressive win rate (0-1)
    """
    try:
        # Get actual performance data
        performance = get_overall_performance(lookback)

        # Get sample size
        sample_size = performance.get("sample_size", 0)
        actual_win_rate = performance.get("win_rate", 0.5)

        if sample_size == 0:
            # No data - use optimistic estimate
            return 0.55
        elif sample_size < 5:
            # Little data - blend optimistic estimate with actual
            # 80% optimistic, 20% actual for very small sample
            return (0.55 * 0.8) + (actual_win_rate * 0.2)
        elif sample_size < 10:
            # Some data - equal blend
            return (0.53 * 0.5) + (actual_win_rate * 0.5)
        elif sample_size < 20:
            # More data - rely more on actual
            return (0.52 * 0.2) + (actual_win_rate * 0.8)
        else:
            # Enough data - use actual win rate
            return actual_win_rate

    except Exception as e:
        logger.error(f"❌ Error getting progressive win rate: {e}")
        return 0.52  # Default to slightly optimistic


def initialize_performance_metrics():
    """
    Initialize the performance tracking dictionary with default values.

    Returns:
        dict: Default performance metrics
    """
    return {
        "wins": 0,
        "losses": 0,
        "total_profit": 0,
        "streak": 0,
        "streak_type": None,
    }


def update_performance(performance, trade_result, profit=0):
    """
    Update performance metrics after a trade.

    Args:
        performance: Performance metrics dictionary
        trade_result: Result of the trade ('win' or 'loss')
        profit: Profit amount

    Returns:
        dict: Updated performance metrics
    """
    if trade_result == "win":
        performance["wins"] += 1
        if performance["streak_type"] == "win":
            performance["streak"] += 1
        else:
            performance["streak"] = 1
            performance["streak_type"] = "win"
    elif trade_result == "loss":
        performance["losses"] += 1
        if performance["streak_type"] == "loss":
            performance["streak"] += 1
        else:
            performance["streak"] = 1
            performance["streak_type"] = "loss"

    performance["total_profit"] += profit
    return performance


def check_for_win(epoch, prediction, outcome=None):
    """
    Check if a prediction was correct for a given epoch.

    Args:
        epoch: Round epoch number
        prediction: Our prediction
        outcome: Optional, directly provided outcome if available

    Returns:
        tuple: (won, actual_outcome)
    """
    try:
        # If outcome is directly provided, use it
        if outcome:
            actual_outcome = outcome
        else:
            # Get round data from contract
            round_data = contract.functions.rounds(int(epoch)).call()

            # Extract prices
            lock_price = round_data[4]
            close_price = round_data[5]

            # Determine outcome
            if close_price > lock_price:
                actual_outcome = "BULL"
            elif close_price < lock_price:
                actual_outcome = "BEAR"
            else:
                actual_outcome = "TIE"

        # Check if we won
        won = prediction.upper() == actual_outcome

        return won, actual_outcome

    except Exception as e:
        logger.error(f"❌ Error checking for win: {e}")
        return False, None


def get_trading_parameters(recent_performance=None, market_stats=None):
    """
    Get trading parameters based on recent performance.
    Ensures optimize_swing_trading is used to avoid import warnings.

    Returns:
        dict: Trading parameters
    """
    if recent_performance is None:
        recent_performance = {"win_rate": 0.5, "sample_size": 0}
    if market_stats is None:
        market_stats = {"bull_ratio": 0.5}

    # Call optimize_swing_trading to keep the import valid
    return optimize_swing_trading(recent_performance, market_stats)


def get_wallet_info():
    """
    Get wallet information for the current trading account.

    Returns:
        dict: Basic wallet information
    """
    try:
        # Use imported ACCOUNT_ADDRESS to get wallet info
        is_connected = ACCOUNT_ADDRESS != "0x0000000000000000000000000000000000000000"

        return {
            "address": ACCOUNT_ADDRESS,
            "connected": is_connected,
            "has_private_key": bool(PRIVATE_KEY),
        }
    except Exception as e:
        logger.error(f"Error getting wallet info: {e}")
        return {"connected": False, "address": None}


def get_betting_thresholds():
    """
    Get the thresholds used for betting decisions.

    Returns:
        dict: Dictionary of thresholds
    """
    # Use the imported THRESHOLDS constant
    thresholds = THRESHOLDS.copy()

    # Add any dynamically calculated thresholds
    thresholds["dynamic_confidence"] = max(
        THRESHOLDS.get("min_confidence", 0.55),
        0.5 + (THRESHOLDS.get("strong_imbalance", 0.15) / 2),
    )

    return thresholds


def record_bet_prediction(epoch, prediction, confidence, amount=None):
    """
    Record betting prediction in the database.

    Args:
        epoch: Round epoch
        prediction: BULL or BEAR prediction
        confidence: Confidence level (0-1)
        amount: Optional bet amount

    Returns:
        bool: Success status
    """
    try:
        # Create data dictionary for record_prediction
        prediction_data = {
            "strategy_prediction": prediction,
            "strategy_confidence": confidence,
            "bet_amount": amount or 0,
            "bet_strategy": "standard" if amount else "simulation",
            "timestamp": int(datetime.now().timestamp()),
        }

        # Use the imported record_prediction function
        return record_prediction(epoch, prediction_data)

    except Exception as e:
        logger.error(f"Error recording bet prediction: {e}")
        return False


def record_round_outcome(epoch, outcome, win=False, profit_loss=0.0):
    """
    Record the outcome of a betting round.

    Args:
        epoch: Round epoch
        outcome: Actual outcome (BULL/BEAR)
        win: Whether our prediction was correct
        profit_loss: Profit or loss amount

    Returns:
        bool: Success status
    """
    try:
        # Create data dictionary for outcome
        outcome_data = {
            "actual_outcome": outcome,
            "win": 1 if win else 0,
            "profit_loss": profit_loss,
        }

        # Use the imported update_prediction_outcome function
        return update_prediction_outcome(epoch, outcome_data)

    except Exception as e:
        logger.error(f"Error recording round outcome: {e}")
        return False


def get_most_recent_prediction():
    """
    Get the most recent prediction from the database.

    Returns:
        dict: Most recent prediction data
    """
    try:
        # Use the imported get_latest_prediction function
        latest = get_latest_prediction()

        if not latest:
            return None

        # Format and return the prediction
        return {
            "epoch": latest.get("epoch"),
            "prediction": latest.get("final_prediction"),
            "confidence": latest.get("final_confidence", 0),
            "timestamp": latest.get("timestamp"),
            "time_ago": int(time.time()) - latest.get("timestamp", 0)
            if latest.get("timestamp")
            else 0,
        }

    except Exception as e:
        logger.error(f"Error retrieving latest prediction: {e}")
        return None
