#!/usr/bin/env python3
"""
Main entry point for the cryptocurrency prediction bot.
"""

import builtins
import logging
import os
import sqlite3
import sys
import time
import traceback
import warnings
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

import pdb

# Import from configuration module
from configuration import config
from models.func_ai_strategy import AIStrategy
from models.func_hybrid import hybrid_prediction
# Import ML models
from models.func_rf import train_model
from models.func_ta import TechnicalAnalysis
# Import analysis functions
from scripts.analysis.market import (bootstrap_market_data,
                                     get_historical_prices,
                                     get_market_direction,
                                     get_market_sentiment)
from scripts.analysis.regime import detect_market_regime
from scripts.analysis.technical import get_technical_prediction
# Import scripts core functionality
from scripts.core.constants import (DB_FILE, STRATEGY_WEIGHTS, TABLES,
                                    THRESHOLDS, contract, web3)
# Import blockchain data functionality
from scripts.data.blockchain import (get_current_epoch,
                                     get_enriched_round_data, get_round_data,
                                     get_time_until_lock,
                                     get_time_until_round_end)
# Import data related functionality
from scripts.data.database import (get_performance_stats, get_prediction,
                                   get_recent_trades, get_test_balance,
                                   initialize_database, record_bet,
                                   record_trade, update_prediction_outcome)
from scripts.diagnosis.check_data_flow import start_monitoring
from scripts.prediction.filtering import filter_signals
# Import prediction handling
from scripts.prediction.handler import PredictionHandler
from scripts.prediction.strategy_selector import select_optimal_strategy
# Import trading functionality
from scripts.trading.betting import check_for_win, place_bet, should_place_bet
from scripts.trading.money import calculate_optimal_bet_size

# Start monitoring (checks every 5 minutes by default)
start_monitoring()


# Initialize placed_bets_tracker in builtins
builtins.placed_bets_tracker = {}

# Global tracking variables
placed_bets = builtins.placed_bets_tracker
claimable_rounds = []
wins = 0
losses = 0
consecutive_losses = 0
last_bull_pressure = 0.5
prediction_handler = None
balance = 0

# Add global AI strategy instance
ai_strategy = None


def calculate_sum(a, b):
    pdb.set_trace()  # Break here to start debugging
    c = a + b
    return c


# In the main function, you call it like:
def main():
    print("Starting calculation...")
    result = calculate_sum(
        10, 20
    )  # Now this works because the function accepts arguments
    print(f"Result is: {result}")


# Setup logging
def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.join("data", "logs")
    os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, "trading_bot.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def select_mode():
    """Allow user to select mode (live or test)."""
    while True:
        print("\n🔄 Select mode:")
        print("1. Live mode (real betting)")
        print("2. Test mode (simulated betting)")

        try:
            choice = input("Enter your choice (1-2): ").strip()

            if choice == "1":
                # Confirm live mode to prevent accidental selection
                confirm = (
                    input(
                        "⚠️ WARNING: Live mode will place REAL bets with REAL money. Are you sure? (yes/no): "
                    )
                    .strip()
                    .lower()
                )
                if confirm == "yes":
                    return "live"
                else:
                    print("Live mode cancelled. Defaulting to test mode.")
                    return "test"
            elif choice == "2":
                return "test"
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except Exception as e:
            print(f"Error: {e}")
            return "test"


def get_wallet_balance(mode="test"):
    """
    Get wallet balance based on mode.

    Args:
        mode: "live" to get real blockchain balance, "test" for simulated balance

    Returns:
        float: Wallet balance in BNB
    """
    try:
        if mode == "live":
            # Get real balance from blockchain
            from web3 import Web3

            from scripts.core.constants import config, web3

            # Get wallet address from config
            wallet_address = config.get("wallet", {}).get("address")
            if not wallet_address:
                print("⚠️ No wallet address configured, using test balance")
                return get_test_balance()

            # Convert address to checksum format
            try:
                checksummed_address = Web3.to_checksum_address(wallet_address)
                print(f"✅ Using checksummed wallet address: {checksummed_address}")
            except Exception as e:
                print(f"❌ Error converting wallet address to checksum format: {e}")
                print(f"⚠️ Falling back to test balance")
                return get_test_balance()

            # Get balance from blockchain
            try:
                balance_wei = web3.eth.get_balance(checksummed_address)
                balance_bnb = Web3.from_wei(balance_wei, "ether")

                print(f"💰 Live wallet balance: {balance_bnb:.6f} BNB")
                return float(balance_bnb)
            except Exception as e:
                print(f"❌ Error getting blockchain balance: {e}")
                print(f"⚠️ Falling back to test balance")
                return get_test_balance()
        else:
            # Return test balance
            return get_test_balance()  # Use database-stored test balance
    except Exception as e:
        print(f"❌ Error getting wallet balance: {e}")
        traceback.print_exc()
        return get_test_balance()  # Fallback to test balance


def check_claimable_rounds():
    """Check for claimable rounds."""
    # In your actual implementation, this would check the blockchain
    return []


def claim_rewards(claimable):
    """Claim rewards for the specified rounds."""
    # In your actual implementation, this would interact with the contract
    return True


def get_optimal_gas_price(web3_instance, strategy="medium"):
    """
    Get gas price based on strategy.
    """
    try:
        # Get base gas price
        base_gas_price = web3_instance.eth.gas_price

        # Apply multiplier based on strategy
        multiplier = 1.0
        if strategy == "slow":
            multiplier = 0.9
        elif strategy == "fast":
            multiplier = 1.2
        elif strategy == "aggressive":
            multiplier = 1.5

        return int(base_gas_price * multiplier)
    except Exception as e:
        print(f"Error getting gas price: {e}")
        return 5 * 10**9  # Default to 5 Gwei


def simulate_bet(epoch, prediction, amount):
    """
    Simulate placing a bet in test mode (no blockchain transaction).

    Args:
        epoch: Round epoch
        prediction: BULL or BEAR
        amount: Bet amount in BNB

    Returns:
        bool: Always True for simulation success
    """
    try:
        # Create bet record with proper datetime
        current_time = datetime.now()
        timestamp = int(current_time.timestamp())

        bet_data = {
            "epoch": epoch,
            "prediction": prediction,
            "amount": amount,
            "strategy": "test_mode",
            "timestamp": timestamp,
            "datetime": current_time.strftime(
                "%Y-%m-%d %H:%M:%S"
            ),  # Use datetime for formatting
            "gas_price": 0,  # No gas used in test mode
            "tx_hash": "test_tx_" + str(epoch),  # Simulated transaction hash
        }

        # Record to database
        record_bet(bet_data)

        # Add to global tracking
        global placed_bets
        placed_bets[epoch] = {
            "prediction": prediction,
            "amount": amount,
            "timestamp": int(time.time()),
            "simulated": True,
        }

        return True

    except Exception as e:
        print(f"❌ Error simulating bet: {e}")
        traceback.print_exc()
        return False


def simulate_round_result(epoch, prediction, amount):
    """
    Simulate the result of a bet when a round completes in test mode.

    Args:
        epoch: Round epoch
        prediction: Our prediction (BULL/BEAR)
        amount: Bet amount

    Returns:
        dict: Simulated round result
    """
    try:
        # Get the actual round outcome
        round_data = get_round_data(epoch)
        if not round_data or not round_data.get("outcome"):
            return None  # Round not complete yet

        # Calculate profit/loss
        outcome = round_data["outcome"]
        won = prediction == outcome

        # Calculate profit (in test mode use 1.9x multiplier to simulate house edge)
        profit_loss = amount * 0.9 if won else -amount

        # Create result data
        result = {
            "epoch": epoch,
            "prediction": prediction,
            "outcome": outcome,
            "amount": amount,
            "won": won,
            "profit_loss": profit_loss,
            "closePrice": round_data.get("closePrice", 0),
            "lockPrice": round_data.get("lockPrice", 0),
        }

        # Update statistics
        global wins, losses, consecutive_losses
        if won:
            wins += 1
            consecutive_losses = 0
        else:
            losses += 1
            consecutive_losses += 1

        return result

    except Exception as e:
        print(f"❌ Error simulating round result: {e}")
        traceback.print_exc()
        return None


def handle_round_completion(epoch, round_result):
    """
    Handle round completion and update prediction outcomes.
    Works for both real and simulated bets.

    Args:
        epoch: Round epoch
        round_result: Dict with round result data
    """
    try:
        # Get the prediction from our database handler
        prediction_data = get_prediction(epoch)

        # If we have a prediction for this epoch
        if prediction_data and prediction_data.get("final_prediction"):
            final_prediction = prediction_data.get("final_prediction")

            # Verify the win using check_for_win function
            is_win, actual_outcome = check_for_win(
                epoch, final_prediction, round_result["outcome"]
            )
            if is_win != round_result["won"]:
                print(f"⚠️ Win verification mismatch for epoch {epoch}")

            # Update outcome in database
            update_prediction_outcome(
                epoch=epoch,
                outcome=round_result["outcome"],
                win=1 if round_result["won"] else 0,
                profit_loss=round_result.get("profit_loss", 0),
            )

            # Print outcome
            result_emoji = "✅" if round_result["outcome"] == final_prediction else "❌"
            print(
                f"{result_emoji} Round {epoch} completed - Prediction: {final_prediction}, Outcome: {round_result['outcome']}"
            )

            # Record as a trade for performance tracking
            trade_data = {
                "bullAmount": prediction_data.get("bullAmount", 0),
                "bearAmount": prediction_data.get("bearAmount", 0),
                "totalAmount": prediction_data.get("totalAmount", 0),
                "bullRatio": prediction_data.get("bullRatio", 0),
                "bearRatio": prediction_data.get("bearRatio", 0),
                "lockPrice": prediction_data.get("lockPrice", 0),
                "closePrice": round_result["closePrice"],
                "outcome": round_result["outcome"],
                "prediction": final_prediction,
                "amount": round_result.get("amount", 0),
                "profit_loss": round_result["profit_loss"],
                "win": 1 if round_result["outcome"] == final_prediction else 0,
                "test_mode": round_result.get("simulated", False),
            }
            record_trade(epoch, trade_data)

            # Update wallet balance based on mode
            mode = "live" if "live" in locals() else "test"
            if mode == "live":
                # For live mode, get actual blockchain balance
                balance = get_wallet_balance(mode="live")
            else:
                # For test mode, update based on recorded profit/loss
                balance = get_test_balance()
        else:
            print(f"⚠️ No prediction found for epoch {epoch}")

    except Exception as e:
        print(f"❌ Error handling round completion: {e}")
        traceback.print_exc()


def is_betting_time(seconds_until_lock):
    """
    Determine if it's an optimal time to place a bet.

    Args:
        seconds_until_lock: Seconds remaining until betting closes

    Returns:
        bool: True if optimal betting time, False otherwise
    """
    # Get optimal betting window parameters from config
    min_seconds = (
        config.get("timing", {})
        .get("optimal_betting_seconds_before_lock", {})
        .get("min", 30)
    )
    max_seconds = (
        config.get("timing", {})
        .get("optimal_betting_seconds_before_lock", {})
        .get("max", 180)
    )

    return max_seconds >= seconds_until_lock >= min_seconds


def display_round_info(epoch):
    """
    Display information about the current round.

    Args:
        epoch: Current round epoch
    """
    try:
        round_data = get_enriched_round_data(epoch)
        if round_data:
            print(f"\n📊 Round {epoch} Data:")
            print(
                f"   Bull/Bear Ratio: {round_data.get('bullRatio', 0):.2f}/{round_data.get('bearRatio', 0):.2f}"
            )
            print(f"   Total Amount: {round_data.get('totalAmount', 0):.4f} BNB")

    except Exception as e:
        print(f"❌ Error displaying round info: {e}")


def print_stats():
    """Print current performance statistics."""
    print(f"\n📊 Performance Stats:")

    # Get detailed performance stats from database
    performance = get_performance_stats()

    # Print stats from database and global counters
    print(
        f"   Wins: {performance.get('wins', wins)}, Losses: {performance.get('losses', losses)}"
    )
    win_rate = performance.get("win_rate", 0)
    print(f"   Win Rate: {win_rate:.2%}")
    print(f"   Profit/Loss: {performance.get('profit_loss', 0):.6f} BNB")
    print(f"   Consecutive Losses: {consecutive_losses}")
    print(f"   Balance: {balance:.6f} BNB")
    print(f"   Total Bets: {performance.get('total_bets', 0)}")

    # Display recent trades
    display_recent_trades(3)


def display_recent_trades(limit=5):
    """Display recent trading history."""
    try:
        print("\n📜 Recent Trading History:")
        trades = get_recent_trades(limit)

        if not trades:
            print("   No recent trades found")
            return

        for trade in trades:
            epoch = trade.get("epoch", "Unknown")
            prediction = trade.get("prediction", "Unknown")
            outcome = trade.get("outcome", "Unknown")
            profit_loss = trade.get("profit_loss", 0)
            win = trade.get("win", 0)

            result_emoji = "✅" if win else "❌"
            profit_sign = "+" if profit_loss >= 0 else ""

            print(
                f"   {result_emoji} Epoch {epoch}: {prediction} vs {outcome} → {profit_sign}{profit_loss:.6f} BNB"
            )

    except Exception as e:
        print(f"❌ Error displaying recent trades: {e}")


def check_database_setup():
    """Verify database tables are set up correctly."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check if key tables exist
        for table_name in TABLES.values():
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
            )
            if not cursor.fetchone():
                print(f"⚠️ Table {table_name} not found, initializing database...")
                initialize_database()
                break

        conn.close()
        return True
    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
        traceback.print_exc()
        return False


def verify_contract_connection():
    """Verify connection to the prediction contract."""
    try:
        if contract:
            # Get current epoch from contract to verify connection
            current_epoch = contract.functions.currentEpoch().call()
            print(f"✅ Contract connected successfully. Current epoch: {current_epoch}")
            return True
        else:
            print("❌ Contract not initialized")
            return False
    except Exception as e:
        print(f"❌ Contract connection error: {e}")
        traceback.print_exc()
        return False


def check_betting_opportunity(confidence, round_data):
    """
    Check if this is a particularly strong betting opportunity.

    Args:
        confidence: Prediction confidence (0-1)
        round_data: Current round data

    Returns:
        bool: True if strong opportunity, False otherwise
    """
    try:
        # Check against high confidence threshold
        if confidence > THRESHOLDS.get("high_confidence", 0.80):
            print(f"⭐ High confidence signal: {confidence:.2f}")
            return True

        # Check for strong market imbalance
        bull_ratio = round_data.get("bullRatio", 0.5)
        bear_ratio = round_data.get("bearRatio", 0.5)

        imbalance = abs(bull_ratio - bear_ratio)
        if imbalance > THRESHOLDS.get("strong_imbalance", 0.15):
            print(f"⭐ Strong market imbalance: {imbalance:.2f}")
            return True

        return False
    except Exception as e:
        print(f"Error checking betting opportunity: {e}")
        return False


def calculate_weighted_confidence(all_predictions):
    """
    Calculate weighted confidence based on multiple predictions.

    Args:
        all_predictions: Dict of predictions from different strategies

    Returns:
        float: Weighted confidence value
    """
    try:
        if not all_predictions:
            return 0.51  # Default slightly bullish

        # Get weights for each strategy
        weights = {}
        total_weight = 0

        for strategy, pred_data in all_predictions.items():
            # Skip any invalid predictions
            if (
                not pred_data
                or "prediction" not in pred_data
                or "confidence" not in pred_data
            ):
                continue

            # Use quality score if available (from filtering), otherwise use standard weight
            if "quality" in pred_data:
                weights[strategy] = pred_data["quality"]
            else:
                # Use a normalized confidence as weight (gives more weight to high confidence)
                weights[strategy] = max(0.5, pred_data["confidence"])

            total_weight += weights[strategy]

        # No valid predictions with weights
        if total_weight <= 0:
            return 0.51

        # Normalize weights
        for strategy in weights:
            weights[strategy] /= total_weight

        # Calculate weighted confidence for each prediction direction (BULL/BEAR)
        bull_weight = 0
        bear_weight = 0

        for strategy, pred_data in all_predictions.items():
            if strategy not in weights:
                continue

            prediction = pred_data["prediction"]
            confidence = pred_data["confidence"]
            strategy_weight = weights[strategy]

            # Adjust for predictions near 0.5 (neutral)
            adjusted_confidence = confidence
            if 0.48 <= confidence <= 0.52:
                adjusted_confidence = 0.5  # Treat as neutral

            # Aggregate weights by direction
            if prediction == "BULL":
                bull_weight += strategy_weight * adjusted_confidence
            elif prediction == "BEAR":
                bear_weight += strategy_weight * (
                    1 - adjusted_confidence
                )  # Convert to same scale

        # Calculate final confidence
        final_confidence = 0.5
        if bull_weight > bear_weight:
            final_confidence = 0.5 + (bull_weight - bear_weight)
        else:
            final_confidence = 0.5 - (bear_weight - bull_weight)

        # If we have any single prediction with very high confidence (>0.75)
        # ensure the weighted result is at least 0.6
        for pred_data in all_predictions.values():
            if pred_data.get("confidence", 0) > 0.75:
                if pred_data["prediction"] == "BULL" and final_confidence < 0.6:
                    final_confidence = max(final_confidence, 0.6)
                elif pred_data["prediction"] == "BEAR" and final_confidence > 0.4:
                    final_confidence = min(final_confidence, 0.4)
                    # Convert to same scale for output
                    final_confidence = 1 - final_confidence

        # Limit to range [0, 1]
        final_confidence = max(0, min(1, final_confidence))

        return final_confidence

    except Exception as e:
        print(f"❌ Error calculating weighted confidence: {e}")
        return 0.51  # Fallback to slightly bullish


def analyze_market_context(round_data):
    """
    Analyze market context for better prediction.

    Args:
        round_data: Current round data

    Returns:
        dict: Market context
    """
    try:
        from scripts.analysis.market import (get_market_direction,
                                             get_market_sentiment)
        from scripts.analysis.technical import get_technical_prediction

        # Initialize result
        market_context = {}

        # Get market sentiment
        sentiment, sentiment_strength = get_market_sentiment(round_data)
        market_context["market_sentiment"] = sentiment
        market_context["sentiment_strength"] = sentiment_strength

        # Get market direction (with specified lookback)
        direction, direction_strength = get_market_direction(lookback=10)
        market_context["market_direction"] = direction
        market_context["direction_strength"] = direction_strength

        # Get technical prediction
        tech_pred, tech_conf = get_technical_prediction(get_historical_prices(20))
        market_context["technical_prediction"] = tech_pred
        market_context["technical_confidence"] = tech_conf

        return market_context

    except Exception as e:
        print(f"❌ Error analyzing market context: {e}")
        return {}


def get_high_confidence_predictions(all_predictions, min_confidence=0.75):
    """
    Get high confidence predictions from all strategies.

    Args:
        all_predictions: Dictionary of predictions from different strategies
        min_confidence: Minimum confidence threshold

    Returns:
        list: High confidence predictions
    """
    try:
        # Get market regime for filtering
        market_regime = detect_market_regime(get_historical_prices(20)).get(
            "regime", "unknown"
        )

        # Pass the market_regime parameter to filter_signals
        filtered_predictions = filter_signals(
            all_predictions, min_confidence, market_regime
        )

        return filtered_predictions

    except Exception as e:
        print(f"❌ Error getting high confidence predictions: {e}")
        traceback.print_exc()
        return []


def select_best_strategy(market_regime, round_data):
    """
    Select the optimal strategy based on current market conditions.

    Args:
        market_regime: Current market regime information
        round_data: Current round data

    Returns:
        str: Selected strategy name
    """
    try:
        # Get performance history for strategy selection
        performance = get_performance_stats()

        # Use the select_optimal_strategy function to choose best strategy
        strategy = select_optimal_strategy(market_regime, performance, round_data)

        print(
            f"🧠 Selected strategy: {strategy.get('primary')} (score: {strategy.get('score', 0):.2f})"
        )

        # If there's a backup strategy, mention it
        if strategy.get("secondary"):
            print(f"🔄 Backup strategy: {strategy.get('secondary')}")

        return strategy
    except Exception as e:
        print(f"❌ Error selecting strategy: {e}")
        return {"primary": "hybrid", "secondary": None, "score": 0.5}


def initialize_ai_strategy():
    """Initialize AI strategy with self-learning capabilities."""
    try:
        global ai_strategy
        print("🧠 Initializing AI prediction model...")
        ai_strategy = AIStrategy()

        # Load trained model if available
        loaded = ai_strategy.load_model()
        if loaded:
            print(
                f"✅ AI model loaded successfully (version {ai_strategy.model_version})"
            )
        else:
            print("⚠️ No pre-trained AI model found, starting with defaults")

        # Get current performance metrics
        performance = ai_strategy.evaluate_performance()
        if performance:
            print(f"📊 AI Model Performance:")
            print(f"   Accuracy: {performance.get('accuracy', 0):.2%}")
            print(f"   Sample Size: {performance.get('sample_size', 0)}")

        return True
    except Exception as e:
        print(f"❌ Error initializing AI strategy: {e}")
        traceback.print_exc()
        return False


def self_optimize_ai(force=False):
    """Perform self-optimization of AI models periodically."""
    try:
        global ai_strategy

        # Check if we have enough new data to optimize
        if not ai_strategy:
            return False

        # Determine if we should optimize based on time or force flag
        current_time = time.time()
        last_optimization = getattr(ai_strategy, "last_optimization_time", 0)
        hours_since_optimization = (current_time - last_optimization) / 3600

        if force or hours_since_optimization >= 24:  # Optimize daily or when forced
            print("🔄 Starting AI self-optimization...")

            # Record performance before optimization
            before_performance = ai_strategy.evaluate_performance()

            # Run self-optimization
            results = ai_strategy.self_optimize()

            # Record new performance
            after_performance = ai_strategy.evaluate_performance()

            # Set last optimization time
            ai_strategy.last_optimization_time = current_time

            # Log results
            if results.get("retrained"):
                print(f"✅ AI model self-optimized successfully")
                if before_performance and after_performance:
                    before_acc = before_performance.get("accuracy", 0)
                    after_acc = after_performance.get("accuracy", 0)
                    change = after_acc - before_acc
                    print(
                        f"📈 Accuracy changed by {change:.2%} ({before_acc:.2%} → {after_acc:.2%})"
                    )
            else:
                print(
                    "ℹ️ AI self-optimization completed with no significant improvements"
                )

            # Periodically retrain all models
            if current_time % 50 == 0:  # Every 50 epochs
                retrain_prediction_models()

            return True
        return False
    except Exception as e:
        print(f"❌ Error in AI self-optimization: {e}")
        traceback.print_exc()
        return False


def retrain_prediction_models():
    """Periodically retrain the prediction models with new data."""
    try:
        print("🔄 Retraining random forest model...")

        # Train main random forest model
        model, scaler = train_model(
            db_file=DB_FILE,
            trades_table=TABLES["trades"],
            model_file="data/random_forest_model.pkl",
            scaler_file="data/random_forest_scaler.pkl",
        )

        if model:
            print("✅ Random forest model retrained successfully")
            return True
        else:
            print("⚠️ Random forest model retraining failed")
            return False

    except Exception as e:
        print(f"❌ Error retraining models: {e}")
        traceback.print_exc()
        return False


def main(mode=None):
    """Main function with the bot's main loop."""

    # Setup logger
    logger = setup_logging()
    logger.info("🚀 Starting trading bot")

    # Check database setup
    if not check_database_setup():
        print("❌ Database check failed")
        return 1

    # Initialize database
    from data import ensure_data_ready

    if not ensure_data_ready():
        print("❌ Database initialization failed")
        return 1

    # Initialize prediction handler
    global prediction_handler
    prediction_handler = PredictionHandler()

    # Initialize balances
    global balance
    balance = get_wallet_balance(mode)
    print(f"💰 Wallet Balance: {balance:.6f} BNB")

    # Select mode if not provided
    if mode is None:
        mode = select_mode()

    print(f"🚀 Running in {mode.upper()} mode")

    # Setup event listeners for blockchain
    from scripts.blockchain.events import (setup_event_listeners,
                                           track_betting_events)

    if contract is not None:
        bull_filter, bear_filter = setup_event_listeners(contract)
        print("✅ Event listeners set up successfully")
    else:
        print("❌ Cannot set up event listeners - contract not initialized")
        bull_filter, bear_filter = None, None

    # After initializing database
    if not verify_contract_connection():
        print("⚠️ Warning: Contract connection issues, some features may not work")

    # At the beginning of main(), after database initialization but before main loop:
    # Bootstrap market data if needed
    print("🔄 Bootstrapping market data...")
    result = bootstrap_market_data()
    if result:
        print("✅ Market data bootstrapped successfully")
    else:
        print("⚠️ Market data bootstrap may be incomplete")

    # Initialize AI strategy
    if not initialize_ai_strategy():
        print("⚠️ AI strategy initialization failed, continuing with other strategies")

    # After bootstrapping market data but before the main loop
    # Try to self-optimize based on existing data
    self_optimize_ai(force=True)

    # Main loop
    try:
        while True:
            try:
                # Get current epoch and timing
                current_epoch = get_current_epoch()
                if not current_epoch:
                    print("⚠️ Cannot get current epoch. Retrying...")
                    time.sleep(10)
                    continue

                seconds_until_lock = get_time_until_lock(current_epoch)
                seconds_until_end = get_time_until_round_end(current_epoch)

                min_lock, sec_lock = divmod(seconds_until_lock, 60)
                min_end, sec_end = divmod(seconds_until_end, 60)

                print(f"\n⏱️ Current Epoch: {current_epoch}")
                print(f"⏱️ Time until lock: {min_lock}m {sec_lock}s")
                print(f"⏱️ Time until end: {min_end}m {sec_end}s")

                # Display round information
                if seconds_until_lock < 300:  # Show info when <5 minutes until lock
                    round_data = get_enriched_round_data(current_epoch)
                    if round_data:
                        display_round_info(current_epoch)

                # At the beginning of the main loop, initialize prediction variables
                prediction = "UNKNOWN"
                confidence = 0.0

                # Check for optimal betting time
                if is_betting_time(seconds_until_lock):
                    # Get round data
                    round_data = get_enriched_round_data(current_epoch)
                    if not round_data:
                        print("❌ Failed to get round data")
                        time.sleep(5)
                        continue

                    # Add before generating prediction
                    market_context = analyze_market_context(round_data)

                    # Get updated market regime
                    market_regime = detect_market_regime(get_historical_prices(30))

                    # Select optimal strategy
                    selected_strategy = select_best_strategy(market_regime, round_data)

                    # Get AI prediction if available
                    if ai_strategy:
                        ai_pred, ai_conf = ai_strategy.predict(round_data)
                        print(f"🤖 AI prediction: {ai_pred} ({ai_conf:.2f} confidence)")

                        # Add AI prediction to the set of predictions
                        prediction_handler.add_prediction("ai", ai_pred, ai_conf)

                    # After getting the AI prediction but before getting combined prediction
                    # Add hybrid prediction for more robust signals
                    if "technical_prediction" in market_context:
                        hybrid_pred, hybrid_conf = hybrid_prediction(
                            round_data,
                            {"prices": get_historical_prices(20)},
                            {"market_context": market_context},
                        )
                        print(
                            f"🔄 Hybrid prediction: {hybrid_pred} ({hybrid_conf:.2f} confidence)"
                        )

                        # Add hybrid prediction to the set of predictions
                        prediction_handler.add_prediction(
                            "hybrid", hybrid_pred, hybrid_conf
                        )

                    try:
                        # Check if the prediction_handler has the needed methods
                        if not hasattr(prediction_handler, "get_strategy_weights"):
                            # Add the method dynamically if it's missing
                            def get_weights_func(self):
                                return {
                                    "model": 0.15,
                                    "trend_following": 0.20,
                                    "contrarian": 0.15,
                                    "volume_analysis": 0.20,
                                    "market_indicators": 0.30,
                                }

                            import types

                            prediction_handler.get_strategy_weights = types.MethodType(
                                get_weights_func, prediction_handler
                            )
                            print(
                                "Added missing get_strategy_weights method dynamically"
                            )

                        # Fix the get_prediction call
                        try:
                            # Try the call with strategy preference
                            prediction, confidence = prediction_handler.get_prediction(
                                round_data, selected_strategy.get("primary")
                            )
                        except TypeError:
                            # Fall back to calling without strategy preference
                            print(
                                "Falling back to get_prediction without strategy preference"
                            )
                            prediction, confidence = prediction_handler.get_prediction(
                                round_data
                            )

                        print(
                            f"\n🔮 Prediction for round {current_epoch}: {prediction} ({confidence:.2f} confidence)"
                        )
                    except Exception as e:
                        print(f"❌ Error generating prediction: {e}")
                        traceback.print_exc()
                        prediction = "UNKNOWN"
                        confidence = 0.5

                    # Determine bet amount
                    bet_amount = calculate_optimal_bet_size(
                        wallet_balance=balance,
                        strategy_confidence=confidence,
                        base_amount=config.get("trading", {}).get(
                            "wager_amount", 0.002
                        ),
                    )

                    # Check if we should place bet
                    min_confidence = config.get("trading", {}).get(
                        "min_confidence", 0.57
                    )
                    if (
                        confidence >= min_confidence
                        and mode == "live"
                        and config.get("trading", {}).get("betting_enabled", False)
                        and should_place_bet(round_data, prediction, confidence, config)
                    ):
                        # Before simulating or placing a bet:
                        current_round_id = current_epoch  # Use the epoch as round ID

                        # Check if we already placed a bet for this round
                        if current_round_id in placed_bets:
                            print(
                                f"⚠️ Already placed a bet for round {current_round_id}, skipping"
                            )
                            continue

                        print(f"✅ Betting conditions met")
                        # Get gas price based on strategy
                        gas_strategy = config.get("trading", {}).get(
                            "gas_strategy", "medium"
                        )
                        gas_price = get_optimal_gas_price(web3, gas_strategy)

                        # After determining bet_amount but before placing bet
                        is_strong_opportunity = check_betting_opportunity(
                            confidence, round_data
                        )
                        if is_strong_opportunity:
                            # Increase bet size for strong opportunities
                            bet_amount *= 1.2  # 20% boost
                            print(
                                f"⭐ Strong opportunity detected! Boosting bet to {bet_amount:.6f} BNB"
                            )

                        # Place bet
                        result = place_bet(
                            epoch=current_epoch,
                            prediction=prediction,
                            amount=bet_amount,
                            gas_price=gas_price,
                        )

                        if result:
                            print(f"✅ Bet placed: {bet_amount:.6f} BNB on {prediction}")
                            # Add to placed_bets for tracking
                            placed_bets[current_round_id] = {
                                "prediction": prediction,
                                "amount": bet_amount,
                                "timestamp": int(time.time()),
                            }
                        else:
                            print("❌ Failed to place bet")
                    elif (
                        mode == "test"
                        and confidence >= min_confidence
                        and should_place_bet(round_data, prediction, confidence, config)
                    ):
                        # Before simulating or placing a bet:
                        current_round_id = current_epoch  # Use the epoch as round ID

                        # Check if we already placed a bet for this round
                        if current_round_id in placed_bets:
                            print(
                                f"⚠️ Already placed a test bet for round {current_round_id}, skipping"
                            )
                            continue

                        print(f"✅ Test betting conditions met")
                        result = simulate_bet(current_epoch, prediction, bet_amount)
                        if result:
                            print(
                                f"✅ Simulated bet recorded: {bet_amount:.6f} BNB on {prediction}"
                            )
                        else:
                            print("❌ Failed to record simulated bet")
                    elif confidence < min_confidence:
                        print(
                            f"⚠️ Not betting: Confidence {confidence:.2f} < minimum {min_confidence:.2f}"
                        )
                    elif not config.get("trading", {}).get("betting_enabled", False):
                        print(f"ℹ️ Betting is disabled in configuration")

                # Check for claimable rewards
                if seconds_until_end < 60 and mode == "live":
                    claimable = check_claimable_rounds()
                    if claimable:
                        print(f"💰 Found {len(claimable)} claimable rounds")
                        result = claim_rewards(claimable)
                        if result:
                            print(f"✅ Successfully claimed rewards")

                # Check for completed test mode bets
                if mode == "test":
                    for epoch, bet_data in list(placed_bets.items()):
                        if "simulated" in bet_data and bet_data["simulated"]:
                            # Get round data to check if round is complete
                            round_data = get_round_data(epoch)

                            # Only process if round is ACTUALLY complete and has an outcome
                            if (
                                round_data
                                and round_data.get("outcome")
                                and round_data.get("closeTimestamp", 0)
                                < int(time.time())
                            ):
                                # Make sure we have both lockPrice and closePrice
                                if not (
                                    round_data.get("lockPrice")
                                    and round_data.get("closePrice")
                                ):
                                    logger.info(
                                        f"⏳ Round {epoch} not fully complete yet (waiting for prices)"
                                    )
                                    continue

                                # Simulate result
                                result = simulate_round_result(
                                    epoch, bet_data["prediction"], bet_data["amount"]
                                )

                                if result:
                                    try:
                                        # Handle round completion with the simulated result
                                        handle_round_completion(epoch, result)

                                        # Print result
                                        win_text = (
                                            "✅ WON" if result["won"] else "❌ LOST"
                                        )
                                        print(
                                            f"🧪 TEST BET RESULT: {win_text} {result['profit_loss']:.6f} BNB"
                                        )

                                        # Remove from tracking
                                        del placed_bets[epoch]
                                    except Exception as e:
                                        logger.error(
                                            f"❌ Error processing round result: {e}"
                                        )
                                        traceback.print_exc()
                            else:
                                logger.info(
                                    f"⏳ Round {epoch} not complete yet (waiting for outcome)"
                                )

                # Print stats periodically
                if seconds_until_lock < 30 or seconds_until_end < 30:
                    print_stats()

                # Sleep time based on how close we are to round transitions
                if seconds_until_lock < 150 or seconds_until_end < 60:
                    time.sleep(5)  # Check more frequently near transitions
                else:
                    time.sleep(15)  # Sleep longer during inactive periods

                # After getting the prediction
                all_predictions = prediction_handler.get_all_predictions()
                weighted_confidence = calculate_weighted_confidence(all_predictions)
                print(
                    f"🔮 Weighted confidence (using strategy weights): {weighted_confidence:.2f}"
                )

                # After calculating weighted_confidence
                high_confidence_signals = get_high_confidence_predictions(
                    all_predictions
                )
                if high_confidence_signals:
                    signal_count = len(high_confidence_signals)
                    print(f"✨ Found {signal_count} high confidence signals")

                # Update AI with new outcome data for learning
                if ai_strategy and round_data and round_data.get("outcome"):
                    ai_updated = ai_strategy.record_outcome(
                        current_epoch, prediction, round_data.get("outcome"), round_data
                    )
                    if ai_updated:
                        print(
                            f"🧠 AI model updated with outcome from round {current_epoch}"
                        )

                # Attempt periodic self-optimization (will only run if enough time has passed)
                if ai_strategy and current_epoch % 10 == 0:  # Check every 10 epochs
                    self_optimize_ai()

                # Periodically retrain all models
                if current_epoch % 50 == 0:  # Every 50 epochs
                    retrain_prediction_models()

            except KeyboardInterrupt:
                print("\n👋 Exiting by user request...")
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                traceback.print_exc()
                time.sleep(5)  # Sleep on error

    except KeyboardInterrupt:
        print("\n👋 Exiting by user request...")

    print("✅ Bot shutdown complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
