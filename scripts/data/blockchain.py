"""
Blockchain data operations for the trading bot.
Handles retrieval of data from the blockchain and local database.
"""

import logging
import sqlite3
import time
import traceback

from web3 import Web3

from ..analysis.market import (get_bnb_price, get_btc_price,
                               get_historical_prices)
from ..core.constants import DB_FILE, TABLES, contract, web3

logger = logging.getLogger(__name__)


def get_round_data(epoch):
    """
    Get data for a specific round from the blockchain.

    Args:
        epoch: Epoch number to retrieve

    Returns:
        dict: Round data or None on failure
    """
    try:
        # Get round info from contract
        round_data = contract.functions.rounds(epoch).call()

        # Extract basic data
        start_timestamp = round_data[1]
        lock_timestamp = round_data[2]
        close_timestamp = round_data[3]
        lockPrice_raw = round_data[4]
        closePrice_raw = round_data[5]

        # Convert wei to BNB
        bullAmount = web3.from_wei(round_data[9], "ether")  # bullAmount
        bearAmount = web3.from_wei(round_data[10], "ether")  # bearAmount
        totalAmount = bullAmount + bearAmount

        # Handle price conversion with multiple strategies
        lockPrice = 0
        closePrice = 0

        # Strategy 1: Standard wei conversion (standard contract behavior)
        try:
            if lockPrice_raw > 0:
                lockPrice = float(web3.from_wei(lockPrice_raw, "ether"))
            if closePrice_raw > 0:
                closePrice = float(web3.from_wei(closePrice_raw, "ether"))
        except:
            pass

        # Strategy 2: Direct value (some contracts store price directly)
        if lockPrice < 0.1 and lockPrice_raw > 0:
            lockPrice = float(lockPrice_raw)
        if closePrice < 0.1 and closePrice_raw > 0:
            closePrice = float(closePrice_raw)

        # Strategy 3: Scaled value (some contracts store scaled prices, e.g., BNB*10^8)
        if lockPrice < 0.1 and lockPrice_raw > 0:
            lockPrice = float(lockPrice_raw) / 1e8
        if closePrice < 0.1 and closePrice_raw > 0:
            closePrice = float(closePrice_raw) / 1e8

        # Strategy 4: If prices still zero, try to get from price oracle
        if (lockPrice == 0 or closePrice == 0) and epoch < get_current_epoch() - 1:
            try:
                current_price = get_bnb_price()
                if lockPrice == 0 and current_price:
                    logger.warning(
                        f"⚠️ Using current price {current_price} as lock price"
                    )
                    lockPrice = current_price
                if closePrice == 0 and current_price:
                    logger.warning(
                        f"⚠️ Using current price {current_price} as close price"
                    )
                    closePrice = current_price
            except Exception as e:
                logger.error(f"Error getting price fallback: {e}")

        # Determine outcome if round is closed
        outcome = None
        if close_timestamp > 0:  # If closeTimestamp > 0
            if closePrice > lockPrice:
                outcome = "BULL"
            elif closePrice < lockPrice:
                outcome = "BEAR"
            else:
                outcome = "DRAW"

        # Calculate ratios
        if totalAmount > 0:
            bullRatio = float(bullAmount) / float(totalAmount)
            bearRatio = float(bearAmount) / float(totalAmount)
        else:
            bullRatio = 0.5
            bearRatio = 0.5

        # Format the data
        formatted_data = {
            "epoch": epoch,
            "startTimestamp": start_timestamp,
            "lockTimestamp": lock_timestamp,
            "closeTimestamp": close_timestamp,
            "lockPrice": lockPrice,
            "closePrice": closePrice,
            "bullAmount": bullAmount,
            "bearAmount": bearAmount,
            "totalAmount": totalAmount,
            "bullRatio": bullRatio,
            "bearRatio": bearRatio,
            "outcome": outcome,
            "oracle_called": round_data[13],
        }

        # Log summary
        logger.info(f"Bull/Bear Ratio: {bullRatio:.2%}/{bearRatio:.2%}")
        if outcome:
            logger.info(
                f"Outcome: {outcome} (Lock: {lockPrice:.2f} → Close: {closePrice:.2f})"
            )

        return formatted_data

    except Exception as e:
        logger.error(f"❌ Error getting round {epoch} data: {e}")
        traceback.print_exc()
        return None


def get_historical_data(epoch):
    """
    Get historical data for a specific round from the database.

    Args:
        epoch: Epoch number to retrieve

    Returns:
        dict: Round data from database or None on failure
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute(f"SELECT * FROM {TABLES['trades']} WHERE epoch = ?", (epoch,))
        data = cursor.fetchone()

        if not data:
            return None

        column_names = [description[0] for description in cursor.description]
        result = dict(zip(column_names, data))

        conn.close()
        return result
    except Exception as e:
        logger.error(f"❌ Error loading historical data for epoch {epoch}: {e}")
        return None


def get_current_epoch():
    """
    Get the current prediction epoch.

    Returns:
        int: Current epoch number, or None on error
    """
    try:
        if contract is None:
            logger.error("❌ Cannot get current epoch: Contract not initialized")
            return None

        current_epoch = contract.functions.currentEpoch().call()
        return current_epoch
    except Exception as e:
        logger.error(f"❌ Error getting current epoch: {e}")
        return None


def get_time_until_lock(epoch=None):
    """
    Get time until the epoch locks.

    Args:
        epoch: Epoch to check, uses current epoch if None

    Returns:
        int: Seconds until lock, or 300 (default) on error
    """
    try:
        if contract is None:
            logger.error("❌ Cannot get lock time: Contract not initialized")
            return 300  # Default value

        if epoch is None:
            epoch = get_current_epoch()
            if not epoch:  # Handle None epoch
                return 300

        # Get current time
        current_time = int(time.time())

        # Try to get the actual lock timestamp from the contract
        round_data = contract.functions.rounds(epoch).call()
        lockTimestamp = round_data[2]  # Index 2 is the lock timestamp

        # If lockTimestamp is 0, it means the round hasn't started yet
        if lockTimestamp == 0:
            # Calculate based on the active round (epoch - 1)
            active_round_data = contract.functions.rounds(epoch - 1).call()
            active_lockTimestamp = active_round_data[2]

            # Each round's lock is at the same time as the current round's close
            lockTimestamp = active_lockTimestamp + 300

        # Calculate seconds until lock
        seconds_until_lock = lockTimestamp - current_time
        if seconds_until_lock < 0:
            seconds_until_lock = 0

        return seconds_until_lock

    except Exception as e:
        logger.error(f"❌ Error calculating time until lock: {e}")
        return 0


def get_betting_epoch():
    """
    Get the correct epoch to place bets on (active round + 1).

    Returns:
        int: Epoch number to bet on or None on failure
    """
    try:
        current_epoch = get_current_epoch()
        if current_epoch is None:
            return None

        # Active round is current_epoch - 1
        active_round = current_epoch - 1

        # We should bet on active_round + a1 (the next round after the active one)
        betting_epoch = active_round + 1

        # This should be equal to the current_epoch
        if betting_epoch != current_epoch:
            logger.warning(
                f"⚠️ Betting epoch calculation mismatch. Expected {current_epoch}, got {betting_epoch}"
            )

        return betting_epoch
    except Exception as e:
        logger.error(f"❌ Error determining betting epoch: {e}")
        return None


def get_time_until_round_end(epoch):
    """
    Get time until the end of the specified epoch.

    Args:
        epoch: Epoch number to check

    Returns:
        int: Seconds until round end or 0 if round ended/error
    """
    try:
        # Get current time
        current_time = int(time.time())

        # Get round data to find the lock timestamp
        round_data = contract.functions.rounds(epoch).call()
        lockTimestamp = round_data[2]  # Index 2 is the lock timestamp

        # Each round is 5 minutes (300 seconds) from lock to close
        round_end_time = lockTimestamp + 300

        # Calculate seconds until round end
        seconds_until_end = round_end_time - current_time
        if seconds_until_end < 0:
            seconds_until_end = 0

        return seconds_until_end

    except Exception as e:
        logger.error(f"❌ Error calculating time until round end: {e}")
        return 0


def get_enriched_round_data(epoch):
    """
    Get round data enriched with historical data and analysis.

    Args:
        epoch: Epoch number to retrieve

    Returns:
        dict: Enriched round data or None on failure
    """
    try:
        # Get basic round data from blockchain
        round_data = get_round_data(epoch)

        if not round_data:
            logger.warning(f"⚠️ No round data available for epoch {epoch}")
            return None

        # Get historical data for additional context
        from ..data.database import get_recent_rounds

        historical_data = get_recent_rounds(10)

        if historical_data:
            # Enrich round data with historical context
            round_data["historical_data"] = historical_data

            # Calculate additional metrics based on historical data
            if len(historical_data) >= 3:
                # Calculate price movement trend
                prices = [
                    data.get("closePrice", 0)
                    for data in historical_data
                    if data.get("closePrice", 0) > 0
                ]
                if len(prices) >= 3:
                    round_data["price_trend"] = (
                        "up" if prices[0] > prices[2] else "down"
                    )
                    round_data["price_volatility"] = (
                        abs(prices[0] - prices[-1]) / prices[-1]
                        if prices[-1] > 0
                        else 0
                    )

            logger.info(f"✅ Enriched round {epoch} with historical data")
        else:
            round_data["historical_data"] = []
            logger.info(f"ℹ️ No historical data available for epoch {epoch}")

        return round_data

    except Exception as e:
        logger.error(f"❌ Error getting enriched round data: {e}")
        traceback.print_exc()
        return None


def get_round_with_market_prices(epoch):
    """
    Get round data with current market prices for BNB and BTC.

    Args:
        epoch: Epoch number to retrieve

    Returns:
        dict: Round data with market prices or None on failure
    """
    try:
        # Get basic round data
        round_data = get_round_data(epoch)

        if not round_data:
            return None

        # Add current market prices
        bnb_price = get_bnb_price()
        btc_price = get_btc_price()

        if bnb_price:
            round_data["current_bnb_price"] = bnb_price
            # Calculate percent difference from lock price
            if round_data.get("lockPrice", 0) > 0:
                round_data["bnb_price_diff"] = (
                    (bnb_price - round_data["lockPrice"])
                    / round_data["lockPrice"]
                    * 100
                )

        if btc_price:
            round_data["current_btc_price"] = btc_price

        # Add market correlation data
        if bnb_price and btc_price:
            logger.info(
                f"📊 Current market prices - BNB: ${bnb_price:.2f}, BTC: ${btc_price:.2f}"
            )

        return round_data

    except Exception as e:
        logger.error(f"❌ Error getting round data with market prices: {e}")
        traceback.print_exc()
        return None


def fetch_rounds_range(start_epoch, end_epoch):
    """
    Fetch data for a range of rounds from the blockchain.

    Args:
        start_epoch: Starting epoch number
        end_epoch: Ending epoch number

    Returns:
        list: List of round data dictionaries
    """
    rounds = []

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            round_data = get_round_data(epoch)
            if round_data:
                rounds.append(round_data)

        return rounds

    except Exception as e:
        logger.error(f"❌ Error fetching rounds range: {e}")
        return rounds


def get_round_info(config):
    """
    Get current round info and timing details based on contract logic.

    Args:
        config: Configuration dictionary with timing settings

    Returns:
        dict: Round timing and status information or None on failure
    """
    try:
        # Get current epoch from contract
        current_epoch = contract.functions.currentEpoch().call()

        # Get interval seconds (time between rounds)
        try:
            interval_seconds = contract.functions.intervalSeconds().call()
        except:
            interval_seconds = 300  # Default to 5 minutes if not available

        # Get data for current epoch (the round we can bet on)
        current_round_data = contract.functions.rounds(current_epoch).call()

        # Get data for active round (the round that's currently running)
        active_round = current_epoch - 1
        active_round_data = contract.functions.rounds(active_round).call()

        # Current timestamp
        current_timestamp = int(time.time())

        # Extract timestamps from contract data
        current_startTimestamp = current_round_data[1]
        current_lockTimestamp = current_round_data[2]
        current_closeTimestamp = current_round_data[3]

        active_startTimestamp = active_round_data[1]
        active_lockTimestamp = active_round_data[2]
        active_closeTimestamp = active_round_data[3]

        # Calculate time until betting closes for current round
        seconds_until_lock = max(0, current_lockTimestamp - current_timestamp)

        # Calculate time until active round ends
        seconds_until_close = max(0, active_closeTimestamp - current_timestamp)

        # Get optimal betting window parameters from config
        timing_config = config.get("timing", {}).get(
            "optimal_betting_seconds_before_lock", {}
        )
        min_seconds = timing_config.get("min", 30)
        max_seconds = timing_config.get("max", 60)

        # Determine if we're in the optimal betting window based on config
        optimal_betting_time = max_seconds >= seconds_until_lock >= min_seconds

        # The general betting window is still from start to lock
        betting_window_open = (
            current_timestamp > current_startTimestamp
            and current_timestamp < current_lockTimestamp
        )

        # Format times for display
        minutes_until_lock = int(seconds_until_lock // 60)
        seconds_until_lock_remainder = int(seconds_until_lock % 60)

        minutes_until_close = int(seconds_until_close // 60)
        seconds_until_close_remainder = int(seconds_until_close % 60)

        # Log current status
        if betting_window_open:
            if optimal_betting_time:
                logger.info(
                    f"🎯 OPTIMAL BETTING WINDOW: {minutes_until_lock}m {seconds_until_lock_remainder}s until lock"
                )
            else:
                logger.info(
                    f"🕒 Betting window open: {minutes_until_lock}m {seconds_until_lock_remainder}s until lock"
                )
        else:
            logger.info(
                f"⏳ Betting window closed. Next round in {minutes_until_close}m {seconds_until_close_remainder}s"
            )

        return {
            "current_epoch": current_epoch,
            "active_round": active_round,
            "betting_round": current_epoch,
            "seconds_until_lock": seconds_until_lock,
            "seconds_until_close": seconds_until_close,
            "startTimestamp": current_startTimestamp,
            "lockTimestamp": current_lockTimestamp,
            "closeTimestamp": current_closeTimestamp,
            "betting_window_open": betting_window_open,
            "optimal_betting_time": optimal_betting_time,
            "interval_seconds": interval_seconds,
        }

    except Exception as e:
        logger.error(f"❌ Error getting round info: {e}")
        traceback.print_exc()
        return None


def convert_wei_to_bnb(wei_amount):
    """
    Convert wei amount to BNB using Web3 utilities.

    Args:
        wei_amount: Amount in wei

    Returns:
        float: Equivalent amount in BNB
    """
    try:
        if isinstance(wei_amount, str):
            wei_amount = int(wei_amount)

        # Use Web3 to convert wei to ether (BNB)
        bnb_amount = Web3.from_wei(wei_amount, "ether")
        return float(bnb_amount)

    except Exception as e:
        print(f"Error converting wei to BNB: {e}")
        traceback.print_exc()
        return 0.0


def analyze_price_history(lookback_days=30, interval="1d"):
    """
    Analyze historical price data from the blockchain.
    Uses get_historical_prices to fetch the data.

    Args:
        lookback_days: Number of days to look back
        interval: Data interval ('1h', '1d', etc)

    Returns:
        dict: Price analysis results
    """
    try:
        # Use the imported get_historical_prices function
        prices = get_historical_prices(lookback_days, interval)

        if not prices or len(prices) < 2:
            logger.warning("Not enough historical price data for analysis")
            return {"status": "insufficient_data"}

        # Calculate some basic statistics
        current_price = prices[-1]
        oldest_price = prices[0]
        price_change = (current_price - oldest_price) / oldest_price

        # Find highest and lowest prices
        highest_price = max(prices)
        lowest_price = min(prices)

        # Calculate 7-day moving average if we have enough data
        ma7 = (
            sum(prices[-7:]) / min(7, len(prices))
            if len(prices) >= 7
            else current_price
        )

        logger.info(
            f"Price analysis: Current: {current_price:.2f}, Change: {price_change:.2%}"
        )

        return {
            "status": "success",
            "current_price": current_price,
            "price_change": price_change,
            "highest_price": highest_price,
            "lowest_price": lowest_price,
            "moving_average_7d": ma7,
            "raw_data": prices,
        }

    except Exception as e:
        logger.error(f"Error analyzing price history: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}
