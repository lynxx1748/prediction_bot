"""
Market analysis functions for the trading bot.
Provides market data, sentiment analysis, and trend detection.
"""

import asyncio
import logging
import sqlite3
import time
import traceback
from datetime import datetime, timedelta
from functools import lru_cache

import aiohttp
import numpy as np
import requests

from scripts.core.constants import config
from scripts.utils.helpers import sleep_and_check_for_interruption

from ..core.constants import DB_FILE, TABLES

# Setup logger
logger = logging.getLogger(__name__)

# Cache for price data
_price_cache = {"timestamp": 0, "price": 0}
_market_data_cache = {}


def get_historical_prices(lookback=30):
    """
    Get historical price data with multiple fallback options including Binance.US.

    Args:
        lookback: Number of data points to fetch

    Returns:
        list: List of historical prices
    """
    try:
        # First attempt: Try CoinGecko API which works in most regions
        # try:
        # url = f"https://api.coingecko.com/api/v3/coins/binancecoin/market_chart?vs_currency=usd&days={lookback/48}&interval=5m"
        # response = requests.get(url, timeout=10)

        # if response.status_code == 200:
        # data = response.json()
        # if 'prices' in data:
        # Extract prices (each price is a [timestamp, price] pair)
        # prices = [price[1] for price in data['prices']]
        # logger.info(f"✅ Retrieved {len(prices)} prices from CoinGecko")
        # return prices
        # else:
        # logger.warning(f"⚠️ CoinGecko API returned {response.status_code}")
        # except Exception as e:
        # logger.warning(f"⚠️ Error fetching from CoinGecko: {e}")

        # Second attempt: Try Binance.US API for US users
        try:
            url = f"https://api.binance.us/api/v3/klines?symbol=BNBUSD&interval=5m&limit={lookback}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                # Extract closing prices
                klines = response.json()
                prices = [float(kline[4]) for kline in klines]
                logger.info(f"✅ Retrieved {len(prices)} prices from Binance.US")
                return prices
            else:
                logger.warning(f"⚠️ Binance.US API returned {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Error fetching from Binance.US: {e}")

        # Third attempt: Regular Binance API (may be blocked in some regions)
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol=BNBUSDT&interval=5m&limit={lookback}"
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                # Extract closing prices
                klines = response.json()
                prices = [float(kline[4]) for kline in klines]
                return prices
            else:
                logger.warning(
                    f"⚠️ Regular Binance API returned {response.status_code}"
                )
        except Exception as e:
            logger.warning(f"⚠️ Error fetching from regular Binance: {e}")

        # Fallback: Try database cached prices
        from ..data.database import get_historical_price_data

        db_prices = get_historical_price_data(lookback)
        if db_prices and len(db_prices) > 0:
            logger.info(f"✅ Retrieved {len(db_prices)} prices from database")
            return db_prices

        # Final fallback: Generate synthetic data for testing
        logger.warning("⚠️ Using synthetic price data as fallback")
        import random

        base_price = 300  # Approximate BNB price
        synthetic_prices = [
            base_price * (1 + random.uniform(-0.02, 0.02)) for _ in range(lookback)
        ]
        return synthetic_prices

    except Exception as e:
        logger.error(f"❌ Error getting historical prices: {e}")
        traceback.print_exc()
        # Return minimal synthetic data as last resort
        return [300 for _ in range(lookback)]


def get_historical_volumes(lookback=30):
    """
    Get historical volume data from Binance API.

    Args:
        lookback: Number of data points to fetch

    Returns:
        list: List of historical volumes or None on failure
    """
    try:
        # Get klines/candlestick data
        url = f"https://api.binance.com/api/v3/klines?symbol=BNBUSDT&interval=5m&limit={lookback}"
        response = requests.get(url)

        if response.status_code != 200:
            logger.warning(
                f"⚠️ Error fetching historical volumes: HTTP {response.status_code}"
            )
            return None

        # Extract volume data
        klines = response.json()
        volumes = [float(kline[5]) for kline in klines]  # 5 is the volume index

        return volumes

    except Exception as e:
        logger.error(f"❌ Error getting historical volumes: {e}")
        traceback.print_exc()
        return None


def get_market_sentiment(round_data=None):
    """
    Analyze market sentiment based on price and volume data.

    Args:
        round_data: Dictionary with current round data (optional)

    Returns:
        tuple: (sentiment, strength) where sentiment is 'bullish', 'bearish', or 'neutral'
    """
    try:
        # Handle case when round_data is not provided
        if round_data is None:
            logger.warning(
                "No round data provided for market sentiment analysis, using defaults"
            )
            return "neutral", 0.5

        # Extract relevant data
        bullRatio = float(round_data.get("bullRatio", 0))
        bearRatio = float(round_data.get("bearRatio", 0))
        bnb_change = float(round_data.get("bnb_change", 0))
        btc_change = float(round_data.get("btc_change", 0))

        # Actually use bearRatio in calculation
        ratio_diff = bullRatio - bearRatio

        # Use BTC price change to adjust sentiment
        if abs(btc_change) > 1.0:  # Significant BTC movement
            # If BTC is moving strongly, it usually influences BNB
            if btc_change > 2.0:  # Strong BTC up move
                if ratio_diff > 0:  # Align with volume
                    sentiment = "bullish"
                    strength = min(0.5 + (btc_change / 10) + ratio_diff, 0.95)
                # BTC movement overrides weaker opposite signals
                elif abs(ratio_diff) < 0.1:
                    sentiment = "bullish"
                    strength = min(0.5 + (btc_change / 15), 0.8)
            elif btc_change < -2.0:  # Strong BTC down move
                if ratio_diff < 0:  # Align with volume
                    sentiment = "bearish"
                    strength = min(0.5 + (abs(btc_change) / 10) + abs(ratio_diff), 0.95)
                # BTC movement overrides weaker opposite signals
                elif abs(ratio_diff) < 0.1:
                    sentiment = "bearish"
                    strength = min(0.5 + (abs(btc_change) / 15), 0.8)

        # Use ratio difference in sentiment calculation
        if ratio_diff > 0.2:
            sentiment = "bullish"
            strength = min(0.5 + ratio_diff, 0.95)
        elif ratio_diff < -0.2:
            sentiment = "bearish"
            strength = min(0.5 + abs(ratio_diff), 0.95)
        else:
            # Use price action when volume is inconclusive
            if bnb_change > 1.0:
                sentiment = "bullish"
                strength = min(0.5 + bnb_change / 10, 0.9)
            elif bnb_change < -1.0:
                sentiment = "bearish"
                strength = min(0.5 + abs(bnb_change) / 10, 0.9)
            else:
                sentiment = "neutral"
                strength = 0.5

        return sentiment, strength

    except Exception as e:
        logger.error(f"❌ Error in market sentiment analysis: {e}")
        return "neutral", 0.5


def get_bnb_price():
    """
    Get current BNB price from Binance API.

    Returns:
        float: Current BNB price or None on failure
    """
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return float(data["price"])

        return None

    except Exception as e:
        logger.error(f"❌ Error getting BNB price: {e}")
        return None


@lru_cache(maxsize=1)
def get_bnb_price_cached(max_age_seconds=30):
    """
    Get cached BNB price with expiration.

    Args:
        max_age_seconds: Maximum age of cached price in seconds

    Returns:
        float: Current BNB price or None if cache expired and fetch failed
    """
    global _price_cache

    # Check if cache is still valid
    now = time.time()
    if now - _price_cache["timestamp"] < max_age_seconds:
        return _price_cache["price"]

    # Cache expired, fetch new price
    price = get_bnb_price()
    if price:
        _price_cache = {"timestamp": now, "price": price}
        return price

    # Return expired cache if fetch failed
    if _price_cache["price"] > 0:
        return _price_cache["price"]

    return None


async def get_price_from_source(url, json_path=None, timeout=3.0):
    """
    Get price from a specific API source.

    Args:
        url: API endpoint URL
        json_path: Path to the price in JSON response
        timeout: Request timeout in seconds

    Returns:
        float: Price or None on failure
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=timeout) as response:
                if response.status == 200:
                    data = await response.json()

                    # Extract price based on json_path
                    if json_path:
                        parts = json_path.split(".")
                        for part in parts:
                            if part.isdigit():
                                part = int(part)
                            if isinstance(data, dict) and part in data:
                                data = data[part]
                            elif (
                                isinstance(data, list)
                                and isinstance(part, int)
                                and part < len(data)
                            ):
                                data = data[part]
                            else:
                                return None

                        if isinstance(data, (int, float, str)):
                            return float(data)

                    # No json_path specified, assume data is already the price
                    elif isinstance(data, (int, float, str)):
                        return float(data)
                    elif isinstance(data, dict) and "price" in data:
                        return float(data["price"])

                return None

    except Exception as e:
        logger.debug(f"Error getting price from {url}: {e}")
        return None


async def get_prices_from_multiple_sources():
    """
    Get BNB price from multiple sources and return consensus.

    Returns:
        float: Consensus price or None if all sources fail
    """
    sources = [
        # Binance
        ("https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT", "price"),
        # CoinGecko
        (
            "https://api.coingecko.com/api/v3/simple/price?ids=binancecoin&vs_currencies=usd",
            "binancecoin.usd",
        ),
        # CryptoCompare
        ("https://min-api.cryptocompare.com/data/price?fsym=BNB&tsyms=USD", "USD"),
    ]

    tasks = [get_price_from_source(url, json_path) for url, json_path in sources]
    results = await asyncio.gather(*tasks)

    # Filter out None results
    valid_prices = [price for price in results if price is not None]

    if not valid_prices:
        return None

    # If we have multiple valid prices, remove outliers and average the rest
    if len(valid_prices) > 2:
        # Sort prices
        valid_prices.sort()
        # Remove potential outliers (lowest and highest)
        valid_prices = valid_prices[1:-1]

    # Return average of remaining prices
    return sum(valid_prices) / len(valid_prices)


def get_market_prices_with_fallback():
    """
    Get market prices with multiple fallback mechanisms.

    Returns:
        dict: Market price data
    """
    try:
        # Try to get price from primary source
        bnb_price = get_bnb_price()

        # If primary source fails, try cached price
        if not bnb_price:
            bnb_price = get_bnb_price_cached()

        # If cache fails, try multiple sources concurrently
        if not bnb_price:
            loop = asyncio.get_event_loop()
            bnb_price = loop.run_until_complete(get_prices_from_multiple_sources())

        # If all API sources fail, try getting from database
        if not bnb_price:
            from ..data.database import get_latest_market_data

            market_data = get_latest_market_data()
            if market_data and "bnb_price" in market_data:
                bnb_price = market_data["bnb_price"]

        # Return data in consistent format
        return {"bnb_price": bnb_price or 0, "timestamp": int(time.time())}

    except Exception as e:
        logger.error(f"❌ Error getting market prices: {e}")
        traceback.print_exc()
        return {"bnb_price": 0, "timestamp": int(time.time())}


def get_market_direction(lookback=None):
    """
    Determine market direction from various signals.

    Args:
        lookback: Optional number of candles to look back (defaults to 10)

    Returns:
        tuple: (direction, strength) where direction is 'bullish', 'bearish', or 'neutral'
    """
    try:
        from ..utils.helpers import get_price_trend

        # Ignore the lookback parameter if provided but use a reasonable default
        actual_lookback = 10 if lookback is None else lookback

        # Get recent price trend
        trend, trend_strength = get_price_trend(lookback=actual_lookback)

        # Get fear & greed index
        fg_value, fg_category, _ = get_fear_greed_index()

        # Get ratio analysis from recent rounds
        from ..data.database import get_market_balance

        market_balance = get_market_balance(actual_lookback)

        # Calculate direction
        signals = []

        # Price trend signal
        if trend == "up":
            signals.append(("bullish", trend_strength))
        elif trend == "down":
            signals.append(("bearish", trend_strength))

        # Fear & greed signal (contrarian approach)
        if fg_value < 30:  # Extreme fear -> bullish signal
            signals.append(("bullish", 0.5 + (30 - fg_value) / 60))
        elif fg_value > 70:  # Extreme greed -> bearish signal
            signals.append(("bearish", 0.5 + (fg_value - 70) / 60))

        # Market balance signal (contrarian approach)
        bull_ratio = market_balance.get("bull_ratio", 0.5)
        if bull_ratio > 0.65:  # Strong bull bias -> bearish signal
            signals.append(("bearish", 0.5 + (bull_ratio - 0.65) * 2))
        elif bull_ratio < 0.35:  # Strong bear bias -> bullish signal
            signals.append(("bullish", 0.5 + (0.35 - bull_ratio) * 2))

        # Add direct signal based on fear/greed category text
        if fg_category == "Extreme Fear":
            signals.append(("bullish", 0.8))  # Strong contrarian bullish signal
        elif fg_category == "Fear":
            signals.append(("bullish", 0.6))  # Moderate contrarian bullish signal
        elif fg_category == "Extreme Greed":
            signals.append(("bearish", 0.8))  # Strong contrarian bearish signal
        elif fg_category == "Greed":
            signals.append(("bearish", 0.6))  # Moderate contrarian bearish signal

        # Aggregate signals
        if not signals:
            return "neutral", 0.5

        # Count signals
        bullish_signals = [s[1] for s in signals if s[0] == "bullish"]
        bearish_signals = [s[1] for s in signals if s[0] == "bearish"]

        # Calculate strength
        bull_strength = sum(bullish_signals) / len(signals) if bullish_signals else 0
        bear_strength = sum(bearish_signals) / len(signals) if bearish_signals else 0

        # Determine direction
        if bull_strength > bear_strength:
            return "bullish", bull_strength
        elif bear_strength > bull_strength:
            return "bearish", bear_strength
        else:
            return "neutral", 0.5

    except Exception as e:
        logger.error(f"❌ Error determining market direction: {e}")
        return "neutral", 0.5


def get_fear_greed_index():
    """
    Get Fear & Greed index from API.

    Returns:
        tuple: (index_value, category, timestamp)
    """
    try:
        # Use alternative.me API
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            if "data" in data and len(data["data"]) > 0:
                return (
                    int(data["data"][0]["value"]),
                    data["data"][0]["value_classification"],
                    int(data["data"][0]["timestamp"]),
                )

        # Fallback to cached data
        cached = _market_data_cache.get("fear_greed")
        if cached and (
            time.time() - cached["timestamp"] < 3600
        ):  # Cache valid for 1 hour
            return cached["value"], cached["category"], cached["timestamp"]

        # Default values if everything fails
        return 50, "Neutral", int(time.time())

    except Exception as e:
        logger.error(f"❌ Error getting Fear & Greed index: {e}")
        return 50, "Neutral", int(time.time())


def analyze_order_book_imbalance(order_book_data):
    """
    Analyze order book for buy/sell imbalance.

    Args:
        order_book_data: Order book data with bids and asks

    Returns:
        tuple: (imbalance_direction, strength) where direction is 'buy', 'sell', or 'neutral'
    """
    try:
        if (
            not order_book_data
            or "bids" not in order_book_data
            or "asks" not in order_book_data
        ):
            return "neutral", 0

        bids = order_book_data.get("bids", [])
        asks = order_book_data.get("asks", [])

        # Calculate volume at different levels
        bid_volume = sum(float(bid[1]) for bid in bids[:10])  # Top 10 bid levels
        ask_volume = sum(float(ask[1]) for ask in asks[:10])  # Top 10 ask levels

        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return "neutral", 0

        # Calculate imbalance ratio
        bid_ratio = bid_volume / total_volume

        # Determine imbalance
        if bid_ratio > 0.6:  # More buying pressure
            imbalance = "buy"
            strength = min((bid_ratio - 0.5) * 2, 0.95)  # Scale to 0-0.95
        elif bid_ratio < 0.4:  # More selling pressure
            imbalance = "sell"
            strength = min((0.5 - bid_ratio) * 2, 0.95)  # Scale to 0-0.95
        else:
            imbalance = "neutral"
            strength = 0.5

        return imbalance, strength

    except Exception as e:
        logger.error(f"❌ Error analyzing order book: {e}")
        return "neutral", 0


def analyze_market_indicators():
    """
    Analyze multiple market indicators for a prediction.

    Returns:
        tuple: (signal, strength) where signal is 'bullish', 'bearish', or 'neutral'
    """
    try:
        signals = []

        # Get Fear & Greed Index
        fg_value, fg_category, _ = get_fear_greed_index()

        # Use both numerical value and category text for signals
        if fg_category in ["Extreme Fear", "Fear"]:
            # Use category directly for more precise signaling
            strength = 0.8 if fg_category == "Extreme Fear" else 0.65
            signals.append(("bullish", strength))
            logger.info(
                f"F&G category signal: {fg_category} → bullish ({strength:.2f})"
            )
        elif fg_category in ["Extreme Greed", "Greed"]:
            strength = 0.8 if fg_category == "Extreme Greed" else 0.65
            signals.append(("bearish", strength))
            logger.info(
                f"F&G category signal: {fg_category} → bearish ({strength:.2f})"
            )
        elif fg_value <= 25:  # Backup using numeric value
            # Extreme fear - contrarian bullish signal
            signals.append(("bullish", 0.7 + (25 - fg_value) / 100))
        elif fg_value >= 75:  # Backup using numeric value
            # Extreme greed - contrarian bearish signal
            signals.append(("bearish", 0.7 + (fg_value - 75) / 100))

        # Get recent price trend
        trend, trend_strength = get_price_trend()

        if trend == "up" and trend_strength > 0.5:
            signals.append(("bullish", trend_strength))
        elif trend == "down" and trend_strength > 0.5:
            signals.append(("bearish", trend_strength))

        # Get BTC trend as overall market indicator
        btc_data = fetch_btc_data()
        if btc_data and "change_24h" in btc_data:
            btc_change = btc_data["change_24h"]
            if btc_change > 2.0:  # Strong BTC up move
                signals.append(("bullish", min(0.5 + btc_change / 10, 0.9)))
            elif btc_change < -2.0:  # Strong BTC down move
                signals.append(("bearish", min(0.5 + abs(btc_change) / 10, 0.9)))

        # Aggregate all signals
        if not signals:
            return "neutral", 0.5

        # Count signals by direction
        bullish = [s for s in signals if s[0] == "bullish"]
        bearish = [s for s in signals if s[0] == "bearish"]

        # Calculate average strength for each direction
        bullish_strength = sum(s[1] for s in bullish) / len(bullish) if bullish else 0
        bearish_strength = sum(s[1] for s in bearish) / len(bearish) if bearish else 0

        # Determine final signal
        if bullish_strength > bearish_strength:
            return "bullish", bullish_strength
        elif bearish_strength > bullish_strength:
            return "bearish", bearish_strength
        else:
            return "neutral", 0.5

    except Exception as e:
        logger.error(f"❌ Error analyzing market indicators: {e}")
        return "neutral", 0.5


def fetch_market_data():
    """
    Fetch current market data from various sources.

    Returns:
        dict: Current market data including price and changes
    """
    try:
        # Get current prices
        prices = get_market_prices_with_fallback()

        # Get BTC price data
        btc_data = fetch_btc_data()

        # Get Fear & Greed Index
        fg_value, fg_category, fg_timestamp = get_fear_greed_index()

        # Combine all data
        market_data = {
            "bnb_price": prices.get("bnb_price", 0),
            "bnb_24h_change": prices.get("bnb_24h_change", 0),
            "btc_price": btc_data.get("price", 0),
            "btc_24h_change": btc_data.get("change_24h", 0),
            "fear_greed_value": fg_value,
            "fear_greed_class": fg_category,
            "fear_greed_updated_at": fg_timestamp,
            "timestamp": int(time.time()),
        }

        return market_data

    except Exception as e:
        logger.error(f"❌ Error fetching market data: {e}")
        traceback.print_exc()
        return None


def fetch_btc_data():
    """
    Fetch Bitcoin price data.

    Returns:
        dict: BTC price data
    """
    try:
        url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd&include_24hr_change=true"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            btc_data = {
                "price": data["bitcoin"]["usd"],
                "change_24h": data["bitcoin"]["usd_24h_change"],
            }
            return btc_data

        # Fallback to binance API
        url = "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            btc_data = {
                "price": float(data["lastPrice"]),
                "change_24h": float(data["priceChangePercent"]),
            }
            return btc_data

        return {"price": 0, "change_24h": 0}

    except Exception as e:
        logger.error(f"❌ Error fetching BTC data: {e}")
        return {"price": 0, "change_24h": 0}


def get_bnb_price_cached():
    """
    Get BNB price from cache.

    Returns:
        float: Cached BNB price or None if cache expired
    """
    global _price_cache

    # Use cache if available and recent (last 60 seconds)
    if _price_cache["timestamp"] > time.time() - 60:
        return _price_cache["price"]

    return None


async def get_prices_from_multiple_sources():
    """
    Get prices from multiple sources concurrently.

    Returns:
        float: BNB price from first successful source
    """
    try:
        async with aiohttp.ClientSession() as session:
            tasks = [
                fetch_from_binance(session),
                fetch_from_coingecko(session),
                fetch_from_coinmarketcap(session),
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Return first valid price
            for result in results:
                if isinstance(result, (int, float)) and result > 0:
                    return result

            return None

    except Exception as e:
        logger.error(f"❌ Error fetching from multiple sources: {e}")
        return None


def bootstrap_market_data():
    """
    Bootstrap initial market data if database is empty.

    Returns:
        bool: Success status
    """
    try:
        # Check if we already have enough data
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute(f"SELECT COUNT(*) FROM {TABLES['trades']}")
        count = cursor.fetchone()[0]
        conn.close()

        if count >= 20:
            return True

        logger.info(
            f"⚠️ Limited historical data (found {count} rounds) - bootstrapping data..."
        )

        # Fetch historical data from API
        import requests

        from scripts.data.database import store_historical_data

        # Get data from PancakeSwap API (last 50 rounds)
        url = "https://api.pancakeswap.com/api/v1/prediction/history"
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                rounds = data.get("rounds", [])

                if rounds:
                    logger.info(f"✅ Retrieved {len(rounds)} rounds from API")
                    for round_data in rounds:
                        store_historical_data(round_data, verbose=False)
                    return True
        except Exception as e:
            logger.warning(f"⚠️ Error fetching from official API: {e}")

        # Only import the blockchain module when needed - this breaks the circular dependency
        from scripts.data.blockchain import get_current_epoch, get_round_data

        # Fetch the last 50 rounds
        current_epoch = get_current_epoch()
        if current_epoch:
            logger.info(f"✅ Current epoch: {current_epoch}")
            for i in range(50):
                epoch = current_epoch - i - 1
                if epoch <= 0:
                    break

                try:
                    round_data = get_round_data(epoch)
                    if round_data:
                        store_historical_data(round_data, verbose=False)
                        logger.info(f"✅ Bootstrapped data for round {epoch}")
                except Exception as e:
                    logger.warning(f"⚠️ Error bootstrapping round {epoch}: {e}")

                # Don't hammer the RPC
                sleep_and_check_for_interruption(0.5)

            logger.info(f"✅ Completed bootstrapping {50} rounds of historical data")
            return True

        return False

    except Exception as e:
        logger.error(f"❌ Error bootstrapping market data: {e}")
        return False


def get_btc_price():
    """
    Get current BTC price from Binance API.

    Returns:
        float: Current BTC price or None on failure
    """
    try:
        url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return float(data["price"])

        return None

    except Exception as e:
        logger.error(f"❌ Error getting BTC price: {e}")
        return None


def get_price_trend(prices, window=10):
    """
    Calculate price trend from historical prices.

    Args:
        prices: List of price points
        window: Window size for trend calculation

    Returns:
        dict: Trend information
    """
    try:
        if not prices or len(prices) < window:
            return {"direction": "neutral", "strength": 0, "change_percent": 0}

        # Get recent and older prices
        recent_price = (
            prices[0]
            if isinstance(prices[0], (int, float))
            else prices[0].get("price", 0)
        )
        older_price = (
            prices[window - 1]
            if isinstance(prices[window - 1], (int, float))
            else prices[window - 1].get("price", 0)
        )

        if older_price == 0:
            return {"direction": "neutral", "strength": 0, "change_percent": 0}

        # Calculate change
        change = recent_price - older_price
        change_percent = (change / older_price) * 100

        # Determine direction and strength
        if change_percent > 0.5:
            direction = "up"
            strength = min(abs(change_percent) / 2, 1.0)  # Cap strength at 1.0
        elif change_percent < -0.5:
            direction = "down"
            strength = min(abs(change_percent) / 2, 1.0)  # Cap strength at 1.0
        else:
            direction = "neutral"
            strength = 0

        return {
            "direction": direction,
            "strength": strength,
            "change_percent": change_percent,
        }

    except Exception as e:
        logger.error(f"Error calculating price trend: {e}")
        return {"direction": "neutral", "strength": 0, "change_percent": 0}


def fetch_from_binance(symbol="BNB"):
    """
    Fetch price data from Binance API.

    Args:
        symbol: Trading symbol (default: BNB)

    Returns:
        float: Current price or None on failure
    """
    try:
        url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return float(data["price"])

        return None

    except Exception as e:
        logger.error(f"Error fetching from Binance: {e}")
        return None


def fetch_from_coingecko(coin_id="binancecoin"):
    """
    Fetch price data from CoinGecko API.

    Args:
        coin_id: Coin ID on CoinGecko (default: binancecoin for BNB)

    Returns:
        float: Current price or None on failure
    """
    try:
        url = f"https://api.coingecko.com/api/v3/simple/price?ids={coin_id}&vs_currencies=usd"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return float(data[coin_id]["usd"])

        return None

    except Exception as e:
        logger.error(f"Error fetching from CoinGecko: {e}")
        return None


def fetch_from_coinmarketcap(symbol="BNB"):
    """
    Fetch price data from CoinMarketCap API.
    Note: This requires an API key to work properly.

    Args:
        symbol: Trading symbol (default: BNB)

    Returns:
        float: Current price or None on failure
    """
    try:
        # This is a stub - CoinMarketCap requires an API key
        # You would need to add your API key to the configuration
        api_key = config.get("api_keys", {}).get("coinmarketcap")
        if not api_key:
            logger.warning("No CoinMarketCap API key found")
            return None

        url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"
        parameters = {"symbol": symbol}
        headers = {
            "Accepts": "application/json",
            "X-CMC_PRO_API_KEY": api_key,
        }

        response = requests.get(url, headers=headers, params=parameters, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return float(data["data"][symbol]["quote"]["USD"]["price"])

        return None

    except Exception as e:
        logger.error(f"Error fetching from CoinMarketCap: {e}")
        return None


def get_market_time_ranges():
    """
    Get time ranges for different market analysis periods.
    Uses datetime for time calculations.

    Returns:
        dict: Various time ranges for market analysis
    """
    now = datetime.now()

    return {
        "today": {"start": datetime(now.year, now.month, now.day, 0, 0, 0), "end": now},
        "yesterday": {
            "start": datetime(now.year, now.month, now.day, 0, 0, 0)
            - timedelta(days=1),
            "end": datetime(now.year, now.month, now.day, 0, 0, 0)
            - timedelta(seconds=1),
        },
        "week": {"start": now - timedelta(days=7), "end": now},
        "month": {"start": now - timedelta(days=30), "end": now},
    }


def calculate_market_statistics(prices):
    """
    Calculate key market statistics using numpy.

    Args:
        prices: List of historical prices

    Returns:
        dict: Statistical metrics about the market data
    """
    if not prices or len(prices) < 2:
        return {}

    # Convert to numpy array for calculations
    price_array = np.array(prices)
    returns = np.diff(price_array) / price_array[:-1]

    stats = {
        "mean": float(np.mean(price_array)),
        "median": float(np.median(price_array)),
        "std_dev": float(np.std(price_array)),
        "min": float(np.min(price_array)),
        "max": float(np.max(price_array)),
        "volatility": float(np.std(returns) * np.sqrt(365)),  # Annualized volatility
        "skewness": float(
            0
            if len(returns) < 3
            else np.sum((returns - np.mean(returns)) ** 3)
            / ((len(returns) - 1) * np.std(returns) ** 3)
        ),
    }

    return stats


def get_market_sentiment_report():
    """
    Generate a comprehensive market sentiment report including Fear & Greed categories.

    Returns:
        dict: Market sentiment report data
    """
    try:
        # Get fear & greed index
        fg_value, fg_category, _ = get_fear_greed_index()

        # Use the category to determine sentiment description
        sentiment_description = ""

        if fg_category == "Extreme Fear":
            sentiment_description = (
                "Market in extreme fear - potential buying opportunity"
            )
        elif fg_category == "Fear":
            sentiment_description = "Market fearful - cautiously bullish"
        elif fg_category == "Neutral":
            sentiment_description = "Market sentiment balanced"
        elif fg_category == "Greed":
            sentiment_description = "Market greedy - consider taking profits"
        elif fg_category == "Extreme Greed":
            sentiment_description = (
                "Market extremely greedy - potential selling opportunity"
            )

        logger.info(
            f"Market sentiment: {fg_category} ({fg_value}) - {sentiment_description}"
        )

        # Rest of function...

        return {
            "fear_greed_value": fg_value,
            "fear_greed_category": fg_category,
            "sentiment_description": sentiment_description,
            # Other keys...
        }
    except Exception as e:
        logger.error(f"Error generating market sentiment report: {e}")
        return {}
