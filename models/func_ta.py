import numpy as np
import traceback
import requests
import logging

# Import from configuration module
from configuration import config

# Setup logger
logger = logging.getLogger(__name__)

class TechnicalAnalysis:
    """Technical analysis tools and indicators for price prediction."""
    
    @staticmethod
    def rsi(prices, period=14):
        """
        Calculate Relative Strength Index.
        
        Args:
            prices: Array of price data
            period: RSI period (default: 14)
            
        Returns:
            float: RSI value
        """
        try:
            deltas = np.diff(prices)
            gain = np.where(deltas > 0, deltas, 0)
            loss = np.where(deltas < 0, -deltas, 0)
            
            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])
            
            for i in range(period, len(gain)):
                avg_gain = (avg_gain * (period - 1) + gain[i]) / period
                avg_loss = (avg_loss * (period - 1) + loss[i]) / period
                
            if avg_loss == 0:
                return 100
                
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))
        except Exception as e:
            logger.warning(f"Error calculating RSI: {e}")
            return 50  # Default to neutral on error
    
    @staticmethod
    def macd(prices, fast=12, slow=26, signal=9):
        """
        Calculate MACD, Signal line, and Histogram.
        
        Args:
            prices: Array of price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            tuple: (MACD line, Signal line, Histogram)
        """
        try:
            # Calculate EMAs
            ema_fast = TechnicalAnalysis.ema(prices, fast)
            ema_slow = TechnicalAnalysis.ema(prices, slow)
            
            # Calculate MACD line
            macd_line = ema_fast - ema_slow
            
            # Calculate Signal line
            signal_line = TechnicalAnalysis.ema(macd_line, signal)
            
            # Calculate histogram
            histogram = macd_line - signal_line
            
            return macd_line, signal_line, histogram
        except Exception as e:
            logger.warning(f"Error calculating MACD: {e}")
            return 0, 0, 0  # Default values on error
    
    @staticmethod
    def bollinger_bands(prices, period=20, std_dev=2):
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Array of price data
            period: Moving average period
            std_dev: Number of standard deviations
            
        Returns:
            tuple: (Upper band, Middle band, Lower band)
        """
        try:
            middle_band = TechnicalAnalysis.sma(prices, period)
            std = np.std(prices[-period:])
            
            upper_band = middle_band + (std_dev * std)
            lower_band = middle_band - (std_dev * std)
            
            return upper_band, middle_band, lower_band
        except Exception as e:
            logger.warning(f"Error calculating Bollinger Bands: {e}")
            return prices[-1], prices[-1], prices[-1]  # Default to price on error
    
    @staticmethod
    def ema(prices, period):
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Array of price data
            period: EMA period
            
        Returns:
            float: EMA value
        """
        try:
            prices = np.array(prices)
            multiplier = 2 / (period + 1)
            ema = [prices[0]]  # Start with first price
            
            for price in prices[1:]:
                ema.append((price - ema[-1]) * multiplier + ema[-1])
                
            return np.array(ema)[-1]  # Return last value
        except Exception as e:
            logger.warning(f"Error calculating EMA: {e}")
            return prices[-1]  # Default to last price on error
    
    @staticmethod
    def sma(prices, period):
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Array of price data
            period: SMA period
            
        Returns:
            float: SMA value
        """
        try:
            return np.mean(prices[-period:])
        except Exception as e:
            logger.warning(f"Error calculating SMA: {e}")
            return prices[-1]  # Default to last price on error

    @staticmethod
    def detect_candlestick_patterns(opens, highs, lows, closes):
        """Detect basic candlestick patterns"""
        patterns = {}
        
        # Check for bullish engulfing
        patterns['bullish_engulfing'] = (opens[-1] < closes[-2]) and (closes[-1] > opens[-2]) and (closes[-1] - opens[-1] > opens[-2] - closes[-2])
        
        # Check for bearish engulfing
        patterns['bearish_engulfing'] = (opens[-1] > closes[-2]) and (closes[-1] < opens[-2]) and (opens[-1] - closes[-1] > closes[-2] - opens[-2])
        
        # Check for hammer (bullish)
        patterns['hammer'] = (closes[-1] > opens[-1]) and ((highs[-1] - closes[-1]) < 0.2 * (closes[-1] - lows[-1])) and ((closes[-1] - opens[-1]) < 0.3 * (closes[-1] - lows[-1]))
        
        # Check for shooting star (bearish)
        patterns['shooting_star'] = (closes[-1] < opens[-1]) and ((closes[-1] - lows[-1]) < 0.2 * (highs[-1] - closes[-1])) and ((opens[-1] - closes[-1]) < 0.3 * (highs[-1] - closes[-1]))
        
        # Check for doji (neutral)
        patterns['doji'] = abs(opens[-1] - closes[-1]) < 0.1 * (highs[-1] - lows[-1])
        
        return patterns

    @staticmethod
    def get_candlestick_patterns():
        """
        Analyze candlestick patterns from recent price data.
        Returns pattern signals and their reliability.
        """
        try:
            # Get candlestick data from Binance API
            url = "https://api.binance.com/api/v3/klines?symbol=BNBUSDT&interval=1m&limit=30"
            response = requests.get(url)
            if response.status_code != 200:
                logger.error(f"❌ Error fetching candlestick data: HTTP {response.status_code}")
                return None, 0
                
            candles = response.json()
            
            # Extract OHLC data
            opens = np.array([float(candle[1]) for candle in candles])
            highs = np.array([float(candle[2]) for candle in candles])
            lows = np.array([float(candle[3]) for candle in candles])
            closes = np.array([float(candle[4]) for candle in candles])
            
            # Detect patterns
            patterns = TechnicalAnalysis.detect_candlestick_patterns(opens, highs, lows, closes)
            
            # Count bullish and bearish signals
            bullish_signals = sum([patterns['bullish_engulfing'], patterns['hammer']])
            bearish_signals = sum([patterns['bearish_engulfing'], patterns['shooting_star']])
            
            # Determine overall pattern signal
            if bullish_signals > bearish_signals:
                pattern_signal = "bull"
                strength = 0.6 + (0.1 * bullish_signals)
            elif bearish_signals > bullish_signals:
                pattern_signal = "bear"
                strength = 0.6 + (0.1 * bearish_signals)
            else:
                # If tied, look at the most recent price movement
                if closes[-1] > opens[-1]:
                    pattern_signal = "bull"
                    strength = 0.55
                elif closes[-1] < opens[-1]:
                    pattern_signal = "bear"
                    strength = 0.55
                else:
                    pattern_signal = None
                    strength = 0
            
            # Print detected patterns
            if pattern_signal:
                logger.info(f"\n🕯️ Candlestick Pattern Analysis:")
                logger.info(f"   Signal: {pattern_signal.upper()} (strength: {strength:.2f})")
                
                if patterns['bullish_engulfing']:
                    logger.info(f"   Detected: Bullish Engulfing")
                if patterns['hammer']:
                    logger.info(f"   Detected: Hammer (Bullish)")
                if patterns['bearish_engulfing']:
                    logger.info(f"   Detected: Bearish Engulfing")
                if patterns['shooting_star']:
                    logger.info(f"   Detected: Shooting Star (Bearish)")
            
            return pattern_signal, strength
            
        except Exception as e:
            logger.error(f"❌ Error analyzing candlestick patterns: {e}")
            traceback.print_exc()
            return None, 0

    @staticmethod
    def analyze_price_pattern(round_data, contract, web3, previous_rounds=3):
        """
        Analyze recent price patterns to identify potential rebounds or continuations.
        
        Args:
            round_data: Dictionary containing current round data
            contract: Web3 contract instance
            web3: Web3 instance
            previous_rounds: Number of previous rounds to analyze
            
        Returns:
            tuple: (signal, confidence) where signal is 'BULL', 'BEAR', or None
        """
        try:
            # Get current epoch
            current_epoch = round_data.get('epoch')
            if not current_epoch:
                current_epoch = contract.functions.currentEpoch().call()
            
            # Initialize variables for tracking patterns
            price_changes = []
            consecutive_same_direction = 0
            last_direction = None
            
            # Get active round (current round that's running)
            active_round = current_epoch - 1
            active_round_data = contract.functions.rounds(active_round).call()
            
            # Check if active round has lock price (it should)
            if active_round_data[4] > 0:  # lockPrice is set
                # Get current price from Binance API
                current_price = TechnicalAnalysis.get_current_price()
                
                if current_price and active_round_data[4] > 0:
                    # Calculate real-time price change during active round
                    lockPrice = web3.from_wei(active_round_data[4], 'ether')  # Convert from wei
                    current_price_float = float(current_price)
                    lockPrice_float = float(lockPrice)
                    
                    # Check for unrealistic values and fix them
                    if lockPrice_float < 0.001 or lockPrice_float > 10000:
                        logger.warning(f"⚠️ Unrealistic lock price detected: {lockPrice_float}, using current price")
                        lockPrice_float = current_price_float
                    
                    # Calculate percentage change
                    price_change_percent = ((current_price_float - lockPrice_float) / lockPrice_float) * 100
                    
                    # Cap unrealistic price changes
                    if abs(price_change_percent) > 20:
                        logger.warning(f"⚠️ Unrealistic price change detected: {price_change_percent:.2f}%, capping at ±5%")
                        price_change_percent = 5.0 if price_change_percent > 0 else -5.0
                    
                    logger.info(f"📊 Active round price change: {price_change_percent:.2f}%")
                    
                    # Check for significant price movements
                    if price_change_percent < -4.0:
                        logger.info(f"📉 MAJOR drop detected in active round: {price_change_percent:.2f}%")
                        return "BULL", min(0.95, abs(price_change_percent) / 8)
                    elif price_change_percent < -2.0:
                        logger.info(f"📉 Significant drop detected in active round: {price_change_percent:.2f}%")
                        return "BULL", min(0.85, abs(price_change_percent) / 10)
                    elif price_change_percent > 4.0:
                        logger.info(f"📈 MAJOR rise detected in active round: {price_change_percent:.2f}%")
                        return "BEAR", min(0.85, price_change_percent / 8)
                    elif price_change_percent > 2.0:
                        logger.info(f"📈 Significant rise detected in active round: {price_change_percent:.2f}%")
                        return "BULL", min(0.75, price_change_percent / 10)
            
            # Analyze previous completed rounds
            for i in range(1, previous_rounds + 1):
                prev_epoch = current_epoch - i
                if prev_epoch <= 0:
                    continue
                    
                prev_round_data = contract.functions.rounds(prev_epoch).call()
                
                if prev_round_data[5] > 0:  # closePrice is set
                    lockPrice = prev_round_data[4]
                    closePrice = prev_round_data[5]
                    
                    if lockPrice > 0:
                        price_change = ((closePrice - lockPrice) / lockPrice) * 100
                        price_changes.append(price_change)
                        
                        current_direction = "up" if price_change > 0 else "down"
                        
                        if last_direction is None:
                            last_direction = current_direction
                        elif last_direction == current_direction:
                            consecutive_same_direction += 1
                        else:
                            break
                            
                        last_direction = current_direction
            
            # Analyze patterns from previous rounds
            if price_changes:
                if price_changes[0] < -3.0:
                    strength = min(0.9, abs(price_changes[0]) / 10)
                    logger.info(f"📉 Detected potential rebound after {price_changes[0]:.2f}% drop in previous round")
                    return "BULL", strength
                    
                elif consecutive_same_direction >= 2:
                    avg_change = sum(price_changes[:consecutive_same_direction+1]) / (consecutive_same_direction+1)
                    strength = min(0.8, abs(avg_change) / 5)
                    prediction = "BULL" if last_direction == "up" else "BEAR"
                    logger.info(f"📈 Detected trend continuation: {consecutive_same_direction + 1} rounds of {last_direction}")
                    return prediction, strength
            
            return None, 0
            
        except Exception as e:
            logger.error(f"❌ Error analyzing price pattern: {e}")
            traceback.print_exc()
            return None, 0

    @staticmethod
    def get_current_price():
        """Get current BNB price from Binance API."""
        try:
            response = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BNBUSDT")
            if response.status_code == 200:
                return float(response.json()['price'])
            return None
        except:
            return None

    def calculate_indicators(self, prices):
        """
        Calculate technical indicators for the given price data.
        
        Args:
            prices: List of price data
            
        Returns:
            dict: Dictionary of technical indicators
        """
        try:
            # Convert to numpy array if it's not already
            price_array = np.array(prices)
            
            # Calculate RSI
            rsi = self.rsi(price_array)
            
            # Calculate MACD
            macd, macd_signal, macd_hist = self.macd(price_array)
            
            # Calculate Bollinger Bands
            upper_band, middle_band, lower_band = self.bollinger_bands(price_array)
            
            # Create indicators dictionary
            indicators = {
                'rsi': rsi if isinstance(rsi, (int, float)) else rsi[-1] if len(rsi) > 0 else 50,
                'macd': macd if isinstance(macd, (int, float)) else macd[-1] if len(macd) > 0 else 0,
                'macd_signal': macd_signal if isinstance(macd_signal, (int, float)) else macd_signal[-1] if len(macd_signal) > 0 else 0,
                'macd_hist': macd_hist if isinstance(macd_hist, (int, float)) else macd_hist[-1] if len(macd_hist) > 0 else 0,
                'upper_band': upper_band if isinstance(upper_band, (int, float)) else upper_band[-1] if len(upper_band) > 0 else 0,
                'middle_band': middle_band if isinstance(middle_band, (int, float)) else middle_band[-1] if len(middle_band) > 0 else 0,
                'lower_band': lower_band if isinstance(lower_band, (int, float)) else lower_band[-1] if len(lower_band) > 0 else 0
            }
            
            return indicators
        
        except Exception as e:
            # Use the module-level logger instead of self.logger
            logger.error(f"Error calculating indicators: {e}")
            return {
                'rsi': 50,
                'macd': 0,
                'macd_signal': 0,
                'macd_hist': 0,
                'upper_band': 0,
                'middle_band': 0,
                'lower_band': 0
            }

def get_technical_indicators(prices, config_override=None):
    """
    Analyze price data using technical indicators.
    
    Args:
        prices: Array of historical price data
        config_override: Optional config override
        
    Returns:
        tuple: (prediction, confidence)
    """
    try:
        # Use provided config or get from global config
        ta_config = config_override or config.get('analysis.technical', {})
        
        # Check if technical analysis is enabled
        if not ta_config.get('enable', True):
            logger.info("Technical analysis disabled in config")
            return "BULL", 0.51  # Default to slight bull bias
        
        # Get lookback periods
        lookback = ta_config.get('lookback_periods', {})
        short_period = lookback.get('short', 14)
        medium_period = lookback.get('medium', 30)
        long_period = lookback.get('long', 50)
        
        # Get indicators to use
        indicators = ta_config.get('indicators', {})
        use_rsi = indicators.get("rsi", True)
        use_macd = indicators.get("macd", True)
        use_bollinger = indicators.get("bollinger_bands", True)
        use_ema = indicators.get("ema", True)
        use_candlestick = indicators.get("candlestick", True)
        
        # Initialize signals and TA instance
        signals = []
        ta = TechnicalAnalysis()
        
        # Calculate and use indicators based on config
        if use_rsi:
            rsi = ta.rsi(np.array(prices))
            logger.info(f"📊 RSI: {rsi:.2f}")
            # More balanced RSI thresholds with slight bull bias
            if rsi < 35:  # Was 30, then 40
                signals.append(("BULL", 0.75))
            elif rsi > 70:  # Keep standard overbought level
                signals.append(("BEAR", 0.7))
        
        if use_macd:
            macd_line, signal_line, histogram = ta.macd(np.array(prices))
            logger.info(f"📊 MACD: {macd_line:.4f}, Signal: {signal_line:.4f}, Hist: {histogram:.4f}")
            if macd_line > signal_line:
                signals.append(("BULL", 0.8))
            elif macd_line < signal_line:
                signals.append(("BEAR", 0.75))
        
        if use_bollinger:
            upper, middle, lower = ta.bollinger_bands(np.array(prices))
            current_price = prices[-1]
            bb_position = (current_price - lower) / (upper - lower) if (upper - lower) > 0 else 0.5
            logger.info(f"📊 Bollinger Bands: Position {bb_position:.2f} (0=lower, 1=upper)")
            
            if current_price < lower:
                signals.append(("BULL", 0.75))
            elif current_price > upper:
                signals.append(("BEAR", 0.7))
            # Keep the middle band signal but with lower confidence
            elif abs(current_price - middle) / middle < 0.01:
                signals.append(("BULL", 0.55))
        
        if use_ema:
            # Calculate short and long EMAs
            short_ema = ta.ema(np.array(prices), 9)
            long_ema = ta.ema(np.array(prices), 21)
            logger.info(f"📊 EMA: Short(9): {short_ema:.2f}, Long(21): {long_ema:.2f}")
            
            if short_ema > long_ema:
                signals.append(("BULL", 0.7))
            elif short_ema < long_ema:
                signals.append(("BEAR", 0.7))
        
        if use_candlestick and len(prices) >= 5:
            # Try to get candlestick patterns
            try:
                pattern_signal, pattern_strength = ta.get_candlestick_patterns()
                if pattern_signal:
                    # Ensure uppercase string
                    pattern_signal = str(pattern_signal).upper()
                    signals.append((pattern_signal, pattern_strength))
                    logger.info(f"🕯️ Candlestick Pattern: {pattern_signal} (strength: {pattern_strength:.2f})")
            except Exception as e:
                logger.warning(f"⚠️ Error getting candlestick patterns: {e}")
        
        # Process signals
        if not signals:
            # Default to neutral when no signals
            return "BULL", 0.51  # Slight bull bias as fallback
            
        bull_signals = [conf for sig, conf in signals if sig == "BULL"]
        bear_signals = [conf for sig, conf in signals if sig == "BEAR"]
        
        if bull_signals and not bear_signals:
            return "BULL", sum(bull_signals) / len(bull_signals)
        elif bear_signals and not bull_signals:
            return "BEAR", sum(bear_signals) / len(bear_signals)
        elif bull_signals and bear_signals:
            bull_strength = sum(bull_signals) / len(bull_signals)
            bear_strength = sum(bear_signals) / len(bear_signals)
            
            # Add a more moderate bull market bias (10% instead of 20%)
            bull_strength *= 1.1
            
            # Print the signal strengths
            logger.info(f"📊 Signal Strengths - Bull: {bull_strength:.2f}, Bear: {bear_strength:.2f}")
            
            if bull_strength >= bear_strength:
                return "BULL", bull_strength
            else:
                return "BEAR", bear_strength
        else:
            return "BULL", 0.51  # Slight bull bias as fallback
            
    except Exception as e:
        logger.error(f"❌ Error analyzing technical indicators: {e}")
        traceback.print_exc()
        return "BULL", 0.51  # Slight bull bias as fallback 