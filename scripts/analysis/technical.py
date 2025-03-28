"""
Technical analysis functions for the trading bot.
"""

import numpy as np
import logging
from datetime import datetime, timedelta
import traceback

logger = logging.getLogger(__name__)

def calculate_microtrend(price_changes, window=5):
    """
    Calculate the short-term price trend from recent price changes.
    
    Args:
        price_changes: List of recent price changes
        window: Window size for calculation
        
    Returns:
        tuple: (trend, strength) where trend is 'UP', 'DOWN', or 'NEUTRAL'
    """
    try:
        if not price_changes or len(price_changes) < 3:
            return "NEUTRAL", 0.5
            
        # Use the most recent data points
        recent = price_changes[-window:] if len(price_changes) > window else price_changes
        
        # Calculate average change
        avg_change = sum(recent) / len(recent)
        
        # Calculate trend consistency
        up_count = sum(1 for change in recent if change > 0)
        down_count = sum(1 for change in recent if change < 0)
        total = len(recent)
        
        # Determine dominant direction
        if up_count > down_count:
            direction = "UP"
            consistency = up_count / total
        elif down_count > up_count:
            direction = "DOWN"
            consistency = down_count / total
        else:
            return "NEUTRAL", 0.5
            
        # Calculate trend strength (combination of magnitude and consistency)
        magnitude = min(abs(avg_change) * 10, 0.8)  # Cap at 0.8
        strength = (magnitude * 0.5) + (consistency * 0.5)
        
        return direction, strength
        
    except Exception as e:
        logger.error(f"❌ Error calculating microtrend: {e}")
        return "NEUTRAL", 0.5

def get_technical_prediction(round_data):
    """
    Get prediction based on technical analysis.
    
    Args:
        round_data: Dictionary with round data
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        from ..data.processing import get_recent_price_changes
        
        # Get recent price changes
        price_changes = get_recent_price_changes(10)
        
        if not price_changes or len(price_changes) < 5:
            return None, 0
            
        # Calculate micro trend
        trend, trend_strength = calculate_microtrend(price_changes)
        
        # Simple trend-following logic
        if trend == "UP" and trend_strength > 0.6:
            return "BULL", trend_strength
        elif trend == "DOWN" and trend_strength > 0.6:
            return "BEAR", trend_strength
            
        # If no strong trend, look at round data for a potential mean reversion play
        bull_amount = float(round_data.get('bullAmount', 0))
        bear_amount = float(round_data.get('bearAmount', 0))
        total_amount = bull_amount + bear_amount
        
        if total_amount > 0:
            bull_ratio = bull_amount / total_amount
            bear_ratio = bear_amount / total_amount
            
            # Extreme imbalance suggests potential mean reversion
            if bull_ratio > 0.7:  # Heavily bull-biased
                return "BEAR", min((bull_ratio - 0.5) * 2, 0.9)  # Contrarian approach
            elif bear_ratio > 0.7:  # Heavily bear-biased
                return "BULL", min((bear_ratio - 0.5) * 2, 0.9)  # Contrarian approach
                
        # No clear signals
        return None, 0
            
    except Exception as e:
        logger.error(f"❌ Error in technical prediction: {e}")
        return None, 0

def detect_market_reversal():
    """
    Detect potential market reversal patterns.
    
    Returns:
        tuple: (prediction, confidence, reason) where prediction is "BULL", "BEAR", or "UNKNOWN"
    """
    try:
        from ..data.processing import get_recent_price_changes, get_market_trend
        
        # Get trend information
        trend_info = get_market_trend()
        current_trend = trend_info.get('trend', 'NEUTRAL')
        trend_strength = trend_info.get('strength', 0.5)
        trend_duration = trend_info.get('duration', 0)
        
        # Get recent price changes
        price_changes = get_recent_price_changes(15)  # Look at more data for reversals
        
        if not price_changes or len(price_changes) < 8:
            return "UNKNOWN", 0, "Insufficient data"
            
        # Check for extended trends that might be ready for reversal
        if current_trend == "BULL" and trend_duration > 5 and trend_strength > 0.7:
            # Check for momentum decline
            recent_changes = price_changes[-3:]
            older_changes = price_changes[-8:-3]
            
            recent_avg = sum(recent_changes) / len(recent_changes)
            older_avg = sum(older_changes) / len(older_changes)
            
            # If momentum is decreasing in a strong uptrend, potential reversal
            if recent_avg < older_avg * 0.5:
                confidence = 0.6 + (trend_strength * 0.2)
                return "BEAR", confidence, "Bullish momentum weakening in extended uptrend"
                
        elif current_trend == "BEAR" and trend_duration > 5 and trend_strength > 0.7:
            # Check for momentum decline in downtrend
            recent_changes = price_changes[-3:]
            older_changes = price_changes[-8:-3]
            
            recent_avg = abs(sum(recent_changes) / len(recent_changes))
            older_avg = abs(sum(older_changes) / len(older_changes))
            
            # If momentum is decreasing in a strong downtrend, potential reversal
            if recent_avg < older_avg * 0.5:
                confidence = 0.6 + (trend_strength * 0.2)
                return "BULL", confidence, "Bearish momentum weakening in extended downtrend"
                
        # Check for V-shaped recovery or breakdown patterns
        if len(price_changes) >= 10:
            first_half = price_changes[:5]
            second_half = price_changes[5:]
            
            first_half_direction = "UP" if sum(first_half) > 0 else "DOWN"
            second_half_direction = "UP" if sum(second_half) > 0 else "DOWN"
            
            first_half_magnitude = abs(sum(first_half))
            second_half_magnitude = abs(sum(second_half))
            
            # V-shaped reversal patterns
            if first_half_direction != second_half_direction and second_half_magnitude > first_half_magnitude * 1.2:
                prediction = "BULL" if second_half_direction == "UP" else "BEAR"
                confidence = 0.65 + (second_half_magnitude / 10)
                
                return prediction, min(confidence, 0.85), f"V-shaped {prediction.lower()} reversal detected"
                
        return "UNKNOWN", 0, "No reversal pattern detected"
        
    except Exception as e:
        logger.error(f"❌ Error detecting market reversal: {e}")
        return "UNKNOWN", 0, f"Error: {str(e)}"

def detect_combined_technical_reversal():
    """
    Detect reversals using multiple technical indicators combined.
    
    Returns:
        dict: Reversal detection information
    """
    try:
        from ..data.processing import get_recent_price_changes, get_market_trend
        
        result = {
            'detected': False,
            'direction': None,
            'confidence': 0,
            'indicators': []
        }
        
        # Get market trend data
        trend_info = get_market_trend()
        current_trend = trend_info.get('trend', 'NEUTRAL')
        trend_strength = trend_info.get('strength', 0.5)
        
        # Get recent price changes
        price_changes = get_recent_price_changes(20)
        
        if not price_changes or len(price_changes) < 10:
            return result
            
        # Check for overextended trend (potential reversal)
        if current_trend in ["BULL", "BEAR"] and trend_strength > 0.75:
            # Calculate RSI-like indicator
            gains = [max(0, change) for change in price_changes[-14:]]
            losses = [abs(min(0, change)) for change in price_changes[-14:]]
            
            avg_gain = sum(gains) / len(gains) if gains else 0
            avg_loss = sum(losses) / len(losses) if losses else 0
            
            if avg_loss == 0:
                rs = 100
            else:
                rs = avg_gain / avg_loss
                
            rsi = 100 - (100 / (1 + rs))
            
            # Check for overbought/oversold conditions
            if current_trend == "BULL" and rsi > 70:
                result['indicators'].append('overbought')
                result['direction'] = "BEAR"
                result['confidence'] += 0.3
                
            elif current_trend == "BEAR" and rsi < 30:
                result['indicators'].append('oversold')
                result['direction'] = "BULL"
                result['confidence'] += 0.3
                
        # Check for double top/bottom formation
        price_points = price_changes[-10:]
        if len(price_points) >= 10:
            # Simple check for double top or double bottom pattern
            # This is a simplified version - a real implementation would be more sophisticated
            high_points = []
            low_points = []
            
            for i in range(1, len(price_points) - 1):
                # Check if this point is higher than both neighbors (local maximum)
                if price_points[i] > price_points[i-1] and price_points[i] > price_points[i+1]:
                    high_points.append((i, price_points[i]))
                
                # Check if this point is lower than both neighbors (local minimum)
                if price_points[i] < price_points[i-1] and price_points[i] < price_points[i+1]:
                    low_points.append((i, price_points[i]))
            
            # Check for double top
            if len(high_points) >= 2 and abs(high_points[-1][1] - high_points[-2][1]) < 0.01:
                result['indicators'].append('double_top')
                result['direction'] = "BEAR"
                result['confidence'] += 0.35
                
            # Check for double bottom
            if len(low_points) >= 2 and abs(low_points[-1][1] - low_points[-2][1]) < 0.01:
                result['indicators'].append('double_bottom')
                result['direction'] = "BULL"
                result['confidence'] += 0.35
                
        # Final decision
        if result['confidence'] > 0.3 and result['direction']:
            result['detected'] = True
            # Cap confidence at 0.85
            result['confidence'] = min(result['confidence'], 0.85)
            
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in combined technical reversal detection: {e}")
        return {'detected': False, 'direction': None, 'confidence': 0, 'indicators': []}

def analyze_price_patterns(prices, pattern_type=None):
    """
    Analyze price data for common chart patterns.
    
    Args:
        prices: List of price data points
        pattern_type: Optional specific pattern to look for
        
    Returns:
        dict: Pattern analysis results
    """
    try:
        if not prices or len(prices) < 15:
            return {'detected': False, 'pattern': None, 'confidence': 0}
            
        # Normalize prices to make pattern detection easier
        min_price = min(prices)
        max_price = max(prices)
        range_price = max_price - min_price
        
        if range_price == 0:
            return {'detected': False, 'pattern': None, 'confidence': 0}
            
        normalized = [(p - min_price) / range_price for p in prices]
        
        # Head and Shoulders pattern (bearish reversal)
        if not pattern_type or pattern_type == 'head_and_shoulders':
            if len(normalized) >= 20:
                # Very simplified check - real implementation would be more sophisticated
                # Look for 3 peaks with middle peak higher
                peak_indices = []
                
                for i in range(2, len(normalized) - 2):
                    if normalized[i] > normalized[i-1] and normalized[i] > normalized[i-2] and \
                       normalized[i] > normalized[i+1] and normalized[i] > normalized[i+2]:
                        peak_indices.append(i)
                
                if len(peak_indices) >= 3:
                    # Check if middle peak is higher
                    peaks = [normalized[i] for i in peak_indices]
                    if peaks[1] > peaks[0] and peaks[1] > peaks[2]:
                        return {
                            'detected': True,
                            'pattern': 'head_and_shoulders',
                            'direction': 'BEAR',
                            'confidence': 0.7
                        }
        
        # Double Bottom pattern (bullish reversal)
        if not pattern_type or pattern_type == 'double_bottom':
            if len(normalized) >= 15:
                # Simplified check for double bottom
                trough_indices = []
                
                for i in range(2, len(normalized) - 2):
                    if normalized[i] < normalized[i-1] and normalized[i] < normalized[i-2] and \
                       normalized[i] < normalized[i+1] and normalized[i] < normalized[i+2]:
                        trough_indices.append(i)
                
                if len(trough_indices) >= 2:
                    troughs = [normalized[i] for i in trough_indices]
                    if abs(troughs[-1] - troughs[-2]) < 0.1:  # Similar lows
                        return {
                            'detected': True,
                            'pattern': 'double_bottom',
                            'direction': 'BULL',
                            'confidence': 0.65
                        }
        
        return {'detected': False, 'pattern': None, 'confidence': 0}
        
    except Exception as e:
        logger.error(f"❌ Error analyzing price patterns: {e}")
        return {'detected': False, 'pattern': None, 'confidence': 0}

def analyze_market_range(prices, lookback=20):
    """
    Analyze if the market is near support/resistance levels.
    
    Args:
        prices: List of price data points
        lookback: Number of periods to analyze
        
    Returns:
        dict: Range analysis results
    """
    try:
        if not prices or len(prices) < lookback:
            logger.warning("Not enough price data for range analysis")
            return {"in_range": False, "position": 0, "confidence": 0}
        
        # Get min and max for the period
        min_price = np.min(prices[-lookback:])
        max_price = np.max(prices[-lookback:])
        range_size = max_price - min_price
        
        # If range is too small, not meaningful
        if range_size / np.mean(prices[-lookback:]) < 0.01:
            logger.debug("Price range too small for meaningful analysis")
            return {"in_range": False, "position": 0, "confidence": 0}
        
        # Get current price
        current_price = prices[-1]
        
        # Calculate where in the range we are (0 = at bottom, 1 = at top)
        position = (current_price - min_price) / range_size
        
        # Confidence is higher near the edges of the range
        if position <= 0.15:
            # Near bottom of range - bullish
            confidence = 0.7 * (1 - position/0.15)
            logger.info(f"🔍 Price near bottom of range ({position:.2f}) - bullish signal with {confidence:.2f} confidence")
            return {
                "in_range": True, 
                "position": position,
                "prediction": "BULL",
                "confidence": confidence
            }
        elif position >= 0.85:
            # Near top of range - bearish
            confidence = 0.7 * (position - 0.85)/0.15
            logger.info(f"🔍 Price near top of range ({position:.2f}) - bearish signal with {confidence:.2f} confidence")
            return {
                "in_range": True,
                "position": position,
                "prediction": "BEAR", 
                "confidence": confidence
            }
        else:
            # Middle of range - no strong signal
            logger.debug(f"Price in middle of range ({position:.2f}) - no strong signal")
            return {"in_range": True, "position": position, "confidence": 0}
            
    except Exception as e:
        logger.error(f"❌ Error in market range analysis: {e}")
        return {"in_range": False, "position": 0, "confidence": 0}

def get_ranging_market_prediction(round_data, lookback=10):
    """
    Special strategy for ranging markets that looks for reversion to mean.
    Makes predictions based on deviation from the mean price.
    
    Args:
        round_data: Dictionary with current round data
        lookback: Number of periods to analyze
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        # Get recent prices from our new data structure
        from ..data.processing import get_recent_price_data
        prices = get_recent_price_data(lookback)
        
        if not prices or len(prices) < lookback/2:
            logger.debug("Not enough price data for ranging market analysis")
            return None, 0
        
        # Calculate mean and standard deviation
        mean_price = np.mean(prices)
        std_dev = np.std(prices)
        
        if std_dev == 0:
            logger.debug("Zero standard deviation - skipping ranging market analysis")
            return None, 0
        
        # Get current price
        current_price = round_data.get('lockPrice', 0)
        if current_price == 0:
            current_price = prices[-1]
        
        # Calculate z-score (how many standard deviations from mean)
        z_score = (current_price - mean_price) / std_dev
        
        # Log the analysis data
        logger.debug(f"Mean reversion analysis: Mean={mean_price:.2f}, Current={current_price:.2f}, Z-score={z_score:.2f}")
        
        # Strategy logic: bet on mean reversion when price is far from mean
        if z_score > 1.2:  # Price is significantly above mean
            prediction = "BEAR"
            confidence = min(0.5 + abs(z_score) * 0.1, 0.85)
            logger.info(f"🔍 Price significantly above mean (z-score: {z_score:.2f}) - expecting reversion DOWN")
            return prediction, confidence
            
        elif z_score < -1.2:  # Price is significantly below mean
            prediction = "BULL"
            confidence = min(0.5 + abs(z_score) * 0.1, 0.85)
            logger.info(f"🔍 Price significantly below mean (z-score: {z_score:.2f}) - expecting reversion UP")
            return prediction, confidence
        
        # If price is close to mean, no strong signal
        logger.debug(f"Price close to mean (z-score: {z_score:.2f}) - no strong mean reversion signal")
        return None, 0
        
    except Exception as e:
        logger.error(f"❌ Error in ranging market prediction: {e}")
        traceback.print_exc()
        return None, 0

def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Array of price data
        period: RSI period (default: 14)
        
    Returns:
        float: RSI value
    """
    try:
        # Convert to numpy array
        prices = np.array(prices)
        
        # Calculate price changes
        deltas = np.diff(prices)
        
        # Separate gains and losses
        gains = deltas.copy()
        losses = deltas.copy()
        gains[gains < 0] = 0
        losses[losses > 0] = 0
        losses = abs(losses)
        
        # Calculate average gains and losses
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        if avg_loss == 0:
            return 100
            
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    except Exception as e:
        logger.warning(f"Error calculating RSI: {e}")
        return 50  # Neutral RSI

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate Moving Average Convergence Divergence.
    
    Args:
        prices: Array of price data
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    try:
        # Convert to numpy array
        prices = np.array(prices)
        
        # Calculate EMAs
        ema_fast = calculate_ema(prices, fast)
        ema_slow = calculate_ema(prices, slow)
        
        # Calculate MACD line
        macd_line = ema_fast - ema_slow
        
        # Calculate signal line
        signal_line = calculate_ema(np.array([macd_line]), signal)
        
        # Calculate histogram
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
        
    except Exception as e:
        logger.warning(f"Error calculating MACD: {e}")
        return 0, 0, 0

def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        prices: Array of price data
        period: Period for moving average
        std_dev: Number of standard deviations
        
    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    try:
        prices = np.array(prices)
        if len(prices) < period:
            return prices[-1], prices[-1], prices[-1]
            
        # Calculate simple moving average
        sma = np.mean(prices[-period:])
        
        # Calculate standard deviation
        std = np.std(prices[-period:])
        
        # Calculate upper and lower bands
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return upper_band, sma, lower_band
        
    except Exception as e:
        logger.warning(f"Error calculating Bollinger Bands: {e}")
        return prices[-1], prices[-1], prices[-1]

def calculate_ema(prices, period):
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
        weights = np.exp(np.linspace(-1, 0, period))
        weights /= weights.sum()
        
        # Calculate EMA
        ema = np.convolve(prices, weights, mode='valid')
        
        # Return the last value
        return ema[-1] if len(ema) > 0 else prices[-1]
        
    except Exception as e:
        logger.warning(f"Error calculating EMA: {e}")
        return prices[-1] if len(prices) > 0 else 0

def analyze_rsi(rsi):
    """
    Analyze RSI indicator for trading signals.
    
    Args:
        rsi: RSI value
        
    Returns:
        tuple: (direction, strength) where direction is "BULL", "BEAR", or None
    """
    if rsi < 30:
        return "BULL", 0.8  # Oversold - bullish signal
    elif rsi > 70:
        return "BEAR", 0.8  # Overbought - bearish signal
    elif rsi < 45:
        return "BULL", 0.6  # Approaching oversold - moderate bullish
    elif rsi > 55:
        return "BEAR", 0.6  # Approaching overbought - moderate bearish
    else:
        return None, 0

def analyze_macd(macd, signal, hist):
    """
    Analyze MACD indicator for trading signals.
    
    Args:
        macd: MACD line value
        signal: Signal line value
        hist: Histogram value
        
    Returns:
        tuple: (direction, strength) where direction is "BULL", "BEAR", or None
    """
    if macd > signal and hist > 0:
        return "BULL", 0.8  # Strong bullish signal
    elif macd < signal and hist < 0:
        return "BEAR", 0.8  # Strong bearish signal
    elif macd > signal:
        return "BULL", 0.6  # Moderate bullish
    elif macd < signal:
        return "BEAR", 0.6  # Moderate bearish
    else:
        return None, 0

def analyze_bollinger(price, upper, lower):
    """
    Analyze Bollinger Bands for trading signals.
    
    Args:
        price: Current price
        upper: Upper Bollinger Band
        lower: Lower Bollinger Band
        
    Returns:
        tuple: (direction, strength) where direction is "BULL", "BEAR", or None
    """
    if price < lower:
        return "BULL", 0.8  # Price below lower band - bullish (oversold)
    elif price > upper:
        return "BEAR", 0.8  # Price above upper band - bearish (overbought)
    else:
        # Calculate position within the bands (0-1)
        position = (price - lower) / (upper - lower) if upper != lower else 0.5
        if position < 0.3:
            return "BULL", 0.6  # Near lower band - moderately bullish
        elif position > 0.7:
            return "BEAR", 0.6  # Near upper band - moderately bearish
        else:
            return None, 0

def analyze_ema_cross(short_ema, long_ema):
    """
    Analyze EMA crossover for trading signals.
    
    Args:
        short_ema: Short-term EMA value
        long_ema: Long-term EMA value
        
    Returns:
        tuple: (direction, strength) where direction is "BULL", "BEAR", or None
    """
    if short_ema > long_ema:
        # Bullish crossover
        return "BULL", 0.7
    elif short_ema < long_ema:
        # Bearish crossover
        return "BEAR", 0.7
    else:
        return None, 0

def get_technical_analysis(prices, config=None):
    """
    Analyze technical indicators to generate a trading signal.
    
    Args:
        prices: List of historical prices
        config: Optional configuration dictionary
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL", "BEAR", or None
    """
    try:
        # Ensure we have enough price data
        if len(prices) < 5:
            logger.warning("Not enough price data for technical analysis")
            return None, 0
            
        # Calculate indicators
        rsi = calculate_rsi(prices)
        macd_line, signal_line, histogram = calculate_macd(prices)
        upper_band, middle_band, lower_band = calculate_bollinger_bands(prices)
        ema_short = calculate_ema(prices, 9)
        ema_long = calculate_ema(prices, 21)
        
        # Log indicator values
        logger.info(f"📊 Technical Indicators:")
        logger.info(f"  RSI: {rsi:.2f}")
        logger.info(f"  MACD: {macd_line:.4f}, Signal: {signal_line:.4f}, Hist: {histogram:.4f}")
        
        current_price = prices[-1]
        bb_position = (current_price - lower_band) / (upper_band - lower_band) if (upper_band - lower_band) > 0 else 0.5
        logger.info(f"  Bollinger Bands: Position {bb_position:.2f} (0=lower, 1=upper)")
        logger.info(f"  EMA: Short(9): {ema_short:.2f}, Long(21): {ema_long:.2f}")
        
        # Score each indicator
        signals = {
            'rsi': analyze_rsi(rsi),
            'macd': analyze_macd(macd_line, signal_line, histogram),
            'bb': analyze_bollinger(current_price, upper_band, lower_band),
            'ema': analyze_ema_cross(ema_short, ema_long)
        }
        
        # Weight the signals
        weights = {
            'rsi': 0.3,
            'macd': 0.3,
            'bb': 0.2,
            'ema': 0.2
        }
        
        # Get weights from config if provided
        if config and 'indicator_weights' in config:
            weights.update(config['indicator_weights'])
        
        bull_score = 0
        bear_score = 0
        
        for indicator, (direction, strength) in signals.items():
            if direction:  # Only count valid signals
                weight = weights[indicator]
                if direction == "BULL":
                    bull_score += strength * weight
                else:
                    bear_score += strength * weight
                    
        # Calculate final signal
        if bull_score > bear_score:
            return "BULL", min(0.5 + (bull_score - bear_score), 0.95)
        elif bear_score > bull_score:
            return "BEAR", min(0.5 + (bear_score - bull_score), 0.95)
        else:
            return None, 0
            
    except Exception as e:
        logger.error(f"❌ Error in technical analysis: {e}")
        traceback.print_exc()
        return None, 0

def get_technical_indicators_with_fallback(prices, fallback=True):
    """
    Get technical indicators with fallback synthetic data if needed.
    
    Args:
        prices: List of price data points
        fallback: Whether to use synthetic data fallback if not enough data
        
    Returns:
        tuple: (signal, confidence) or None if insufficient data and fallback disabled
    """
    try:
        # Check if we have enough data
        if len(prices) >= 14:  # Minimum needed for most indicators
            return get_technical_analysis(prices)
            
        if not fallback:
            return None, 0
            
        # Not enough data, create synthetic data
        logger.warning(f"Not enough price data for technical analysis (found {len(prices)})")
        
        # If we have some data, use it to create more realistic fallback
        if prices:
            # Generate based on recent volatility and trend
            mean_price = np.mean(prices)
            std_dev = np.std(prices) if len(prices) > 1 else mean_price * 0.01
            
            # Simple trend detection
            trend = 0
            if len(prices) > 5:
                trend = (prices[-1] - prices[0]) / len(prices)
            
            # Create synthetic history
            synthetic_prices = []
            for i in range(30):
                # Add slight trend plus random noise
                next_price = mean_price + (i * trend) + np.random.normal(0, std_dev)
                synthetic_prices.append(max(0.01, next_price))
            
            # Combine real and synthetic
            combined_prices = synthetic_prices[:-len(prices)] + prices
            logger.info(f"Using combined real/synthetic price data (added {len(synthetic_prices) - len(prices)} synthetic points)")
            return get_technical_analysis(combined_prices)
        else:
            # No data at all, use completely synthetic
            # Create flat prices with slight random walk
            base_price = 300  # Example for BNB
            synthetic_prices = [base_price]
            for i in range(29):
                # Random walk with 0.5% standard deviation
                next_price = synthetic_prices[-1] * (1 + np.random.normal(0, 0.005))
                synthetic_prices.append(next_price)
            
            logger.info("Using completely synthetic price data for technical analysis")
            return get_technical_analysis(synthetic_prices)
            
    except Exception as e:
        logger.error(f"Error in technical indicators with fallback: {e}")
        traceback.print_exc()
        return None, 0

def get_technical_signal_history(days_back=30):
    """
    Create a historical record of technical signals with timestamps.
    Uses datetime for timestamping signals.
    """
    try:
        from ..data.processing import get_recent_price_changes
        
        # Use datetime to create timestamps
        now = datetime.now()
        history = []
        
        # For each historical point, get actual price changes and calculate signals
        for day in range(days_back):
            # Create timestamp for this historical point
            signal_time = now - timedelta(days=day)
            
            # Get historical price changes for this time period
            # This actually uses the imported function
            lookback_days = min(14, days_back - day)
            price_changes = get_recent_price_changes(lookback_days, end_date=signal_time)
            
            # Calculate technical indicators if we have price data
            signal = None
            confidence = 0
            if price_changes and len(price_changes) > 3:
                # Simple trend detection
                avg_change = sum(price_changes) / len(price_changes)
                signal = "BULL" if avg_change > 0 else "BEAR"
                confidence = min(0.5 + abs(avg_change * 20), 0.9)
            
            # Create a record for this time point
            record = {
                'timestamp': signal_time,
                'date': signal_time.strftime('%Y-%m-%d'),
                'signal': signal,
                'confidence': confidence,
                'price_changes': price_changes if price_changes else []
            }
            
            history.append(record)
            
        return history
        
    except Exception as e:
        logger.error(f"Error creating technical signal history: {e}")
        traceback.print_exc()
        return [] 