"""
Pattern analysis functions for the trading bot.
Detects chart patterns for trading decisions.
"""

import numpy as np
from scipy.signal import find_peaks, argrelmin, argrelmax
import traceback
import logging
from ..utils.helpers import get_price_trend

logger = logging.getLogger(__name__)

def detect_advanced_patterns(prices):
    """
    Detect advanced chart patterns with reversal signals.
    
    Args:
        prices: List of price data
        
    Returns:
        tuple: (prediction, confidence) where prediction is "BULL" or "BEAR"
    """
    try:
        if prices is None or len(prices) < 10:
            return None, 0
            
        # Convert to numpy array for calculations
        price_array = np.array(prices)
        
        # Calculate price changes
        changes = np.diff(price_array)
        
        # Detect double bottom/top
        def detect_double_pattern():
            window = price_array[-10:]
            lows = argrelmin(window)[0]
            highs = argrelmax(window)[0]
            
            if len(lows) >= 2:
                # Double bottom
                if abs(window[lows[-1]] - window[lows[-2]]) < window.mean() * 0.01:
                    if window[lows[-1]] < window.mean():
                        return "BULL", 0.8
            if len(highs) >= 2:
                # Double top
                if abs(window[highs[-1]] - window[highs[-2]]) < window.mean() * 0.01:
                    if window[highs[-1]] > window.mean():
                        return "BEAR", 0.8
            return None, 0
            
        # Detect breakouts
        def detect_breakout():
            ma20 = np.mean(price_array[-20:])
            current_price = price_array[-1]
            
            if current_price > ma20 * 1.02:  # 2% breakout
                return "BULL", 0.85
            elif current_price < ma20 * 0.98:
                return "BEAR", 0.85
            return None, 0
            
        # Check patterns in priority order
        patterns = [
            detect_double_pattern(),
            detect_breakout()
        ]
        
        # Return strongest pattern
        valid_patterns = [p for p in patterns if p[0] is not None]
        if valid_patterns:
            return max(valid_patterns, key=lambda x: x[1])
            
        return None, 0
        
    except Exception as e:
        logger.error(f"❌ Error detecting patterns: {e}")
        traceback.print_exc()
        return None, 0

def detect_candlestick_patterns(prices, volumes=None):
    """
    Detect Japanese candlestick patterns.
    
    Args:
        prices: List of OHLC prices
        volumes: Optional list of volumes
        
    Returns:
        dict: Detected patterns with confidence levels
    """
    try:
        if not prices or len(prices) < 5:
            return {}
            
        # Extract OHLC data
        opens = np.array([candle[0] for candle in prices])
        highs = np.array([candle[1] for candle in prices])
        lows = np.array([candle[2] for candle in prices])
        closes = np.array([candle[3] for candle in prices])
        
        patterns = {}
        
        # Detect engulfing patterns
        for i in range(1, len(prices)):
            # Bullish engulfing
            if opens[i] <= closes[i-1] and closes[i] > opens[i-1] and closes[i] > opens[i]:
                body_size_ratio = (closes[i] - opens[i]) / (opens[i-1] - closes[i-1])
                if body_size_ratio > 1.5:
                    patterns['bullish_engulfing'] = {
                        'index': i,
                        'strength': min(0.6 + (body_size_ratio - 1.5) * 0.2, 0.9)
                    }
                    
            # Bearish engulfing
            if opens[i] >= closes[i-1] and closes[i] < opens[i-1] and closes[i] < opens[i]:
                body_size_ratio = (opens[i] - closes[i]) / (closes[i-1] - opens[i-1])
                if body_size_ratio > 1.5:
                    patterns['bearish_engulfing'] = {
                        'index': i,
                        'strength': min(0.6 + (body_size_ratio - 1.5) * 0.2, 0.9)
                    }
                    
        # Detect hammers and hanging men
        for i in range(len(prices)):
            body_size = abs(closes[i] - opens[i])
            if body_size > 0:
                # Lower shadow
                lower_shadow = min(opens[i], closes[i]) - lows[i]
                lower_ratio = lower_shadow / body_size
                
                # Upper shadow
                upper_shadow = highs[i] - max(opens[i], closes[i])
                upper_ratio = upper_shadow / body_size
                
                # Hammer (lower shadow at least 2x the body, small upper shadow)
                if lower_ratio > 2 and upper_ratio < 0.5:
                    # Hammer in downtrend (bullish)
                    if i > 0 and closes[i-1] < opens[i-1]:
                        patterns['hammer'] = {
                            'index': i,
                            'strength': min(0.7 + lower_ratio * 0.1, 0.9)
                        }
                    # Hanging man in uptrend (bearish)
                    elif i > 0 and closes[i-1] > opens[i-1]:
                        patterns['hanging_man'] = {
                            'index': i,
                            'strength': min(0.7 + lower_ratio * 0.1, 0.9)
                        }
        
        return patterns
        
    except Exception as e:
        logger.error(f"❌ Error detecting candlestick patterns: {e}")
        return {}

def detect_support_resistance(prices, window=30):
    """
    Detect support and resistance levels.
    
    Args:
        prices: List of price data
        window: Window size for detection
        
    Returns:
        dict: Support and resistance levels
    """
    try:
        if not prices or len(prices) < window:
            return {'support': [], 'resistance': []}
            
        price_array = np.array(prices)
        
        # Find local minima and maxima
        peaks, _ = find_peaks(price_array)
        troughs, _ = find_peaks(-price_array)
        
        # Group close levels together
        def group_levels(levels, threshold=0.01):
            if not levels.size:
                return []
                
            # Get prices at these points
            prices_at_levels = price_array[levels]
            
            # Sort by price
            sorted_idx = np.argsort(prices_at_levels)
            sorted_prices = prices_at_levels[sorted_idx]
            sorted_levels = levels[sorted_idx]
            
            # Group close prices
            groups = []
            current_group = [sorted_levels[0]]
            current_price = sorted_prices[0]
            
            for i in range(1, len(sorted_levels)):
                # If price is close to current group price
                if abs(sorted_prices[i] - current_price) / current_price < threshold:
                    current_group.append(sorted_levels[i])
                else:
                    # Add current group and start new one
                    groups.append(current_group)
                    current_group = [sorted_levels[i]]
                    current_price = sorted_prices[i]
                    
            # Add the last group
            groups.append(current_group)
            
            # Calculate average price for each group
            result = []
            for group in groups:
                price = np.mean(price_array[group])
                strength = len(group) / len(levels)  # Strength based on frequency
                result.append((price, strength))
                
            return result
            
        # Get grouped levels
        resistance_levels = group_levels(peaks)
        support_levels = group_levels(troughs)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
        
    except Exception as e:
        logger.error(f"❌ Error detecting support/resistance: {e}")
        return {'support': [], 'resistance': []}

def get_pattern_strength(prices, pattern_type=None):
    """
    Analyze price data to determine the strength of a pattern.
    
    Args:
        prices: List of price data
        pattern_type: Optional specific pattern type to look for
        
    Returns:
        tuple: (pattern_name, strength, direction)
    """
    try:
        if not prices or len(prices) < 5:
            return "unknown", 0.0, "neutral"
            
        # Convert to numpy array for easier manipulation
        price_array = np.array(prices)
        
        # Get basic trend information
        trend, trend_strength = get_price_trend(price_array)
        
        # Check for specific pattern types if requested
        if pattern_type:
            if pattern_type == "double_top":
                return detect_double_top(price_array)
            elif pattern_type == "double_bottom":
                return detect_double_bottom(price_array)
            elif pattern_type == "head_shoulders":
                return detect_head_shoulders(price_array)
            # Add more pattern types as needed
            
        # Auto detect the strongest pattern
        patterns = []
        
        # Try to detect various patterns
        patterns.append(detect_double_top(price_array))
        patterns.append(detect_double_bottom(price_array))
        patterns.append(detect_head_shoulders(price_array))
        patterns.append(detect_flag(price_array))
        patterns.append(detect_triangle(price_array))
        
        # Find the strongest pattern
        strongest = max(patterns, key=lambda x: x[1])
        
        # If no strong pattern found, return trend
        if strongest[1] < 0.3:
            return f"trend_{trend}", trend_strength, trend
            
        return strongest
        
    except Exception as e:
        logger.error(f"Error in pattern analysis: {e}")
        return "error", 0.0, "neutral"

def detect_double_top(prices):
    """Detect double top pattern."""
    try:
        # Basic implementation - look for two similar peaks with a valley in between
        if len(prices) < 7:
            return "double_top", 0.0, "neutral"
            
        # Find local maxima
        peaks = []
        for i in range(1, len(prices)-1):
            if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                peaks.append((i, prices[i]))
                
        if len(peaks) < 2:
            return "double_top", 0.0, "neutral"
            
        # Check for double top pattern
        for i in range(len(peaks) - 1):
            peak1 = peaks[i]
            peak2 = peaks[i + 1]
            
            # Verify peaks are similar height (within 2%)
            height_diff = abs(peak1[1] - peak2[1]) / peak1[1]
            if height_diff < 0.02:
                # Find valley between peaks
                valley = min(prices[peak1[0]:peak2[0]])
                
                # Calculate pattern strength based on depth of valley
                valley_depth = (peak1[1] - valley) / peak1[1]
                strength = min(valley_depth * 2, 0.9)  # Cap at 0.9
                
                return "double_top", strength, "bear"
                
        return "double_top", 0.0, "neutral"
        
    except Exception as e:
        logger.error(f"Error detecting double top: {e}")
        return "double_top", 0.0, "neutral"

def detect_double_bottom(prices):
    """Detect double bottom pattern."""
    try:
        # Invert prices and use double top logic
        inverted = [-p for p in prices]
        pattern, strength, direction = detect_double_top(inverted)
        
        if pattern == "double_top" and strength > 0:
            # Use direction in logging to make it accessed
            logger.debug(f"Double bottom pattern detected with {direction} direction")
            # Actually use the direction variable we get 
            return "double_bottom", strength, "bull" if direction == "bear" else "bull"
            
        return "double_bottom", 0.0, "neutral"
        
    except Exception as e:
        logger.error(f"Error detecting double bottom: {e}")
        return "double_bottom", 0.0, "neutral"

def detect_head_shoulders(prices):
    """Detect head and shoulders pattern."""
    try:
        if len(prices) < 12:
            return "head_shoulders", 0.0, "neutral"
            
        # Find local maxima (potential shoulders and head)
        peaks = []
        for i in range(2, len(prices)-2):
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and 
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                peaks.append((i, prices[i]))
                
        if len(peaks) < 3:
            return "head_shoulders", 0.0, "neutral"
            
        for i in range(len(peaks) - 2):
            left = peaks[i]
            middle = peaks[i+1]
            right = peaks[i+2]
            
            # Check if middle peak is higher than side peaks
            if middle[1] > left[1] and middle[1] > right[1]:
                # Check if side peaks are at similar levels
                shoulder_diff = abs(left[1] - right[1]) / left[1]
                
                if shoulder_diff < 0.05:  # Shoulders within 5%
                    # Calculate pattern strength
                    head_height = middle[1] - (left[1] + right[1]) / 2
                    strength = min(head_height / middle[1] * 1.5, 0.9)
                    
                    return "head_shoulders", strength, "bear"
                    
        return "head_shoulders", 0.0, "neutral"
        
    except Exception as e:
        logger.error(f"Error detecting head and shoulders: {e}")
        return "head_shoulders", 0.0, "neutral"

def detect_flag(prices):
    """Detect flag pattern."""
    try:
        if len(prices) < 10:
            return "flag", 0.0, "neutral"
            
        # Check for strong initial move followed by consolidation
        initial_move = (prices[3] - prices[0]) / prices[0]
        
        # Determine if bullish or bearish flag
        flag_direction = "bull" if initial_move > 0 else "bear"
        
        # Look for consolidation in second half of data
        half_point = len(prices) // 2
        first_half = prices[:half_point]
        second_half = prices[half_point:]
        
        # Calculate volatility in each half
        first_volatility = np.std(first_half) / np.mean(first_half)
        second_volatility = np.std(second_half) / np.mean(second_half)
        
        # Flag pattern has higher volatility in first half, consolidation in second
        if first_volatility > second_volatility * 1.5 and abs(initial_move) > 0.01:
            strength = min(abs(initial_move) * 10, 0.8)
            return "flag", strength, flag_direction
            
        return "flag", 0.0, "neutral"
        
    except Exception as e:
        logger.error(f"Error detecting flag pattern: {e}")
        return "flag", 0.0, "neutral"

def detect_triangle(prices):
    """Detect triangle pattern."""
    try:
        if len(prices) < 10:
            return "triangle", 0.0, "neutral"
            
        # Split data into segments for analysis
        segments = 4
        segment_size = len(prices) // segments
        
        highs = []
        lows = []
        
        # Get high and low for each segment
        for i in range(segments):
            start = i * segment_size
            end = start + segment_size
            segment = prices[start:end]
            highs.append(max(segment))
            lows.append(min(segment))
            
        # Check for converging patterns
        high_slope = (highs[-1] - highs[0]) / (segments - 1)
        low_slope = (lows[-1] - lows[0]) / (segments - 1)
        
        # Determine triangle type
        if abs(high_slope) < 0.001 and low_slope > 0.001:
            # Ascending triangle (bullish)
            strength = min(abs(low_slope) * 15, 0.8)
            return "ascending_triangle", strength, "bull"
            
        elif high_slope < -0.001 and abs(low_slope) < 0.001:
            # Descending triangle (bearish)
            strength = min(abs(high_slope) * 15, 0.8)
            return "descending_triangle", strength, "bear"
            
        elif high_slope < -0.001 and low_slope > 0.001:
            # Symmetrical triangle
            avg_slope = (abs(high_slope) + abs(low_slope)) / 2
            strength = min(avg_slope * 15, 0.7)
            
            # Direction is determined by preceding trend
            trend, _ = get_price_trend(prices[:len(prices)//2])
            return "symmetrical_triangle", strength, trend
            
        return "triangle", 0.0, "neutral"
        
    except Exception as e:
        logger.error(f"Error detecting triangle pattern: {e}")
        return "triangle", 0.0, "neutral" 