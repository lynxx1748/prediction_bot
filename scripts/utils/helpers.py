"""
Helper functions for the trading bot.
"""

import time
import numpy as np
import threading
import logging

# Try to import keyboard but provide a fallback if not available
try:
    import keyboard  # type: ignore
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    print("ℹ️ Keyboard module not available - ESC key detection disabled")

logger = logging.getLogger(__name__)

def get_price_trend(prices=None, lookback=8):
    """
    Calculate the price trend from historical data.
    
    Args:
        prices: Array of price data (if None, fetches from database)
        lookback: Number of data points to consider
        
    Returns:
        tuple: (trend, strength) where trend is 'up', 'down', or 'neutral'
    """
    try:
        if prices is None or len(prices) < 2:
            # If no prices provided, fetch from database
            from ..analysis.market import get_historical_prices
            prices = get_historical_prices(lookback)
            
        if not prices or len(prices) < 2:
            return "neutral", 0.0
            
        # Calculate trend
        price_changes = np.array(prices)
        
        # Calculate smoothed average change
        avg_change = np.mean(price_changes)
        
        # Calculate trend strength based on consistency
        changes_sign = np.sign(price_changes)
        consistence = np.sum(changes_sign == np.sign(avg_change)) / len(changes_sign)
        
        # Strength combines magnitude and consistency
        strength = min(abs(avg_change) * 20, 0.7) + (consistence * 0.3)
        strength = min(strength, 0.95)  # Cap at 0.95
        
        if avg_change > 0.0001:
            return "up", strength
        elif avg_change < -0.0001:
            return "down", strength
        else:
            return "neutral", 0.0
            
    except Exception as e:
        logger.error(f"❌ Error calculating price trend: {e}")
        return "neutral", 0.0

def sleep_and_check_for_interruption(seconds):
    """
    Sleep for specified time but check for keyboard interrupt.
    
    Args:
        seconds: Time to sleep in seconds
        
    Returns:
        bool: True if interrupted, False otherwise
    """
    interrupted = threading.Event()
    
    def check_for_esc():
        try:
            if KEYBOARD_AVAILABLE and keyboard.is_pressed('esc'):
                logger.info("ESC key pressed, interrupting...")
                interrupted.set()
        except Exception:
            pass
    
    # If keyboard module is not available, just sleep
    if not KEYBOARD_AVAILABLE:
        time.sleep(seconds)
        return False
        
    end_time = time.time() + seconds
    
    # Check every 0.1 seconds for interruption
    while time.time() < end_time and not interrupted.is_set():
        check_for_esc()
        time.sleep(0.1)
        
    return interrupted.is_set()

def get_historical_data(epoch, limit=10):
    """
    Get historical data for a range of epochs.
    
    Args:
        epoch: Current epoch number
        limit: Number of previous epochs to fetch
        
    Returns:
        list: List of historical data points
    """
    try:
        from ..data.database import get_recent_rounds
        
        # Ensure epoch is an integer
        epoch = int(epoch)
        
        # Get data for specified range
        history = get_recent_rounds(limit)
        
        return history
        
    except Exception as e:
        logger.error(f"❌ Error getting historical data: {e}")
        return []

def format_price_change(change):
    """
    Format price change for display.
    
    Args:
        change: Price change value
        
    Returns:
        str: Formatted price change
    """
    if change > 0:
        return f"🟢 +{change:.2f}%"
    elif change < 0:
        return f"🔴 {change:.2f}%"
    else:
        return f"⚪ {change:.2f}%"

def calculate_price_change(start_price, end_price):
    """
    Calculate percentage price change.
    
    Args:
        start_price: Starting price
        end_price: Ending price
        
    Returns:
        float: Percentage change
    """
    if start_price == 0:
        return 0
    return ((end_price - start_price) / start_price) * 100

def get_recent_outcomes(n=10):
    """
    Get recent market outcomes.
    
    Args:
        n: Number of outcomes to fetch
        
    Returns:
        list: List of recent outcomes ('bull' or 'bear')
    """
    try:
        from ..data.database import get_recent_rounds
        
        # Get recent rounds
        rounds = get_recent_rounds(n)
        
        # Extract outcomes
        outcomes = []
        for r in rounds:
            if 'closePrice' in r and 'lockPrice' in r:
                if r['closePrice'] > r['lockPrice']:
                    outcomes.append('bull')
                else:
                    outcomes.append('bear')
                    
        return outcomes
        
    except Exception as e:
        logger.error(f"❌ Error getting recent outcomes: {e}")
        return []

def optimize_swing_trading(recent_performance, market_stats, consecutive_losses=0):
    """
    Optimize betting parameters for swing trading strategy based on recent performance.
    
    Args:
        recent_performance: Recent performance metrics
        market_stats: Market statistics data
        consecutive_losses: Current streak of consecutive losses
        
    Returns:
        dict: Optimized betting parameters
    """
    try:
        # Default parameters
        params = {
            'enabled': True,
            'confidence_threshold': 0.65,
            'min_bet_amount': 10,
            'max_bet_amount': 50,
            'martingale_factor': 1.0,
            'risk_factor': 0.5,
            'skip_if_trend_against': False
        }
        
        # Extract key metrics
        win_rate = recent_performance.get('win_rate', 0.5)
        bull_win_rate = recent_performance.get('bull_win_rate', 0.5)
        bear_win_rate = recent_performance.get('bear_win_rate', 0.5)
        sample_size = recent_performance.get('sample_size', 0)
        is_valid = recent_performance.get('valid', False)
        
        # Market balance and trend
        bull_ratio = market_stats.get('bull_ratio', 0.5) if market_stats else 0.5
        
        # If we don't have enough data, use default conservative settings
        if not is_valid or sample_size < 10:
            logger.info("📊 Not enough data for swing trade optimization, using default parameters")
            return params
            
        # Adjust confidence threshold based on win rate
        if win_rate > 0.65:
            params['confidence_threshold'] = 0.6  # Lower threshold if win rate is high
        elif win_rate < 0.45:
            params['confidence_threshold'] = 0.75  # Higher threshold if win rate is low
            
        # Adjust risk factor based on overall performance
        if win_rate > 0.55:
            params['risk_factor'] = 0.6  # Slightly more aggressive if performing well
        elif win_rate < 0.45:
            params['risk_factor'] = 0.3  # More conservative if performing poorly
            
        # Adjust martingale factor based on consecutive losses
        if consecutive_losses == 0:
            params['martingale_factor'] = 1.0
        elif consecutive_losses == 1:
            params['martingale_factor'] = 1.5
        elif consecutive_losses == 2:
            params['martingale_factor'] = 2.0
        elif consecutive_losses >= 3:
            params['martingale_factor'] = 2.5
            
        # Limit maximum martingale factor to prevent excessive betting
        params['martingale_factor'] = min(params['martingale_factor'], 3.0)
        
        # Adjust bull/bear specific parameters
        if abs(bull_win_rate - bear_win_rate) > 0.2:
            # There's a significant difference in bull vs bear prediction accuracy
            if bull_win_rate > bear_win_rate:
                logger.info(f"📊 BULL predictions more accurate ({bull_win_rate:.2f} vs {bear_win_rate:.2f})")
                params['skip_if_trend_against'] = bull_win_rate < 0.5  # Skip if BULL accuracy below 50%
            else:
                logger.info(f"📊 BEAR predictions more accurate ({bear_win_rate:.2f} vs {bull_win_rate:.2f})")
                params['skip_if_trend_against'] = bear_win_rate < 0.5  # Skip if BEAR accuracy below 50%
                
        # Factor in market balance (avoid betting against strong trends)
        if bull_ratio > 0.65:
            logger.info(f"📊 Strong BULL market detected ({bull_ratio:.2f})")
            # Increase threshold for BEAR bets in a bull market
            params['bear_confidence_threshold'] = params['confidence_threshold'] + 0.1
        elif bull_ratio < 0.35:
            logger.info(f"📊 Strong BEAR market detected ({1-bull_ratio:.2f})")
            # Increase threshold for BULL bets in a bear market
            params['bull_confidence_threshold'] = params['confidence_threshold'] + 0.1
            
        logger.info(f"📊 Swing trading parameters optimized: threshold={params['confidence_threshold']}, risk={params['risk_factor']}")
        return params
        
    except Exception as e:
        logger.error(f"❌ Error optimizing swing trading: {e}")
        # Return conservative default parameters
        return {
            'enabled': True,
            'confidence_threshold': 0.7,
            'min_bet_amount': 10,
            'max_bet_amount': 30,
            'martingale_factor': 1.0,
            'risk_factor': 0.4,
            'skip_if_trend_against': False
        }

def detect_swing_pattern(price_changes, market_stats=None):
    """
    Detect swing trading patterns from price changes.
    
    Args:
        price_changes: List of recent price changes
        market_stats: Optional market statistics
        
    Returns:
        dict: Swing pattern detection result
    """
    try:
        # Default result structure
        result = {
            'swing_opportunity': False,
            'direction': None,
            'confidence': 0.0
        }
        
        # If not enough data, return default result
        if not price_changes or not isinstance(price_changes, list) or len(price_changes) < 3:
            return result
            
        # Basic trend detection
        up_count = sum(1 for change in price_changes[:3] if change > 0)
        down_count = sum(1 for change in price_changes[:3] if change < 0)
        
        # Look for reversal opportunities
        if up_count >= 3:  # Three consecutive up movements
            result['swing_opportunity'] = True
            result['direction'] = "BEAR"  # Contrarian approach
            result['confidence'] = 0.65
        elif down_count >= 3:  # Three consecutive down movements
            result['swing_opportunity'] = True
            result['direction'] = "BULL"  # Contrarian approach 
            result['confidence'] = 0.65
        
        # Get market bias from stats if available
        if market_stats and isinstance(market_stats, dict):
            bull_ratio = market_stats.get('bull_ratio', 0.5)
            if bull_ratio > 0.7:  # Strong bull market - contrarian
                result['swing_opportunity'] = True
                result['direction'] = "BEAR" 
                result['confidence'] = 0.6
            elif bull_ratio < 0.3:  # Strong bear market - contrarian
                result['swing_opportunity'] = True
                result['direction'] = "BULL"
                result['confidence'] = 0.6
                
        return result
            
    except Exception as e:
        logger.error(f"❌ Error detecting swing pattern: {e}")
        return {'swing_opportunity': False, 'direction': None, 'confidence': 0.0} 