"""
Trade timing optimization for the trading bot.
Calculates optimal entry times based on market conditions and round timing.
"""

import time
import logging
import traceback
from functools import lru_cache

logger = logging.getLogger(__name__)

# Simple in-memory cache to replace the cache_manager import
_timing_cache = {}

def get_cache(key):
    """Get value from cache"""
    return _timing_cache.get(key)

def set_cache(key, value, expiry=None):
    """Set value in cache"""
    _timing_cache[key] = value
    # Expiry not implemented in this simple version

@lru_cache(maxsize=128)
def get_optimal_entry_time(current_epoch, round_data_str, market_regime_str):
    """
    Determine the optimal time to enter a trade based on market conditions.
    
    Args:
        current_epoch: Current blockchain epoch
        round_data_str: Data about the current round (as string for caching)
        market_regime_str: Current market regime information (as string for caching)
        
    Returns:
        dict: Timing information with optimal entry window details
    """
    # Convert string inputs back to dicts
    round_data = eval(round_data_str)
    market_regime = eval(market_regime_str)
    
    try:
        # Create cache key using current_epoch for epoch-specific timing
        cache_key = f"timing_epoch_{current_epoch}"
        
        # Use cache_key to check if we already calculated timing for this epoch
        cached_timing = get_cache(cache_key)
        if cached_timing:
            logger.debug(f"Using cached timing for epoch {current_epoch}")
            return cached_timing
        
        # Check if we have stored timing preferences for this epoch
        # This would come from a storage mechanism in a real implementation
        stored_preferences = {}  # Would be loaded from storage in real implementation
        
        # Get round timestamps
        start_time = round_data.get('startTimestamp', 0)
        lock_time = round_data.get('lockTimestamp', 0)
        close_time = round_data.get('closeTimestamp', 0)
        
        # Calculate round duration
        round_duration = close_time - start_time
        lock_duration = lock_time - start_time
        
        # Log the epoch we're calculating timing for
        logger.info(f"Calculating optimal entry time for epoch {current_epoch}")
        
        if lock_duration <= 0:
            logger.warning("Invalid round timing data (lock_duration <= 0)")
            return {
                "epoch": current_epoch,
                "is_optimal_time": False,
                "optimal_seconds_before_lock": 30,  # Safe default
                "current_seconds_before_lock": 0,
                "optimal_pct": 0.7,
                "current_pct": 0
            }
        
        # Current time
        current_time = int(time.time())
        
        # Time left until lock
        time_to_lock = max(0, lock_time - current_time)
        
        # Base timing strategy on market regime
        regime = market_regime.get('regime', 'unknown')
        
        # After loading stored_preferences, use them if available
        if stored_preferences:
            # Check if we have stored timing preferences for this specific epoch
            if str(current_epoch) in stored_preferences:
                epoch_prefs = stored_preferences[str(current_epoch)]
                logger.info(f"Found stored timing preferences for epoch {current_epoch}")
                
                # Use stored optimal_pct if available
                if 'optimal_pct' in epoch_prefs:
                    optimal_pct = epoch_prefs['optimal_pct']
                    logger.info(f"Using stored optimal percentage: {optimal_pct}")
                    
                # Could also use other stored preferences here
                
            # Or use preferences for the regime if available
            elif regime in stored_preferences:
                regime_prefs = stored_preferences[regime]
                if 'optimal_pct' in regime_prefs:
                    optimal_pct = regime_prefs['optimal_pct']
                    logger.info(f"Using {regime} regime stored timing: {optimal_pct}")
        
        if regime == "volatile":
            # In volatile markets, wait longer to catch last-minute moves
            # Wait until 75-85% of the way to lock
            optimal_pct = 0.8  # 80% of the way to lock
            logger.debug(f"Using volatile market timing strategy (optimal_pct={optimal_pct})")
            
        elif regime == "trending" or regime == "uptrend" or regime == "downtrend":
            # In trending markets, earlier entry is usually fine
            # Enter around 60-70% of the way to lock
            optimal_pct = 0.65
            logger.debug(f"Using trending market timing strategy (optimal_pct={optimal_pct})")
            
        else:  # ranging or unknown
            # Default timing - around 70-75% of the way to lock
            optimal_pct = 0.7
            logger.debug(f"Using default timing strategy (optimal_pct={optimal_pct})")
        
        # Calculate optimal seconds before lock
        optimal_seconds = lock_duration * (1 - optimal_pct)
        
        # Determine if current time is optimal for entry
        current_pct = (current_time - start_time) / lock_duration
        
        # INCREASED the optimal window from 0.1 to 0.2
        is_optimal_time = abs(current_pct - optimal_pct) <= 0.2  # Within 20% of optimal
        
        if is_optimal_time:
            logger.info(f"🕒 OPTIMAL ENTRY TIME: {time_to_lock:.1f}s before lock ({current_pct:.2f} vs optimal {optimal_pct:.2f})")
        else:
            logger.debug(f"Not optimal entry time: {time_to_lock:.1f}s before lock ({current_pct:.2f} vs optimal {optimal_pct:.2f})")
        
        result = {
            "epoch": current_epoch,  # Include epoch in result
            "is_optimal_time": is_optimal_time,
            "optimal_seconds_before_lock": optimal_seconds,
            "current_seconds_before_lock": time_to_lock,
            "optimal_pct": optimal_pct,
            "current_pct": current_pct
        }
        
        # After calculation, cache the result before returning
        set_cache(cache_key, result, expiry=1800)  # Cache for 30 minutes
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Error calculating optimal entry time for epoch {current_epoch}: {e}")
        traceback.print_exc()
        return {
            "epoch": current_epoch,
            "is_optimal_time": True,  # Default to True on error to avoid missing trades
            "optimal_seconds_before_lock": 30,  # Safe default
            "current_seconds_before_lock": 30,
            "optimal_pct": 0.7,
            "current_pct": 0.7,
            "error": str(e)
        } 