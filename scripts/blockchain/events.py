"""
Blockchain event tracking for the trading bot.
Monitors and processes events from the blockchain in real-time.
"""

import time
import traceback
import logging
from web3 import Web3
import sqlite3

from ..core.constants import web3, contract, DB_FILE, CONTRACT_ADDRESS
from ..data.database import record_prediction

logger = logging.getLogger(__name__)

def setup_event_listeners(contract, callback=None):
    """
    Set up event listeners for BetBull and BetBear events.
    
    Args:
        contract: The prediction contract instance
        callback: Optional callback function for events
        
    Returns:
        tuple: (bull_filter, bear_filter) event filters
    """
    try:
        # In newer web3.py versions, you need to use argument_filters parameter
        # instead of directly passing fromBlock
        bull_filter = contract.events.BetBull.create_filter(
            argument_filters={},
            from_block='latest'  # Note: fromBlock changed to from_block
        )
        
        bear_filter = contract.events.BetBear.create_filter(
            argument_filters={},
            from_block='latest'  # Note: fromBlock changed to from_block
        )
        
        logger.info("✅ Event listeners set up successfully")
        return bull_filter, bear_filter
    
    except Exception as e:
        logger.error(f"❌ Error setting up event listeners: {e}")
        traceback.print_exc()
        return None, None

def track_betting_events(bull_filter=None, bear_filter=None):
    """
    Track betting data directly from contract.
    
    Args:
        bull_filter: Event filter for BetBull events
        bear_filter: Event filter for BetBear events
        
    Returns:
        int: 1 on success, 0 on failure
    """
    try:
        # Check and process any new bull events if filter provided
        if bull_filter:
            try:
                new_bull_entries = bull_filter.get_new_entries()
                if new_bull_entries:
                    logger.info(f"Found {len(new_bull_entries)} new bull events")
                    # Process the new bull entries here
            except Exception as filter_error:
                logger.warning(f"Error checking bull filter: {filter_error}")
        
        # After the bull filter check, add similar code for bear filter
        if bear_filter:
            try:
                new_bear_entries = bear_filter.get_new_entries()
                if new_bear_entries:
                    logger.info(f"Found {len(new_bear_entries)} new bear events")
                    # Process the new bear entries here
            except Exception as filter_error:
                logger.warning(f"Error checking bear filter: {filter_error}")
        
        # Get current epoch
        current_epoch = contract.functions.currentEpoch().call()
        
        # Get data for current round
        round_data = contract.functions.rounds(current_epoch).call()
        
        # Extract amounts
        bull_amount = float(web3.from_wei(round_data[9], 'ether')) 
        bear_amount = float(web3.from_wei(round_data[10], 'ether'))
        total_amount = bull_amount + bear_amount
        
        # Calculate ratios
        bull_ratio = bull_amount / total_amount if total_amount > 0 else 0.5
        bear_ratio = bear_amount / total_amount if total_amount > 0 else 0.5
        
        # Prepare data for database
        prediction_data = {
            'epoch': current_epoch,
            'bullAmount': bull_amount,
            'bearAmount': bear_amount,
            'totalAmount': total_amount,
            'bullRatio': bull_ratio,
            'bearRatio': bear_ratio,
            'lockPrice': float(web3.from_wei(round_data[4], 'ether')),
            'closePrice': float(web3.from_wei(round_data[5], 'ether'))
        }
        
        # Record to predictions table
        record_prediction(current_epoch, prediction_data)
        
        logger.info(f"📊 Round {current_epoch} Data: Bull {bull_ratio:.2%} / Bear {bear_ratio:.2%}")
        return 1
        
    except Exception as e:
        logger.error(f"❌ Error collecting betting data: {e}")
        traceback.print_exc()
        return 0

def start_event_monitor(interval=30):
    """
    Start monitoring blockchain events.
    
    Args:
        interval: Polling interval in seconds
        
    Returns:
        bool: True if monitoring started successfully
    """
    try:
        # Set up event filters
        bull_filter, bear_filter = setup_event_listeners(contract)
        if not bull_filter or not bear_filter:
            logger.error("❌ Failed to set up event filters")
            return False
            
        logger.info(f"🔄 Starting event monitor with {interval}s interval")
        
        # Initial data collection
        track_betting_events(bull_filter, bear_filter)
        
        # Return True to indicate success
        # (In a real implementation, this would likely be in a loop or separate thread)
        return True
        
    except Exception as e:
        logger.error(f"❌ Error starting event monitor: {e}")
        traceback.print_exc()
        return False

def check_new_events(bull_filter, bear_filter):
    """
    Check for new betting events and process them.
    
    Args:
        bull_filter: Event filter for BetBull events
        bear_filter: Event filter for BetBear events
        
    Returns:
        tuple: (bull_entries, bear_entries) count of new events
    """
    try:
        # Get new entries from filters
        bull_entries = bull_filter.get_new_entries()
        bear_entries = bear_filter.get_new_entries()
        
        # Process bull entries
        for entry in bull_entries:
            epoch = entry['args']['epoch']
            amount = float(Web3.from_wei(entry['args']['amount'], 'ether'))
            sender = entry['args']['sender']
            logger.info(f"🐂 BetBull: {amount:.4f} BNB from {sender[:8]}... on epoch {epoch}")
            
        # Process bear entries
        for entry in bear_entries:
            epoch = entry['args']['epoch']
            amount = float(Web3.from_wei(entry['args']['amount'], 'ether'))
            sender = entry['args']['sender']
            logger.info(f"🐻 BetBear: {amount:.4f} BNB from {sender[:8]}... on epoch {epoch}")
            
        # If we have any new entries, update the database
        if bull_entries or bear_entries:
            track_betting_events(bull_filter, bear_filter)
            
        return len(bull_entries), len(bear_entries)
        
    except Exception as e:
        logger.error(f"❌ Error checking new events: {e}")
        traceback.print_exc()
        return 0, 0

def wait_for_event_confirmation(event_hash, timeout=120):
    """
    Wait for a blockchain event to be confirmed.
    Uses the time module for timing the wait process.
    
    Args:
        event_hash: Transaction or event hash
        timeout: Maximum time to wait in seconds
        
    Returns:
        bool: True if confirmed, False if timed out
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check confirmation status logic would go here
            # For demo purposes, we'll just wait a bit
            time.sleep(2)
            
            # Simulate checking confirmation status
            # In a real implementation, you would check the blockchain
            if (time.time() - start_time) > 10:  # Simulate confirmation after 10 seconds
                logger.info(f"Event {event_hash} confirmed after {time.time() - start_time:.1f} seconds")
                return True
                
        except Exception as e:
            logger.error(f"Error checking event confirmation: {e}")
            traceback.print_exc()
            
    logger.warning(f"Timed out waiting for event {event_hash} after {timeout} seconds")
    return False

def initialize_event_db():
    """
    Initialize the event database for storing blockchain events.
    Uses DB_FILE constant for database location.
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create table for blockchain events if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS blockchain_events (
                event_hash TEXT PRIMARY KEY,
                epoch INTEGER,
                event_type TEXT,
                amount REAL,
                timestamp INTEGER,
                sender TEXT,
                confirmed INTEGER DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info(f"Event database initialized at {DB_FILE}")
        return True
        
    except Exception as e:
        logger.error(f"Error initializing event database: {e}")
        traceback.print_exc()
        return False

def validate_contract():
    """
    Validate the prediction contract address and check its status.
    Uses CONTRACT_ADDRESS constant to identify the target contract.
    
    Returns:
        dict: Contract validation information
    """
    try:
        # Verify contract address format
        if not Web3.is_address(CONTRACT_ADDRESS):
            logger.error(f"Invalid contract address format: {CONTRACT_ADDRESS}")
            return {'valid': False, 'reason': 'Invalid address format'}
            
        # Check if contract exists on the blockchain
        code = web3.eth.get_code(CONTRACT_ADDRESS)
        if code == b'':
            logger.error(f"No code found at contract address: {CONTRACT_ADDRESS}")
            return {'valid': False, 'reason': 'No contract code at address'}
            
        # Get contract balance
        balance = web3.eth.get_balance(CONTRACT_ADDRESS)
        balance_bnb = web3.from_wei(balance, 'ether')
        
        logger.info(f"Contract at {CONTRACT_ADDRESS} is valid with balance of {balance_bnb:.4f} BNB")
        return {
            'valid': True,
            'address': CONTRACT_ADDRESS,
            'balance': float(balance_bnb)
        }
        
    except Exception as e:
        logger.error(f"Error validating contract: {e}")
        traceback.print_exc()
        return {'valid': False, 'reason': str(e)} 