"""
Data collection functionality for trading bot.
For fetching historical and real-time data.
"""

import logging
import time
import sqlite3
import requests
import json
import os
from datetime import datetime, timedelta

from scripts.core.constants import DB_FILE, TABLES, CONTRACT_ADDRESS, web3, contract, MARKET_API

logger = logging.getLogger(__name__)

def fetch_historical_data(days=30, save_to_db=True):
    """
    Fetch historical price and betting data.
    
    Args:
        days: Number of days of history to fetch
        save_to_db: Whether to save fetched data to database
        
    Returns:
        dict: Historical data
    """
    logger.info(f"Fetching {days} days of historical data")
    
    try:
        # Calculate start time
        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        # Initialize data structure
        historical_data = {
            'prices': [],
            'rounds': [],
            'start_timestamp': start_time,
            'end_timestamp': int(datetime.now().timestamp())
        }
        
        # Fetch price data from API
        if MARKET_API:
            url = f"{MARKET_API}/historical-prices?days={days}"
            response = requests.get(url)
            if response.status_code == 200:
                price_data = response.json()
                historical_data['prices'] = price_data
                logger.info(f"Fetched {len(price_data)} price points")
            else:
                logger.error(f"Failed to fetch historical prices: {response.status_code}")
        
        # Fetch round data from contract if web3 is connected
        if web3 and web3.is_connected() and contract:
            current_epoch = contract.functions.currentEpoch().call()
            # Get up to 500 recent rounds
            max_rounds = min(500, current_epoch)
            
            for epoch in range(current_epoch - max_rounds, current_epoch):
                try:
                    # Fetch round from contract
                    round_data = contract.functions.rounds(epoch).call()
                    
                    # Convert to dictionary
                    formatted_round = {
                        'epoch': epoch,
                        'startTimestamp': round_data[0],
                        'lockTimestamp': round_data[1],
                        'closeTimestamp': round_data[2],
                        'lockPrice': round_data[3] / 1e8,
                        'closePrice': round_data[4] / 1e8,
                        'lockOracleId': round_data[5],
                        'closeOracleId': round_data[6],
                        'totalAmount': round_data[7] / 1e18,
                        'bullAmount': round_data[8] / 1e18,
                        'bearAmount': round_data[9] / 1e18,
                        'rewardBaseCalAmount': round_data[10] / 1e18,
                        'rewardAmount': round_data[11] / 1e18,
                        'oracleCalled': round_data[12]
                    }
                    
                    # Add calculated fields
                    formatted_round['bullRatio'] = 0
                    formatted_round['bearRatio'] = 0
                    
                    if formatted_round['totalAmount'] > 0:
                        formatted_round['bullRatio'] = formatted_round['bullAmount'] / formatted_round['totalAmount']
                        formatted_round['bearRatio'] = formatted_round['bearAmount'] / formatted_round['totalAmount']
                    
                    # Determine outcome
                    if formatted_round['closePrice'] > formatted_round['lockPrice']:
                        formatted_round['outcome'] = 'BULL'
                    elif formatted_round['closePrice'] < formatted_round['lockPrice']:
                        formatted_round['outcome'] = 'BEAR'
                    else:
                        formatted_round['outcome'] = 'DRAW'
                        
                    # Add to historical data
                    historical_data['rounds'].append(formatted_round)
                    
                except Exception as e:
                    logger.error(f"Error fetching round {epoch}: {e}")
            
            logger.info(f"Fetched {len(historical_data['rounds'])} historical rounds")
        
        # Save to database if requested
        if save_to_db:
            save_historical_data_to_db(historical_data)
        
        # Add metadata about the fetch
        historical_data['fetched_at'] = datetime.now().isoformat()
        historical_data['days_fetched'] = days
        
        logger.info(f"Successfully fetched historical data since {datetime.fromtimestamp(start_time)}")
        return historical_data
        
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {'prices': [], 'rounds': []}

def save_historical_data_to_db(historical_data):
    """
    Save historical data to database.
    
    Args:
        historical_data: Dictionary of historical prices and rounds
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Save price data
        for price_point in historical_data.get('prices', []):
            try:
                cursor.execute(f'''
                    INSERT OR IGNORE INTO {TABLES['market_data']} 
                    (timestamp, price, volume, market_cap, dominance)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    price_point.get('timestamp', 0),
                    price_point.get('price', 0),
                    price_point.get('volume', 0),
                    price_point.get('market_cap', 0),
                    price_point.get('dominance', 0)
                ))
            except Exception as e:
                logger.error(f"Error saving price point: {e}")
        
        # Save round data
        for round_data in historical_data.get('rounds', []):
            try:
                # First check if this round already exists
                cursor.execute(f'''
                    SELECT id FROM {TABLES['trades']} WHERE epoch = ?
                ''', (round_data.get('epoch', 0),))
                
                if cursor.fetchone() is None:
                    cursor.execute(f'''
                        INSERT INTO {TABLES['trades']} 
                        (epoch, timestamp, lockPrice, closePrice, bullAmount, bearAmount, 
                        totalAmount, bullRatio, bearRatio, outcome)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        round_data.get('epoch', 0),
                        round_data.get('closeTimestamp', 0),
                        round_data.get('lockPrice', 0),
                        round_data.get('closePrice', 0),
                        round_data.get('bullAmount', 0),
                        round_data.get('bearAmount', 0),
                        round_data.get('totalAmount', 0),
                        round_data.get('bullRatio', 0),
                        round_data.get('bearRatio', 0),
                        round_data.get('outcome', 'UNKNOWN')
                    ))
            except Exception as e:
                logger.error(f"Error saving round data: {e}")
        
        conn.commit()
        conn.close()
        logger.info("Historical data saved to database")
        
    except Exception as e:
        logger.error(f"Error saving historical data to database: {e}")

def collect_real_time_data():
    """
    Collect real-time market data.
    
    Returns:
        dict: Current market data
    """
    logger.info("Collecting real-time market data")
    
    try:
        # Initialize data structure
        real_time_data = {
            'timestamp': int(time.time()),
            'price': 0,
            'volume': 0,
            'market_cap': 0,
            'latest_rounds': []
        }
        
        # Get current price data
        if MARKET_API:
            try:
                url = f"{MARKET_API}/current-price"
                response = requests.get(url)
                if response.status_code == 200:
                    price_data = response.json()
                    real_time_data.update(price_data)
                    logger.info(f"Current price: ${price_data.get('price', 0)}")
                else:
                    logger.error(f"Failed to fetch current price: {response.status_code}")
            except Exception as e:
                logger.error(f"Error fetching current price: {e}")
        
        # Get latest round data from contract
        if web3 and web3.is_connected() and contract:
            try:
                current_epoch = contract.functions.currentEpoch().call()
                
                # Get data for current and previous rounds
                for epoch in range(current_epoch - 5, current_epoch + 1):
                    try:
                        round_data = contract.functions.rounds(epoch).call()
                        
                        formatted_round = {
                            'epoch': epoch,
                            'startTimestamp': round_data[0],
                            'lockTimestamp': round_data[1],
                            'closeTimestamp': round_data[2],
                            'lockPrice': round_data[3] / 1e8,
                            'closePrice': round_data[4] / 1e8,
                            'totalAmount': round_data[7] / 1e18,
                            'bullAmount': round_data[8] / 1e18,
                            'bearAmount': round_data[9] / 1e18,
                            'oracleCalled': round_data[12]
                        }
                        
                        # Add calculated fields
                        formatted_round['bullRatio'] = 0
                        formatted_round['bearRatio'] = 0
                        
                        if formatted_round['totalAmount'] > 0:
                            formatted_round['bullRatio'] = formatted_round['bullAmount'] / formatted_round['totalAmount']
                            formatted_round['bearRatio'] = formatted_round['bearAmount'] / formatted_round['totalAmount']
                        
                        # Determine status
                        now = int(time.time())
                        if now < formatted_round['lockTimestamp']:
                            formatted_round['status'] = 'betting'
                        elif now < formatted_round['closeTimestamp']:
                            formatted_round['status'] = 'locked'
                        else:
                            formatted_round['status'] = 'closed'
                            
                            # Determine outcome for closed rounds
                            if formatted_round['closePrice'] > formatted_round['lockPrice']:
                                formatted_round['outcome'] = 'BULL'
                            elif formatted_round['closePrice'] < formatted_round['lockPrice']:
                                formatted_round['outcome'] = 'BEAR'
                            else:
                                formatted_round['outcome'] = 'DRAW'
                        
                        real_time_data['latest_rounds'].append(formatted_round)
                        
                    except Exception as e:
                        logger.error(f"Error fetching round {epoch}: {e}")
                
                logger.info(f"Collected data for {len(real_time_data['latest_rounds'])} latest rounds")
                
            except Exception as e:
                logger.error(f"Error fetching current epoch: {e}")
        
        return real_time_data
        
    except Exception as e:
        logger.error(f"Error collecting real-time data: {e}")
        return {}

def ensure_data_directories():
    """
    Ensure that all necessary data directories exist.
    Uses os module to check and create directories.
    
    Returns:
        list: Created or verified directory paths
    """
    data_dirs = [
        'data',
        'data/raw',
        'data/processed', 
        'data/models',
        'data/logs'
    ]
    
    created_dirs = []
    
    for directory in data_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            created_dirs.append(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
            
    # Get current script directory for reference
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    return {
        'created': created_dirs,
        'all_dirs': data_dirs,
        'script_dir': script_dir
    }

def get_contract_info():
    """
    Get information about the prediction contract.
    Uses CONTRACT_ADDRESS constant to identify the contract.
    
    Returns:
        dict: Contract information
    """
    try:
        # Check if contract address is valid
        if not web3.is_address(CONTRACT_ADDRESS):
            return {
                'status': 'error',
                'message': f'Invalid contract address: {CONTRACT_ADDRESS}'
            }
            
        # Get contract code and balance
        contract_code = web3.eth.get_code(CONTRACT_ADDRESS)
        contract_balance = web3.eth.get_balance(CONTRACT_ADDRESS)
        balance_bnb = web3.from_wei(contract_balance, 'ether')
        
        logger.info(f"Contract info retrieved for {CONTRACT_ADDRESS}")
        
        return {
            'status': 'success',
            'address': CONTRACT_ADDRESS,
            'has_code': len(contract_code) > 0,
            'balance_wei': contract_balance,
            'balance_bnb': float(balance_bnb),
            'network': web3.net.version
        }
        
    except Exception as e:
        logger.error(f"Error getting contract info: {e}")
        return {
            'status': 'error',
            'message': str(e),
            'address': CONTRACT_ADDRESS
        }

if __name__ == "__main__":
    # If run directly, fetch some historical data for testing
    logging.basicConfig(level=logging.INFO)
    data = fetch_historical_data(days=7)
    print(f"Fetched {len(data['prices'])} price points and {len(data['rounds'])} rounds") 