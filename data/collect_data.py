import time
import sqlite3
import json
import requests
from web3 import Web3
from datetime import datetime
import os
import logging
from pathlib import Path

# Import from config module
from configuration import config, get_contract_abi

# Setup logger
logger = logging.getLogger(__name__)

# Load settings from configuration
RPC_URL = config.get('blockchain.rpc.primary')
PREDICTION_CONTRACT_ADDRESS = config.get('blockchain.contract_address')
DB_FILE = config.get('database.file')
TABLE_NAME = config.get('database.tables.trades')
MARKET_API_URL = config.get('api.market')

# Initialize Web3
web3 = Web3(Web3.HTTPProvider(RPC_URL))

# Load ABI
contract_abi = get_contract_abi()
if contract_abi:
    contract = web3.eth.contract(address=PREDICTION_CONTRACT_ADDRESS, abi=contract_abi)
else:
    logger.error("Failed to load contract ABI")
    raise ValueError("Contract ABI could not be loaded")

def create_table():
    """Create the database table if it doesn't exist."""
    # Make sure directory exists
    os.makedirs(os.path.dirname(DB_FILE), exist_ok=True)
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            epoch INTEGER PRIMARY KEY,
            datetime TEXT,
            hour INTEGER,
            minute INTEGER,
            second INTEGER,
            lockPrice REAL,
            closePrice REAL,
            bullAmount REAL,
            bearAmount REAL,
            totalAmount REAL,
            bullRatio REAL,
            bearRatio REAL,
            bnbPrice REAL,
            btcPrice REAL,
            outcome TEXT
        )
    """)
    conn.commit()
    conn.close()
    logger.info(f"Database table {TABLE_NAME} created/confirmed")

def get_last_saved_epoch():
    """Get the most recent epoch saved in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT MAX(epoch) FROM {TABLE_NAME}")
    last_epoch = cursor.fetchone()[0]
    conn.close()
    return last_epoch if last_epoch is not None else 0

def fetch_market_data():
    """Fetch current market prices from API."""
    try:
        response = requests.get(MARKET_API_URL)
        data = response.json()
        bnb_price = next((item["price"] for item in data if item["symbol"] == "BNBUSDT"), None)
        btc_price = next((item["price"] for item in data if item["symbol"] == "BTCUSDT"), None)
        return float(bnb_price), float(btc_price)
    except Exception as e:
        logger.error(f"Error fetching market data: {e}")
        return None, None

def fetch_round_data(epoch):
    """Fetch data for a specific round/epoch."""
    try:
        round_data = contract.functions.rounds(epoch).call()
        total_amount = round_data[8] / 1e18
        bull_amount = round_data[9] / 1e18
        bear_amount = round_data[10] / 1e18
        bull_ratio = bull_amount / total_amount if total_amount > 0 else 0
        bear_ratio = bear_amount / total_amount if total_amount > 0 else 0
        bnb_price, btc_price = fetch_market_data()
        close_timestamp = round_data[3]
        dt = datetime.fromtimestamp(close_timestamp, datetime.timezone.utc)
        
        lock_price = round_data[4] / 1e18
        close_price = round_data[5] / 1e18
        outcome = "Bull" if close_price > lock_price else "Bear" if close_price < lock_price else "Draw"

        return {
            "epoch": epoch,
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "hour": dt.hour,
            "minute": dt.minute,
            "second": dt.second,
            "lockPrice": lock_price,
            "closePrice": close_price,
            "bullAmount": bull_amount,
            "bearAmount": bear_amount,
            "totalAmount": total_amount,
            "bullRatio": bull_ratio,
            "bearRatio": bear_ratio,
            "bnbPrice": bnb_price,
            "btcPrice": btc_price,
            "outcome": outcome
        }
    except Exception as e:
        logger.error(f"Error fetching round {epoch} data: {e}")
        return None

def save_to_db(data):
    """Save round data to the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE epoch = ?", (data['epoch'],))
    count = cursor.fetchone()[0]
    if count == 0:
        cursor.execute(f"""
            INSERT INTO {TABLE_NAME} (epoch, datetime, hour, minute, second, lockPrice, closePrice, bullAmount, bearAmount, totalAmount, bullRatio, bearRatio, bnbPrice, btcPrice, outcome)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (data['epoch'], data['datetime'], data['hour'], data['minute'], data['second'], data['lockPrice'], data['closePrice'], data['bullAmount'], data['bearAmount'], data['totalAmount'], data['bullRatio'], data['bearRatio'], data['bnbPrice'], data['btcPrice'], data['outcome']))
        logger.info(f"Inserted round {data['epoch']}")
    conn.commit()
    conn.close()

def fetch_historical_data():
    """Fetch and save historical rounds data."""
    last_epoch = get_last_saved_epoch()
    current_epoch = contract.functions.currentEpoch().call()
    logger.info(f"Current epoch: {current_epoch}, Last saved epoch: {last_epoch}")
    
    if last_epoch == 0:
        start_epoch = max(1, current_epoch - 10000)
    else:
        start_epoch = last_epoch + 1
    
    if start_epoch >= current_epoch:
        logger.info("Database is up to date.")
        return
    
    logger.info(f"Fetching from epoch {start_epoch} to {current_epoch - 1}")
    for epoch in range(start_epoch, current_epoch):
        round_data = fetch_round_data(epoch)
        if round_data:
            save_to_db(round_data)
        else:
            logger.warning(f"Skipping round {epoch} (no data).")
    logger.info("Historical data catch-up complete.")

def collect_real_time_data():
    """Continuously collect new round data in real-time."""
    logger.info("Starting real-time data collection...")
    last_epoch = get_last_saved_epoch()
    while True:
        current_epoch = contract.functions.currentEpoch().call()
        if current_epoch > last_epoch:
            round_data = fetch_round_data(current_epoch)
            if round_data:
                save_to_db(round_data)
                last_epoch = current_epoch
                logger.info(f"Saved real-time round {current_epoch}.")
        else:
            logger.debug(f"Waiting for new round... (Current: {current_epoch})")
        time.sleep(300)  # 5 minutes between checks

if __name__ == "__main__":
    # Setup basic logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    create_table()
    fetch_historical_data()
    collect_real_time_data()
