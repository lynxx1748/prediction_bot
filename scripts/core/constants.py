"""
Constants and configuration for the trading bot.
"""

import json
import os
from web3 import Web3
from eth_typing import ChecksumAddress
import logging
import traceback

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

logger = logging.getLogger(__name__)

# Look for config.json 
config_paths = [
    "configuration/config.json"
]

config = None
for config_path in config_paths:
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
            print(f"✅ Loaded config from {config_path}")
            break
    except FileNotFoundError:
        continue
    except json.JSONDecodeError:

        print(f"❌ Invalid JSON in {config_path}")
        raise

if config is None:
    print("❌ Could not find config.json in any of these locations:")
    for path in config_paths:
        print(f"  - {path}")
    raise FileNotFoundError("config.json not found")

# Constants from config
DB_FILE = os.path.join("data", "historical_data.db")
TABLES = {
    "trades": "trades",
    "predictions": "predictions",
    "signal_performance": "signal_performance",
    "strategy_performance": "strategy_performance",
    "market_data": "market_data",
    "bets": "bets"
}

# Model files
MODEL_FILE = config.get("model", {}).get("model_file", "./data/random_forest_model.pkl")
SCALER_FILE = config.get("model", {}).get("scaler_file", "./data/random_forest_scaler.pkl")

# Network setup
CONTRACT_ADDRESS = config.get("blockchain", {}).get("contract_address")
RPC_URL = config.get("blockchain", {}).get("rpc", {}).get("primary")
MARKET_API = config.get("market_api")

# More robust handling for ABI file path
try:
    abi_filename = config.get('abi_file', 'abi.json')
    ABI_FILE = os.path.join("configuration", abi_filename)
    if not os.path.exists(ABI_FILE):
        logger.warning(f"ABI file not found at {ABI_FILE}, using default path")
        ABI_FILE = os.path.join("configuration", "abi.json")
except Exception as e:
    logger.error(f"Error configuring ABI file path: {e}")
    ABI_FILE = os.path.join("configuration", "abi.json")

logger.info(f"Using ABI file: {ABI_FILE}")

# Initialize Web3 first
web3 = Web3(Web3.HTTPProvider(RPC_URL))

# Load contract ABI
abi_loaded = False
abi_paths_to_try = [
    "abi.json",  # Add the root location first
    os.path.join("configuration", "abi.json"),
    ABI_FILE,
    os.path.join(".", "abi.json"),
    os.path.join("..", "abi.json"),
    os.path.join("../configuration", "abi.json")
]

print("🔍 Looking for ABI file in these locations:")
for path in abi_paths_to_try:
    print(f"  - {os.path.abspath(path)}")

for abi_path in abi_paths_to_try:
    try:
        with open(abi_path, "r") as abi_file:
            contract_abi = json.load(abi_file)
        print(f"✅ ABI loaded from {abi_path}")
        abi_loaded = True
        break
    except FileNotFoundError:
        continue
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in {abi_path}!")
        continue

if not abi_loaded:
    print("❌ Could not load ABI from any of these locations:")
    for path in abi_paths_to_try:
        print(f"  - {path}")
    raise FileNotFoundError("ABI file not found")

# Contract initialization section - fix the import and loading issues
try:
    # Import directly from configuration module to ensure proper loading
    from configuration import get_contract_abi
    
    # Get ABI directly from the function
    contract_abi = get_contract_abi()
    if not contract_abi:
        raise ValueError("Failed to load contract ABI")
    
    # Get contract address from config
    if not CONTRACT_ADDRESS:
        raise ValueError("CONTRACT_ADDRESS is empty in config")
        
    # Verify web3 connection first
    try:
        block_number = web3.eth.block_number
        print(f"✅ Web3 connected to {RPC_URL}. Current block: {block_number}")
        
        # Ensure address is properly checksummed
        checksum_address = Web3.to_checksum_address(CONTRACT_ADDRESS)
        print(f"✅ Address checksummed: {checksum_address}")
        
        # Initialize contract with direct reference to ABI
        contract = web3.eth.contract(address=checksum_address, abi=contract_abi)
        print(f"✅ Contract initialized at {checksum_address}")
        
        # Test contract connection
        try:
            current_epoch = contract.functions.currentEpoch().call()
            print(f"✅ Contract successfully verified! Current epoch: {current_epoch}")
        except Exception as verify_error:
            print(f"❌ Contract verification failed: {verify_error}")
            print("RPC connection works but contract calls fail - likely ABI mismatch or wrong contract address")
            contract = None
    except Exception as web3_error:
        print(f"❌ Web3 connection failed: {web3_error}")
        print(f"Cannot connect to RPC: {RPC_URL}")
        contract = None
except Exception as e:
    print(f"❌ Error initializing contract: {e}")
    print(f"CONTRACT_ADDRESS: {CONTRACT_ADDRESS}")
    print(f"RPC_URL: {RPC_URL}")
    traceback.print_exc()
    contract = None

# Final verification
if contract is None:
    print("❌ CONTRACT IS NOT INITIALIZED. Blockchain functions will not work.")
    print("Please check:")
    print("1. CONTRACT_ADDRESS in your config.json")
    print("2. RPC_URL in your config.json") 
    print("3. That your ABI matches the contract at the specified address")
else:
    print("✅ CONTRACT SUCCESSFULLY INITIALIZED")

# Trading constants
THRESHOLDS = {
    'strong_imbalance': 0.15,     # 15% imbalance is considered strong
    'min_total_amount': 0.1,      # REDUCED from 1.0 to 0.1 BNB
    'high_volatility': 5.0,       # 5% change is high volatility
    'extreme_volatility': 10.0,   # 10% change is extreme
    'strong_trend': 3.0,          # 3% price change indicates strong trend
    'strong_ratio': 0.65,         # 65% ratio is considered strong
    'extreme_ratio': 0.80,        # 80% ratio is considered extreme
    'min_confidence': 0.52,       # REDUCED from 0.55 to 0.52
    'high_confidence': 0.80,      # High confidence threshold
    'reverse_after': 4            # Reverse strategy after X consecutive losses
}

STRATEGY_WEIGHTS = config.get("trading", {}).get("prediction_weights", {
    "model": 0.15,
    "pattern": 0.20,
    "market": 0.20,
    "technical": 0.20,
    "sentiment": 0.25
})

STOP_LOSS = config.get("trading", {}).get("stop_loss", 5)
WAGER_AMOUNT = config.get("trading", {}).get("wager_amount", 0.005)
BETTING_MODE = config.get("trading", {}).get("betting_mode", "test")

# Wallet constants
try:
    # Use get() with defaults for wallet configuration
    wallet_config = config.get("wallet", {})
    ACCOUNT_ADDRESS = web3.to_checksum_address(wallet_config.get("address", "0x0000000000000000000000000000000000000000"))
    PRIVATE_KEY = wallet_config.get("private_key", "")
    
    # Only verify wallet setup if we're not in test mode
    if BETTING_MODE.lower() != "test" and (not ACCOUNT_ADDRESS or not PRIVATE_KEY):
        print("⚠️ Warning: Missing wallet credentials, only test mode will be available")
    
    # Test wallet connection if we have an address (that's not the default)
    if ACCOUNT_ADDRESS != "0x0000000000000000000000000000000000000000":
        balance = web3.eth.get_balance(ACCOUNT_ADDRESS)
        print(f"✅ Wallet connected - Balance: {web3.from_wei(balance, 'ether'):.4f} BNB")
    else:
        print("ℹ️ No wallet address configured, running in test-only mode")
    
except Exception as e:
    print(f"⚠️ Wallet setup issue: {e}")
    print("ℹ️ Running in test-only mode")
    ACCOUNT_ADDRESS = "0x0000000000000000000000000000000000000000"
    PRIVATE_KEY = ""

def get_backup_rpc():
    """Get list of backup RPC endpoints"""
    return config["rpc"].get("backup_endpoints", [])

def get_gas_multiplier(strategy):
    """Get gas multiplier for given strategy"""
    return config["gas_price"]["multipliers"].get(strategy, 1.0)

# Market bias configuration
MARKET_BIAS = {
    'enabled': config.get('market_bias', {}).get('enabled', True),
    'bias_direction': config.get('market_bias', {}).get('bias_direction', 'BULL'),
    'bias_strength': config.get('market_bias', {}).get('bias_strength', 0.15),
    'min_confidence': config.get('market_bias', {}).get('min_confidence', 0.45)
}

def ensure_checksum_address(address: str) -> ChecksumAddress:
    """
    Convert a string ethereum address to a ChecksumAddress.
    
    Args:
        address: Ethereum address string
        
    Returns:
        ChecksumAddress: Address in checksum format
        
    Raises:
        ValueError: If the address is invalid
    """
    # Type check
    if not isinstance(address, str):
        logging.error(f"Address must be a string, got {type(address)}")
        raise ValueError(f"Address must be a string, got {type(address)}")
    
    # Format check
    if not Web3.is_address(address):
        logging.error(f"Invalid Ethereum address format: {address}")
        raise ValueError(f"Invalid Ethereum address format: {address}")
    
    # Process and return
    return ChecksumAddress(Web3.to_checksum_address(address))