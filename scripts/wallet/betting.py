"""
Betting functionality for the trading bot.
Handles bet amount calculations and bet placement.
"""

import logging
import traceback
from web3 import Web3

from ..core.constants import config, web3, contract, ACCOUNT_ADDRESS, PRIVATE_KEY
from .balance import get_wallet_balance

logger = logging.getLogger(__name__)

def calculate_bet_amount(config, confidence=0.5, consecutive_losses=0):
    """
    Calculate the optimal bet amount based on config settings and prediction confidence.
    
    Args:
        config: The loaded config dictionary
        confidence: Prediction confidence (0.0-1.0)
        consecutive_losses: Number of consecutive losses for martingale strategy
        
    Returns:
        float: Amount to bet in BNB
    """
    try:
        # Get trading config with defaults
        trading_config = config.get("trading", {})
        
        # Get base bet amount from config - use wager_amount as the key
        base_amount = float(trading_config.get("wager_amount", 0.005))  # Default 0.005 BNB
        logger.info(f"Base bet amount from config: {base_amount} BNB")
        
        # Get min and max bet amounts with defaults
        min_amount = float(trading_config.get("min_bet_amount", base_amount * 0.5))
        max_amount = float(trading_config.get("max_bet_amount", base_amount * 2.0))
        
        # Get bet strategy with default
        strategy = trading_config.get("bet_strategy", "fixed")
        logger.info(f"Using bet strategy: {strategy}")
        
        # Calculate bet amount based on strategy
        if strategy == "fixed":
            # Always use the base amount
            amount = base_amount
            
        elif strategy == "confidence":
            # Scale bet amount based on prediction confidence
            confidence_factor = 0.5 + (confidence * 0.5)  # Range: 0.5-1.0
            amount = base_amount * confidence_factor
            
        elif strategy == "kelly":
            # Kelly criterion - bet size based on edge and bankroll
            wallet_balance = get_wallet_balance(web3, ACCOUNT_ADDRESS)
            
            # Calculate edge based on confidence
            edge = (2 * confidence) - 1  # Range: -1 to 1
            
            if edge <= 0:
                amount = min_amount
            else:
                kelly_fraction = edge * 0.5  # Half-Kelly for safety
                kelly_fraction = min(0.2, max(0.01, kelly_fraction))
                amount = wallet_balance * kelly_fraction
        
        elif strategy == "martingale":
            # Martingale strategy - double bet after loss
            amount = base_amount * (2 ** consecutive_losses)
        
        else:
            # Unknown strategy, use base amount
            logger.warning(f"Unknown betting strategy: {strategy}, using fixed amount")
            amount = base_amount
        
        # Ensure amount is within min/max limits
        amount = max(min_amount, min(amount, max_amount))
        
        # Round to 6 decimal places (BNB precision)
        amount = round(amount, 6)
        
        logger.info(f"Calculated bet amount: {amount} BNB (strategy: {strategy})")
        return amount
        
    except Exception as e:
        logger.error(f"Error calculating bet amount: {e}")
        traceback.print_exc()
        # Return a safe default amount
        return 0.005  # Default to 0.005 BNB if there's an error

def place_bet(epoch, prediction, amount, gas_strategy="medium"):
    """
    Place a bet for the given epoch.
    
    Args:
        epoch: Epoch number to bet on
        prediction: Prediction direction ("BULL" or "BEAR")
        amount: Bet amount in BNB
        gas_strategy: Gas price strategy ('low', 'medium', 'high')
        
    Returns:
        bool: True if successfully placed bet, False otherwise
    """
    try:
        account_address = Web3.to_checksum_address(ACCOUNT_ADDRESS)
        
        # Verify valid prediction
        if prediction.upper() not in ["BULL", "BEAR"]:
            logger.error(f"Invalid prediction: {prediction}. Must be 'BULL' or 'BEAR'")
            return False
            
        # Determine which contract function to call
        if prediction.upper() == "BULL":
            bet_function = contract.functions.betBull
        else:
            bet_function = contract.functions.betBear
        
        # Get gas settings from config
        gas_config = config.get("gas_price", {})
        gas_multiplier = gas_config.get("multipliers", {}).get(gas_strategy, 1.0)
        min_gas = web3.to_wei(gas_config.get("min_gwei", 1.0), 'gwei')
        max_gas = web3.to_wei(gas_config.get("max_gwei", 5.0), 'gwei')
        
        # Calculate gas price with configured values
        base_gas_price = web3.eth.gas_price
        gas_price = int(base_gas_price * gas_multiplier)
        
        # Ensure gas price is within min/max limits
        gas_price = max(min_gas, min(max_gas, gas_price))
        
        # Calculate gas limit from config
        gas_limit = config.get("bet_gas_limit", 300000)
        
        # Convert amount to wei
        amount_wei = web3.to_wei(amount, 'ether')
        
        # Check balance before proceeding
        balance = web3.eth.get_balance(account_address)
        estimated_cost = amount_wei + (gas_price * gas_limit)
        
        if balance < estimated_cost:
            logger.warning(f"Insufficient funds to place {prediction} bet for epoch {epoch}")
            logger.warning(f"Balance: {web3.from_wei(balance, 'ether'):.6f} BNB")
            logger.warning(f"Required: {web3.from_wei(estimated_cost, 'ether'):.6f} BNB")
            return False
        
        # Build transaction
        bet_tx = bet_function(epoch).build_transaction({
            'from': account_address,
            'value': amount_wei,
            'gas': gas_limit,
            'gasPrice': gas_price,
            'nonce': web3.eth.get_transaction_count(account_address),
            'chainId': web3.eth.chain_id
        })
        
        logger.info(f"Placing {prediction} bet of {amount} BNB for epoch {epoch}")
        logger.info(f"Gas price: {web3.from_wei(gas_price, 'gwei'):.2f} gwei, Gas limit: {gas_limit}")
        
        # Sign and send transaction
        signed_tx = web3.eth.account.sign_transaction(bet_tx, PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        logger.info(f"Waiting for bet transaction to be mined... (tx: {tx_hash.hex()})")
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status == 1:
            logger.info(f"Successfully placed {prediction} bet of {amount} BNB for epoch {epoch}")
            return True
        else:
            logger.error(f"Failed to place bet for epoch {epoch}")
            return False
            
    except Exception as e:
        logger.error(f"Error placing bet: {e}")
        traceback.print_exc()
        return False 