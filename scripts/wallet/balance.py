"""
Wallet balance operations for the trading bot.
"""

import logging
import traceback
from web3 import Web3

logger = logging.getLogger(__name__)

def get_wallet_balance(web3, wallet_address):
    """
    Get the current wallet balance in BNB.
    
    Args:
        web3: Web3 instance
        wallet_address: Wallet address to check
        
    Returns:
        float: Balance in BNB
    """
    try:
        # Convert address to checksum format
        checksum_address = web3.to_checksum_address(wallet_address)
        
        # Get wallet balance in wei
        balance_wei = web3.eth.get_balance(checksum_address)
        
        # Convert to BNB (1 BNB = 10^18 wei)
        balance_bnb = balance_wei / (10 ** 18)
        
        logger.debug(f"Current wallet balance: {balance_bnb:.6f} BNB")
        return balance_bnb
        
    except Exception as e:
        logger.error(f"Error getting wallet balance: {e}")
        traceback.print_exc()
        return 0.0 