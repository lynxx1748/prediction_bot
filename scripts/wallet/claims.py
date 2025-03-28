"""
Claim processing for the trading bot.
Handles reward claims and tracking of claimable rounds.
"""

import logging
import traceback
import re
from web3 import Web3

from ..core.constants import contract, web3, ACCOUNT_ADDRESS, PRIVATE_KEY, config
from ..data.blockchain import get_round_data
from ..data.collector import DataCollector
from ..data.database import update_prediction_outcome

logger = logging.getLogger(__name__)

def claim_rewards(epoch, gas_strategy="medium"):
    """
    Claim rewards for a specific epoch.
    
    Args:
        epoch: Epoch number to claim rewards for
        gas_strategy: Gas price strategy ('low', 'medium', 'high')
        
    Returns:
        bool: True if successfully claimed, False otherwise
    """
    try:
        account_address = Web3.to_checksum_address(ACCOUNT_ADDRESS)
        
        # Get gas settings from config
        gas_config = config.get("gas_price", {})
        gas_multiplier = gas_config.get("multipliers", {}).get(gas_strategy, 1.0)
        min_gas = web3.to_wei(gas_config.get("min_gwei", 1.0), 'gwei')
        max_gas = web3.to_wei(gas_config.get("max_gwei", 5.0), 'gwei')
        
        # Calculate gas price with configured values
        base_gas_price = web3.eth.gas_price
        gas_price = int(base_gas_price * gas_multiplier)
        
        # Ensure gas price is within min/max limits and at least 1 gwei
        gas_price = max(min_gas, min(max_gas, gas_price))
        network_min_gas = web3.to_wei(1.0, 'gwei')  # Ensure at least 1 gwei
        if gas_price < network_min_gas:
            gas_price = network_min_gas
            logger.warning(f"Increased gas price to minimum network requirement: {web3.from_wei(gas_price, 'gwei')} gwei")
        
        # Calculate gas limit from config
        gas_limit = config.get("claim_gas_limit", 200000)
        
        # Check balance before proceeding
        balance = web3.eth.get_balance(account_address)
        estimated_cost = gas_price * gas_limit
        
        if balance < estimated_cost:
            logger.warning(f"Insufficient funds to claim rewards for epoch {epoch}")
            logger.warning(f"Balance: {web3.from_wei(balance, 'ether'):.6f} BNB")
            logger.warning(f"Required: {web3.from_wei(estimated_cost, 'ether'):.6f} BNB")
            return False
        
        # Convert epoch to integer
        epoch_int = int(epoch)
        
        # Build transaction with configured gas values
        claim_tx = contract.functions.claim([epoch_int]).build_transaction({
            'from': account_address,
            'gas': gas_limit,
            'gasPrice': gas_price,
            'nonce': web3.eth.get_transaction_count(account_address),
            'chainId': web3.eth.chain_id
        })
        
        logger.info(f"Using gas strategy: {gas_strategy}, Gas price: {web3.from_wei(gas_price, 'gwei'):.2f} gwei, Gas limit: {gas_limit}")
        
        # Sign and send transaction
        signed_tx = web3.eth.account.sign_transaction(claim_tx, PRIVATE_KEY)
        tx_hash = web3.eth.send_raw_transaction(signed_tx.raw_transaction)
        
        logger.info(f"Waiting for claim transaction to be mined... (tx: {tx_hash.hex()})")
        tx_receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
        
        if tx_receipt.status == 1:
            logger.info(f"Successfully claimed rewards for epoch {epoch}")
            return True
        else:
            logger.error(f"Failed to claim rewards for epoch {epoch}")
            return False
            
    except Exception as e:
        # Check for "transaction underpriced" error
        if "transaction underpriced" in str(e):
            # Extract the minimum required gas from the error message
            min_gas_match = re.search(r'minimum needed (\d+)', str(e))
            if min_gas_match:
                min_gas_needed = int(min_gas_match.group(1))
                logger.warning(f"Transaction underpriced. Retrying with minimum gas: {web3.from_wei(min_gas_needed, 'gwei')} gwei")
                
                # Retry with the exact minimum gas needed plus a small buffer
                return claim_rewards(epoch, "medium")  # Use medium strategy which should be higher
        
        logger.error(f"Error claiming rewards: {e}")
        traceback.print_exc()
        return False

def check_claimable_rounds(active_round=None, placed_bets={}, claimable_rounds=[], wins=0, losses=0, consecutive_losses=0):
    """
    Check for rounds with claimable winnings and record outcomes.
    
    Args:
        active_round: Current active round
        placed_bets: Dictionary of placed bets
        claimable_rounds: List of rounds with claimable winnings
        wins: Total wins count
        losses: Total losses count
        consecutive_losses: Current consecutive losses
        
    Returns:
        tuple: (placed_bets, claimable_rounds, wins, losses, consecutive_losses)
    """
    try:
        # Get current epoch if not provided
        if active_round is None:
            active_round = contract.functions.currentEpoch().call() - 1
            
        # Check all previous bets
        for round_epoch in list(placed_bets.keys()):
            if int(round_epoch) >= active_round:
                continue
                
            # Get round data
            round_data = get_round_data(int(round_epoch))
            if not round_data or not round_data.get('closePrice'):
                continue
            
            # Record outcome in database
            update_prediction_outcome(round_epoch, round_data)
            
            our_prediction = placed_bets[round_epoch]
            actual_outcome = round_data['outcome']
            
            if actual_outcome:
                # Record round data for AI learning
                data_point = {
                    'epoch': round_epoch,
                    'bullAmount': round_data['bullAmount'],
                    'bearAmount': round_data['bearAmount'],
                    'totalAmount': round_data['totalAmount'],
                    'bullRatio': round_data['bullRatio'],
                    'bearRatio': round_data['bearRatio'],
                    'lockPrice': round_data['lockPrice'],
                    'closePrice': round_data['closePrice'],
                    'prediction': our_prediction,
                    'actual_outcome': actual_outcome,
                    'win': 1 if our_prediction.upper() == actual_outcome.upper() else 0
                }
                
                # Record to database
                collector = DataCollector(config)
                collector.record_data_point(data_point, round_data)
                
                # Update stats
                if our_prediction.upper() == actual_outcome.upper():
                    wins += 1
                    consecutive_losses = 0
                    logger.info(f"Round {round_epoch}: {our_prediction.upper()} bet WON!")
                    
                    # Add to claimable rounds if we won
                    if round_epoch not in claimable_rounds:
                        claimable_rounds.append(round_epoch)
                        logger.info(f"Added round {round_epoch} to claimable rounds")
                else:
                    losses += 1
                    consecutive_losses += 1
                    logger.info(f"Round {round_epoch}: {our_prediction.upper()} bet LOST!")

                del placed_bets[round_epoch]
                
        # Claim winnings if we have any claimable rounds
        if claimable_rounds and config.get("auto_claim", True):
            logger.info(f"Attempting to claim winnings for rounds: {claimable_rounds}")
            for epoch in list(claimable_rounds):
                if claim_rewards(epoch):
                    claimable_rounds.remove(epoch)
                    logger.info(f"Successfully claimed rewards for round {epoch}")
                else:
                    logger.warning(f"Failed to claim rewards for round {epoch}")

        return placed_bets, claimable_rounds, wins, losses, consecutive_losses

    except Exception as e:
        logger.error(f"Error checking claimable rounds: {e}")
        traceback.print_exc()
        return placed_bets, claimable_rounds, wins, losses, consecutive_losses 