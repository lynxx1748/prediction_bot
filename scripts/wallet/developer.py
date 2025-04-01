"""
Developer wallet verification for the trading bot.
Handles developer identification and fee-related functionality.
"""

import base64
import hashlib
import logging
import os
import traceback

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


def create_encrypted_wallet():
    """
    Create encrypted developer wallet file.

    This function encrypts the developer wallet address and stores it in a hidden file.
    Only the developer should run this function to set up the identification.
    """
    try:
        # Developer wallet address (only public address, NO private key!)
        DEV_WALLET = "0xc3f4718e83e78a7986258b2c3e076436e5c77192"

        # Add a checksum for verification
        checksum = hashlib.sha256(DEV_WALLET.encode()).hexdigest()[:8]
        data_to_encrypt = f"{DEV_WALLET}:{checksum}"

        # Generate key from a constant string unique to the project
        key = base64.b64encode(b"UglyBotV102DevWallet2025" + b"=" * 11)
        cipher = Fernet(key)

        # Encrypt the wallet address with checksum
        encrypted_wallet = cipher.encrypt(data_to_encrypt.encode())

        # Save to hidden file
        wallet_file = os.path.join(
            os.path.dirname(__file__), "..", "..", ".dev_wallet.enc"
        )
        with open(wallet_file, "wb") as f:
            f.write(encrypted_wallet)

        logger.info("Developer wallet file created successfully")
        return True

    except Exception as e:
        logger.error(f"Error creating encrypted wallet: {e}")
        traceback.print_exc()
        return False


def get_dev_wallet():
    """
    Get developer wallet address from encrypted file.

    Returns:
        str: Developer wallet address if verification successful, None otherwise
    """
    try:
        # Generate same key used for encryption
        key = base64.b64encode(b"UglyBotV102DevWallet2025" + b"=" * 11)
        cipher = Fernet(key)

        # Read and decrypt
        wallet_file = os.path.join(
            os.path.dirname(__file__), "..", "..", ".dev_wallet.enc"
        )
        with open(wallet_file, "rb") as file:
            encrypted_wallet = file.read()

        decrypted_data = cipher.decrypt(encrypted_wallet).decode()
        wallet, stored_checksum = decrypted_data.split(":")

        # Verify checksum to ensure wallet address hasn't been tampered with
        current_checksum = hashlib.sha256(wallet.encode()).hexdigest()[:8]
        if current_checksum != stored_checksum:
            logger.warning("Wallet verification failed - file may be tampered")
            return None

        return wallet

    except FileNotFoundError:
        logger.warning("Developer wallet file not found")
        return None
    except Exception as e:
        logger.error(f"Developer wallet verification failed: {e}")
        traceback.print_exc()
        return None


def is_dev_wallet(wallet_address):
    """
    Check if the provided wallet address matches the developer wallet.

    Args:
        wallet_address: Wallet address to check

    Returns:
        bool: True if wallet is the developer's, False otherwise
    """
    try:
        dev_wallet = get_dev_wallet()
        if not dev_wallet:
            return False

        # Case-insensitive comparison
        return wallet_address.lower() == dev_wallet.lower()

    except Exception as e:
        logger.error(f"Error checking developer wallet: {e}")
        return False


def calculate_dev_fee(amount, is_dev=False):
    """
    Calculate developer fee amount (2% of winnings).

    Args:
        amount: Total winning amount
        is_dev: Whether the user is the developer

    Returns:
        float: Fee amount (0 if user is developer)
    """
    if is_dev:
        return 0.0

    # 2% dev fee
    fee = amount * 0.02

    # Round to 6 decimal places (BNB precision)
    return round(fee, 6)
