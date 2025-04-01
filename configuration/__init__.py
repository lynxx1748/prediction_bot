"""
Configuration module for the trading bot.
Provides easy access to configuration settings from a centralized location.
"""

import json
import logging
import os
from pathlib import Path

# Define the configuration directory
CONFIG_DIR = Path(__file__).parent


class Config:
    """Configuration manager for the trading bot."""

    _instance = None
    _config = None

    def __new__(cls):
        """Implement singleton pattern for Config."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._load_config()
        return cls._instance

    @classmethod
    def _load_config(cls):
        """Load configuration from JSON file."""
        config_path = CONFIG_DIR / "config.json"
        try:
            with open(config_path, "r") as f:
                cls._config = json.load(f)
            logging.info("Configuration loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            cls._config = {}

    @classmethod
    def get(cls, key=None, default=None):
        """
        Get configuration value.

        Args:
            key: The configuration key to retrieve (dot notation supported)
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        if cls._config is None:
            cls._load_config()

        if key is None:
            return cls._config

        # Handle nested keys with dot notation (e.g., "trading.bet_strategy")
        keys = key.split(".")
        value = cls._config

        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default

    @classmethod
    def save(cls):
        """Save current configuration to file."""
        config_path = CONFIG_DIR / "config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(cls._config, f, indent=4)
            logging.info("Configuration saved successfully.")
            return True
        except Exception as e:
            logging.error(f"Error saving configuration: {e}")
            return False

    @classmethod
    def update(cls, key, value):
        """
        Update a configuration value.

        Args:
            key: The configuration key to update (dot notation supported)
            value: The new value

        Returns:
            True if successful, False otherwise
        """
        if cls._config is None:
            cls._load_config()

        keys = key.split(".")
        config = cls._config

        # Navigate to the nested dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        # Update the value
        config[keys[-1]] = value
        return cls.save()


# Function to get contract ABI
def get_contract_abi():
    """Load the contract ABI from the JSON file."""
    abi_path = CONFIG_DIR / "abi.json"
    try:
        with open(abi_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading contract ABI: {e}")
        return None


# Create a global config instance for easy imports
config = Config()

# Export public interface
__all__ = ["config", "get_contract_abi", "CONFIG_DIR"]
