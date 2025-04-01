"""
Configuration management for the trading bot.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "database": {
        "file": "data/historical_data.db",
        "tables": {
            "trades": "trades",
            "predictions": "predictions",
            "signal_performance": "signal_performance",
            "strategy_performance": "strategy_performance",
            "market_data": "market_data",
        },
    },
    "model": {
        "model_file": "./data/random_forest_model.pkl",
        "scaler_file": "./data/random_forest_scaler.pkl",
        "update_threshold": 20,
    },
    "trading": {
        "min_confidence": 0.52,
        "wager_amount": 0.005,
        "betting_mode": "test",
        "stop_loss": 5,
        "prediction_weights": {
            "model": 0.15,
            "pattern": 0.20,
            "market": 0.20,
            "technical": 0.20,
            "sentiment": 0.25,
        },
    },
    "market_bias": {
        "enabled": True,
        "bias_direction": "BULL",
        "bias_strength": 0.15,
        "min_confidence": 0.45,
    },
}


class Config:
    """Configuration management class."""

    def __init__(self):
        """Initialize configuration with defaults."""
        self.config = DEFAULT_CONFIG.copy()

    def load(self, config_path=None):
        """
        Load configuration from file.

        Args:
            config_path: Path to config file (if None, searches in standard locations)

        Returns:
            bool: True if config loaded successfully
        """
        # Locations to search
        config_paths = [
            config_path,
            "config.json",
            "configuration/config.json",
            "../config.json",
            "./config.json",
        ]

        # Filter out None values
        config_paths = [p for p in config_paths if p]

        # Try each path
        for path in config_paths:
            try:
                with open(path, "r") as config_file:
                    loaded_config = json.load(config_file)

                    # Deep merge with default config
                    self._deep_update(self.config, loaded_config)

                    logger.info(f"✅ Loaded config from {path}")
                    return True
            except FileNotFoundError:
                continue
            except json.JSONDecodeError:
                logger.error(f"❌ Invalid JSON in {path}")
                continue

        logger.error("❌ Could not find valid config.json")
        return False

    def _deep_update(self, d, u):
        """
        Recursively update nested dictionaries.

        Args:
            d: Dictionary to update
            u: Dictionary with updates
        """
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._deep_update(d[k], v)
            else:
                d[k] = v

    def get(self, key_path=None, default=None):
        """
        Get configuration value.

        Args:
            key_path: Dot-separated path to config value (e.g., 'database.file')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if key_path is None:
            return self.config

        # Navigate through nested dictionaries
        keys = key_path.split(".")
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path, value):
        """
        Set configuration value.

        Args:
            key_path: Dot-separated path to config value
            value: Value to set

        Returns:
            bool: True if successful
        """
        # Handle empty key
        if not key_path:
            return False

        # Navigate through nested dictionaries
        keys = key_path.split(".")
        config = self.config

        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config or not isinstance(config[key], dict):
                config[key] = {}
            config = config[key]

        # Set the value at the target key
        config[keys[-1]] = value
        return True

    def save(self, file_path="config.json"):
        """
        Save configuration to file.

        Args:
            file_path: Path to save config file

        Returns:
            bool: True if successful
        """
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            # Write config to file
            with open(file_path, "w") as f:
                json.dump(self.config, f, indent=2)

            logger.info(f"✅ Saved config to {file_path}")
            return True
        except Exception as e:
            logger.error(f"❌ Error saving config: {e}")
            return False


# Create singleton instance
config = Config()
