"""
Main entry point for the trading bot.
"""

import logging
import argparse
import traceback
import time
import sys

from .core import config
from .prediction import PredictionHandler
from .data.blockchain import BlockchainDataHandler
from .wallet import get_wallet_balance
from .ui import start_web_interface, start_control_panel

logger = logging.getLogger(__name__)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("logs/bot.log"),
            logging.StreamHandler()
        ]
    )

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UglyBot Trading System')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode (no GUI)')
    parser.add_argument('--web-only', action='store_true', help='Run only the web interface')
    parser.add_argument('--desktop-only', action='store_true', help='Run only the desktop interface')
    return parser.parse_args()

def main():
    """Main function to start the trading bot."""
    try:
        # Setup
        setup_logging()
        args = parse_arguments()
        
        logger.info("Starting UglyBot Trading System")
        
        # Start blockchain data handler
        blockchain_handler = BlockchainDataHandler()
        blockchain_handler.start()
        
        # Start prediction handler
        prediction_handler = PredictionHandler()
        
        # Start user interfaces based on arguments
        if not args.headless:
            if not args.desktop_only:
                web_thread = start_web_interface()
                logger.info("Web interface started")
                
            if not args.web_only:
                desktop_thread = start_control_panel()
                logger.info("Desktop interface started")
        
        # Main loop
        while True:
            # Main bot logic
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Shutting down trading bot")
        # Cleanup code
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 