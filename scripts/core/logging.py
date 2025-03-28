"""
Logging functionality for the trading bot.
"""

import json
from datetime import datetime
import os
import logging

logger = logging.getLogger(__name__)

def log_prediction_details(epoch, predictions, final_prediction, outcome=None):
    """
    Log detailed prediction information for each epoch
    
    Args:
        epoch: Current round epoch
        predictions: Dictionary of all strategy predictions
        final_prediction: The final chosen prediction
        outcome: Actual outcome (if known)
    """
    try:
        # Create logs directory if it doesn't exist
        if not os.path.exists('logs'):
            os.makedirs('logs')
            
        # Create the log entry
        log_entry = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'strategy_predictions': {},
            'final_prediction': final_prediction
        }
        
        # Add individual strategy predictions
        for strategy, pred_data in predictions.items():
            if strategy != 'final':  # Skip the final prediction from the predictions dict
                log_entry['strategy_predictions'][strategy] = {
                    'prediction': pred_data.get('prediction'),
                    'confidence': float(pred_data.get('confidence', 0)),
                    'quality': float(pred_data.get('quality', 0))
                }
        
        # Add outcome if provided
        if outcome:
            log_entry['outcome'] = outcome
            log_entry['won'] = (final_prediction.upper() == outcome.upper()) if outcome else None
            
        # Write to daily log file
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_file = f'logs/predictions_{date_str}.json'
        
        # Read existing logs
        existing_logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                try:
                    existing_logs = json.load(f)
                except json.JSONDecodeError:
                    existing_logs = []
        
        # Append new log
        existing_logs.append(log_entry)
        
        # Write updated logs
        with open(log_file, 'w') as f:
            json.dump(existing_logs, f, indent=2)
            
        # Print summary to console
        logger.info("\n📝 Prediction Summary:")
        logger.info(f"Epoch: {epoch}")
        logger.info("Strategy Predictions:")
        for strategy, data in log_entry['strategy_predictions'].items():
            logger.info(f"  {strategy}: {data['prediction']} ({data['confidence']:.2f} confidence)")
        logger.info(f"Final Prediction: {final_prediction}")
        if outcome:
            result = "✅ WON" if log_entry['won'] else "❌ LOST"
            logger.info(f"Outcome: {outcome} {result}")
        logger.info("-" * 50)
            
    except Exception as e:
        logger.error(f"❌ Error logging prediction details: {e}")

def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Setup logging configuration for the application
    
    Args:
        log_level: Logging level (default: INFO)
        log_file: Optional file path for logging
    """
    # Create logs directory if log_file specified and directory doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger 