"""
Web interface for the trading bot.
Provides a Flask-based dashboard for monitoring and controlling the bot.
"""

import os
import json
import time
import logging
import psutil
import traceback
from datetime import datetime, timedelta
from threading import Thread
from flask import Flask, render_template, jsonify, request, send_from_directory, send_file

from ..core import config
from ..wallet import get_wallet_balance
from ..data.database import get_recent_trades, get_trade_performance

logger = logging.getLogger(__name__)
app = Flask(__name__)

# Store this in memory for quick access
latest_stats = {
    'last_prediction': None,
    'win_rate': 0,
    'total_trades': 0,
    'current_streak': 0,
    'last_updated': None
}

@app.route('/')
def home():
    """Render the main dashboard."""
    return render_template('index.html')

@app.route('/api/stats')
def get_stats():
    """Get current statistics for the dashboard."""
    try:
        # Read the latest log file
        date_str = datetime.now().strftime('%Y-%m-%d')
        log_file = f'logs/predictions_{date_str}.json'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                logs = json.load(f)
                if logs:
                    latest = logs[-1]
                    
                    # Calculate performance for different periods
                    now = datetime.now()
                    week_ago = now - timedelta(days=7)
                    month_ago = now - timedelta(days=30)
                    
                    def calculate_period_stats(start_date):
                        period_logs = [log for log in logs 
                                     if datetime.strptime(log['timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date]
                        wins = sum(1 for log in period_logs if log.get('won') == True)
                        total = len([log for log in period_logs if log.get('won') is not None])
                        profit = sum(float(log.get('profit_loss', 0)) for log in period_logs)
                        win_rate = (wins / total * 100) if total > 0 else 0
                        return profit, win_rate
                    
                    today_profit, today_winrate = calculate_period_stats(now.replace(hour=0, minute=0, second=0))
                    week_profit, week_winrate = calculate_period_stats(week_ago)
                    month_profit, month_winrate = calculate_period_stats(month_ago)
                    
                    return jsonify({
                        'status': 'success',
                        'data': {
                            'last_prediction': latest.get('final_prediction'),
                            'strategies': latest.get('strategy_predictions', {}),
                            'today_profit': f"{today_profit:.4f} BNB",
                            'today_winrate': f"{today_winrate:.1f}%",
                            'week_profit': f"{week_profit:.4f} BNB",
                            'week_winrate': f"{week_winrate:.1f}%",
                            'month_profit': f"{month_profit:.4f} BNB",
                            'month_winrate': f"{month_winrate:.1f}%",
                            'last_result': '✅ Won' if latest.get('won') else '❌ Lost' if latest.get('won') is not None else 'Pending',
                            'last_updated': latest.get('timestamp')
                        }
                    })
        
        return jsonify({
            'status': 'error',
            'message': 'No data available'
        })
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# Additional routes and functions from the original file would be included here...

def start_web_interface():
    """
    Start the web interface in a separate thread.
    
    Returns:
        Thread: The web interface thread
    """
    try:
        # Create templates directory if it doesn't exist
        os.makedirs('templates', exist_ok=True)
        
        web_thread = Thread(target=app.run, kwargs={
            'host': '0.0.0.0',
            'port': 5000,
            'debug': False,
            'use_reloader': False
        })
        web_thread.daemon = True  # This ensures the thread stops when the main program stops
        web_thread.start()
        logger.info("Web interface started on port 5000")
        return web_thread
    except Exception as e:
        logger.error(f"Failed to start web interface: {e}")
        traceback.print_exc()
        return None 