"""
Database operations for the trading bot.
Handles data storage, retrieval, and analysis.
"""

import sqlite3
import traceback
import logging
from collections import deque
from datetime import datetime
import json
import os
import time

from ..core.constants import DB_FILE, TABLES

# Setup logger
logger = logging.getLogger(__name__)

# Store recent predictions for dynamic weight adjustment
prediction_history = deque(maxlen=50)

# Keep a small cache of recent predictions for quick access
prediction_cache = deque(maxlen=50)

def initialize_prediction_db():
    """Initialize the SQLite database for storing predictions."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if predictions table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['predictions'],))
        table_exists = cursor.fetchone() is not None
        
        # Only create the table if it doesn't exist
        if not table_exists:
            logger.info(f"🔧 Creating predictions table as it doesn't exist")
            # Create predictions table with ALL columns
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLES["predictions"]} (
                epoch INTEGER PRIMARY KEY,
                timestamp INTEGER,
                time INTEGER,
                
                bullAmount REAL,
                bearAmount REAL,
                totalAmount REAL,
                bullRatio REAL,
                bearRatio REAL,
                lockPrice REAL,
                closePrice REAL,
                
                ai_prediction TEXT,
                market_prediction TEXT,
                final_prediction TEXT,
                
                model_confidence REAL,
                trend_confidence REAL,
                market_confidence REAL,
                volume_confidence REAL,
                pattern_confidence REAL,
                ai_confidence REAL,
                technical_confidence REAL,
                final_confidence REAL,
                
                actual_outcome TEXT,
                win INTEGER DEFAULT 0,
                profit_loss REAL DEFAULT 0.0,
                strategy_prediction TEXT,
                strategy_confidence REAL,
                
                model_weight REAL,
                trend_following_weight REAL,
                contrarian_weight REAL,
                volume_analysis_weight REAL,
                market_indicators_weight REAL,
                market_regime_weight REAL,
                
                market_regime_prediction TEXT,
                market_regime_confidence REAL,
                trend_following_prediction TEXT, 
                trend_following_confidence REAL,
                contrarian_prediction TEXT,
                contrarian_confidence REAL,
                volume_analysis_prediction TEXT,
                volume_analysis_confidence REAL,
                mean_reversion_prediction TEXT,
                mean_reversion_confidence REAL,
                volatility_breakout_prediction TEXT,
                volatility_breakout_confidence REAL,
                
                bet_amount REAL DEFAULT 0.0,
                bet_strategy TEXT
            )''')
        else:
            # If table exists, check for and add missing columns
            # Get existing columns
            cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
            existing_columns = [column[1] for column in cursor.fetchall()]
            
            # Define columns that should exist
            required_columns = [
                ('time', 'INTEGER'),
                ('bullRatio', 'REAL'),
                ('bearRatio', 'REAL'),
                ('market_prediction', 'TEXT'),
                ('market_confidence', 'REAL')
            ]
            
            # Add missing columns
            for col_name, col_type in required_columns:
                if col_name not in existing_columns:
                    try:
                        logger.info(f"Adding missing column {col_name} to predictions table")
                        cursor.execute(f"ALTER TABLE {TABLES['predictions']} ADD COLUMN {col_name} {col_type}")
                    except Exception as e:
                        logger.error(f"Error adding column {col_name}: {e}")
        
        # Also ensure the trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['trades'],))
        trades_table_exists = cursor.fetchone() is not None
        
        if not trades_table_exists:
            logger.info(f"🔧 Creating trades table as it doesn't exist")
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLES["trades"]} (
                epoch INTEGER PRIMARY KEY,
                timestamp INTEGER,
                startTime INTEGER,
                lockTime INTEGER,
                closeTime INTEGER,
                lockPrice REAL,
                closePrice REAL,
                totalAmount REAL,
                bullAmount REAL,
                bearAmount REAL,
                bullRatio REAL,
                bearRatio REAL,
                outcome TEXT
            )''')
            
        # Create market data table if it doesn't exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['market_data'],))
        market_table_exists = cursor.fetchone() is not None
        
        if not market_table_exists:
            logger.info(f"🔧 Creating market_data table as it doesn't exist")
            cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {TABLES["market_data"]} (
                timestamp INTEGER PRIMARY KEY,
                datetime TEXT,
                bnb_price REAL,
                bnb_24h_volume REAL,
                bnb_24h_change REAL,
                bnb_high REAL,
                bnb_low REAL,
                btc_price REAL,
                btc_24h_volume REAL,
                btc_24h_change REAL,
                fear_greed_value INTEGER,
                fear_greed_class TEXT,
                current_epoch INTEGER,
                totalAmount REAL,
                bullAmount REAL,
                bearAmount REAL,
                bullRatio REAL,
                bearRatio REAL
            )''')
            
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        traceback.print_exc()
        return False

def record_prediction(epoch, data):
    """
    Record prediction data to database.
    
    Args:
        epoch: Betting round epoch
        data: Dictionary of prediction data
        
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create table if it doesn't exist
        initialize_prediction_db()
        
        # Get column info to avoid inserting into non-existent columns
        cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
        existing_columns = [column[1] for column in cursor.fetchall()]
        
        # Check if record already exists
        cursor.execute(f"SELECT * FROM {TABLES['predictions']} WHERE epoch = ?", (epoch,))
        existing = cursor.fetchone()
        
        # Convert underscore naming to camelCase for database compatibility
        renamed_data = data.copy()
        
        # Handle common naming mismatches
        column_mappings = {
            'bull_ratio': 'bullRatio',
            'bear_ratio': 'bearRatio',
            'time': 'timestamp'
        }
        
        for original, db_name in column_mappings.items():
            if original in renamed_data and db_name in existing_columns:
                renamed_data[db_name] = renamed_data.pop(original)
        
        # Create a filtered data dictionary that only contains valid columns
        filtered_data = {}
        for key, value in renamed_data.items():
            if key in existing_columns:
                filtered_data[key] = value
            else:
                logger.warning(f"Skipping column '{key}' - not in predictions table")
        
        # IMPORTANT: Make sure timestamp is present (critical for many operations)
        # If timestamp is missing but time is available, use time as timestamp
        if 'timestamp' in existing_columns and 'timestamp' not in filtered_data:
            if 'time' in data:
                filtered_data['timestamp'] = data['time']
            else:
                # Use current time as fallback
                filtered_data['timestamp'] = int(time.time())
                logger.info(f"Adding current timestamp {filtered_data['timestamp']} as fallback")
        
        # Prepare SQL based on whether we're inserting or updating
        if existing:
            # Build SET clause
            set_clauses = []
            params = []
            
            for key, value in filtered_data.items():
                if key != 'epoch':  # Skip the primary key
                    set_clauses.append(f"{key} = ?")
                    params.append(value)
                    
            params.append(epoch)  # Add epoch for the WHERE clause
            
            # Execute update
            sql = f"UPDATE {TABLES['predictions']} SET {', '.join(set_clauses)} WHERE epoch = ?"
            cursor.execute(sql, params)
            
        else:
            # Build column and param placeholders lists
            columns = ['epoch'] + [key for key in filtered_data.keys() if key != 'epoch']
            placeholders = ["?"] * len(columns)
            
            # Put epoch first in params
            params = [epoch] + [filtered_data[key] for key in columns[1:]]
            
            # Execute insert
            sql = f"INSERT INTO {TABLES['predictions']} ({', '.join(columns)}) VALUES ({', '.join(placeholders)})"
            cursor.execute(sql, params)
            
        # Update cache
        prediction_cache.append({
            'epoch': epoch,
            **filtered_data
        })
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error recording prediction data: {e}")
        traceback.print_exc()
        return False

def record_prediction_outcome(epoch, outcome):
    """
    Record actual outcome of a prediction.
    
    Args:
        epoch: Betting round epoch
        outcome: Actual outcome ("BULL" or "BEAR")
        
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get the prediction for this epoch
        cursor.execute(f"SELECT final_prediction FROM {TABLES['predictions']} WHERE epoch = ?", (epoch,))
        row = cursor.fetchone()
        
        if row is None:
            logger.warning(f"⚠️ No prediction found for epoch {epoch}")
            conn.close()
            return False
            
        prediction = row[0]
        
        # Determine if prediction was correct
        win = 1 if prediction == outcome else 0
        
        # Record the outcome
        cursor.execute(f'''
            UPDATE {TABLES['predictions']} 
            SET actual_outcome = ?, win = ? 
            WHERE epoch = ?
        ''', (outcome, win, epoch))
        
        # Also update cached predictions
        for i, pred in enumerate(prediction_cache):
            if pred.get('epoch') == epoch:
                prediction_cache[i]['actual_outcome'] = outcome
                prediction_cache[i]['win'] = win
                break
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Recorded outcome for epoch {epoch}: {outcome}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error recording prediction outcome: {e}")
        traceback.print_exc()
        return False

def update_prediction_outcome(epoch, outcome=None, win=None, profit_loss=None, bet_amount=None):
    """
    Update prediction outcome with additional data.
    
    Args:
        epoch: Betting round epoch
        outcome: Optional outcome to update
        win: Optional win status (1 for win, 0 for loss)
        profit_loss: Optional profit/loss amount
        bet_amount: Optional bet amount
    
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if record exists
        cursor.execute(f"SELECT * FROM {TABLES['predictions']} WHERE epoch = ?", (epoch,))
        existing = cursor.fetchone()
        
        if existing is None:
            logger.warning(f"⚠️ No prediction found for epoch {epoch}")
            conn.close()
            return False
        
        # Build update clause
        set_clauses = []
        params = []
        
        if outcome is not None:
            set_clauses.append("actual_outcome = ?")
            params.append(outcome)
        
        if win is not None:
            set_clauses.append("win = ?")
            params.append(win)
        
        if profit_loss is not None:
            set_clauses.append("profit_loss = ?")
            params.append(profit_loss)
            
        if bet_amount is not None:
            set_clauses.append("bet_amount = ?")
            params.append(bet_amount)
            
        if not set_clauses:
            logger.warning("⚠️ No updates provided for prediction outcome")
            conn.close()
            return False
            
        params.append(epoch)  # For WHERE clause
        
        # Execute update
        cursor.execute(f"""
            UPDATE {TABLES['predictions']} 
            SET {', '.join(set_clauses)} 
            WHERE epoch = ?
        """, params)
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating prediction outcome: {e}")
        traceback.print_exc()
        return False

def get_prediction_accuracy(min_samples=5):
    """
    Calculate recent prediction accuracy.
    
    Args:
        min_samples: Minimum number of predictions needed for accuracy calculation
        
    Returns:
        tuple: (accuracy, consecutive_wrong)
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get recent predictions with outcomes
        cursor.execute(f'''
            SELECT final_prediction, actual_outcome
            FROM {TABLES['predictions']}
            WHERE final_prediction IS NOT NULL AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT 20
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results or len(results) < min_samples:
            return 0.5, 0  # Not enough data
            
        # Calculate overall accuracy
        total = len(results)
        correct = sum(1 for r in results if r[0] == r[1])
        accuracy = correct / total
        
        # Calculate consecutive wrong predictions
        consecutive_wrong = 0
        for r in results:
            if r[0] != r[1]:  # Wrong prediction
                consecutive_wrong += 1
            else:
                break  # Stop counting at first correct prediction
                
        return accuracy, consecutive_wrong
        
    except Exception as e:
        logger.error(f"❌ Error calculating prediction accuracy: {e}")
        traceback.print_exc()
        return 0.5, 0  # Default values

def get_recent_predictions(limit=10):
    """
    Get recent predictions from database.
    
    Args:
        limit: Number of predictions to return
        
    Returns:
        list: List of recent predictions
    """
    # Check cache first
    if prediction_cache and len(prediction_cache) >= limit:
        return list(prediction_cache)[-limit:]
        
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # Return results as dictionaries
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT *
            FROM {TABLES['predictions']}
            ORDER BY epoch DESC
            LIMIT {limit}
        ''')
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        # Update cache with database results
        for row in results:
            # Only add to cache if not already present
            if not any(p.get('epoch') == row['epoch'] for p in prediction_cache):
                prediction_cache.append(row)
                
        return results
        
    except Exception as e:
        logger.error(f"❌ Error getting recent predictions: {e}")
        traceback.print_exc()
        return []

def get_overall_performance(lookback=100):
    """
    Get overall prediction performance.
    
    Args:
        lookback: Number of recent predictions to analyze
        
    Returns:
        dict: Performance metrics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get predictions with outcomes
        cursor.execute(f'''
            SELECT win, final_prediction, actual_outcome
            FROM {TABLES['predictions']}
            WHERE final_prediction IS NOT NULL AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT {lookback}
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {
                'accuracy': 0.5,
                'sample_size': 0,
                'bull_accuracy': 0.5,
                'bear_accuracy': 0.5
            }
        
        # Calculate overall metrics
        total = len(results)
        wins = sum(r[0] for r in results)
        accuracy = wins / total if total > 0 else 0.5
        
        # Calculate bull/bear specific metrics
        bull_preds = [r for r in results if r[1] == 'BULL']
        bull_wins = sum(1 for r in bull_preds if r[0] == 1)
        bull_accuracy = bull_wins / len(bull_preds) if bull_preds else 0.5
        
        bear_preds = [r for r in results if r[1] == 'BEAR']
        bear_wins = sum(1 for r in bear_preds if r[0] == 1)
        bear_accuracy = bear_wins / len(bear_preds) if bear_preds else 0.5
        
        # Calculate streak
        current_streak = 1
        streak_type = 'win' if results[0][0] == 1 else 'loss'
        
        for i in range(1, len(results)):
            if (results[i][0] == 1 and streak_type == 'win') or (results[i][0] == 0 and streak_type == 'loss'):
                current_streak += 1
            else:
                break
                
        return {
            'accuracy': accuracy,
            'sample_size': total,
            'bull_accuracy': bull_accuracy,
            'bear_accuracy': bear_accuracy,
            'bull_predictions': len(bull_preds),
            'bear_predictions': len(bear_preds),
            'streak': current_streak,
            'streak_type': streak_type
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting overall performance: {e}")
        traceback.print_exc()
        return {
            'accuracy': 0.5,
            'sample_size': 0,
            'bull_accuracy': 0.5,
            'bear_accuracy': 0.5
        }

def get_recent_price_changes(lookback=10):
    """Get recent price changes for short-term analysis"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute(f"""
            SELECT lockPrice, closePrice
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {lookback}
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        price_changes = []
        for lock, close in results:
            if lock and close and lock > 0:
                price_changes.append((close - lock) / lock)
            
        return price_changes
    
    except Exception as e:
        logger.error(f"❌ Error getting recent price changes: {e}")
        return []

def get_recent_rounds(count=10):
    """
    Get recent rounds from database.
    
    Args:
        count: Number of rounds to return
        
    Returns:
        list: List of recent rounds as dictionaries
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row  # Return results as dictionaries
        cursor = conn.cursor()
        
        cursor.execute(f'''
            SELECT *
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {count}
        ''')
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Error getting recent rounds: {e}")
        traceback.print_exc()
        return []

def record_market_data(data):
    """
    Record market data to database.
    
    Args:
        data: Dictionary of market data
        
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Ensure table exists
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["market_data"]} (
            timestamp INTEGER PRIMARY KEY,
            datetime TEXT,
            bnb_price REAL,
            bnb_24h_volume REAL,
            bnb_24h_change REAL,
            bnb_high REAL,
            bnb_low REAL,
            btc_price REAL,
            btc_24h_volume REAL,
            btc_24h_change REAL,
            fear_greed_value INTEGER,
            fear_greed_class TEXT,
            current_epoch INTEGER,
            totalAmount REAL,
            bullAmount REAL,
            bearAmount REAL,
            bullRatio REAL,
            bearRatio REAL
        )''')
        
        # Add datetime if missing
        if 'datetime' not in data and 'timestamp' in data:
            data['datetime'] = datetime.fromtimestamp(data['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            
        # Get columns and values
        columns = list(data.keys())
        placeholders = ', '.join(['?'] * len(columns))
        values = [data[col] for col in columns]
        
        # Insert data
        cursor.execute(f'''
            INSERT OR REPLACE INTO {TABLES["market_data"]} 
            ({', '.join(columns)})
            VALUES ({placeholders})
        ''', values)
        
        conn.commit()
        conn.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Error recording market data: {e}")
        traceback.print_exc()
        return False

def get_adaptive_weights():
    """
    Get adaptive weights based on strategy performance.
    
    Returns:
        dict: Strategy weights
    """
    try:
        # Default weights
        weights = {
            "model": 0.15,
            "pattern": 0.20,
            "market": 0.20, 
            "technical": 0.25,
            "sentiment": 0.20
        }
        
        # Read performance data if available
        performance_file = os.path.join('data', 'strategy_performance.json')
        if os.path.exists(performance_file):
            try:
                with open(performance_file, 'r') as f:
                    performance = json.load(f)
                    
                # Adjust weights based on accuracy
                if 'strategies' in performance:
                    total_accuracy = 0
                    strategy_accuracies = {}
                    
                    # Get accuracies for each strategy
                    for strategy, stats in performance['strategies'].items():
                        if 'accuracy' in stats and stats.get('sample_size', 0) >= 10:
                            accuracy = stats['accuracy']
                            strategy_accuracies[strategy] = accuracy
                            total_accuracy += accuracy
                    
                    # Calculate normalized weights
                    if total_accuracy > 0:
                        for strategy, accuracy in strategy_accuracies.items():
                            weights[strategy] = 0.1 + (accuracy / total_accuracy) * 0.9
                            
            except Exception as e:
                logger.error(f"❌ Error reading performance data: {e}")
                
        # Ensure weights sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            for key in weights:
                weights[key] = weights[key] / total_weight
                
        return weights
        
    except Exception as e:
        logger.error(f"❌ Error calculating adaptive weights: {e}")
        return {
            "model": 0.15,
            "pattern": 0.20,
            "market": 0.20,
            "technical": 0.25,
            "sentiment": 0.20
        }

def store_historical_data(round_data, verbose=True):
    """
    Store historical round data in database.
    
    Args:
        round_data: Dictionary with round data
        verbose: Whether to print detailed info
        
    Returns:
        bool: Success status
    """
    try:
        # Normalize the data format
        epoch = round_data.get('epoch') or round_data.get('id')
        if not epoch:
            if verbose:
                logger.warning(f"⚠️ No epoch/id found in round data")
            return False
            
        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Ensure trades table exists
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["trades"]} (
            epoch INTEGER PRIMARY KEY,
            timestamp INTEGER,
            startTime INTEGER,
            lockTime INTEGER,
            closeTime INTEGER,
            lockPrice REAL,
            closePrice REAL,
            totalAmount REAL,
            bullAmount REAL,
            bearAmount REAL,
            bullRatio REAL,
            bearRatio REAL,
            outcome TEXT
        )''')
        
        # Extract data from round_data
        start_time = round_data.get('startTimestamp', round_data.get('startTime', 0))
        lock_time = round_data.get('lockTimestamp', round_data.get('lockTime', 0))
        close_time = round_data.get('closeTimestamp', round_data.get('closeTime', 0))
        lock_price = round_data.get('lockPrice', 0)
        close_price = round_data.get('closePrice', 0)
        
        if isinstance(lock_price, str):
            lock_price = float(lock_price)
        if isinstance(close_price, str):
            close_price = float(close_price)
            
        # Bull and bear amounts
        bull_amount = round_data.get('bullAmount', 0)
        bear_amount = round_data.get('bearAmount', 0)
        
        if isinstance(bull_amount, str):
            bull_amount = float(bull_amount)
        if isinstance(bear_amount, str):
            bear_amount = float(bear_amount)
            
        # Calculate totals and ratios
        total_amount = bull_amount + bear_amount
        bull_ratio = bull_amount / total_amount if total_amount > 0 else 0.5
        bear_ratio = bear_amount / total_amount if total_amount > 0 else 0.5
        
        # Determine outcome
        outcome = None
        if close_price > lock_price:
            outcome = "BULL"
        elif close_price < lock_price:
            outcome = "BEAR"
            
        # Insert data
        cursor.execute(f'''
        INSERT OR REPLACE INTO {TABLES["trades"]} (
            epoch, timestamp, startTime, lockTime, closeTime,
            lockPrice, closePrice, totalAmount, bullAmount,
            bearAmount, bullRatio, bearRatio, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            epoch, int(time.time()), start_time, lock_time, close_time,
            lock_price, close_price, total_amount, bull_amount,
            bear_amount, bull_ratio, bear_ratio, outcome
        ))
        
        conn.commit()
        conn.close()
        
        if verbose:
            logger.info(f"✅ Stored historical data for round {epoch}")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Error storing historical data: {e}")
        traceback.print_exc()
        return False 

def update_signal_performance(signal_type, was_correct):
    """
    Update the performance metrics for a prediction signal.
    
    Args:
        signal_type: Type of prediction signal
        was_correct: Whether the prediction was correct
        
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create signal_performance table if it doesn't exist
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["signal_performance"]} (
            signal_type TEXT PRIMARY KEY,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL
        )
        ''')
        
        # Update or insert performance record
        cursor.execute(f'''
        INSERT INTO {TABLES["signal_performance"]} (signal_type, total_predictions, correct_predictions)
        VALUES (?, 1, ?)
        ON CONFLICT(signal_type) DO UPDATE SET
            total_predictions = total_predictions + 1,
            correct_predictions = correct_predictions + ?,
            accuracy = CAST(correct_predictions + ? AS REAL) / (total_predictions + 1)
        ''', (signal_type, int(was_correct), int(was_correct), int(was_correct)))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error updating signal performance: {e}")
        traceback.print_exc()
        return False

def initialize_database(force=False):
    """
    Initialize all database tables.
    
    Args:
        force: Whether to force delete and recreate the database
        
    Returns:
        bool: Success status
    """
    try:
        # For a clean start, delete the DB file if force=True
        if force and os.path.exists(DB_FILE):
            os.remove(DB_FILE)
            logger.info(f"✅ Deleted existing database: {DB_FILE}")
            
        # Initialize main prediction DB with all tables
        initialize_prediction_db()
        
        # Add strategy prediction columns
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Add all strategy prediction columns
        for strategy_name in ["market_regime", "trend_following", "mean_reversion", "volatility_breakout"]:
            try:
                cursor.execute(f"ALTER TABLE {TABLES['predictions']} ADD COLUMN {strategy_name}_prediction TEXT")
                cursor.execute(f"ALTER TABLE {TABLES['predictions']} ADD COLUMN {strategy_name}_confidence REAL")
                logger.info(f"✅ Added {strategy_name} prediction columns to database")
            except sqlite3.OperationalError:
                pass  # Column may already exist
                
        conn.commit()
        conn.close()
        
        logger.info("✅ Database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing database: {e}")
        traceback.print_exc()
        return False

def initialize_all_tables():
    """
    Initialize or fix all tables in the database.
    
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["predictions"]} (
            epoch INTEGER PRIMARY KEY,
            timestamp INTEGER,
            
            bullAmount REAL,
            bearAmount REAL,
            totalAmount REAL,
            bullRatio REAL,
            bearRatio REAL,
            lockPrice REAL,
            closePrice REAL,
            
            ai_prediction TEXT,
            final_prediction TEXT,
            
            model_confidence REAL,
            trend_confidence REAL,
            market_confidence REAL,
            volume_confidence REAL,
            pattern_confidence REAL,
            ai_confidence REAL,
            technical_confidence REAL,
            final_confidence REAL,
            
            actual_outcome TEXT,
            win INTEGER DEFAULT 0,
            profit_loss REAL DEFAULT 0.0,
            strategy_prediction TEXT,
            strategy_confidence REAL
        )
        ''')
        
        # Create trades table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["trades"]} (
            epoch INTEGER PRIMARY KEY,
            timestamp INTEGER,
            startTime INTEGER,
            lockTime INTEGER,
            closeTime INTEGER,
            lockPrice REAL,
            closePrice REAL,
            totalAmount REAL,
            bullAmount REAL,
            bearAmount REAL,
            bullRatio REAL,
            bearRatio REAL,
            outcome TEXT
        )
        ''')
        
        # Create signal performance table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["signal_performance"]} (
            signal_type TEXT PRIMARY KEY,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL
        )
        ''')
        
        # Create strategy performance table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["strategy_performance"]} (
            strategy_name TEXT PRIMARY KEY,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL,
            last_updated INTEGER
        )
        ''')
        
        # Create market data table
        cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {TABLES["market_data"]} (
            timestamp INTEGER PRIMARY KEY,
            bnb_price REAL,
            bnb_change_24h REAL,
            btc_price REAL,
            btc_change_24h REAL,
            eth_price REAL,
            eth_change_24h REAL,
            fear_greed_index INTEGER,
            market_sentiment TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("✅ All tables initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error initializing all tables: {e}")
        traceback.print_exc()
        return False

def get_prediction_sample_size():
    """
    Get the number of samples we have for win rate calculation.
    
    Returns:
        int: Number of samples
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Count predictions with outcomes
        cursor.execute(f"""
            SELECT COUNT(*) 
            FROM {TABLES['predictions']} 
            WHERE final_prediction IS NOT NULL 
            AND actual_outcome IS NOT NULL
        """)
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count
        
    except Exception as e:
        logger.error(f"❌ Error getting prediction sample size: {e}")
        return 0 

def get_recent_trades(limit=20):
    """
    Get recent trade data from the database.
    
    Args:
        limit: Maximum number of trades to retrieve
        
    Returns:
        list: Recent trade records
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check if trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['trades'],))
        if not cursor.fetchone():
            logger.warning("⚠️ Trades table does not exist")
            return []
        
        # Get column information
        cursor.execute(f"PRAGMA table_info({TABLES['trades']})")
        columns = [info[1] for info in cursor.fetchall()]
        
        # Build query based on available columns
        select_columns = []
        for col in ['epoch', 'timestamp', 'lockPrice', 'closePrice', 'outcome', 
                   'prediction', 'amount', 'profit_loss', 'win', 'bullRatio', 
                   'bearRatio', 'totalAmount']:
            if col in columns:
                select_columns.append(col)
        
        # If timestamp is missing, use a default or fallback column
        if 'timestamp' not in columns and 'epoch' in columns:
            select_columns.append("epoch as timestamp")  # Use epoch as fallback
        
        # Build and execute query
        query = f"SELECT {', '.join(select_columns)} FROM {TABLES['trades']} ORDER BY epoch DESC LIMIT ?"
        cursor.execute(query, (limit,))
        
        # Convert to list of dictionaries
        rows = cursor.fetchall()
        trades = [dict(row) for row in rows]
        
        conn.close()
        
        # Add additional calculated fields
        for trade in trades:
            # Calculate price change
            if trade.get('lockPrice') and trade.get('closePrice'):
                price_change = ((trade['closePrice'] - trade['lockPrice']) / trade['lockPrice']) * 100
                trade['price_change_pct'] = price_change
                
            # Add formatted timestamp if available
            if trade.get('timestamp'):
                try:
                    trade['datetime'] = datetime.fromtimestamp(trade['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
                except:
                    # If timestamp can't be converted, use a placeholder
                    trade['datetime'] = 'Unknown'
        
        return trades
        
    except Exception as e:
        logger.error(f"❌ Error getting recent trades: {e}")
        traceback.print_exc()
        return []

def get_market_balance_stats(lookback=100):
    """
    Get market balance statistics for recent rounds.
    
    Args:
        lookback: Number of rounds to look back
        
    Returns:
        dict: Market balance statistics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if the trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['trades'],))
        if not cursor.fetchone():
            logger.warning("⚠️ Trades table does not exist")
            return {'avg_bull_ratio': 0.5, 'avg_bear_ratio': 0.5}
            
        # Check which columns exist in the table
        cursor.execute(f"PRAGMA table_info({TABLES['trades']})")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Choose appropriate ORDER BY clause based on available columns
        order_by = "ORDER BY epoch DESC"  # Default fallback
        if 'timestamp' in columns:
            order_by = "ORDER BY timestamp DESC"
        elif 'closeTime' in columns:
            order_by = "ORDER BY closeTime DESC" 
            
        # Execute query with dynamic ORDER BY
        cursor.execute(f'''
            SELECT 
                AVG(bullRatio) as avg_bull_ratio,
                AVG(bearRatio) as avg_bear_ratio,
                SUM(CASE WHEN bullRatio > bearRatio THEN 1 ELSE 0 END) as bull_dominant_count,
                SUM(CASE WHEN bearRatio > bullRatio THEN 1 ELSE 0 END) as bear_dominant_count,
                AVG(totalAmount) as avg_total_amount,
                MAX(totalAmount) as max_total_amount,
                MIN(totalAmount) as min_total_amount,
                COUNT(*) as total_rounds
            FROM (
                SELECT bullRatio, bearRatio, totalAmount
                FROM {TABLES['trades']}
                {order_by}
                LIMIT ?
            )
        ''', (lookback,))
        
        result = cursor.fetchone()
        
        if result:
            stats = {
                'avg_bull_ratio': result[0] or 0,
                'avg_bear_ratio': result[1] or 0,
                'bull_dominant_count': result[2] or 0,
                'bear_dominant_count': result[3] or 0,
                'avg_total_amount': result[4] or 0,
                'max_total_amount': result[5] or 0,
                'min_total_amount': result[6] or 0,
                'total_rounds': result[7] or 0
            }
            
            # Calculate some additional metrics
            if stats['total_rounds'] > 0:
                stats['bull_dominant_ratio'] = stats['bull_dominant_count'] / stats['total_rounds']
                stats['bear_dominant_ratio'] = stats['bear_dominant_count'] / stats['total_rounds']
                stats['market_bias'] = stats['avg_bull_ratio'] - stats['avg_bear_ratio']
                
                # Add these fields for compatibility with the new code
                stats['bull_ratio'] = stats['avg_bull_ratio']
                stats['bear_ratio'] = stats['avg_bear_ratio']
            else:
                stats['bull_dominant_ratio'] = 0
                stats['bear_dominant_ratio'] = 0
                stats['market_bias'] = 0
                stats['bull_ratio'] = 0.5
                stats['bear_ratio'] = 0.5
                
            conn.close()
            return stats
        else:
            conn.close()
            return {
                'avg_bull_ratio': 0.5,
                'avg_bear_ratio': 0.5,
                'bull_dominant_count': 0,
                'bear_dominant_count': 0,
                'avg_total_amount': 0,
                'max_total_amount': 0,
                'min_total_amount': 0,
                'total_rounds': 0,
                'bull_dominant_ratio': 0,
                'bear_dominant_ratio': 0,
                'market_bias': 0,
                'bull_ratio': 0.5,
                'bear_ratio': 0.5
            }
            
    except Exception as e:
        logger.error(f"Error getting market balance stats: {e}")
        return {
            'avg_bull_ratio': 0.5,
            'avg_bear_ratio': 0.5,
            'bull_dominant_count': 0,
            'bear_dominant_count': 0,
            'avg_total_amount': 0,
            'max_total_amount': 0,
            'min_total_amount': 0,
            'total_rounds': 0,
            'bull_dominant_ratio': 0,
            'bear_dominant_ratio': 0,
            'market_bias': 0,
            'bull_ratio': 0.5,
            'bear_ratio': 0.5
        }

def get_market_balance_stats_robust(lookback=10):
    """
    Get market balance statistics from trades, with robust column checking.
    This is a backup for when get_market_balance_stats fails.
    
    Args:
        lookback: Number of rounds to analyze
        
    Returns:
        dict: Market balance statistics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['trades'],))
        if not cursor.fetchone():
            logger.warning("⚠️ Trades table does not exist")
            return {'bull_ratio': 0.5, 'bear_ratio': 0.5}
        
        # Get column info
        cursor.execute(f"PRAGMA table_info({TABLES['trades']})")
        columns = [column[1] for column in cursor.fetchall()]
        
        # Determine sorting column
        if 'timestamp' in columns:
            order_by = "timestamp DESC"
        elif 'closeTime' in columns:
            order_by = "closeTime DESC"
        else:
            order_by = "epoch DESC"
        
        # Ensure the essential columns exist
        if 'bullAmount' in columns and 'bearAmount' in columns:
            cursor.execute(f"""
                SELECT bullAmount, bearAmount, 
                       bullRatio, bearRatio, epoch
                FROM {TABLES['trades']}
                ORDER BY {order_by}
                LIMIT ?
            """, (lookback,))
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {'bull_ratio': 0.5, 'bear_ratio': 0.5}
                
            total_bull = 0
            total_bear = 0
            bull_ratios = []
            bear_ratios = []
            
            for row in results:
                bull_amount = row[0] or 0
                bear_amount = row[1] or 0
                total_bull += bull_amount
                total_bear += bear_amount
                
                # Get ratios if available
                if len(row) >= 4 and row[2] is not None and row[3] is not None:
                    bull_ratios.append(row[2])
                    bear_ratios.append(row[3])
            
            # Calculate ratios
            total = total_bull + total_bear
            if total > 0:
                bull_ratio = total_bull / total
                bear_ratio = total_bear / total
            else:
                bull_ratio = 0.5
                bear_ratio = 0.5
                
            # Calculate average ratios if available
            avg_bull_ratio = sum(bull_ratios) / len(bull_ratios) if bull_ratios else bull_ratio
            avg_bear_ratio = sum(bear_ratios) / len(bear_ratios) if bear_ratios else bear_ratio
            
            # Build the stats object in compatible format with original function
            return {
                'avg_bull_ratio': avg_bull_ratio,
                'avg_bear_ratio': avg_bear_ratio,
                'bull_dominant_count': sum(1 for r in bull_ratios if r > 0.5),
                'bear_dominant_count': sum(1 for r in bear_ratios if r > 0.5),
                'avg_total_amount': (total_bull + total_bear) / len(results) if results else 0,
                'max_total_amount': max([(row[0] or 0) + (row[1] or 0) for row in results]) if results else 0,
                'min_total_amount': min([(row[0] or 0) + (row[1] or 0) for row in results]) if results else 0,
                'total_rounds': len(results),
                'bull_dominant_ratio': sum(1 for r in bull_ratios if r > 0.5) / len(bull_ratios) if bull_ratios else 0.5,
                'bear_dominant_ratio': sum(1 for r in bear_ratios if r > 0.5) / len(bear_ratios) if bear_ratios else 0.5,
                'market_bias': avg_bull_ratio - avg_bear_ratio,
                'bull_ratio': bull_ratio,  # Adding these for compatibility with new code
                'bear_ratio': bear_ratio
            }
            
        else:
            conn.close()
            return {'bull_ratio': 0.5, 'bear_ratio': 0.5}
        
    except Exception as e:
        logger.error(f"Error in robust market balance stats: {e}")
        return {
            'avg_bull_ratio': 0.5,
            'avg_bear_ratio': 0.5,
            'bull_dominant_count': 0,
            'bear_dominant_count': 0,
            'avg_total_amount': 0,
            'max_total_amount': 0,
            'min_total_amount': 0,
            'total_rounds': 0,
            'bull_dominant_ratio': 0,
            'bear_dominant_ratio': 0,
            'market_bias': 0,
            'bull_ratio': 0.5,
            'bear_ratio': 0.5
        }

def get_recent_volume_data(lookback=50):
    """
    Get recent volume data for analysis.
    
    Args:
        lookback: Number of rounds to look back
        
    Returns:
        dict: Volume data including trends and statistics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get recent volume data from trades table
        cursor.execute(f'''
            SELECT 
                epoch,
                timestamp,
                totalAmount,
                bullAmount,
                bearAmount,
                outcome
            FROM {TABLES['trades']}
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (lookback,))
        
        columns = [column[0] for column in cursor.description]
        rounds = []
        
        for row in cursor.fetchall():
            round_data = dict(zip(columns, row))
            rounds.append(round_data)
            
        # Get additional volume stats
        cursor.execute(f'''
            SELECT 
                AVG(totalAmount) as avg_volume,
                MAX(totalAmount) as max_volume,
                MIN(totalAmount) as min_volume,
                AVG(bullAmount) as avg_bull_volume,
                AVG(bearAmount) as avg_bear_volume,
                SUM(totalAmount) as total_volume
            FROM (
                SELECT totalAmount, bullAmount, bearAmount
                FROM {TABLES['trades']}
                ORDER BY timestamp DESC
                LIMIT ?
            )
        ''', (lookback,))
        
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            stats = {
                'avg_volume': result[0] or 0,
                'max_volume': result[1] or 0,
                'min_volume': result[2] or 0,
                'avg_bull_volume': result[3] or 0,
                'avg_bear_volume': result[4] or 0,
                'total_volume': result[5] or 0,
                'rounds': rounds,
                'lookback': lookback
            }
            
            # Calculate volume trend (is volume increasing or decreasing?)
            if len(rounds) >= 10:
                recent_5 = sum(r['totalAmount'] for r in rounds[:5]) / 5
                previous_5 = sum(r['totalAmount'] for r in rounds[5:10]) / 5
                stats['volume_trend'] = recent_5 / previous_5 if previous_5 > 0 else 1.0
                stats['volume_trend_pct'] = (stats['volume_trend'] - 1) * 100
            else:
                stats['volume_trend'] = 1.0
                stats['volume_trend_pct'] = 0.0
                
            # Calculate bull vs bear volume ratio
            if stats['avg_bear_volume'] > 0:
                stats['bull_bear_ratio'] = stats['avg_bull_volume'] / stats['avg_bear_volume']
            else:
                stats['bull_bear_ratio'] = 1.0
            
            return stats
        else:
            return {
                'avg_volume': 0,
                'max_volume': 0,
                'min_volume': 0,
                'avg_bull_volume': 0,
                'avg_bear_volume': 0,
                'total_volume': 0,
                'rounds': [],
                'lookback': lookback,
                'volume_trend': 1.0,
                'volume_trend_pct': 0.0,
                'bull_bear_ratio': 1.0
            }
            
    except Exception as e:
        logger.error(f"Error getting recent volume data: {e}")
        return {
            'avg_volume': 0,
            'max_volume': 0,
            'min_volume': 0,
            'avg_bull_volume': 0,
            'avg_bear_volume': 0,
            'total_volume': 0,
            'rounds': [],
            'lookback': lookback,
            'volume_trend': 1.0,
            'volume_trend_pct': 0.0,
            'bull_bear_ratio': 1.0
        } 

def get_prediction(epoch):
    """
    Get prediction data for a specific epoch from the database.
    
    Args:
        epoch: The epoch number to retrieve
        
    Returns:
        dict: Prediction data or None if not found
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query the predictions table
        cursor.execute(
            f"SELECT * FROM {TABLES['predictions']} WHERE epoch = ?",
            (epoch,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Convert to dictionary
            prediction_data = dict(result)
            return prediction_data
        else:
            return None
            
    except sqlite3.Error as e:
        logging.error(f"Database error retrieving prediction for epoch {epoch}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error getting prediction: {e}")
        return None 

def get_performance_stats():
    """
    Get comprehensive performance statistics for the bot.
    
    Returns:
        dict: Dictionary containing performance metrics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get win/loss statistics
        cursor.execute(f'''
            SELECT 
                COUNT(*) as total_bets,
                SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
                SUM(profit_loss) as total_profit_loss
            FROM {TABLES['predictions']}
            WHERE bet_amount > 0 AND actual_outcome IS NOT NULL
        ''')
        
        result = cursor.fetchone()
        
        # Get most recent consecutive losses
        cursor.execute(f'''
            SELECT win
            FROM {TABLES['predictions']}
            WHERE bet_amount > 0 AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT 10
        ''')
        
        recent_results = cursor.fetchall()
        
        conn.close()
        
        # Calculate statistics
        total_bets = result[0] if result[0] else 0
        wins = result[1] if result[1] else 0
        losses = result[2] if result[2] else 0
        profit_loss = result[3] if result[3] else 0
        
        # Calculate win rate
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        # Calculate consecutive losses
        consecutive_losses = 0
        for win_result in recent_results:
            if win_result[0] == 0:  # Loss
                consecutive_losses += 1
            else:
                break
                
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_loss': profit_loss,
            'consecutive_losses': consecutive_losses
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting performance stats: {e}")
        traceback.print_exc()
        return {
            'total_bets': 0,
            'wins': 0,
            'losses': 0,
            'win_rate': 0,
            'profit_loss': 0,
            'consecutive_losses': 0
        } 

def record_trade(epoch, trade_data):
    """
    Record a trade to the database with proper type conversion.
    
    Args:
        epoch: Round epoch
        trade_data: Dictionary containing trade data
        
    Returns:
        bool: Success status
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['trades'],))
        if not cursor.fetchone():
            # Create table with all necessary columns
            cursor.execute(f'''
                CREATE TABLE IF NOT EXISTS {TABLES['trades']} (
                    epoch INTEGER PRIMARY KEY,
                    timestamp INTEGER,
                    datetime TEXT,
                    startTime INTEGER,
                    lockTime INTEGER,
                    closeTime INTEGER,
                    lockPrice REAL,
                    closePrice REAL,
                    bullAmount REAL,
                    bearAmount REAL,
                    totalAmount REAL,
                    bullRatio REAL,
                    bearRatio REAL,
                    outcome TEXT,
                    prediction TEXT,
                    amount REAL,
                    profit_loss REAL,
                    win INTEGER
                )
            ''')
            conn.commit()
        
        # Prepare data, ensuring epoch is passed correctly and types are converted
        trade_data_with_epoch = {}
        
        # Add the epoch
        trade_data_with_epoch['epoch'] = epoch
        
        # Add current timestamp if not present
        if 'timestamp' not in trade_data or not trade_data.get('timestamp'):
            trade_data_with_epoch['timestamp'] = int(time.time())
        else:
            trade_data_with_epoch['timestamp'] = int(trade_data.get('timestamp', 0))
            
        # Add formatted datetime for readability
        trade_data_with_epoch['datetime'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Copy and convert numeric fields
        for field in ['startTime', 'lockTime', 'closeTime']:
            if field in trade_data:
                try:
                    trade_data_with_epoch[field] = int(trade_data[field])
                except (ValueError, TypeError):
                    trade_data_with_epoch[field] = 0
                    
        for field in ['lockPrice', 'closePrice', 'bullAmount', 'bearAmount', 
                     'totalAmount', 'bullRatio', 'bearRatio', 'amount', 'profit_loss']:
            if field in trade_data:
                try:
                    trade_data_with_epoch[field] = float(trade_data[field])
                except (ValueError, TypeError):
                    trade_data_with_epoch[field] = 0.0
        
        # Copy text fields
        for field in ['outcome', 'prediction']:
            if field in trade_data:
                trade_data_with_epoch[field] = str(trade_data[field]) if trade_data[field] else None
        
        # Handle win field separately
        if 'win' in trade_data:
            try:
                trade_data_with_epoch['win'] = int(trade_data['win'])
            except (ValueError, TypeError):
                trade_data_with_epoch['win'] = 0
        
        # Get the columns that exist in the table
        cursor.execute(f"PRAGMA table_info({TABLES['trades']})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Create column list and placeholders for SQL (only including columns that exist in table)
        valid_columns = [col for col in trade_data_with_epoch.keys() if col in columns]
        placeholders = ','.join(['?' for _ in valid_columns])
        
        # Create SQL statement
        sql = f'''
            INSERT OR REPLACE INTO {TABLES['trades']} 
            ({','.join(valid_columns)})
            VALUES ({placeholders})
        '''
        
        # Create values tuple in same order as columns
        values = tuple(trade_data_with_epoch.get(col) for col in valid_columns)
        
        # Print values for debugging
        print("SQL Values:", values)
        print("SQL Types:", [type(v).__name__ for v in values])
        
        # Execute the insert
        cursor.execute(sql, values)
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Recorded trade for epoch {epoch}")
        return True
    except Exception as e:
        logger.error(f"❌ Error recording trade: {e}")
        traceback.print_exc()
        return False

def record_bet(bet_data):
    """Record a bet to the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # First create the table if it doesn't exist with all needed columns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS bets (
                epoch INTEGER PRIMARY KEY,
                timestamp INTEGER,
                datetime TEXT,
                prediction TEXT,
                amount REAL,
                outcome TEXT,
                win INTEGER DEFAULT 0,
                profit_loss REAL DEFAULT 0.0,
                strategy TEXT,
                gas_price REAL,
                tx_hash TEXT,
                test_mode INTEGER DEFAULT 0
            )
        """)
        
        # Check for all required columns and add any that are missing
        cursor.execute("PRAGMA table_info(bets)")
        existing_columns = [col[1] for col in cursor.fetchall()]
        
        # Define required columns (name: type)
        required_columns = {
            'datetime': 'TEXT',
            'strategy': 'TEXT',
            'gas_price': 'REAL',
            'tx_hash': 'TEXT',
            'test_mode': 'INTEGER'
        }
        
        # Add any missing columns
        for column, dtype in required_columns.items():
            if column not in existing_columns:
                cursor.execute(f"ALTER TABLE bets ADD COLUMN {column} {dtype}")
                conn.commit()
                logger.info(f"Added {column} column to bets table")
        
        # Now proceed with the insert
        cursor.execute('''
            INSERT OR REPLACE INTO bets 
            (epoch, timestamp, datetime, prediction, amount, strategy, gas_price, tx_hash, test_mode) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            bet_data.get('epoch'),
            bet_data.get('timestamp', int(time.time())),
            bet_data.get('datetime', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
            bet_data.get('prediction'),
            bet_data.get('amount'),
            bet_data.get('strategy'),
            bet_data.get('gas_price', 0),
            bet_data.get('tx_hash'),
            1 if bet_data.get('simulated', False) else 0
        ))
        
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"❌ Error recording bet: {e}")
        traceback.print_exc()
        return False

def get_strategy_performance():
    """
    Get performance metrics for different prediction strategies.
    
    Returns:
        dict: Dictionary with performance metrics for each strategy
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check table schema to find the correct column name
        cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Determine which column to use for strategy filtering
        strategy_column = 'bet_strategy'  # Default to bet_strategy
        if 'strategy' in columns:
            strategy_column = 'strategy'
        elif 'bet_strategy' not in columns:
            logger.warning("No strategy column found in predictions table")
            strategy_column = None
        
        strategies = ["hybrid", "ai", "technical", "market_regime", 
                      "trend_following", "contrarian", "volume_analysis"]
        
        results = {}
        
        for strategy in strategies:
            # Get performance data for this strategy
            if strategy_column:
                query = f"""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
                        SUM(profit_loss) as profit_loss
                    FROM {TABLES['predictions']} 
                    WHERE final_prediction IS NOT NULL 
                    AND actual_outcome IS NOT NULL
                    AND {strategy_column} = ?
                """
                cursor.execute(query, (strategy,))
            else:
                # If no strategy column exists, get overall stats instead
                query = f"""
                    SELECT 
                        COUNT(*) as total_predictions,
                        SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
                        SUM(profit_loss) as profit_loss
                    FROM {TABLES['predictions']} 
                    WHERE final_prediction IS NOT NULL 
                    AND actual_outcome IS NOT NULL
                """
                cursor.execute(query)
            
            row = cursor.fetchone()
            
            if row and row[0] > 0:
                total = row[0]
                wins = row[1] or 0
                losses = row[2] or 0
                profit_loss = row[3] or 0
                
                # Calculate win rate
                win_rate = wins / total if total > 0 else 0
                
                results[strategy] = {
                    'total': total,  # Add total key for compatibility with filtering.py
                    'total_predictions': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': win_rate,
                    'profit_loss': profit_loss,
                    'sample_size': total  # Add sample_size for compatibility
                }
            else:
                # Default values for strategies with no data
                results[strategy] = {
                    'total': 0,
                    'total_predictions': 0,
                    'wins': 0,
                    'losses': 0,
                    'win_rate': 0,
                    'profit_loss': 0,
                    'sample_size': 0
                }
        
        conn.close()
        return results
        
    except Exception as e:
        logger.error(f"❌ Error getting strategy performance: {e}")
        traceback.print_exc()
        
        # Return default data for all strategies with the 'total' key
        default_data = {}
        for strategy in ["hybrid", "ai", "technical", "market_regime", "trend_following", 
                       "contrarian", "volume_analysis", "mean_reversion", "volatility_breakout"]:
            default_data[strategy] = {
                'total': 0,
                'total_predictions': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'profit_loss': 0,
                'sample_size': 0
            }
        return default_data

def get_recent_signals(limit=20):
    """
    Get recent prediction signals from the database.
    
    Args:
        limit: Maximum number of signals to return
        
    Returns:
        list: Recent signals with prediction information
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query recent predictions with their signals and outcomes
        cursor.execute(f'''
            SELECT 
                epoch,
                timestamp,
                final_prediction,
                final_confidence,
                trend_following_prediction,
                trend_following_confidence,
                market_regime_prediction,
                market_regime_confidence,
                contrarian_prediction,
                contrarian_confidence,
                volume_analysis_prediction,
                volume_analysis_confidence,
                technical_confidence,
                actual_outcome,
                win
            FROM {TABLES['predictions']}
            WHERE final_prediction IS NOT NULL
            ORDER BY epoch DESC
            LIMIT ?
        ''', (limit,))
        
        # Get results and convert to dictionaries
        rows = cursor.fetchall()
        signals = [dict(row) for row in rows]
        
        conn.close()
        return signals
        
    except Exception as e:
        logger.error(f"❌ Error getting recent signals: {e}")
        traceback.print_exc()
        return [] 

def get_strategy_performance_by_regime(regime=None):
    """
    Get performance statistics for each strategy broken down by market regime.
    
    Args:
        regime: Optional specific regime to filter by
        
    Returns:
        dict: Performance metrics by strategy and regime
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Common strategies
        strategies = ["hybrid", "ai", "technical", "market_regime", "trend_following", 
                      "contrarian", "volume_analysis", "mean_reversion", "volatility_breakout"]
        
        # Common regimes
        regimes = ["trending_up", "trending_down", "ranging", "volatile", "breakout", "reversal"]
        
        performance_data = {}
        
        # If a specific regime is requested, only query for that one
        if regime:
            target_regimes = [regime]
        else:
            target_regimes = regimes
            
        # For each market regime, get performance by strategy
        for current_regime in target_regimes:
            regime_data = {}
            
            for strategy in strategies:
                # Query win/loss for this strategy in this regime
                cursor.execute(f'''
                    SELECT 
                        COUNT(*) as total_preds,
                        SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
                        SUM(profit_loss) as total_profit_loss
                    FROM {TABLES['predictions']}
                    WHERE bet_strategy = ? 
                    AND market_regime = ?
                    AND actual_outcome IS NOT NULL
                ''', (strategy, current_regime))
                
                strategy_result = cursor.fetchone()
                
                if strategy_result and strategy_result[0]:
                    total_preds = strategy_result[0] or 0
                    wins = strategy_result[1] or 0
                    losses = strategy_result[2] or 0
                    profit_loss = strategy_result[3] or 0
                    
                    # Calculate win rate
                    win_rate = wins / total_preds if total_preds > 0 else 0
                    
                    # Store in performance data
                    regime_data[strategy] = {
                        'total_predictions': total_preds,
                        'wins': wins,
                        'losses': losses,
                        'win_rate': win_rate,
                        'profit_loss': profit_loss,
                        'sample_size': total_preds
                    }
                else:
                    # Default data if no results
                    regime_data[strategy] = {
                        'total_predictions': 0,
                        'wins': 0,
                        'losses': 0,
                        'win_rate': 0,
                        'profit_loss': 0,
                        'sample_size': 0
                    }
            
            # Add this regime's data to the main results
            performance_data[current_regime] = regime_data
        
        conn.close()
        return performance_data
        
    except Exception as e:
        logger.error(f"❌ Error getting strategy performance by regime: {e}")
        traceback.print_exc()
        
        # Return default data
        if regime:
            return {regime: {s: {'total_predictions': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 
                              'profit_loss': 0, 'sample_size': 0} for s in strategies}}
        else:
            return {r: {s: {'total_predictions': 0, 'wins': 0, 'losses': 0, 'win_rate': 0, 
                           'profit_loss': 0, 'sample_size': 0} for s in strategies} for r in regimes} 

def get_prediction_history(limit=50, days=None, strategy=None):
    """
    Get historical prediction data with outcomes.
    
    Args:
        limit: Maximum number of predictions to retrieve
        days: Optional, retrieve only predictions from past X days
        strategy: Optional, filter by specific strategy
        
    Returns:
        list: List of prediction records with outcomes
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get column information to avoid querying non-existent columns
        cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
        available_columns = [row[1] for row in cursor.fetchall()]
        
        # Build query conditions
        conditions = ["final_prediction IS NOT NULL AND actual_outcome IS NOT NULL"]
        query_params = []
        
        if days is not None:
            # Calculate timestamp for X days ago
            days_ago = int(time.time()) - (days * 24 * 60 * 60)
            conditions.append("timestamp >= ?")
            query_params.append(days_ago)
            
        if strategy is not None:
            if 'bet_strategy' in available_columns:
                conditions.append("bet_strategy = ?")
                query_params.append(strategy)
            else:
                # If strategy column doesn't exist, return empty list
                logger.warning(f"Strategy column 'bet_strategy' not found in table")
                return []
            
        # Combine conditions
        where_clause = " AND ".join(conditions)
        
        # Build select list based on available columns
        select_columns = ["epoch", "timestamp", "final_prediction", "final_confidence", 
                         "actual_outcome", "win", "profit_loss"]
                         
        # Only include columns that actually exist in the table
        select_list = [col for col in select_columns if col in available_columns]
        
        # Check for additional columns that might be used
        for col in ["market_regime", "bet_strategy", "bet_amount", "bullRatio", 
                   "bearRatio", "totalAmount", "trend_following_prediction", 
                   "contrarian_prediction", "volume_analysis_prediction", 
                   "market_regime_prediction", "technical_confidence", "ai_confidence"]:
            if col in available_columns:
                select_list.append(col)
        
        # Build and execute query
        query = f'''
            SELECT {", ".join(select_list)}
            FROM {TABLES['predictions']}
            WHERE {where_clause}
            ORDER BY epoch DESC
            LIMIT ?
        '''
        
        query_params.append(limit)
        cursor.execute(query, query_params)
        
        # Convert to list of dictionaries
        rows = cursor.fetchall()
        history = [dict(row) for row in rows]
        
        conn.close()
        return history
        
    except Exception as e:
        logger.error(f"❌ Error getting prediction history: {e}")
        traceback.print_exc()
        return [] 

def get_signal_performance(days=30, min_samples=5):
    """
    Get performance metrics for different prediction signals.
    
    Args:
        days: Number of days to look back for signal performance
        min_samples: Minimum number of samples required for valid stats
        
    Returns:
        dict: Performance metrics for each signal type
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Calculate timestamp for X days ago
        days_ago = int(time.time()) - (days * 24 * 60 * 60)
        
        # Signal types to analyze
        signal_types = [
            "trend_following", "contrarian", "volume_analysis", 
            "market_regime", "mean_reversion", "volatility_breakout", 
            "technical", "ai", "hybrid", "final"
        ]
        
        performance_data = {}
        
        # For each signal type, get performance metrics
        for signal_type in signal_types:
            # Get prediction and win data
            pred_column = f"{signal_type}_prediction"
            conf_column = f"{signal_type}_confidence"
            
            # Handle special case for some signal types
            if signal_type in ["technical", "ai", "hybrid"]:
                # These don't have their own prediction column
                pred_column = "final_prediction"
                
            # Check if column exists
            cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
            columns = [col[1] for col in cursor.fetchall()]
            
            if pred_column not in columns:
                performance_data[signal_type] = {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sample_size": 0,
                    "valid": False
                }
                continue
                
            # Query for this signal type
            query = f'''
                SELECT 
                    COUNT(*) as total_preds,
                    SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END) as losses,
                    SUM(profit_loss) as total_profit,
                    SUM(CASE WHEN win = 0 THEN bet_amount ELSE 0 END) as total_loss,
                    AVG({conf_column}) as avg_confidence
                FROM {TABLES['predictions']}
                WHERE {pred_column} IS NOT NULL
                AND actual_outcome IS NOT NULL
                AND timestamp >= ?
            '''
            
            cursor.execute(query, (days_ago,))
            result = cursor.fetchone()
            
            # Calculate metrics
            if result and result[0] and result[0] >= min_samples:
                total = result[0]
                wins = result[1] or 0
                losses = result[2] or 0
                profit = result[3] or 0
                loss_amount = result[4] or 0.01
                avg_conf = result[5] or 0
                
                # Calculate metrics
                win_rate = wins / total if total > 0 else 0
                profit_factor = abs(profit / loss_amount) if loss_amount > 0 else 0
                
                performance_data[signal_type] = {
                    "win_rate": win_rate,
                    "profit_factor": profit_factor,
                    "sample_size": total,
                    "valid": total >= min_samples,
                    "avg_confidence": avg_conf,
                    "wins": wins,
                    "losses": losses
                }
            else:
                performance_data[signal_type] = {
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "sample_size": result[0] if result and result[0] else 0,
                    "valid": False,
                    "avg_confidence": 0.0
                }
                
        conn.close()
        return performance_data
        
    except Exception as e:
        logger.error(f"❌ Error getting signal performance: {e}")
        traceback.print_exc()
        
        # Return empty data
        return {signal: {
            "win_rate": 0.0,
            "profit_factor": 0.0, 
            "sample_size": 0,
            "valid": False,
            "avg_confidence": 0.0
        } for signal in [
            "trend_following", "contrarian", "volume_analysis", 
            "market_regime", "mean_reversion", "volatility_breakout", 
            "technical", "ai", "hybrid", "final"
        ]} 

def get_latest_prediction():
    """
    Get the most recent prediction from the database.
    
    Returns:
        dict: Latest prediction data or None if not found
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Query the predictions table for the most recent entry
        cursor.execute(f'''
            SELECT * FROM {TABLES['predictions']} 
            ORDER BY epoch DESC 
            LIMIT 1
        ''')
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            # Convert to dictionary
            prediction_data = dict(result)
            return prediction_data
        else:
            return None
            
    except Exception as e:
        logger.error(f"❌ Error getting latest prediction: {e}")
        traceback.print_exc()
        return None 

def get_historical_price_data(lookback=30):
    """
    Get historical price data from the database.
    
    Args:
        lookback: Number of price points to retrieve
        
    Returns:
        list: List of historical prices
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Try to get prices from market_data table first
        cursor.execute(f"""
            SELECT bnb_price 
            FROM {TABLES['market_data']} 
            ORDER BY timestamp DESC 
            LIMIT {lookback}
        """)
        
        results = cursor.fetchall()
        
        # If we don't have enough data in market_data, try the trades table
        if len(results) < lookback:
            cursor.execute(f"""
                SELECT closePrice 
                FROM {TABLES['trades']} 
                WHERE closePrice IS NOT NULL AND closePrice > 0
                ORDER BY epoch DESC 
                LIMIT {lookback}
            """)
            
            additional_results = cursor.fetchall()
            results.extend(additional_results)
            
        conn.close()
        
        # Extract prices and return as a list
        prices = [row[0] for row in results if row[0] is not None]
        
        return prices
        
    except Exception as e:
        logger.error(f"❌ Error getting historical price data: {e}")
        traceback.print_exc()
        return [] 

def get_market_balance(lookback=20):
    """
    Get market balance data (bull vs bear ratio) from recent rounds.
    
    Args:
        lookback: Number of recent rounds to analyze
        
    Returns:
        dict: Market balance statistics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Check if trades table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (TABLES['trades'],))
        if not cursor.fetchone():
            logger.warning("⚠️ Trades table does not exist")
            return {'bull_ratio': 0.5, 'bear_ratio': 0.5}
        
        # Query trades table for bull/bear amounts
        cursor.execute(f"""
            SELECT bullAmount, bearAmount, bullRatio, bearRatio
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {lookback}
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {
                'bull_ratio': 0.5,
                'bear_ratio': 0.5,
                'total_bull': 0,
                'total_bear': 0
            }
            
        # Calculate averages
        total_bull = sum(row[0] for row in results if row[0] is not None)
        total_bear = sum(row[1] for row in results if row[1] is not None)
        
        # Calculate overall ratios
        total = total_bull + total_bear
        if total > 0:
            bull_ratio = total_bull / total
            bear_ratio = total_bear / total
        else:
            bull_ratio = 0.5
            bear_ratio = 0.5
            
        # Get average ratios
        avg_bull_ratio = sum(row[2] for row in results if row[2] is not None) / len(results)
        avg_bear_ratio = sum(row[3] for row in results if row[3] is not None) / len(results)
        
        return {
            'bull_ratio': bull_ratio,
            'bear_ratio': bear_ratio,
            'avg_bull_ratio': avg_bull_ratio,
            'avg_bear_ratio': avg_bear_ratio,
            'total_bull': total_bull,
            'total_bear': total_bear
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting market balance: {e}")
        return {
            'bull_ratio': 0.5,
            'bear_ratio': 0.5,
            'avg_bull_ratio': 0.5,
            'avg_bear_ratio': 0.5,
            'total_bull': 0,
            'total_bear': 0
        } 