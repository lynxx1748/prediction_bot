"""
Unified database schema for the trading bot.
This module defines the schema for all tables in the unified database.
"""

import sqlite3
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Database configuration
DB_FILE = "data/trading_bot.db"

# Table names
TABLES = {
    "trades": "trades",
    "predictions": "predictions",
    "market_data": "market_data",
    "historical_prices": "historical_prices",
    "strategy_performance": "strategy_performance",
    "settings": "settings",
    "blockchain_events": "blockchain_events",
    "mid_round_swings": "mid_round_swings"
}

def initialize_unified_db():
    """Initialize the unified SQLite database with all necessary tables."""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Record initialization time
        init_time = datetime.now().isoformat()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS db_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)
        cursor.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)",
            ("initialized_at", init_time)
        )

        # Create trades table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['trades']} (
            epoch INTEGER PRIMARY KEY,
            timestamp INTEGER,
            datetime TEXT,
            startTime INTEGER,
            lockTime INTEGER,
            closeTime INTEGER,
            
            -- Market data
            lockPrice REAL,
            closePrice REAL,
            bullAmount REAL,
            bearAmount REAL,
            totalAmount REAL,
            bullRatio REAL,
            bearRatio REAL,
            bnbPrice REAL,
            btcPrice REAL,
            
            -- Prediction data
            final_prediction TEXT,
            model_prediction TEXT,
            strategy_prediction TEXT,
            prediction_confidence REAL,
            
            -- Outcome data
            outcome TEXT,
            profit_loss REAL,
            win INTEGER DEFAULT 0,
            
            -- Additional info
            bet_amount REAL DEFAULT 0.0,
            bet_strategy TEXT,
            mode TEXT DEFAULT 'test'
        )
        """)

        # Create predictions table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['predictions']} (
            epoch INTEGER PRIMARY KEY,
            timestamp INTEGER,
            
            -- Individual strategy predictions
            ai_prediction TEXT,
            ai_confidence REAL,
            technical_prediction TEXT,
            technical_confidence REAL,
            hybrid_prediction TEXT,
            hybrid_confidence REAL,
            
            -- Component predictions
            trend_prediction TEXT,
            trend_confidence REAL,
            volume_prediction TEXT,
            volume_confidence REAL,
            pattern_prediction TEXT,
            pattern_confidence REAL,
            market_prediction TEXT,
            market_confidence REAL,
            
            -- Strategy weights
            model_weight REAL DEFAULT 0.0,
            trend_following_weight REAL DEFAULT 0.0,
            contrarian_weight REAL DEFAULT 0.0,
            volume_analysis_weight REAL DEFAULT 0.0,
            market_indicators_weight REAL DEFAULT 0.0,
            market_regime_weight REAL DEFAULT 0.0,
            
            -- Market regime
            market_regime TEXT,
            market_regime_confidence REAL,
            
            -- Final prediction
            final_prediction TEXT,
            final_confidence REAL,
            
            -- Outcome
            actual_outcome TEXT,
            win INTEGER DEFAULT 0
        )
        """)

        # Create market_data table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['market_data']} (
            timestamp INTEGER PRIMARY KEY,
            price REAL,
            volume REAL,
            change_24h REAL,
            high_24h REAL,
            low_24h REAL,
            market_cap REAL,
            
            -- BNB specific data
            bnb_price REAL,
            bnb_24h_volume REAL,
            bnb_24h_change REAL,
            bnb_high REAL,
            bnb_low REAL,
            
            -- BTC specific data
            btc_price REAL,
            btc_24h_volume REAL,
            btc_24h_change REAL,
            
            -- Market sentiment
            fear_greed_index INTEGER,
            fear_greed_value REAL,
            fear_greed_class TEXT,
            
            -- Current round data
            total_rounds REAL,
            current_epoch REAL,
            totalAmount REAL,
            bullAmount REAL,
            bearAmount REAL,
            bullRatio REAL,
            bearRatio REAL
        )
        """)

        # Create historical_prices table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['historical_prices']} (
            timestamp INTEGER PRIMARY KEY,
            price REAL,
            volume REAL,
            high REAL,
            low REAL,
            open REAL,
            close REAL,
            epoch INTEGER,
            lockPrice REAL,
            closePrice REAL
        )
        """)

        # Create strategy_performance table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['strategy_performance']} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy TEXT,
            regime TEXT,
            prediction TEXT,
            actual_outcome TEXT,
            win INTEGER,
            epoch INTEGER,
            timestamp INTEGER
        )
        """)

        # Create settings table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['settings']} (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """)

        # Create blockchain events table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['blockchain_events']} (
            event_hash TEXT PRIMARY KEY,
            epoch INTEGER,
            event_type TEXT,
            amount REAL,
            timestamp INTEGER,
            sender TEXT,
            confirmed INTEGER DEFAULT 0
        )
        """)

        # Create mid-round swings table
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLES['mid_round_swings']} (
            epoch INTEGER PRIMARY KEY,
            timestamp INTEGER,
            initial_prediction TEXT,
            swing_direction TEXT,
            magnitude REAL,
            elapsed_seconds INTEGER
        )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Successfully initialized unified database schema")
        return True

    except Exception as e:
        logger.error(f"❌ Error initializing unified database: {e}")
        return False 