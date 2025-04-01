"""
Database initialization and setup.
"""

import logging
import os
import sqlite3

from scripts.core.constants import DB_FILE, TABLES

logger = logging.getLogger(__name__)


def initialize_databases():
    """Initialize all database tables if they don't exist."""
    logger.info("Initializing database tables")

    try:
        # Create data directory if it doesn't exist
        db_dir = os.path.dirname(DB_FILE)
        os.makedirs(db_dir, exist_ok=True)

        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Create trades table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLES['trades']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER,
                timestamp INTEGER,
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
        """
        )

        # Create predictions table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLES['predictions']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                epoch INTEGER,
                timestamp INTEGER,
                bullAmount REAL,
                bearAmount REAL,
                totalAmount REAL,
                bullRatio REAL,
                bearRatio REAL,
                lockPrice REAL,
                technical_prediction TEXT,
                technical_confidence REAL,
                model_prediction TEXT,
                model_confidence REAL,
                pattern_prediction TEXT,
                pattern_confidence REAL,
                market_prediction TEXT,
                market_confidence REAL,
                sentiment_prediction TEXT,
                sentiment_confidence REAL,
                final_prediction TEXT,
                confidence REAL,
                actual_outcome TEXT,
                bet_amount REAL,
                profit_loss REAL
            )
        """
        )

        # Create signal performance table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLES['signal_performance']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_type TEXT,
                timestamp INTEGER,
                correct_count INTEGER,
                total_count INTEGER,
                accuracy REAL,
                market_regime TEXT
            )
        """
        )

        # Create strategy performance table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLES['strategy_performance']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy TEXT,
                timestamp INTEGER,
                correct_count INTEGER,
                total_count INTEGER,
                accuracy REAL,
                market_regime TEXT,
                profit_loss REAL
            )
        """
        )

        # Create market data table
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLES['market_data']} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp INTEGER,
                price REAL,
                volume REAL,
                market_cap REAL,
                dominance REAL,
                regime TEXT
            )
        """
        )

        conn.commit()
        conn.close()

        logger.info("Database initialization complete")
        return True

    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        return False


def reset_database():
    """Reset database by dropping and recreating all tables."""
    try:
        logger.warning("Resetting database - all data will be lost!")

        # Connect to database
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Drop all tables
        for table in TABLES.values():
            cursor.execute(f"DROP TABLE IF EXISTS {table}")

        conn.commit()
        conn.close()

        # Reinitialize tables
        initialize_databases()

        logger.info("Database reset complete")
        return True

    except Exception as e:
        logger.error(f"Error resetting database: {e}")
        return False


# Run initialization if this script is executed directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    initialize_databases()
