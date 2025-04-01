import logging
import os
import sqlite3
from pathlib import Path

# Import from configuration module
from configuration import config

# Setup logger
logger = logging.getLogger(__name__)

# Get database paths from configuration
DB_PATH = Path(config.get("database.file")).parent
PREDICTION_DB_FILE = DB_PATH / "prediction_history.db"
HISTORICAL_DB_FILE = DB_PATH / "historical_data.db"

# Get table names from configuration
PREDICTION_TABLE_NAME = config.get("database.tables.predictions")
HISTORICAL_TABLE_NAME = config.get("database.tables.trades")
SIGNAL_PERFORMANCE_TABLE = config.get("database.tables.signal_performance")


def create_or_update_table(db_file, table_name, schema):
    """
    Create or update a table in the specified database.

    Args:
        db_file: Path to the database file
        table_name: Name of the table
        schema: SQL schema for the table
    """
    # Ensure the database directory exists
    os.makedirs(db_file.parent, exist_ok=True)

    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(schema)

    conn.commit()
    conn.close()
    logger.info(f"Table '{table_name}' in '{db_file}' is up to date!")


def initialize_databases():
    """
    Initialize both databases with their respective tables.
    """
    # Schema for the predictions table
    prediction_schema = f"""
    CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE_NAME} (
        epoch INTEGER PRIMARY KEY,
        timestamp TEXT,
        prediction TEXT,
        actual_outcome TEXT,
        confidence REAL,
        market_conditions TEXT,
        bet_amount REAL,
        outcome_type TEXT,
        strategy_used TEXT,
        model_prediction TEXT,
        pattern_prediction TEXT,
        market_prediction TEXT,
        technical_prediction TEXT,
        sentiment_prediction TEXT,
        order_book_prediction TEXT
    )
    """

    # Schema for the historical table
    analytics_schema = f"""
    CREATE TABLE IF NOT EXISTS {HISTORICAL_TABLE_NAME} (
            epoch INTEGER PRIMARY KEY,
            datetime TEXT,
            hour INTEGER,
            minute INTEGER,
            second INTEGER,
            lockPrice REAL,
            closePrice REAL,
            bullAmount REAL,
            bearAmount REAL,
            totalAmount REAL,
            bullRatio REAL,
            bearRatio REAL,
            bnbPrice REAL,
            btcPrice REAL,
            outcome TEXT
    )
    """

    # Schema for the signal performance table
    signal_performance_schema = f"""
    CREATE TABLE IF NOT EXISTS {SIGNAL_PERFORMANCE_TABLE} (
        signal_name TEXT PRIMARY KEY,
        correct_predictions INTEGER DEFAULT 0,
        total_predictions INTEGER DEFAULT 0,
        success_rate REAL DEFAULT 0.0,
        recent_success_rate REAL DEFAULT 0.0,
        consecutive_wins INTEGER DEFAULT 0,
        consecutive_losses INTEGER DEFAULT 0,
        last_updated TEXT,
        weight REAL DEFAULT 0.0
    )
    """

    # Create/update tables in both databases
    create_or_update_table(PREDICTION_DB_FILE, PREDICTION_TABLE_NAME, prediction_schema)
    create_or_update_table(
        PREDICTION_DB_FILE, SIGNAL_PERFORMANCE_TABLE, signal_performance_schema
    )
    create_or_update_table(HISTORICAL_DB_FILE, HISTORICAL_TABLE_NAME, analytics_schema)

    logger.info("All database tables initialized successfully")


if __name__ == "__main__":
    # Setup basic logging when run directly
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    initialize_databases()
