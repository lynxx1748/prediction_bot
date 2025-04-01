"""
Migration script to move data from old databases to the new unified database.
This script handles the migration of data from historical_data.db and prediction_history.db
to the new trading_bot.db.
"""

import sqlite3
import logging
import traceback
from datetime import datetime

from .unified_schema import DB_FILE, TABLES, initialize_unified_db

logger = logging.getLogger(__name__)

# Old database files
OLD_DB_FILES = {
    "historical": "data/historical_data.db",
    "predictions": "data/prediction_history.db"
}

def migrate_trades(old_conn, new_conn):
    """Migrate trades data from historical_data.db to new database."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get all trades from old database
        old_cursor.execute("SELECT * FROM trades")
        trades = old_cursor.fetchall()

        # Get column names from old database
        old_cursor.execute("PRAGMA table_info(trades)")
        old_columns = [col[1] for col in old_cursor.fetchall()]

        # Get column names from new database
        new_cursor.execute(f"PRAGMA table_info({TABLES['trades']})")
        new_columns = [col[1] for col in new_cursor.fetchall()]

        # Create mapping of old columns to new columns
        column_mapping = {col: col for col in old_columns if col in new_columns}

        # Insert data into new database
        for trade in trades:
            # Create dictionary of old data
            old_data = dict(zip(old_columns, trade))
            
            # Create new data dictionary with mapped columns
            new_data = {new_col: old_data[old_col] 
                       for old_col, new_col in column_mapping.items()}

            # Format timestamp if it exists and is not None
            if 'timestamp' in new_data and new_data['timestamp'] is not None:
                new_data['datetime'] = datetime.fromtimestamp(new_data['timestamp']).isoformat()
            else:
                new_data['datetime'] = datetime.now().isoformat()

            # Build INSERT statement
            columns = list(new_data.keys())
            values = [new_data[col] for col in columns]
            placeholders = ["?" for _ in columns]
            
            sql = f"""
            INSERT INTO {TABLES['trades']} 
            ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
            """
            
            new_cursor.execute(sql, values)

        new_conn.commit()
        logger.info(f"✅ Migrated {len(trades)} trades")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating trades: {e}")
        traceback.print_exc()
        return False

def migrate_predictions(old_conn, new_conn):
    """Migrate predictions data from historical_data.db to new database."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get all predictions from old database
        old_cursor.execute("SELECT * FROM predictions")
        predictions = old_cursor.fetchall()

        # Get column names from old database
        old_cursor.execute("PRAGMA table_info(predictions)")
        old_columns = [col[1] for col in old_cursor.fetchall()]

        # Get column names from new database
        new_cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
        new_columns = [col[1] for col in new_cursor.fetchall()]

        # Create mapping of old columns to new columns
        column_mapping = {col: col for col in old_columns if col in new_columns}

        # Insert data into new database
        for pred in predictions:
            # Create dictionary of old data
            old_data = dict(zip(old_columns, pred))
            
            # Create new data dictionary with mapped columns
            new_data = {new_col: old_data[old_col] 
                       for old_col, new_col in column_mapping.items()}

            # Build INSERT statement
            columns = list(new_data.keys())
            values = [new_data[col] for col in columns]
            placeholders = ["?" for _ in columns]
            
            sql = f"""
            INSERT INTO {TABLES['predictions']} 
            ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
            """
            
            new_cursor.execute(sql, values)

        new_conn.commit()
        logger.info(f"✅ Migrated {len(predictions)} predictions")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating predictions: {e}")
        traceback.print_exc()
        return False

def migrate_market_data(old_conn, new_conn):
    """Migrate market data from historical_data.db to new database."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get all market data from old database
        old_cursor.execute("SELECT * FROM market_data")
        market_data = old_cursor.fetchall()

        # Get column names from old database
        old_cursor.execute("PRAGMA table_info(market_data)")
        old_columns = [col[1] for col in old_cursor.fetchall()]

        # Get column names from new database
        new_cursor.execute(f"PRAGMA table_info({TABLES['market_data']})")
        new_columns = [col[1] for col in new_cursor.fetchall()]

        # Create mapping of old columns to new columns
        column_mapping = {col: col for col in old_columns if col in new_columns}

        # Insert data into new database
        for data in market_data:
            # Create dictionary of old data
            old_data = dict(zip(old_columns, data))
            
            # Create new data dictionary with mapped columns
            new_data = {new_col: old_data[old_col] 
                       for old_col, new_col in column_mapping.items()}

            # Build INSERT statement
            columns = list(new_data.keys())
            values = [new_data[col] for col in columns]
            placeholders = ["?" for _ in columns]
            
            sql = f"""
            INSERT INTO {TABLES['market_data']} 
            ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
            """
            
            new_cursor.execute(sql, values)

        new_conn.commit()
        logger.info(f"✅ Migrated {len(market_data)} market data records")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating market data: {e}")
        traceback.print_exc()
        return False

def migrate_historical_prices(old_conn, new_conn):
    """Migrate historical prices from historical_data.db to new database."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get all historical prices from old database
        old_cursor.execute("SELECT * FROM historical_prices")
        prices = old_cursor.fetchall()

        # Get column names from old database
        old_cursor.execute("PRAGMA table_info(historical_prices)")
        old_columns = [col[1] for col in old_cursor.fetchall()]

        # Get column names from new database
        new_cursor.execute(f"PRAGMA table_info({TABLES['historical_prices']})")
        new_columns = [col[1] for col in new_cursor.fetchall()]

        # Create mapping of old columns to new columns
        column_mapping = {col: col for col in old_columns if col in new_columns}

        # Insert data into new database
        for price in prices:
            # Create dictionary of old data
            old_data = dict(zip(old_columns, price))
            
            # Create new data dictionary with mapped columns
            new_data = {new_col: old_data[old_col] 
                       for old_col, new_col in column_mapping.items()}

            # Build INSERT statement
            columns = list(new_data.keys())
            values = [new_data[col] for col in columns]
            placeholders = ["?" for _ in columns]
            
            sql = f"""
            INSERT INTO {TABLES['historical_prices']} 
            ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
            """
            
            new_cursor.execute(sql, values)

        new_conn.commit()
        logger.info(f"✅ Migrated {len(prices)} historical prices")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating historical prices: {e}")
        traceback.print_exc()
        return False

def migrate_strategy_performance(old_conn, new_conn):
    """Migrate strategy performance data from historical_data.db to new database."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get all strategy performance data from old database
        old_cursor.execute("SELECT * FROM strategy_regime_performance")
        performance = old_cursor.fetchall()

        # Get column names from old database
        old_cursor.execute("PRAGMA table_info(strategy_regime_performance)")
        old_columns = [col[1] for col in old_cursor.fetchall()]

        # Get column names from new database
        new_cursor.execute(f"PRAGMA table_info({TABLES['strategy_performance']})")
        new_columns = [col[1] for col in new_cursor.fetchall()]

        # Create mapping of old columns to new columns
        column_mapping = {col: col for col in old_columns if col in new_columns}

        # Insert data into new database
        for perf in performance:
            # Create dictionary of old data
            old_data = dict(zip(old_columns, perf))
            
            # Create new data dictionary with mapped columns
            new_data = {new_col: old_data[old_col] 
                       for old_col, new_col in column_mapping.items()}

            # Build INSERT statement
            columns = list(new_data.keys())
            values = [new_data[col] for col in columns]
            placeholders = ["?" for _ in columns]
            
            sql = f"""
            INSERT INTO {TABLES['strategy_performance']} 
            ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
            """
            
            new_cursor.execute(sql, values)

        new_conn.commit()
        logger.info(f"✅ Migrated {len(performance)} strategy performance records")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating strategy performance: {e}")
        traceback.print_exc()
        return False

def migrate_settings(old_conn, new_conn):
    """Migrate settings from historical_data.db to new database."""
    try:
        old_cursor = old_conn.cursor()
        new_cursor = new_conn.cursor()

        # Get all settings from old database
        old_cursor.execute("SELECT * FROM settings")
        settings = old_cursor.fetchall()

        # Get column names from old database
        old_cursor.execute("PRAGMA table_info(settings)")
        old_columns = [col[1] for col in old_cursor.fetchall()]

        # Get column names from new database
        new_cursor.execute(f"PRAGMA table_info({TABLES['settings']})")
        new_columns = [col[1] for col in new_cursor.fetchall()]

        # Create mapping of old columns to new columns
        column_mapping = {col: col for col in old_columns if col in new_columns}

        # Insert data into new database
        for setting in settings:
            # Create dictionary of old data
            old_data = dict(zip(old_columns, setting))
            
            # Create new data dictionary with mapped columns
            new_data = {new_col: old_data[old_col] 
                       for old_col, new_col in column_mapping.items()}

            # Build INSERT statement
            columns = list(new_data.keys())
            values = [new_data[col] for col in columns]
            placeholders = ["?" for _ in columns]
            
            sql = f"""
            INSERT INTO {TABLES['settings']} 
            ({', '.join(columns)}) 
            VALUES ({', '.join(placeholders)})
            """
            
            new_cursor.execute(sql, values)

        new_conn.commit()
        logger.info(f"✅ Migrated {len(settings)} settings")
        return True

    except Exception as e:
        logger.error(f"❌ Error migrating settings: {e}")
        traceback.print_exc()
        return False

def migrate_all():
    """Migrate all data from old databases to the new unified database."""
    try:
        # Initialize new database
        if not initialize_unified_db():
            logger.error("Failed to initialize new database")
            return False

        # Connect to old databases
        historical_conn = sqlite3.connect(OLD_DB_FILES["historical"])
        predictions_conn = sqlite3.connect(OLD_DB_FILES["predictions"])
        new_conn = sqlite3.connect(DB_FILE)

        # Migrate data from historical_data.db
        success = (
            migrate_trades(historical_conn, new_conn) and
            migrate_predictions(historical_conn, new_conn) and
            migrate_market_data(historical_conn, new_conn) and
            migrate_historical_prices(historical_conn, new_conn) and
            migrate_strategy_performance(historical_conn, new_conn) and
            migrate_settings(historical_conn, new_conn)
        )

        # Close connections
        historical_conn.close()
        predictions_conn.close()
        new_conn.close()

        if success:
            logger.info("✅ Successfully migrated all data to new unified database")
        else:
            logger.error("❌ Some data migration failed")

        return success

    except Exception as e:
        logger.error(f"❌ Error during migration: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run migration
    migrate_all() 