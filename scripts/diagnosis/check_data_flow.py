"""
Data flow monitoring service for the trading bot.
Monitors data flow from blockchain to database and logs any issues.
"""

import logging
import os
import sqlite3
import sys
import threading
import time
import traceback
from datetime import datetime

# Set up logger
logger = logging.getLogger("DataFlowMonitor")

# Import the functions we need to test
from scripts.core.constants import DB_FILE, TABLES, contract, web3
from scripts.data.blockchain import (get_current_epoch,
                                     get_enriched_round_data, get_round_data)
from scripts.data.database import (get_recent_rounds, initialize_prediction_db,
                                   record_trade)


class DataFlowMonitor:
    """Monitor data flow and database health"""

    def __init__(self, interval=300):
        """
        Initialize the monitor

        Args:
            interval: Check interval in seconds (default: 5 minutes)
        """
        self.interval = interval
        self.running = False
        self.thread = None
        self.stats = {
            "rounds_checked": 0,
            "rounds_stored": 0,
            "blockchain_errors": 0,
            "database_errors": 0,
            "last_check": None,
            "last_status": "Not started",
        }

    def start(self):
        """Start the monitoring service"""
        if self.running:
            logger.info("Monitoring already running")
            return False

        self.running = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        logger.info(
            f"📊 Data flow monitoring started (checking every {self.interval} seconds)"
        )
        return True

    def stop(self):
        """Stop the monitoring service"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        logger.info("📊 Data flow monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                self._check_data_flow()
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                traceback.print_exc()
                time.sleep(30)  # Shorter sleep on error

    def _check_data_flow(self):
        """Check data flow and log results"""
        logger.info("🔍 Running data flow check...")
        self.stats["last_check"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check blockchain connection
        blockchain_ok = self._check_blockchain()

        # Check database connection
        database_ok = self._check_database()

        # If both are OK, check data retrieval and storage
        if blockchain_ok and database_ok:
            self._check_data_storage()

        # Log overall status
        if blockchain_ok and database_ok:
            self.stats["last_status"] = "HEALTHY"
            logger.info(
                f"✅ Data flow healthy - Stored {self.stats['rounds_stored']}/{self.stats['rounds_checked']} rounds"
            )
        else:
            self.stats["last_status"] = "ISSUES DETECTED"
            logger.warning(
                f"⚠️ Data flow issues - BC: {'✅' if blockchain_ok else '❌'}, DB: {'✅' if database_ok else '❌'}"
            )

    def _check_blockchain(self):
        """Check blockchain connection"""
        try:
            # Check web3 connection
            if not web3.is_connected():
                logger.error("❌ Web3 not connected")
                self.stats["blockchain_errors"] += 1
                return False

            # Check contract access
            current_epoch = contract.functions.currentEpoch().call()
            if not current_epoch:
                logger.error("❌ Could not get current epoch")
                self.stats["blockchain_errors"] += 1
                return False

            logger.info(f"✅ Blockchain connection OK (Current epoch: {current_epoch})")
            return True

        except Exception as e:
            logger.error(f"❌ Blockchain connection error: {e}")
            traceback.print_exc()
            self.stats["blockchain_errors"] += 1
            return False

    def _check_database(self):
        """Check database connection and tables"""
        try:
            # Initialize prediction database to ensure tables exist
            initialize_prediction_db()

            # Try to connect to database
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            # Check if trades table exists
            cursor.execute(
                f"SELECT name FROM sqlite_master WHERE type='table' AND name='{TABLES['trades']}'"
            )
            if not cursor.fetchone():
                logger.error(f"❌ Table '{TABLES['trades']}' not found")
                conn.close()
                self.stats["database_errors"] += 1
                return False

            conn.close()

            # Check if we can retrieve recent rounds
            recent_rounds = get_recent_rounds(5)  # Get 5 most recent rounds
            if recent_rounds:
                logger.info(
                    f"✅ Database connection OK. Retrieved {len(recent_rounds)} recent rounds."
                )
            else:
                logger.warning("⚠️ Database connection OK but no recent rounds found.")

            return True

        except Exception as e:
            logger.error(f"❌ Database connection error: {e}")
            traceback.print_exc()
            self.stats["database_errors"] += 1
            return False

    def _check_data_storage(self):
        """Check data retrieval and storage for recent rounds"""
        try:
            # Get current epoch
            current_epoch = get_current_epoch()
            if not current_epoch:
                logger.error("❌ Could not get current epoch")
                return False

            # Check the most recent completed round (current - 2 to ensure it's closed)
            test_epoch = current_epoch - 2

            # Get round data
            logger.info(f"Checking data for epoch {test_epoch}...")
            round_data = get_round_data(test_epoch)

            if not round_data:
                logger.error(f"❌ Could not retrieve data for epoch {test_epoch}")
                self.stats["blockchain_errors"] += 1
                return False

            self.stats["rounds_checked"] += 1

            # Try to store it
            try:
                success = record_trade(test_epoch, round_data)
                if success:
                    logger.info(f"✅ Successfully recorded data for epoch {test_epoch}")
                    self.stats["rounds_stored"] += 1
                else:
                    logger.error(f"❌ Failed to record data for epoch {test_epoch}")
                    self.stats["database_errors"] += 1
            except Exception as e:
                logger.error(f"❌ Error recording trade: {e}")
                self.stats["database_errors"] += 1

            # Check if we can retrieve it back
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            cursor.execute(
                f"SELECT * FROM {TABLES['trades']} WHERE epoch = ?", (test_epoch,)
            )
            data = cursor.fetchone()
            conn.close()

            if data:
                logger.info(f"✅ Successfully retrieved data for epoch {test_epoch}")

                # Auto-collect missing data if needed (e.g., first run)
                if self.stats["rounds_checked"] == 1:
                    self._collect_missing_data(current_epoch)

                return True
            else:
                logger.error(
                    f"❌ Could not retrieve data for epoch {test_epoch} from database"
                )
                self.stats["database_errors"] += 1
                return False

        except Exception as e:
            logger.error(f"❌ Data storage check error: {e}")
            traceback.print_exc()
            return False

    def _collect_missing_data(self, current_epoch, limit=25):
        """
        Collect missing data for recent rounds

        Args:
            current_epoch: Current blockchain epoch
            limit: Maximum number of rounds to check
        """
        logger.info(f"🔄 Checking for missing round data in last {limit} epochs...")

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        collected = 0
        for epoch in range(current_epoch - limit, current_epoch):
            # Check if we already have this round
            cursor.execute(
                f"SELECT epoch FROM {TABLES['trades']} WHERE epoch = ?", (epoch,)
            )
            if cursor.fetchone():
                continue

            # Get and store missing round
            try:
                # Get basic round data first
                round_data = get_round_data(epoch)

                # Use get_enriched_round_data for detailed information
                if round_data and round_data.get("closePrice"):
                    # Only get enriched data for completed rounds
                    enriched_data = get_enriched_round_data(epoch)
                    if enriched_data:
                        # Merge enriched data into round_data
                        round_data.update(enriched_data)
                        logger.info(f"✅ Retrieved enriched data for epoch {epoch}")

                # Store the data (now with enrichments if available)
                if round_data:
                    if record_trade(epoch, round_data):
                        collected += 1
                        logger.info(f"✅ Collected missing data for epoch {epoch}")
                    else:
                        logger.error(
                            f"❌ Failed to record missing data for epoch {epoch}"
                        )
            except Exception as e:
                logger.error(f"Error collecting missing data for epoch {epoch}: {e}")

        conn.close()

        if collected > 0:
            logger.info(f"✅ Collected {collected} missing rounds")
        else:
            logger.info("✅ No missing rounds found")

    def get_status(self):
        """Get monitor status"""
        return {
            "running": self.running,
            "stats": self.stats,
            "health": "Good"
            if self.stats["last_status"] == "HEALTHY"
            else "Issues Detected",
        }


# Singleton instance
monitor = DataFlowMonitor()


def start_monitoring(interval=300):
    """Start the data flow monitoring service"""
    return monitor.start()


def stop_monitoring():
    """Stop the data flow monitoring service"""
    return monitor.stop()


def get_monitor_status():
    """Get the current monitoring status"""
    return monitor.get_status()


# For command-line usage
if __name__ == "__main__":
    from scripts.core.logging import setup_logging

    # Setup logging
    setup_logging()

    print("📊 Starting Data Flow Monitor...")
    monitor.start()

    try:
        # Run for specified time or until interrupted
        if len(sys.argv) > 1:
            duration = int(sys.argv[1])
            print(f"📊 Monitor will run for {duration} seconds")
            time.sleep(duration)
            monitor.stop()
        else:
            print("📊 Monitor running (Press Ctrl+C to stop)")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print("\n📊 Stopping monitor...")
        monitor.stop()
        print("📊 Monitor stopped")
