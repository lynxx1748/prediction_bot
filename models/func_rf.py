import logging
import sqlite3
import traceback

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Import from configuration module
from configuration import config

# Setup logger
logger = logging.getLogger(__name__)


def train_model(db_file=None, trades_table=None, model_file=None, scaler_file=None):
    """
    Train the Random Forest model.

    Args:
        db_file: Optional path to database file (overrides config)
        trades_table: Optional table name (overrides config)
        model_file: Optional path to model file (overrides config)
        scaler_file: Optional path to scaler file (overrides config)

    Returns:
        tuple: (model, scaler) or (None, None) on failure
    """
    try:
        # Use provided parameters or get from config
        db_file = db_file or config.get("database.file")
        trades_table = trades_table or config.get("database.tables.trades")
        model_file = model_file or config.get("paths.model_file")
        scaler_file = scaler_file or config.get("paths.scaler_file")

        # Load historical data
        conn = sqlite3.connect(db_file)
        query = f"""
        SELECT * FROM {trades_table}
        WHERE closePrice IS NOT NULL
        ORDER BY epoch DESC
        LIMIT 10000
        """
        df = pd.read_sql(query, conn)
        conn.close()

        if df.empty:
            logger.error("No training data available")
            return None, None

        # Define required features and check availability
        required_features = ["bullRatio", "bearRatio", "bnb_change", "btc_change"]
        available_features = [
            feature for feature in required_features if feature in df.columns
        ]

        if len(available_features) < 2:
            logger.error(f"Not enough features available. Found: {available_features}")
            return None, None

        # Add missing columns with default values if needed
        for feature in required_features:
            if feature not in df.columns:
                logger.warning(
                    f"Feature '{feature}' not found, adding with default value 0"
                )
                df[feature] = 0

        # Prepare features and target
        X = df[required_features]
        y = (df["closePrice"] > df["lockPrice"]).astype(int)

        # Train model
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

        model.fit(X, y)

        # Save model and scaler
        joblib.dump(model, model_file)

        logger.info(f"\n📊 Loaded {len(df):,} training samples from historical data")
        logger.info("\n📈 Feature Importance:")
        for feat, imp in zip(X.columns, model.feature_importances_):
            logger.info(f"   {feat}: {imp:.3f}")
        logger.info(f"\n🎯 Training Accuracy: {model.score(X, y):.2%}\n")

        return model, None

    except Exception as e:
        logger.error(f"❌ Error training model: {e}")
        traceback.print_exc()
        return None, None
