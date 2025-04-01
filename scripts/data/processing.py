"""
Data processing functions for the trading bot.
Handles transformation, analysis, and preparation of data.
"""

import json
import logging
import os
import sqlite3
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from ..core.constants import DB_FILE, TABLES

logger = logging.getLogger(__name__)


def get_recent_price_changes(lookback=10):
    """
    Get recent price changes from the database.

    Args:
        lookback: Number of recent rounds to analyze

    Returns:
        list: List of price change percentages
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        cursor.execute(
            f"""
            SELECT lockPrice, closePrice
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {lookback}
        """
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return []

        # Calculate price changes as percentages
        changes = []
        for lock_price, close_price in results:
            # Skip entries where either price is None or zero
            if lock_price is not None and close_price is not None and lock_price > 0:
                change = ((close_price - lock_price) / lock_price) * 100
                changes.append(change)

        return changes

    except Exception as e:
        logger.error(f"❌ Error getting recent price changes: {e}")
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

        cursor.execute(
            f"""
            SELECT bullAmount, bearAmount, bullRatio, bearRatio
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {lookback}
        """
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return {
                "bull_ratio": 0.5,
                "bear_ratio": 0.5,
                "total_bull": 0,
                "total_bear": 0,
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
        avg_bull_ratio = sum(row[2] for row in results if row[2] is not None) / len(
            results
        )
        avg_bear_ratio = sum(row[3] for row in results if row[3] is not None) / len(
            results
        )

        return {
            "bull_ratio": bull_ratio,
            "bear_ratio": bear_ratio,
            "avg_bull_ratio": avg_bull_ratio,
            "avg_bear_ratio": avg_bear_ratio,
            "total_bull": total_bull,
            "total_bear": total_bear,
        }

    except Exception as e:
        logger.error(f"❌ Error getting market balance: {e}")
        return {
            "bull_ratio": 0.5,
            "bear_ratio": 0.5,
            "avg_bull_ratio": 0.5,
            "avg_bear_ratio": 0.5,
            "total_bull": 0,
            "total_bear": 0,
        }


def get_market_trend(lookback=10):
    """
    Get market trend from recent price movements.

    Args:
        lookback: Number of recent rounds to analyze

    Returns:
        tuple: (trend, strength) where trend is 'up', 'down', or 'neutral'
    """
    try:
        # Get recent price changes
        changes = get_recent_price_changes(lookback)

        if not changes:
            return "neutral", 0

        # Calculate average change
        avg_change = sum(changes) / len(changes)

        # Calculate trend consistency
        up_count = sum(1 for change in changes if change > 0)
        down_count = sum(1 for change in changes if change < 0)

        # Determine trend
        if up_count > down_count:
            trend = "up"
            strength = (
                min((up_count / len(changes)) * avg_change * 0.5, 0.95)
                if avg_change > 0
                else 0.5
            )
        elif down_count > up_count:
            trend = "down"
            strength = (
                min((down_count / len(changes)) * abs(avg_change) * 0.5, 0.95)
                if avg_change < 0
                else 0.5
            )
        else:
            trend = "neutral"
            strength = 0.5

        return trend, strength

    except Exception as e:
        logger.error(f"❌ Error determining market trend: {e}")
        return "neutral", 0


def get_prediction_statistics(lookback=50):
    """
    Get statistics about prediction accuracy.

    Args:
        lookback: Number of recent predictions to analyze

    Returns:
        dict: Prediction statistics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Get completed predictions
        cursor.execute(
            f"""
            SELECT final_prediction, actual_outcome
            FROM {TABLES['predictions']}
            WHERE final_prediction IS NOT NULL 
            AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT {lookback}
        """
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return {
                "accuracy": 0.5,
                "bull_accuracy": 0.5,
                "bear_accuracy": 0.5,
                "bull_count": 0,
                "bear_count": 0,
                "total_count": 0,
            }

        # Calculate overall accuracy
        correct = sum(1 for pred, actual in results if pred == actual)
        accuracy = correct / len(results) if results else 0.5

        # Calculate bull accuracy
        bull_preds = [row for row in results if row[0] == "BULL"]
        bull_correct = sum(1 for pred, actual in bull_preds if pred == actual)
        bull_accuracy = bull_correct / len(bull_preds) if bull_preds else 0.5

        # Calculate bear accuracy
        bear_preds = [row for row in results if row[0] == "BEAR"]
        bear_correct = sum(1 for pred, actual in bear_preds if pred == actual)
        bear_accuracy = bear_correct / len(bear_preds) if bear_preds else 0.5

        return {
            "accuracy": accuracy,
            "bull_accuracy": bull_accuracy,
            "bear_accuracy": bear_accuracy,
            "bull_count": len(bull_preds),
            "bear_count": len(bear_preds),
            "total_count": len(results),
        }

    except Exception as e:
        logger.error(f"❌ Error getting prediction statistics: {e}")
        return {
            "accuracy": 0.5,
            "bull_accuracy": 0.5,
            "bear_accuracy": 0.5,
            "bull_count": 0,
            "bear_count": 0,
            "total_count": 0,
        }


def process_round_data(round_data):
    """
    Process round data for storage and analysis.

    Args:
        round_data: Dictionary with round data

    Returns:
        dict: Processed round data
    """
    try:
        processed = {}

        # Extract and normalize basic fields
        for field in ["epoch", "startTimestamp", "lockTimestamp", "closeTimestamp"]:
            if field in round_data:
                processed[field] = round_data[field]

        # Process amounts
        bull_amount = round_data.get("bullAmount", 0)
        bear_amount = round_data.get("bearAmount", 0)

        # Convert string amounts to float if needed
        if isinstance(bull_amount, str):
            bull_amount = float(bull_amount)
        if isinstance(bear_amount, str):
            bear_amount = float(bear_amount)

        processed["bullAmount"] = bull_amount
        processed["bearAmount"] = bear_amount
        processed["totalAmount"] = bull_amount + bear_amount

        # Calculate ratios
        if processed["totalAmount"] > 0:
            processed["bullRatio"] = bull_amount / processed["totalAmount"]
            processed["bearRatio"] = bear_amount / processed["totalAmount"]
        else:
            processed["bullRatio"] = 0.5
            processed["bearRatio"] = 0.5

        # Process prices
        lock_price = round_data.get("lockPrice")
        close_price = round_data.get("closePrice")

        if isinstance(lock_price, str):
            lock_price = float(lock_price)
        if isinstance(close_price, str):
            close_price = float(close_price)

        processed["lockPrice"] = lock_price
        processed["closePrice"] = close_price

        # Determine outcome
        if close_price and lock_price:
            if close_price > lock_price:
                processed["outcome"] = "BULL"
            elif close_price < lock_price:
                processed["outcome"] = "BEAR"
            else:
                processed["outcome"] = "TIE"

        return processed

    except Exception as e:
        logger.error(f"❌ Error processing round data: {e}")
        traceback.print_exc()
        return round_data


def calculate_strategy_performance():
    """
    Calculate performance metrics for different prediction strategies.

    Returns:
        dict: Performance metrics for each strategy
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Get strategy predictions with outcomes
        cursor.execute(
            f"""
            SELECT 
                epoch,
                model_prediction, model_confidence,
                trend_following_prediction, trend_following_confidence,
                contrarian_prediction, contrarian_confidence,
                volume_analysis_prediction, volume_analysis_confidence,
                final_prediction, actual_outcome
            FROM {TABLES['predictions']}
            WHERE final_prediction IS NOT NULL 
            AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT 100
        """
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return {
                "strategies": {
                    "model": {"accuracy": 0.5, "sample_size": 0},
                    "trend_following": {"accuracy": 0.5, "sample_size": 0},
                    "contrarian": {"accuracy": 0.5, "sample_size": 0},
                    "volume_analysis": {"accuracy": 0.5, "sample_size": 0},
                },
                "overall": {"accuracy": 0.5, "sample_size": 0},
            }

        # Calculate performance for each strategy
        strategies = {
            "model": {"correct": 0, "total": 0},
            "trend_following": {"correct": 0, "total": 0},
            "contrarian": {"correct": 0, "total": 0},
            "volume_analysis": {"correct": 0, "total": 0},
            "final": {"correct": 0, "total": 0},
        }

        # Column indices for each strategy
        strategy_indices = {
            "model": (1, 2),  # prediction, confidence
            "trend_following": (3, 4),
            "contrarian": (5, 6),
            "volume_analysis": (7, 8),
            "final": (9, None),  # final prediction, no confidence
        }

        for row in results:
            actual = row[10]  # actual outcome

            # Check each strategy
            for strategy, (pred_idx, _) in strategy_indices.items():
                prediction = row[pred_idx]

                if prediction and actual:
                    strategies[strategy]["total"] += 1
                    if prediction == actual:
                        strategies[strategy]["correct"] += 1

        # Calculate accuracy for each strategy
        performance = {"strategies": {}, "overall": {}}

        for strategy, stats in strategies.items():
            if stats["total"] > 0:
                accuracy = stats["correct"] / stats["total"]
            else:
                accuracy = 0.5

            if strategy == "final":
                performance["overall"] = {
                    "accuracy": accuracy,
                    "sample_size": stats["total"],
                }
            else:
                performance["strategies"][strategy] = {
                    "accuracy": accuracy,
                    "sample_size": stats["total"],
                }

        # Save performance data to file
        performance_file = os.path.join("data", "strategy_performance.json")
        os.makedirs(os.path.dirname(performance_file), exist_ok=True)

        with open(performance_file, "w") as f:
            json.dump(performance, f, indent=2)

        return performance

    except Exception as e:
        logger.error(f"❌ Error calculating strategy performance: {e}")
        traceback.print_exc()
        return {
            "strategies": {
                "model": {"accuracy": 0.5, "sample_size": 0},
                "trend_following": {"accuracy": 0.5, "sample_size": 0},
                "contrarian": {"accuracy": 0.5, "sample_size": 0},
                "volume_analysis": {"accuracy": 0.5, "sample_size": 0},
            },
            "overall": {"accuracy": 0.5, "sample_size": 0},
        }


def get_winning_streak():
    """
    Get current winning/losing streak from recent predictions.

    Returns:
        tuple: (streak_count, is_winning)
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Get recent predictions with outcomes
        cursor.execute(
            f"""
            SELECT final_prediction, actual_outcome
            FROM {TABLES['predictions']}
            WHERE final_prediction IS NOT NULL 
            AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT 20
        """
        )

        results = cursor.fetchall()
        conn.close()

        if not results:
            return 0, False

        # Calculate streak
        streak = 0
        is_winning = (
            results[0][0] == results[0][1]
        )  # Win or loss for most recent prediction

        for prediction, actual in results:
            if (prediction == actual) == is_winning:
                streak += 1
            else:
                break

        return streak, is_winning

    except Exception as e:
        logger.error(f"❌ Error calculating winning streak: {e}")
        return 0, False


def get_market_prediction_performance(lookback=50):
    """
    Get market prediction performance metrics, safely handling missing columns.

    Args:
        lookback: Number of recent predictions to analyze

    Returns:
        dict: Performance metrics
    """
    try:
        # Handle lookback if it's a dictionary
        if isinstance(lookback, dict):
            logger.warning(
                "⚠️ Warning: lookback parameter is a dictionary, using default value"
            )
            lookback = 50

        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check what columns exist in the predictions table
        cursor.execute(f"PRAGMA table_info({TABLES['predictions']})")
        existing_columns = [row[1] for row in cursor.fetchall()]

        # Try different prediction column names in order of preference
        prediction_columns = [
            "market_prediction",  # Try first
            "trend_prediction",  # Closest equivalent
            "final_prediction",  # Fall back to final prediction
            "ai_prediction",  # Last resort
        ]

        # Find the first column that exists
        prediction_column = None
        for col in prediction_columns:
            if col in existing_columns:
                prediction_column = col
                break

        if not prediction_column:
            logger.warning(
                "⚠️ No suitable column found for market prediction performance calculation"
            )
            return {
                "accuracy": 0.5,
                "sample_size": 0,
                "total": 0,
                "wins": 0,
                "losses": 0,
            }

        # Calculate performance metrics using the available column
        cursor.execute(
            f"""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN {prediction_column} = actual_outcome THEN 1 ELSE 0 END) as wins
            FROM {TABLES['predictions']}
            WHERE {prediction_column} IS NOT NULL
            AND actual_outcome IS NOT NULL
            ORDER BY epoch DESC
            LIMIT {lookback}
        """
        )

        row = cursor.fetchone()
        conn.close()

        if row and row[0] > 0:
            total, wins = row
            return {
                "accuracy": wins / total,
                "sample_size": total,
                "total": total,  # Include both keys for compatibility
                "wins": wins,
                "losses": total - wins,
            }
        else:
            return {
                "accuracy": 0.5,
                "sample_size": 0,
                "total": 0,
                "wins": 0,
                "losses": 0,
            }

    except Exception as e:
        logger.error(f"❌ Error getting market prediction performance: {e}")
        traceback.print_exc()
        return {"accuracy": 0.5, "sample_size": 0, "total": 0, "wins": 0, "losses": 0}


def get_market_balance_stats(lookback=100):
    """
    Get statistics about market balance between bull and bear outcomes.

    Args:
        lookback: Number of recent rounds to analyze

    Returns:
        dict: Market balance statistics
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (TABLES["trades"],),
        )
        if not cursor.fetchone():
            logger.warning("⚠️ Trades table does not exist")
            return {"bull_ratio": 0.5, "sample_size": 0}

        # Query actual market outcomes
        cursor.execute(
            f"""
            SELECT outcome, COUNT(*) as count
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT {lookback}
        """
        )

        results = cursor.fetchall()
        conn.close()

        # Calculate bull/bear ratios
        total = sum(r[1] for r in results)
        bull_count = sum(r[1] for r in results if r[0] == "BULL")
        bear_count = sum(r[1] for r in results if r[0] == "BEAR")

        if total > 0:
            bull_ratio = bull_count / total
            bear_ratio = bear_count / total

            logger.info(
                f"🔍 Market Balance: {bull_ratio:.2f} BULL / {bear_ratio:.2f} BEAR over {total} rounds"
            )

            # Print alert if balance is skewed
            if bull_ratio > 0.65:
                logger.warning(
                    f"⚠️ STRONG BULL MARKET DETECTED: {bull_ratio:.2f} of recent outcomes are BULL"
                )
            elif bear_ratio > 0.65:
                logger.warning(
                    f"⚠️ STRONG BEAR MARKET DETECTED: {bear_ratio:.2f} of recent outcomes are BEAR"
                )

            return {
                "bull_ratio": bull_ratio,
                "bear_ratio": bear_ratio,
                "sample_size": total,
                "bull_count": bull_count,
                "bear_count": bear_count,
            }
        else:
            return {"bull_ratio": 0.5, "sample_size": 0}

    except Exception as e:
        logger.error(f"❌ Error getting market balance stats: {e}")
        traceback.print_exc()
        return {"bull_ratio": 0.5, "sample_size": 0}


def get_recent_performance(days=3, min_samples=5):
    """
    Get recent performance metrics for the trading bot.

    Args:
        days: Number of days to look back
        min_samples: Minimum number of samples required for valid stats

    Returns:
        dict: Recent performance metrics
    """
    try:
        from .database import get_prediction_history

        # Get recent prediction history
        recent_history = get_prediction_history(limit=100, days=days)

        if not recent_history or len(recent_history) < min_samples:
            logger.warning(
                f"Insufficient data for recent performance analysis ({len(recent_history) if recent_history else 0} samples)"
            )
            return {
                "win_rate": 0.0,
                "bull_win_rate": 0.0,
                "bear_win_rate": 0.0,
                "avg_profit": 0.0,
                "total_profit": 0.0,
                "sample_size": len(recent_history) if recent_history else 0,
                "valid": False,
            }

        # Calculate metrics
        total_bets = len(recent_history)
        wins = sum(1 for pred in recent_history if pred.get("win") == 1)

        # Bull/Bear specific metrics
        bull_bets = [
            pred for pred in recent_history if pred.get("final_prediction") == "BULL"
        ]
        bull_wins = sum(1 for pred in bull_bets if pred.get("win") == 1)

        bear_bets = [
            pred for pred in recent_history if pred.get("final_prediction") == "BEAR"
        ]
        bear_wins = sum(1 for pred in bear_bets if pred.get("win") == 1)

        # Profit calculations
        profits = [pred.get("profit_loss", 0) for pred in recent_history]
        total_profit = sum(profits)
        avg_profit = total_profit / total_bets if total_bets > 0 else 0

        # Win rates
        win_rate = wins / total_bets if total_bets > 0 else 0
        bull_win_rate = bull_wins / len(bull_bets) if bull_bets else 0
        bear_win_rate = bear_wins / len(bear_bets) if bear_bets else 0

        return {
            "win_rate": win_rate,
            "bull_win_rate": bull_win_rate,
            "bear_win_rate": bear_win_rate,
            "avg_profit": avg_profit,
            "total_profit": total_profit,
            "sample_size": total_bets,
            "valid": total_bets >= min_samples,
            "recent_outcomes": [
                pred.get("actual_outcome") for pred in recent_history[:10]
            ],
        }

    except Exception as e:
        logger.error(f"❌ Error calculating recent performance: {e}")
        traceback.print_exc()
        return {
            "win_rate": 0.0,
            "bull_win_rate": 0.0,
            "bear_win_rate": 0.0,
            "avg_profit": 0.0,
            "total_profit": 0.0,
            "sample_size": 0,
            "valid": False,
        }


def get_overall_performance():
    """
    Get overall performance metrics for the trading bot across all history.

    Returns:
        dict: Comprehensive performance metrics
    """
    try:
        from .database import get_performance_stats, get_strategy_performance

        # Get basic performance stats from database
        perf_stats = get_performance_stats()
        strategy_perf = get_strategy_performance()

        # Extract key metrics
        total_bets = perf_stats.get("total_bets", 0)
        wins = perf_stats.get("wins", 0)
        losses = perf_stats.get("losses", 0)
        win_rate = perf_stats.get("win_rate", 0.0)
        profit_loss = perf_stats.get("profit_loss", 0.0)

        # Calculate additional metrics
        avg_profit = profit_loss / total_bets if total_bets > 0 else 0.0

        # Get best performing strategy
        best_strategy = None
        best_win_rate = 0.0

        for strategy, metrics in strategy_perf.items():
            if (
                metrics.get("win_rate", 0) > best_win_rate
                and metrics.get("sample_size", 0) >= 10
            ):
                best_win_rate = metrics.get("win_rate", 0)
                best_strategy = strategy

        return {
            "total_bets": total_bets,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "profit_loss": profit_loss,
            "avg_profit": avg_profit,
            "roi": (profit_loss / total_bets) * 100 if total_bets > 0 else 0.0,
            "best_strategy": best_strategy,
            "best_strategy_win_rate": best_win_rate,
            "consecutive_losses": perf_stats.get("consecutive_losses", 0),
            "valid": total_bets >= 10,
        }

    except Exception as e:
        logger.error(f"❌ Error calculating overall performance: {e}")
        traceback.print_exc()
        return {
            "total_bets": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "profit_loss": 0.0,
            "avg_profit": 0.0,
            "roi": 0.0,
            "best_strategy": None,
            "best_strategy_win_rate": 0.0,
            "consecutive_losses": 0,
            "valid": False,
        }


def get_recent_trades(limit=20):
    """Get recent trade data from the database."""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Query recent trades with safe SQL that handles NULL values
        cursor.execute(
            f"""
            SELECT 
                epoch, 
                datetime as timestamp,
                COALESCE(lockPrice, 0) as lockPrice, 
                COALESCE(closePrice, 0) as closePrice, 
                COALESCE(outcome, '') as outcome, 
                COALESCE(prediction, '') as prediction, 
                COALESCE(amount, 0) as amount, 
                COALESCE(profit_loss, 0) as profit_loss, 
                COALESCE(win, 0) as win,
                COALESCE(bullRatio, 0) as bullRatio,
                COALESCE(bearRatio, 0) as bearRatio,
                COALESCE(totalAmount, 0) as totalAmount
            FROM {TABLES['trades']}
            ORDER BY epoch DESC
            LIMIT ?
        """,
            (limit,),
        )

        # Convert to list of dictionaries with safer conversion
        rows = cursor.fetchall()
        trades = []

        for row in rows:
            trade_dict = dict(row)
            # Ensure numeric values are properly converted
            for key in [
                "lockPrice",
                "closePrice",
                "amount",
                "profit_loss",
                "bullRatio",
                "bearRatio",
                "totalAmount",
            ]:
                if key in trade_dict:
                    try:
                        trade_dict[key] = float(trade_dict[key] or 0)
                    except (TypeError, ValueError):
                        trade_dict[key] = 0.0

            # Ensure win is an integer
            if "win" in trade_dict:
                try:
                    trade_dict["win"] = int(trade_dict["win"] or 0)
                except (TypeError, ValueError):
                    trade_dict["win"] = 0

            trades.append(trade_dict)

        conn.close()
        print(f"Database returned {len(trades)} recent trades")

        # Calculate price change safely
        for trade in trades:
            lock_price = trade.get("lockPrice", 0)
            close_price = trade.get("closePrice", 0)

            # Calculate price change only if both values exist and lock_price is non-zero
            if lock_price and close_price and lock_price != 0:
                price_change = ((close_price - lock_price) / lock_price) * 100
                trade["price_change_pct"] = price_change
            else:
                trade["price_change_pct"] = 0

            # Format timestamp if needed
            if "timestamp" in trade and not isinstance(trade["timestamp"], str):
                try:
                    trade["datetime"] = datetime.fromtimestamp(
                        trade["timestamp"]
                    ).strftime("%Y-%m-%d %H:%M:%S")
                except (TypeError, ValueError):
                    trade["datetime"] = "Unknown"

        return trades

    except Exception as e:
        logger.error(f"❌ Error getting recent trades: {e}")
        traceback.print_exc()
        # Return an empty list on error rather than None
        return []


def get_time_range(days_back=30, hours_back=0, minutes_back=0):
    """
    Get a time range from now going back by the specified duration.
    Explicitly demonstrates timedelta usage.

    Args:
        days_back: Days to look back
        hours_back: Hours to look back
        minutes_back: Minutes to look back

    Returns:
        tuple: (start_time, end_time) as datetime objects
    """
    end_time = datetime.now()

    # Create a timedelta explicitly
    delta = timedelta(days=days_back, hours=hours_back, minutes=minutes_back)

    # Use the timedelta to calculate start time
    start_time = end_time - delta

    return start_time, end_time


def calculate_trade_statistics(trades):
    """
    Calculate statistical metrics for trades using numpy.

    Args:
        trades: List of trade dictionaries

    Returns:
        dict: Statistical metrics
    """
    try:
        if not trades or len(trades) < 2:
            return {"count": 0}

        # Extract price changes and convert to numpy array
        price_changes = [trade.get("price_change_pct", 0) for trade in trades]
        price_array = np.array(price_changes)

        # Calculate statistics using numpy
        stats = {
            "count": len(trades),
            "mean_change": float(np.mean(price_array)),
            "median_change": float(np.median(price_array)),
            "std_dev": float(np.std(price_array)),
            "min_change": float(np.min(price_array)),
            "max_change": float(np.max(price_array)),
            "volatility": float(np.std(price_array)),
            "positive_count": int(np.sum(price_array > 0)),
            "negative_count": int(np.sum(price_array < 0)),
        }

        return stats

    except Exception as e:
        logger.error(f"Error calculating trade statistics: {e}")
        return {"count": 0, "error": str(e)}


def get_trades_dataframe(trades):
    """
    Convert trades list to pandas DataFrame for analysis.

    Args:
        trades: List of trade dictionaries

    Returns:
        pd.DataFrame: DataFrame containing trade data
    """
    try:
        if not trades:
            return pd.DataFrame()

        # Create DataFrame from trades
        df = pd.DataFrame(trades)

        # Add additional calculated columns
        if "timestamp" in df.columns:
            df["date"] = pd.to_datetime(df["timestamp"], unit="s")
            df["hour"] = df["date"].dt.hour
            df["day_of_week"] = df["date"].dt.dayofweek

        # Add win rate statistics
        if "win" in df.columns:
            df["win_numeric"] = df["win"].astype(int)

        # Add rolling metrics when enough data is available
        if len(df) >= 5 and "price_change_pct" in df.columns:
            df["rolling_volatility"] = df["price_change_pct"].rolling(5).std()

        logger.info(
            f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns"
        )
        return df

    except Exception as e:
        logger.error(f"Error creating trades DataFrame: {e}")
        traceback.print_exc()
        return pd.DataFrame()
