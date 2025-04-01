"""
Prediction handler for the trading bot.
Coordinates different prediction strategies and produces final decisions.
"""

import asyncio
import logging
import time
import traceback
from datetime import datetime

from ..analysis.market import get_historical_prices
from ..core.constants import MARKET_BIAS, STRATEGY_WEIGHTS
from ..core.logging import log_prediction_details
from ..data.database import get_adaptive_weights
from ..utils.async_utils import run_prediction_with_timeout

logger = logging.getLogger(__name__)


class PredictionHandler:
    """Handles prediction generation and aggregation."""

    def __init__(self, config=None):
        """
        Initialize PredictionHandler.

        Args:
            config: Optional configuration override
        """
        self.config = config or {}
        self.recent_predictions = {}

        # Market regime tracking
        self.current_regime = "unknown"
        self.regime_confidence = 0
        self.last_regime_update = 0

        # Strategy instances
        self.strategies = {}

        # Get initial strategy weights
        self.strategy_weights = get_adaptive_weights()

        logger.info("🧠 Initialized prediction handler")

    async def get_prediction(self, round_data, strategy_preference=None):
        """
        Get prediction for current round.

        Args:
            round_data: Dictionary with current round data
            strategy_preference: Optional strategy to prioritize

        Returns:
            tuple: (final_prediction, confidence, predictions_dict)
        """
        try:
            # Update market regime if needed
            current_time = time.time()
            if current_time - self.last_regime_update > 600:  # 10 minutes
                await self._update_market_regime()
                self.last_regime_update = current_time

            # Gather predictions from all strategies
            predictions = {}
            confidences = {}

            # Run predictions concurrently
            tasks = []
            for strategy_name, strategy_instance in self.strategies.items():
                if hasattr(strategy_instance, "predict") and callable(
                    strategy_instance.predict
                ):
                    tasks.append(
                        self._get_strategy_prediction(
                            strategy_name, strategy_instance, round_data
                        )
                    )

            results = await asyncio.gather(*tasks)

            # Process results
            for strategy_name, prediction, confidence in results:
                if prediction:
                    predictions[strategy_name] = prediction.upper()
                    confidences[strategy_name] = confidence

            # Get weights for each strategy
            weights = {}
            for strategy_name in predictions:
                # Prioritize the preferred strategy if specified
                if strategy_preference and strategy_name == strategy_preference:
                    weights[strategy_name] = (
                        self.strategy_weights.get(strategy_name, 0.1) * 1.5
                    )  # 50% boost
                else:
                    weights[strategy_name] = self.strategy_weights.get(
                        strategy_name, 0.1
                    )

            # If no predictions, return default
            if not predictions:
                logger.warning("⚠️ No predictions generated, using default")
                return "BULL", 0.51, {}

            # Apply market bias if enabled
            predictions, confidences, weights = self._apply_market_bias(
                predictions, confidences, weights
            )

            # Calculate weighted prediction
            bull_weight = 0
            bear_weight = 0
            total_weight = 0

            for strategy_name, prediction in predictions.items():
                weight = weights[strategy_name] * confidences[strategy_name]
                total_weight += weight

                if prediction == "BULL":
                    bull_weight += weight
                elif prediction == "BEAR":
                    bear_weight += weight

            if total_weight == 0:
                return "BULL", 0.51, {}

            # Determine final prediction
            final_prediction = "BULL" if bull_weight >= bear_weight else "BEAR"
            final_confidence = max(bull_weight, bear_weight) / total_weight

            # Create prediction data dictionary
            prediction_data = {
                "final": {
                    "prediction": final_prediction,
                    "confidence": final_confidence,
                }
            }

            for strategy_name in predictions:
                prediction_data[strategy_name] = {
                    "prediction": predictions[strategy_name],
                    "confidence": confidences[strategy_name],
                    "weight": weights[strategy_name],
                }

            # Store in recent predictions
            epoch = round_data.get("epoch", 0)
            self.recent_predictions[epoch] = {
                "timestamp": datetime.now().isoformat(),
                "data": prediction_data,
            }

            # Log prediction summary
            logger.info(
                f"📊 Final prediction: {final_prediction} with {final_confidence:.2f} confidence"
            )
            for strategy_name, prediction in predictions.items():
                logger.info(
                    f"  - {strategy_name}: {prediction} ({confidences[strategy_name]:.2f}, weight: {weights[strategy_name]:.2f})"
                )

            return final_prediction, final_confidence, prediction_data

        except Exception as e:
            logger.error(f"❌ Error getting prediction: {e}")
            traceback.print_exc()
            return "BULL", 0.51, {}

    def _apply_market_bias(self, predictions, confidences, weights):
        """
        Apply market bias to predictions if enabled.

        Args:
            predictions: Dictionary of strategy predictions
            confidences: Dictionary of confidence levels
            weights: Dictionary of strategy weights

        Returns:
            tuple: (updated_predictions, updated_confidences, updated_weights)
        """
        # Check if market bias is enabled
        if not MARKET_BIAS.get("enabled", False):
            return predictions, confidences, weights

        # Get bias parameters
        bias_direction = MARKET_BIAS.get("bias_direction", "BULL")
        bias_strength = MARKET_BIAS.get("bias_strength", 0.15)
        min_confidence = MARKET_BIAS.get("min_confidence", 0.45)

        # Add bias as a virtual strategy
        avg_confidence = (
            sum(confidences.values()) / len(confidences) if confidences else 0.5
        )

        if avg_confidence >= min_confidence:
            # Add market bias to prediction mix
            predictions["market_bias"] = bias_direction
            confidences["market_bias"] = 0.7  # Moderate confidence for bias
            weights["market_bias"] = bias_strength

            # Normalize other weights
            total_other_weight = sum(
                w for k, w in weights.items() if k != "market_bias"
            )
            factor = (1 - bias_strength) / total_other_weight

            for k in list(weights.keys()):
                if k != "market_bias":
                    weights[k] *= factor

            logger.info(
                f"🧲 Applied {bias_direction} market bias (strength: {bias_strength:.2f})"
            )

        return predictions, confidences, weights

    async def _get_strategy_prediction(self, strategy_name, strategy, round_data):
        """
        Get prediction from a specific strategy with timeout handling.

        Args:
            strategy_name: Name of the strategy
            strategy: Strategy instance
            round_data: Round data

        Returns:
            tuple: (strategy_name, prediction, confidence)
        """
        try:
            # Check if strategy has async predict method
            if hasattr(strategy, "predict_async") and callable(strategy.predict_async):
                prediction, confidence = await run_prediction_with_timeout(
                    strategy.predict_async, round_data, timeout=2.0
                )
            else:
                # Fallback to synchronous prediction
                prediction, confidence = strategy.predict(round_data)

            return strategy_name, prediction, confidence

        except Exception as e:
            logger.error(f"❌ Error getting prediction from {strategy_name}: {e}")
            return strategy_name, None, 0

    async def _update_market_regime(self):
        """Update market regime detection."""
        try:
            from ..analysis.market import get_historical_prices
            from ..analysis.regime import detect_market_regime

            prices = await asyncio.to_thread(get_historical_prices, 30)

            if prices and len(prices) > 15:
                regime_data = await asyncio.to_thread(detect_market_regime, prices)

                self.current_regime = regime_data["regime"]
                self.regime_confidence = regime_data["confidence"]

                logger.info(
                    f"🔍 Updated Market Regime: {self.current_regime.upper()} (confidence: {self.regime_confidence:.2f})"
                )

        except Exception as e:
            logger.error(f"❌ Error updating market regime: {e}")

    def get_all_predictions(self):
        """
        Get all predictions from different strategies.

        Returns:
            dict: Dictionary of predictions by strategy type
        """
        # Create a dictionary with all stored predictions
        all_predictions = {}

        # Add model prediction if available
        if hasattr(self, "model_prediction") and self.model_prediction:
            all_predictions["model"] = {
                "prediction": self.model_prediction,
                "confidence": getattr(self, "model_confidence", 0.5),
            }

        # Add trend following prediction
        if (
            hasattr(self, "trend_following_prediction")
            and self.trend_following_prediction
        ):
            all_predictions["trend_following"] = {
                "prediction": self.trend_following_prediction,
                "confidence": getattr(self, "trend_following_confidence", 0.5),
            }

        # Add contrarian prediction
        if hasattr(self, "contrarian_prediction") and self.contrarian_prediction:
            all_predictions["contrarian"] = {
                "prediction": self.contrarian_prediction,
                "confidence": getattr(self, "contrarian_confidence", 0.5),
            }

        # Add volume analysis prediction
        if (
            hasattr(self, "volume_analysis_prediction")
            and self.volume_analysis_prediction
        ):
            all_predictions["volume_analysis"] = {
                "prediction": self.volume_analysis_prediction,
                "confidence": getattr(self, "volume_analysis_confidence", 0.5),
            }

        # Add technical prediction
        if hasattr(self, "technical_prediction") and self.technical_prediction:
            all_predictions["technical"] = {
                "prediction": self.technical_prediction,
                "confidence": getattr(self, "technical_confidence", 0.5),
            }

        # Add AI prediction
        if hasattr(self, "ai_prediction") and self.ai_prediction:
            all_predictions["ai"] = {
                "prediction": self.ai_prediction,
                "confidence": getattr(self, "ai_confidence", 0.5),
            }

        # Add hybrid prediction
        if hasattr(self, "hybrid_prediction") and self.hybrid_prediction:
            all_predictions["hybrid"] = {
                "prediction": self.hybrid_prediction,
                "confidence": getattr(self, "hybrid_confidence", 0.5),
            }

        # Add final prediction
        if hasattr(self, "final_prediction") and self.final_prediction:
            all_predictions["final"] = {
                "prediction": self.final_prediction,
                "confidence": getattr(self, "final_confidence", 0.5),
            }

        return all_predictions

    def add_prediction(self, strategy_type, prediction, confidence):
        """
        Add a prediction from a specific strategy.

        Args:
            strategy_type: Type of strategy (e.g., 'ai', 'technical', 'hybrid')
            prediction: The prediction value ('BULL' or 'BEAR')
            confidence: Confidence level (0.0 to 1.0)

        Returns:
            bool: Success status
        """
        try:
            # Set attributes on the handler object for the specified strategy
            setattr(self, f"{strategy_type}_prediction", prediction)
            setattr(self, f"{strategy_type}_confidence", confidence)

            logger.info(
                f"✅ Added {strategy_type} prediction: {prediction} ({confidence:.2f} confidence)"
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error adding {strategy_type} prediction: {e}")
            return False

    def get_strategy_weights(self):
        """
        Get current strategy weights based on configuration.

        Returns:
            dict: Dictionary of weights for each strategy
        """
        try:
            # First check if we have adaptive weights
            if hasattr(self, "adaptive_weights") and self.adaptive_weights:
                logger.info("Using adaptive strategy weights")
                return self.adaptive_weights

            # Use the imported STRATEGY_WEIGHTS constant as base weights
            base_weights = STRATEGY_WEIGHTS.copy()

            # Apply any runtime adjustments based on market conditions
            if (
                hasattr(self, "market_condition")
                and self.market_condition == "volatile"
            ):
                # Increase weight for contrarian strategy in volatile markets
                base_weights["contrarian"] = base_weights.get("contrarian", 0.15) * 1.5
                # Normalize weights
                total = sum(base_weights.values())
                for key in base_weights:
                    base_weights[key] /= total

            logger.info(f"Strategy weights from config: {dict(base_weights)}")
            return base_weights

        except Exception as e:
            logger.error(f"❌ Error getting strategy weights: {e}")
            # Return default weights if there's an error
            return {
                "model": 0.15,
                "trend_following": 0.20,
                "contrarian": 0.15,
                "volume_analysis": 0.20,
                "market_indicators": 0.30,
            }

    def get_prediction(self, round_data, strategy_preference=None):
        """
        Synchronous wrapper for get_prediction_async.

        Args:
            round_data: Dictionary with current round data
            strategy_preference: Optional strategy to prioritize

        Returns:
            tuple: (prediction, confidence)
        """
        try:
            # Initialize all predictions and get them from stored attributes
            all_predictions = self.get_all_predictions()

            # If we have a final prediction already, return it
            if "final" in all_predictions:
                prediction = all_predictions["final"]["prediction"]
                confidence = all_predictions["final"]["confidence"]
                return prediction, confidence

            # Determine which strategy to use based on priority
            if strategy_preference and strategy_preference in all_predictions:
                # Use the specified strategy
                prediction = all_predictions[strategy_preference]["prediction"]
                confidence = all_predictions[strategy_preference]["confidence"]
                logger.info(
                    f"Using {strategy_preference} strategy: {prediction} ({confidence:.2f})"
                )

                # Store as final prediction
                self.final_prediction = prediction
                self.final_confidence = confidence

                return prediction, confidence

            # If no specific strategy worked, use hybrid approach
            if "hybrid" in all_predictions:
                prediction = all_predictions["hybrid"]["prediction"]
                confidence = all_predictions["hybrid"]["confidence"]
            elif "ai" in all_predictions:
                prediction = all_predictions["ai"]["prediction"]
                confidence = all_predictions["ai"]["confidence"]
            elif "technical" in all_predictions:
                prediction = all_predictions["technical"]["prediction"]
                confidence = all_predictions["technical"]["confidence"]
            else:
                # Fallback to default
                prediction = "BULL"  # Default prediction
                confidence = 0.51  # Just above 50% confidence

            # Store as final prediction
            self.final_prediction = prediction
            self.final_confidence = confidence

            return prediction, confidence

        except Exception as e:
            logger.error(f"❌ Error getting prediction: {e}")
            traceback.print_exc()
            return "BULL", 0.51  # Default fallback

    def process_prediction_result(prediction_data, epoch, source="ensemble"):
        """
        Process and log prediction results.
        Uses log_prediction_details to record the prediction information.

        Args:
            prediction_data: Dictionary containing prediction information
            epoch: Current round epoch
            source: Source of the prediction

        Returns:
            dict: Enhanced prediction data
        """
        try:
            # Enhance prediction data with additional metadata
            enhanced_data = {
                **prediction_data,
                "epoch": epoch,
                "timestamp": int(time.time()),
                "source": source,
            }

            # Use the imported log_prediction_details function
            log_prediction_details(
                epoch=epoch,
                prediction=enhanced_data.get("prediction"),
                confidence=enhanced_data.get("confidence", 0),
                strategy=source,
                additional_data=enhanced_data,
            )

            return enhanced_data

        except Exception as e:
            logger.error(f"Error processing prediction result: {e}")
            return prediction_data

    def get_historical_price_context(lookback_days=7, interval="1d"):
        """
        Get historical price context for prediction.
        Uses get_historical_prices to retrieve price data.

        Args:
            lookback_days: Number of days to look back
            interval: Price interval

        Returns:
            dict: Historical price context data
        """
        try:
            # Use the imported get_historical_prices function
            prices = get_historical_prices(lookback_days, interval)

            if not prices or len(prices) < 2:
                logger.warning("Insufficient historical price data")
                return {"status": "insufficient_data"}

            # Calculate basic price metrics
            current_price = prices[-1]
            oldest_price = prices[0]
            price_change_pct = ((current_price - oldest_price) / oldest_price) * 100

            return {
                "status": "success",
                "current_price": current_price,
                "price_change_pct": price_change_pct,
                "prices": prices,
                "days": lookback_days,
                "interval": interval,
            }

        except Exception as e:
            logger.error(f"Error getting historical price context: {e}")
            return {"status": "error", "message": str(e)}
