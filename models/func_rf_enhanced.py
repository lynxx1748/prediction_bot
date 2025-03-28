"""
Enhanced Random Forest model with incremental learning and self-optimization.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import sqlite3
import joblib
import traceback
import logging
import os
from pathlib import Path
from datetime import datetime
import time

# Import from configuration module
from configuration import config
from .model_evaluation import ModelEvaluator
from .model_version_control import ModelVersionControl

# Setup logger
logger = logging.getLogger(__name__)

class AdaptiveRandomForest:
    """Random Forest model with adaptive learning capabilities."""
    
    def __init__(self, model_name="adaptive_rf"):
        """
        Initialize the Adaptive Random Forest model.
        
        Args:
            model_name: Name identifier for the model
        """
        self.model_name = model_name
        self.model = None
        self.feature_importance = {}
        self.training_data = []
        self.recent_predictions = []
        self.max_history = 1000  # Maximum samples to keep in memory
        
        # Get configuration
        self.config = config.get('random_forest', {})
        self.min_samples = self.config.get('min_samples', 100)
        self.recency_weight = self.config.get('recency_weight', True)
        self.recency_halflife = self.config.get('recency_halflife', 500)  # Samples after which weight is halved
        
        # Model parameters (with defaults)
        self.model_params = self.config.get('parameters', {
            'n_estimators': 100,
            'max_depth': 6,
            'min_samples_split': 10,
            'class_weight': 'balanced'
        })
        
        # Setup file paths
        self.model_path = Path(config.get('paths.model_file', 'models')) / f"{model_name}.joblib"
        
        # Initialize evaluator and version control
        self.evaluator = ModelEvaluator(model_name)
        self.version_control = ModelVersionControl(model_name)
        
        # Load existing model or create new one
        self._initialize_model()
        
    def _initialize_model(self):
        """Load existing model or create new one."""
        try:
            if os.path.exists(self.model_path):
                logger.info(f"Loading existing model from {self.model_path}")
                model_data = joblib.load(self.model_path)
                
                # Handle different possible formats
                if isinstance(model_data, tuple) and len(model_data) >= 2:
                    self.model, metadata = model_data[:2]
                    if isinstance(metadata, dict) and 'feature_importance' in metadata:
                        self.feature_importance = metadata['feature_importance']
                elif hasattr(model_data, 'predict_proba'):
                    # Just the model was saved
                    self.model = model_data
                
                logger.info(f"Model loaded successfully")
            else:
                logger.info(f"No existing model found, will train from scratch")
                self.model = RandomForestClassifier(**self.model_params)
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            traceback.print_exc()
            self.model = RandomForestClassifier(**self.model_params)
    
    def load_historical_data(self, db_file=None, table=None, limit=10000):
        """
        Load historical data from database for training.
        
        Args:
            db_file: Path to database file
            table: Table name
            limit: Maximum number of rows to load
            
        Returns:
            bool: Success status
        """
        try:
            # Use provided parameters or get from config
            db_file = db_file or config.get('database.file')
            table = table or config.get('database.tables.trades')
            
            # Connect to database
            conn = sqlite3.connect(db_file)
            query = f"""
            SELECT * FROM {table}
            WHERE closePrice IS NOT NULL
            ORDER BY epoch DESC
            LIMIT {limit}
            """
            
            # Load data
            df = pd.read_sql(query, conn)
            conn.close()
            
            if df.empty:
                logger.warning("No historical data found in database")
                return False
            
            # Calculate outcome
            df['outcome'] = df.apply(lambda row: 'BULL' if row['closePrice'] > row['lockPrice'] else 'BEAR', axis=1)
            
            # Store data internally
            self.training_data = df.to_dict('records')
            logger.info(f"Loaded {len(self.training_data)} historical samples")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            traceback.print_exc()
            return False
    
    def train(self, force=False, save=True):
        """
        Train the Random Forest model.
        
        Args:
            force: Force training even with few samples
            save: Save model after training
            
        Returns:
            bool: Success status
        """
        start_time = time.time()
        
        try:
            # Check if we have enough data
            total_samples = len(self.training_data) + len(self.recent_predictions)
            if total_samples < self.min_samples and not force:
                logger.warning(f"Not enough samples for training: {total_samples} < {self.min_samples}")
                return False
            
            # Combine historical data with recent predictions
            all_data = self.training_data + [p for p in self.recent_predictions if p.get('actual') is not None]
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # Basic feature set
            features = ['bullRatio', 'bearRatio', 'bnb_change', 'btc_change']
            
            # Add additional features if available
            available_features = [f for f in features if f in df.columns]
            
            # Verify we have enough features
            if len(available_features) < 2:
                logger.warning(f"Not enough features available: {available_features}")
                return False
            
            # Prepare training data
            X = df[available_features]
            y = df['outcome']
            
            # Apply recency weighting if enabled
            if self.recency_weight and len(df) > 10:
                # Create weights based on recency (more recent = higher weight)
                sample_weights = np.exp(-np.arange(len(df))[::-1] / self.recency_halflife)
                # Normalize weights
                sample_weights = sample_weights / sample_weights.sum()
            else:
                sample_weights = None
            
            # Reinitialize the model
            self.model = RandomForestClassifier(**self.model_params)
            
            # Train the model
            if sample_weights is not None:
                self.model.fit(X, y, sample_weight=sample_weights)
            else:
                self.model.fit(X, y)
            
            # Store feature importance
            self.feature_importance = dict(zip(available_features, self.model.feature_importances_))
            
            # Calculate training accuracy
            training_accuracy = self.model.score(X, y)
            
            # Save the model if requested
            if save:
                metadata = {
                    'feature_importance': self.feature_importance,
                    'training_time': time.time() - start_time,
                    'sample_count': len(df),
                    'training_accuracy': training_accuracy,
                    'features': available_features
                }
                
                joblib.dump((self.model, metadata), self.model_path)
                
                # Create a version with our version control system
                version = self.version_control.create_version(
                    self.model_path,
                    metadata=metadata,
                    description=f"Trained on {len(df)} samples with {training_accuracy:.2%} accuracy"
                )
                
                # Record metrics
                self.evaluator.record_metrics({
                    'accuracy': training_accuracy,
                    'sample_count': len(df),
                    'training_time': time.time() - start_time
                })
            
            # Log results
            logger.info(f"Training completed in {time.time() - start_time:.2f}s with {training_accuracy:.2%} accuracy")
            logger.info(f"Feature importance:")
            for feat, imp in sorted(self.feature_importance.items(), key=lambda x: -x[1]):
                logger.info(f"  {feat}: {imp:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            traceback.print_exc()
            return False
    
    def predict(self, data):
        """
        Make a prediction using the model.
        
        Args:
            data: Dictionary containing features
            
        Returns:
            tuple: (prediction, confidence)
        """
        try:
            if self.model is None:
                logger.warning("Model not initialized, using default prediction")
                return "BULL", 0.51
            
            # Get all features the model was trained on
            features = list(self.feature_importance.keys())
            
            # Extract available features from input data
            X = []
            missing_features = []
            for feature in features:
                if feature in data:
                    X.append(data[feature])
                else:
                    missing_features.append(feature)
                    X.append(0.0)  # Default value
            
            if missing_features:
                logger.warning(f"Missing features in prediction data: {missing_features}")
            
            # Make prediction
            X = np.array([X])
            proba = self.model.predict_proba(X)[0]
            
            # Get class labels
            classes = self.model.classes_
            bull_idx = np.where(classes == 'BULL')[0][0] if 'BULL' in classes else 0
            
            # Determine prediction and confidence
            confidence = proba[bull_idx]
            prediction = "BULL" if confidence >= 0.5 else "BEAR"
            confidence = max(confidence, 1 - confidence)
            
            # Store prediction for later feedback
            self.recent_predictions.append({
                'epoch': data.get('epoch', 0),
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'confidence': float(confidence),
                'features': {k: data.get(k) for k in features if k in data},
                'actual': None  # Will be updated later
            })
            
            # Trim history if needed
            if len(self.recent_predictions) > self.max_history:
                self.recent_predictions = self.recent_predictions[-self.max_history:]
            
            return prediction, float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            traceback.print_exc()
            return "BULL", 0.51
    
    def record_outcome(self, epoch, actual_outcome):
        """
        Record actual outcome for a prediction.
        
        Args:
            epoch: Epoch number
            actual_outcome: Actual outcome (BULL or BEAR)
            
        Returns:
            bool: True if outcome was recorded
        """
        try:
            recorded = False
            
            # Find the prediction for this epoch
            for pred in self.recent_predictions:
                if pred['epoch'] == epoch and pred['actual'] is None:
                    # Record actual outcome
                    pred['actual'] = actual_outcome
                    recorded = True
                    
                    # Check if prediction was correct
                    correct = pred['prediction'] == actual_outcome
                    logger.info(f"Prediction for epoch {epoch} was {'correct' if correct else 'incorrect'} (predicted {pred['prediction']}, actual {actual_outcome})")
                    
                    break
            
            # If we've accumulated enough outcomes, consider retraining
            completed_predictions = sum(1 for p in self.recent_predictions if p.get('actual') is not None)
            if completed_predictions >= self.config.get('retrain_threshold', 20):
                logger.info(f"Accumulated {completed_predictions} completed predictions, considering retraining")
                
                # Calculate recent accuracy
                correct_count = sum(1 for p in self.recent_predictions 
                                 if p.get('actual') is not None and p['prediction'] == p['actual'])
                if completed_predictions > 0:
                    accuracy = correct_count / completed_predictions
                    logger.info(f"Recent accuracy: {accuracy:.2%}")
                    
                    # Retrain if accuracy is below threshold
                    if accuracy < self.config.get('retrain_accuracy_threshold', 0.55):
                        logger.info(f"Accuracy below threshold, retraining model")
                        self.train(force=True)
            
            return recorded
            
        except Exception as e:
            logger.error(f"Error recording outcome: {e}")
            traceback.print_exc()
            return False
    
    def get_performance_metrics(self):
        """
        Get performance metrics for the model.
        
        Returns:
            dict: Performance metrics
        """
        try:
            # Calculate metrics from recent predictions
            completed = [p for p in self.recent_predictions if p.get('actual') is not None]
            
            if not completed:
                return {
                    'accuracy': None,
                    'sample_count': 0,
                    'bull_accuracy': None,
                    'bear_accuracy': None
                }
            
            # Overall accuracy
            correct = sum(1 for p in completed if p['prediction'] == p['actual'])
            accuracy = correct / len(completed) if completed else 0
            
            # Bull accuracy
            bull_preds = [p for p in completed if p['prediction'] == 'BULL']
            bull_correct = sum(1 for p in bull_preds if p['actual'] == 'BULL')
            bull_accuracy = bull_correct / len(bull_preds) if bull_preds else 0
            
            # Bear accuracy
            bear_preds = [p for p in completed if p['prediction'] == 'BEAR']
            bear_correct = sum(1 for p in bear_preds if p['actual'] == 'BEAR')
            bear_accuracy = bear_correct / len(bear_preds) if bear_preds else 0
            
            return {
                'accuracy': accuracy,
                'sample_count': len(completed),
                'bull_accuracy': bull_accuracy,
                'bear_accuracy': bear_accuracy,
                'bull_predictions': len(bull_preds),
                'bear_predictions': len(bear_preds)
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            traceback.print_exc()
            return {'error': str(e)} 