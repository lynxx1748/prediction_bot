"""
Model evaluation utilities for tracking performance metrics over time.
"""

import json
import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from configuration import config

# Setup logger
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Tracks and evaluates model performance over time."""
    
    def __init__(self, model_name, metrics_file=None):
        """
        Initialize model evaluator.
        
        Args:
            model_name: Identifier for the model being evaluated
            metrics_file: Path to store metrics (defaults to config value)
        """
        self.model_name = model_name
        self.metrics_file = metrics_file or Path(config.get('paths.metrics_dir', 'metrics')) / f"{model_name}_metrics.json"
        self.metrics_history = self._load_metrics()
        self.current_version = self._get_latest_version()
        self.samples_since_training = 0
        
        # Create metrics directory if it doesn't exist
        os.makedirs(os.path.dirname(self.metrics_file), exist_ok=True)
    
    def _load_metrics(self):
        """Load metrics history from file."""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    return json.load(f)
            return {'versions': [], 'metrics': []}
        except Exception as e:
            logger.error(f"Error loading metrics file: {e}")
            return {'versions': [], 'metrics': []}
    
    def _save_metrics(self):
        """Save metrics history to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving metrics file: {e}")
            return False
    
    def _get_latest_version(self):
        """Get the latest model version from metrics history."""
        if not self.metrics_history['versions']:
            return 1
        return max(v['version'] for v in self.metrics_history['versions'])
    
    def increment_version(self, reason=None):
        """
        Increment the model version.
        
        Args:
            reason: Reason for version increment
            
        Returns:
            int: New version number
        """
        self.current_version += 1
        version_info = {
            'version': self.current_version,
            'date': datetime.datetime.now().isoformat(),
            'reason': reason or 'Routine update'
        }
        self.metrics_history['versions'].append(version_info)
        self._save_metrics()
        
        logger.info(f"Model {self.model_name} version incremented to {self.current_version}: {reason}")
        return self.current_version
    
    def record_metrics(self, metrics_dict, prediction_count=None):
        """
        Record performance metrics.
        
        Args:
            metrics_dict: Dictionary of metric names and values
            prediction_count: Number of predictions this metric covers
            
        Returns:
            bool: Success status
        """
        metrics_entry = {
            'version': self.current_version,
            'timestamp': datetime.datetime.now().isoformat(),
            'prediction_count': prediction_count,
            'metrics': metrics_dict
        }
        
        self.metrics_history['metrics'].append(metrics_entry)
        success = self._save_metrics()
        
        if success:
            logger.info(f"Recorded metrics for {self.model_name} v{self.current_version}: {metrics_dict}")
        
        return success
    
    def get_performance_trend(self, metric_name, last_n=10):
        """
        Get trend for a specific metric.
        
        Args:
            metric_name: Name of the metric to analyze
            last_n: Number of most recent entries to consider
            
        Returns:
            dict: Trend information including stability and direction
        """
        if not self.metrics_history['metrics']:
            return {'stable': False, 'direction': 'unknown', 'values': []}
        
        # Extract the requested metric from history
        values = []
        for entry in self.metrics_history['metrics'][-last_n:]:
            if metric_name in entry['metrics']:
                values.append(entry['metrics'][metric_name])
        
        if not values:
            return {'stable': False, 'direction': 'unknown', 'values': []}
        
        # Calculate trend
        slope = 0
        if len(values) > 1:
            # Simple linear regression for slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
        
        # Determine stability and direction
        stability = np.std(values) / max(0.0001, np.mean(values))
        direction = 'improving' if slope > 0.01 else 'declining' if slope < -0.01 else 'stable'
        
        return {
            'stable': stability < 0.1,  # Less than 10% standard deviation
            'direction': direction,
            'current': values[-1] if values else None,
            'average': np.mean(values) if values else None,
            'values': values,
            'slope': slope
        }
    
    def should_retrain(self):
        """
        Determine if retraining is required based on performance metrics.
        
        Returns:
            tuple: (should_retrain, reason) where should_retrain is bool
        """
        # Get accuracy trend
        accuracy_trend = self.get_performance_trend('accuracy', last_n=5)
        
        # Set threshold for retraining
        accuracy_threshold = 0.55
        
        # Check if accuracy is below threshold
        if 'current' in accuracy_trend and accuracy_trend['current'] is not None and accuracy_trend['current'] < accuracy_threshold:
            return True, f"Accuracy below threshold: {accuracy_trend['current']:.2f} < {accuracy_threshold:.2f}"
        
        # Check for declining accuracy trend
        if ('current' in accuracy_trend and 'previous' in accuracy_trend and 
            accuracy_trend['current'] is not None and accuracy_trend['previous'] is not None and
            accuracy_trend['current'] < accuracy_trend['previous'] * 0.9):
            return True, f"Declining accuracy: {accuracy_trend['current']:.2f} < {accuracy_trend['previous']:.2f}"
        
        # Check time since last training
        days_since_training = self._get_days_since_training()
        max_days_without_training = 7
        
        if days_since_training > max_days_without_training:
            return True, f"Time since last training: {days_since_training} days > {max_days_without_training} days"
        
        # Check sample count since last training
        if self.samples_since_training > 200:
            return True, f"Many new samples since training: {self.samples_since_training} > 200"
        
        return False, "Model performance is satisfactory"
    
    def plot_metrics(self, metric_names=None, save_path=None):
        """
        Generate plots for metrics over time.
        
        Args:
            metric_names: List of metrics to plot (None for all)
            save_path: Path to save the plot (None for display only)
            
        Returns:
            bool: Success status
        """
        try:
            if not self.metrics_history['metrics']:
                logger.warning("No metrics data to plot")
                return False
            
            # Prepare data
            metrics_df = pd.DataFrame([
                {
                    'date': pd.to_datetime(entry['timestamp']),
                    'version': entry['version'],
                    **entry['metrics']
                }
                for entry in self.metrics_history['metrics']
            ])
            
            # Default to all metrics if none specified
            if metric_names is None:
                # Get all column names that aren't date or version
                metric_names = [col for col in metrics_df.columns 
                               if col not in ['date', 'version']]
            
            # Create plot
            plt.figure(figsize=(12, 8))
            for metric in metric_names:
                if metric in metrics_df.columns:
                    plt.plot(metrics_df['date'], metrics_df[metric], 
                            marker='o', label=metric)
            
            # Add version change indicators
            for version_info in self.metrics_history['versions']:
                version_date = pd.to_datetime(version_info['date'])
                if version_date >= metrics_df['date'].min() and version_date <= metrics_df['date'].max():
                    plt.axvline(x=version_date, color='r', linestyle='--', alpha=0.3)
                    plt.text(version_date, plt.ylim()[1] * 0.95, f"v{version_info['version']}", 
                             rotation=90, verticalalignment='top')
            
            plt.title(f"{self.model_name} Performance Metrics")
            plt.xlabel("Date")
            plt.ylabel("Metric Value")
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            
            # Save or display
            if save_path:
                plt.savefig(save_path)
                plt.close()
                logger.info(f"Metrics plot saved to {save_path}")
            else:
                plt.tight_layout()
                plt.show()
            
            return True
            
        except Exception as e:
            logger.error(f"Error plotting metrics: {e}")
            return False
    
    def _get_days_since_training(self):
        """
        Calculate the number of days since the last model training.
        
        Returns:
            float: Number of days since last training, or a large number if never trained
        """
        try:
            # Get the latest version data
            if not self.metrics_history['versions']:
                return 999.0  # If no versions exist, return a large number
            
            latest_version = max(self.metrics_history['versions'], key=lambda v: v['version'])
            latest_date_str = latest_version.get('date')
            
            if not latest_date_str:
                return 999.0
            
            # Parse the date
            latest_date = datetime.datetime.fromisoformat(latest_date_str)
            
            # Calculate time difference
            time_diff = datetime.datetime.now() - latest_date
            
            # Return days
            return time_diff.total_seconds() / (24 * 3600)
        
        except Exception as e:
            logger.error(f"Error calculating days since training: {e}")
            return 999.0  # Return a large number on error 