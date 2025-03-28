"""
Model version control system for tracking and managing model versions.
"""

import os
import json
import shutil
import datetime
import logging
from pathlib import Path

from configuration import config

# Setup logger
logger = logging.getLogger(__name__)

class ModelVersionControl:
    """Manages versioning of machine learning models."""
    
    def __init__(self, model_name):
        """
        Initialize version control for a model.
        
        Args:
            model_name: Name of the model to track
        """
        self.model_name = model_name
        self.versions_dir = Path(config.get('paths.model_versions_dir', 'model_versions')) / model_name
        self.version_file = self.versions_dir / 'versions.json'
        
        # Create directory if it doesn't exist
        os.makedirs(self.versions_dir, exist_ok=True)
        
        # Load or initialize versions data
        self.versions = self._load_versions()
    
    def _load_versions(self):
        """Load versions data from file."""
        try:
            if os.path.exists(self.version_file):
                with open(self.version_file, 'r') as f:
                    return json.load(f)
            return {
                'model_name': self.model_name,
                'current_version': 0,
                'versions': []
            }
        except Exception as e:
            logger.error(f"Error loading versions file: {e}")
            return {
                'model_name': self.model_name,
                'current_version': 0,
                'versions': []
            }
    
    def _save_versions(self):
        """Save versions data to file."""
        try:
            with open(self.version_file, 'w') as f:
                json.dump(self.versions, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving versions file: {e}")
            return False
    
    def create_version(self, model_file, metadata=None, description=""):
        """
        Create a new version of the model.
        
        Args:
            model_file: Path to the model file
            metadata: Optional metadata about the model
            description: Description of the version
            
        Returns:
            int: Version number
        """
        try:
            if not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_file}")
                return None
            
            # Increment version number
            new_version = self.versions['current_version'] + 1
            self.versions['current_version'] = new_version
            
            # Create version directory
            version_dir = self.versions_dir / f"v{new_version}"
            os.makedirs(version_dir, exist_ok=True)
            
            # Copy model file to version directory
            dest_file = version_dir / os.path.basename(model_file)
            shutil.copy2(model_file, dest_file)
            
            # Record version info
            version_info = {
                'version': new_version,
                'timestamp': datetime.datetime.now().isoformat(),
                'file': str(dest_file),
                'metadata': metadata or {},
                'description': description
            }
            
            self.versions['versions'].append(version_info)
            self._save_versions()
            
            logger.info(f"Created version {new_version} of {self.model_name}")
            return new_version
            
        except Exception as e:
            logger.error(f"Error creating version: {e}")
            return None
    
    def get_version(self, version):
        """
        Get information about a specific version.
        
        Args:
            version: Version number
            
        Returns:
            dict: Version information
        """
        for v in self.versions['versions']:
            if v['version'] == version:
                return v
        return None
    
    def get_latest_version(self):
        """
        Get the latest version information.
        
        Returns:
            dict: Version information
        """
        if not self.versions['versions']:
            return None
        return self.versions['versions'][-1]
    
    def rollback(self, version, target_file):
        """
        Roll back to a previous version.
        
        Args:
            version: Version to roll back to
            target_file: File to write the model to
            
        Returns:
            bool: Success status
        """
        try:
            version_info = self.get_version(version)
            if not version_info:
                logger.error(f"Version {version} not found")
                return False
            
            # Copy the version file to the target location
            source_file = version_info['file']
            if not os.path.exists(source_file):
                logger.error(f"Version file not found: {source_file}")
                return False
            
            # Make sure target directory exists
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            
            # Copy the file
            shutil.copy2(source_file, target_file)
            
            logger.info(f"Rolled back to version {version} of {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back to version {version}: {e}")
            return False
    
    def compare_versions(self, version1, version2):
        """
        Compare two versions based on their metadata.
        
        Args:
            version1: First version number
            version2: Second version number
            
        Returns:
            dict: Comparison results
        """
        try:
            v1 = self.get_version(version1)
            v2 = self.get_version(version2)
            
            if not v1 or not v2:
                missing = []
                if not v1:
                    missing.append(version1)
                if not v2:
                    missing.append(version2)
                return {"error": f"Missing versions: {missing}"}
            
            # Compare metadata
            comparison = {
                'timestamp_delta': (
                    datetime.datetime.fromisoformat(v2['timestamp']) - 
                    datetime.datetime.fromisoformat(v1['timestamp'])
                ).total_seconds(),
                'metadata_diff': {}
            }
            
            # Compare metadata fields
            for key in set(v1['metadata'].keys()) | set(v2['metadata'].keys()):
                if key in v1['metadata'] and key in v2['metadata']:
                    if v1['metadata'][key] != v2['metadata'][key]:
                        comparison['metadata_diff'][key] = {
                            'v1': v1['metadata'][key],
                            'v2': v2['metadata'][key]
                        }
                elif key in v1['metadata']:
                    comparison['metadata_diff'][key] = {
                        'v1': v1['metadata'][key],
                        'v2': 'missing'
                    }
                else:
                    comparison['metadata_diff'][key] = {
                        'v1': 'missing',
                        'v2': v2['metadata'][key]
                    }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return {"error": str(e)}
    
    def list_versions(self):
        """
        List all versions of the model.
        
        Returns:
            list: Version information
        """
        return self.versions['versions'] 