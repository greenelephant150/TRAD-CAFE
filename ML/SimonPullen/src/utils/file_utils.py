"""
File utilities for Simon Pullen trading system
"""

import os
import json
import pickle
import logging
from typing import Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class FileUtils:
    """Utility functions for file operations"""
    
    @staticmethod
    def ensure_dir(directory: str) -> bool:
        """Ensure directory exists, create if it doesn't"""
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            return True
        except Exception as e:
            logger.error(f"Error creating directory {directory}: {e}")
            return False
    
    @staticmethod
    def save_json(data: Any, filepath: str, indent: int = 2) -> bool:
        """Save data to JSON file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=indent, default=str)
            return True
        except Exception as e:
            logger.error(f"Error saving JSON to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_json(filepath: str) -> Optional[Any]:
        """Load data from JSON file"""
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON from {filepath}: {e}")
            return None
    
    @staticmethod
    def save_pickle(data: Any, filepath: str) -> bool:
        """Save data to pickle file"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Error saving pickle to {filepath}: {e}")
            return False
    
    @staticmethod
    def load_pickle(filepath: str) -> Optional[Any]:
        """Load data from pickle file"""
        try:
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading pickle from {filepath}: {e}")
            return None
    
    @staticmethod
    def get_file_size(filepath: str) -> Optional[int]:
        """Get file size in bytes"""
        try:
            return os.path.getsize(filepath)
        except Exception as e:
            logger.error(f"Error getting file size for {filepath}: {e}")
            return None
