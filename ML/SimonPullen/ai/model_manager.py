import os
import pickle
import glob
import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import json
import sys
import types

logger = logging.getLogger(__name__)

# ============================================================================
# PATCH SCIKIT-LEARN TREE CLASS AT RUNTIME
# ============================================================================
try:
    import sklearn
    from sklearn.tree._tree import Tree
    
    # Check which version we're dealing with
    sklearn_version = tuple(map(int, sklearn.__version__.split('.')[:3]))
    logger.info(f"Detected scikit-learn version: {sklearn.__version__}")
    
    # For scikit-learn >= 1.4, we need to handle legacy models
    if sklearn_version >= (1, 4, 0):
        logger.info("Patching scikit-learn Tree class for legacy model compatibility")
        
        # Store the original __setstate__ method
        if hasattr(Tree, '__setstate__'):
            original_setstate = Tree.__setstate__
        else:
            original_setstate = None
        
        def patched_setstate(self, state):
            """Patched __setstate__ that adds missing_go_to_left if needed"""
            # Handle different state formats
            if original_setstate:
                try:
                    # Try original first
                    original_setstate(self, state)
                except Exception as e:
                    logger.warning(f"Original __setstate__ failed: {e}")
                    # Manual initialization
                    if isinstance(state, tuple):
                        # Old format: (children_left, children_right, feature, threshold, 
                        #              impurity, n_node_samples, weighted_n_node_samples)
                        if len(state) >= 7:
                            self.children_left = state[0]
                            self.children_right = state[1]
                            self.feature = state[2]
                            self.threshold = state[3]
                            self.impurity = state[4]
                            self.n_node_samples = state[5]
                            self.weighted_n_node_samples = state[6]
                            self.node_count = len(self.children_left)
            else:
                # Manual initialization if no original
                if isinstance(state, tuple):
                    if len(state) >= 7:
                        self.children_left = state[0]
                        self.children_right = state[1]
                        self.feature = state[2]
                        self.threshold = state[3]
                        self.impurity = state[4]
                        self.n_node_samples = state[5]
                        self.weighted_n_node_samples = state[6]
                        self.node_count = len(self.children_left)
            
            # Add missing_go_to_left if it doesn't exist
            if not hasattr(self, 'missing_go_to_left'):
                if hasattr(self, 'node_count'):
                    self.missing_go_to_left = np.zeros(self.node_count, dtype=np.uint8)
                elif hasattr(self, 'n_node_samples') and hasattr(self.n_node_samples, '__len__'):
                    self.missing_go_to_left = np.zeros(len(self.n_node_samples), dtype=np.uint8)
                    self.node_count = len(self.n_node_samples)
                else:
                    # Try to determine node_count from state
                    if isinstance(state, tuple) and len(state) > 0:
                        for item in state:
                            if hasattr(item, '__len__'):
                                self.missing_go_to_left = np.zeros(len(item), dtype=np.uint8)
                                self.node_count = len(item)
                                break
        
        # Apply the patch
        Tree.__setstate__ = patched_setstate
        
        # Also patch the __reduce__ method to ensure compatibility
        if hasattr(Tree, '__reduce__'):
            original_reduce = Tree.__reduce__
            
            def patched_reduce(self):
                """Patched __reduce__ that includes missing_go_to_left"""
                reduce_data = original_reduce(self)
                # Ensure missing_go_to_left is included in state
                if isinstance(reduce_data, tuple) and len(reduce_data) >= 2:
                    state = reduce_data[1]
                    if isinstance(state, tuple):
                        # Add missing_go_to_left to state if not present
                        if len(state) == 7:  # Old format
                            new_state = state + (np.zeros(self.node_count, dtype=np.uint8),)
                            reduce_data = (reduce_data[0], new_state) + reduce_data[2:]
                return reduce_data
            
            Tree.__reduce__ = patched_reduce
        
        logger.info("Successfully patched Tree class for legacy compatibility")
    else:
        logger.info(f"Running with older scikit-learn {sklearn.__version__}, no patching needed")
        
except ImportError as e:
    logger.warning(f"Could not patch scikit-learn: {e}")
except Exception as e:
    logger.warning(f"Error patching scikit-learn: {e}")
    import traceback
    traceback.print_exc()


# Custom unpickler to handle scikit-learn version compatibility (as fallback)
class SklearnCompatUnpickler(pickle.Unpickler):
    """Custom unpickler that handles scikit-learn version mismatches"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug("Initializing SklearnCompatUnpickler")
    
    def find_class(self, module, name):
        # Handle scikit-learn tree structure changes
        if module == 'sklearn.tree._tree' and name == 'Tree':
            logger.debug("Intercepting Tree class creation")
            # Return the already patched Tree class
            try:
                from sklearn.tree._tree import Tree
                return Tree
            except:
                pass
        elif module.startswith('sklearn.'):
            try:
                return super().find_class(module, name)
            except (AttributeError, ModuleNotFoundError) as e:
                logger.debug(f"Could not find {module}.{name}: {e}")
                # Try older module paths
                if module == 'sklearn.tree._tree':
                    try:
                        return super().find_class('sklearn.tree', name)
                    except:
                        pass
                elif module == 'sklearn.ensemble._forest':
                    try:
                        return super().find_class('sklearn.ensemble.forest', name)
                    except:
                        pass
        return super().find_class(module, name)


def safe_pickle_load(file_path):
    """
    Safely load a pickle file with compatibility handling for scikit-learn versions
    
    Args:
        file_path: Path to pickle file
    
    Returns:
        Loaded object or None if failed
    """
    logger.debug(f"Attempting to load {file_path}")
    
    try:
        # First try normal loading
        logger.debug("Trying normal pickle loading")
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
            logger.debug("Normal pickle loading succeeded")
            return obj
    except Exception as e:
        logger.warning(f"Normal pickle loading failed for {file_path}: {e}")
        
        # Try with custom unpickler for scikit-learn compatibility
        try:
            logger.debug("Trying custom unpickler")
            with open(file_path, 'rb') as f:
                unpickler = SklearnCompatUnpickler(f)
                obj = unpickler.load()
                logger.debug("Custom unpickler succeeded")
                
                # Post-process to add missing fields to all trees
                if hasattr(obj, 'estimators_'):
                    logger.debug(f"Found RandomForest with {len(obj.estimators_)} estimators")
                    for i, estimator in enumerate(obj.estimators_):
                        if hasattr(estimator, 'tree_'):
                            tree = estimator.tree_
                            # Add missing_go_to_left
                            if not hasattr(tree, 'missing_go_to_left'):
                                if hasattr(tree, 'node_count'):
                                    logger.debug(f"Adding missing_go_to_left to tree {i}")
                                    tree.missing_go_to_left = np.zeros(tree.node_count, dtype=np.uint8)
                            
                            # Add monotonic_cst for the tree itself
                            if not hasattr(estimator, 'monotonic_cst'):
                                logger.debug(f"Adding monotonic_cst to estimator {i}")
                                estimator.monotonic_cst = None
                else:
                    logger.debug("Object is not a RandomForest model")
                
                return obj
                
        except Exception as e2:
            logger.error(f"Safe pickle loading also failed for {file_path}: {e2}")
            
            # Last resort: try loading with different encoding
            try:
                logger.debug("Trying latin1 encoding")
                with open(file_path, 'rb') as f:
                    return pickle.load(f, encoding='latin1')
            except Exception as e3:
                logger.error(f"All loading methods failed for {file_path}: {e3}")
                return None


class ModelManager:
    """
    Manages models created by pkltrainer3.py and makes them available to the AI module
    """
    
    def __init__(self, 
                 model_dir: str = "/mnt2/Trading-Cafe/ML/SPullen/ai/trained_models/",
                 parquet_base_path: str = "/home/grct/Forex_Parquet"):
        """
        Args:
            model_dir: Directory where pkltrainer3.py saves models
            parquet_base_path: Path to parquet data for validation
        """
        self.model_dir = model_dir
        self.parquet_base_path = parquet_base_path
        self.models = {}  # Cache of loaded models
        self.model_index = {}  # Quick lookup index
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        logger.info(f"ModelManager initialized with model_dir: {model_dir}")
        
        # Build initial index
        self._rebuild_index()
        
    def _rebuild_index(self):
        """Rebuild the model index from disk"""
        logger.debug("Rebuilding model index")
        self.model_index = self.discover_models()
        logger.info(f"Found {len(self.model_index)} models")
        
    def discover_models(self) -> Dict[str, Dict]:
        """
        Discover all models created by pkltrainer3.py
        
        Returns:
            Dictionary of {model_filename: metadata}
        """
        model_files = glob.glob(os.path.join(self.model_dir, "*.pkl"))
        discovered = {}
        
        for model_path in model_files:
            try:
                # Extract model info from filename
                filename = os.path.basename(model_path)
                
                # Parse filename format: DDMMYYYY--DDMMYYYY--SPullen--PAIR--S5.pkl
                parts = filename.replace('.pkl', '').split('--')
                if len(parts) >= 5:
                    min_date_str, max_date_str, creator, pair, timeframe = parts
                    
                    # Convert dates
                    min_date = datetime.strptime(min_date_str, "%d%m%Y")
                    max_date = datetime.strptime(max_date_str, "%d%m%Y")
                    
                    # Quick metadata without loading full model
                    discovered[filename] = {
                        'path': model_path,
                        'pair': pair,
                        'min_date': min_date,
                        'max_date': max_date,
                        'timeframe': timeframe,
                        'filename': filename,
                        'size_mb': os.path.getsize(model_path) / (1024*1024),
                        'model_name': f"{pair}_{max_date.strftime('%Y%m%d')}"
                    }
                    logger.debug(f"Added model from filename: {pair}")
                else:
                    logger.debug(f"Filename doesn't match expected format: {filename}")
                    # Try to load metadata from within the file safely
                    try:
                        model_data = safe_pickle_load(model_path)
                        
                        if model_data and isinstance(model_data, dict) and 'metadata' in model_data:
                            meta = model_data['metadata']
                            pair = meta.get('pair', 'unknown')
                            data_range = meta.get('data_range', {})
                            
                            min_date_str = data_range.get('min_date', '')
                            max_date_str = data_range.get('max_date', '')
                            
                            if min_date_str and max_date_str:
                                min_date = datetime.fromisoformat(min_date_str)
                                max_date = datetime.fromisoformat(max_date_str)
                            else:
                                continue
                            
                            discovered[filename] = {
                                'path': model_path,
                                'pair': pair,
                                'min_date': min_date,
                                'max_date': max_date,
                                'timeframe': meta.get('model_params', {}).get('timeframe', '1h'),
                                'filename': filename,
                                'size_mb': os.path.getsize(model_path) / (1024*1024),
                                'model_name': f"{pair}_{max_date.strftime('%Y%m%d')}",
                                'accuracy': meta.get('performance', {}).get('accuracy', 0),
                                'samples': meta.get('samples', 0),
                                'device': meta.get('device', 'unknown')
                            }
                            logger.debug(f"Added model from metadata: {pair}")
                    except Exception as e:
                        logger.debug(f"Could not load metadata from {filename}: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error parsing {model_path}: {e}")
                continue
                
        return discovered
    
    def load_model(self, model_filename: str) -> Optional[Dict]:
        """
        Load a specific model with all metadata
        Handles legacy scikit-learn models with missing fields
        
        Args:
            model_filename: Filename of the model (e.g., '01012020--31122020--SPullen--EUR_USD--S5.pkl')
        
        Returns:
            Complete model package with metadata
        """
        logger.info(f"Loading model: {model_filename}")
        
        # Check cache first
        if model_filename in self.models:
            logger.info(f"Returning cached model: {model_filename}")
            return self.models[model_filename]
        
        model_path = os.path.join(self.model_dir, model_filename)
        
        try:
            # Use safe loading function
            model_package = safe_pickle_load(model_path)
            
            if model_package is None:
                logger.error(f"Failed to load {model_filename} after multiple attempts")
                return None
            
            logger.debug(f"Successfully loaded model package of type: {type(model_package)}")
            
            # Ensure it has the expected structure
            if not isinstance(model_package, dict):
                logger.debug(f"Converting legacy format (not a dict)")
                # Convert legacy format
                model_package = {
                    'model': model_package,
                    'metadata': self.model_index.get(model_filename, {
                        'pair': 'unknown',
                        'features': []
                    })
                }
            
            # Get the model object
            model = model_package.get('model')
            
            # Apply additional patches for scikit-learn compatibility
            if hasattr(model, 'estimators_'):
                logger.debug(f"Model is a RandomForest with {len(model.estimators_)} estimators")
                for i, estimator in enumerate(model.estimators_):
                    # Add monotonic_cst if missing
                    if not hasattr(estimator, 'monotonic_cst'):
                        logger.debug(f"Adding monotonic_cst to estimator {i}")
                        estimator.monotonic_cst = None
                    
                    # Add missing tree attributes
                    if hasattr(estimator, 'tree_'):
                        tree = estimator.tree_
                        if not hasattr(tree, 'missing_go_to_left') and hasattr(tree, 'node_count'):
                            logger.debug(f"Adding missing_go_to_left to tree {i}")
                            tree.missing_go_to_left = np.zeros(tree.node_count, dtype=np.uint8)
                        
                        # Add other potential missing attributes
                        if not hasattr(tree, 'n_outputs'):
                            logger.debug(f"Adding n_outputs to tree {i}")
                            tree.n_outputs = 1
            else:
                logger.debug("Model is not a RandomForest (no estimators_ attribute)")
            
            # Cache it
            self.models[model_filename] = model_package
            logger.info(f"Successfully loaded and cached: {model_filename}")
            return model_package
            
        except Exception as e:
            logger.error(f"Error loading {model_filename}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_model_for_pair(self, pair: str, as_of_date: Optional[datetime] = None) -> Optional[Dict]:
        """
        Get the most recent model for a specific pair
        
        Args:
            pair: Trading pair (e.g., 'EUR_USD')
            as_of_date: Date to use for model selection (default: now)
        
        Returns:
            Most appropriate model package
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        logger.info(f"Getting model for {pair} as of {as_of_date.date()}")
        
        # Refresh index
        self._rebuild_index()
        
        # Filter for this pair
        pair_models = []
        for filename, info in self.model_index.items():
            if info['pair'] == pair:
                # Check if model covers the as_of_date
                if info['min_date'] <= as_of_date <= info['max_date']:
                    pair_models.append((info['max_date'], filename, info))
                    logger.debug(f"Found matching model: {filename}")
        
        if not pair_models:
            logger.warning(f"No model found for {pair} as of {as_of_date.date()}")
            return None
        
        # Get most recent model (by max_date)
        pair_models.sort(reverse=True)  # Most recent first
        best_model_filename = pair_models[0][1]
        logger.info(f"Selected model: {best_model_filename}")
        
        # Load and return the model
        return self.load_model(best_model_filename)
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict]:
        """
        Get a model by its friendly name (pair_YYYYMMDD)
        
        Args:
            model_name: Friendly name (e.g., 'EUR_USD_20201231')
        
        Returns:
            Model package
        """
        logger.info(f"Getting model by name: {model_name}")
        for filename, info in self.model_index.items():
            if info.get('model_name') == model_name:
                return self.load_model(filename)
        return None
    
    def get_latest_models_summary(self) -> pd.DataFrame:
        """
        Get a summary of all available models
        
        Returns:
            DataFrame with model information
        """
        self._rebuild_index()
        
        rows = []
        for filename, info in self.model_index.items():
            # Ensure min_date and max_date are datetime objects
            min_date = info['min_date']
            max_date = info['max_date']
            
            # Convert to datetime if they're strings
            if isinstance(min_date, str):
                try:
                    from datetime import datetime
                    min_date = datetime.fromisoformat(min_date)
                except:
                    min_date = datetime.now()
            
            if isinstance(max_date, str):
                try:
                    from datetime import datetime
                    max_date = datetime.fromisoformat(max_date)
                except:
                    max_date = datetime.now()
            
            # Calculate days difference safely
            if hasattr(max_date, 'date') and hasattr(min_date, 'date'):
                days_diff = (max_date.date() - min_date.date()).days
            else:
                days_diff = 0
            
            rows.append({
                'Pair': info['pair'],
                'From': min_date,  # Keep as datetime object
                'To': max_date,    # Keep as datetime object
                'Days': days_diff,
                'Accuracy': info.get('accuracy', 'N/A'),
                'Samples': info.get('samples', 0),
                'Device': info.get('device', 'unknown'),
                'Size (MB)': round(info['size_mb'], 1),
                'Model Name': info.get('model_name', filename),
                'Filename': filename
            })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(['Pair', 'To'], ascending=[True, False])
        
        logger.info(f"Generated summary with {len(df)} models")
        return df
    
    def delete_model(self, model_filename: str) -> bool:
        """
        Delete a model file
        
        Args:
            model_filename: Filename to delete
        
        Returns:
            True if successful
        """
        model_path = os.path.join(self.model_dir, model_filename)
        logger.info(f"Deleting model: {model_filename}")
        try:
            os.remove(model_path)
            # Remove from cache
            if model_filename in self.models:
                del self.models[model_filename]
            # Rebuild index
            self._rebuild_index()
            logger.info(f"Deleted model: {model_filename}")
            return True
        except Exception as e:
            logger.error(f"Error deleting {model_filename}: {e}")
            return False
    
    def get_model_features(self, model_filename: str) -> List[str]:
        """
        Get the feature list used by a model
        
        Args:
            model_filename: Filename of the model
        
        Returns:
            List of feature names
        """
        model_package = self.load_model(model_filename)
        if model_package:
            if isinstance(model_package, dict):
                metadata = model_package.get('metadata', {})
                return metadata.get('features', [])
        return []
    
    def compare_models(self, pair: str) -> pd.DataFrame:
        """
        Compare all models for a specific pair
        
        Args:
            pair: Trading pair
        
        Returns:
            DataFrame with model comparison
        """
        self._rebuild_index()
        
        rows = []
        for filename, info in self.model_index.items():
            if info['pair'] == pair:
                rows.append({
                    'Model': info.get('model_name', filename),
                    'Trained To': info['max_date'].date(),
                    'Data From': info['min_date'].date(),
                    'Coverage Days': (info['max_date'] - info['min_date']).days,
                    'Accuracy': info.get('accuracy', 'N/A'),
                    'Samples': info.get('samples', 0),
                    'Device': info.get('device', 'unknown'),
                    'Size MB': round(info['size_mb'], 1)
                })
        
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values('Trained To', ascending=False)
        
        return df