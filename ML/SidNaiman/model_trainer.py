"""
Model Training Module
Trains ML models to predict trade success probability
Supports both CPU (scikit-learn) and GPU (cuml) backends
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Try to import scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                f1_score, roc_auc_score, confusion_matrix)
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError as e:
    SKLEARN_AVAILABLE = False
    logger.warning(f"scikit-learn not available: {e}")

# Try to import XGBoost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Try to import LightGBM
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# Try to import cuML for GPU acceleration
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cumlRandomForestClassifier
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


class ModelTrainer:
    """
    Train and evaluate ML models for trade prediction
    Supports multiple algorithms with GPU acceleration
    """
    
    def __init__(self, accelerator=None, model_dir='ai/trained_models/'):
        """
        Args:
            accelerator: AIAccelerator instance for GPU/CPU management
            model_dir: Directory to save trained models
        """
        self.accelerator = accelerator
        self.model_dir = model_dir
        self.models = {}
        self.metrics = {}
        self.scaler = None
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Detect available backends
        self.backends = {
            'sklearn': SKLEARN_AVAILABLE,
            'xgboost': XGBOOST_AVAILABLE,
            'lightgbm': LIGHTGBM_AVAILABLE,
            'cuml': CUML_AVAILABLE
        }
        
        logger.info(f"ModelTrainer initialized with backends: {self.backends}")
        logger.info(f"Models will be saved to: {os.path.abspath(model_dir)}")
    
    def prepare_data(self, df: pd.DataFrame, feature_cols: List[str], 
                      target_col: str = 'outcome',
                      test_size: float = 0.2,
                      scale_features: bool = True) -> Dict:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and target
            feature_cols: List of feature column names
            target_col: Name of target column
            test_size: Proportion of data to use for testing
            scale_features: Whether to scale features
        
        Returns:
            Dictionary with train/test splits and scaler
        """
        if df.empty:
            raise ValueError("Empty DataFrame provided")
        
        # Extract features and target
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Convert categorical features if needed
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = pd.Categorical(X[col]).codes
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        if scale_features:
            self.scaler = StandardScaler()
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_names': feature_cols,
            'scaler': self.scaler
        }
    
    def train_random_forest(self, X_train, y_train, X_test=None, y_test=None,
                              params: Optional[Dict] = None,
                              use_gpu: bool = False) -> Tuple[Any, Dict]:
        """
        Train Random Forest classifier
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features (optional)
            y_test: Test labels (optional)
            params: Model parameters
            use_gpu: Whether to use GPU if available
        
        Returns:
            Tuple of (model, metrics)
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        }
        
        if params:
            default_params.update(params)
        
        model = None
        metrics = {}
        
        # Try GPU first if requested and available
        if use_gpu and CUML_AVAILABLE:
            try:
                logger.info("Training Random Forest on GPU (cuML)")
                model = cumlRandomForestClassifier(
                    n_estimators=default_params['n_estimators'],
                    max_depth=default_params['max_depth'],
                    random_state=default_params['random_state']
                )
                model.fit(X_train, y_train)
                backend_used = 'cuml'
            except Exception as e:
                logger.warning(f"GPU training failed, falling back to CPU: {e}")
                model = None
        
        # Fall back to scikit-learn
        if model is None and SKLEARN_AVAILABLE:
            logger.info("Training Random Forest on CPU (scikit-learn)")
            model = RandomForestClassifier(**default_params)
            model.fit(X_train, y_train)
            backend_used = 'sklearn'
        
        if model is None:
            raise RuntimeError("No ML backend available for Random Forest")
        
        # Evaluate if test data provided
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['backend'] = backend_used
            metrics['params'] = default_params
        
        return model, metrics
    
    def train_xgboost(self, X_train, y_train, X_test=None, y_test=None,
                        params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Train XGBoost classifier"""
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        if params:
            default_params.update(params)
        
        logger.info("Training XGBoost model")
        model = xgb.XGBClassifier(**default_params)
        model.fit(X_train, y_train)
        
        metrics = {}
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['params'] = default_params
        
        return model, metrics
    
    def train_lightgbm(self, X_train, y_train, X_test=None, y_test=None,
                         params: Optional[Dict] = None) -> Tuple[Any, Dict]:
        """Train LightGBM classifier"""
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not installed")
        
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        logger.info("Training LightGBM model")
        model = lgb.LGBMClassifier(**default_params)
        model.fit(X_train, y_train)
        
        metrics = {}
        if X_test is not None and y_test is not None:
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
            metrics['params'] = default_params
        
        return model, metrics
    
    def train_all_models(self, X_train, y_train, X_test, y_test,
                           use_gpu: bool = False) -> Dict[str, Tuple[Any, Dict]]:
        """Train all available models and return best one"""
        results = {}
        
        # Random Forest
        if SKLEARN_AVAILABLE or (use_gpu and CUML_AVAILABLE):
            try:
                model, metrics = self.train_random_forest(
                    X_train, y_train, X_test, y_test, use_gpu=use_gpu
                )
                results['random_forest'] = (model, metrics)
            except Exception as e:
                logger.error(f"Random Forest failed: {e}")
        
        # XGBoost
        if XGBOOST_AVAILABLE:
            try:
                model, metrics = self.train_xgboost(X_train, y_train, X_test, y_test)
                results['xgboost'] = (model, metrics)
            except Exception as e:
                logger.error(f"XGBoost failed: {e}")
        
        # LightGBM
        if LIGHTGBM_AVAILABLE:
            try:
                model, metrics = self.train_lightgbm(X_train, y_train, X_test, y_test)
                results['lightgbm'] = (model, metrics)
            except Exception as e:
                logger.error(f"LightGBM failed: {e}")
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """Calculate classification metrics"""
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0))
        }
        
        if y_pred_proba is not None:
            try:
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
                else:
                    metrics['roc_auc'] = float(roc_auc_score(y_true, y_pred_proba))
            except:
                metrics['roc_auc'] = 0.5
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        return metrics
    
    def save_model(self, model, model_name: str, metrics: Optional[Dict] = None,
                    feature_names: Optional[List[str]] = None):
        """Save trained model and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = os.path.join(self.model_dir, f"{model_name}_{timestamp}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        if metrics or feature_names or self.scaler:
            metadata = {
                'model_name': model_name,
                'timestamp': timestamp,
                'metrics': metrics or {},
                'feature_names': feature_names or [],
                'scaler': self.scaler
            }
            
            meta_path = os.path.join(self.model_dir, f"{model_name}_{timestamp}_meta.json")
            
            # Convert scaler to serializable format
            if self.scaler:
                metadata['scaler_mean'] = self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None
                metadata['scaler_scale'] = self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
            
            with open(meta_path, 'w') as f:
                # Convert numpy types to Python types for JSON serialization
                def convert(obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, dict):
                        return {k: convert(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [convert(v) for v in obj]
                    return obj
                
                json.dump(convert(metadata), f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str) -> Tuple[Any, Optional[Dict]]:
        """Load trained model and metadata"""
        # Load model
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load metadata if exists
        meta_path = model_path.replace('.pkl', '_meta.json')
        metadata = None
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        return model, metadata
    
    def get_best_model(self, results: Dict[str, Tuple[Any, Dict]]) -> Tuple[str, Any, Dict]:
        """Get best model based on F1 score"""
        best_score = -1
        best_name = None
        best_model = None
        best_metrics = None
        
        for name, (model, metrics) in results.items():
            score = metrics.get('f1', 0)
            if score > best_score:
                best_score = score
                best_name = name
                best_model = model
                best_metrics = metrics
        
        return best_name, best_model, best_metrics
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        import glob
        models = glob.glob(os.path.join(self.model_dir, "*.pkl"))
        return [os.path.basename(m) for m in models]
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get information about a specific model"""
        model_path = os.path.join(self.model_dir, model_name)
        if not os.path.exists(model_path):
            return None
        
        _, metadata = self.load_model(model_path)
        return metadata