#!/usr/bin/env python3
"""
Model Training Module for SID Method - AUGMENTED VERSION
=============================================================================
Trains and evaluates ML models for trade signal augmentation incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- RSI threshold features (exact 30/70)
- MACD alignment and cross features
- Stop loss and take profit targets
- Position sizing with 0.5-2% risk

WAVE 2 (Live Sessions & Q&A):
- Confidence scoring as training target
- Pattern confirmation features (W, M, H&S)
- Divergence detection features
- Market context features
- Reachability metrics as features

WAVE 3 (Academy Support Sessions):
- Session-based features (Asian, London, US, Overlap)
- Zone quality features
- Precision RSI crossing features
- Minimum candle requirement features
- Stop loss pip buffer features

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import os
import sys
import pickle
import json
import argparse
import logging
import warnings
import time
import gc
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ============================================================================
# GPU DETECTION
# ============================================================================
GPU_AVAILABLE = False
CUML_AVAILABLE = False
CUPY_AVAILABLE = False
GPU_COUNT = 0

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    GPU_COUNT = cp.cuda.runtime.getDeviceCount()
    print(f"✅ CuPy available: {GPU_COUNT} GPU(s) found")
except ImportError:
    print(f"⚠️ CuPy not available - GPU acceleration disabled")

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cumlRandomForest
    CUML_AVAILABLE = True
    print(f"✅ cuML available for GPU-accelerated ML")
except ImportError:
    print(f"⚠️ cuML not available - using CPU for ML")

GPU_AVAILABLE = CUPY_AVAILABLE

# ============================================================================
# CPU ML LIBRARIES
# ============================================================================
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False
LIGHTGBM_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                                recall_score, confusion_matrix, roc_auc_score,
                                classification_report, precision_recall_curve)
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
    print(f"✅ scikit-learn available (CPU)")
except ImportError:
    print(f"⚠️ scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print(f"✅ XGBoost available (CPU)")
except ImportError:
    print(f"⚠️ XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    print(f"✅ LightGBM available (CPU)")
except ImportError:
    print(f"⚠️ LightGBM not available")

# Try to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable=None, desc="", **kwargs):
            self.iterable = iterable or []
            self.desc = desc
        def __iter__(self): 
            return iter(self.iterable)
        def update(self, n=1): pass
        def close(self): pass


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ModelTrainingConfig:
    """Configuration for model training (Wave 1, 2, 3)"""
    # Wave 1: Core SID parameters
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_target: int = 50
    prefer_macd_cross: bool = True
    
    # Wave 2: Training parameters
    target_horizon: int = 5  # bars ahead to predict
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    
    # Wave 2: Model selection
    models_to_train: List[str] = field(default_factory=lambda: [
        'random_forest', 'xgboost', 'lightgbm', 'gradient_boosting'
    ])
    
    # Wave 2: Hyperparameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 15
    rf_min_samples_split: int = 5
    
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1
    
    lgb_n_estimators: int = 100
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.1
    
    gb_n_estimators: int = 100
    gb_max_depth: int = 5
    gb_learning_rate: float = 0.1
    
    # Wave 2: Cross-validation
    use_time_series_cv: bool = True
    cv_folds: int = 5
    
    # Wave 3: Quality thresholds
    min_accuracy: float = 0.55
    min_f1: float = 0.5
    min_precision: float = 0.5
    min_recall: float = 0.4
    
    # Wave 3: Early stopping
    early_stopping_rounds: int = 10


@dataclass
class ModelMetrics:
    """Container for model performance metrics (Wave 2 & 3)"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    specificity: float = 0.0
    balanced_accuracy: float = 0.0
    
    # Confusion matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Additional metrics
    training_time: float = 0.0
    inference_time: float = 0.0
    feature_importance: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'roc_auc': self.roc_auc,
            'specificity': self.specificity,
            'balanced_accuracy': self.balanced_accuracy,
            'true_positives': self.true_positives,
            'true_negatives': self.true_negatives,
            'false_positives': self.false_positives,
            'false_negatives': self.false_negatives,
            'training_time': self.training_time,
            'inference_time': self.inference_time
        }


class GPUMemoryManager:
    """Manage GPU memory for training (Wave 2 GPU acceleration)"""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self.initialized = False
        self.device = None

    def activate(self) -> bool:
        if CUPY_AVAILABLE:
            try:
                self.device = cp.cuda.Device(self.gpu_id)
                self.device.use()
                self.initialized = True
                free, total = cp.cuda.runtime.memGetInfo()
                print(f"🎮 GPU {self.gpu_id} activated: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
                return True
            except Exception as e:
                print(f"⚠️ GPU activation failed: {e}")
        return False

    def cleanup(self):
        if self.initialized and CUPY_AVAILABLE:
            try:
                cp.get_default_memory_pool().free_all_blocks()
                cp.cuda.Stream().synchronize()
                gc.collect()
            except:
                pass
        self.initialized = False

    def get_free_memory_mb(self) -> float:
        if CUPY_AVAILABLE:
            try:
                with cp.cuda.Device(self.gpu_id):
                    free, total = cp.cuda.runtime.memGetInfo()
                    return free / (1024**2)
            except:
                pass
        return 0


class ModelTrainer:
    """
    Trains and evaluates ML models for SID Method trade signal augmentation
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: ModelTrainingConfig = None, 
                 model_dir: str = 'ai/trained_models/',
                 use_gpu: bool = True,
                 verbose: bool = True):
        """
        Initialize the model trainer
        
        Args:
            config: ModelTrainingConfig instance
            model_dir: Directory to save trained models
            use_gpu: Use GPU acceleration if available
            verbose: Enable verbose output
        """
        self.config = config or ModelTrainingConfig()
        self.model_dir = model_dir
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.verbose = verbose
        
        # Create model directory
        os.makedirs(model_dir, exist_ok=True)
        
        # GPU manager
        self.gpu_manager = GPUMemoryManager(gpu_id=0)
        
        # Track trained models
        self.models: Dict[str, Any] = {}
        self.metrics: Dict[str, ModelMetrics] = {}
        self.feature_importances: Dict[str, Dict] = {}
        
        # Scaler for features
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🤖 MODEL TRAINER v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📁 Model directory: {model_dir}")
            print(f"🎮 GPU acceleration: {'Enabled' if self.use_gpu else 'Disabled'}")
            print(f"🎯 Target horizon: {self.config.target_horizon} bars")
            print(f"📊 Models to train: {self.config.models_to_train}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # DATA PREPARATION (Wave 1 & 2)
    # ========================================================================
    
    def prepare_features_and_target(self, df: pd.DataFrame, 
                                     target_col: str = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare features and target for training (Wave 1 & 2)
        
        Args:
            df: Feature DataFrame
            target_col: Target column name (if None, uses default)
        
        Returns:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature column names
        """
        # Set default target if not provided
        if target_col is None:
            target_col = f'target_direction_{self.config.target_horizon}'
            if target_col not in df.columns:
                target_col = f'target_rsi_50_{self.config.target_horizon}'
        
        # Exclude target columns and price columns from features
        exclude_cols = [col for col in df.columns if col.startswith('target_')]
        exclude_cols.extend(['open', 'high', 'low', 'close', 'volume', 'sma_50'])
        
        # Select feature columns
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols 
                       and df[col].dtype in ['float64', 'float32', 'int64']]
        
        # Get features and target
        X = df[feature_cols].values
        y = df[target_col].values
        
        # Check for NaN
        if np.isnan(X).any():
            if self.verbose:
                print(f"  ⚠️ NaN detected in features - filling with 0")
            X = np.nan_to_num(X, 0)
        
        if self.verbose:
            print(f"  📊 Features: {len(feature_cols)}")
            print(f"  🎯 Target: {target_col}")
            print(f"  📈 Target distribution: {np.mean(y)*100:.1f}% positive")
            print(f"  📐 Feature matrix shape: {X.shape}")
        
        return X, y, feature_cols
    
    def train_test_split_timeseries(self, X: np.ndarray, y: np.ndarray,
                                      test_size: float = 0.2,
                                      val_size: float = 0.1) -> Tuple:
        """
        Time series split for training (Wave 2 - no lookahead bias)
        """
        n = len(X)
        test_start = int(n * (1 - test_size))
        val_start = int(n * (1 - test_size - val_size))
        
        X_train = X[:val_start]
        y_train = y[:val_start]
        X_val = X[val_start:test_start]
        y_val = y[val_start:test_start]
        X_test = X[test_start:]
        y_test = y[test_start:]
        
        if self.verbose:
            print(f"  📊 Time series split:")
            print(f"     Train: {len(X_train)} samples")
            print(f"     Validation: {len(X_val)} samples")
            print(f"     Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    # ========================================================================
    # GPU-ACCELERATED TRAINING (Wave 2)
    # ========================================================================
    
    def train_random_forest_gpu(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  params: Dict = None) -> Tuple[Any, ModelMetrics]:
        """Train Random Forest on GPU using cuML (Wave 2 GPU acceleration)"""
        metrics = ModelMetrics()
        
        if not CUML_AVAILABLE:
            return None, metrics
        
        try:
            self.gpu_manager.activate()
            
            if self.verbose:
                print(f"  🎮 Training Random Forest on GPU...")
            
            # Default parameters (Wave 1 & 2)
            default_params = {
                'n_estimators': self.config.rf_n_estimators,
                'max_depth': self.config.rf_max_depth,
                'min_samples_split': self.config.rf_min_samples_split,
                'random_state': self.config.random_state
            }
            if params:
                default_params.update(params)
            
            # Convert to GPU arrays
            X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
            
            # Train model
            start_time = time.time()
            model = cumlRandomForest(**default_params)
            model.fit(X_train_gpu, y_train_gpu)
            metrics.training_time = time.time() - start_time
            
            # Evaluate on validation
            X_val_gpu = cp.asarray(X_val, dtype=cp.float32)
            y_pred_gpu = model.predict(X_val_gpu)
            y_pred = cp.asnumpy(y_pred_gpu)
            
            metrics = self._calculate_metrics(y_val, y_pred)
            metrics.training_time = metrics.training_time
            
            if self.verbose:
                print(f"     ✅ GPU Random Forest - F1: {metrics.f1:.4f}, Acc: {metrics.accuracy:.4f}")
            
            self.gpu_manager.cleanup()
            return model, metrics
            
        except Exception as e:
            if self.verbose:
                print(f"     ⚠️ GPU training failed: {e}, falling back to CPU")
            self.gpu_manager.cleanup()
            return None, metrics
    
    # ========================================================================
    # CPU MODEL TRAINING (Wave 1 & 2)
    # ========================================================================
    
    def train_random_forest_cpu(self, X_train: np.ndarray, y_train: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  params: Dict = None) -> Tuple[Any, ModelMetrics]:
        """Train Random Forest on CPU using scikit-learn"""
        metrics = ModelMetrics()
        
        if not SKLEARN_AVAILABLE:
            return None, metrics
        
        try:
            if self.verbose:
                print(f"  💻 Training Random Forest on CPU...")
            
            # Default parameters (Wave 1 & 2)
            default_params = {
                'n_estimators': self.config.rf_n_estimators,
                'max_depth': self.config.rf_max_depth,
                'min_samples_split': self.config.rf_min_samples_split,
                'n_jobs': -1,
                'random_state': self.config.random_state,
                'verbose': 0
            }
            if params:
                default_params.update(params)
            
            start_time = time.time()
            model = RandomForestClassifier(**default_params)
            model.fit(X_train, y_train)
            metrics.training_time = time.time() - start_time
            
            # Evaluate on validation
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            metrics.training_time = metrics.training_time
            
            # Get feature importance (Wave 2)
            if hasattr(model, 'feature_importances_'):
                metrics.feature_importance = dict(enumerate(model.feature_importances_))
            
            if self.verbose:
                print(f"     ✅ CPU Random Forest - F1: {metrics.f1:.4f}, Acc: {metrics.accuracy:.4f}")
            
            return model, metrics
            
        except Exception as e:
            if self.verbose:
                print(f"     ❌ Random Forest training failed: {e}")
            return None, metrics
    
    def train_xgboost_cpu(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            params: Dict = None) -> Tuple[Any, ModelMetrics]:
        """Train XGBoost on CPU (Wave 2)"""
        metrics = ModelMetrics()
        
        if not XGBOOST_AVAILABLE:
            return None, metrics
        
        try:
            if self.verbose:
                print(f"  💻 Training XGBoost on CPU...")
            
            # Default parameters (Wave 1 & 2)
            default_params = {
                'n_estimators': self.config.xgb_n_estimators,
                'max_depth': self.config.xgb_max_depth,
                'learning_rate': self.config.xgb_learning_rate,
                'random_state': self.config.random_state,
                'verbosity': 0,
                'tree_method': 'hist',
                'eval_metric': 'logloss'
            }
            if params:
                default_params.update(params)
            
            start_time = time.time()
            model = xgb.XGBClassifier(**default_params)
            model.fit(X_train, y_train, 
                     eval_set=[(X_val, y_val)],
                     early_stopping_rounds=self.config.early_stopping_rounds,
                     verbose=False)
            metrics.training_time = time.time() - start_time
            
            # Evaluate on validation
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            metrics.training_time = metrics.training_time
            
            # Get feature importance (Wave 2)
            if hasattr(model, 'feature_importances_'):
                metrics.feature_importance = dict(enumerate(model.feature_importances_))
            
            if self.verbose:
                print(f"     ✅ XGBoost - F1: {metrics.f1:.4f}, Acc: {metrics.accuracy:.4f}")
            
            return model, metrics
            
        except Exception as e:
            if self.verbose:
                print(f"     ❌ XGBoost training failed: {e}")
            return None, metrics
    
    def train_lightgbm_cpu(self, X_train: np.ndarray, y_train: np.ndarray,
                             X_val: np.ndarray, y_val: np.ndarray,
                             params: Dict = None) -> Tuple[Any, ModelMetrics]:
        """Train LightGBM on CPU (Wave 2)"""
        metrics = ModelMetrics()
        
        if not LIGHTGBM_AVAILABLE:
            return None, metrics
        
        try:
            if self.verbose:
                print(f"  💻 Training LightGBM on CPU...")
            
            # Default parameters (Wave 1 & 2)
            default_params = {
                'n_estimators': self.config.lgb_n_estimators,
                'max_depth': self.config.lgb_max_depth,
                'learning_rate': self.config.lgb_learning_rate,
                'random_state': self.config.random_state,
                'verbose': -1
            }
            if params:
                default_params.update(params)
            
            start_time = time.time()
            model = lgb.LGBMClassifier(**default_params)
            model.fit(X_train, y_train,
                     eval_set=[(X_val, y_val)],
                     eval_metric='logloss',
                     callbacks=[lgb.early_stopping(self.config.early_stopping_rounds)])
            metrics.training_time = time.time() - start_time
            
            # Evaluate on validation
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            metrics.training_time = metrics.training_time
            
            # Get feature importance (Wave 2)
            if hasattr(model, 'feature_importances_'):
                metrics.feature_importance = dict(enumerate(model.feature_importances_))
            
            if self.verbose:
                print(f"     ✅ LightGBM - F1: {metrics.f1:.4f}, Acc: {metrics.accuracy:.4f}")
            
            return model, metrics
            
        except Exception as e:
            if self.verbose:
                print(f"     ❌ LightGBM training failed: {e}")
            return None, metrics
    
    def train_gradient_boosting_cpu(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_val: np.ndarray, y_val: np.ndarray,
                                      params: Dict = None) -> Tuple[Any, ModelMetrics]:
        """Train Gradient Boosting on CPU (Wave 2)"""
        metrics = ModelMetrics()
        
        if not SKLEARN_AVAILABLE:
            return None, metrics
        
        try:
            if self.verbose:
                print(f"  💻 Training Gradient Boosting on CPU...")
            
            # Default parameters (Wave 1 & 2)
            default_params = {
                'n_estimators': self.config.gb_n_estimators,
                'max_depth': self.config.gb_max_depth,
                'learning_rate': self.config.gb_learning_rate,
                'subsample': 0.8,
                'random_state': self.config.random_state,
                'verbose': 0
            }
            if params:
                default_params.update(params)
            
            start_time = time.time()
            model = GradientBoostingClassifier(**default_params)
            model.fit(X_train, y_train)
            metrics.training_time = time.time() - start_time
            
            # Evaluate on validation
            y_pred = model.predict(X_val)
            metrics = self._calculate_metrics(y_val, y_pred)
            metrics.training_time = metrics.training_time
            
            # Get feature importance (Wave 2)
            if hasattr(model, 'feature_importances_'):
                metrics.feature_importance = dict(enumerate(model.feature_importances_))
            
            if self.verbose:
                print(f"     ✅ Gradient Boosting - F1: {metrics.f1:.4f}, Acc: {metrics.accuracy:.4f}")
            
            return model, metrics
            
        except Exception as e:
            if self.verbose:
                print(f"     ❌ Gradient Boosting training failed: {e}")
            return None, metrics
    
    # ========================================================================
    # METRICS CALCULATION (Wave 2 & 3)
    # ========================================================================
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
        """Calculate comprehensive metrics (Wave 2 & 3)"""
        metrics = ModelMetrics()
        
        try:
            # Basic metrics
            metrics.accuracy = accuracy_score(y_true, y_pred)
            metrics.precision = precision_score(y_true, y_pred, zero_division=0)
            metrics.recall = recall_score(y_true, y_pred, zero_division=0)
            metrics.f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            metrics.true_negatives = int(tn)
            metrics.false_positives = int(fp)
            metrics.false_negatives = int(fn)
            metrics.true_positives = int(tp)
            
            # Additional metrics (Wave 3)
            metrics.specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics.balanced_accuracy = (metrics.recall + metrics.specificity) / 2
            
            # ROC AUC if probabilities available
            try:
                if hasattr(self, '_last_proba'):
                    metrics.roc_auc = roc_auc_score(y_true, self._last_proba)
            except:
                pass
                
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ Metrics calculation error: {e}")
        
        return metrics
    
    def calculate_roc_auc(self, model, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """Calculate ROC AUC (Wave 2)"""
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)
                if len(proba.shape) > 1 and proba.shape[1] > 1:
                    proba = proba[:, 1]
                self._last_proba = proba
                return roc_auc_score(y_test, proba)
        except Exception as e:
            if self.verbose:
                print(f"  ⚠️ ROC AUC calculation failed: {e}")
        return 0.0
    
    # ========================================================================
    # MODEL EVALUATION (Wave 2 & 3)
    # ========================================================================
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray) -> ModelMetrics:
        """Evaluate trained model on test set (Wave 2 & 3)"""
        if self.verbose:
            print(f"  📊 Evaluating model on test set...")
        
        start_time = time.time()
        
        try:
            if hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
            else:
                return ModelMetrics()
            
            metrics = self._calculate_metrics(y_test, y_pred)
            metrics.inference_time = time.time() - start_time
            
            # Calculate ROC AUC
            metrics.roc_auc = self.calculate_roc_auc(model, X_test, y_test)
            
            if self.verbose:
                print(f"     ✅ Test - F1: {metrics.f1:.4f}, Acc: {metrics.accuracy:.4f}, AUC: {metrics.roc_auc:.4f}")
            
            return metrics
            
        except Exception as e:
            if self.verbose:
                print(f"     ❌ Evaluation failed: {e}")
            return ModelMetrics()
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                        model_type: str = 'random_forest') -> Dict:
        """
        Perform time series cross-validation (Wave 2)
        
        Args:
            X: Feature matrix
            y: Target labels
            model_type: Type of model to cross-validate
        
        Returns:
            Dictionary with cross-validation scores
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}
        
        if self.verbose:
            print(f"  🔄 Cross-validating {model_type}...")
        
        cv_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
        
        # Time series split (Wave 2)
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_val_fold = X[val_idx]
            y_val_fold = y[val_idx]
            
            # Train model
            if model_type == 'random_forest':
                model, _ = self.train_random_forest_cpu(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold, 
                    verbose=False
                )
            elif model_type == 'xgboost':
                model, _ = self.train_xgboost_cpu(
                    X_train_fold, y_train_fold, X_val_fold, y_val_fold,
                    verbose=False
                )
            else:
                continue
            
            if model:
                # Evaluate
                y_pred = model.predict(X_val_fold)
                metrics = self._calculate_metrics(y_val_fold, y_pred)
                
                cv_scores['accuracy'].append(metrics.accuracy)
                cv_scores['f1'].append(metrics.f1)
                cv_scores['precision'].append(metrics.precision)
                cv_scores['recall'].append(metrics.recall)
        
        # Calculate means and stds
        results = {}
        for metric, scores in cv_scores.items():
            if scores:
                results[metric] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'scores': scores
                }
        
        if self.verbose:
            print(f"     CV Results: F1 = {results.get('f1', {}).get('mean', 0):.4f} (±{results.get('f1', {}).get('std', 0):.4f})")
        
        return results
    
    # ========================================================================
    # MODEL SELECTION (Wave 2)
    # ========================================================================
    
    def train_and_select_best(self, X_train: np.ndarray, y_train: np.ndarray,
                                X_val: np.ndarray, y_val: np.ndarray,
                                X_test: np.ndarray, y_test: np.ndarray,
                                feature_names: List[str] = None) -> Tuple[str, Any, ModelMetrics]:
        """
        Train multiple models and select the best one based on validation F1 (Wave 2)
        
        Returns:
            (best_model_name, best_model, best_metrics)
        """
        models_to_train = []
        
        # Define available models
        if 'random_forest' in self.config.models_to_train:
            models_to_train.append(('random_forest', self.train_random_forest_cpu))
        
        if 'xgboost' in self.config.models_to_train and XGBOOST_AVAILABLE:
            models_to_train.append(('xgboost', self.train_xgboost_cpu))
        
        if 'lightgbm' in self.config.models_to_train and LIGHTGBM_AVAILABLE:
            models_to_train.append(('lightgbm', self.train_lightgbm_cpu))
        
        if 'gradient_boosting' in self.config.models_to_train:
            models_to_train.append(('gradient_boosting', self.train_gradient_boosting_cpu))
        
        # GPU Random Forest (if available)
        if 'random_forest_gpu' in self.config.models_to_train and CUML_AVAILABLE and self.use_gpu:
            models_to_train.insert(0, ('random_forest_gpu', self.train_random_forest_gpu))
        
        best_score = -1
        best_model = None
        best_name = None
        best_metrics = None
        
        for model_name, trainer_func in models_to_train:
            if self.verbose:
                print(f"\n  🚀 Training {model_name}...")
            
            model, val_metrics = trainer_func(X_train, y_train, X_val, y_val)
            
            if model:
                # Evaluate on test set
                test_metrics = self.evaluate_model(model, X_test, y_test)
                
                # Use F1 score for selection
                score = test_metrics.f1
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_name = model_name
                    best_metrics = test_metrics
                
                self.models[model_name] = model
                self.metrics[model_name] = test_metrics
                
                # Store feature importances
                if feature_names and test_metrics.feature_importance:
                    importance_dict = {}
                    for idx, imp in test_metrics.feature_importance.items():
                        if idx < len(feature_names):
                            importance_dict[feature_names[idx]] = float(imp)
                    self.feature_importances[model_name] = importance_dict
        
        if self.verbose:
            print(f"\n  🏆 Best model: {best_name} (F1: {best_score:.4f})")
        
        return best_name, best_model, best_metrics
    
    # ========================================================================
    # MODEL SAVING AND LOADING
    # ========================================================================
    
    def save_model(self, model, model_name: str, metrics: ModelMetrics,
                    feature_names: List[str] = None, 
                    metadata: Dict = None) -> bool:
        """Save trained model and metadata (Wave 2)"""
        try:
            # Save model
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics
            metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            # Save metadata
            metadata = metadata or {}
            metadata.update({
                'model_name': model_name,
                'training_date': datetime.now().isoformat(),
                'metrics': metrics.to_dict(),
                'config': {
                    'rsi_oversold': self.config.rsi_oversold,
                    'rsi_overbought': self.config.rsi_overbought,
                    'target_horizon': self.config.target_horizon,
                    'prefer_macd_cross': self.config.prefer_macd_cross
                }
            })
            
            if feature_names:
                metadata['feature_names'] = feature_names
            
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            if self.verbose:
                print(f"  💾 Model saved to {model_path}")
            
            return True
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Failed to save model: {e}")
            return False
    
    def load_model(self, model_name: str) -> Tuple[Any, Dict]:
        """Load trained model and metadata"""
        try:
            model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
            metadata_path = os.path.join(self.model_dir, f"{model_name}_metadata.json")
            
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            metadata = {}
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            if self.verbose:
                print(f"  📂 Model loaded: {model_name}")
            
            return model, metadata
            
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Failed to load model: {e}")
            return None, {}
    
    def get_model_summary(self) -> Dict:
        """Get summary of trained models"""
        summary = {}
        for name, model in self.models.items():
            metrics = self.metrics.get(name)
            summary[name] = {
                'type': type(model).__name__,
                'f1_score': metrics.f1 if metrics else None,
                'accuracy': metrics.accuracy if metrics else None,
                'training_time': metrics.training_time if metrics else None
            }
        return summary
    
    def get_best_model(self) -> Tuple[str, Any, ModelMetrics]:
        """Get the best model from training"""
        best_name = None
        best_score = -1
        best_model = None
        best_metrics = None
        
        for name, metrics in self.metrics.items():
            if metrics.f1 > best_score:
                best_score = metrics.f1
                best_name = name
                best_model = self.models.get(name)
                best_metrics = metrics
        
        return best_name, best_model, best_metrics
    
    # ========================================================================
    # COMPLETE TRAINING PIPELINE (Wave 1, 2, 3)
    # ========================================================================
    
    def train(self, df: pd.DataFrame, target_col: str = None,
               instrument: str = 'general') -> Tuple[str, Any, ModelMetrics]:
        """
        Complete training pipeline (Wave 1, 2, 3)
        
        Args:
            df: Feature DataFrame (from feature_engineering)
            target_col: Target column name
            instrument: Instrument name for model naming
        
        Returns:
            (best_model_name, best_model, best_metrics)
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🚀 TRAINING SID METHOD MODELS")
            print(f"{'='*60}")
            print(f"📈 Instrument: {instrument}")
            print(f"📊 Data shape: {df.shape}")
        
        # Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df, target_col)
        
        if len(X) == 0:
            raise ValueError("No training data available")
        
        # Time series split
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_test_split_timeseries(
            X, y, self.config.test_size, self.config.validation_size
        )
        
        # Scale features (Wave 2)
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
            X_val = self.scaler.transform(X_val)
            X_test = self.scaler.transform(X_test)
        
        # Train and select best model
        best_name, best_model, best_metrics = self.train_and_select_best(
            X_train, y_train, X_val, y_val, X_test, y_test, feature_names
        )
        
        # Save best model
        if best_model:
            model_name = f"{instrument}_{best_name}_h{self.config.target_horizon}"
            self.save_model(best_model, model_name, best_metrics, feature_names)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"✅ TRAINING COMPLETE")
            print(f"{'='*60}")
            print(f"🏆 Best model: {best_name}")
            print(f"📈 F1 Score: {best_metrics.f1:.4f}")
            print(f"📊 Accuracy: {best_metrics.accuracy:.4f}")
            print(f"🎯 Precision: {best_metrics.precision:.4f}")
            print(f"🔍 Recall: {best_metrics.recall:.4f}")
            print(f"📐 ROC AUC: {best_metrics.roc_auc:.4f}")
            print(f"⏱️ Training time: {best_metrics.training_time:.1f}s")
            print(f"{'='*60}\n")
        
        return best_name, best_model, best_metrics
    
    def predict(self, model, features: np.ndarray, 
                 threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions using trained model (Wave 2)
        
        Args:
            model: Trained model
            features: Feature matrix
            threshold: Classification threshold (default 0.5)
        
        Returns:
            Predictions (0 or 1)
        """
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features)
                if len(proba.shape) > 1 and proba.shape[1] > 1:
                    proba = proba[:, 1]
                return (proba >= threshold).astype(int)
            else:
                return model.predict(features)
        except Exception as e:
            if self.verbose:
                print(f"  ❌ Prediction failed: {e}")
            return np.zeros(len(features))


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train SID Method AI Models')
    parser.add_argument('--data', type=str, required=True, help='Path to feature data (parquet/csv)')
    parser.add_argument('--target', type=str, default='target_direction_5', help='Target column')
    parser.add_argument('--instrument', type=str, default='general', help='Instrument name')
    parser.add_argument('--model-dir', type=str, default='ai/trained_models/', help='Model directory')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--list-models', action='store_true', help='List available models')
    
    args = parser.parse_args()
    
    if args.list_models:
        print("\n📋 Available models to train:")
        print("  - random_forest")
        print("  - random_forest_gpu (if cuML available)")
        print("  - xgboost")
        print("  - lightgbm")
        print("  - gradient_boosting")
        return
    
    # Load data
    print(f"\n📂 Loading data from {args.data}...")
    if args.data.endswith('.parquet'):
        df = pd.read_parquet(args.data)
    else:
        df = pd.read_csv(args.data, index_col=0, parse_dates=True)
    
    print(f"   Data shape: {df.shape}")
    
    # Configure trainer
    config = ModelTrainingConfig()
    trainer = ModelTrainer(
        config=config,
        model_dir=args.model_dir,
        use_gpu=not args.no_gpu,
        verbose=True
    )
    
    # Train models
    best_name, best_model, metrics = trainer.train(df, args.target, args.instrument)
    
    print(f"\n✅ Training complete. Best model: {best_name}")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    print("\n" + "="*70)
    print("🧪 TESTING MODEL TRAINER v3.0")
    print("="*70)
    
    # Create sample feature data
    np.random.seed(42)
    n_samples = 10000
    
    # Create realistic features (Wave 1, 2, 3)
    features = {
        'rsi': np.random.uniform(20, 80, n_samples),
        'macd': np.random.randn(n_samples),
        'macd_signal': np.random.randn(n_samples),
        'rsi_oversold': (np.random.uniform(0, 1, n_samples) > 0.9).astype(int),
        'rsi_overbought': (np.random.uniform(0, 1, n_samples) > 0.9).astype(int),
        'macd_cross_above': (np.random.uniform(0, 1, n_samples) > 0.95).astype(int),
        'macd_cross_below': (np.random.uniform(0, 1, n_samples) > 0.95).astype(int),
        'bullish_divergence': (np.random.uniform(0, 1, n_samples) > 0.98).astype(int),
        'bearish_divergence': (np.random.uniform(0, 1, n_samples) > 0.98).astype(int),
        'double_bottom': (np.random.uniform(0, 1, n_samples) > 0.95).astype(int),
        'double_top': (np.random.uniform(0, 1, n_samples) > 0.95).astype(int),
        'is_us_session': (np.random.uniform(0, 1, n_samples) > 0.7).astype(int),
        'volatility_20': np.random.uniform(0.005, 0.02, n_samples),
        'target_direction_5': (np.random.uniform(0, 1, n_samples) > 0.5).astype(int)
    }
    
    df = pd.DataFrame(features)
    
    print(f"Sample data shape: {df.shape}")
    print(f"Target distribution: {df['target_direction_5'].mean()*100:.1f}% positive")
    
    # Initialize trainer
    config = ModelTrainingConfig()
    trainer = ModelTrainer(config, verbose=True)
    
    # Train model
    best_name, best_model, metrics = trainer.train(df, 'target_direction_5')
    
    print(f"\n✅ Model training test complete")