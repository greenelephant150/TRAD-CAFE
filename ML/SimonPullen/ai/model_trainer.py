"""
Model Training Module
Trains various ML models to predict trade success without modifying rules
"""

import os
import pickle
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates ML models for trade signal augmentation
    Supports both CPU (scikit-learn) and GPU (cuML, TensorFlow) backends
    """
    
    def __init__(self, accelerator, model_dir='ai/trained_models/'):
        """
        Args:
            accelerator: AIAccelerator instance
            model_dir: Directory to save trained models
        """
        self.acc = accelerator
        self.model_dir = model_dir
        self.np = accelerator.get_numpy()
        self.pd = accelerator.get_pandas()
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize ML libraries
        self.sk = accelerator.get_sklearn_or_cuml()
        self.tf = accelerator.get_tensorflow()
        
        # Track trained models
        self.models = {}
        self.metrics = {}
        
        logger.info(f"ModelTrainer initialized with backend: {accelerator.device_type}")
    
    def train_random_forest(self, X_train, y_train, params: Optional[Dict] = None) -> Any:
        """
        Train Random Forest classifier
        Uses cuML if GPU available, otherwise scikit-learn
        """
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        try:
            # Try GPU version first
            if self.acc.device_type != 'cpu' and hasattr(self.sk, 'ensemble'):
                from cuml.ensemble import RandomForestClassifier as cumlRF
                model = cumlRF(**default_params)
                logger.info("Training Random Forest on GPU (cuML)")
            else:
                # CPU version
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**default_params)
                logger.info("Training Random Forest on CPU (scikit-learn)")
            
            model.fit(X_train, y_train)
            return model
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            return None
    
    def train_xgboost(self, X_train, y_train, params: Optional[Dict] = None) -> Any:
        """Train XGBoost classifier (CPU only currently)"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42
        }
        
        if params:
            default_params.update(params)
        
        try:
            import xgboost as xgb
            model = xgb.XGBClassifier(**default_params)
            model.fit(X_train, y_train)
            logger.info("Training XGBoost on CPU")
            return model
        except ImportError:
            logger.warning("XGBoost not installed, skipping")
            return None
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            return None
    
    def train_lightgbm(self, X_train, y_train, params: Optional[Dict] = None) -> Any:
        """Train LightGBM classifier (CPU)"""
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1
        }
        
        if params:
            default_params.update(params)
        
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**default_params)
            model.fit(X_train, y_train)
            logger.info("Training LightGBM on CPU")
            return model
        except ImportError:
            logger.warning("LightGBM not installed, skipping")
            return None
        except Exception as e:
            logger.error(f"Error training LightGBM: {e}")
            return None
    
    def train_tensorflow(self, X_train, y_train, 
                          input_shape: int,
                          params: Optional[Dict] = None) -> Any:
        """Train TensorFlow neural network"""
        if not self.tf:
            logger.warning("TensorFlow not available, skipping")
            return None
        
        default_params = {
            'epochs': 50,
            'batch_size': 32,
            'validation_split': 0.2,
            'patience': 5
        }
        
        if params:
            default_params.update(params)
        
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
            from tensorflow.keras.callbacks import EarlyStopping
            
            # Build model
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_shape,)),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy', 'AUC']
            )
            
            # Early stopping
            early_stop = EarlyStopping(
                monitor='val_loss',
                patience=default_params['patience'],
                restore_best_weights=True
            )
            
            # Train
            history = model.fit(
                X_train, y_train,
                epochs=default_params['epochs'],
                batch_size=default_params['batch_size'],
                validation_split=default_params['validation_split'],
                callbacks=[early_stop],
                verbose=0
            )
            
            logger.info(f"TensorFlow model trained for {len(history.history['loss'])} epochs")
            
            # Add training history to model object
            model.history = history.history
            return model
            
        except Exception as e:
            logger.error(f"Error training TensorFlow model: {e}")
            return None
    
    def evaluate_model(self, model, X_test, y_test, model_name: str) -> Dict:
        """Evaluate model performance"""
        metrics = {}
        
        try:
            # Get predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                y_pred = (y_pred_proba > 0.5).astype(int)
            elif hasattr(model, 'predict'):
                y_pred = model.predict(X_test)
                if hasattr(y_pred, 'numpy'):
                    y_pred = y_pred.numpy()
                y_pred_proba = y_pred  # For metrics that need probabilities
            else:
                return {'error': 'Model cannot predict'}
            
            # Convert to numpy if needed
            if hasattr(y_test, 'values'):
                y_test = y_test.values
            if hasattr(y_pred, 'get'):
                y_pred = y_pred.get()
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics['accuracy'] = accuracy_score(y_test, y_pred)
            metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
            metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
            metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
            
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except:
                metrics['roc_auc'] = 0.5
            
            # Confusion matrix
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            metrics = {'error': str(e)}
        
        self.metrics[model_name] = metrics
        return metrics
    
    def save_model(self, model, model_name: str, metrics: Optional[Dict] = None):
        """Save trained model and metrics"""
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
        
        try:
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Save metrics if provided
            if metrics:
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            logger.info(f"Model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_name: str) -> Tuple[Any, Optional[Dict]]:
        """Load trained model and metrics"""
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        metrics_path = os.path.join(self.model_dir, f"{model_name}_metrics.json")
        
        try:
            # Load model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Load metrics if available
            metrics = None
            if os.path.exists(metrics_path):
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
            
            logger.info(f"Model loaded from {model_path}")
            return model, metrics
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None, None
    
    def get_best_model(self, X_train, y_train, X_val, y_val) -> Tuple[str, Any, Dict]:
        """
        Train multiple models and return the best one based on validation F1 score
        """
        models_to_try = [
            ('random_forest', self.train_random_forest),
            ('xgboost', self.train_xgboost),
            ('lightgbm', self.train_lightgbm)
        ]
        
        # Add TensorFlow if available
        if self.tf:
            models_to_try.append(('tensorflow', self.train_tensorflow))
        
        best_score = -1
        best_model = None
        best_name = None
        best_metrics = None
        
        for model_name, trainer_func in models_to_try:
            logger.info(f"Training {model_name}...")
            
            if model_name == 'tensorflow':
                model = trainer_func(X_train, y_train, X_train.shape[1])
            else:
                model = trainer_func(X_train, y_train)
            
            if model:
                metrics = self.evaluate_model(model, X_val, y_val, model_name)
                f1_score = metrics.get('f1', 0)
                
                logger.info(f"  {model_name} F1 Score: {f1_score:.3f}")
                
                if f1_score > best_score:
                    best_score = f1_score
                    best_model = model
                    best_name = model_name
                    best_metrics = metrics
        
        return best_name, best_model, best_metrics