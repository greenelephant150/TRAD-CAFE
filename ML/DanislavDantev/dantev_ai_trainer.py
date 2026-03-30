#!/usr/bin/env python3
"""
Danislav Dantev AI Model Trainer
Trains models to predict institutional order flow
Features: Order block strength, FVG size, liquidity sweeps, BOS/CHoCH
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class DantevAITrainer:
    """
    AI Model Trainer for Danislav Dantev's Institutional Concepts
    """
    
    def __init__(self, model_dir: str = "/mnt2/Trading-Cafe/ML/DDantev/ai/trained_models/"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models = {}
        self.scaler = None
        self.feature_names = []
        
        logger.info(f"DantevAITrainer initialized with model_dir: {self.model_dir}")
    
    def create_training_features(self, analysis: Dict) -> np.ndarray:
        """
        Create feature vector from institutional analysis
        """
        features = []
        
        # Order Block features
        if analysis.get('order_blocks'):
            best_ob = max(analysis['order_blocks'], key=lambda x: x.strength)
            features.append(best_ob.strength)
            features.append(1 if best_ob.type == 'bullish' else 0)
            features.append(best_ob.body_ratio)
            features.append(best_ob.range_size / best_ob.close if best_ob.close > 0 else 0)
        else:
            features.extend([0, 0, 0, 0])
        
        # FVG features
        if analysis.get('fair_value_gaps'):
            best_fvg = max(analysis['fair_value_gaps'], key=lambda x: x.strength)
            features.append(best_fvg.strength)
            features.append(best_fvg.gap_size_pips / 100)  # Normalize
            features.append(1 if best_fvg.filled else 0)
            features.append(best_fvg.volume_imbalance)
        else:
            features.extend([0, 0, 0, 0])
        
        # Liquidity features
        liquidity = analysis.get('liquidity_levels', {})
        swing_highs = liquidity.get('swing_highs', [])
        swing_lows = liquidity.get('swing_lows', [])
        
        swept_highs = sum(1 for h in swing_highs if h.swept)
        swept_lows = sum(1 for l in swing_lows if l.swept)
        total_highs = len(swing_highs)
        total_lows = len(swing_lows)
        
        features.append(swept_highs / max(total_highs, 1))
        features.append(swept_lows / max(total_lows, 1))
        features.append(len(liquidity.get('equal_highs', [])) / 10)
        features.append(len(liquidity.get('equal_lows', [])) / 10)
        
        # Average liquidity strength
        avg_high_strength = sum(h.strength for h in swing_highs) / max(total_highs, 1)
        avg_low_strength = sum(l.strength for l in swing_lows) / max(total_lows, 1)
        features.append(avg_high_strength)
        features.append(avg_low_strength)
        
        # BOS features
        bos_list = analysis.get('break_of_structure', [])
        if bos_list:
            latest_bos = bos_list[-1]
            features.append(1 if latest_bos.confirmed else 0)
            features.append(1 if latest_bos.volume_spike else 0)
            features.append(1 if latest_bos.direction == 'bullish' else 0)
            features.append(latest_bos.strength)
            features.append(len(bos_list) / 10)
        else:
            features.extend([0, 0, 0, 0, 0])
        
        # CHoCH features
        choch_list = analysis.get('change_of_character', [])
        if choch_list:
            latest_choch = choch_list[-1]
            features.append(1 if latest_choch.confirmed else 0)
            features.append(latest_choch.strength)
            features.append(len(choch_list) / 5)
        else:
            features.extend([0, 0, 0])
        
        # Premium/Discount features
        features.append(analysis.get('premium_discount', 0.5))
        features.append(1 if analysis.get('is_premium', False) else 0)
        features.append(1 if analysis.get('is_discount', False) else 0)
        
        # OTE alignment
        current_price = analysis.get('current_price', 0)
        ote = analysis.get('ote_levels', {})
        if current_price and ote.get('golden_ratio'):
            distance_to_ote = abs(current_price - ote['golden_ratio']) / current_price
            features.append(1 if distance_to_ote < 0.002 else 0)  # Within 0.2%
            features.append(distance_to_ote)
        else:
            features.extend([0, 0])
        
        # Market structure features
        features.append(analysis.get('trend_strength', 0.5))
        features.append(analysis.get('confluence_score', 0) / 100)
        
        # Return as numpy array
        return np.array(features, dtype=np.float32)
    
    def prepare_dataset(self, analyses: List[Dict], outcomes: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare dataset for training
        """
        X_list = []
        for analysis in analyses:
            X = self.create_training_features(analysis)
            X_list.append(X)
        
        X = np.array(X_list)
        y = np.array(outcomes)
        
        # Store feature names
        self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        return X, y
    
    def train_random_forest(self, X: np.ndarray, y: np.ndarray, 
                            params: Optional[Dict] = None) -> Any:
        """
        Train Random Forest model
        """
        try:
            from sklearn.ensemble import RandomForestClassifier
            
            default_params = {
                'n_estimators': 200,
                'max_depth': 12,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42,
                'class_weight': 'balanced'
            }
            
            if params:
                default_params.update(params)
            
            model = RandomForestClassifier(**default_params)
            model.fit(X, y)
            
            logger.info(f"Random Forest trained: {model.n_estimators} trees, depth={model.max_depth}")
            return model
            
        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            return None
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray,
                      params: Optional[Dict] = None) -> Any:
        """
        Train XGBoost model
        """
        try:
            import xgboost as xgb
            
            default_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'use_label_encoder': False,
                'eval_metric': 'logloss'
            }
            
            if params:
                default_params.update(params)
            
            model = xgb.XGBClassifier(**default_params)
            model.fit(X, y)
            
            logger.info(f"XGBoost trained: {model.n_estimators} trees")
            return model
            
        except ImportError:
            logger.warning("XGBoost not available")
            return None
        except Exception as e:
            logger.error(f"Error training XGBoost: {e}")
            return None
    
    def train_lightgbm(self, X: np.ndarray, y: np.ndarray,
                       params: Optional[Dict] = None) -> Any:
        """
        Train LightGBM model
        """
        try:
            import lightgbm as lgb
            
            default_params = {
                'n_estimators': 200,
                'max_depth': 8,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'verbose': -1,
                'class_weight': 'balanced'
            }
            
            if params:
                default_params.update(params)
            
            model = lgb.LGBMClassifier(**default_params)
            model.fit(X, y)
            
            logger.info(f"LightGBM trained: {model.n_estimators} trees")
            return model
            
        except ImportError:
            logger.warning("LightGBM not available")
            return None
        except Exception as e:
            logger.error(f"Error training LightGBM: {e}")
            return None
    
    def train_all(self, X: np.ndarray, y: np.ndarray, 
                  validation_split: float = 0.2,
                  test_split: float = 0.1) -> Dict:
        """
        Train all available models and return best
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_split, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_split / (1 - test_split), 
            random_state=42, stratify=y_temp
        )
        
        logger.info(f"Training split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        results = {}
        
        # Random Forest
        rf = self.train_random_forest(X_train, y_train)
        if rf:
            y_pred = rf.predict(X_val)
            y_pred_proba = rf.predict_proba(X_val)
            results['random_forest'] = {
                'model': rf,
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred, average='weighted'),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted')
            }
        
        # XGBoost
        xgb = self.train_xgboost(X_train, y_train)
        if xgb:
            y_pred = xgb.predict(X_val)
            results['xgboost'] = {
                'model': xgb,
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred, average='weighted'),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted')
            }
        
        # LightGBM
        lgb = self.train_lightgbm(X_train, y_train)
        if lgb:
            y_pred = lgb.predict(X_val)
            results['lightgbm'] = {
                'model': lgb,
                'accuracy': accuracy_score(y_val, y_pred),
                'f1': f1_score(y_val, y_pred, average='weighted'),
                'precision': precision_score(y_val, y_pred, average='weighted'),
                'recall': recall_score(y_val, y_pred, average='weighted')
            }
        
        # Find best model
        best_name = None
        best_score = -1
        best_model = None
        best_metrics = None
        
        for name, result in results.items():
            if result['f1'] > best_score:
                best_score = result['f1']
                best_name = name
                best_model = result['model']
                best_metrics = result
        
        # Test best model on test set
        test_metrics = {}
        if best_model:
            y_pred_test = best_model.predict(X_test)
            test_metrics = {
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'test_f1': f1_score(y_test, y_pred_test, average='weighted'),
                'test_precision': precision_score(y_test, y_pred_test, average='weighted'),
                'test_recall': recall_score(y_test, y_pred_test, average='weighted')
            }
        
        return {
            'best_model': best_name,
            'best_f1': best_score,
            'best_accuracy': best_metrics.get('accuracy', 0) if best_metrics else 0,
            'all_results': results,
            'model': best_model,
            'test_metrics': test_metrics,
            'feature_count': X.shape[1],
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'test_samples': len(X_test)
        }
    
    def save_model(self, model: Any, model_name: str, metadata: Dict = None) -> str:
        """
        Save model with metadata
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"{model_name}_{timestamp}.pkl"
        
        package = {
            'model': model,
            'metadata': metadata or {},
            'training_date': datetime.now().isoformat(),
            'model_type': model_name,
            'feature_names': self.feature_names,
            'feature_count': len(self.feature_names)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(package, f)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(self, model_path: str) -> Optional[Any]:
        """
        Load saved model
        """
        model_path = Path(model_path)
        
        if not model_path.exists():
            # Try to find latest model
            models = list(self.model_dir.glob(f"{model_path.stem}*.pkl"))
            if not models:
                logger.warning(f"No model found matching {model_path}")
                return None
            model_path = max(models, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using latest model: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                package = pickle.load(f)
            
            if isinstance(package, dict):
                model = package.get('model')
                self.feature_names = package.get('feature_names', [])
                logger.info(f"Model loaded from {model_path} with {len(self.feature_names)} features")
                return model
            else:
                logger.warning(f"Model package format unknown: {type(package)}")
                return package
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def predict(self, model: Any, analysis: Dict) -> Dict:
        """
        Make prediction from institutional analysis
        """
        features = self.create_training_features(analysis)
        
        # Reshape for prediction
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            if len(proba) > 1:
                prediction = 1 if proba[1] > 0.5 else 0
                confidence = proba[1] if prediction == 1 else 1 - proba[0]
            else:
                prediction = 1 if proba[0] > 0.5 else 0
                confidence = proba[0] if prediction == 1 else 1 - proba[0]
        elif hasattr(model, 'predict'):
            prediction = model.predict(features)[0]
            confidence = 0.5  # Default
            if hasattr(model, 'decision_function'):
                try:
                    decision = model.decision_function(features)
                    confidence = 1 / (1 + np.exp(-decision[0])) if hasattr(decision[0], '__len__') else 1 / (1 + np.exp(-decision))
                except:
                    pass
        else:
            prediction = 0
            confidence = 0
        
        return {
            'prediction': int(prediction),
            'confidence': float(confidence),
            'signal_strength': self._get_signal_strength(confidence)
        }
    
    def _get_signal_strength(self, confidence: float) -> str:
        """Get signal strength label"""
        if confidence >= 0.8:
            return 'STRONG'
        elif confidence >= 0.65:
            return 'MODERATE'
        elif confidence >= 0.55:
            return 'WEAK'
        else:
            return 'AVOID'
    
    def get_model_info(self, model_path: str = None) -> Dict:
        """
        Get information about saved models
        """
        models = list(self.model_dir.glob("*.pkl"))
        
        info = {
            'total_models': len(models),
            'models': []
        }
        
        for model_path in models:
            try:
                with open(model_path, 'rb') as f:
                    package = pickle.load(f)
                
                if isinstance(package, dict):
                    info['models'].append({
                        'filename': model_path.name,
                        'training_date': package.get('training_date', 'unknown'),
                        'model_type': package.get('model_type', 'unknown'),
                        'feature_count': package.get('feature_count', 0),
                        'size_mb': model_path.stat().st_size / (1024 * 1024)
                    })
            except:
                continue
        
        # Sort by date
        info['models'].sort(key=lambda x: x.get('training_date', ''), reverse=True)
        
        return info