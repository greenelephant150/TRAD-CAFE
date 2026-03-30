"""
Signal Prediction Module
Generates AI confidence scores for zones/patterns without overriding rules
Handles feature alignment between training and prediction
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging
import os
import glob
import json

# Import ModelManager
from ai.model_manager import ModelManager

logger = logging.getLogger(__name__)


class SignalPredictor:
    """
    Generates AI-based confidence scores for trade signals
    Augments rule-based signals with ML predictions
    Handles feature alignment automatically
    """
    
    def __init__(self, accelerator, model_dir='ai/trained_models/'):
        """
        Args:
            accelerator: AIAccelerator instance
            model_dir: Directory containing trained models
        """
        self.acc = accelerator
        self.model_dir = model_dir
        self.np = accelerator.get_numpy() if accelerator else np
        self.pd = accelerator.get_pandas() if accelerator else pd
        
        # Initialize ModelManager
        self.model_manager = ModelManager(model_dir=model_dir)
        
        # Load available models
        self.models = {}  # Will store model info with feature names
        self.model_cache = {}  # Cache for loaded models
        self.load_models()
        
        # Default feature list (will be updated when model loads)
        self.expected_features = []
        
        logger.info(f"SignalPredictor initialized with {len(self.models)} models")
    
    def load_models(self):
        """Load all trained models using ModelManager"""
        # Get model summary from manager
        models_df = self.model_manager.get_latest_models_summary()
        
        if models_df.empty:
            logger.warning("No models found in directory")
            return
        
        # Load each model
        for _, row in models_df.iterrows():
            try:
                filename = row['Filename']
                model_name = row['Model Name']
                
                # Load model package
                model_package = self.model_manager.load_model(filename)
                
                if model_package:
                    # Extract metadata
                    if isinstance(model_package, dict):
                        metadata = model_package.get('metadata', {})
                        features = metadata.get('features', [])
                        
                        self.models[model_name] = {
                            'filename': filename,
                            'pair': row['Pair'],
                            'model': model_package['model'],
                            'feature_names': features,
                            'metadata': metadata,
                            'trained_to': row['To'],
                            'accuracy': row.get('Accuracy', 0),
                            'lookback': metadata.get('model_params', {}).get('lookback', 20)
                        }
                        
                        logger.info(f"✅ Loaded model: {model_name} with {len(features)} features, lookback={self.models[model_name]['lookback']}")
                
            except Exception as e:
                logger.error(f"Error loading model {row.get('Filename', 'unknown')}: {e}")
        
        # Set default expected features from first model
        if self.models:
            first_model = list(self.models.values())[0]
            self.expected_features = first_model.get('feature_names', [])
    
    def reload_models(self):
        """Reload all models (useful after training new ones)"""
        self.models = {}
        self.model_cache = {}
        self.load_models()
        logger.info(f"Reloaded {len(self.models)} models")
    
    def _align_features(self, features: Dict, expected_features: List[str], lookback: int = 20) -> np.ndarray:
        """
        Align input features with model's expected features
        
        Args:
            features: Dictionary of input features (keys like 'returns_0', 'returns_1', etc.)
            expected_features: List of feature names the model expects
            lookback: Lookback period used in training
        
        Returns:
            Numpy array of features in correct order
        """
        # If this is a lookback model (many features, e.g., 320)
        if len(expected_features) > 100:
            logger.info(f"Model expects {len(expected_features)} features with lookback={lookback}")
            logger.info(f"Features dict has {len(features)} keys")
            
            # Log sample of expected features for debugging
            logger.info(f"Sample expected features: {expected_features[:10]}")
            logger.info(f"Sample feature keys: {list(features.keys())[:10]}")
            
            # Create array in the exact order expected by the model
            feature_array = []
            missing_features = []
            
            for feat_name in expected_features:
                if feat_name in features:
                    feature_array.append(features[feat_name])
                else:
                    # If feature missing, use 0.0
                    feature_array.append(0.0)
                    missing_features.append(feat_name)
            
            if missing_features:
                logger.warning(f"Missing {len(missing_features)} features. Sample: {missing_features[:10]}")
            
            result = np.array(feature_array, dtype=np.float32).reshape(1, -1)
            logger.info(f"Created feature array with shape: {result.shape}")
            return result
        
        # Traditional feature alignment for non-lookback models
        logger.info(f"Model expects {len(expected_features)} features (non-lookback)")
        aligned = []
        missing_features = []
        
        for feat_name in expected_features:
            if feat_name in features:
                aligned.append(features[feat_name])
            else:
                # Feature missing, use default value based on typical ranges
                if 'quality' in feat_name:
                    default_val = 50.0
                elif 'rsi' in feat_name:
                    default_val = 50.0
                elif 'macd' in feat_name:
                    default_val = 0.0
                elif 'volume' in feat_name:
                    default_val = 1.0
                elif 'hour' in feat_name or 'day' in feat_name:
                    default_val = 12.0
                else:
                    default_val = 0.0
                
                aligned.append(default_val)
                missing_features.append(feat_name)
        
        if missing_features:
            logger.debug(f"Missing features: {missing_features[:5]}... using defaults")
        
        return np.array(aligned, dtype=np.float32).reshape(1, -1)
    
    def _dict_to_array(self, features: Dict) -> np.ndarray:
        """Convert feature dict to numpy array (legacy method)"""
        numeric_features = []
        feature_names = []
        
        for key in sorted(features.keys()):
            if isinstance(features[key], (int, float)):
                numeric_features.append(features[key])
                feature_names.append(key)
        
        logger.warning(f"Legacy conversion using {len(numeric_features)} features: {feature_names[:5]}...")
        return np.array(numeric_features, dtype=np.float32).reshape(1, -1)
    
    def get_model_for_current_time(self, pair: str, as_of_date: Optional[datetime] = None) -> Optional[str]:
        """Get the appropriate model name for current trading"""
        if as_of_date is None:
            as_of_date = datetime.now()
        
        model_package = self.model_manager.get_model_for_pair(pair, as_of_date)
        
        if model_package and isinstance(model_package, dict):
            metadata = model_package.get('metadata', {})
            data_range = metadata.get('data_range', {})
            max_date_str = data_range.get('max_date', '')
            
            if max_date_str:
                try:
                    max_date = datetime.fromisoformat(max_date_str)
                    return f"{pair}_{max_date.strftime('%Y%m%d')}"
                except:
                    pass
        
        # Fallback: look through loaded models
        pair_models = []
        for name, info in self.models.items():
            if info.get('pair') == pair:
                pair_models.append((info.get('trained_to', datetime.min), name))
        
        if pair_models:
            pair_models.sort(reverse=True)
            return pair_models[0][1]
        
        return None
    
    def predict_zone_success(self, features: Dict, model_name: str = 'best', pair: Optional[str] = None) -> Dict:
        """
        Predict probability of zone trade success
        
        Args:
            features: Feature dictionary from FeatureEngineer
            model_name: Name of model to use ('best', 'random_forest', etc.) or specific model name
            pair: Trading pair (for auto-selection if model_name='auto')
        
        Returns:
            Dict with prediction results
        """
        if not self.models:
            logger.warning("No models loaded, using default 0.5 probability")
            return {
                'success_probability': 0.5,
                'confidence': 0,
                'model_used': None,
                'signal_strength': 'neutral',
                'features_used': 0
            }
        
        # Auto-select model based on pair
        if model_name == 'auto' and pair:
            model_name = self.get_model_for_current_time(pair)
            if not model_name:
                model_name = 'best'
        
        # Select model
        if model_name == 'best' or model_name not in self.models:
            if pair:
                pair_models = []
                for name, info in self.models.items():
                    if info.get('pair') == pair:
                        accuracy = info.get('accuracy', 0)
                        if accuracy != 'N/A' and isinstance(accuracy, (int, float)):
                            pair_models.append((accuracy, name))
                
                if pair_models:
                    pair_models.sort(reverse=True)
                    model_name = pair_models[0][1]
                else:
                    model_name = list(self.models.keys())[0]
            else:
                model_name = list(self.models.keys())[0]
        
        model_info = self.models.get(model_name)
        
        if not model_info:
            logger.warning(f"Model {model_name} not found, using first available")
            model_name = list(self.models.keys())[0]
            model_info = self.models[model_name]
        
        model = model_info['model']
        expected_features = model_info.get('feature_names', [])
        lookback = model_info.get('lookback', 20)
        
        # Log debugging information
        logger.info(f"Using model: {model_name} for pair: {model_info.get('pair', 'unknown')}")
        logger.info(f"Model expects {len(expected_features)} features with lookback={lookback}")
        logger.info(f"Features dict has {len(features)} keys")
        logger.info(f"Sample feature keys from input: {list(features.keys())[:10]}")
        if expected_features:
            logger.info(f"Sample expected features: {expected_features[:10]}")
        
        try:
            # Align features based on expected count
            feature_array = self._align_features(features, expected_features, lookback)
            logger.info(f"Final feature array shape: {feature_array.shape}")
            
            # Get prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array)[0]
                if len(proba) > 1:
                    success_prob = proba[1]
                else:
                    success_prob = proba[0]
            elif hasattr(model, 'predict'):
                pred = model.predict(feature_array)[0]
                if hasattr(pred, 'item'):
                    pred = pred.item()
                success_prob = float(pred)
            else:
                success_prob = 0.5
            
            # Calculate confidence
            confidence = abs(success_prob - 0.5) * 200
            signal_strength = self._get_signal_strength(success_prob, confidence)
            
            model_accuracy = model_info.get('accuracy', 'N/A')
            if model_accuracy != 'N/A' and isinstance(model_accuracy, (int, float)):
                confidence = confidence * model_accuracy
            
            result = {
                'success_probability': float(success_prob),
                'confidence': float(min(confidence, 100)),
                'model_used': model_name,
                'model_pair': model_info.get('pair', 'unknown'),
                'signal_strength': signal_strength,
                'features_used': feature_array.shape[1],
                'expected_features': len(expected_features) if expected_features else 0,
                'model_accuracy': model_accuracy
            }
            
            logger.info(f"Prediction result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            logger.error(f"Features dict had keys: {list(features.keys())}")
            logger.error(f"Model expected {len(expected_features)} features")
            logger.error(f"Sample expected: {expected_features[:10] if expected_features else 'None'}")
            import traceback
            traceback.print_exc()
            return {
                'success_probability': 0.5,
                'confidence': 0,
                'model_used': model_name,
                'signal_strength': 'neutral',
                'error': str(e)
            }
    
    def _get_signal_strength(self, probability: float, confidence: float) -> str:
        """Get signal strength label"""
        conf_norm = confidence / 100
        combined = probability * conf_norm
        
        if combined >= 0.75:
            return 'VERY_STRONG'
        elif combined >= 0.65:
            return 'STRONG'
        elif combined >= 0.55:
            return 'MODERATE'
        elif combined >= 0.45:
            return 'WEAK'
        else:
            return 'VERY_WEAK'
    
    def augment_zone_signal(self, zone_data: Dict, ai_prediction: Dict) -> Dict:
        """Augment zone signal with AI prediction"""
        augmented = zone_data.copy()
        
        zone = zone_data.get('zone', {})
        if isinstance(zone_data, dict) and 'zone' not in zone_data:
            zone = zone_data
            augmented = {'zone': zone_data}
        
        augmented['ai_confidence'] = ai_prediction.get('confidence', 0)
        augmented['ai_success_probability'] = ai_prediction.get('success_probability', 0.5)
        augmented['ai_signal_strength'] = ai_prediction.get('signal_strength', 'neutral')
        augmented['ai_model_used'] = ai_prediction.get('model_used', None)
        augmented['ai_model_pair'] = ai_prediction.get('model_pair', 'unknown')
        augmented['ai_model_accuracy'] = ai_prediction.get('model_accuracy', 'N/A')
        augmented['ai_features_used'] = ai_prediction.get('features_used', 0)
        
        rule_score = zone.get('quality_score', 50)
        if isinstance(rule_score, (list, dict)):
            rule_score = 50
        
        ai_score = ai_prediction.get('success_probability', 0.5) * 100
        
        ai_confidence = ai_prediction.get('confidence', 0) / 100
        model_accuracy = ai_prediction.get('model_accuracy', 0.5)
        if model_accuracy == 'N/A' or not isinstance(model_accuracy, (int, float)):
            model_accuracy = 0.5
        
        ai_weight = min(ai_confidence * model_accuracy, 0.4)
        rule_weight = 1 - ai_weight
        
        augmented['combined_score'] = (rule_score * rule_weight) + (ai_score * ai_weight)
        augmented['ai_weight'] = ai_weight
        augmented['rule_weight'] = rule_weight
        
        zone_type = zone.get('zone_type', 'unknown')
        if augmented['combined_score'] >= 80:
            augmented['ai_recommendation'] = f"STRONG {'BUY' if zone_type == 'demand' else 'SELL'}"
        elif augmented['combined_score'] >= 65:
            augmented['ai_recommendation'] = f"{'BUY' if zone_type == 'demand' else 'SELL'}"
        elif augmented['combined_score'] >= 50:
            augmented['ai_recommendation'] = 'NEUTRAL'
        else:
            augmented['ai_recommendation'] = 'AVOID'
        
        confidence = augmented['ai_confidence']
        if confidence >= 80:
            augmented['ai_confidence_color'] = 'green'
        elif confidence >= 60:
            augmented['ai_confidence_color'] = 'lightgreen'
        elif confidence >= 40:
            augmented['ai_confidence_color'] = 'yellow'
        elif confidence >= 20:
            augmented['ai_confidence_color'] = 'orange'
        else:
            augmented['ai_confidence_color'] = 'red'
        
        return augmented
    
    def get_model_summary(self) -> Dict:
        """Get summary of loaded models"""
        summary = {
            'total_models': len(self.models),
            'models': list(self.models.keys()),
            'active_model': list(self.models.keys())[0] if self.models else None
        }
        
        feature_counts = {}
        pair_counts = {}
        lookback_values = {}
        for name, info in self.models.items():
            feature_counts[name] = len(info.get('feature_names', []))
            pair_counts[name] = info.get('pair', 'unknown')
            lookback_values[name] = info.get('lookback', 20)
        
        summary['feature_counts'] = feature_counts
        summary['pair_models'] = pair_counts
        summary['lookback_values'] = lookback_values
        
        accuracies = {}
        for name, info in self.models.items():
            acc = info.get('accuracy', 'N/A')
            if acc != 'N/A':
                accuracies[name] = acc
        summary['model_accuracies'] = accuracies
        
        return summary
    
    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get detailed info about a specific model"""
        if model_name not in self.models:
            return None
        
        info = self.models[model_name].copy()
        if 'model' in info:
            del info['model']
        
        return info
    
    def get_best_model_for_pair(self, pair: str) -> Optional[str]:
        """Get the best model name for a specific pair"""
        return self.get_model_for_current_time(pair)
    
    def predict_batch(self, features_list: List[Dict], model_name: str = 'best', pair: Optional[str] = None) -> List[Dict]:
        """Predict for multiple feature sets"""
        results = []
        for features in features_list:
            result = self.predict_zone_success(features, model_name, pair)
            results.append(result)
        return results