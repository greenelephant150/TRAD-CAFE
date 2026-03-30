"""
SID Method Inference Module
Makes predictions using trained models to augment Sid's rule-based signals
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import os
import glob

logger = logging.getLogger(__name__)


class SidInference:
    """
    Makes predictions using trained Sid Method models
    Augments rule-based signals with ML confidence scores
    """
    
    def __init__(self, model_dir: str = None):
        """
        Args:
            model_dir: Directory containing trained models
        """
        if model_dir is None:
            # Auto-detect model directory
            possible_paths = [
                "/mnt2/Trading-Cafe/ML/SNaiman2/ai/trained_models/",
                "./ai/trained_models/",
                "../ai/trained_models/"
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    model_dir = path
                    break
        
        self.model_dir = model_dir
        self.models = {}  # Cache of loaded models
        self.model_info = {}  # Model metadata
        self.load_models()
        
        logger.info(f"SidInference initialized with {len(self.models)} models from {self.model_dir}")
    
    def load_models(self):
        """Load all trained Sid Method models"""
        if not self.model_dir or not os.path.exists(self.model_dir):
            logger.warning(f"Model directory not found: {self.model_dir}")
            return
        
        model_files = glob.glob(os.path.join(self.model_dir, "*SidMethod*.pkl"))
        
        for model_path in model_files:
            try:
                filename = os.path.basename(model_path)
                
                # Parse filename: DDMMYYYY--SidMethod--PAIR--TARGET--L{lookback}.pkl
                parts = filename.replace('.pkl', '').split('--')
                if len(parts) >= 5:
                    date_str, method, pair, target, lookback_part = parts[:5]
                    lookback = int(lookback_part.replace('L', '')) if lookback_part.startswith('L') else 10
                    
                    with open(model_path, 'rb') as f:
                        model_package = pickle.load(f)
                    
                    if isinstance(model_package, dict) and 'model' in model_package:
                        self.models[filename] = model_package['model']
                        self.model_info[filename] = {
                            'pair': pair,
                            'target': target,
                            'lookback': lookback,
                            'date': date_str,
                            'accuracy': model_package.get('metadata', {}).get('accuracy', 0),
                            'samples': model_package.get('metadata', {}).get('samples', 0),
                            'best_model': model_package.get('metadata', {}).get('best_model', 'unknown'),
                            'path': model_path
                        }
                        logger.debug(f"Loaded model: {filename} (acc={self.model_info[filename]['accuracy']:.3f})")
            
            except Exception as e:
                logger.error(f"Error loading model {model_path}: {e}")
    
    def get_model_for_pair(self, pair: str, target: str = None) -> Optional[tuple]:
        """
        Get the best model for a specific pair
        Returns (model, model_info)
        """
        pair_models = []
        
        for filename, info in self.model_info.items():
            if info['pair'] == pair:
                if target is None or info['target'] == target:
                    pair_models.append((info['accuracy'], filename, info))
        
        if not pair_models:
            return None
        
        # Get model with highest accuracy
        pair_models.sort(reverse=True)
        best_filename = pair_models[0][1]
        
        return self.models[best_filename], self.model_info[best_filename]
    
    def extract_features_from_candle(self, df: pd.DataFrame, idx: int, 
                                       lookback: int = 10) -> Dict:
        """
        Extract features from a single candle for prediction
        Matches the features used in training
        """
        if idx < lookback:
            return {}
        
        features = {}
        
        # Use lookback window of candles
        for i in range(lookback):
            candle_idx = idx - i
            if candle_idx < 0:
                continue
            candle = df.iloc[candle_idx]
            
            # RSI features
            features[f'rsi_{i}'] = candle.get('rsi', 50)
            features[f'rsi_oversold_{i}'] = 1 if candle.get('rsi', 50) < 30 else 0
            features[f'rsi_overbought_{i}'] = 1 if candle.get('rsi', 50) > 70 else 0
            features[f'rsi_mid_{i}'] = 1 if 45 <= candle.get('rsi', 50) <= 55 else 0
            
            # MACD features
            features[f'macd_{i}'] = candle.get('macd', 0)
            features[f'macd_signal_{i}'] = candle.get('macd_signal', 0)
            features[f'macd_hist_{i}'] = candle.get('macd_hist', 0)
            features[f'macd_cross_above_{i}'] = candle.get('macd_cross_above', 0)
            features[f'macd_cross_below_{i}'] = candle.get('macd_cross_below', 0)
            
            # Price action
            features[f'returns_{i}'] = candle.get('returns', 0)
            features[f'volatility_5_{i}'] = candle.get('volatility_5', 0.01)
            features[f'body_ratio_{i}'] = candle.get('body_ratio', 0.5)
        
        return features
    
    def predict_rsi_50_probability(self, df: pd.DataFrame, idx: int,
                                     pair: str) -> Dict:
        """
        Predict probability that RSI will reach 50 within next bars
        """
        # Try to get model for direction prediction first
        model_result = self.get_model_for_pair(pair, 'target_direction')
        if not model_result:
            return {
                'probability': 0.5,
                'confidence': 0,
                'model_used': None,
                'message': 'No model found for this pair'
            }
        
        model, model_info = model_result
        
        # Extract features
        features = self.extract_features_from_candle(
            df, idx, lookback=model_info['lookback']
        )
        
        if not features:
            return {
                'probability': 0.5,
                'confidence': 0,
                'model_used': model_info,
                'message': 'Insufficient data for feature extraction'
            }
        
        # Convert features to array in correct order
        feature_array = np.array(list(features.values()), dtype=np.float32).reshape(1, -1)
        
        try:
            # Get prediction
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(feature_array)[0]
                if len(proba) > 1:
                    probability = proba[1]  # Probability of class 1 (up direction)
                else:
                    probability = proba[0]
            elif hasattr(model, 'predict'):
                prediction = model.predict(feature_array)[0]
                probability = float(prediction)
            else:
                probability = 0.5
            
            # Calculate confidence based on model accuracy
            confidence = model_info['accuracy'] * 100
            
            return {
                'probability': float(probability),
                'confidence': float(confidence),
                'model_used': model_info,
                'target': model_info['target'],
                'lookback': model_info['lookback'],
                'prediction': 'UP' if probability > 0.5 else 'DOWN'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'probability': 0.5,
                'confidence': 0,
                'model_used': model_info,
                'error': str(e)
            }
    
    def augment_signal(self, rsi_value: float, macd_aligned: bool,
                        ml_prediction: Dict) -> Dict:
        """
        Augment Sid's rule-based signal with ML prediction
        """
        # Rule-based confidence
        rule_confidence = 0
        
        if rsi_value < 30:
            rule_confidence = 70  # Strong oversold signal
        elif rsi_value > 70:
            rule_confidence = 70  # Strong overbought signal
        else:
            rule_confidence = 30  # Neutral
        
        if macd_aligned:
            rule_confidence += 20
        
        # ML confidence
        ml_prob = ml_prediction.get('probability', 0.5)
        ml_confidence = ml_prob * 100
        
        # Combine (weighted average)
        # Give more weight to rules (Sid's method is rule-based first)
        combined_confidence = (rule_confidence * 0.7 + ml_confidence * 0.3)
        
        # Determine signal strength
        if combined_confidence >= 80:
            strength = 'VERY_STRONG'
            recommendation = 'STRONG BUY' if rsi_value < 30 else 'STRONG SELL'
        elif combined_confidence >= 65:
            strength = 'STRONG'
            recommendation = 'BUY' if rsi_value < 30 else 'SELL'
        elif combined_confidence >= 50:
            strength = 'MODERATE'
            recommendation = 'CONSIDER' if rsi_value < 30 or rsi_value > 70 else 'NEUTRAL'
        else:
            strength = 'WEAK'
            recommendation = 'AVOID'
        
        return {
            'rule_confidence': rule_confidence,
            'ml_confidence': ml_confidence,
            'combined_confidence': combined_confidence,
            'strength': strength,
            'recommendation': recommendation,
            'ml_prediction': ml_prediction
        }
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all loaded models"""
        if not self.model_info:
            return pd.DataFrame()
        
        rows = []
        for filename, info in self.model_info.items():
            rows.append({
                'Pair': info['pair'],
                'Target': info['target'],
                'Lookback': info['lookback'],
                'Accuracy': f"{info['accuracy']:.3f}",
                'Samples': info['samples'],
                'Best Model': info['best_model'],
                'Date': info['date']
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['Pair', 'Accuracy'], ascending=[True, False])
        return df