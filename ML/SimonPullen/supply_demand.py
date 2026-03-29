#!/usr/bin/env python3
"""
Supply and Demand Zone Detector with AI Augmentation
Detects institutional supply/demand zones and augments with ML predictions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import json
import os

# Import AI components
from ai.ai_accelerator import AIAccelerator
from ai.signal_predictor import SignalPredictor
from ai.feature_engineering import FeatureEngineer
from ai.model_manager import ModelManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplyDemand:
    """
    Detects supply and demand zones with AI-powered confidence scoring
    """
    
    def __init__(self, 
                 lookback_left: int = 30,
                 lookback_right: int = 30,
                 zone_threshold: float = 0.02,
                 use_ai: bool = True,
                 model_dir: str = "/mnt2/Trading-Cafe/ML/SPullen/ai/trained_models/",
                 data_path: str = "/home/grct/Forex_Parquet"):
        """
        Args:
            lookback_left: Bars to look left for swing detection
            lookback_right: Bars to look right for confirmation
            zone_threshold: Threshold for zone detection
            use_ai: Whether to use AI augmentation
            model_dir: Directory containing trained models
            data_path: Path to price data
        """
        self.lookback_left = lookback_left
        self.lookback_right = lookback_right
        self.zone_threshold = zone_threshold
        self.use_ai = use_ai
        
        # Initialize AI components if enabled
        if use_ai:
            self.accelerator = AIAccelerator()
            self.model_manager = ModelManager(model_dir=model_dir)
            self.predictor = SignalPredictor(self.accelerator, model_dir=model_dir)
            self.feature_engineer = FeatureEngineer(self.accelerator)
            logger.info(f"AI components initialized with {len(self.predictor.models)} models")
        else:
            self.accelerator = None
            self.model_manager = None
            self.predictor = None
            self.feature_engineer = None
            
        logger.info(f"SupplyDemand initialized: lookback={lookback_left}/{lookback_right}, threshold={zone_threshold}, AI={use_ai}")
    
    def detect_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing highs and lows in the price data
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            DataFrame with swing points marked
        """
        df = df.copy()
        df['swing_high'] = False
        df['swing_low'] = False
        
        # Need enough data
        if len(df) < self.lookback_left + self.lookback_right + 1:
            return df
        
        # Detect swing highs
        for i in range(self.lookback_left, len(df) - self.lookback_right):
            # Check if this is a swing high
            is_swing_high = True
            current_high = df['high'].iloc[i]
            
            # Check left side
            for j in range(i - self.lookback_left, i):
                if df['high'].iloc[j] >= current_high:
                    is_swing_high = False
                    break
            
            # Check right side
            if is_swing_high:
                for j in range(i + 1, i + self.lookback_right + 1):
                    if df['high'].iloc[j] >= current_high:
                        is_swing_high = False
                        break
            
            if is_swing_high:
                df.loc[df.index[i], 'swing_high'] = True
        
        # Detect swing lows
        for i in range(self.lookback_left, len(df) - self.lookback_right):
            # Check if this is a swing low
            is_swing_low = True
            current_low = df['low'].iloc[i]
            
            # Check left side
            for j in range(i - self.lookback_left, i):
                if df['low'].iloc[j] <= current_low:
                    is_swing_low = False
                    break
            
            # Check right side
            if is_swing_low:
                for j in range(i + 1, i + self.lookback_right + 1):
                    if df['low'].iloc[j] <= current_low:
                        is_swing_low = False
                        break
            
            if is_swing_low:
                df.loc[df.index[i], 'swing_low'] = True
        
        return df
    
    def identify_initial_zones(self, df: pd.DataFrame) -> List[Dict]:
        """
        Identify initial supply/demand zones from swing points
        
        Args:
            df: DataFrame with swing points marked
        
        Returns:
            List of zone dictionaries
        """
        zones = []
        
        # Get swing points
        swing_highs = df[df['swing_high'] == True].index.tolist()
        swing_lows = df[df['swing_low'] == True].index.tolist()
        
        # Create demand zones from swing lows
        for idx in swing_lows:
            # Find the swing low price
            low_price = df.loc[idx, 'low']
            
            # Find the two most recent swing highs around this low
            left_highs = [h for h in swing_highs if h < idx]
            right_highs = [h for h in swing_highs if h > idx]
            
            if len(left_highs) > 0 and len(right_highs) > 0:
                left_high = left_highs[-1]  # Most recent before
                right_high = right_highs[0]  # First after
                
                # Calculate zone boundaries
                left_high_price = df.loc[left_high, 'high']
                right_high_price = df.loc[right_high, 'high']
                
                # Zone base is the swing low, top is min of surrounding highs
                zone_top = min(left_high_price, right_high_price)
                
                # Calculate zone quality metrics
                consolidation_bars = abs(df.index.get_loc(idx) - df.index.get_loc(left_high))
                price_range = (zone_top - low_price) / low_price * 100
                
                # Quality score (higher is better)
                if consolidation_bars > 0 and price_range > 0:
                    quality = min(95, max(30, 70 - price_range * 10 + consolidation_bars / 2))
                else:
                    quality = 50
                
                zone = {
                    'type': 'demand',
                    'price_level': low_price,
                    'zone_top': zone_top,
                    'zone_bottom': low_price * 0.998,  # Slight buffer
                    'formation_start': left_high,
                    'formation_end': idx,
                    'consolidation_bars': consolidation_bars,
                    'price_range_pct': price_range,
                    'quality_score': quality,
                    'strength': self._calculate_strength(quality, price_range, consolidation_bars),
                    'is_fresh': True,
                    'touch_count': 0,
                    'created_at': idx
                }
                
                zones.append(zone)
        
        # Create supply zones from swing highs
        for idx in swing_highs:
            # Find the swing high price
            high_price = df.loc[idx, 'high']
            
            # Find the two most recent swing lows around this high
            left_lows = [l for l in swing_lows if l < idx]
            right_lows = [l for l in swing_lows if l > idx]
            
            if len(left_lows) > 0 and len(right_lows) > 0:
                left_low = left_lows[-1]  # Most recent before
                right_low = right_lows[0]  # First after
                
                # Calculate zone boundaries
                left_low_price = df.loc[left_low, 'low']
                right_low_price = df.loc[right_low, 'low']
                
                # Zone base is the swing high, bottom is max of surrounding lows
                zone_bottom = max(left_low_price, right_low_price)
                
                # Calculate zone quality metrics
                consolidation_bars = abs(df.index.get_loc(idx) - df.index.get_loc(left_low))
                price_range = (high_price - zone_bottom) / zone_bottom * 100
                
                # Quality score (higher is better)
                if consolidation_bars > 0 and price_range > 0:
                    quality = min(95, max(30, 70 - price_range * 10 + consolidation_bars / 2))
                else:
                    quality = 50
                
                zone = {
                    'type': 'supply',
                    'price_level': high_price,
                    'zone_top': high_price * 1.002,  # Slight buffer
                    'zone_bottom': zone_bottom,
                    'formation_start': left_low,
                    'formation_end': idx,
                    'consolidation_bars': consolidation_bars,
                    'price_range_pct': price_range,
                    'quality_score': quality,
                    'strength': self._calculate_strength(quality, price_range, consolidation_bars),
                    'is_fresh': True,
                    'touch_count': 0,
                    'created_at': idx
                }
                
                zones.append(zone)
        
        return zones
    
    def _calculate_strength(self, quality: float, price_range: float, consolidation_bars: int) -> str:
        """Calculate zone strength based on metrics"""
        if quality >= 80 and price_range < 1.0 and consolidation_bars > 15:
            return 'STRONG'
        elif quality >= 60 and price_range < 2.0 and consolidation_bars > 10:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators exactly as in pkltrainer3.py
        This ensures feature consistency with training
        """
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Ranges
        df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
        hl_diff = (df['high'] - df['low']).replace(0, 1)
        df['close_position'] = (df['close'] - df['low']) / hl_diff
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
            ma_period = df[f'ma_{period}'].replace(0, 1)
            df[f'ma_ratio_{period}'] = df['close'] / ma_period
        
        # Volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Target (not used in prediction)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        return df.dropna()
    
    def extract_zone_features(self, zone: Dict, df: pd.DataFrame, current_idx: Optional[pd.Timestamp] = None) -> Dict:
        """
        Extract features from a zone for AI prediction - Matches training data format
        
        Args:
            zone: Zone dictionary
            df: DataFrame with price data and indicators
            current_idx: Current timestamp (default: last index)
        
        Returns:
            Feature dictionary for AI model that matches training features
        """
        if current_idx is None:
            current_idx = df.index[-1]
        
        # Get the lookback window (20 periods as in training)
        lookback = 20
        idx_pos = df.index.get_loc(current_idx) if current_idx in df.index else len(df) - 1
        
        if idx_pos < lookback:
            logger.warning(f"Not enough data for lookback {lookback} at position {idx_pos}")
            idx_pos = lookback
        
        # Get the lookback window of data
        lookback_df = df.iloc[idx_pos - lookback:idx_pos]
        
        # Calculate all indicators for the lookback window
        # This matches exactly what was done in training
        features = {}
        
        # For each row in the lookback window, add flattened features
        for i, (_, row) in enumerate(lookback_df.iterrows()):
            # Add all indicator columns that were used in training
            indicator_cols = [
                'returns', 'log_returns', 'high_low_ratio', 'close_position',
                'ma_5', 'ma_ratio_5', 'ma_10', 'ma_ratio_10', 
                'ma_20', 'ma_ratio_20', 'ma_50', 'ma_ratio_50',
                'volatility_5', 'volatility_10', 'volatility_20',
                'rsi_14'
            ]
            
            for col in indicator_cols:
                if col in row:
                    features[f'{col}_{i}'] = row[col]
                else:
                    features[f'{col}_{i}'] = 0.0

        logger.info(f"Extracted {len(features)} features for zone at {zone.get('price_level')}")
        logger.info(f"Feature keys sample: {list(features.keys())[:10]}")
        return features
    
    def augment_zones_with_ai(self, zones: List[Dict], df: pd.DataFrame, pair: str) -> List[Dict]:
        """
        Augment zones with AI predictions
        
        Args:
            zones: List of zone dictionaries
            df: DataFrame with price data
            pair: Trading pair
        
        Returns:
            Zones with AI augmentation
        """
        if not self.use_ai or not self.predictor:
            return zones
        
        augmented_zones = []
        
        for zone in zones:
            # Extract features for this zone using lookback window
            features = self.extract_zone_features(zone, df)
            
            # Get AI prediction
            prediction = self.predictor.predict_zone_success(
                features=features,
                model_name='auto',
                pair=pair
            )
            
            # Create zone data in format expected by augment_zone_signal
            zone_data = {'zone': zone}
            
            # Augment zone with AI prediction
            augmented_zone_data = self.predictor.augment_zone_signal(zone_data, prediction)
            
            # Extract augmented zone
            if 'zone' in augmented_zone_data:
                augmented_zone = augmented_zone_data['zone']
                # Add AI fields to the zone
                augmented_zone['ai_confidence'] = augmented_zone_data.get('ai_confidence', 0)
                augmented_zone['ai_success_probability'] = augmented_zone_data.get('ai_success_probability', 0.5)
                augmented_zone['ai_signal_strength'] = augmented_zone_data.get('ai_signal_strength', 'neutral')
                augmented_zone['ai_model_used'] = augmented_zone_data.get('ai_model_used', None)
                augmented_zone['combined_score'] = augmented_zone_data.get('combined_score', zone['quality_score'])
                augmented_zone['ai_recommendation'] = augmented_zone_data.get('ai_recommendation', 'NEUTRAL')
                augmented_zone['ai_weight'] = augmented_zone_data.get('ai_weight', 0)
                augmented_zone['ai_confidence_color'] = augmented_zone_data.get('ai_confidence_color', 'gray')
                
                augmented_zones.append(augmented_zone)
            else:
                # If augmentation fails, keep original zone
                augmented_zones.append(zone)
        
        return augmented_zones
    
    def filter_zones_by_price(self, zones: List[Dict], current_price: float, max_distance_pct: float = 5.0) -> List[Dict]:
        """
        Filter zones by distance from current price
        
        Args:
            zones: List of zones
            current_price: Current price
            max_distance_pct: Maximum distance percentage
        
        Returns:
            Filtered zones
        """
        filtered = []
        for zone in zones:
            distance = abs(current_price - zone['price_level']) / current_price * 100
            if distance <= max_distance_pct:
                zone['distance_pct'] = distance
                filtered.append(zone)
        
        return sorted(filtered, key=lambda x: x['distance_pct'])
    
    def update_zone_status(self, zones: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """
        Update zone status based on price action
        
        Args:
            zones: List of zones
            df: DataFrame with price data
        
        Returns:
            Updated zones
        """
        current_price = df['close'].iloc[-1]
        updated_zones = []
        
        for zone in zones:
            # Check if zone has been touched
            zone_price = zone['price_level']
            
            if zone['type'] == 'demand':
                # For demand zones, price touching from above
                touched = any(df['low'].iloc[-20:] <= zone_price * 1.001)
            else:
                # For supply zones, price touching from below
                touched = any(df['high'].iloc[-20:] >= zone_price * 0.999)
            
            if touched:
                zone['touch_count'] += 1
                zone['is_fresh'] = False
            
            # Check if zone is broken
            if zone['type'] == 'demand':
                broken = current_price < zone_price * 0.995
            else:
                broken = current_price > zone_price * 1.005
            
            if broken:
                zone['is_broken'] = True
                continue  # Don't include broken zones
            
            updated_zones.append(zone)
        
        return updated_zones
    
    def get_trading_signals(self, zones: List[Dict], current_price: float) -> List[Dict]:
        """
        Generate trading signals from zones
        
        Args:
            zones: List of zones
            current_price: Current price
        
        Returns:
            List of trading signals
        """
        signals = []
        
        for zone in zones:
            # Calculate distance
            distance = abs(current_price - zone['price_level']) / current_price * 100
            
            # Determine signal type based on zone type and price position
            if zone['type'] == 'demand':
                if current_price > zone['price_level']:
                    # Price above demand zone - potential buy
                    signal_type = 'BUY'
                    confidence = zone.get('combined_score', zone['quality_score'])
                    
                    # Adjust confidence based on AI if available
                    if 'ai_confidence' in zone:
                        confidence = (confidence * 0.7 + zone['ai_confidence'] * 0.3)
                    
                    signals.append({
                        'type': signal_type,
                        'zone_type': 'demand',
                        'price_level': zone['price_level'],
                        'current_price': current_price,
                        'distance_pct': distance,
                        'confidence': confidence,
                        'ai_confidence': zone.get('ai_confidence', 0),
                        'ai_signal_strength': zone.get('ai_signal_strength', 'neutral'),
                        'ai_recommendation': zone.get('ai_recommendation', 'NEUTRAL'),
                        'zone_quality': zone['quality_score'],
                        'strength': zone['strength'],
                        'target': zone['price_level'] * 1.02,  # 2% target
                        'stop_loss': zone['price_level'] * 0.995,  # Just below zone
                        'timestamp': datetime.now()
                    })
            
            else:  # supply
                if current_price < zone['price_level']:
                    # Price below supply zone - potential sell
                    signal_type = 'SELL'
                    confidence = zone.get('combined_score', zone['quality_score'])
                    
                    # Adjust confidence based on AI if available
                    if 'ai_confidence' in zone:
                        confidence = (confidence * 0.7 + zone['ai_confidence'] * 0.3)
                    
                    signals.append({
                        'type': signal_type,
                        'zone_type': 'supply',
                        'price_level': zone['price_level'],
                        'current_price': current_price,
                        'distance_pct': distance,
                        'confidence': confidence,
                        'ai_confidence': zone.get('ai_confidence', 0),
                        'ai_signal_strength': zone.get('ai_signal_strength', 'neutral'),
                        'ai_recommendation': zone.get('ai_recommendation', 'NEUTRAL'),
                        'zone_quality': zone['quality_score'],
                        'strength': zone['strength'],
                        'target': zone['price_level'] * 0.98,  # 2% target
                        'stop_loss': zone['price_level'] * 1.005,  # Just above zone
                        'timestamp': datetime.now()
                    })
        
        # Sort by confidence (highest first)
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        
        return signals
    
    def analyze(self, df: pd.DataFrame, pair: str, current_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Complete analysis: detect zones, augment with AI, generate signals
        
        Args:
            df: DataFrame with OHLC data
            pair: Trading pair
            current_price: Current price (default: last close)
        
        Returns:
            Dictionary with analysis results
        """
        if current_price is None:
            current_price = df['close'].iloc[-1]
        
        # First calculate all indicators to match training data
        df = self._calculate_all_indicators(df)
        
        # Detect swing points
        df_with_swings = self.detect_swing_points(df)
        
        # Identify initial zones
        initial_zones = self.identify_initial_zones(df_with_swings)
        logger.info(f"Identified {len(initial_zones)} initial zones")
        
        # Update zone status
        active_zones = self.update_zone_status(initial_zones, df)
        logger.info(f"{len(active_zones)} zones still active")
        
        # Filter to nearby zones
        nearby_zones = self.filter_zones_by_price(active_zones, current_price, max_distance_pct=5.0)
        logger.info(f"{len(nearby_zones)} zones within 5% of current price")
        
        # Augment with AI
        if self.use_ai and nearby_zones:
            augmented_zones = self.augment_zones_with_ai(nearby_zones, df, pair)
            logger.info(f"Augmented {len(augmented_zones)} zones with AI predictions")
        else:
            augmented_zones = nearby_zones
        
        # Generate trading signals
        signals = self.get_trading_signals(augmented_zones, current_price)
        logger.info(f"Generated {len(signals)} trading signals")
        
        return {
            'pair': pair,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'total_zones': len(initial_zones),
            'active_zones': len(active_zones),
            'nearby_zones': len(nearby_zones),
            'augmented_zones': len(augmented_zones),
            'signals': signals,
            'zones': augmented_zones,
            'ai_enabled': self.use_ai,
            'df': df_with_swings  # Return for plotting
        }


# Example usage
if __name__ == "__main__":
    # Test the class
    from ai.feature_engineering import FeatureEngineer
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 1.0,
        'high': np.random.randn(1000).cumsum() + 1.2,
        'low': np.random.randn(1000).cumsum() + 0.8,
        'close': np.random.randn(1000).cumsum() + 1.0,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Initialize detector with AI
    sd = SupplyDemand(use_ai=True)
    
    # Analyze
    pair = "EUR_USD"
    results = sd.analyze(df, pair)
    
    print(f"\n📊 Analysis Results for {pair}:")
    print(f"  Total Zones: {results['total_zones']}")
    print(f"  Active Zones: {results['active_zones']}")
    print(f"  Nearby Zones: {results['nearby_zones']}")
    print(f"  Signals: {len(results['signals'])}")
    
    if results['signals']:
        print("\n📈 Top Signals:")
        for i, signal in enumerate(results['signals'][:3]):
            print(f"\n  Signal {i+1}:")
            print(f"    Type: {signal['type']}")
            print(f"    Confidence: {signal['confidence']:.1f}")
            print(f"    AI Confidence: {signal['ai_confidence']:.1f}")
            print(f"    AI Strength: {signal['ai_signal_strength']}")
            print(f"    AI Rec: {signal['ai_recommendation']}")
            print(f"    Price: {signal['price_level']:.5f}")
            print(f"    Distance: {signal['distance_pct']:.2f}%")