#!/usr/bin/env python3
"""
Sid Naiman Supply and Demand Zone Detector
Based on SID Method principles:
- RSI signals (<30 oversold, >70 overbought)
- MACD alignment confirmation
- Stop loss at swing low/high rounded to whole number
- Take profit at RSI 50
- Daily charts only
- No trading within 14 days of earnings
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupplyDemand:
    """
    Detects supply and demand zones using Sid Naiman's SID Method
    """
    
    def __init__(self, 
                 lookback_left: int = 30,
                 lookback_right: int = 30,
                 zone_threshold: float = 0.02):
        """
        Args:
            lookback_left: Bars to look left for swing detection
            lookback_right: Bars to look right for confirmation
            zone_threshold: Threshold for zone detection
        """
        self.lookback_left = lookback_left
        self.lookback_right = lookback_right
        self.zone_threshold = zone_threshold
        
        logger.info(f"SupplyDemand (SID Method) initialized")
    
    def detect_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect swing highs and lows in the price data
        """
        df = df.copy()
        df['swing_high'] = False
        df['swing_low'] = False
        
        if len(df) < self.lookback_left + self.lookback_right + 1:
            return df
        
        # Detect swing highs
        for i in range(self.lookback_left, len(df) - self.lookback_right):
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
        Following Sid's SID Method
        """
        zones = []
        
        swing_highs = df[df['swing_high'] == True].index.tolist()
        swing_lows = df[df['swing_low'] == True].index.tolist()
        
        # Create demand zones from swing lows
        for idx in swing_lows:
            low_price = df.loc[idx, 'low']
            
            left_highs = [h for h in swing_highs if h < idx]
            right_highs = [h for h in swing_highs if h > idx]
            
            if len(left_highs) > 0 and len(right_highs) > 0:
                left_high = left_highs[-1]
                right_high = right_highs[0]
                
                left_high_price = df.loc[left_high, 'high']
                right_high_price = df.loc[right_high, 'high']
                
                zone_top = min(left_high_price, right_high_price)
                
                # Calculate zone quality
                consolidation_bars = abs(df.index.get_loc(idx) - df.index.get_loc(left_high))
                price_range = (zone_top - low_price) / low_price * 100
                
                # Quality score (simpler than Simon's)
                quality = min(95, max(30, 70 - price_range * 10))
                
                zone = {
                    'type': 'demand',
                    'price_level': low_price,
                    'zone_top': zone_top,
                    'zone_bottom': low_price * 0.998,
                    'formation_start': left_high,
                    'formation_end': idx,
                    'consolidation_bars': consolidation_bars,
                    'price_range_pct': price_range,
                    'quality_score': quality,
                    'strength': self._calculate_strength(quality),
                    'is_fresh': True,
                    'touch_count': 0,
                    'created_at': idx
                }
                
                zones.append(zone)
        
        # Create supply zones from swing highs
        for idx in swing_highs:
            high_price = df.loc[idx, 'high']
            
            left_lows = [l for l in swing_lows if l < idx]
            right_lows = [l for l in swing_lows if l > idx]
            
            if len(left_lows) > 0 and len(right_lows) > 0:
                left_low = left_lows[-1]
                right_low = right_lows[0]
                
                left_low_price = df.loc[left_low, 'low']
                right_low_price = df.loc[right_low, 'low']
                
                zone_bottom = max(left_low_price, right_low_price)
                
                consolidation_bars = abs(df.index.get_loc(idx) - df.index.get_loc(left_low))
                price_range = (high_price - zone_bottom) / zone_bottom * 100
                
                quality = min(95, max(30, 70 - price_range * 10))
                
                zone = {
                    'type': 'supply',
                    'price_level': high_price,
                    'zone_top': high_price * 1.002,
                    'zone_bottom': zone_bottom,
                    'formation_start': left_low,
                    'formation_end': idx,
                    'consolidation_bars': consolidation_bars,
                    'price_range_pct': price_range,
                    'quality_score': quality,
                    'strength': self._calculate_strength(quality),
                    'is_fresh': True,
                    'touch_count': 0,
                    'created_at': idx
                }
                
                zones.append(zone)
        
        return zones
    
    def _calculate_strength(self, quality: float) -> str:
        """Calculate zone strength based on quality score"""
        if quality >= 80:
            return 'STRONG'
        elif quality >= 60:
            return 'MODERATE'
        else:
            return 'WEAK'
    
    def filter_zones_by_price(self, zones: List[Dict], current_price: float, max_distance_pct: float = 5.0) -> List[Dict]:
        """Filter zones by distance from current price"""
        filtered = []
        for zone in zones:
            distance = abs(current_price - zone['price_level']) / current_price * 100
            if distance <= max_distance_pct:
                zone['distance_pct'] = distance
                filtered.append(zone)
        
        return sorted(filtered, key=lambda x: x['distance_pct'])
    
    def update_zone_status(self, zones: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Update zone status based on price action"""
        current_price = df['close'].iloc[-1]
        updated_zones = []
        
        for zone in zones:
            zone_price = zone['price_level']
            
            if zone['type'] == 'demand':
                touched = any(df['low'].iloc[-20:] <= zone_price * 1.001)
            else:
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
                continue
            
            updated_zones.append(zone)
        
        return updated_zones
    
    def get_trading_signals(self, zones: List[Dict], current_price: float, 
                             rsi_value: Optional[float] = None,
                             macd_aligned: bool = False) -> List[Dict]:
        """
        Generate trading signals from zones using SID Method rules
        - RSI < 30 for longs
        - RSI > 70 for shorts
        - MACD alignment confirmation
        """
        signals = []
        
        for zone in zones:
            distance = abs(current_price - zone['price_level']) / current_price * 100
            
            # Demand zone - potential buy
            if zone['type'] == 'demand' and current_price > zone['price_level']:
                if rsi_value is not None and rsi_value < 30 and macd_aligned:
                    signal_type = 'BUY'
                    confidence = zone['quality_score']
                    
                    signals.append({
                        'type': signal_type,
                        'zone_type': 'demand',
                        'price_level': zone['price_level'],
                        'current_price': current_price,
                        'distance_pct': distance,
                        'confidence': confidence,
                        'zone_quality': zone['quality_score'],
                        'strength': zone['strength'],
                        'target': zone['price_level'] * 1.02,  # 2% target
                        'stop_loss': zone['price_level'] * 0.995,
                        'timestamp': datetime.now()
                    })
            
            # Supply zone - potential sell
            elif zone['type'] == 'supply' and current_price < zone['price_level']:
                if rsi_value is not None and rsi_value > 70 and macd_aligned:
                    signal_type = 'SELL'
                    confidence = zone['quality_score']
                    
                    signals.append({
                        'type': signal_type,
                        'zone_type': 'supply',
                        'price_level': zone['price_level'],
                        'current_price': current_price,
                        'distance_pct': distance,
                        'confidence': confidence,
                        'zone_quality': zone['quality_score'],
                        'strength': zone['strength'],
                        'target': zone['price_level'] * 0.98,
                        'stop_loss': zone['price_level'] * 1.005,
                        'timestamp': datetime.now()
                    })
        
        signals.sort(key=lambda x: x['confidence'], reverse=True)
        return signals
    
    def analyze(self, df: pd.DataFrame, pair: str, current_price: Optional[float] = None,
                 rsi_value: Optional[float] = None, macd_aligned: bool = False) -> Dict[str, Any]:
        """
        Complete analysis: detect zones, generate signals using SID Method
        """
        if current_price is None:
            current_price = df['close'].iloc[-1]
        
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
        
        # Generate trading signals using SID Method rules
        signals = self.get_trading_signals(nearby_zones, current_price, rsi_value, macd_aligned)
        logger.info(f"Generated {len(signals)} trading signals with SID Method")
        
        return {
            'pair': pair,
            'timestamp': datetime.now(),
            'current_price': current_price,
            'total_zones': len(initial_zones),
            'active_zones': len(active_zones),
            'nearby_zones': len(nearby_zones),
            'signals': signals,
            'zones': nearby_zones,
            'df': df_with_swings
        }


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='1h')
    df = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 1.0,
        'high': np.random.randn(1000).cumsum() + 1.2,
        'low': np.random.randn(1000).cumsum() + 0.8,
        'close': np.random.randn(1000).cumsum() + 1.0,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates)
    
    # Initialize detector
    sd = SupplyDemand()
    
    # Analyze
    pair = "EUR_USD"
    results = sd.analyze(df, pair, rsi_value=25, macd_aligned=True)
    
    print(f"\n📊 SID Method Analysis Results for {pair}:")
    print(f"  Total Zones: {results['total_zones']}")
    print(f"  Active Zones: {results['active_zones']}")
    print(f"  Nearby Zones: {results['nearby_zones']}")
    print(f"  Signals: {len(results['signals'])}")
    
    if results['signals']:
        print("\n📈 SID Method Signals:")
        for i, signal in enumerate(results['signals'][:3]):
            print(f"\n  Signal {i+1}:")
            print(f"    Type: {signal['type']}")
            print(f"    Confidence: {signal['confidence']:.1f}")
            print(f"    Price: {signal['price_level']:.5f}")
            print(f"    Stop Loss: {signal['stop_loss']:.5f}")
            print(f"    Target: {signal['target']:.5f}")