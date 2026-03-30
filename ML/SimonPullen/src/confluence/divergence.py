"""
RSI Divergence Detection
Simon Pullen: Divergence increases win rate from 71% to 80% (40% profit increase)
Types:
- Regular Bullish: Price lower low, RSI higher low
- Regular Bearish: Price higher high, RSI lower high
- Hidden Bullish: Price higher low, RSI lower low
- Hidden Bearish: Price lower high, RSI higher high
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import talib


@dataclass
class DivergenceSignal:
    """Represents a divergence signal"""
    divergence_type: str  # 'regular_bullish', 'regular_bearish', 'hidden_bullish', 'hidden_bearish'
    start_idx: int
    end_idx: int
    start_price: float
    end_price: float
    start_rsi: float
    end_rsi: float
    strength: float  # 0-1, how clear the divergence is
    detected_at: Optional[pd.Timestamp] = None


class DivergenceDetector:
    """
    Detects RSI divergences following Simon Pullen's methodology
    Uses standard RSI settings (14 periods)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rsi_period = config.get('rsi_period', 14)
        self.lookback = config.get('divergence_lookback', 100)
        self.min_swing_points = config.get('min_swing_points', 5)
        
    def calculate_rsi(self, df: pd.DataFrame) -> pd.Series:
        """Calculate RSI for the dataframe"""
        close_prices = df['close'].values
        rsi = talib.RSI(close_prices, timeperiod=self.rsi_period)
        return pd.Series(rsi, index=df.index)
    
    def detect_all(self, df: pd.DataFrame) -> List[DivergenceSignal]:
        """Detect all types of divergence"""
        rsi = self.calculate_rsi(df)
        
        # Find swing highs and lows in price and RSI
        price_swing_highs = self._find_swing_highs(df['high'])
        price_swing_lows = self._find_swing_lows(df['low'])
        rsi_swing_highs = self._find_swing_highs(rsi)
        rsi_swing_lows = self._find_swing_lows(rsi)
        
        divergences = []
        
        # Regular Bullish: Price lower low, RSI higher low
        divergences.extend(self._detect_regular_bullish(df, rsi, price_swing_lows, rsi_swing_lows))
        
        # Regular Bearish: Price higher high, RSI lower high
        divergences.extend(self._detect_regular_bearish(df, rsi, price_swing_highs, rsi_swing_highs))
        
        # Hidden Bullish: Price higher low, RSI lower low
        divergences.extend(self._detect_hidden_bullish(df, rsi, price_swing_lows, rsi_swing_lows))
        
        # Hidden Bearish: Price lower high, RSI higher high
        divergences.extend(self._detect_hidden_bearish(df, rsi, price_swing_highs, rsi_swing_highs))
        
        return divergences
    
    def _find_swing_highs(self, series: pd.Series, window: int = 5) -> List[Dict]:
        """Find swing highs in a series"""
        swings = []
        
        for i in range(window, len(series) - window):
            if i < window or i >= len(series) - window:
                continue
                
            current = series.iloc[i]
            left_max = series.iloc[i-window:i].max()
            right_max = series.iloc[i+1:i+window+1].max()
            
            if current > left_max and current > right_max:
                swings.append({
                    'idx': i,
                    'value': current,
                    'timestamp': series.index[i]
                })
        
        return swings
    
    def _find_swing_lows(self, series: pd.Series, window: int = 5) -> List[Dict]:
        """Find swing lows in a series"""
        swings = []
        
        for i in range(window, len(series) - window):
            if i < window or i >= len(series) - window:
                continue
                
            current = series.iloc[i]
            left_min = series.iloc[i-window:i].min()
            right_min = series.iloc[i+1:i+window+1].min()
            
            if current < left_min and current < right_min:
                swings.append({
                    'idx': i,
                    'value': current,
                    'timestamp': series.index[i]
                })
        
        return swings
    
    def _detect_regular_bullish(self, df: pd.DataFrame, rsi: pd.Series, 
                                 price_lows: List[Dict], rsi_lows: List[Dict]) -> List[DivergenceSignal]:
        """
        Regular Bullish Divergence
        Price makes lower low, RSI makes higher low
        """
        signals = []
        
        for i in range(1, len(price_lows)):
            p1 = price_lows[i-1]
            p2 = price_lows[i]
            
            # Price lower low
            if p2['value'] >= p1['value']:
                continue
                
            # Find corresponding RSI lows
            r1 = self._find_closest_swing(rsi_lows, p1['idx'], before=True)
            r2 = self._find_closest_swing(rsi_lows, p2['idx'], before=True)
            
            if r1 is None or r2 is None:
                continue
                
            # RSI higher low
            if r2['value'] <= r1['value']:
                continue
                
            # Valid divergence
            strength = self._calculate_strength(p1['value'], p2['value'], r1['value'], r2['value'])
            
            signals.append(DivergenceSignal(
                divergence_type='regular_bullish',
                start_idx=p1['idx'],
                end_idx=p2['idx'],
                start_price=p1['value'],
                end_price=p2['value'],
                start_rsi=r1['value'],
                end_rsi=r2['value'],
                strength=strength,
                detected_at=df.index[-1]
            ))
        
        return signals
    
    def _detect_regular_bearish(self, df: pd.DataFrame, rsi: pd.Series,
                                 price_highs: List[Dict], rsi_highs: List[Dict]) -> List[DivergenceSignal]:
        """
        Regular Bearish Divergence
        Price makes higher high, RSI makes lower high
        """
        signals = []
        
        for i in range(1, len(price_highs)):
            p1 = price_highs[i-1]
            p2 = price_highs[i]
            
            # Price higher high
            if p2['value'] <= p1['value']:
                continue
                
            # Find corresponding RSI highs
            r1 = self._find_closest_swing(rsi_highs, p1['idx'], before=True)
            r2 = self._find_closest_swing(rsi_highs, p2['idx'], before=True)
            
            if r1 is None or r2 is None:
                continue
                
            # RSI lower high
            if r2['value'] >= r1['value']:
                continue
                
            strength = self._calculate_strength(p1['value'], p2['value'], r1['value'], r2['value'])
            
            signals.append(DivergenceSignal(
                divergence_type='regular_bearish',
                start_idx=p1['idx'],
                end_idx=p2['idx'],
                start_price=p1['value'],
                end_price=p2['value'],
                start_rsi=r1['value'],
                end_rsi=r2['value'],
                strength=strength,
                detected_at=df.index[-1]
            ))
        
        return signals
    
    def _detect_hidden_bullish(self, df: pd.DataFrame, rsi: pd.Series,
                                price_lows: List[Dict], rsi_lows: List[Dict]) -> List[DivergenceSignal]:
        """
        Hidden Bullish Divergence
        Price makes higher low, RSI makes lower low
        """
        signals = []
        
        for i in range(1, len(price_lows)):
            p1 = price_lows[i-1]
            p2 = price_lows[i]
            
            # Price higher low
            if p2['value'] <= p1['value']:
                continue
                
            # Find corresponding RSI lows
            r1 = self._find_closest_swing(rsi_lows, p1['idx'], before=True)
            r2 = self._find_closest_swing(rsi_lows, p2['idx'], before=True)
            
            if r1 is None or r2 is None:
                continue
                
            # RSI lower low
            if r2['value'] >= r1['value']:
                continue
                
            strength = self._calculate_strength(p1['value'], p2['value'], r1['value'], r2['value'])
            
            signals.append(DivergenceSignal(
                divergence_type='hidden_bullish',
                start_idx=p1['idx'],
                end_idx=p2['idx'],
                start_price=p1['value'],
                end_price=p2['value'],
                start_rsi=r1['value'],
                end_rsi=r2['value'],
                strength=strength,
                detected_at=df.index[-1]
            ))
        
        return signals
    
    def _detect_hidden_bearish(self, df: pd.DataFrame, rsi: pd.Series,
                                price_highs: List[Dict], rsi_highs: List[Dict]) -> List[DivergenceSignal]:
        """
        Hidden Bearish Divergence
        Price makes lower high, RSI makes higher high
        """
        signals = []
        
        for i in range(1, len(price_highs)):
            p1 = price_highs[i-1]
            p2 = price_highs[i]
            
            # Price lower high
            if p2['value'] >= p1['value']:
                continue
                
            # Find corresponding RSI highs
            r1 = self._find_closest_swing(rsi_highs, p1['idx'], before=True)
            r2 = self._find_closest_swing(rsi_highs, p2['idx'], before=True)
            
            if r1 is None or r2 is None:
                continue
                
            # RSI higher high
            if r2['value'] <= r1['value']:
                continue
                
            strength = self._calculate_strength(p1['value'], p2['value'], r1['value'], r2['value'])
            
            signals.append(DivergenceSignal(
                divergence_type='hidden_bearish',
                start_idx=p1['idx'],
                end_idx=p2['idx'],
                start_price=p1['value'],
                end_price=p2['value'],
                start_rsi=r1['value'],
                end_rsi=r2['value'],
                strength=strength,
                detected_at=df.index[-1]
            ))
        
        return signals
    
    def _find_closest_swing(self, swings: List[Dict], target_idx: int, before: bool = True) -> Optional[Dict]:
        """Find closest swing point to target index"""
        if not swings:
            return None
            
        if before:
            # Find closest swing before target
            valid = [s for s in swings if s['idx'] <= target_idx]
            if not valid:
                return None
            return max(valid, key=lambda x: x['idx'])
        else:
            # Find closest swing after target
            valid = [s for s in swings if s['idx'] >= target_idx]
            if not valid:
                return None
            return min(valid, key=lambda x: x['idx'])
    
    def _calculate_strength(self, p1: float, p2: float, r1: float, r2: float) -> float:
        """Calculate divergence strength (0-1)"""
        price_diff_pct = abs(p2 - p1) / p1
        rsi_diff_pct = abs(r2 - r1) / max(r1, r2)
        
        # Stronger when both differences are significant
        strength = min(price_diff_pct * 50, 1.0) * min(rsi_diff_pct * 10, 1.0)
        return min(strength, 1.0)
    
    def get_divergence_for_pattern(self, df: pd.DataFrame, pattern_start: int, pattern_end: int) -> List[DivergenceSignal]:
        """Get divergences that align with a pattern"""
        all_divergences = self.detect_all(df)
        
        # Filter divergences that end near the pattern start
        aligned = []
        for div in all_divergences:
            if div.end_idx >= pattern_start - 10 and div.end_idx <= pattern_start + 5:
                aligned.append(div)
        
        return aligned
