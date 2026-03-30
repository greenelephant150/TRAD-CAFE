"""
Pattern Detection Module for Simon Pullen's Trading System
Detects M-Top and W-Bottom patterns following Academy rules:
- Minimum 7 candles for valid patterns
- Entry on break of neckline (close of candle)
- Stop loss behind the pattern
- Rogue wick identification
- Room for completion checking
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from scipy.signal import argrelextrema

logger = logging.getLogger(__name__)


class Pattern:
    """Represents a detected M or W pattern"""
    
    def __init__(self,
                 pattern_type: str,  # 'M-Top' or 'W-Bottom'
                 start_idx: int,
                 end_idx: int,
                 left_price: float,
                 center_price: float,
                 right_price: float,
                 neckline_price: float,
                 entry_price: float,
                 stop_loss: float,
                 take_profit: float,
                 candle_count: int,
                 wonkiness: float,
                 confidence: float):
        
        self.pattern_type = pattern_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.left_price = left_price
        self.center_price = center_price
        self.right_price = right_price
        self.neckline_price = neckline_price
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.candle_count = candle_count
        self.wonkiness = wonkiness
        self.confidence = confidence
        self.valid = candle_count >= 7 and wonkiness < 5.0
    
    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type,
            'neckline': self.neckline_price,
            'entry': self.entry_price,
            'stop': self.stop_loss,
            'target': self.take_profit,
            'candle_count': self.candle_count,
            'wonkiness': self.wonkiness,
            'confidence': self.confidence,
            'valid': self.valid
        }
    
    def has_room_for_completion(self, df: pd.DataFrame) -> bool:
        """Check if there's enough price movement for target"""
        if self.end_idx >= len(df) - 5:
            return False
        
        future_prices = df['close'].iloc[self.end_idx:self.end_idx+10]
        
        if self.pattern_type == 'W-Bottom':
            return future_prices.max() >= self.take_profit
        else:  # M-Top
            return future_prices.min() <= self.take_profit


class PatternDetector:
    """Detects M and W patterns following Academy rules"""
    
    def __init__(self, swing_window: int = 5):
        self.swing_window = swing_window
        self.logger = logging.getLogger(__name__)
    
    def find_swing_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows"""
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = argrelextrema(highs, np.greater, order=self.swing_window)[0]
        swing_lows = argrelextrema(lows, np.less, order=self.swing_window)[0]
        
        return list(swing_highs), list(swing_lows)
    
    def identify_rogue_wicks(self, df: pd.DataFrame, indices: List[int]) -> List[int]:
        """
        Identify rogue wicks that should be ignored in pattern drawing
        From sessions: wicks that stick out and aren't part of pattern structure
        """
        rogue = []
        
        for idx in indices:
            if idx <= 0 or idx >= len(df)-1:
                continue
            
            candle = df.iloc[idx]
            prev_close = df.iloc[idx-1]['close']
            next_close = df.iloc[idx+1]['close'] if idx+1 < len(df) else prev_close
            
            body = abs(candle['close'] - candle['open'])
            wick_upper = candle['high'] - max(candle['close'], candle['open'])
            wick_lower = min(candle['close'], candle['open']) - candle['low']
            
            # Check if wick is isolated and extreme
            if wick_upper > body * 3:
                if candle['high'] > max(prev_close, next_close) * 1.02:
                    rogue.append(idx)
            elif wick_lower > body * 3:
                if candle['low'] < min(prev_close, next_close) * 0.98:
                    rogue.append(idx)
        
        return rogue
    
    def calculate_wonkiness(self, left: float, center: float, right: float) -> float:
        """Calculate pattern wonkiness (distortion)"""
        if left == 0 or right == 0:
            return 100.0
        
        symmetry = abs(left - right) / ((left + right) / 2) * 100
        return symmetry
    
    def detect_w_bottoms(self, df: pd.DataFrame, swing_lows: List[int]) -> List[Pattern]:
        """Detect W-Bottom (double bottom) patterns"""
        patterns = []
        
        if len(swing_lows) < 2:
            return patterns
        
        for i in range(len(swing_lows) - 1):
            idx1 = swing_lows[i]
            idx2 = swing_lows[i+1]
            
            # Minimum candle count (7)
            if idx2 - idx1 < 6:
                continue
            
            # Get prices
            low1 = df['low'].iloc[idx1]
            low2 = df['low'].iloc[idx2]
            
            # Lows should be at similar level (within 2%)
            if abs(low1 - low2) / low1 > 0.02:
                continue
            
            # Find peak between lows
            peak_idx = df['high'].iloc[idx1:idx2].idxmax()
            peak_price = df.loc[peak_idx, 'high']
            peak_loc = df.index.get_loc(peak_idx)
            
            # Peak should be at least 3% higher
            if peak_price < min(low1, low2) * 1.03:
                continue
            
            # Calculate wonkiness
            wonkiness = self.calculate_wonkiness(low1, peak_price, low2)
            
            # Identify rogue wicks
            pattern_indices = list(range(idx1, idx2+1))
            rogue = self.identify_rogue_wicks(df, pattern_indices)
            
            # Adjust stop for rogue wicks
            if rogue:
                non_rogue_lows = [df['low'].iloc[i] for i in pattern_indices if i not in rogue]
                stop_price = min(non_rogue_lows) * 0.999 if non_rogue_lows else min(low1, low2) * 0.999
            else:
                stop_price = min(low1, low2) * 0.999
            
            # Entry on break of neckline (close)
            entry_price = peak_price * 1.001
            
            # Target (measured move)
            avg_low = (low1 + low2) / 2
            target_price = entry_price + (peak_price - avg_low) * 2
            
            # Confidence calculation
            volume_conf = self._volume_confidence(df, idx1, idx2)
            pattern_conf = 100 - wonkiness
            confidence = (volume_conf + pattern_conf) / 2
            
            pattern = Pattern(
                pattern_type='W-Bottom',
                start_idx=idx1,
                end_idx=idx2,
                left_price=low1,
                center_price=peak_price,
                right_price=low2,
                neckline_price=peak_price,
                entry_price=entry_price,
                stop_loss=stop_price,
                take_profit=target_price,
                candle_count=idx2 - idx1 + 1,
                wonkiness=wonkiness,
                confidence=confidence
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def detect_m_tops(self, df: pd.DataFrame, swing_highs: List[int]) -> List[Pattern]:
        """Detect M-Top (double top) patterns"""
        patterns = []
        
        if len(swing_highs) < 2:
            return patterns
        
        for i in range(len(swing_highs) - 1):
            idx1 = swing_highs[i]
            idx2 = swing_highs[i+1]
            
            if idx2 - idx1 < 6:
                continue
            
            high1 = df['high'].iloc[idx1]
            high2 = df['high'].iloc[idx2]
            
            if abs(high1 - high2) / high1 > 0.02:
                continue
            
            valley_idx = df['low'].iloc[idx1:idx2].idxmin()
            valley_price = df.loc[valley_idx, 'low']
            valley_loc = df.index.get_loc(valley_idx)
            
            if valley_price > min(high1, high2) * 0.97:
                continue
            
            wonkiness = self.calculate_wonkiness(high1, valley_price, high2)
            
            pattern_indices = list(range(idx1, idx2+1))
            rogue = self.identify_rogue_wicks(df, pattern_indices)
            
            if rogue:
                non_rogue_highs = [df['high'].iloc[i] for i in pattern_indices if i not in rogue]
                stop_price = max(non_rogue_highs) * 1.001 if non_rogue_highs else max(high1, high2) * 1.001
            else:
                stop_price = max(high1, high2) * 1.001
            
            entry_price = valley_price * 0.999
            avg_high = (high1 + high2) / 2
            target_price = entry_price - (avg_high - valley_price) * 2
            
            volume_conf = self._volume_confidence(df, idx1, idx2)
            pattern_conf = 100 - wonkiness
            confidence = (volume_conf + pattern_conf) / 2
            
            pattern = Pattern(
                pattern_type='M-Top',
                start_idx=idx1,
                end_idx=idx2,
                left_price=high1,
                center_price=valley_price,
                right_price=high2,
                neckline_price=valley_price,
                entry_price=entry_price,
                stop_loss=stop_price,
                take_profit=target_price,
                candle_count=idx2 - idx1 + 1,
                wonkiness=wonkiness,
                confidence=confidence
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def _volume_confidence(self, df: pd.DataFrame, start: int, end: int) -> float:
        """Calculate confidence based on volume"""
        try:
            volume = df['volume'].iloc[start:end+1]
            if len(volume) > 1:
                trend = (volume.iloc[-1] / volume.iloc[0]) - 1
                if trend > 0.2:
                    return 90.0
                elif trend > 0.1:
                    return 70.0
                elif trend > 0:
                    return 50.0
            return 50.0
        except:
            return 50.0
    
    def detect_patterns(self, df: pd.DataFrame, min_confidence: float = 50.0) -> List[Pattern]:
        """Detect all patterns"""
        if df.empty or len(df) < 30:
            return []
        
        swing_highs, swing_lows = self.find_swing_points(df)
        
        w_bottoms = self.detect_w_bottoms(df, swing_lows)
        m_tops = self.detect_m_tops(df, swing_highs)
        
        all_patterns = w_bottoms + m_tops
        
        # Filter by validity and confidence
        valid = [p for p in all_patterns if p.valid and p.confidence >= min_confidence]
        valid.sort(key=lambda x: x.confidence, reverse=True)
        
        return valid
    
    def get_best_pattern(self, df: pd.DataFrame) -> Optional[Pattern]:
        """Get highest confidence pattern"""
        patterns = self.detect_patterns(df)
        return patterns[0] if patterns else None