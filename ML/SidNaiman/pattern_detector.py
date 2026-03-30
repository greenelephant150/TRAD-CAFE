"""
Pattern Detection Module for Sid Naiman's SID Method
Detects simple double top and double bottom patterns for confirmation
Following Sid's rules: patterns are confirmation only, not required
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Pattern:
    """Represents a detected pattern (double top/double bottom)"""
    
    def __init__(self,
                 pattern_type: str,  # 'double_top' or 'double_bottom'
                 start_idx: int,
                 end_idx: int,
                 left_price: float,
                 center_price: float,
                 right_price: float,
                 neckline_price: float,
                 candle_count: int,
                 confidence: float):
        
        self.pattern_type = pattern_type
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.left_price = left_price
        self.center_price = center_price
        self.right_price = right_price
        self.neckline_price = neckline_price
        self.candle_count = candle_count
        self.confidence = confidence
    
    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type,
            'neckline': self.neckline_price,
            'candle_count': self.candle_count,
            'confidence': self.confidence
        }


class PatternDetector:
    """
    Detects simple double top and double bottom patterns for Sid's SID Method
    Patterns are for confirmation only, not required for trades
    """
    
    def __init__(self, swing_window: int = 5):
        self.swing_window = swing_window
        self.logger = logging.getLogger(__name__)
    
    def find_swing_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows"""
        highs = df['high'].values
        lows = df['low'].values
        
        swing_highs = []
        swing_lows = []
        
        # Find swing highs
        for i in range(self.swing_window, len(df) - self.swing_window):
            if highs[i] == max(highs[i-self.swing_window:i+self.swing_window+1]):
                swing_highs.append(i)
        
        # Find swing lows
        for i in range(self.swing_window, len(df) - self.swing_window):
            if lows[i] == min(lows[i-self.swing_window:i+self.swing_window+1]):
                swing_lows.append(i)
        
        return swing_highs, swing_lows
    
    def detect_double_bottoms(self, df: pd.DataFrame, swing_lows: List[int]) -> List[Pattern]:
        """Detect double bottom (W) patterns"""
        patterns = []
        
        if len(swing_lows) < 2:
            return patterns
        
        for i in range(len(swing_lows) - 1):
            idx1 = swing_lows[i]
            idx2 = swing_lows[i + 1]
            
            # Need enough candles between
            if idx2 - idx1 < 5:
                continue
            
            # Get prices
            low1 = df['low'].iloc[idx1]
            low2 = df['low'].iloc[idx2]
            
            # Lows should be at similar level (within 2%)
            if abs(low1 - low2) / low1 > 0.02:
                continue
            
            # Find peak between lows
            peak_idx = df['high'].iloc[idx1:idx2 + 1].idxmax()
            peak_price = df.loc[peak_idx, 'high']
            
            # Peak should be at least 3% higher
            if peak_price < min(low1, low2) * 1.03:
                continue
            
            # Calculate confidence
            symmetry = 100 - (abs(low1 - low2) / low1 * 100)
            depth = (peak_price / min(low1, low2) - 1) * 100
            confidence = (symmetry * 0.5 + min(depth * 10, 50))
            
            pattern = Pattern(
                pattern_type='double_bottom',
                start_idx=idx1,
                end_idx=idx2,
                left_price=low1,
                center_price=peak_price,
                right_price=low2,
                neckline_price=peak_price,
                candle_count=idx2 - idx1 + 1,
                confidence=confidence
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def detect_double_tops(self, df: pd.DataFrame, swing_highs: List[int]) -> List[Pattern]:
        """Detect double top (M) patterns"""
        patterns = []
        
        if len(swing_highs) < 2:
            return patterns
        
        for i in range(len(swing_highs) - 1):
            idx1 = swing_highs[i]
            idx2 = swing_highs[i + 1]
            
            if idx2 - idx1 < 5:
                continue
            
            high1 = df['high'].iloc[idx1]
            high2 = df['high'].iloc[idx2]
            
            if abs(high1 - high2) / high1 > 0.02:
                continue
            
            valley_idx = df['low'].iloc[idx1:idx2 + 1].idxmin()
            valley_price = df.loc[valley_idx, 'low']
            
            if valley_price > min(high1, high2) * 0.97:
                continue
            
            # Calculate confidence
            symmetry = 100 - (abs(high1 - high2) / high1 * 100)
            depth = (min(high1, high2) / valley_price - 1) * 100
            confidence = (symmetry * 0.5 + min(depth * 10, 50))
            
            pattern = Pattern(
                pattern_type='double_top',
                start_idx=idx1,
                end_idx=idx2,
                left_price=high1,
                center_price=valley_price,
                right_price=high2,
                neckline_price=valley_price,
                candle_count=idx2 - idx1 + 1,
                confidence=confidence
            )
            
            patterns.append(pattern)
        
        return patterns
    
    def detect_patterns(self, df: pd.DataFrame, min_confidence: float = 50.0) -> List[Pattern]:
        """Detect all patterns"""
        if df.empty or len(df) < 30:
            return []
        
        swing_highs, swing_lows = self.find_swing_points(df)
        
        double_bottoms = self.detect_double_bottoms(df, swing_lows)
        double_tops = self.detect_double_tops(df, swing_highs)
        
        all_patterns = double_bottoms + double_tops
        
        # Filter by confidence
        valid = [p for p in all_patterns if p.confidence >= min_confidence]
        valid.sort(key=lambda x: x.confidence, reverse=True)
        
        return valid
    
    def get_best_pattern(self, df: pd.DataFrame) -> Optional[Pattern]:
        """Get highest confidence pattern"""
        patterns = self.detect_patterns(df)
        return patterns[0] if patterns else None