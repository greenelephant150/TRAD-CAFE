"""
Entry candle analysis for Head and Shoulders patterns
Simon's preferred entries: pin bars, engulfing, tweezer patterns
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EntryCandleType(Enum):
    PIN = "pin"
    ENGULFING = "engulfing"
    TWEEZER = "tweezer"
    NONE = "none"


class EntryCandleAnalyzer:
    """
    Analyzes candles for valid entry signals
    Simon's entry candles: pin bars, engulfing, tweezer patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pin_threshold = config.get('pin_threshold', 0.66)  # Wick > 66% of range
        self.engulfing_required = config.get('engulfing_required', True)
        
    def analyze(self, df: pd.DataFrame, idx: int, pattern_type: str, direction: str) -> Tuple[EntryCandleType, float]:
        """
        Analyze candle at idx for entry signals
        Returns (candle_type, confidence)
        """
        if idx >= len(df):
            return EntryCandleType.NONE, 0.0
            
        candle = df.iloc[idx]
        
        # Check for pin bar
        pin_type, pin_conf = self._is_pin_bar(candle, direction)
        if pin_type != EntryCandleType.NONE:
            return pin_type, pin_conf
            
        # Check for engulfing
        if idx > 0:
            prev_candle = df.iloc[idx - 1]
            engulf_type, engulf_conf = self._is_engulfing(candle, prev_candle, direction)
            if engulf_type != EntryCandleType.NONE:
                return engulf_type, engulf_conf
        
        # Check for tweezer (needs current and next candle)
        if idx < len(df) - 1:
            next_candle = df.iloc[idx + 1]
            tweezer_type, tweezer_conf = self._is_tweezer(candle, next_candle, direction)
            if tweezer_type != EntryCandleType.NONE:
                return tweezer_type, tweezer_conf
        
        return EntryCandleType.NONE, 0.0
    
    def _is_pin_bar(self, candle, direction: str) -> Tuple[EntryCandleType, float]:
        """
        Check if candle is a pin bar
        Pin bar: long wick, small body, wick > 2/3 of range
        """
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return EntryCandleType.NONE, 0.0
            
        wick_size = total_range - body_size
        wick_ratio = wick_size / total_range
        
        # Pin bar has wick > threshold
        if wick_ratio < self.pin_threshold:
            return EntryCandleType.NONE, 0.0
            
        # Check direction
        if direction == 'long':
            # For long entry, pin should have lower wick (rejection of lows)
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            lower_wick_ratio = lower_wick / total_range
            if lower_wick_ratio > 0.5:
                confidence = min(wick_ratio * 1.2, 1.0)
                return EntryCandleType.PIN, confidence
        else:  # short
            # For short entry, pin should have upper wick (rejection of highs)
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            upper_wick_ratio = upper_wick / total_range
            if upper_wick_ratio > 0.5:
                confidence = min(wick_ratio * 1.2, 1.0)
                return EntryCandleType.PIN, confidence
        
        return EntryCandleType.NONE, 0.0
    
    def _is_engulfing(self, candle, prev_candle, direction: str) -> Tuple[EntryCandleType, float]:
        """
        Check if candle engulfs previous candle
        """
        if direction == 'long':
            # Bullish engulfing: green candle engulfs previous red
            if (candle['close'] > candle['open'] and  # Green candle
                prev_candle['close'] < prev_candle['open'] and  # Previous red
                candle['open'] < prev_candle['close'] and
                candle['close'] > prev_candle['open']):
                
                # Calculate engulf percentage
                engulf_range = candle['close'] - candle['open']
                prev_range = prev_candle['open'] - prev_candle['close']
                engulf_ratio = engulf_range / prev_range if prev_range > 0 else 2.0
                
                confidence = min(engulf_ratio * 0.5, 1.0)
                return EntryCandleType.ENGULFING, confidence
                
        else:  # short
            # Bearish engulfing: red candle engulfs previous green
            if (candle['close'] < candle['open'] and  # Red candle
                prev_candle['close'] > prev_candle['open'] and  # Previous green
                candle['open'] > prev_candle['close'] and
                candle['close'] < prev_candle['open']):
                
                engulf_range = candle['open'] - candle['close']
                prev_range = prev_candle['close'] - prev_candle['open']
                engulf_ratio = engulf_range / prev_range if prev_range > 0 else 2.0
                
                confidence = min(engulf_ratio * 0.5, 1.0)
                return EntryCandleType.ENGULFING, confidence
        
        return EntryCandleType.NONE, 0.0
    
    def _is_tweezer(self, candle1, candle2, direction: str) -> Tuple[EntryCandleType, float]:
        """
        Check if two candles form a tweezer pattern
        Tweezer top: same highs (for short entry)
        Tweezer bottom: same lows (for long entry)
        """
        if direction == 'long':
            # Tweezer bottom - same lows
            low_diff = abs(candle1['low'] - candle2['low']) / candle1['low']
            if low_diff < 0.001:  # Very close lows
                # Both should close near highs
                if (candle1['close'] > candle1['open'] and 
                    candle2['close'] > candle2['open']):
                    return EntryCandleType.TWEEZER, 0.8
        else:  # short
            # Tweezer top - same highs
            high_diff = abs(candle1['high'] - candle2['high']) / candle1['high']
            if high_diff < 0.001:  # Very close highs
                # Both should close near lows
                if (candle1['close'] < candle1['open'] and 
                    candle2['close'] < candle2['open']):
                    return EntryCandleType.TWEEZER, 0.8
        
        return EntryCandleType.NONE, 0.0
    
    def find_best_entry_candle(self, df: pd.DataFrame, start_idx: int, 
                               direction: str, pattern_type: str,
                               max_lookback: int = 10) -> Tuple[Optional[int], EntryCandleType, float]:
        """
        Find the best entry candle within lookback period
        """
        best_idx = None
        best_type = EntryCandleType.NONE
        best_conf = 0.0
        
        for i in range(start_idx, min(start_idx + max_lookback, len(df))):
            candle_type, confidence = self.analyze(df, i, pattern_type, direction)
            if confidence > best_conf:
                best_conf = confidence
                best_type = candle_type
                best_idx = i
                
            # If we found a good entry, stop looking
            if confidence > 0.7:
                break
        
        return best_idx, best_type, best_conf
