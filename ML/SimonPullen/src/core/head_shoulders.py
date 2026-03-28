"""
Simon Pullen Head and Shoulders Pattern Detection
Rules:
- Minimum 30 candles, maximum 120 candles in pattern
- Head flanked by two lower shoulders
- Neckline based on bodies only, can slope
- Process: Break → Close → Retest → Entry Candle → Enter
- Entry candles: pin bars, engulfing, tweezer patterns
- Three stop loss options with different R:R profiles
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class StopLossType(Enum):
    CONSERVATIVE = "behind_head"
    MODERATE = "behind_shoulder"
    AGGRESSIVE = "behind_entry"


@dataclass
class HeadShouldersPattern:
    """Represents a Head and Shoulders pattern (normal or inverted)"""
    pattern_type: str  # 'normal' or 'inverted'
    instrument: str
    timeframe: str
    left_shoulder_idx: int
    left_shoulder_price: float
    head_idx: int
    head_price: float
    right_shoulder_idx: int
    right_shoulder_price: float
    neckline_start_idx: int
    neckline_start_price: float
    neckline_end_idx: int
    neckline_end_price: float
    neckline_slope: float
    break_idx: Optional[int] = None
    break_price: Optional[float] = None
    retest_idx: Optional[int] = None
    retest_price: Optional[float] = None
    entry_candle_idx: Optional[int] = None
    entry_candle_type: Optional[str] = None  # 'pin', 'engulfing', 'tweezer'
    entry_price: Optional[float] = None
    stop_loss_options: Dict[str, float] = field(default_factory=dict)
    take_profit_price: Optional[float] = None
    candle_count: int = 0
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    confluence_score: float = 0.0
    detected_at: Optional[pd.Timestamp] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'pattern_type': self.pattern_type,
            'instrument': self.instrument,
            'timeframe': self.timeframe,
            'left_shoulder_idx': self.left_shoulder_idx,
            'left_shoulder_price': self.left_shoulder_price,
            'head_idx': self.head_idx,
            'head_price': self.head_price,
            'right_shoulder_idx': self.right_shoulder_idx,
            'right_shoulder_price': self.right_shoulder_price,
            'neckline_start_idx': self.neckline_start_idx,
            'neckline_start_price': self.neckline_start_price,
            'neckline_end_idx': self.neckline_end_idx,
            'neckline_end_price': self.neckline_end_price,
            'neckline_slope': self.neckline_slope,
            'break_idx': self.break_idx,
            'break_price': self.break_price,
            'retest_idx': self.retest_idx,
            'retest_price': self.retest_price,
            'entry_candle_idx': self.entry_candle_idx,
            'entry_candle_type': self.entry_candle_type,
            'entry_price': self.entry_price,
            'stop_loss_options': self.stop_loss_options,
            'take_profit_price': self.take_profit_price,
            'candle_count': self.candle_count,
            'valid': self.valid,
            'validation_errors': self.validation_errors,
            'confluence_score': self.confluence_score,
            'detected_at': str(self.detected_at) if self.detected_at else None
        }


class EntryCandleType(Enum):
    PIN = "pin"
    ENGULFING = "engulfing"
    TWEEZER = "tweezer"


class HeadShouldersDetector:
    """
    Detects Head and Shoulders patterns following Simon Pullen's strict rules
    
    Key rules:
    1. Head must be higher than shoulders (normal) or lower (inverted)
    2. 30-120 candles from left shoulder start to right shoulder end
    3. Neckline based on bodies, can slope
    4. Must have break + close + retest + entry candle
    5. Entry candles: pin, engulfing, tweezer
    6. Three stop loss options with different risk profiles
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_candles = config.get('min_hs_candles', 30)
        self.max_candles = config.get('max_hs_candles', 120)
        self.shoulder_similarity_tolerance = config.get('shoulder_similarity_tolerance', 0.03)
        self.require_break_close = config.get('require_break_close', True)
        self.require_retest = config.get('require_retest', True)
        self.require_entry_candle = config.get('require_entry_candle', True)
        
    def detect_normal(self, df: pd.DataFrame, instrument: str = "", timeframe: str = "1h") -> List[HeadShouldersPattern]:
        """
        Detect normal Head and Shoulders (bearish reversal)
        Pattern: left shoulder < head > right shoulder
        """
        peaks = self._find_peaks(df)
        patterns = []
        
        # Need at least 3 peaks for H&S
        for i in range(1, len(peaks) - 1):
            left = peaks[i-1]
            head = peaks[i]
            right = peaks[i+1]
            
            # Check head is highest
            if head['price'] <= left['price'] or head['price'] <= right['price']:
                continue
                
            # Shoulders should be similar in height
            if not self._shoulders_similar(left, right):
                continue
                
            # Get integer positions
            left_pos = df.index.get_loc(left['timestamp'])
            head_pos = df.index.get_loc(head['timestamp'])
            right_pos = df.index.get_loc(right['timestamp'])
            
            # Check candle count
            candle_count = right_pos - left_pos + 1
            if not (self.min_candles <= candle_count <= self.max_candles):
                continue
                
            # Find neckline (bodies between left shoulder-head and head-right shoulder)
            neckline = self._find_neckline(df, left, head, right, left_pos, head_pos, right_pos)
            if neckline is None:
                continue
                
            # Check if pattern completed with break, retest, entry
            pattern = self._validate_pattern(
                df, left, head, right, left_pos, head_pos, right_pos, neckline, 'normal', instrument, timeframe
            )
            
            if pattern and pattern.valid:
                patterns.append(pattern)
                
        return patterns
    
    def detect_inverted(self, df: pd.DataFrame, instrument: str = "", timeframe: str = "1h") -> List[HeadShouldersPattern]:
        """
        Detect inverted Head and Shoulders (bullish reversal)
        Pattern: left shoulder > head < right shoulder
        """
        troughs = self._find_troughs(df)
        patterns = []
        
        for i in range(1, len(troughs) - 1):
            left = troughs[i-1]
            head = troughs[i]
            right = troughs[i+1]
            
            # Check head is lowest
            if head['price'] >= left['price'] or head['price'] >= right['price']:
                continue
                
            # Shoulders should be similar in height
            if not self._shoulders_similar(left, right):
                continue
                
            # Get integer positions
            left_pos = df.index.get_loc(left['timestamp'])
            head_pos = df.index.get_loc(head['timestamp'])
            right_pos = df.index.get_loc(right['timestamp'])
            
            # Check candle count
            candle_count = right_pos - left_pos + 1
            if not (self.min_candles <= candle_count <= self.max_candles):
                continue
                
            # Find neckline (bodies between left shoulder-head and head-right shoulder)
            neckline = self._find_neckline_inverted(df, left, head, right, left_pos, head_pos, right_pos)
            if neckline is None:
                continue
                
            # Check if pattern completed
            pattern = self._validate_pattern(
                df, left, head, right, left_pos, head_pos, right_pos, neckline, 'inverted', instrument, timeframe
            )
            
            if pattern and pattern.valid:
                patterns.append(pattern)
                
        return patterns
    
    def _find_peaks(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find local maxima (peaks) in price"""
        peaks = []
        
        for i in range(window, len(df) - window):
            current_high = df.iloc[i]['high']
            
            left_max = df.iloc[i-window:i]['high'].max()
            right_max = df.iloc[i+1:i+window+1]['high'].max()
            
            if current_high > left_max and current_high > right_max:
                peaks.append({
                    'idx': i,
                    'price': current_high,
                    'timestamp': df.index[i],
                    'low': df.iloc[i]['low']
                })
        
        return peaks
    
    def _find_troughs(self, df: pd.DataFrame, window: int = 5) -> List[Dict]:
        """Find local minima (troughs) in price"""
        troughs = []
        
        for i in range(window, len(df) - window):
            current_low = df.iloc[i]['low']
            
            left_min = df.iloc[i-window:i]['low'].min()
            right_min = df.iloc[i+1:i+window+1]['low'].min()
            
            if current_low < left_min and current_low < right_min:
                troughs.append({
                    'idx': i,
                    'price': current_low,
                    'timestamp': df.index[i],
                    'high': df.iloc[i]['high']
                })
        
        return troughs
    
    def _shoulders_similar(self, left: Dict, right: Dict) -> bool:
        """Check if shoulders are at similar levels"""
        price_diff_pct = abs(right['price'] - left['price']) / left['price']
        return price_diff_pct <= self.shoulder_similarity_tolerance
    
    def _find_neckline(self, df: pd.DataFrame, left: Dict, head: Dict, right: Dict,
                       left_pos: int, head_pos: int, right_pos: int) -> Optional[Dict]:
        """
        Find neckline for normal H&S
        Neckline connects the lows between left shoulder-head and head-right shoulder
        Uses bodies only, not wicks
        """
        # Find lowest point between left shoulder and head (using lows for bodies)
        between_left_head = df.iloc[left_pos:head_pos+1]
        left_neck_idx = between_left_head['low'].idxmin()
        left_neck = df.loc[left_neck_idx]
        
        # Find lowest point between head and right shoulder
        between_head_right = df.iloc[head_pos:right_pos+1]
        right_neck_idx = between_head_right['low'].idxmin()
        right_neck = df.loc[right_neck_idx]
        
        # Convert to integer positions
        left_neck_pos = df.index.get_loc(left_neck_idx)
        right_neck_pos = df.index.get_loc(right_neck_idx)
        
        # Check if neckline cuts through any bodies (invalid if it does)
        if self._neckline_cuts_bodies(df, left_neck, right_neck, left_neck_pos, right_neck_pos, 'normal'):
            return None
            
        return {
            'start_idx': left_neck_pos,
            'start_price': left_neck['low'],
            'end_idx': right_neck_pos,
            'end_price': right_neck['low']
        }
    
    def _find_neckline_inverted(self, df: pd.DataFrame, left: Dict, head: Dict, right: Dict,
                                 left_pos: int, head_pos: int, right_pos: int) -> Optional[Dict]:
        """
        Find neckline for inverted H&S
        Neckline connects the highs between left shoulder-head and head-right shoulder
        """
        # Find highest point between left shoulder and head (using highs for bodies)
        between_left_head = df.iloc[left_pos:head_pos+1]
        left_neck_idx = between_left_head['high'].idxmax()
        left_neck = df.loc[left_neck_idx]
        
        # Find highest point between head and right shoulder
        between_head_right = df.iloc[head_pos:right_pos+1]
        right_neck_idx = between_head_right['high'].idxmax()
        right_neck = df.loc[right_neck_idx]
        
        # Convert to integer positions
        left_neck_pos = df.index.get_loc(left_neck_idx)
        right_neck_pos = df.index.get_loc(right_neck_idx)
        
        # Check if neckline cuts through any bodies
        if self._neckline_cuts_bodies(df, left_neck, right_neck, left_neck_pos, right_neck_pos, 'inverted'):
            return None
            
        return {
            'start_idx': left_neck_pos,
            'start_price': left_neck['high'],
            'end_idx': right_neck_pos,
            'end_price': right_neck['high']
        }
    
    def _neckline_cuts_bodies(self, df: pd.DataFrame, neck1, neck2, 
                              neck1_pos: int, neck2_pos: int, pattern_type: str) -> bool:
        """
        Check if neckline cuts through any candle bodies
        Critical Simon rule: Neckline must not intersect bodies
        """
        # Calculate neckline slope (using integer positions)
        if pattern_type == 'normal':
            slope = (neck2['low'] - neck1['low']) / (neck2_pos - neck1_pos)
        else:
            slope = (neck2['high'] - neck1['high']) / (neck2_pos - neck1_pos)
        
        # Check each candle between neck points
        for i in range(neck1_pos + 1, neck2_pos):
            if i >= len(df):
                break
            candle = df.iloc[i]
            
            # Calculate neckline price at this index
            if pattern_type == 'normal':
                neck_price = neck1['low'] + slope * (i - neck1_pos)
                # Check if neckline goes through body (between low and high)
                if neck_price < candle['high'] and neck_price > candle['low']:
                    return True
            else:
                neck_price = neck1['high'] + slope * (i - neck1_pos)
                if neck_price < candle['high'] and neck_price > candle['low']:
                    return True
        
        return False
    
    def _check_break_close(self, df: pd.DataFrame, neckline: Dict, pattern_type: str) -> Optional[int]:
        """
        Check for break and close beyond neckline
        Returns index of break candle if valid
        """
        for i in range(neckline['end_idx'] + 1, min(neckline['end_idx'] + 20, len(df))):
            candle = df.iloc[i]
            
            if pattern_type == 'normal':
                # For normal H&S, break below neckline
                if candle['low'] < neckline['end_price']:
                    # Check if it closed below
                    if candle['close'] < neckline['end_price']:
                        return i
            else:
                # For inverted H&S, break above neckline
                if candle['high'] > neckline['end_price']:
                    if candle['close'] > neckline['end_price']:
                        return i
        
        return None
    
    def _check_retest(self, df: pd.DataFrame, break_idx: int, neckline: Dict, pattern_type: str) -> Optional[int]:
        """
        Check for retest of neckline after break
        Simon: Price must come back to neckline after break
        """
        for i in range(break_idx + 1, min(break_idx + 15, len(df))):
            candle = df.iloc[i]
            
            if pattern_type == 'normal':
                # Retest from below - price comes up to neckline
                if candle['high'] >= neckline['end_price'] and candle['close'] < neckline['end_price']:
                    return i
            else:
                # Retest from above - price comes down to neckline
                if candle['low'] <= neckline['end_price'] and candle['close'] > neckline['end_price']:
                    return i
        
        return None
    
    def _find_entry_candle(self, df: pd.DataFrame, retest_idx: int, neckline: Dict, pattern_type: str) -> Tuple[Optional[int], Optional[str]]:
        """
        Find entry candle after retest
        Simon's preferred entries: pin bars, engulfing, tweezer patterns
        """
        for i in range(retest_idx + 1, min(retest_idx + 10, len(df))):
            candle = df.iloc[i]
            
            # Check for pin bar
            if self._is_pin_bar(candle, pattern_type):
                return i, 'pin'
                
            # Check for engulfing
            if i > 0:
                prev_candle = df.iloc[i-1]
                if self._is_engulfing(candle, prev_candle, pattern_type):
                    return i, 'engulfing'
                    
            # Check for tweezer (needs 2 candles)
            if i > 0 and i < len(df) - 1:
                next_candle = df.iloc[i+1]
                if self._is_tweezer(candle, next_candle, pattern_type):
                    return i, 'tweezer'
        
        return None, None
    
    def _is_pin_bar(self, candle, pattern_type: str) -> bool:
        """Check if candle is a pin bar (long wick, small body)"""
        body_size = abs(candle['close'] - candle['open'])
        total_range = candle['high'] - candle['low']
        
        if total_range == 0:
            return False
            
        wick_ratio = (total_range - body_size) / total_range
        
        # Pin bar has wick > 2/3 of range
        if wick_ratio < 0.66:
            return False
            
        # Check direction
        if pattern_type == 'normal':  # For normal H&S, we want bearish pin (upper wick)
            upper_wick = candle['high'] - max(candle['open'], candle['close'])
            return upper_wick > body_size
        else:  # For inverted H&S, we want bullish pin (lower wick)
            lower_wick = min(candle['open'], candle['close']) - candle['low']
            return lower_wick > body_size
    
    def _is_engulfing(self, candle, prev_candle, pattern_type: str) -> bool:
        """Check if candle engulfs previous candle"""
        if pattern_type == 'normal':
            # Bearish engulfing
            return (candle['open'] > prev_candle['close'] and 
                    candle['close'] < prev_candle['open'] and
                    candle['high'] > prev_candle['high'] and
                    candle['low'] < prev_candle['low'])
        else:
            # Bullish engulfing
            return (candle['open'] < prev_candle['close'] and 
                    candle['close'] > prev_candle['open'] and
                    candle['high'] > prev_candle['high'] and
                    candle['low'] < prev_candle['low'])
    
    def _is_tweezer(self, candle1, candle2, pattern_type: str) -> bool:
        """Check if two candles form a tweezer pattern"""
        if pattern_type == 'normal':
            # Tweezer top - same highs
            high_diff = abs(candle1['high'] - candle2['high']) / candle1['high']
            return high_diff < 0.001 and candle1['close'] < candle1['open'] and candle2['close'] < candle2['open']
        else:
            # Tweezer bottom - same lows
            low_diff = abs(candle1['low'] - candle2['low']) / candle1['low']
            return low_diff < 0.001 and candle1['close'] > candle1['open'] and candle2['close'] > candle2['open']
    
    def _calculate_stop_losses(self, df: pd.DataFrame, pattern: HeadShouldersPattern) -> Dict[str, float]:
        """Calculate three stop loss options"""
        stops = {}
        
        # Conservative: behind head
        if pattern.pattern_type == 'normal':
            stops[StopLossType.CONSERVATIVE.value] = pattern.head_price + (pattern.head_price * 0.002)  # 0.2% buffer
        else:
            stops[StopLossType.CONSERVATIVE.value] = pattern.head_price - (pattern.head_price * 0.002)
        
        # Moderate: behind right shoulder
        if pattern.entry_candle_idx is not None:
            if pattern.pattern_type == 'normal':
                stops[StopLossType.MODERATE.value] = pattern.right_shoulder_price + (pattern.right_shoulder_price * 0.001)
            else:
                stops[StopLossType.MODERATE.value] = pattern.right_shoulder_price - (pattern.right_shoulder_price * 0.001)
        
        # Aggressive: behind entry candle
        if pattern.entry_candle_idx is not None:
            entry_candle = df.iloc[pattern.entry_candle_idx]
            if pattern.pattern_type == 'normal':
                stops[StopLossType.AGGRESSIVE.value] = entry_candle['high'] + (entry_candle['high'] * 0.0005)
            else:
                stops[StopLossType.AGGRESSIVE.value] = entry_candle['low'] - (entry_candle['low'] * 0.0005)
        
        return stops
    
    def _calculate_take_profit(self, df: pd.DataFrame, pattern: HeadShouldersPattern, neckline: Dict) -> float:
        """
        Calculate take profit (pattern completion level)
        Distance from neckline to head, projected beyond entry
        """
        if pattern.pattern_type == 'normal':
            head_to_neckline = pattern.head_price - neckline['end_price']
            return pattern.entry_price - head_to_neckline if pattern.entry_price else 0
        else:
            head_to_neckline = neckline['end_price'] - pattern.head_price
            return pattern.entry_price + head_to_neckline if pattern.entry_price else 0
    
    def _validate_pattern(self, df, left, head, right, left_pos, head_pos, right_pos, 
                          neckline, pattern_type, instrument, timeframe) -> Optional[HeadShouldersPattern]:
        """Validate complete H&S pattern with all rules"""
        validation_errors = []
        
        # Create pattern object
        pattern = HeadShouldersPattern(
            pattern_type=pattern_type,
            instrument=instrument,
            timeframe=timeframe,
            left_shoulder_idx=left['idx'],
            left_shoulder_price=left['price'],
            head_idx=head['idx'],
            head_price=head['price'],
            right_shoulder_idx=right['idx'],
            right_shoulder_price=right['price'],
            neckline_start_idx=neckline['start_idx'],
            neckline_start_price=neckline['start_price'],
            neckline_end_idx=neckline['end_idx'],
            neckline_end_price=neckline['end_price'],
            neckline_slope=(neckline['end_price'] - neckline['start_price']) / (neckline['end_idx'] - neckline['start_idx']) if neckline['end_idx'] != neckline['start_idx'] else 0,
            candle_count=right_pos - left_pos + 1,
            detected_at=df.index[-1] if len(df) > 0 else None
        )
        
        # Initialize variables
        break_idx = None
        retest_idx = None
        entry_idx = None
        entry_type = None
        
        # Check for break and close
        break_idx = self._check_break_close(df, neckline, pattern_type)
        if break_idx is None:
            validation_errors.append("No break and close below neckline")
        else:
            pattern.break_idx = break_idx
            pattern.break_price = df.iloc[break_idx]['low'] if pattern_type == 'normal' else df.iloc[break_idx]['high']
        
        # Check for retest (only if break occurred)
        if break_idx is not None:
            retest_idx = self._check_retest(df, break_idx, neckline, pattern_type)
            if retest_idx is None and self.require_retest:
                validation_errors.append("No retest of neckline after break")
            else:
                pattern.retest_idx = retest_idx
                if retest_idx is not None:
                    pattern.retest_price = df.iloc[retest_idx]['high'] if pattern_type == 'normal' else df.iloc[retest_idx]['low']
        
        # Check for entry candle (only if retest occurred)
        if retest_idx is not None:
            entry_idx, entry_type = self._find_entry_candle(df, retest_idx, neckline, pattern_type)
            if entry_idx is None and self.require_entry_candle:
                validation_errors.append("No entry candle after retest")
            else:
                pattern.entry_candle_idx = entry_idx
                pattern.entry_candle_type = entry_type
                if entry_idx is not None:
                    if pattern_type == 'normal':
                        pattern.entry_price = df.iloc[entry_idx]['low']
                    else:
                        pattern.entry_price = df.iloc[entry_idx]['high']
        
        # Calculate stop losses and take profit
        if pattern.entry_price is not None:
            pattern.stop_loss_options = self._calculate_stop_losses(df, pattern)
            pattern.take_profit_price = self._calculate_take_profit(df, pattern, neckline)
        
        pattern.valid = len(validation_errors) == 0
        pattern.validation_errors = validation_errors
        
        return pattern if pattern.valid else None
