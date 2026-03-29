"""
Stop loss calculation following Simon Pullen's rules
- Can ignore single rogue wicks
- Three options for H&S: behind head, behind shoulder, behind entry
"""

import numpy as np
from typing import Dict, Optional, Any

from src.core.mw_pattern import MWPattern
from src.core.head_shoulders import HeadShouldersPattern, StopLossType


class StopLossCalculator:
    """
    Calculates stop loss levels for different pattern types
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ignore_rogue_wicks = config.get('ignore_rogue_wicks', True)
        
    def calculate_for_mw(self, pattern: MWPattern, df, ignore_rogue: bool = True) -> float:
        """
        Calculate stop loss for M&W patterns
        Stop loss behind the pattern, can ignore single rogue wicks
        """
        if pattern.pattern_type == 'M':
            # For M-Top, stop above right peak
            stop = pattern.right_peak_price
            if ignore_rogue and self._is_rogue_wick(df, pattern.right_peak_idx, 'high'):
                # Use next highest nearby price
                stop = self._get_next_highest(df, pattern.right_peak_idx)
        else:  # W
            # For W-Bottom, stop below right trough
            stop = pattern.right_peak_price
            if ignore_rogue and self._is_rogue_wick(df, pattern.right_peak_idx, 'low'):
                stop = self._get_next_lowest(df, pattern.right_peak_idx)
        
        return stop
    
    def calculate_for_hs(self, pattern: HeadShouldersPattern, df, 
                         stop_type: str = 'moderate') -> float:
        """
        Calculate stop loss for H&S patterns
        Three options: conservative (behind head), moderate (behind shoulder), aggressive (behind entry)
        """
        if stop_type == StopLossType.CONSERVATIVE.value:
            # Behind head
            if pattern.pattern_type == 'normal':
                return pattern.head_price + (pattern.head_price * 0.002)
            else:
                return pattern.head_price - (pattern.head_price * 0.002)
                
        elif stop_type == StopLossType.MODERATE.value:
            # Behind right shoulder
            if pattern.pattern_type == 'normal':
                return pattern.right_shoulder_price + (pattern.right_shoulder_price * 0.001)
            else:
                return pattern.right_shoulder_price - (pattern.right_shoulder_price * 0.001)
                
        elif stop_type == StopLossType.AGGRESSIVE.value:
            # Behind entry candle
            if pattern.entry_candle_idx is not None:
                entry_candle = df.iloc[pattern.entry_candle_idx]
                if pattern.pattern_type == 'normal':
                    return entry_candle['high'] + (entry_candle['high'] * 0.0005)
                else:
                    return entry_candle['low'] - (entry_candle['low'] * 0.0005)
        
        # Default to moderate
        return self.calculate_for_hs(pattern, df, StopLossType.MODERATE.value)
    
    def _is_rogue_wick(self, df, idx: int, price_type: str, window: int = 3) -> bool:
        """
        Check if a price point is a rogue wick (significantly different from neighbors)
        """
        start = max(0, idx - window)
        end = min(len(df), idx + window + 1)
        
        prices = []
        for i in range(start, end):
            if i == idx:
                continue
            prices.append(df.iloc[i][price_type])
        
        if not prices:
            return False
            
        current_price = df.iloc[idx][price_type]
        avg_price = np.mean(prices)
        
        # If current is more than 2% higher/lower than average, it's a rogue wick
        return abs(current_price - avg_price) / avg_price > 0.02
    
    def _get_next_highest(self, df, idx: int, window: int = 3) -> float:
        """Get the next highest price near the given index"""
        start = max(0, idx - window)
        end = min(len(df), idx + window + 1)
        
        prices = []
        for i in range(start, end):
            if i == idx:
                continue
            prices.append(df.iloc[i]['high'])
        
        return max(prices) if prices else df.iloc[idx]['high']
    
    def _get_next_lowest(self, df, idx: int, window: int = 3) -> float:
        """Get the next lowest price near the given index"""
        start = max(0, idx - window)
        end = min(len(df), idx + window + 1)
        
        prices = []
        for i in range(start, end):
            if i == idx:
                continue
            prices.append(df.iloc[i]['low'])
        
        return min(prices) if prices else df.iloc[idx]['low']
