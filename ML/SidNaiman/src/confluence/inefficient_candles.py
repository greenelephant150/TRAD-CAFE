"""
Inefficient Candle Detection
Simon Pullen: Inefficient candles (not squared up) act as magnets for price
Market doesn't like inefficiencies - 99.99% of candles get squared up
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class InefficientCandle:
    """Represents an inefficient candle that hasn't been squared up"""
    idx: int
    high: float
    low: float
    midpoint: float
    strength: float  # 0-1, based on size and time since formation
    detected_at: Optional[pd.Timestamp] = None


class InefficientCandleDetector:
    """
    Detects inefficient candles that haven't been squared up
    An inefficient candle is one where price hasn't retraced to the 50% level
    
    Simon: "The market doesn't like inefficiencies. They like to square up every candle."
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback = config.get('inefficient_lookback', 100)
        self.min_candle_size = config.get('min_inefficient_size', 0.001)  # Minimum size to consider
        self.square_threshold = config.get('square_threshold', 0.5)  # 50% retracement
        
    def detect(self, df: pd.DataFrame, current_idx: int = -1) -> List[InefficientCandle]:
        """
        Detect inefficient candles up to current_idx
        """
        if current_idx < 0:
            current_idx = len(df) - 1
            
        start_idx = max(0, current_idx - self.lookback)
        inefficient = []
        
        for i in range(start_idx, current_idx):
            candle = df.iloc[i]
            candle_size = candle['high'] - candle['low']
            
            # Skip tiny candles
            if candle_size < self.min_candle_size:
                continue
                
            # Check if this candle has been squared up
            if self._is_squared_up(df, i, current_idx):
                continue
                
            # Calculate strength based on size and age
            age = current_idx - i
            size_factor = min(candle_size / 0.01, 1.0)  # Normalize to 1% move
            age_factor = 1.0 - min(age / self.lookback, 0.8)  # Older candles have less pull
            
            strength = (size_factor * 0.6 + age_factor * 0.4)
            
            inefficient.append(InefficientCandle(
                idx=i,
                high=candle['high'],
                low=candle['low'],
                midpoint=(candle['high'] + candle['low']) / 2,
                strength=strength,
                detected_at=df.index[current_idx]
            ))
        
        return inefficient
    
    def _is_squared_up(self, df: pd.DataFrame, candle_idx: int, current_idx: int) -> bool:
        """
        Check if a candle has been squared up
        Squared up = price has retraced to at least 50% of the candle's range
        """
        candle = df.iloc[candle_idx]
        midpoint = (candle['high'] + candle['low']) / 2
        
        # Check all candles after this one
        for i in range(candle_idx + 1, min(current_idx + 1, len(df))):
            later_candle = df.iloc[i]
            
            # Check if price touched the midpoint
            if later_candle['low'] <= midpoint <= later_candle['high']:
                return True
                
            # Also check closes (some definitions use close)
            if abs(later_candle['close'] - midpoint) / (candle['high'] - candle['low']) < 0.1:
                return True
        
        return False
    
    def get_nearest_inefficient(self, df: pd.DataFrame, price: float, 
                                direction: str, current_idx: int = -1) -> Optional[InefficientCandle]:
        """
        Find the nearest inefficient candle in the direction of trade
        Used to extend profit targets
        """
        inefficient = self.detect(df, current_idx)
        
        if not inefficient:
            return None
            
        if direction == 'long':
            # Look for inefficient candles above current price (acting as upside magnet)
            above = [c for c in inefficient if c.midpoint > price]
            if not above:
                return None
            return min(above, key=lambda c: c.midpoint - price)
        else:
            # Look for inefficient candles below current price (acting as downside magnet)
            below = [c for c in inefficient if c.midpoint < price]
            if not below:
                return None
            return min(below, key=lambda c: price - c.midpoint)
    
    def would_extend_target(self, df: pd.DataFrame, entry_price: float, 
                           target_price: float, direction: str) -> Tuple[bool, Optional[float]]:
        """
        Check if an inefficient candle would extend the target
        Returns (would_extend, better_target)
        """
        nearest = self.get_nearest_inefficient(df, entry_price, direction, len(df) - 1)
        
        if not nearest:
            return False, None
            
        if direction == 'long' and nearest.midpoint > target_price:
            return True, nearest.midpoint
        elif direction == 'short' and nearest.midpoint < target_price:
            return True, nearest.midpoint
            
        return False, None
