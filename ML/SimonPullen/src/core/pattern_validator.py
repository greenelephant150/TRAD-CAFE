"""
Pattern validation rules for Simon Pullen's methodology
Handles candle count validation, peak similarity, and other rules
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PatternValidator:
    """
    Validates patterns against Simon's strict rules
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mw_min_candles = config.get('min_mw_candles', 7)
        self.mw_max_candles = config.get('max_mw_candles', 30)
        self.hs_min_candles = config.get('min_hs_candles', 30)
        self.hs_max_candles = config.get('max_hs_candles', 120)
        self.peak_similarity_tolerance = config.get('peak_similarity_tolerance', 0.02)
        
    def validate_mw_candle_count(self, left_idx: int, right_idx: int) -> Tuple[bool, str]:
        """
        Validate candle count for M&W patterns
        Must be between 7 and 30 candles
        """
        candle_count = right_idx - left_idx + 1
        
        if candle_count < self.mw_min_candles:
            return False, f"Too few candles: {candle_count} < {self.mw_min_candles}"
        elif candle_count > self.mw_max_candles:
            return False, f"Too many candles: {candle_count} > {self.mw_max_candles}"
        else:
            return True, f"Candle count OK: {candle_count}"
    
    def validate_hs_candle_count(self, left_idx: int, right_idx: int) -> Tuple[bool, str]:
        """
        Validate candle count for Head & Shoulders patterns
        Must be between 30 and 120 candles
        """
        candle_count = right_idx - left_idx + 1
        
        if candle_count < self.hs_min_candles:
            return False, f"Too few candles: {candle_count} < {self.hs_min_candles}"
        elif candle_count > self.hs_max_candles:
            return False, f"Too many candles: {candle_count} > {self.hs_max_candles}"
        else:
            return True, f"Candle count OK: {candle_count}"
    
    def validate_peak_similarity(self, price1: float, price2: float) -> Tuple[bool, float]:
        """
        Validate that two peaks/troughs are at similar levels
        Returns (valid, difference_percent)
        """
        diff_pct = abs(price2 - price1) / price1
        valid = diff_pct <= self.peak_similarity_tolerance
        return valid, diff_pct
    
    def validate_impulsive_move(self, df: pd.DataFrame, start_idx: int, 
                                lookback: int = 20, direction: str = None) -> Tuple[bool, float]:
        """
        Validate that price move into pattern was impulsive
        Simon: "Beautiful clean price action going into it"
        
        Returns (is_impulsive, strength_score)
        """
        if start_idx < lookback:
            lookback = start_idx
            
        pre_df = df.iloc[start_idx - lookback:start_idx]
        
        # Calculate price change
        price_change = (pre_df['close'].iloc[-1] - pre_df['close'].iloc[0]) / pre_df['close'].iloc[0]
        
        # Determine trend direction
        if direction is None:
            direction = 1 if price_change > 0 else -1
        
        # Count pullbacks (candles opposite to trend)
        pullbacks = 0
        for i in range(1, len(pre_df)):
            candle_change = pre_df['close'].iloc[i] - pre_df['close'].iloc[i-1]
            if candle_change * direction < 0:
                pullbacks += 1
        
        pullback_ratio = pullbacks / len(pre_df)
        
        # Calculate consecutive candles in trend direction
        max_consecutive = 0
        current = 0
        for i in range(1, len(pre_df)):
            candle_change = pre_df['close'].iloc[i] - pre_df['close'].iloc[i-1]
            if candle_change * direction > 0:
                current += 1
                max_consecutive = max(max_consecutive, current)
            else:
                current = 0
        
        # Score impulsiveness (0-1)
        # Low pullbacks + high consecutive candles = impulsive
        pullback_score = 1.0 - min(pullback_ratio * 2, 1.0)  # Lower pullbacks = higher score
        consecutive_score = min(max_consecutive / 10, 1.0)  # More consecutive = higher score
        move_score = min(abs(price_change) * 100, 1.0)  # Bigger move = higher score
        
        strength = (pullback_score * 0.4 + consecutive_score * 0.3 + move_score * 0.3)
        
        # Simon's rule: impulsive if pullbacks < 30% and decent move
        is_impulsive = pullback_ratio < 0.3 and abs(price_change) > 0.005
        
        return is_impulsive, strength
    
    def validate_tp_within_leg(self, df: pd.DataFrame, start_idx: int, 
                               tp_price: float, pattern_type: str) -> bool:
        """
        Validate that take profit is within the last impulsive leg
        Critical Simon rule: TP must be within the price range of the move into pattern
        """
        # Look back up to 20 candles before pattern
        lookback = min(20, start_idx)
        leg_df = df.iloc[start_idx - lookback:start_idx]
        
        if pattern_type in ['M', 'normal']:
            # For M-Top or normal H&S, TP is lower
            leg_low = leg_df['low'].min()
            return tp_price >= leg_low
        else:
            # For W-Bottom or inverted H&S, TP is higher
            leg_high = leg_df['high'].max()
            return tp_price <= leg_high
    
    def get_recommended_timeframe(self, candle_count: int, pattern_type: str) -> str:
        """
        Get recommended timeframe based on candle count
        If too many candles, go up a timeframe
        If too few, go down a timeframe
        """
        if pattern_type in ['M', 'W']:
            if candle_count > self.mw_max_candles:
                return "4h"  # Go up
            elif candle_count < self.mw_min_candles:
                return "15m"  # Go down
            else:
                return "1h"  # Current is fine
        else:  # H&S
            if candle_count > self.hs_max_candles:
                return "4h"  # Go up
            elif candle_count < self.hs_min_candles:
                return "15m"  # Go down
            else:
                return "1h"  # Current is fine
