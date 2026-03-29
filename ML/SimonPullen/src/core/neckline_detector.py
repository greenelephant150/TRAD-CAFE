"""
Neckline detection for pattern trading
Simon Pullen: Necklines based on bodies only, never wicks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class NecklineDetector:
    """
    Detects and validates necklines for M&W and H&S patterns
    Rules:
    - Based on bodies only (never wicks)
    - Cannot cut through any candle bodies
    - For M&W: lowest body between peaks / highest body between troughs
    - For H&S: connects lows between left shoulder-head and head-right shoulder
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_bodies_only = config.get('neckline_based_on', 'bodies') == 'bodies'
        
    def detect_mw_neckline(self, df: pd.DataFrame, left_idx: int, right_idx: int, pattern_type: str) -> Tuple[Optional[int], Optional[float]]:
        """
        Detect neckline for M-Top or W-Bottom
        For M-Top: lowest body between peaks
        For W-Bottom: highest body between troughs
        """
        between_df = df.iloc[left_idx:right_idx + 1]
        
        if pattern_type == 'M':
            # For M-Top: lowest body (use low as proxy for bottom of body)
            neckline_idx = between_df['low'].idxmin()
            neckline_price = between_df.loc[neckline_idx, 'low']
        else:  # 'W'
            # For W-Bottom: highest body (use high as proxy for top of body)
            neckline_idx = between_df['high'].idxmax()
            neckline_price = between_df.loc[neckline_idx, 'high']
            
        return neckline_idx, neckline_price
    
    def detect_hs_neckline(self, df: pd.DataFrame, left_shoulder_idx: int, 
                           head_idx: int, right_shoulder_idx: int, 
                           pattern_type: str) -> Optional[Dict]:
        """
        Detect neckline for Head and Shoulders
        Returns dict with start_idx, start_price, end_idx, end_price
        """
        if pattern_type == 'normal':
            # Normal H&S: connect lows between left shoulder-head and head-right shoulder
            between_left_head = df.iloc[left_shoulder_idx:head_idx + 1]
            left_neck = between_left_head.loc[between_left_head['low'].idxmin()]
            
            between_head_right = df.iloc[head_idx:right_shoulder_idx + 1]
            right_neck = between_head_right.loc[between_head_right['low'].idxmin()]
        else:  # inverted
            # Inverted H&S: connect highs
            between_left_head = df.iloc[left_shoulder_idx:head_idx + 1]
            left_neck = between_left_head.loc[between_left_head['high'].idxmax()]
            
            between_head_right = df.iloc[head_idx:right_shoulder_idx + 1]
            right_neck = between_head_right.loc[between_head_right['high'].idxmax()]
        
        # Check if neckline cuts through any bodies
        if self._neckline_cuts_bodies(df, left_neck, right_neck, left_shoulder_idx, right_shoulder_idx, pattern_type):
            logger.debug("Neckline cuts through bodies - invalid")
            return None
            
        return {
            'start_idx': left_neck.name,
            'start_price': left_neck['low'] if pattern_type == 'normal' else left_neck['high'],
            'end_idx': right_neck.name,
            'end_price': right_neck['low'] if pattern_type == 'normal' else right_neck['high']
        }
    
    def _neckline_cuts_bodies(self, df: pd.DataFrame, neck1, neck2, 
                              start_idx: int, end_idx: int, pattern_type: str) -> bool:
        """
        Check if neckline cuts through any candle bodies
        Critical Simon rule: Neckline must not intersect bodies
        """
        # Calculate neckline slope
        if pattern_type == 'normal':
            slope = (neck2['low'] - neck1['low']) / (neck2.name - neck1.name)
        else:
            slope = (neck2['high'] - neck1['high']) / (neck2.name - neck1.name)
        
        # Check each candle between neck points
        for i in range(neck1.name + 1, neck2.name):
            if i >= len(df):
                break
            candle = df.iloc[i]
            
            # Calculate neckline price at this index
            if pattern_type == 'normal':
                neck_price = neck1['low'] + slope * (i - neck1.name)
                # Check if neckline goes through body (between low and high)
                if neck_price < candle['high'] and neck_price > candle['low']:
                    return True
            else:
                neck_price = neck1['high'] + slope * (i - neck1.name)
                if neck_price < candle['high'] and neck_price > candle['low']:
                    return True
        
        return False
    
    def calculate_neckline_slope(self, neckline: Dict) -> float:
        """Calculate slope of neckline"""
        if neckline['end_idx'] == neckline['start_idx']:
            return 0
        return (neckline['end_price'] - neckline['start_price']) / (neckline['end_idx'] - neckline['start_idx'])
    
    def extend_neckline(self, df: pd.DataFrame, neckline: Dict, bars_forward: int = 20) -> pd.Series:
        """
        Extend neckline forward for support/resistance
        Simon uses extended necklines as future support/resistance levels
        """
        if neckline['end_idx'] >= len(df) - 1:
            return pd.Series(index=df.index)
            
        slope = self.calculate_neckline_slope(neckline)
        
        # Create series
        extension = pd.Series(index=df.index, dtype=float)
        
        # Fill from start to end
        for i in range(neckline['start_idx'], neckline['end_idx'] + 1):
            extension.iloc[i] = neckline['start_price'] + slope * (i - neckline['start_idx'])
        
        # Extend forward
        for i in range(neckline['end_idx'] + 1, min(neckline['end_idx'] + bars_forward + 1, len(df))):
            extension.iloc[i] = neckline['end_price'] + slope * (i - neckline['end_idx'])
        
        return extension
