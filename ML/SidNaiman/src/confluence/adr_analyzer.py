"""
Average Daily Range (ADR) Analyzer
Simon Pullen: ADR probabilities - 50% (99%), 75% (79%), 100% (41%), 125% (14%)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ADRLevels:
    """Average Daily Range levels for current day"""
    date: pd.Timestamp
    low: float
    high: float
    adr_pips: float
    level_50: float  # 50% of ADR
    level_75: float  # 75% of ADR
    level_100: float  # 100% of ADR
    level_125: float  # 125% of ADR
    prob_50: float = 0.99  # 99% chance of hitting
    prob_75: float = 0.79  # 79% chance
    prob_100: float = 0.41  # 41% chance
    prob_125: float = 0.14  # 14% chance


class ADRAnalyzer:
    """
    Analyzes Average Daily Range levels
    Simon's probabilities:
    - 50% of ADR: 99% chance of being hit
    - 75% of ADR: 79% chance
    - 100% of ADR: 41% chance
    - 125% of ADR: 14% chance (strong reversal signal)
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.period = config.get('adr_period', 14)
        self.probabilities = config.get('adr_probabilities', {
            0.50: 0.99,
            0.75: 0.79,
            1.00: 0.41,
            1.25: 0.14
        })
        
    def calculate_adr(self, df: pd.DataFrame) -> float:
        """
        Calculate Average Daily Range
        Average of (high - low) over last N days
        """
        if len(df) < self.period:
            return 0
            
        # Calculate daily ranges
        if df.index.freq and df.index.freq.name == 'D':
            # Already daily data
            ranges = df['high'] - df['low']
        else:
            # Need to resample to daily
            daily = df.resample('D').agg({
                'high': 'max',
                'low': 'min'
            }).dropna()
            ranges = daily['high'] - daily['low']
        
        # Calculate average
        adr = ranges.tail(self.period).mean()
        return adr
    
    def get_today_levels(self, df: pd.DataFrame, current_idx: int = -1) -> Optional[ADRLevels]:
        """
        Get ADR levels for current day
        """
        if current_idx < 0:
            current_idx = len(df) - 1
            
        current_candle = df.iloc[current_idx]
        current_date = df.index[current_idx].date()
        
        # Get today's range so far
        day_start = df.index[current_idx].normalize()
        today_df = df[df.index >= day_start]
        
        if today_df.empty:
            return None
            
        day_low = today_df['low'].min()
        day_high = today_df['high'].max()
        
        # Calculate ADR
        adr = self.calculate_adr(df)
        adr_pips = adr
        
        # Calculate levels
        levels = ADRLevels(
            date=pd.Timestamp(current_date),
            low=day_low,
            high=day_high,
            adr_pips=adr_pips,
            level_50=day_low + adr * 0.5,
            level_75=day_low + adr * 0.75,
            level_100=day_low + adr,
            level_125=day_low + adr * 1.25
        )
        
        return levels
    
    def get_reversal_probability(self, price: float, levels: ADRLevels, direction: str) -> float:
        """
        Get probability of reversal at current price level
        Higher probability at extreme levels (125% ADR)
        """
        if direction == 'long':
            # For long entries, we want price at low levels
            if price <= levels.low + levels.adr_pips * 0.1:
                return 0.8  # Near low of day
            elif price <= levels.level_50:
                return 0.6
            else:
                return 0.3
        else:
            # For short entries, we want price at high levels
            if price >= levels.high - levels.adr_pips * 0.1:
                return 0.8  # Near high of day
            elif price >= levels.level_50:
                return 0.6
            else:
                return 0.3
    
    def is_extended(self, price: float, levels: ADRLevels, direction: str) -> Tuple[bool, float]:
        """
        Check if price is extended (beyond 100% ADR)
        Returns (is_extended, reversal_probability)
        """
        if direction == 'long':
            # For long, extended means below 100% level (oversold)
            if price <= levels.level_100:
                return True, self.probabilities.get(1.25, 0.14)
            elif price <= levels.level_125:
                return True, self.probabilities.get(1.25, 0.14) * 1.5
        else:
            # For short, extended means above 100% level (overbought)
            if price >= levels.level_100:
                return True, self.probabilities.get(1.25, 0.14)
            elif price >= levels.level_125:
                return True, self.probabilities.get(1.25, 0.14) * 1.5
        
        return False, 0.0
    
    def would_reverse_at_level(self, price: float, levels: ADRLevels) -> Tuple[bool, str]:
        """
        Check if price is likely to reverse at current ADR level
        """
        # Check 125% level (strong reversal)
        if abs(price - levels.level_125) / levels.adr_pips < 0.05:
            return True, "125% ADR - strong reversal signal"
            
        # Check 100% level (moderate reversal)
        if abs(price - levels.level_100) / levels.adr_pips < 0.05:
            return True, "100% ADR - moderate reversal signal"
            
        return False, ""
