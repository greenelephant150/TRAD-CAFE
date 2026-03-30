"""
Weekly Trendline Detection
Simon Pullen: Never trade against weekly trendlines - they're too powerful
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class WeeklyTrendline:
    """Represents a weekly trendline"""
    start_idx: int
    start_price: float
    end_idx: int
    end_price: float
    slope: float
    touches: List[int]  # Indices where price touched the line
    strength: float  # 0-1, based on number of touches and time span


class WeeklyTrendlineDetector:
    """
    Detects weekly trendlines
    Simon: "I don't ever mess with a weekly trend line - they're so powerful"
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_touches = config.get('trendline_min_touches', 3)
        self.touch_tolerance = config.get('trendline_touch_tolerance', 0.002)  # 0.2%
        
    def detect_trendlines(self, df: pd.DataFrame) -> List[WeeklyTrendline]:
        """
        Detect weekly trendlines
        """
        # Ensure we have weekly data
        if df.index.freq and df.index.freq.name != 'W':
            weekly = df.resample('W').agg({
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).dropna()
        else:
            weekly = df
        
        trendlines = []
        
        # Detect uptrend lines (connecting lows)
        uptrends = self._find_uptrend_lines(weekly)
        trendlines.extend(uptrends)
        
        # Detect downtrend lines (connecting highs)
        downtrends = self._find_downtrend_lines(weekly)
        trendlines.extend(downtrends)
        
        return trendlines
    
    def _find_uptrend_lines(self, df: pd.DataFrame) -> List[WeeklyTrendline]:
        """
        Find uptrend lines connecting swing lows
        """
        # Find swing lows
        swing_lows = []
        for i in range(2, len(df) - 2):
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                swing_lows.append((i, df['low'].iloc[i]))
        
        trendlines = []
        
        # Check combinations of swing lows
        for i in range(len(swing_lows)):
            for j in range(i + 1, len(swing_lows)):
                idx1, price1 = swing_lows[i]
                idx2, price2 = swing_lows[j]
                
                # Calculate slope
                slope = (price2 - price1) / (idx2 - idx1)
                
                # Count touches
                touches = [idx1, idx2]
                
                for k in range(j + 1, len(swing_lows)):
                    idx3, price3 = swing_lows[k]
                    expected = price1 + slope * (idx3 - idx1)
                    
                    # Check if swing low touches trendline
                    if abs(price3 - expected) / price1 < self.touch_tolerance:
                        touches.append(idx3)
                
                if len(touches) >= self.min_touches:
                    strength = min(len(touches) / 10, 1.0)
                    trendlines.append(WeeklyTrendline(
                        start_idx=idx1,
                        start_price=price1,
                        end_idx=touches[-1],
                        end_price=price1 + slope * (touches[-1] - idx1),
                        slope=slope,
                        touches=touches,
                        strength=strength
                    ))
        
        return trendlines
    
    def _find_downtrend_lines(self, df: pd.DataFrame) -> List[WeeklyTrendline]:
        """
        Find downtrend lines connecting swing highs
        """
        # Find swing highs
        swing_highs = []
        for i in range(2, len(df) - 2):
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                swing_highs.append((i, df['high'].iloc[i]))
        
        trendlines = []
        
        # Check combinations of swing highs
        for i in range(len(swing_highs)):
            for j in range(i + 1, len(swing_highs)):
                idx1, price1 = swing_highs[i]
                idx2, price2 = swing_highs[j]
                
                # Calculate slope (should be negative for downtrend)
                slope = (price2 - price1) / (idx2 - idx1)
                
                if slope >= 0:  # Not a downtrend
                    continue
                
                # Count touches
                touches = [idx1, idx2]
                
                for k in range(j + 1, len(swing_highs)):
                    idx3, price3 = swing_highs[k]
                    expected = price1 + slope * (idx3 - idx1)
                    
                    # Check if swing high touches trendline
                    if abs(price3 - expected) / price1 < self.touch_tolerance:
                        touches.append(idx3)
                
                if len(touches) >= self.min_touches:
                    strength = min(len(touches) / 10, 1.0)
                    trendlines.append(WeeklyTrendline(
                        start_idx=idx1,
                        start_price=price1,
                        end_idx=touches[-1],
                        end_price=price1 + slope * (touches[-1] - idx1),
                        slope=slope,
                        touches=touches,
                        strength=strength
                    ))
        
        return trendlines
    
    def would_block_trade(self, trendlines: List[WeeklyTrendline], 
                          entry_price: float, target_price: float,
                          direction: str) -> Tuple[bool, Optional[WeeklyTrendline]]:
        """
        Check if a weekly trendline would block the trade
        Simon: Never trade against weekly trendlines
        """
        for trendline in trendlines:
            # Check if trendline lies between entry and target
            if direction == 'long':
                if entry_price < trendline.start_price < target_price:
                    return True, trendline
            else:
                if target_price < trendline.start_price < entry_price:
                    return True, trendline
        
        return False, None
    
    def get_price_at_idx(self, trendline: WeeklyTrendline, idx: int) -> float:
        """Get trendline price at a given index"""
        return trendline.start_price + trendline.slope * (idx - trendline.start_idx)
