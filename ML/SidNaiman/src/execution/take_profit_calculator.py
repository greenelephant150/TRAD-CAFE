"""
Take profit calculation following Simon Pullen's rules
- M&W: 1:1 risk-to-reward minimum
- H&S: Pattern completion level (distance from neckline to head)
- Can extend targets for inefficient candles
"""

from typing import Optional, Tuple
import numpy as np

from src.core.mw_pattern import MWPattern
from src.core.head_shoulders import HeadShouldersPattern


class TakeProfitCalculator:
    """
    Calculates take profit levels for different pattern types
    """
    
    def __init__(self, config):
        self.config = config
        self.default_rr = config.get('default_risk_reward', 1.0)
        
    def calculate_for_mw(self, pattern: MWPattern) -> float:
        """
        Calculate take profit for M&W patterns (1:1 minimum)
        """
        if pattern.pattern_type == 'M':
            # For M-Top, TP below entry
            risk = pattern.stop_loss_price - pattern.entry_price
            return pattern.entry_price - (risk * self.default_rr)
        else:  # W
            # For W-Bottom, TP above entry
            risk = pattern.entry_price - pattern.stop_loss_price
            return pattern.entry_price + (risk * self.default_rr)
    
    def calculate_for_hs(self, pattern: HeadShouldersPattern) -> float:
        """
        Calculate take profit for H&S patterns (pattern completion level)
        Distance from neckline to head, projected beyond entry
        """
        if pattern.pattern_type == 'normal':
            # For normal H&S, TP below entry
            head_to_neckline = pattern.head_price - pattern.neckline_end_price
            return pattern.entry_price - head_to_neckline
        else:  # inverted
            # For inverted H&S, TP above entry
            head_to_neckline = pattern.neckline_end_price - pattern.head_price
            return pattern.entry_price + head_to_neckline
    
    def extend_for_inefficient_candle(self, base_tp: float, inefficient_price: float, 
                                      direction: str) -> Tuple[float, bool]:
        """
        Extend take profit to include inefficient candle target
        Returns (new_tp, extended)
        """
        if direction == 'long':
            if inefficient_price > base_tp:
                return inefficient_price, True
        else:  # short
            if inefficient_price < base_tp:
                return inefficient_price, True
        
        return base_tp, False
    
    def adjust_for_adr(self, base_tp: float, adr_levels, direction: str) -> Tuple[float, str]:
        """
        Adjust take profit based on ADR levels
        Simon: 50% (99% hit), 75% (79%), 100% (41%), 125% (14%)
        """
        if direction == 'long':
            # For long trades, TP should be below ADR resistance
            if base_tp > adr_levels.level_125:
                return adr_levels.level_125, "adjusted_to_125%_ADR"
            elif base_tp > adr_levels.level_100:
                return adr_levels.level_100, "adjusted_to_100%_ADR"
        else:
            # For short trades, TP should be above ADR support
            if base_tp < adr_levels.level_125:
                return adr_levels.level_125, "adjusted_to_125%_ADR"
            elif base_tp < adr_levels.level_100:
                return adr_levels.level_100, "adjusted_to_100%_ADR"
        
        return base_tp, "no_adjustment"
