"""
Exit rule engine following Simon Pullen's rules
- M&W: Close after 15 bars sideways
- H&S: Close after pattern-width sideways
- News: Close before high-impact news
- Take profit hits
- Stop loss hits
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.core.mw_pattern import MWPattern
from src.core.head_shoulders import HeadShouldersPattern
from src.execution.entry_rules import EntrySignal


class ExitRuleEngine:
    """
    Manages exit rules for open positions
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mw_max_sideways_bars = config.get('mw_max_sideways_bars', 15)
        self.hs_max_sideways_multiplier = config.get('hs_max_sideways_multiplier', 1.0)
        self.news_avoid_hours = config.get('news_avoid_hours', 2)
        
    def should_exit(self, df: pd.DataFrame, entry: EntrySignal, 
                    entry_idx: int, current_idx: int,
                    upcoming_news: List[Dict] = None) -> Tuple[bool, str]:
        """
        Check if position should be exited
        Returns (exit, reason)
        """
        # Check if take profit hit
        if self._take_profit_hit(df, entry, entry_idx, current_idx):
            return True, "take_profit_hit"
            
        # Check if stop loss hit
        if self._stop_loss_hit(df, entry, entry_idx, current_idx):
            return True, "stop_loss_hit"
            
        # Check sideways rules
        if self._sideways_exit(df, entry, entry_idx, current_idx):
            return True, "sideways_too_long"
            
        # Check news
        if upcoming_news and self._news_risk(upcoming_news):
            return True, "news_avoidance"
            
        return False, ""
    
    def _take_profit_hit(self, df: pd.DataFrame, entry: EntrySignal,
                          entry_idx: int, current_idx: int) -> bool:
        """Check if take profit level has been hit"""
        for i in range(entry_idx, current_idx + 1):
            candle = df.iloc[i]
            if entry.direction == 'long':
                if candle['high'] >= entry.take_profit:
                    return True
            else:  # short
                if candle['low'] <= entry.take_profit:
                    return True
        return False
    
    def _stop_loss_hit(self, df: pd.DataFrame, entry: EntrySignal,
                        entry_idx: int, current_idx: int) -> bool:
        """Check if stop loss level has been hit"""
        for i in range(entry_idx, current_idx + 1):
            candle = df.iloc[i]
            if entry.direction == 'long':
                if candle['low'] <= entry.stop_loss:
                    return True
            else:  # short
                if candle['high'] >= entry.stop_loss:
                    return True
        return False
    
    def _sideways_exit(self, df: pd.DataFrame, entry: EntrySignal,
                        entry_idx: int, current_idx: int) -> bool:
        """
        Check if price has gone sideways too long
        M&W: 15 bars sideways
        H&S: pattern-width sideways
        """
        if current_idx - entry_idx < 5:  # Need at least 5 bars to judge
            return False
        
        # Calculate price range since entry
        entry_price = entry.entry_price
        prices = []
        for i in range(entry_idx, current_idx + 1):
            prices.append(df.iloc[i]['close'])
        
        price_range = max(prices) - min(prices)
        avg_candle_range = np.mean([df.iloc[i]['high'] - df.iloc[i]['low'] for i in range(entry_idx, current_idx + 1)])
        
        # If range is less than 2x average candle, it's sideways
        is_sideways = price_range < (avg_candle_range * 2)
        
        if not is_sideways:
            return False
            
        # Check duration based on pattern type
        if entry.pattern_type in ['M', 'W']:
            bars_sideways = current_idx - entry_idx
            return bars_sideways >= self.mw_max_sideways_bars
        else:  # H&S
            pattern = entry.pattern_ref
            if hasattr(pattern, 'candle_count') and pattern.candle_count:
                max_sideways = int(pattern.candle_count * self.hs_max_sideways_multiplier)
                bars_sideways = current_idx - entry_idx
                return bars_sideways >= max_sideways
        
        return False
    
    def _news_risk(self, upcoming_news: List[Dict]) -> bool:
        """
        Check if high-impact news is approaching
        Simon: Never hold through high-impact news
        """
        now = datetime.now()
        
        for news in upcoming_news:
            if news.get('impact', '').lower() in ['high', 'medium']:
                news_time = news.get('time')
                if news_time:
                    time_diff = (news_time - now).total_seconds() / 3600
                    if 0 < time_diff < self.news_avoid_hours:
                        return True
        
        return False
    
    def get_exit_status(self, df: pd.DataFrame, entry: EntrySignal,
                        entry_idx: int, exit_idx: int) -> Dict:
        """
        Get detailed exit status for trade logging
        """
        exit_candle = df.iloc[exit_idx]
        
        if entry.direction == 'long':
            exit_price = exit_candle['close']
            pnl_pips = exit_price - entry.entry_price
            max_favorable = max([df.iloc[i]['high'] for i in range(entry_idx, exit_idx + 1)]) - entry.entry_price
            max_adverse = entry.entry_price - min([df.iloc[i]['low'] for i in range(entry_idx, exit_idx + 1)])
        else:
            exit_price = exit_candle['close']
            pnl_pips = entry.entry_price - exit_price
            max_favorable = entry.entry_price - min([df.iloc[i]['low'] for i in range(entry_idx, exit_idx + 1)])
            max_adverse = max([df.iloc[i]['high'] for i in range(entry_idx, exit_idx + 1)]) - entry.entry_price
        
        pnl_pct = (pnl_pips / entry.entry_price) * 100 if entry.entry_price != 0 else 0
        
        # Check if exit was due to take profit
        exit_reason = "manual"
        if self._take_profit_hit(df, entry, entry_idx, exit_idx):
            exit_reason = "take_profit"
        elif self._stop_loss_hit(df, entry, entry_idx, exit_idx):
            exit_reason = "stop_loss"
        
        return {
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry.entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pips': pnl_pips,
            'pnl_pct': pnl_pct,
            'bars_held': exit_idx - entry_idx,
            'max_favorable': max_favorable,
            'max_adverse': max_adverse,
        }
