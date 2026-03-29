"""
Exit rule engine for Sid Naiman's SID Method
- Primary exit: RSI 50 reached
- Stop loss hit
- Two consecutive reversal days
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class ExitRuleEngine:
    """
    Manages exit rules for Sid's SID Method
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rsi_target = config.get('rsi_target', 50)
        self.reversal_days = config.get('reversal_exit_days', 2)
    
    def check_rsi_target(self, df: pd.DataFrame, entry_idx: int,
                           current_idx: int, direction: str) -> bool:
        """Check if RSI has reached target (50)"""
        for i in range(entry_idx, current_idx + 1):
            rsi = df['rsi'].iloc[i]
            if direction == 'long':
                # For long trades, we want RSI to go UP to 50
                if rsi >= self.rsi_target:
                    return True
            else:
                # For short trades, we want RSI to go DOWN to 50
                if rsi <= self.rsi_target:
                    return True
        return False
    
    def check_stop_loss(self, df: pd.DataFrame, entry_idx: int,
                          current_idx: int, stop_loss: float,
                          direction: str) -> bool:
        """Check if stop loss has been hit"""
        for i in range(entry_idx, current_idx + 1):
            candle = df.iloc[i]
            if direction == 'long':
                if candle['low'] <= stop_loss:
                    return True
            else:
                if candle['high'] >= stop_loss:
                    return True
        return False
    
    def check_reversal_days(self, df: pd.DataFrame, entry_idx: int,
                              current_idx: int, direction: str) -> bool:
        """Check for 2 consecutive reversal days"""
        if current_idx - entry_idx < self.reversal_days:
            return False
        
        # Check last 'reversal_days' candles
        for i in range(current_idx - self.reversal_days + 1, current_idx + 1):
            candle = df.iloc[i]
            prev_candle = df.iloc[i - 1]
            
            if direction == 'long':
                # Reversal means price going down
                reversal1 = candle['close'] < candle['open']
                reversal2 = prev_candle['close'] < prev_candle['open']
            else:
                # Reversal means price going up
                reversal1 = candle['close'] > candle['open']
                reversal2 = prev_candle['close'] > prev_candle['open']
            
            if not (reversal1 and reversal2):
                return False
        
        return True
    
    def should_exit(self, df: pd.DataFrame, entry_idx: int,
                     current_idx: int, stop_loss: float,
                     direction: str) -> Tuple[bool, str]:
        """
        Check if position should be exited
        Returns (exit, reason)
        """
        # Check RSI target
        if self.check_rsi_target(df, entry_idx, current_idx, direction):
            return True, "rsi_target_hit"
        
        # Check stop loss
        if self.check_stop_loss(df, entry_idx, current_idx, stop_loss, direction):
            return True, "stop_loss_hit"
        
        # Check reversal days
        if self.check_reversal_days(df, entry_idx, current_idx, direction):
            return True, "reversal_days"
        
        return False, ""
    
    def get_exit_status(self, df: pd.DataFrame, entry_idx: int,
                        exit_idx: int, entry_price: float,
                        stop_loss: float, direction: str) -> Dict:
        """Get detailed exit status for trade logging"""
        exit_candle = df.iloc[exit_idx]
        
        if direction == 'long':
            exit_price = exit_candle['close']
            pnl_pips = exit_price - entry_price
            max_favorable = max(df['high'].iloc[entry_idx:exit_idx + 1]) - entry_price
            max_adverse = entry_price - min(df['low'].iloc[entry_idx:exit_idx + 1])
        else:
            exit_price = exit_candle['close']
            pnl_pips = entry_price - exit_price
            max_favorable = entry_price - min(df['low'].iloc[entry_idx:exit_idx + 1])
            max_adverse = max(df['high'].iloc[entry_idx:exit_idx + 1]) - entry_price
        
        pnl_pct = (pnl_pips / entry_price) * 100 if entry_price != 0 else 0
        
        # Determine exit reason
        if self.check_rsi_target(df, entry_idx, exit_idx, direction):
            exit_reason = "rsi_target"
        elif self.check_stop_loss(df, entry_idx, exit_idx, stop_loss, direction):
            exit_reason = "stop_loss"
        elif self.check_reversal_days(df, entry_idx, exit_idx, direction):
            exit_reason = "reversal_days"
        else:
            exit_reason = "manual"
        
        return {
            'entry_idx': entry_idx,
            'exit_idx': exit_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pips': pnl_pips,
            'pnl_pct': pnl_pct,
            'bars_held': exit_idx - entry_idx,
            'max_favorable': max_favorable,
            'max_adverse': max_adverse
        }