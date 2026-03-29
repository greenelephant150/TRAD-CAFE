"""
Entry rule engine for Sid Naiman's SID Method
Simpler than Simon's - just RSI signal + MACD alignment
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


class EntrySignal:
    """Represents an entry signal from Sid's SID Method"""
    
    def __init__(self,
                 instrument: str,
                 signal_type: str,  # 'oversold' or 'overbought'
                 direction: str,    # 'long' or 'short'
                 entry_price: float,
                 stop_loss: float,
                 take_profit: float,
                 rsi_value: float,
                 macd_aligned: bool,
                 macd_crossed: bool,
                 confidence: float,
                 signal_date: datetime,
                 entry_date: datetime,
                 pattern_confirmation: Optional[str] = None):
        
        self.instrument = instrument
        self.signal_type = signal_type
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.rsi_value = rsi_value
        self.macd_aligned = macd_aligned
        self.macd_crossed = macd_crossed
        self.confidence = confidence
        self.signal_date = signal_date
        self.entry_date = entry_date
        self.pattern_confirmation = pattern_confirmation


class EntryRuleEngine:
    """
    Manages entry rules for Sid's SID Method
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.require_macd_alignment = config.get('require_macd_alignment', True)
        self.require_macd_cross = config.get('require_macd_cross', False)  # Optional
        self.require_earnings_buffer = config.get('require_earnings_buffer', True)
        self.earnings_days = config.get('earnings_days_buffer', 14)
    
    def check_rsi_signal(self, df: pd.DataFrame, current_idx: int) -> Tuple[bool, str]:
        """
        Check if RSI gives a signal
        Returns (has_signal, signal_type)
        """
        if current_idx < 0:
            current_idx = len(df) - 1
        
        rsi_value = df['rsi'].iloc[current_idx]
        
        if rsi_value < self.rsi_oversold:
            return True, 'oversold'
        elif rsi_value > self.rsi_overbought:
            return True, 'overbought'
        else:
            return False, 'neutral'
    
    def check_macd_alignment(self, df: pd.DataFrame, current_idx: int,
                               signal_type: str) -> bool:
        """
        Check if MACD aligns with RSI signal
        For oversold: MACD should be pointing up
        For overbought: MACD should be pointing down
        """
        if current_idx < 1:
            return False
        
        current_macd = df['macd'].iloc[current_idx]
        prev_macd = df['macd'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            return current_macd > prev_macd
        elif signal_type == 'overbought':
            return current_macd < prev_macd
        else:
            return False
    
    def check_macd_cross(self, df: pd.DataFrame, current_idx: int,
                           signal_type: str) -> bool:
        """
        Check if MACD has crossed the signal line
        """
        if current_idx < 1:
            return False
        
        current_macd = df['macd'].iloc[current_idx]
        current_signal = df['macd_signal'].iloc[current_idx]
        prev_macd = df['macd'].iloc[current_idx - 1]
        prev_signal = df['macd_signal'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            return prev_macd <= prev_signal and current_macd > current_signal
        elif signal_type == 'overbought':
            return prev_macd >= prev_signal and current_macd < current_signal
        else:
            return False
    
    def find_signal_date(self, df: pd.DataFrame, entry_idx: int,
                           signal_type: str) -> datetime:
        """Find when RSI first crossed threshold"""
        rsi_values = df['rsi'].iloc[:entry_idx + 1]
        
        if signal_type == 'oversold':
            mask = rsi_values < self.rsi_oversold
        else:
            mask = rsi_values > self.rsi_overbought
        
        if mask.any():
            signal_idx = mask[mask].index[-1]
            return signal_idx
        else:
            return df.index[entry_idx]
    
    def calculate_stop_loss(self, df: pd.DataFrame,
                              signal_date: datetime,
                              entry_date: datetime,
                              signal_type: str) -> float:
        """
        Calculate stop loss using Sid's rules
        """
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]
        
        if period_df.empty:
            return 0.0
        
        if signal_type == 'oversold':
            lowest_low = period_df['low'].min()
            return np.floor(lowest_low)
        else:
            highest_high = period_df['high'].max()
            return np.ceil(highest_high)
    
    def check_earnings(self, earnings_date: Optional[datetime], 
                        entry_date: datetime) -> bool:
        """Check earnings buffer"""
        if earnings_date is None:
            return True
        
        days_before = (earnings_date - entry_date).days
        return days_before >= self.earnings_days
    
    def should_enter(self, df: pd.DataFrame, current_idx: int,
                      instrument: str,
                      earnings_date: Optional[datetime] = None,
                      pattern_confirmation: Optional[str] = None) -> Optional[EntrySignal]:
        """
        Main entry check for Sid's SID Method
        """
        if current_idx < 0:
            current_idx = len(df) - 1
        
        # Check RSI signal
        has_signal, signal_type = self.check_rsi_signal(df, current_idx)
        if not has_signal:
            return None
        
        # Check MACD alignment
        aligned = self.check_macd_alignment(df, current_idx, signal_type)
        if not aligned and self.require_macd_alignment:
            return None
        
        # Optional MACD cross (stronger confirmation)
        crossed = self.check_macd_cross(df, current_idx, signal_type)
        if self.require_macd_cross and not crossed:
            return None
        
        # Find signal date for stop loss calculation
        signal_date = self.find_signal_date(df, current_idx, signal_type)
        entry_date = df.index[current_idx]
        
        # Calculate stop loss
        stop_loss = self.calculate_stop_loss(df, signal_date, entry_date, signal_type)
        
        # Determine direction
        direction = 'long' if signal_type == 'oversold' else 'short'
        entry_price = df['close'].iloc[current_idx]
        
        # Simple confidence score
        confidence = 60  # Base confidence
        if crossed:
            confidence += 20  # MACD cross adds confidence
        if pattern_confirmation:
            confidence += 10  # Pattern adds confidence
        
        # Create entry signal
        signal = EntrySignal(
            instrument=instrument,
            signal_type=signal_type,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=0.0,  # Will be set by exit rules
            rsi_value=df['rsi'].iloc[current_idx],
            macd_aligned=aligned,
            macd_crossed=crossed,
            confidence=min(confidence, 100),
            signal_date=signal_date,
            entry_date=entry_date,
            pattern_confirmation=pattern_confirmation
        )
        
        return signal

