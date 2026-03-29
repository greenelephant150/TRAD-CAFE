#!/usr/bin/env python3
"""
Sid Naiman's SID Method - Complete Strategy Implementation
With verbose output and tqdm progress bars
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import sys

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm, trange
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Create dummy tqdm
    class tqdm:
        def __init__(self, iterable=None, desc="", **kwargs):
            self.iterable = iterable or []
            self.desc = desc
        def __iter__(self): 
            return iter(self.iterable)
        def update(self, n=1): pass
        def close(self): pass
        def set_description(self, desc): self.desc = desc
        def set_postfix(self, **kwargs): pass
    trange = lambda *args, **kwargs: tqdm(range(*args), **kwargs)

logger = logging.getLogger(__name__)


class SidMethod:
    """
    Complete implementation of Sid Naiman's SID Method
    """
    
    def __init__(self, account_balance: float = 10000, verbose: bool = True):
        self.account_balance = account_balance
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_target = 50
        self.earnings_days_buffer = 14
        self.max_risk_percent = 2.0
        self.min_risk_percent = 0.5
        self.default_risk_percent = 1.0
        self.max_consecutive_losses = 3
        self.verbose = verbose
        
        # Track open positions
        self.open_positions = []
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_loss = 0.0
        
        if self.verbose:
            print(f"[SidMethod] Initialized with balance: ${account_balance:,.2f}")
            print(f"[SidMethod] RSI thresholds: oversold={self.rsi_oversold}, overbought={self.rsi_overbought}")
            print(f"[SidMethod] Risk: {self.min_risk_percent}%-{self.max_risk_percent}%")
            print(f"[SidMethod] TQDM available: {TQDM_AVAILABLE}")
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14, desc: str = "Calculating RSI") -> pd.Series:
        """Calculate RSI (Relative Strength Index) with progress"""
        if self.verbose:
            print(f"[SidMethod] {desc} (period={period})...")
        
        delta = df['close'].diff()
        
        # Use tqdm for rolling calculations if large dataset
        if len(df) > 100000 and TQDM_AVAILABLE:
            print(f"[SidMethod]   Processing {len(df):,} rows...")
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        else:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        
        if self.verbose:
            print(f"[SidMethod]   RSI range: {rsi.min():.2f} - {rsi.max():.2f}")
        
        return rsi.fillna(50)
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, 
                        signal: int = 9, desc: str = "Calculating MACD") -> pd.DataFrame:
        """Calculate MACD with progress"""
        if self.verbose:
            print(f"[SidMethod] {desc} (fast={fast}, slow={slow}, signal={signal})...")
        
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        if self.verbose:
            print(f"[SidMethod]   MACD range: {macd_line.min():.4f} - {macd_line.max():.4f}")
            print(f"[SidMethod]   Signal range: {signal_line.min():.4f} - {signal_line.max():.4f}")
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def check_rsi_signal(self, rsi_value: float) -> str:
        """Check RSI signal with debug output"""
        if rsi_value < self.rsi_oversold:
            if self.verbose:
                print(f"[SidMethod]   RSI={rsi_value:.2f} -> OVERSOLD")
            return 'oversold'
        elif rsi_value > self.rsi_overbought:
            if self.verbose:
                print(f"[SidMethod]   RSI={rsi_value:.2f} -> OVERBOUGHT")
            return 'overbought'
        else:
            if self.verbose:
                print(f"[SidMethod]   RSI={rsi_value:.2f} -> neutral")
            return 'neutral'
    
    def check_macd_alignment(self, macd_df: pd.DataFrame, current_idx: int, 
                               signal_type: str) -> bool:
        """Check MACD alignment with debug"""
        if current_idx < 2:
            return False
        
        current_macd = macd_df['macd'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            aligned = current_macd > prev_macd
            if self.verbose and aligned:
                print(f"[SidMethod]   MACD aligned UP for oversold: {prev_macd:.4f} -> {current_macd:.4f}")
            return aligned
        elif signal_type == 'overbought':
            aligned = current_macd < prev_macd
            if self.verbose and aligned:
                print(f"[SidMethod]   MACD aligned DOWN for overbought: {prev_macd:.4f} -> {current_macd:.4f}")
            return aligned
        else:
            return False
    
    def check_macd_cross(self, macd_df: pd.DataFrame, current_idx: int,
                           signal_type: str) -> bool:
        """Check MACD cross with debug"""
        if current_idx < 1:
            return False
        
        current_macd = macd_df['macd'].iloc[current_idx]
        current_signal = macd_df['signal'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        prev_signal = macd_df['signal'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            crossed = (prev_macd <= prev_signal and current_macd > current_signal)
            if self.verbose and crossed:
                print(f"[SidMethod]   MACD CROSSED UP: {prev_macd:.4f} <= {prev_signal:.4f} -> {current_macd:.4f} > {current_signal:.4f}")
            return crossed
        elif signal_type == 'overbought':
            crossed = (prev_macd >= prev_signal and current_macd < current_signal)
            if self.verbose and crossed:
                print(f"[SidMethod]   MACD CROSSED DOWN: {prev_macd:.4f} >= {prev_signal:.4f} -> {current_macd:.4f} < {current_signal:.4f}")
            return crossed
        else:
            return False
    
    def calculate_stop_loss(self, df: pd.DataFrame, 
                             signal_date: datetime,
                             entry_date: datetime,
                             signal_type: str) -> float:
        """Calculate stop loss with debug"""
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]
        
        if period_df.empty:
            if self.verbose:
                print(f"[SidMethod]   WARNING: Empty period for stop loss")
            return 0.0
        
        if signal_type == 'oversold':
            lowest_low = period_df['low'].min()
            stop_loss = np.floor(lowest_low)
            if self.verbose:
                print(f"[SidMethod]   Stop loss (oversold): lowest={lowest_low:.5f}, rounded down={stop_loss:.5f}")
            return float(stop_loss)
        else:
            highest_high = period_df['high'].max()
            stop_loss = np.ceil(highest_high)
            if self.verbose:
                print(f"[SidMethod]   Stop loss (overbought): highest={highest_high:.5f}, rounded up={stop_loss:.5f}")
            return float(stop_loss)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                  risk_percent: float = None) -> Dict:
        """Calculate position size with debug"""
        if risk_percent is None:
            # Adjust risk based on consecutive losses
            if self.consecutive_losses == 0:
                risk_percent = self.default_risk_percent
            elif self.consecutive_losses == 1:
                risk_percent = 0.75
            elif self.consecutive_losses == 2:
                risk_percent = 0.5
            else:
                risk_percent = 0.25
            
            if self.verbose:
                print(f"[SidMethod]   Risk adjusted for {self.consecutive_losses} losses: {risk_percent:.2f}%")
        
        risk_percent = max(self.min_risk_percent, min(risk_percent, self.max_risk_percent))
        risk_amount = self.account_balance * (risk_percent / 100)
        
        # Calculate risk per unit
        if entry_price > stop_loss:
            risk_per_unit = entry_price - stop_loss
            direction = 'long'
        else:
            risk_per_unit = stop_loss - entry_price
            direction = 'short'
        
        if risk_per_unit <= 0:
            if self.verbose:
                print(f"[SidMethod]   ERROR: Invalid stop loss (risk_per_unit={risk_per_unit})")
            return {'error': 'Invalid stop loss'}
        
        units = risk_amount / risk_per_unit
        units = np.floor(units)
        
        if self.verbose:
            print(f"[SidMethod]   Position size: {units:.0f} units")
            print(f"[SidMethod]   Risk: ${risk_amount:.2f} ({risk_percent:.1f}%)")
            print(f"[SidMethod]   Risk per unit: ${risk_per_unit:.5f}")
            print(f"[SidMethod]   Position value: ${units * entry_price:.2f}")
        
        return {
            'units': units,
            'direction': direction,
            'risk_amount': risk_amount,
            'risk_percent': risk_percent,
            'risk_per_unit': risk_per_unit,
            'position_value': units * entry_price,
            'entry_price': entry_price,
            'stop_loss': stop_loss
        }
    
    def find_trade_opportunities(self, df: pd.DataFrame, 
                                   earnings_dates: Dict[str, datetime] = None,
                                   progress_callback: callable = None) -> List[Dict]:
        """
        Find all trade opportunities with progress tracking
        """
        if df.empty or len(df) < 50:
            if self.verbose:
                print(f"[SidMethod] Insufficient data: {len(df)} rows")
            return []
        
        if self.verbose:
            print(f"[SidMethod] Finding trade opportunities in {len(df):,} rows...")
        
        # Calculate indicators with progress
        df = df.copy()
        
        if self.verbose:
            print(f"[SidMethod]   Step 1/4: Calculating RSI...")
        df['rsi'] = self.calculate_rsi(df)
        
        if self.verbose:
            print(f"[SidMethod]   Step 2/4: Calculating MACD...")
        macd_df = self.calculate_macd(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        df['macd_hist'] = macd_df['histogram']
        
        if self.verbose:
            print(f"[SidMethod]   Step 3/4: Scanning for signals...")
        
        opportunities = []
        
        # Use tqdm for progress if available
        if TQDM_AVAILABLE and len(df) > 5000:
            iterator = tqdm(range(20, len(df) - 1), desc="  Scanning", unit="bars")
        else:
            iterator = range(20, len(df) - 1)
        
        for i in iterator:
            current_date = df.index[i]
            rsi_value = df['rsi'].iloc[i]
            
            # Check RSI signal
            signal_type = self.check_rsi_signal(rsi_value)
            if signal_type == 'neutral':
                continue
            
            # Check earnings
            if earnings_dates and current_date in earnings_dates:
                earnings_date = earnings_dates.get(current_date)
                if not self.check_earnings(earnings_date, current_date):
                    if self.verbose:
                        print(f"[SidMethod]   Skipping {current_date}: within {self.earnings_days_buffer} days of earnings")
                    continue
            
            # Check MACD alignment
            aligned = self.check_macd_alignment(macd_df, i, signal_type)
            if not aligned:
                continue
            
            # Check MACD cross
            crossed = self.check_macd_cross(macd_df, i, signal_type)
            
            # Find signal date
            signal_date = self._find_signal_date(df, i, signal_type)
            
            # Calculate stop loss
            stop_loss = self.calculate_stop_loss(df, signal_date, current_date, signal_type)
            
            # Determine direction
            direction = 'long' if signal_type == 'oversold' else 'short'
            entry_price = df['close'].iloc[i]
            take_profit = self.calculate_take_profit(entry_price, stop_loss, direction, 'rsi_50')
            
            opportunity = {
                'date': current_date,
                'signal_type': signal_type,
                'direction': direction,
                'rsi_value': rsi_value,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_date': signal_date,
                'macd_aligned': aligned,
                'macd_crossed': crossed,
                'index': i
            }
            
            opportunities.append(opportunity)
            
            if progress_callback and len(opportunities) % 100 == 0:
                progress_callback(len(opportunities))
        
        if self.verbose:
            print(f"[SidMethod]   Step 4/4: Found {len(opportunities)} opportunities")
        
        return opportunities
    
    def _find_signal_date(self, df: pd.DataFrame, entry_idx: int, signal_type: str) -> datetime:
        """Find the date when RSI first crossed the threshold"""
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
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                                direction: str, method: str = 'rsi_50') -> float:
        """Calculate take profit levels"""
        risk_distance = abs(entry_price - stop_loss)
        
        if method == 'rsi_50':
            if direction == 'long':
                return entry_price + (risk_distance * 1.0)
            else:
                return entry_price - (risk_distance * 1.0)
        elif method == 'points':
            points = 4 if entry_price < 200 else 8
            if direction == 'long':
                return entry_price + points
            else:
                return entry_price - points
        else:
            return 0.0
    
    def check_earnings(self, earnings_date: Optional[datetime], entry_date: datetime) -> bool:
        """Check earnings buffer"""
        if earnings_date is None:
            return True
        days_before = (earnings_date - entry_date).days
        return days_before >= self.earnings_days_buffer
    
    def check_reversal_days(self, df: pd.DataFrame, entry_idx: int, 
                              current_idx: int, direction: str) -> bool:
        """Check for 2 consecutive reversal days"""
        if current_idx - entry_idx < 2:
            return False
        
        candle1 = df.iloc[current_idx - 1]
        candle2 = df.iloc[current_idx]
        
        if direction == 'long':
            reversal1 = candle1['close'] < candle1['open']
            reversal2 = candle2['close'] < candle2['open']
        else:
            reversal1 = candle1['close'] > candle1['open']
            reversal2 = candle2['close'] > candle2['open']
        
        return reversal1 and reversal2
    
    def update_account_balance(self, new_balance: float):
        """Update account balance"""
        if self.verbose:
            print(f"[SidMethod] Account balance updated: ${self.account_balance:,.2f} -> ${new_balance:,.2f}")
        self.account_balance = new_balance
    
    def update_trade_result(self, result: str, loss_amount: float = 0):
        """Update tracking after trade closes"""
        if result == 'loss':
            self.consecutive_losses += 1
            self.daily_loss += loss_amount
            if self.verbose:
                print(f"[SidMethod] Trade LOSS: ${loss_amount:.2f}, consecutive losses: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            if self.verbose:
                print(f"[SidMethod] Trade WIN! consecutive losses reset")
    
    def reset_daily(self):
        """Reset daily counters"""
        if self.verbose:
            print(f"[SidMethod] Daily reset: losses={self.daily_loss:.2f}, consecutive={self.consecutive_losses}")
        self.daily_loss = 0.0
        self.consecutive_losses = 0