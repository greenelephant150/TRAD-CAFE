"""
Sid Naiman's SID Method - Complete Strategy Implementation
Based on meticulous review of 50+ hours of Sid's sessions

Core Principles:
1. RSI Signal (oversold <30, overbought >70)
2. MACD Alignment (both pointing same direction)
3. Earnings Check (no trades within 14 calendar days before earnings)
4. Entry at current price when aligned
5. Stop Loss: 
   - For oversold: lowest low between signal and entry, rounded DOWN
   - For overbought: highest high between signal and entry, rounded UP
6. Take Profit: RSI 50 (set alert)
7. Exit after 2 consecutive reversal days
8. Position Size Calculator: 0.5% to 2% risk per trade
9. Daily charts only
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SidMethod:
    """
    Complete implementation of Sid Naiman's SID Method
    """
    
    def __init__(self, account_balance: float = 10000):
        self.account_balance = account_balance
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.rsi_target = 50
        self.earnings_days_buffer = 14  # No trades within 14 days before earnings
        self.max_risk_percent = 2.0
        self.min_risk_percent = 0.5
        self.default_risk_percent = 1.0
        self.max_consecutive_losses = 3
        
        # Track open positions
        self.open_positions = []
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_loss = 0.0
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def check_rsi_signal(self, rsi_value: float) -> str:
        """
        Check RSI signal
        Returns: 'oversold', 'overbought', or 'neutral'
        """
        if rsi_value < self.rsi_oversold:
            return 'oversold'
        elif rsi_value > self.rsi_overbought:
            return 'overbought'
        else:
            return 'neutral'
    
    def check_macd_alignment(self, macd_df: pd.DataFrame, current_idx: int, 
                               signal_type: str) -> bool:
        """
        Check if MACD aligns with RSI signal
        For oversold: MACD should be pointing up
        For overbought: MACD should be pointing down
        """
        if current_idx < 2:
            return False
        
        current_macd = macd_df['macd'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            # MACD should be increasing
            return current_macd > prev_macd
        elif signal_type == 'overbought':
            # MACD should be decreasing
            return current_macd < prev_macd
        else:
            return False
    
    def check_macd_cross(self, macd_df: pd.DataFrame, current_idx: int,
                           signal_type: str) -> bool:
        """
        Check if MACD has crossed the signal line
        Provides stronger confirmation (less risk, less profit)
        """
        if current_idx < 1:
            return False
        
        current_macd = macd_df['macd'].iloc[current_idx]
        current_signal = macd_df['signal'].iloc[current_idx]
        prev_macd = macd_df['macd'].iloc[current_idx - 1]
        prev_signal = macd_df['signal'].iloc[current_idx - 1]
        
        if signal_type == 'oversold':
            # MACD crossing above signal line
            return (prev_macd <= prev_signal and current_macd > current_signal)
        elif signal_type == 'overbought':
            # MACD crossing below signal line
            return (prev_macd >= prev_signal and current_macd < current_signal)
        else:
            return False
    
    def check_earnings(self, earnings_date: Optional[datetime], entry_date: datetime) -> bool:
        """
        Check if entry date is at least 14 calendar days before earnings
        Sid's rule: no trades within 14 days BEFORE earnings
        """
        if earnings_date is None:
            return True
        
        days_before = (earnings_date - entry_date).days
        return days_before >= self.earnings_days_buffer
    
    def calculate_stop_loss(self, df: pd.DataFrame, 
                             signal_date: datetime,
                             entry_date: datetime,
                             signal_type: str) -> float:
        """
        Calculate stop loss using Sid's rules:
        - Oversold: lowest low between signal and entry, rounded DOWN
        - Overbought: highest high between signal and entry, rounded UP
        """
        # Get data between signal and entry
        mask = (df.index >= signal_date) & (df.index <= entry_date)
        period_df = df.loc[mask]
        
        if period_df.empty:
            return 0.0
        
        if signal_type == 'oversold':
            # For oversold: lowest low, rounded down
            lowest_low = period_df['low'].min()
            stop_loss = np.floor(lowest_low)
        else:
            # For overbought: highest high, rounded up
            highest_high = period_df['high'].max()
            stop_loss = np.ceil(highest_high)
        
        return float(stop_loss)
    
    def calculate_position_size(self, entry_price: float, stop_loss: float,
                                  risk_percent: float = None) -> Dict:
        """
        Sid's position size calculator
        Risk per trade: 0.5% to 2% of account
        """
        if risk_percent is None:
            # Adjust risk based on consecutive losses
            if self.consecutive_losses == 0:
                risk_percent = self.default_risk_percent
            elif self.consecutive_losses == 1:
                risk_percent = 0.75
            elif self.consecutive_losses == 2:
                risk_percent = 0.5
            else:
                risk_percent = 0.25  # After 3+ losses, trade very small
        
        risk_percent = max(self.min_risk_percent, min(risk_percent, self.max_risk_percent))
        
        risk_amount = self.account_balance * (risk_percent / 100)
        
        # Calculate risk per unit
        if entry_price > stop_loss:  # Long trade
            risk_per_unit = entry_price - stop_loss
            direction = 'long'
        else:  # Short trade
            risk_per_unit = stop_loss - entry_price
            direction = 'short'
        
        if risk_per_unit <= 0:
            return {'error': 'Invalid stop loss'}
        
        # Calculate shares/units
        units = risk_amount / risk_per_unit
        
        # Round down to stay within risk (Sid's rule)
        units = np.floor(units)
        
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
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                                direction: str, method: str = 'rsi_50') -> float:
        """
        Calculate take profit levels
        Primary: RSI 50 (monitored by alert)
        Alternative: Add 4 points for stocks under $200, 8 points for over $200
        """
        if method == 'rsi_50':
            # RSI 50 will be monitored separately
            # Return a conservative estimate for order placement
            risk_distance = abs(entry_price - stop_loss)
            if direction == 'long':
                return entry_price + (risk_distance * 1.0)  # 1:1 risk/reward
            else:
                return entry_price - (risk_distance * 1.0)
        
        elif method == 'points':
            # For stocks under $200: add 4 points
            # For stocks over $200: add 8 points
            if entry_price < 200:
                points = 4
            else:
                points = 8
            
            if direction == 'long':
                return entry_price + points
            else:
                return entry_price - points
        
        else:
            return 0.0
    
    def check_reversal_days(self, df: pd.DataFrame, entry_idx: int, 
                              current_idx: int, direction: str) -> bool:
        """
        Check if there have been 2 consecutive reversal days
        If yes, exit the trade (Sid's rule for volatile markets)
        """
        if current_idx - entry_idx < 2:
            return False
        
        # Check last 2 candles
        candle1 = df.iloc[current_idx - 1]
        candle2 = df.iloc[current_idx]
        
        if direction == 'long':
            # For long trades, reversal means price going down
            reversal1 = candle1['close'] < candle1['open']
            reversal2 = candle2['close'] < candle2['open']
        else:
            # For short trades, reversal means price going up
            reversal1 = candle1['close'] > candle1['open']
            reversal2 = candle2['close'] > candle2['open']
        
        return reversal1 and reversal2
    
    def find_trade_opportunities(self, df: pd.DataFrame, 
                                   earnings_dates: Dict[str, datetime] = None) -> List[Dict]:
        """
        Find all trade opportunities in the dataframe using Sid's rules
        """
        if df.empty or len(df) < 50:
            return []
        
        # Calculate indicators
        df = df.copy()
        df['rsi'] = self.calculate_rsi(df)
        macd_df = self.calculate_macd(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        df['macd_hist'] = macd_df['histogram']
        
        opportunities = []
        
        for i in range(20, len(df) - 1):
            current_date = df.index[i]
            rsi_value = df['rsi'].iloc[i]
            
            # Check RSI signal
            signal_type = self.check_rsi_signal(rsi_value)
            if signal_type == 'neutral':
                continue
            
            # Check earnings (if available)
            if earnings_dates and current_date in earnings_dates:
                earnings_date = earnings_dates.get(current_date)
                if not self.check_earnings(earnings_date, current_date):
                    continue
            
            # Check MACD alignment
            aligned = self.check_macd_alignment(macd_df, i, signal_type)
            if not aligned:
                continue
            
            # Check MACD cross (for stronger confirmation)
            crossed = self.check_macd_cross(macd_df, i, signal_type)
            
            # Find the signal date (first time RSI crossed threshold)
            signal_date = self._find_signal_date(df, i, signal_type)
            
            # Calculate stop loss
            stop_loss = self.calculate_stop_loss(df, signal_date, current_date, signal_type)
            
            # Determine direction
            direction = 'long' if signal_type == 'oversold' else 'short'
            
            # Calculate entry price (current price at alignment)
            entry_price = df['close'].iloc[i]
            
            # Calculate take profit (1:1 risk/reward for order placement)
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
        
        return opportunities
    
    def _find_signal_date(self, df: pd.DataFrame, entry_idx: int, signal_type: str) -> datetime:
        """Find the date when RSI first crossed the threshold"""
        rsi_values = df['rsi'].iloc[:entry_idx + 1]
        
        if signal_type == 'oversold':
            # Find first time RSI went below 30
            mask = rsi_values < self.rsi_oversold
        else:
            # Find first time RSI went above 70
            mask = rsi_values > self.rsi_overbought
        
        if mask.any():
            signal_idx = mask[mask].index[-1]
            return signal_idx
        else:
            return df.index[entry_idx]
    
    def update_account_balance(self, new_balance: float):
        """Update account balance"""
        self.account_balance = new_balance
    
    def update_trade_result(self, result: str, loss_amount: float = 0):
        """Update tracking after trade closes"""
        if result == 'loss':
            self.consecutive_losses += 1
            self.daily_loss += loss_amount
        else:
            self.consecutive_losses = 0
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_loss = 0.0
        self.consecutive_losses = 0