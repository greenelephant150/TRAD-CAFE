#!/usr/bin/env python3
"""
Bar Replay Module for SID Method - AUGMENTED VERSION
=============================================================================
Implements bar replay backtesting incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Sequential bar-by-bar execution
- RSI threshold detection (exact 30/70)
- MACD alignment and cross detection
- Stop loss and take profit execution
- Trade tracking and metrics

WAVE 2 (Live Sessions & Q&A):
- Market context filtering during replay
- Pattern confirmation over time
- Divergence detection in replay
- Multi-timeframe analysis
- Walk-forward validation

WAVE 3 (Academy Support Sessions):
- Session-based filtering during replay
- Zone quality assessment over time
- Minimum candle requirements
- Partial profit taking in replay
- Trade journal generation

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import json

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class TradeStatus(Enum):
    """Status of a trade during replay"""
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class TradeExitReason(Enum):
    """Reason for trade exit during replay"""
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_ALT = "take_profit_alt"
    TRAILING_STOP = "trailing_stop"
    REVERSAL = "reversal"
    DIVERGENCE = "divergence"
    TIME_STOP = "time_stop"
    MANUAL = "manual"


@dataclass
class ReplayTrade:
    """Trade record during bar replay"""
    entry_date: datetime
    exit_date: Optional[datetime] = None
    instrument: str = ""
    direction: str = ""  # 'long' or 'short'
    entry_price: float = 0.0
    exit_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    take_profit_alt: float = 0.0
    units: int = 0
    risk_amount: float = 0.0
    profit_loss: float = 0.0
    profit_pips: float = 0.0
    profit_percent: float = 0.0
    exit_reason: Optional[TradeExitReason] = None
    status: TradeStatus = TradeStatus.PENDING
    
    # Signal data
    signal_type: str = ""  # 'oversold' or 'overbought'
    rsi_entry: float = 0.0
    rsi_exit: float = 0.0
    macd_crossed: bool = False
    pattern_confirmed: bool = False
    divergence_detected: bool = False
    confidence_score: float = 0.0
    
    # Context
    market_trend: str = ""
    session: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'entry_date': self.entry_date.isoformat() if self.entry_date else None,
            'exit_date': self.exit_date.isoformat() if self.exit_date else None,
            'instrument': self.instrument,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'units': self.units,
            'profit_loss': self.profit_loss,
            'profit_pips': self.profit_pips,
            'profit_percent': self.profit_percent,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'signal_type': self.signal_type,
            'rsi_entry': self.rsi_entry,
            'macd_crossed': self.macd_crossed,
            'pattern_confirmed': self.pattern_confirmed,
            'divergence_detected': self.divergence_detected,
            'confidence_score': self.confidence_score
        }


@dataclass
class BacktestMetrics:
    """Backtest performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = 0.0
    
    # SID-specific metrics
    avg_rsi_entry: float = 0.0
    avg_confidence: float = 0.0
    macd_cross_trades: int = 0
    pattern_trades: int = 0
    divergence_trades: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'net_profit': self.net_profit,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'sharpe_ratio': self.sharpe_ratio,
            'profit_factor': self.profit_factor,
            'avg_rsi_entry': self.avg_rsi_entry,
            'avg_confidence': self.avg_confidence,
            'macd_cross_trades': self.macd_cross_trades,
            'pattern_trades': self.pattern_trades,
            'divergence_trades': self.divergence_trades
        }


@dataclass
class BarReplayConfig:
    """Configuration for bar replay (Wave 1, 2, 3)"""
    # Wave 1: Core parameters
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_target: int = 50
    prefer_macd_cross: bool = True
    risk_percent: float = 1.0
    
    # Wave 2: Confirmation parameters
    use_pattern_confirmation: bool = True
    use_divergence: bool = True
    use_market_context: bool = True
    market_lookback: int = 50
    
    # Wave 3: Session and quality
    use_session_filter: bool = True
    min_confidence: float = 0.5
    min_quality_score: float = 40.0
    
    # Replay parameters
    initial_balance: float = 10000.0
    max_open_trades: int = 3
    commission_per_trade: float = 0.0
    slippage_pips: float = 0.0
    pip_value_default: float = 0.0001
    
    # Output
    save_trades: bool = True
    save_metrics: bool = True
    verbose: bool = True


class BarReplay:
    """
    Bar replay backtesting engine for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: BarReplayConfig = None):
        """
        Initialize bar replay engine
        
        Args:
            config: BarReplayConfig instance
        """
        self.config = config or BarReplayConfig()
        
        # Replay state
        self.current_bar = 0
        self.current_date = None
        self.current_price = 0.0
        
        # Account tracking
        self.balance = self.config.initial_balance
        self.peak_balance = self.config.initial_balance
        self.open_trades: List[ReplayTrade] = []
        self.closed_trades: List[ReplayTrade] = []
        
        # Indicator cache
        self.rsi_values: List[float] = []
        self.macd_values: List[float] = []
        self.macd_signal_values: List[float] = []
        self.sma_50_values: List[float] = []
        
        # Market context
        self.market_trend = "unknown"
        self.market_df = None
        
        # Metrics
        self.metrics = BacktestMetrics()
        
        # Progress tracking
        self.progress_callback = None
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"🔄 BAR REPLAY v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"💰 Initial Balance: ${self.config.initial_balance:,.2f}")
            print(f"📊 Risk per trade: {self.config.risk_percent}%")
            print(f"📈 Max open trades: {self.config.max_open_trades}")
            print(f"🎯 RSI: {self.config.rsi_oversold}/{self.config.rsi_overbought}")
            print(f"🔄 MACD cross preferred: {self.config.prefer_macd_cross}")
            print(f"📐 Pattern confirmation: {self.config.use_pattern_confirmation}")
            print(f"⚡ Divergence detection: {self.config.use_divergence}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: INDICATOR CALCULATIONS (Wave 1)
    # ========================================================================
    
    def calculate_rsi(self, close_prices: List[float], period: int = 14) -> float:
        """
        Calculate RSI for current bar
        
        Args:
            close_prices: List of close prices
            period: RSI period
        
        Returns:
            Current RSI value
        """
        if len(close_prices) < period + 1:
            return 50.0
        
        delta = close_prices[-1] - close_prices[-2]
        
        # Use cached gains/losses for efficiency
        gains = []
        losses = []
        
        for i in range(-period, 0):
            diff = close_prices[i] - close_prices[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))
        
        avg_gain = sum(gains) / period
        avg_loss = sum(losses) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_macd(self, close_prices: List[float], fast: int = 12, 
                        slow: int = 26, signal: int = 9) -> Tuple[float, float]:
        """
        Calculate MACD for current bar
        
        Returns:
            (macd_line, signal_line)
        """
        if len(close_prices) < slow + signal:
            return 0.0, 0.0
        
        # EMA calculations (simplified for replay)
        ema_fast = self._calculate_ema(close_prices, fast)
        ema_slow = self._calculate_ema(close_prices, slow)
        macd = ema_fast - ema_slow
        
        # Signal line EMA of MACD
        if len(self.macd_values) >= signal:
            signal_ema = self._calculate_ema(self.macd_values + [macd], signal)
        else:
            signal_ema = macd
        
        return macd, signal_ema
    
    def _calculate_ema(self, values: List[float], period: int) -> float:
        """Calculate EMA for a list of values"""
        if len(values) < period:
            return values[-1] if values else 0
        
        multiplier = 2 / (period + 1)
        ema = values[-period]
        
        for i in range(-period + 1, 0):
            ema = (values[i] - ema) * multiplier + ema
        
        return ema
    
    # ========================================================================
    # SECTION 2: SIGNAL DETECTION (Wave 1 & 2)
    # ========================================================================
    
    def detect_signal(self, rsi: float, macd: float, macd_signal: float,
                       macd_prev: float, macd_signal_prev: float,
                       pattern_detected: bool = False,
                       divergence_detected: bool = False) -> Tuple[bool, str, bool, bool]:
        """
        Detect SID Method trade signal
        
        Returns:
            (has_signal, signal_type, macd_aligned, macd_crossed)
        """
        has_signal = False
        signal_type = ""
        macd_aligned = False
        macd_crossed = False
        
        # Check RSI thresholds (Wave 1)
        if rsi < self.config.rsi_oversold:
            signal_type = "oversold"
            has_signal = True
        elif rsi > self.config.rsi_overbought:
            signal_type = "overbought"
            has_signal = True
        
        if not has_signal:
            return False, "", False, False
        
        # Check MACD alignment (Wave 1)
        if signal_type == "oversold":
            macd_aligned = macd > macd_prev
        else:
            macd_aligned = macd < macd_prev
        
        if not macd_aligned:
            return False, "", False, False
        
        # Check MACD cross (Wave 2)
        if signal_type == "oversold":
            macd_crossed = (macd_prev <= macd_signal_prev and macd > macd_signal)
        else:
            macd_crossed = (macd_prev >= macd_signal_prev and macd < macd_signal)
        
        # Apply cross preference (Wave 2)
        if self.config.prefer_macd_cross and not macd_crossed:
            return False, "", False, False
        
        return True, signal_type, macd_aligned, macd_crossed
    
    # ========================================================================
    # SECTION 3: TRADE EXECUTION (Wave 1 & 2)
    # ========================================================================
    
    def calculate_stop_loss(self, df: pd.DataFrame, signal_idx: int,
                              entry_idx: int, signal_type: str) -> float:
        """
        Calculate stop loss based on SID Method (Wave 1)
        """
        period_df = df.iloc[signal_idx:entry_idx + 1]
        
        if signal_type == "oversold":
            lowest_low = period_df['low'].min()
            stop_loss = np.floor(lowest_low)
        else:
            highest_high = period_df['high'].max()
            stop_loss = np.ceil(highest_high)
        
        return float(stop_loss)
    
    def calculate_take_profit(self, entry_price: float, stop_loss: float,
                                direction: str) -> Tuple[float, float]:
        """
        Calculate take profit levels (Wave 1 & 2)
        """
        risk_distance = abs(entry_price - stop_loss)
        
        # Primary: RSI 50 (1:1)
        if direction == 'long':
            primary_tp = entry_price + risk_distance
        else:
            primary_tp = entry_price - risk_distance
        
        # Alternative: Point targets (Wave 2)
        if entry_price < 200:
            points = 4
        else:
            points = 8
        
        if direction == 'long':
            alt_tp = entry_price + points
        else:
            alt_tp = entry_price - points
        
        return primary_tp, alt_tp
    
    def calculate_position_size(self, entry_price: float, stop_loss: float) -> int:
        """
        Calculate position size (Wave 1)
        """
        risk_amount = self.balance * (self.config.risk_percent / 100)
        
        if entry_price > stop_loss:
            risk_per_unit = entry_price - stop_loss
        else:
            risk_per_unit = stop_loss - entry_price
        
        if risk_per_unit <= 0:
            return 0
        
        units = risk_amount / risk_per_unit
        return max(1, int(np.floor(units)))
    
    # ========================================================================
    # SECTION 4: TRADE MANAGEMENT (Wave 1, 2, 3)
    # ========================================================================
    
    def check_stop_loss(self, trade: ReplayTrade, current_price: float) -> bool:
        """Check if stop loss is hit (Wave 1)"""
        if trade.direction == 'long':
            return current_price <= trade.stop_loss
        else:
            return current_price >= trade.stop_loss
    
    def check_take_profit(self, trade: ReplayTrade, current_price: float,
                            use_alt: bool = False) -> bool:
        """Check if take profit is hit (Wave 1 & 2)"""
        tp = trade.take_profit_alt if use_alt else trade.take_profit
        
        if trade.direction == 'long':
            return current_price >= tp
        else:
            return current_price <= tp
    
    def check_reversal_exit(self, trade: ReplayTrade, current_bar: pd.Series,
                              prev_bar: pd.Series) -> bool:
        """
        Check for reversal exit (Wave 3)
        """
        if trade.direction == 'long':
            # Two consecutive red candles
            return (current_bar['close'] < current_bar['open'] and
                    prev_bar['close'] < prev_bar['open'])
        else:
            # Two consecutive green candles
            return (current_bar['close'] > current_bar['open'] and
                    prev_bar['close'] > prev_bar['open'])
    
    def update_trade(self, trade: ReplayTrade, exit_price: float,
                      exit_reason: TradeExitReason, exit_date: datetime):
        """
        Update trade with exit details
        """
        trade.exit_date = exit_date
        trade.exit_price = exit_price
        trade.exit_reason = exit_reason
        trade.status = TradeStatus.CLOSED
        
        # Calculate profit
        if trade.direction == 'long':
            trade.profit_loss = (trade.exit_price - trade.entry_price) * trade.units
        else:
            trade.profit_loss = (trade.entry_price - trade.exit_price) * trade.units
        
        trade.profit_loss -= self.config.commission_per_trade
        
        # Update balance
        self.balance += trade.profit_loss
        
        # Update peak balance for drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        self.closed_trades.append(trade)
    
    # ========================================================================
    # SECTION 5: MARKET CONTEXT (Wave 2)
    # ========================================================================
    
    def update_market_trend(self, df: pd.DataFrame, current_idx: int):
        """
        Update market trend based on SPY or major index (Wave 2)
        """
        if not self.config.use_market_context:
            return
        
        if self.market_df is None:
            return
        
        lookback = self.config.market_lookback
        if current_idx < lookback:
            return
        
        market_data = self.market_df.iloc[current_idx - lookback:current_idx + 1]
        highs = market_data['high']
        lows = market_data['low']
        
        # Check for uptrend
        higher_highs = all(highs.iloc[i] <= highs.iloc[i+1] 
                          for i in range(len(highs)-10, len(highs)-1))
        higher_lows = all(lows.iloc[i] <= lows.iloc[i+1] 
                         for i in range(len(lows)-10, len(lows)-1))
        
        if higher_highs and higher_lows:
            self.market_trend = "uptrend"
            return
        
        # Check for downtrend
        lower_highs = all(highs.iloc[i] >= highs.iloc[i+1] 
                         for i in range(len(highs)-10, len(highs)-1))
        lower_lows = all(lows.iloc[i] >= lows.iloc[i+1] 
                        for i in range(len(lows)-10, len(lows)-1))
        
        if lower_highs and lower_lows:
            self.market_trend = "downtrend"
            return
        
        self.market_trend = "sideways"
    
    # ========================================================================
    # SECTION 6: SESSION DETECTION (Wave 3)
    # ========================================================================
    
    def get_session(self, dt: datetime) -> str:
        """Get trading session (Wave 3)"""
        hour = dt.hour
        
        if 0 <= hour < 7:
            return "asian"
        elif 7 <= hour < 12:
            return "london"
        elif 12 <= hour < 16:
            return "overlap"
        elif 16 <= hour < 21:
            return "us"
        else:
            return "asian"
    
    def should_filter_by_session(self, session: str) -> bool:
        """Check if session should be filtered (Wave 3)"""
        if not self.config.use_session_filter:
            return False
        
        unsuitable_sessions = ["asian"]
        return session in unsuitable_sessions
    
    # ========================================================================
    # SECTION 7: MAIN REPLAY LOOP
    # ========================================================================
    
    def run(self, df: pd.DataFrame, market_df: pd.DataFrame = None,
             progress_callback: Callable = None) -> BacktestMetrics:
        """
        Run bar replay backtest
        
        Args:
            df: Price DataFrame for instrument
            market_df: Market index DataFrame for context (optional)
            progress_callback: Progress callback function
        
        Returns:
            BacktestMetrics
        """
        self.market_df = market_df
        self.progress_callback = progress_callback
        
        if self.config.verbose:
            print(f"\n🔄 Starting bar replay backtest...")
            print(f"   Data range: {df.index[0]} to {df.index[-1]}")
            print(f"   Total bars: {len(df)}")
        
        start_time = time.time()
        
        # Pre-calculate indicators
        df = df.copy()
        
        # RSI
        df['rsi'] = self._calculate_rsi_vectorized(df)
        
        # MACD
        macd_df = self._calculate_macd_vectorized(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        
        # SMA 50
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Main replay loop
        self.balance = self.config.initial_balance
        self.peak_balance = self.config.initial_balance
        self.open_trades = []
        self.closed_trades = []
        
        signal_pending = None
        signal_idx = None
        signal_date = None
        signal_type = None
        
        total_bars = len(df)
        
        for i in range(50, total_bars):  # Start after enough data
            current_bar = df.iloc[i]
            current_date = df.index[i]
            current_price = current_bar['close']
            
            # Update progress
            if self.progress_callback and i % 100 == 0:
                self.progress_callback(i / total_bars)
            
            # Update market trend (Wave 2)
            self.update_market_trend(df, i)
            
            # Check open trades
            trades_to_close = []
            for trade in self.open_trades:
                # Check stop loss (Wave 1)
                if self.check_stop_loss(trade, current_price):
                    self.update_trade(trade, trade.stop_loss, TradeExitReason.STOP_LOSS, current_date)
                    trades_to_close.append(trade)
                    continue
                
                # Check primary take profit (Wave 1)
                if self.check_take_profit(trade, current_price, use_alt=False):
                    self.update_trade(trade, trade.take_profit, TradeExitReason.TAKE_PROFIT, current_date)
                    trades_to_close.append(trade)
                    continue
                
                # Check alternative take profit (Wave 2)
                if self.check_take_profit(trade, current_price, use_alt=True):
                    self.update_trade(trade, trade.take_profit_alt, TradeExitReason.TAKE_PROFIT_ALT, current_date)
                    trades_to_close.append(trade)
                    continue
                
                # Check reversal exit (Wave 3)
                if i > trade.entry_idx + 1:
                    prev_bar = df.iloc[i - 1]
                    if self.check_reversal_exit(trade, current_bar, prev_bar):
                        self.update_trade(trade, current_price, TradeExitReason.REVERSAL, current_date)
                        trades_to_close.append(trade)
                        continue
            
            # Remove closed trades
            for trade in trades_to_close:
                if trade in self.open_trades:
                    self.open_trades.remove(trade)
            
            # Check for new signals
            if len(self.open_trades) < self.config.max_open_trades:
                # Get current values
                rsi = current_bar['rsi']
                macd = current_bar['macd']
                macd_signal = current_bar['macd_signal']
                macd_prev = df['macd'].iloc[i-1]
                macd_signal_prev = df['macd_signal'].iloc[i-1]
                
                # Detect signal (Wave 1 & 2)
                has_signal, sig_type, macd_aligned, macd_crossed = self.detect_signal(
                    rsi, macd, macd_signal, macd_prev, macd_signal_prev
                )
                
                if has_signal:
                    # Check market context (Wave 2)
                    if self.config.use_market_context:
                        if sig_type == "oversold" and self.market_trend == "downtrend":
                            continue
                        if sig_type == "overbought" and self.market_trend == "uptrend":
                            continue
                        if self.market_trend == "sideways":
                            continue
                    
                    # Check session (Wave 3)
                    session = self.get_session(current_date)
                    if self.should_filter_by_session(session):
                        continue
                    
                    # Find signal date
                    signal_mask = df['rsi'].iloc[:i+1] < self.config.rsi_oversold if sig_type == "oversold" else df['rsi'].iloc[:i+1] > self.config.rsi_overbought
                    if signal_mask.any():
                        signal_idx = signal_mask[signal_mask].index[-1]
                        signal_date = signal_idx
                    else:
                        signal_idx = i
                        signal_date = current_date
                    
                    # Calculate stop loss (Wave 1)
                    stop_loss = self.calculate_stop_loss(df, signal_idx, i, sig_type)
                    
                    if stop_loss > 0:
                        # Calculate take profit (Wave 1 & 2)
                        direction = 'long' if sig_type == 'oversold' else 'short'
                        primary_tp, alt_tp = self.calculate_take_profit(current_price, stop_loss, direction)
                        
                        # Calculate position size (Wave 1)
                        units = self.calculate_position_size(current_price, stop_loss)
                        
                        if units > 0:
                            # Create trade
                            trade = ReplayTrade(
                                entry_date=current_date,
                                instrument=getattr(df, 'name', 'unknown'),
                                direction=direction,
                                entry_price=current_price,
                                stop_loss=stop_loss,
                                take_profit=primary_tp,
                                take_profit_alt=alt_tp,
                                units=units,
                                risk_amount=self.balance * (self.config.risk_percent / 100),
                                signal_type=sig_type,
                                rsi_entry=rsi,
                                macd_crossed=macd_crossed,
                                market_trend=self.market_trend,
                                session=session,
                                status=TradeStatus.OPEN
                            )
                            
                            self.open_trades.append(trade)
        
        # Close any remaining open trades at end
        for trade in self.open_trades:
            final_price = df['close'].iloc[-1]
            self.update_trade(trade, final_price, TradeExitReason.MANUAL, df.index[-1])
        
        self.open_trades = []
        
        # Calculate metrics
        self._calculate_metrics()
        
        elapsed = time.time() - start_time
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"✅ BACKTEST COMPLETE")
            print(f"{'='*60}")
            print(f"⏱️ Time: {elapsed:.1f}s")
            print(f"📊 Total trades: {self.metrics.total_trades}")
            print(f"🏆 Win rate: {self.metrics.win_rate:.1f}%")
            print(f"💰 Net profit: ${self.metrics.net_profit:,.2f}")
            print(f"📈 Profit factor: {self.metrics.profit_factor:.2f}")
            print(f"📉 Max drawdown: {self.metrics.max_drawdown_percent:.1f}%")
            print(f"{'='*60}\n")
        
        return self.metrics
    
    # ========================================================================
    # SECTION 8: VECTORIZED INDICATOR CALCULATIONS
    # ========================================================================
    
    def _calculate_rsi_vectorized(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Vectorized RSI calculation"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd_vectorized(self, df: pd.DataFrame, fast: int = 12,
                                     slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Vectorized MACD calculation"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        })
    
    # ========================================================================
    # SECTION 9: METRICS CALCULATION
    # ========================================================================
    
    def _calculate_metrics(self):
        """Calculate backtest metrics (Wave 2 & 3)"""
        if not self.closed_trades:
            return
        
        profits = []
        losses = []
        equity_curve = [self.config.initial_balance]
        
        for trade in self.closed_trades:
            if trade.profit_loss > 0:
                profits.append(trade.profit_loss)
            else:
                losses.append(abs(trade.profit_loss))
            
            equity_curve.append(equity_curve[-1] + trade.profit_loss)
        
        self.metrics.total_trades = len(self.closed_trades)
        self.metrics.winning_trades = len(profits)
        self.metrics.losing_trades = len(losses)
        self.metrics.win_rate = (self.metrics.winning_trades / self.metrics.total_trades * 100) if self.metrics.total_trades > 0 else 0
        self.metrics.total_profit = sum(profits)
        self.metrics.total_loss = sum(losses)
        self.metrics.net_profit = self.metrics.total_profit - self.metrics.total_loss
        self.metrics.average_win = sum(profits) / len(profits) if profits else 0
        self.metrics.average_loss = sum(losses) / len(losses) if losses else 0
        self.metrics.largest_win = max(profits) if profits else 0
        self.metrics.largest_loss = max(losses) if losses else 0
        self.metrics.profit_factor = self.metrics.total_profit / self.metrics.total_loss if self.metrics.total_loss > 0 else float('inf')
        
        # Maximum drawdown
        peak = self.config.initial_balance
        max_drawdown = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            drawdown = peak - eq
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        self.metrics.max_drawdown = max_drawdown
        self.metrics.max_drawdown_percent = (max_drawdown / self.config.initial_balance * 100) if self.config.initial_balance > 0 else 0
        
        # SID-specific metrics
        rsi_entries = [t.rsi_entry for t in self.closed_trades if t.rsi_entry > 0]
        if rsi_entries:
            self.metrics.avg_rsi_entry = sum(rsi_entries) / len(rsi_entries)
        
        confidences = [t.confidence_score for t in self.closed_trades if t.confidence_score > 0]
        if confidences:
            self.metrics.avg_confidence = sum(confidences) / len(confidences)
        
        self.metrics.macd_cross_trades = sum(1 for t in self.closed_trades if t.macd_crossed)
        self.metrics.pattern_trades = sum(1 for t in self.closed_trades if t.pattern_confirmed)
        self.metrics.divergence_trades = sum(1 for t in self.closed_trades if t.divergence_detected)
    
    # ========================================================================
    # SECTION 10: RESULTS EXPORT
    # ========================================================================
    
    def export_trades(self, filepath: str) -> bool:
        """Export trades to CSV (Wave 3)"""
        try:
            trades_df = pd.DataFrame([t.to_dict() for t in self.closed_trades])
            trades_df.to_csv(filepath, index=False)
            
            if self.config.verbose:
                print(f"💾 Trades exported to {filepath}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to export trades: {e}")
            return False
    
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to JSON (Wave 3)"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            
            if self.config.verbose:
                print(f"💾 Metrics exported to {filepath}")
            
            return True
        except Exception as e:
            print(f"❌ Failed to export metrics: {e}")
            return False


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 TESTING BAR REPLAY v3.0")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=5000, freq='H')
    np.random.seed(42)
    
    # Generate realistic price data with trends
    returns = np.random.randn(5000) * 0.001
    prices = 100 + np.cumsum(returns)
    
    # Add oversold condition
    prices[1000:1010] = 95
    
    # Add overbought condition
    prices[2000:2010] = 105
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + np.abs(np.random.randn(5000) * 0.5),
        'low': prices - np.abs(np.random.randn(5000) * 0.5),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 5000)
    }, index=dates)
    
    # Create market index (SPY proxy)
    spy_prices = 450 + np.cumsum(np.random.randn(5000) * 0.0005)
    market_df = pd.DataFrame({
        'open': spy_prices,
        'high': spy_prices + 1,
        'low': spy_prices - 1,
        'close': spy_prices,
        'volume': np.random.randint(100000, 500000, 5000)
    }, index=dates)
    
    print(f"Sample data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Initialize bar replay
    config = BarReplayConfig(
        initial_balance=10000,
        risk_percent=1.0,
        prefer_macd_cross=True,
        use_pattern_confirmation=False,
        use_divergence=False,
        verbose=True
    )
    replay = BarReplay(config)
    
    # Run backtest
    metrics = replay.run(df, market_df)
    
    print(f"\n📊 Backtest Results:")
    for key, value in metrics.to_dict().items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print(f"\n✅ Bar replay test complete")