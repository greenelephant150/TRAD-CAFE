#!/usr/bin/env python3
"""
Pattern Backtesting Script for SID Method - AUGMENTED VERSION
=============================================================================
Backtests pattern performance incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Basic pattern detection (double bottom, double top, head & shoulders)
- RSI and MACD confirmation for patterns
- Stop loss and take profit calculation
- Trade performance metrics

WAVE 2 (Live Sessions & Q&A):
- Pattern quality scoring
- Neckline break confirmation
- Volume confirmation
- Multi-timeframe pattern validation
- Pattern success rate by market context

WAVE 3 (Academy Support Sessions):
- Minimum candle requirements validation
- Rogue wick handling in patterns
- Pattern invalidation rules
- Confluence detection with SID signals
- Zone quality integration

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules
try:
    from config.pattern_rules import PatternRules, PatternRulesConfig, DetectedPattern, PatternType, PatternQuality, PatternDirection
    PATTERN_RULES_AVAILABLE = True
except ImportError:
    PATTERN_RULES_AVAILABLE = False
    print("⚠️ pattern_rules.py not available")

try:
    from sid_method import SidMethod
    SID_AVAILABLE = True
except ImportError:
    SID_AVAILABLE = False
    print("⚠️ sid_method.py not available")

try:
    from src.backtesting.bar_replay import BarReplay, BarReplayConfig
    BAR_REPLAY_AVAILABLE = True
except ImportError:
    BAR_REPLAY_AVAILABLE = False
    print("⚠️ bar_replay.py not available")

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable=None, desc="", **kwargs):
            self.iterable = iterable or []
            self.desc = desc
        def __iter__(self): 
            return iter(self.iterable)
        def update(self, n=1): pass
        def close(self): pass


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PatternTradeResult:
    """Result of a pattern-based trade"""
    pattern_type: str
    pattern_direction: str
    pattern_quality: str
    pattern_quality_score: float
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    actual_profit: float
    actual_profit_pips: float
    actual_profit_percent: float
    was_win: bool
    bars_held: int
    exit_reason: str
    rsi_at_entry: float
    macd_aligned: bool
    volume_confirmed: bool
    sid_signal_present: bool
    
    def to_dict(self) -> Dict:
        return {
            'pattern_type': self.pattern_type,
            'pattern_direction': self.pattern_direction,
            'pattern_quality': self.pattern_quality,
            'pattern_quality_score': self.pattern_quality_score,
            'entry_date': self.entry_date.isoformat(),
            'exit_date': self.exit_date.isoformat(),
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'actual_profit': self.actual_profit,
            'actual_profit_pips': self.actual_profit_pips,
            'actual_profit_percent': self.actual_profit_percent,
            'was_win': self.was_win,
            'bars_held': self.bars_held,
            'exit_reason': self.exit_reason,
            'rsi_at_entry': self.rsi_at_entry,
            'macd_aligned': self.macd_aligned,
            'volume_confirmed': self.volume_confirmed,
            'sid_signal_present': self.sid_signal_present
        }


@dataclass
class PatternBacktestMetrics:
    """Metrics for pattern backtest results"""
    total_patterns: int = 0
    valid_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_profit: float = 0.0
    total_loss: float = 0.0
    net_profit: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    # Pattern-specific metrics
    patterns_by_type: Dict[str, int] = field(default_factory=dict)
    patterns_by_quality: Dict[str, int] = field(default_factory=dict)
    win_rate_by_quality: Dict[str, float] = field(default_factory=dict)
    win_rate_by_type: Dict[str, float] = field(default_factory=dict)
    avg_profit_by_type: Dict[str, float] = field(default_factory=dict)
    
    # SID integration metrics
    sid_confirmed_trades: int = 0
    sid_confirmed_win_rate: float = 0.0
    pattern_only_win_rate: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'total_patterns': self.total_patterns,
            'valid_trades': self.valid_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': self.win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'net_profit': self.net_profit,
            'average_win': self.average_win,
            'average_loss': self.average_loss,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'patterns_by_type': self.patterns_by_type,
            'patterns_by_quality': self.patterns_by_quality,
            'win_rate_by_quality': self.win_rate_by_quality,
            'win_rate_by_type': self.win_rate_by_type,
            'avg_profit_by_type': self.avg_profit_by_type,
            'sid_confirmed_trades': self.sid_confirmed_trades,
            'sid_confirmed_win_rate': self.sid_confirmed_win_rate,
            'pattern_only_win_rate': self.pattern_only_win_rate
        }


@dataclass
class BacktestConfig:
    """Configuration for pattern backtest (Wave 1, 2, 3)"""
    # Pattern detection parameters
    min_pattern_candles: int = 7
    double_bottom_tolerance: float = 0.02
    double_top_tolerance: float = 0.02
    head_shoulders_tolerance: float = 0.03
    require_neckline_break: bool = True
    neckline_break_bars: int = 2
    
    # SID integration
    use_sid_confirmation: bool = True
    require_sid_signal: bool = False  # If True, only trade patterns with SID signal
    sid_rsi_oversold: int = 30
    sid_rsi_overbought: int = 70
    
    # Trade management
    risk_percent: float = 1.0
    use_trailing_stop: bool = False
    max_hold_bars: int = 50
    
    # Data range
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Output
    output_dir: str = "backtest_results/"
    verbose: bool = True


class PatternBacktest:
    """
    Backtest pattern performance for SID Method
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: BacktestConfig = None):
        """
        Initialize pattern backtest engine
        
        Args:
            config: BacktestConfig instance
        """
        self.config = config or BacktestConfig()
        
        # Initialize pattern rules
        if PATTERN_RULES_AVAILABLE:
            pattern_config = PatternRulesConfig(
                min_pattern_candles=self.config.min_pattern_candles,
                double_bottom_tolerance=self.config.double_bottom_tolerance,
                double_top_tolerance=self.config.double_top_tolerance,
                head_shoulders_tolerance=self.config.head_shoulders_tolerance,
                require_neckline_break=self.config.require_neckline_break,
                neckline_break_confirmation_bars=self.config.neckline_break_bars
            )
            self.pattern_rules = PatternRules(pattern_config, verbose=False)
        else:
            self.pattern_rules = None
        
        # Initialize Sid Method
        if SID_AVAILABLE and self.config.use_sid_confirmation:
            self.sid = SidMethod(verbose=False)
        else:
            self.sid = None
        
        # Results storage
        self.trades: List[PatternTradeResult] = []
        self.metrics = PatternBacktestMetrics()
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"📊 PATTERN BACKTEST v3.0 (Fully Augmented)")
            print(f"{'='*60}")
            print(f"📐 Min pattern candles: {self.config.min_pattern_candles}")
            print(f"🎯 Neckline break: {self.config.require_neckline_break} ({self.config.neckline_break_bars} bars)")
            print(f"🔄 SID confirmation: {self.config.use_sid_confirmation}")
            print(f"💰 Risk per trade: {self.config.risk_percent}%")
            print(f"📈 Max hold bars: {self.config.max_hold_bars}")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: DATA PREPARATION
    # ========================================================================
    
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load price data from file
        
        Args:
            filepath: Path to CSV or Parquet file
        
        Returns:
            DataFrame with OHLCV data
        """
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Ensure required columns
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert to numeric
        for col in required:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=required)
        
        # Filter by date range
        if self.config.start_date:
            df = df[df.index >= self.config.start_date]
        if self.config.end_date:
            df = df[df.index <= self.config.end_date]
        
        if self.config.verbose:
            print(f"📊 Loaded {len(df):,} bars")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
        
        return df
    
    # ========================================================================
    # SECTION 2: SID SIGNAL CHECK (Wave 2)
    # ========================================================================
    
    def check_sid_signal(self, df: pd.DataFrame, idx: int, direction: str) -> Tuple[bool, float, bool]:
        """
        Check if SID Method confirms pattern (Wave 2)
        
        Returns:
            (has_signal, rsi_value, macd_aligned)
        """
        if not self.sid:
            return False, 0, False
        
        # Calculate RSI if not already present
        if 'rsi' not in df.columns:
            df['rsi'] = self.sid.calculate_rsi(df, verbose=False)
        
        rsi = df['rsi'].iloc[idx]
        
        if direction == 'bullish':
            # For bullish patterns, look for oversold (RSI < 30)
            has_signal = rsi < self.config.sid_rsi_oversold
        else:
            # For bearish patterns, look for overbought (RSI > 70)
            has_signal = rsi > self.config.sid_rsi_overbought
        
        # Check MACD alignment (simplified)
        macd_aligned = False
        if 'macd' in df.columns and idx > 0:
            if direction == 'bullish':
                macd_aligned = df['macd'].iloc[idx] > df['macd'].iloc[idx - 1]
            else:
                macd_aligned = df['macd'].iloc[idx] < df['macd'].iloc[idx - 1]
        
        return has_signal, rsi, macd_aligned
    
    # ========================================================================
    # SECTION 3: TRADE EXECUTION (Wave 1 & 2)
    # ========================================================================
    
    def execute_pattern_trade(self, df: pd.DataFrame, pattern: DetectedPattern,
                                entry_idx: int) -> Optional[PatternTradeResult]:
        """
        Execute a trade based on pattern detection
        
        Args:
            df: Price DataFrame
            pattern: Detected pattern
            entry_idx: Index where pattern was confirmed
        
        Returns:
            PatternTradeResult or None
        """
        # Determine entry price (break of neckline)
        if pattern.direction == PatternDirection.BULLISH:
            entry_price = pattern.neckline_price
            stop_loss = pattern.stop_loss
            take_profit = pattern.target_price
        else:
            entry_price = pattern.neckline_price
            stop_loss = pattern.stop_loss
            take_profit = pattern.target_price
        
        # Check SID confirmation (Wave 2)
        sid_confirmed, rsi_value, macd_aligned = self.check_sid_signal(df, entry_idx, pattern.direction.value)
        
        if self.config.require_sid_signal and not sid_confirmed:
            if self.config.verbose:
                print(f"  Skipping {pattern.pattern_type.value}: SID signal not confirmed")
            return None
        
        # Track trade progression
        entry_date = df.index[entry_idx]
        current_idx = entry_idx + 1
        bars_held = 0
        
        highest_price = entry_price
        lowest_price = entry_price
        trailing_stop = None
        
        while current_idx < len(df) and bars_held < self.config.max_hold_bars:
            current_price = df['close'].iloc[current_idx]
            current_date = df.index[current_idx]
            
            # Update highest/lowest for trailing stop
            if pattern.direction == PatternDirection.BULLISH:
                if current_price > highest_price:
                    highest_price = current_price
            else:
                if current_price < lowest_price:
                    lowest_price = current_price
            
            # Check stop loss
            if pattern.direction == PatternDirection.BULLISH:
                if current_price <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                    break
            else:
                if current_price >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "stop_loss"
                    break
            
            # Check take profit
            if pattern.direction == PatternDirection.BULLISH:
                if current_price >= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                    break
            else:
                if current_price <= take_profit:
                    exit_price = take_profit
                    exit_reason = "take_profit"
                    break
            
            # Check trailing stop (Wave 2)
            if self.config.use_trailing_stop:
                if pattern.direction == PatternDirection.BULLISH:
                    trail_distance = (highest_price - entry_price) * 0.5
                    trailing_stop = highest_price - trail_distance
                    if current_price <= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "trailing_stop"
                        break
                else:
                    trail_distance = (entry_price - lowest_price) * 0.5
                    trailing_stop = lowest_price + trail_distance
                    if current_price >= trailing_stop:
                        exit_price = trailing_stop
                        exit_reason = "trailing_stop"
                        break
            
            # Check reversal (Wave 3)
            if bars_held > 2:
                if pattern.direction == PatternDirection.BULLISH:
                    if (df['close'].iloc[current_idx-1] < df['open'].iloc[current_idx-1] and
                        df['close'].iloc[current_idx] < df['open'].iloc[current_idx]):
                        exit_price = current_price
                        exit_reason = "reversal"
                        break
                else:
                    if (df['close'].iloc[current_idx-1] > df['open'].iloc[current_idx-1] and
                        df['close'].iloc[current_idx] > df['open'].iloc[current_idx]):
                        exit_price = current_price
                        exit_reason = "reversal"
                        break
            
            current_idx += 1
            bars_held += 1
        
        # If max bars reached, exit at current price
        if current_idx >= len(df) or bars_held >= self.config.max_hold_bars:
            exit_price = df['close'].iloc[current_idx - 1] if current_idx < len(df) else df['close'].iloc[-1]
            exit_reason = "max_bars"
            current_idx = len(df) - 1
        
        exit_date = df.index[current_idx] if current_idx < len(df) else df.index[-1]
        
        # Calculate profit
        if pattern.direction == PatternDirection.BULLISH:
            profit = exit_price - entry_price
        else:
            profit = entry_price - exit_price
        
        # Calculate pip value
        pip_value = 0.0001
        profit_pips = profit / pip_value
        
        # Calculate position size
        risk_amount = 10000 * (self.config.risk_percent / 100)  # Assuming $10,000 account
        risk_distance = abs(entry_price - stop_loss)
        units = risk_amount / risk_distance if risk_distance > 0 else 0
        
        actual_profit = profit * units
        
        # Determine if win
        was_win = actual_profit > 0
        
        # Volume confirmation
        volume_confirmed = False
        if 'volume' in df.columns:
            avg_volume = df['volume'].iloc[entry_idx-20:entry_idx].mean() if entry_idx > 20 else df['volume'].iloc[:entry_idx].mean()
            volume_confirmed = df['volume'].iloc[entry_idx] > avg_volume * 1.5
        
        return PatternTradeResult(
            pattern_type=pattern.pattern_type.value,
            pattern_direction=pattern.direction.value,
            pattern_quality=pattern.quality.value,
            pattern_quality_score=pattern.quality_score,
            entry_date=entry_date,
            exit_date=exit_date,
            entry_price=entry_price,
            exit_price=exit_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            actual_profit=actual_profit,
            actual_profit_pips=profit_pips,
            actual_profit_percent=(profit / entry_price * 100) if entry_price > 0 else 0,
            was_win=was_win,
            bars_held=bars_held,
            exit_reason=exit_reason,
            rsi_at_entry=rsi_value,
            macd_aligned=macd_aligned,
            volume_confirmed=volume_confirmed,
            sid_signal_present=sid_confirmed
        )
    
    # ========================================================================
    # SECTION 4: MAIN BACKTEST LOOP
    # ========================================================================
    
    def run_backtest(self, df: pd.DataFrame) -> PatternBacktestMetrics:
        """
        Run pattern backtest on data
        
        Args:
            df: Price DataFrame
        
        Returns:
            PatternBacktestMetrics
        """
        if self.config.verbose:
            print(f"\n🔄 Running pattern backtest...")
        
        # Pre-calculate indicators
        df = df.copy()
        
        # Calculate RSI for SID confirmation
        if self.sid and 'rsi' not in df.columns:
            df['rsi'] = self.sid.calculate_rsi(df, verbose=False)
        
        # Calculate MACD
        if self.sid and 'macd' not in df.columns:
            macd_df = self.sid.calculate_macd(df, verbose=False)
            df['macd'] = macd_df['macd']
            df['macd_signal'] = macd_df['signal']
        
        # Main loop - scan for patterns at each bar
        min_bars = self.config.min_pattern_candles * 2
        
        # Use tqdm for progress
        if TQDM_AVAILABLE and self.config.verbose:
            iterator = tqdm(range(min_bars, len(df) - self.config.neckline_break_bars), desc="Scanning patterns")
        else:
            iterator = range(min_bars, len(df) - self.config.neckline_break_bars)
        
        for i in iterator:
            # Detect patterns at this bar
            patterns = self.pattern_rules.detect_all_patterns(df.iloc[:i+1], current_idx=i) if self.pattern_rules else []
            
            for pattern in patterns:
                # Check if pattern is valid quality (Wave 3)
                if pattern.quality in [PatternQuality.POOR, PatternQuality.INVALID]:
                    continue
                
                # Check neckline break (Wave 2)
                if self.config.require_neckline_break:
                    is_broken, bars_broken = self.pattern_rules.check_neckline_break(
                        df, pattern, i, df['close'].iloc[i]
                    )
                    if not is_broken:
                        continue
                
                # Execute trade
                trade = self.execute_pattern_trade(df, pattern, i)
                if trade:
                    self.trades.append(trade)
        
        # Calculate metrics
        self._calculate_metrics()
        
        if self.config.verbose:
            print(f"\n✅ Backtest complete: {len(self.trades)} trades executed")
        
        return self.metrics
    
    # ========================================================================
    # SECTION 5: METRICS CALCULATION (Wave 2 & 3)
    # ========================================================================
    
    def _calculate_metrics(self):
        """Calculate backtest metrics"""
        if not self.trades:
            return
        
        # Basic metrics
        self.metrics.total_patterns = len(self.trades)
        self.metrics.valid_trades = len([t for t in self.trades if t.was_win is not None])
        self.metrics.winning_trades = len([t for t in self.trades if t.was_win])
        self.metrics.losing_trades = len([t for t in self.trades if not t.was_win])
        self.metrics.win_rate = (self.metrics.winning_trades / self.metrics.valid_trades * 100) if self.metrics.valid_trades > 0 else 0
        
        # Profit metrics
        wins = [t.actual_profit for t in self.trades if t.was_win]
        losses = [abs(t.actual_profit) for t in self.trades if not t.was_win]
        
        self.metrics.total_profit = sum(wins)
        self.metrics.total_loss = sum(losses)
        self.metrics.net_profit = self.metrics.total_profit - self.metrics.total_loss
        self.metrics.average_win = sum(wins) / len(wins) if wins else 0
        self.metrics.average_loss = sum(losses) / len(losses) if losses else 0
        self.metrics.profit_factor = self.metrics.total_profit / self.metrics.total_loss if self.metrics.total_loss > 0 else float('inf')
        
        # Pattern-specific metrics
        for trade in self.trades:
            # By type
            self.metrics.patterns_by_type[trade.pattern_type] = self.metrics.patterns_by_type.get(trade.pattern_type, 0) + 1
            
            # By quality
            self.metrics.patterns_by_quality[trade.pattern_quality] = self.metrics.patterns_by_quality.get(trade.pattern_quality, 0) + 1
        
        # Win rate by quality
        for quality in set(t.pattern_quality for t in self.trades):
            quality_trades = [t for t in self.trades if t.pattern_quality == quality]
            quality_wins = len([t for t in quality_trades if t.was_win])
            self.metrics.win_rate_by_quality[quality] = (quality_wins / len(quality_trades) * 100) if quality_trades else 0
        
        # Win rate by type
        for ptype in set(t.pattern_type for t in self.trades):
            type_trades = [t for t in self.trades if t.pattern_type == ptype]
            type_wins = len([t for t in type_trades if t.was_win])
            self.metrics.win_rate_by_type[ptype] = (type_wins / len(type_trades) * 100) if type_trades else 0
            self.metrics.avg_profit_by_type[ptype] = sum(t.actual_profit for t in type_trades) / len(type_trades) if type_trades else 0
        
        # SID integration metrics
        sid_confirmed = [t for t in self.trades if t.sid_signal_present]
        pattern_only = [t for t in self.trades if not t.sid_signal_present]
        
        self.metrics.sid_confirmed_trades = len(sid_confirmed)
        self.metrics.sid_confirmed_win_rate = (len([t for t in sid_confirmed if t.was_win]) / len(sid_confirmed) * 100) if sid_confirmed else 0
        self.metrics.pattern_only_win_rate = (len([t for t in pattern_only if t.was_win]) / len(pattern_only) * 100) if pattern_only else 0
        
        # Drawdown calculation
        equity_curve = [0]
        cumulative = 0
        peak = 0
        max_dd = 0
        
        for trade in self.trades:
            cumulative += trade.actual_profit
            equity_curve.append(cumulative)
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
        
        self.metrics.max_drawdown = max_dd
        self.metrics.max_drawdown_percent = (max_dd / 10000 * 100) if max_dd > 0 else 0  # Assuming $10,000 starting
    
    # ========================================================================
    # SECTION 6: RESULTS EXPORT (Wave 3)
    # ========================================================================
    
    def export_results(self, output_dir: str = None) -> Dict[str, str]:
        """
        Export backtest results to files
        
        Returns:
            Dictionary of file paths
        """
        if output_dir is None:
            output_dir = self.config.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exported = {}
        
        # Export trades
        trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
        trades_path = os.path.join(output_dir, f"pattern_trades_{timestamp}.csv")
        trades_df.to_csv(trades_path, index=False)
        exported['trades'] = trades_path
        
        # Export metrics
        metrics_path = os.path.join(output_dir, f"pattern_metrics_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        exported['metrics'] = metrics_path
        
        # Export summary report
        report_path = os.path.join(output_dir, f"pattern_report_{timestamp}.txt")
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("PATTERN BACKTEST REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total Patterns Detected: {self.metrics.total_patterns}\n")
            f.write(f"Valid Trades: {self.metrics.valid_trades}\n")
            f.write(f"Winning Trades: {self.metrics.winning_trades}\n")
            f.write(f"Losing Trades: {self.metrics.losing_trades}\n")
            f.write(f"Win Rate: {self.metrics.win_rate:.2f}%\n\n")
            f.write(f"Net Profit: ${self.metrics.net_profit:.2f}\n")
            f.write(f"Profit Factor: {self.metrics.profit_factor:.2f}\n")
            f.write(f"Max Drawdown: ${self.metrics.max_drawdown:.2f} ({self.metrics.max_drawdown_percent:.2f}%)\n\n")
            f.write("Patterns by Type:\n")
            for ptype, count in self.metrics.patterns_by_type.items():
                f.write(f"  {ptype}: {count} (Win Rate: {self.metrics.win_rate_by_type.get(ptype, 0):.1f}%)\n")
            f.write("\nPatterns by Quality:\n")
            for quality, count in self.metrics.patterns_by_quality.items():
                f.write(f"  {quality}: {count} (Win Rate: {self.metrics.win_rate_by_quality.get(quality, 0):.1f}%)\n")
            f.write("\nSID Integration:\n")
            f.write(f"  SID Confirmed Trades: {self.metrics.sid_confirmed_trades} (Win Rate: {self.metrics.sid_confirmed_win_rate:.1f}%)\n")
            f.write(f"  Pattern Only Trades: {self.metrics.valid_trades - self.metrics.sid_confirmed_trades} (Win Rate: {self.metrics.pattern_only_win_rate:.1f}%)\n")
        
        exported['report'] = report_path
        
        if self.config.verbose:
            print(f"\n💾 Results exported to {output_dir}")
            print(f"   Trades: {trades_path}")
            print(f"   Metrics: {metrics_path}")
            print(f"   Report: {report_path}")
        
        return exported
    
    # ========================================================================
    # SECTION 7: PRINT SUMMARY (Wave 2 & 3)
    # ========================================================================
    
    def print_summary(self):
        """Print backtest summary"""
        print("\n" + "="*60)
        print("📊 PATTERN BACKTEST SUMMARY")
        print("="*60)
        print(f"\n📈 Overall Performance:")
        print(f"   Total Patterns: {self.metrics.total_patterns}")
        print(f"   Win Rate: {self.metrics.win_rate:.1f}%")
        print(f"   Net Profit: ${self.metrics.net_profit:.2f}")
        print(f"   Profit Factor: {self.metrics.profit_factor:.2f}")
        print(f"   Max Drawdown: {self.metrics.max_drawdown_percent:.1f}%")
        
        print(f"\n📊 Patterns by Type:")
        for ptype, count in self.metrics.patterns_by_type.items():
            win_rate = self.metrics.win_rate_by_type.get(ptype, 0)
            print(f"   {ptype}: {count} trades (Win Rate: {win_rate:.1f}%)")
        
        print(f"\n⭐ Patterns by Quality:")
        for quality, count in self.metrics.patterns_by_quality.items():
            win_rate = self.metrics.win_rate_by_quality.get(quality, 0)
            print(f"   {quality}: {count} trades (Win Rate: {win_rate:.1f}%)")
        
        print(f"\n🔄 SID Integration:")
        print(f"   With SID Confirmation: {self.metrics.sid_confirmed_trades} trades (Win Rate: {self.metrics.sid_confirmed_win_rate:.1f}%)")
        print(f"   Pattern Only: {self.metrics.valid_trades - self.metrics.sid_confirmed_trades} trades (Win Rate: {self.metrics.pattern_only_win_rate:.1f}%)")
        print("="*60 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Pattern Backtest for SID Method')
    parser.add_argument('--data', type=str, required=True, help='Path to data file (CSV/Parquet)')
    parser.add_argument('--output', type=str, default='backtest_results/', help='Output directory')
    parser.add_argument('--min-candles', type=int, default=7, help='Minimum pattern candles')
    parser.add_argument('--require-neckline', action='store_true', default=True, help='Require neckline break')
    parser.add_argument('--no-neckline', dest='require_neckline', action='store_false', help='Don\'t require neckline break')
    parser.add_argument('--require-sid', action='store_true', help='Require SID signal confirmation')
    parser.add_argument('--risk', type=float, default=1.0, help='Risk percent per trade')
    parser.add_argument('--max-bars', type=int, default=50, help='Maximum bars to hold')
    parser.add_argument('--no-tqdm', action='store_true', help='Disable progress bar')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    # Create configuration
    config = BacktestConfig(
        min_pattern_candles=args.min_candles,
        require_neckline_break=args.require_neckline,
        require_sid_signal=args.require_sid,
        risk_percent=args.risk,
        max_hold_bars=args.max_bars,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    # Initialize backtest
    backtest = PatternBacktest(config)
    
    # Load data
    df = backtest.load_data(args.data)
    
    # Run backtest
    metrics = backtest.run_backtest(df)
    
    # Print summary
    backtest.print_summary()
    
    # Export results
    backtest.export_results()
    
    print(f"\n✅ Pattern backtest complete")


if __name__ == "__main__":
    # Test with sample data
    print("\n" + "="*70)
    print("🧪 TESTING PATTERN BACKTEST v3.0")
    print("="*70)
    
    # Create sample data with patterns
    dates = pd.date_range(start='2024-01-01', periods=1000, freq='H')
    np.random.seed(42)
    
    # Create double bottom pattern
    prices = 100 + np.random.randn(1000) * 0.5
    
    # Double bottom around index 200-300
    prices[200:210] = 95
    prices[210:240] = 98
    prices[240:250] = 94
    prices[250:280] = 99
    
    # Double top around index 500-600
    prices[500:510] = 105
    prices[510:540] = 102
    prices[540:550] = 106
    prices[550:580] = 101
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    print(f"Sample data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Run backtest
    config = BacktestConfig(
        min_pattern_candles=7,
        require_neckline_break=True,
        require_sid_signal=False,
        verbose=True
    )
    backtest = PatternBacktest(config)
    
    metrics = backtest.run_backtest(df)
    backtest.print_summary()
    
    print(f"\n✅ Pattern backtest test complete")