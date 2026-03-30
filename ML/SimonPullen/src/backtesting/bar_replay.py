"""
Bar replay system for backtesting Simon Pullen's strategies
Simulates TradingView's bar replay functionality
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Generator, Tuple
from datetime import datetime, timedelta
import logging

from src.core.mw_pattern import MWPatternDetector, MWPattern
from src.core.head_shoulders import HeadShouldersDetector, HeadShouldersPattern
from src.confluence.divergence import DivergenceDetector
from src.execution.entry_rules import EntryRuleEngine, EntrySignal
from src.execution.exit_rules import ExitRuleEngine
from src.risk.position_sizer import PositionSizer
from src.backtesting.trade_logger import TradeLogger
from src.utils.color_logger import get_logger, Colors, EMOJIS

logger = logging.getLogger(__name__)


class BarReplay:
    """
    TradingView-style bar replay for backtesting
    
    Allows stepping through historical data bar by bar
    to simulate real-time pattern recognition
    """
    
    def __init__(self, config: Dict[str, Any], initial_balance: float = 10000):
        self.config = config
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.verbose = config.get('verbose', False)
        self.color_logger = get_logger(self.verbose)
        
        # Initialize components
        self.mw_detector = MWPatternDetector(config)
        self.hs_detector = HeadShouldersDetector(config)
        self.div_detector = DivergenceDetector(config)
        self.entry_engine = EntryRuleEngine(config)
        self.exit_engine = ExitRuleEngine(config)
        self.position_sizer = PositionSizer(config, initial_balance)
        self.trade_logger = TradeLogger()
        
        # State
        self.df = None
        self.current_idx = 0
        self.max_idx = 0
        self.instrument = ""
        self.timeframe = ""
        
        # Detected patterns (indexed by detection bar)
        self.patterns: Dict[int, List] = {}
        
        # Open positions
        self.positions: List[Dict] = []
        
        # Statistics
        self.patterns_detected = 0
        self.entries_attempted = 0
        self.entries_executed = 0
        
    def load_data(self, df: pd.DataFrame, instrument: str, timeframe: str):
        """Load historical data for replay"""
        self.df = df
        self.max_idx = len(df) - 1
        self.instrument = instrument
        self.timeframe = timeframe
        self.current_idx = 0
        
        if self.verbose:
            self.color_logger.debug(f"Loaded {len(df)} bars for {instrument} {timeframe}")
        
    def reset(self):
        """Reset to beginning"""
        self.current_idx = 0
        self.current_balance = self.initial_balance
        self.patterns = {}
        self.positions = []
        self.trade_logger = TradeLogger()
        self.patterns_detected = 0
        self.entries_attempted = 0
        self.entries_executed = 0
        
    def step(self, steps: int = 1) -> bool:
        """
        Move forward one or more bars
        Returns True if more bars available
        """
        self.current_idx = min(self.current_idx + steps, self.max_idx)
        return self.current_idx < self.max_idx
    
    def step_to_end(self):
        """Step through all remaining bars"""
        while self.step():
            self._process_current_bar()
    
    def _process_current_bar(self):
        """Process current bar - detect patterns, check entries/exits"""
        if self.current_idx < 10:  # Need enough data
            return
            
        current_df = self.df.iloc[:self.current_idx + 1]
        
        # Detect patterns
        self._detect_patterns(current_df)
        
        # Check exits for open positions
        self._check_exits(current_df)
        
        # Check entries for detected patterns
        self._check_entries(current_df)
        
    def _detect_patterns(self, df: pd.DataFrame):
        """Detect patterns up to current bar"""
        # M-Tops
        m_patterns = self.mw_detector.detect_m_top(df, self.instrument, self.timeframe)
        for pattern in m_patterns:
            if pattern.valid:
                self.patterns_detected += 1
                if self.verbose and self.patterns_detected % 10 == 0:
                    self.color_logger.pattern_found("M-Top", pattern.candle_count, 0.7)
                    
                if self.current_idx not in self.patterns:
                    self.patterns[self.current_idx] = []
                self.patterns[self.current_idx].append(pattern)
                
        # W-Bottoms
        w_patterns = self.mw_detector.detect_w_bottom(df, self.instrument, self.timeframe)
        for pattern in w_patterns:
            if pattern.valid:
                self.patterns_detected += 1
                if self.verbose and self.patterns_detected % 10 == 0:
                    self.color_logger.pattern_found("W-Bottom", pattern.candle_count, 0.7)
                    
                if self.current_idx not in self.patterns:
                    self.patterns[self.current_idx] = []
                self.patterns[self.current_idx].append(pattern)
                
        # Head & Shoulders
        hs_patterns = self.hs_detector.detect_normal(df, self.instrument, self.timeframe)
        hs_patterns.extend(self.hs_detector.detect_inverted(df, self.instrument, self.timeframe))
        
        for pattern in hs_patterns:
            if pattern.valid:
                self.patterns_detected += 1
                if self.verbose and self.patterns_detected % 10 == 0:
                    ptype = "H&S Normal" if pattern.pattern_type == 'normal' else "H&S Inverted"
                    self.color_logger.pattern_found(ptype, pattern.candle_count, 0.75)
                    
                if self.current_idx not in self.patterns:
                    self.patterns[self.current_idx] = []
                self.patterns[self.current_idx].append(pattern)
    
    def _check_entries(self, df: pd.DataFrame):
        """Check if any detected patterns have triggered entries"""
        # Check patterns detected in last 20 bars
        for detect_idx in range(max(0, self.current_idx - 20), self.current_idx + 1):
            if detect_idx not in self.patterns:
                continue
                
            for pattern in self.patterns[detect_idx]:
                # Check if pattern has triggered entry
                try:
                    self.entries_attempted += 1
                    
                    if isinstance(pattern, MWPattern):
                        signal = self.entry_engine.check_mw_entry(df, pattern, self.current_idx)
                    else:  # HeadShouldersPattern
                        signal = self.entry_engine.check_hs_entry(df, pattern, self.current_idx)
                    
                    if signal:
                        self.entries_executed += 1
                        self._execute_entry(signal, df)
                except Exception as e:
                    if self.verbose:
                        self.color_logger.debug(f"Error checking entry: {e}")
                    continue
    
    def _check_exits(self, df: pd.DataFrame):
        """Check exits for all open positions"""
        for position in self.positions[:]:  # Copy list to allow removal
            entry_idx = position['entry_idx']
            signal = position['entry_signal']
            
            try:
                exit_now, reason = self.exit_engine.should_exit(
                    df, signal, entry_idx, self.current_idx
                )
                
                if exit_now:
                    self._execute_exit(position, reason, df)
            except Exception as e:
                if self.verbose:
                    self.color_logger.debug(f"Error checking exit: {e}")
                continue
    
    def _execute_entry(self, signal: EntrySignal, df: pd.DataFrame):
        """Execute an entry signal"""
        # Check if we can take this trade
        can_take, reason = self.position_sizer.can_take_trade(
            signal.instrument, signal.confidence
        )
        
        if not can_take:
            if self.verbose:
                self.color_logger.debug(f"Skipping trade: {reason}")
            return
            
        # Calculate position size
        instrument_info = {'step_size': 1000}  # TODO: Get from config
        position_size = self.position_sizer.calculate_position_size(
            signal.instrument, signal.direction,
            signal.entry_price, signal.stop_loss,
            signal.confidence, instrument_info
        )
        
        if position_size is None:
            if self.verbose:
                self.color_logger.debug(f"Invalid position size for {signal.pattern_type} on {signal.instrument}")
            return
        
        # Get entry time (try to get datetime, fallback to index)
        try:
            entry_time = df.index[self.current_idx] if hasattr(df.index, '__getitem__') else self.current_idx
        except:
            entry_time = self.current_idx
        
        # Create position
        position = {
            'entry_idx': self.current_idx,
            'entry_time': entry_time,
            'entry_signal': signal,
            'position_size': position_size,
            'status': 'open'
        }
        
        self.positions.append(position)
        self.position_sizer.add_position({
            'instrument': signal.instrument,
            'risk_percent': position_size.risk_percent,
            'direction': signal.direction
        })
        
        self.trade_logger.log_entry(position, self.current_balance)
        
        if self.verbose:
            self.color_logger.trade_entry(
                signal.instrument,
                f"{signal.pattern_type} ({signal.stop_loss_type})",
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit
            )
    
    def _execute_exit(self, position: Dict, reason: str, df: pd.DataFrame):
        """Execute an exit"""
        signal = position['entry_signal']
        
        # Get exit details
        exit_status = self.exit_engine.get_exit_status(
            df, signal, position['entry_idx'], self.current_idx
        )
        exit_status['exit_reason'] = reason
        
        # Update balance
        pnl = exit_status['pnl_pct'] / 100 * position['position_size'].risk_amount / position['position_size'].risk_percent * 100
        self.current_balance += pnl
        
        # Get exit time
        try:
            exit_time = df.index[self.current_idx] if hasattr(df.index, '__getitem__') else self.current_idx
        except:
            exit_time = self.current_idx
        
        position['exit_time'] = exit_time
        
        # Remove from positions
        self.positions.remove(position)
        self.position_sizer.remove_position(signal.instrument, position['position_size'].risk_percent)
        
        # Log trade
        self.trade_logger.log_exit(position, exit_status, self.current_balance)
        
        if self.verbose:
            self.color_logger.trade_exit(
                signal.instrument,
                exit_status['pnl_pct'],
                reason,
                exit_status['bars_held']
            )
    
    def get_results(self) -> Dict:
        """Get backtest results"""
        trades = self.trade_logger.get_all_trades()
        
        if not trades:
            return {
                'total_trades': 0,
                'patterns_detected': self.patterns_detected,
                'entries_attempted': self.entries_attempted
            }
            
        winners = [t for t in trades if t.get('exit_status', {}).get('pnl_pct', 0) > 0]
        losers = [t for t in trades if t.get('exit_status', {}).get('pnl_pct', 0) <= 0]
        
        total_pnl = sum(t.get('exit_status', {}).get('pnl_pct', 0) for t in trades)
        
        # Group by pattern type
        by_pattern = {}
        for t in trades:
            if 'entry_signal' in t and hasattr(t['entry_signal'], 'pattern_type'):
                ptype = t['entry_signal'].pattern_type
            else:
                ptype = 'unknown'
                
            if ptype not in by_pattern:
                by_pattern[ptype] = {'wins': 0, 'losses': 0, 'pnl': 0}
            
            pnl = t.get('exit_status', {}).get('pnl_pct', 0)
            if pnl > 0:
                by_pattern[ptype]['wins'] += 1
            else:
                by_pattern[ptype]['losses'] += 1
            by_pattern[ptype]['pnl'] += pnl
        
        # Calculate average trade metrics
        avg_win = np.mean([t.get('exit_status', {}).get('pnl_pct', 0) for t in winners]) if winners else 0
        avg_loss = np.mean([t.get('exit_status', {}).get('pnl_pct', 0) for t in losers]) if losers else 0
        
        return {
            'total_trades': len(trades),
            'winners': len(winners),
            'losers': len(losers),
            'win_rate': len(winners) / len(trades) * 100 if trades else 0,
            'total_pnl_pct': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'final_balance': self.current_balance,
            'return_pct': (self.current_balance - self.initial_balance) / self.initial_balance * 100,
            'by_pattern': by_pattern,
            'patterns_detected': self.patterns_detected,
            'entries_attempted': self.entries_attempted,
            'entries_executed': self.entries_executed,
            'trades': trades
        }
    
    def run_full_backtest(self, df: pd.DataFrame, instrument: str, timeframe: str) -> Dict:
        """Run complete backtest on all data"""
        self.load_data(df, instrument, timeframe)
        self.step_to_end()
        return self.get_results()
    
    def analyze_stop_loss_strategies(self, df: pd.DataFrame, instrument: str, timeframe: str) -> Dict:
        """
        Compare different stop loss strategies
        Like Simon's analysis that showed aggressive stops yield higher returns
        """
        results = {}
        
        strategies = [
            ('conservative', '🔵 Conservative'),
            ('moderate', '🟡 Moderate'),
            ('aggressive', '🔴 Aggressive')
        ]
        
        for stop_type, display_name in strategies:
            if self.verbose:
                self.color_logger.section(f"Testing {display_name} Strategy")
                
            self.reset()
            self.load_data(df, instrument, timeframe)
            
            # Override default stop type
            self.entry_engine.default_stop_type = stop_type
            
            self.step_to_end()
            results[stop_type] = self.get_results()
            
            if self.verbose:
                win_rate = results[stop_type]['win_rate']
                total_pnl = results[stop_type]['return_pct']
                color = Colors.GREEN if total_pnl > 0 else Colors.RED
                print(f"  {display_name}: Win Rate {win_rate:.1f}%, "
                      f"Return {color}{total_pnl:+.2f}%{Colors.END}")
        
        return results
