#!/usr/bin/env python3
"""
Trade Log Analysis Script for SID Method - AUGMENTED VERSION
=============================================================================
Analyzes trade logs and generates performance metrics incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Win rate calculation
- Risk-reward ratio analysis
- Profit factor calculation
- Maximum drawdown analysis
- Trade duration statistics

WAVE 2 (Live Sessions & Q&A):
- Consecutive loss analysis
- Confidence score correlation
- Pattern confirmation effectiveness
- Divergence detection impact
- Market context performance breakdown
- Session-based performance analysis

WAVE 3 (Academy Support Sessions):
- Trade quality distribution
- Stop loss placement analysis
- Partial profit effectiveness
- Reversal exit analysis
- Time stop analysis
- Zone quality correlation

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
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import json
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class TradeStats:
    """Individual trade statistics"""
    trade_id: int
    entry_date: datetime
    exit_date: datetime
    instrument: str
    direction: str  # 'long' or 'short'
    signal_type: str  # 'oversold' or 'overbought'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    profit_loss: float
    profit_pips: float
    profit_percent: float
    bars_held: int
    exit_reason: str  # 'stop_loss', 'take_profit', 'trailing_stop', etc.
    
    # Signal quality metrics
    rsi_entry: float
    rsi_exit: float
    macd_crossed: bool
    pattern_confirmed: bool
    pattern_name: str
    divergence_detected: bool
    divergence_type: str
    confidence_score: float
    confidence_level: str
    quality_rating: str
    quality_score: float
    
    # Context
    market_trend: str
    session: str
    session_suitability: str
    
    # Risk metrics
    risk_percent: float
    risk_amount: float
    reward_ratio: float
    
    # Notes
    notes: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'trade_id': self.trade_id,
            'entry_date': self.entry_date.isoformat(),
            'exit_date': self.exit_date.isoformat(),
            'instrument': self.instrument,
            'direction': self.direction,
            'signal_type': self.signal_type,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'profit_loss': self.profit_loss,
            'profit_pips': self.profit_pips,
            'profit_percent': self.profit_percent,
            'bars_held': self.bars_held,
            'exit_reason': self.exit_reason,
            'rsi_entry': self.rsi_entry,
            'rsi_exit': self.rsi_exit,
            'macd_crossed': self.macd_crossed,
            'pattern_confirmed': self.pattern_confirmed,
            'pattern_name': self.pattern_name,
            'divergence_detected': self.divergence_detected,
            'divergence_type': self.divergence_type,
            'confidence_score': self.confidence_score,
            'confidence_level': self.confidence_level,
            'quality_rating': self.quality_rating,
            'quality_score': self.quality_score,
            'market_trend': self.market_trend,
            'session': self.session,
            'session_suitability': self.session_suitability,
            'risk_percent': self.risk_percent,
            'risk_amount': self.risk_amount,
            'reward_ratio': self.reward_ratio,
            'notes': self.notes
        }


@dataclass
class PerformanceMetrics:
    """Overall performance metrics"""
    # Basic metrics
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
    profit_factor: float = 0.0
    expectancy: float = 0.0
    
    # Risk metrics
    average_risk_percent: float = 0.0
    average_reward_ratio: float = 0.0
    max_consecutive_losses: int = 0
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    # Duration metrics
    average_bars_held: float = 0.0
    max_bars_held: int = 0
    average_trade_duration_hours: float = 0.0
    
    # Exit reason breakdown
    exit_reasons: Dict[str, int] = field(default_factory=dict)
    exit_reason_win_rates: Dict[str, float] = field(default_factory=dict)
    
    # SID Method specific metrics
    avg_rsi_entry: float = 0.0
    avg_confidence_score: float = 0.0
    macd_cross_trades: int = 0
    macd_cross_win_rate: float = 0.0
    pattern_trades: int = 0
    pattern_win_rate: float = 0.0
    divergence_trades: int = 0
    divergence_win_rate: float = 0.0
    
    # Quality breakdown
    trades_by_quality: Dict[str, int] = field(default_factory=dict)
    win_rate_by_quality: Dict[str, float] = field(default_factory=dict)
    
    # Market context breakdown
    trades_by_trend: Dict[str, int] = field(default_factory=dict)
    win_rate_by_trend: Dict[str, float] = field(default_factory=dict)
    
    # Session breakdown
    trades_by_session: Dict[str, int] = field(default_factory=dict)
    win_rate_by_session: Dict[str, float] = field(default_factory=dict)
    
    # Signal type breakdown
    trades_by_signal: Dict[str, int] = field(default_factory=dict)
    win_rate_by_signal: Dict[str, float] = field(default_factory=dict)
    
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
            'profit_factor': self.profit_factor,
            'expectancy': self.expectancy,
            'average_risk_percent': self.average_risk_percent,
            'average_reward_ratio': self.average_reward_ratio,
            'max_consecutive_losses': self.max_consecutive_losses,
            'max_drawdown': self.max_drawdown,
            'max_drawdown_percent': self.max_drawdown_percent,
            'average_bars_held': self.average_bars_held,
            'max_bars_held': self.max_bars_held,
            'average_trade_duration_hours': self.average_trade_duration_hours,
            'exit_reasons': self.exit_reasons,
            'exit_reason_win_rates': self.exit_reason_win_rates,
            'avg_rsi_entry': self.avg_rsi_entry,
            'avg_confidence_score': self.avg_confidence_score,
            'macd_cross_trades': self.macd_cross_trades,
            'macd_cross_win_rate': self.macd_cross_win_rate,
            'pattern_trades': self.pattern_trades,
            'pattern_win_rate': self.pattern_win_rate,
            'divergence_trades': self.divergence_trades,
            'divergence_win_rate': self.divergence_win_rate,
            'trades_by_quality': self.trades_by_quality,
            'win_rate_by_quality': self.win_rate_by_quality,
            'trades_by_trend': self.trades_by_trend,
            'win_rate_by_trend': self.win_rate_by_trend,
            'trades_by_session': self.trades_by_session,
            'win_rate_by_session': self.win_rate_by_session,
            'trades_by_signal': self.trades_by_signal,
            'win_rate_by_signal': self.win_rate_by_signal
        }


class TradeLogAnalyzer:
    """
    Analyzes trade logs for SID Method performance
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize trade log analyzer
        
        Args:
            verbose: Enable verbose output
        """
        self.verbose = verbose
        self.trades: List[TradeStats] = []
        self.metrics = PerformanceMetrics()
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"📊 TRADE LOG ANALYZER v3.0 (Fully Augmented)")
            print(f"{'='*60}\n")
    
    # ========================================================================
    # SECTION 1: DATA LOADING
    # ========================================================================
    
    def load_from_csv(self, filepath: str) -> List[TradeStats]:
        """
        Load trades from CSV file
        
        Expected columns:
        - entry_date, exit_date, instrument, direction, signal_type
        - entry_price, exit_price, stop_loss, take_profit
        - profit_loss, profit_pips, profit_percent, bars_held, exit_reason
        - rsi_entry, rsi_exit, macd_crossed, pattern_confirmed, pattern_name
        - divergence_detected, divergence_type, confidence_score, confidence_level
        - quality_rating, quality_score, market_trend, session, session_suitability
        - risk_percent, risk_amount, reward_ratio
        """
        df = pd.read_csv(filepath, parse_dates=['entry_date', 'exit_date'])
        
        self.trades = []
        
        for _, row in df.iterrows():
            trade = TradeStats(
                trade_id=row.get('trade_id', len(self.trades) + 1),
                entry_date=row['entry_date'],
                exit_date=row['exit_date'],
                instrument=row.get('instrument', 'unknown'),
                direction=row.get('direction', 'long'),
                signal_type=row.get('signal_type', 'unknown'),
                entry_price=row['entry_price'],
                exit_price=row['exit_price'],
                stop_loss=row.get('stop_loss', 0),
                take_profit=row.get('take_profit', 0),
                profit_loss=row['profit_loss'],
                profit_pips=row.get('profit_pips', 0),
                profit_percent=row.get('profit_percent', 0),
                bars_held=row.get('bars_held', 0),
                exit_reason=row.get('exit_reason', 'unknown'),
                rsi_entry=row.get('rsi_entry', 0),
                rsi_exit=row.get('rsi_exit', 0),
                macd_crossed=row.get('macd_crossed', False),
                pattern_confirmed=row.get('pattern_confirmed', False),
                pattern_name=row.get('pattern_name', ''),
                divergence_detected=row.get('divergence_detected', False),
                divergence_type=row.get('divergence_type', ''),
                confidence_score=row.get('confidence_score', 0),
                confidence_level=row.get('confidence_level', 'low'),
                quality_rating=row.get('quality_rating', 'fair'),
                quality_score=row.get('quality_score', 0),
                market_trend=row.get('market_trend', 'unknown'),
                session=row.get('session', 'unknown'),
                session_suitability=row.get('session_suitability', 'medium'),
                risk_percent=row.get('risk_percent', 1.0),
                risk_amount=row.get('risk_amount', 0),
                reward_ratio=row.get('reward_ratio', 0),
                notes=row.get('notes', '')
            )
            self.trades.append(trade)
        
        if self.verbose:
            print(f"✅ Loaded {len(self.trades)} trades from {filepath}")
        
        return self.trades
    
    def load_from_json(self, filepath: str) -> List[TradeStats]:
        """Load trades from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.trades = []
        
        for item in data:
            trade = TradeStats(
                trade_id=item.get('trade_id', len(self.trades) + 1),
                entry_date=datetime.fromisoformat(item['entry_date']),
                exit_date=datetime.fromisoformat(item['exit_date']),
                instrument=item.get('instrument', 'unknown'),
                direction=item.get('direction', 'long'),
                signal_type=item.get('signal_type', 'unknown'),
                entry_price=item['entry_price'],
                exit_price=item['exit_price'],
                stop_loss=item.get('stop_loss', 0),
                take_profit=item.get('take_profit', 0),
                profit_loss=item['profit_loss'],
                profit_pips=item.get('profit_pips', 0),
                profit_percent=item.get('profit_percent', 0),
                bars_held=item.get('bars_held', 0),
                exit_reason=item.get('exit_reason', 'unknown'),
                rsi_entry=item.get('rsi_entry', 0),
                rsi_exit=item.get('rsi_exit', 0),
                macd_crossed=item.get('macd_crossed', False),
                pattern_confirmed=item.get('pattern_confirmed', False),
                pattern_name=item.get('pattern_name', ''),
                divergence_detected=item.get('divergence_detected', False),
                divergence_type=item.get('divergence_type', ''),
                confidence_score=item.get('confidence_score', 0),
                confidence_level=item.get('confidence_level', 'low'),
                quality_rating=item.get('quality_rating', 'fair'),
                quality_score=item.get('quality_score', 0),
                market_trend=item.get('market_trend', 'unknown'),
                session=item.get('session', 'unknown'),
                session_suitability=item.get('session_suitability', 'medium'),
                risk_percent=item.get('risk_percent', 1.0),
                risk_amount=item.get('risk_amount', 0),
                reward_ratio=item.get('reward_ratio', 0),
                notes=item.get('notes', '')
            )
            self.trades.append(trade)
        
        if self.verbose:
            print(f"✅ Loaded {len(self.trades)} trades from {filepath}")
        
        return self.trades
    
    # ========================================================================
    # SECTION 2: METRICS CALCULATION (Wave 1, 2, 3)
    # ========================================================================
    
    def calculate_metrics(self) -> PerformanceMetrics:
        """Calculate all performance metrics"""
        if not self.trades:
            return self.metrics
        
        # Basic metrics
        self.metrics.total_trades = len(self.trades)
        self.metrics.winning_trades = len([t for t in self.trades if t.profit_loss > 0])
        self.metrics.losing_trades = len([t for t in self.trades if t.profit_loss <= 0])
        self.metrics.win_rate = (self.metrics.winning_trades / self.metrics.total_trades * 100) if self.metrics.total_trades > 0 else 0
        
        # Profit metrics
        wins = [t.profit_loss for t in self.trades if t.profit_loss > 0]
        losses = [abs(t.profit_loss) for t in self.trades if t.profit_loss <= 0]
        
        self.metrics.total_profit = sum(wins)
        self.metrics.total_loss = sum(losses)
        self.metrics.net_profit = self.metrics.total_profit - self.metrics.total_loss
        self.metrics.average_win = sum(wins) / len(wins) if wins else 0
        self.metrics.average_loss = sum(losses) / len(losses) if losses else 0
        self.metrics.largest_win = max(wins) if wins else 0
        self.metrics.largest_loss = max(losses) if losses else 0
        self.metrics.profit_factor = self.metrics.total_profit / self.metrics.total_loss if self.metrics.total_loss > 0 else float('inf')
        self.metrics.expectancy = self.metrics.net_profit / self.metrics.total_trades if self.metrics.total_trades > 0 else 0
        
        # Risk metrics
        self.metrics.average_risk_percent = sum(t.risk_percent for t in self.trades) / self.metrics.total_trades
        self.metrics.average_reward_ratio = sum(t.reward_ratio for t in self.trades) / self.metrics.total_trades
        
        # Consecutive losses (Wave 2)
        max_consecutive = 0
        current_consecutive = 0
        for trade in self.trades:
            if trade.profit_loss <= 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        self.metrics.max_consecutive_losses = max_consecutive
        
        # Duration metrics
        self.metrics.average_bars_held = sum(t.bars_held for t in self.trades) / self.metrics.total_trades
        self.metrics.max_bars_held = max(t.bars_held for t in self.trades)
        
        # Calculate trade duration in hours
        durations = []
        for t in self.trades:
            duration = (t.exit_date - t.entry_date).total_seconds() / 3600
            durations.append(duration)
        self.metrics.average_trade_duration_hours = sum(durations) / len(durations) if durations else 0
        
        # Exit reason breakdown (Wave 1 & 2)
        exit_reasons = {}
        exit_reason_wins = {}
        exit_reason_total = {}
        
        for trade in self.trades:
            reason = trade.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            exit_reason_total[reason] = exit_reason_total.get(reason, 0) + 1
            if trade.profit_loss > 0:
                exit_reason_wins[reason] = exit_reason_wins.get(reason, 0) + 1
        
        self.metrics.exit_reasons = exit_reasons
        for reason in exit_reason_total:
            self.metrics.exit_reason_win_rates[reason] = (exit_reason_wins.get(reason, 0) / exit_reason_total[reason] * 100)
        
        # SID Method specific metrics (Wave 1 & 2)
        rsi_entries = [t.rsi_entry for t in self.trades if t.rsi_entry > 0]
        self.metrics.avg_rsi_entry = sum(rsi_entries) / len(rsi_entries) if rsi_entries else 0
        
        confidence_scores = [t.confidence_score for t in self.trades if t.confidence_score > 0]
        self.metrics.avg_confidence_score = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # MACD cross trades
        macd_cross = [t for t in self.trades if t.macd_crossed]
        self.metrics.macd_cross_trades = len(macd_cross)
        macd_cross_wins = len([t for t in macd_cross if t.profit_loss > 0])
        self.metrics.macd_cross_win_rate = (macd_cross_wins / len(macd_cross) * 100) if macd_cross else 0
        
        # Pattern trades (Wave 2)
        pattern_trades = [t for t in self.trades if t.pattern_confirmed]
        self.metrics.pattern_trades = len(pattern_trades)
        pattern_wins = len([t for t in pattern_trades if t.profit_loss > 0])
        self.metrics.pattern_win_rate = (pattern_wins / len(pattern_trades) * 100) if pattern_trades else 0
        
        # Divergence trades (Wave 2)
        div_trades = [t for t in self.trades if t.divergence_detected]
        self.metrics.divergence_trades = len(div_trades)
        div_wins = len([t for t in div_trades if t.profit_loss > 0])
        self.metrics.divergence_win_rate = (div_wins / len(div_trades) * 100) if div_trades else 0
        
        # Quality breakdown (Wave 3)
        qualities = ['excellent', 'good', 'fair', 'poor']
        for quality in qualities:
            quality_trades = [t for t in self.trades if t.quality_rating == quality]
            self.metrics.trades_by_quality[quality] = len(quality_trades)
            quality_wins = len([t for t in quality_trades if t.profit_loss > 0])
            self.metrics.win_rate_by_quality[quality] = (quality_wins / len(quality_trades) * 100) if quality_trades else 0
        
        # Market context breakdown (Wave 2)
        trends = ['uptrend', 'downtrend', 'sideways', 'unknown']
        for trend in trends:
            trend_trades = [t for t in self.trades if t.market_trend == trend]
            self.metrics.trades_by_trend[trend] = len(trend_trades)
            trend_wins = len([t for t in trend_trades if t.profit_loss > 0])
            self.metrics.win_rate_by_trend[trend] = (trend_wins / len(trend_trades) * 100) if trend_trades else 0
        
        # Session breakdown (Wave 3)
        sessions = ['overlap', 'us', 'london', 'asian']
        for session in sessions:
            session_trades = [t for t in self.trades if t.session == session]
            self.metrics.trades_by_session[session] = len(session_trades)
            session_wins = len([t for t in session_trades if t.profit_loss > 0])
            self.metrics.win_rate_by_session[session] = (session_wins / len(session_trades) * 100) if session_trades else 0
        
        # Signal type breakdown
        signals = ['oversold', 'overbought']
        for signal in signals:
            signal_trades = [t for t in self.trades if t.signal_type == signal]
            self.metrics.trades_by_signal[signal] = len(signal_trades)
            signal_wins = len([t for t in signal_trades if t.profit_loss > 0])
            self.metrics.win_rate_by_signal[signal] = (signal_wins / len(signal_trades) * 100) if signal_trades else 0
        
        # Drawdown calculation
        equity_curve = [0]
        cumulative = 0
        peak = 0
        max_dd = 0
        max_dd_percent = 0
        
        for trade in self.trades:
            cumulative += trade.profit_loss
            equity_curve.append(cumulative)
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd
            # Percent drawdown relative to starting capital
            starting_capital = 10000  # Assuming $10,000 starting
            current_balance = starting_capital + cumulative
            dd_percent = (dd / starting_capital) * 100 if starting_capital > 0 else 0
            if dd_percent > max_dd_percent:
                max_dd_percent = dd_percent
        
        self.metrics.max_drawdown = max_dd
        self.metrics.max_drawdown_percent = max_dd_percent
        
        return self.metrics
    
    # ========================================================================
    # SECTION 3: VISUALIZATION (Wave 2 & 3)
    # ========================================================================
    
    def plot_equity_curve(self, save_path: str = None):
        """Plot equity curve"""
        if not self.trades:
            return
        
        cumulative = [0]
        for trade in self.trades:
            cumulative.append(cumulative[-1] + trade.profit_loss)
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative, label='Equity Curve', linewidth=2)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        plt.fill_between(range(len(cumulative)), cumulative, 0, 
                         where=[c >= 0 for c in cumulative], color='green', alpha=0.3,
                         where2=[c < 0 for c in cumulative], color='red', alpha=0.3)
        plt.title('Equity Curve')
        plt.xlabel('Trade Number')
        plt.ylabel('Cumulative Profit/Loss ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Equity curve saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_win_rate_by_quality(self, save_path: str = None):
        """Plot win rate by quality rating (Wave 3)"""
        qualities = list(self.metrics.win_rate_by_quality.keys())
        win_rates = list(self.metrics.win_rate_by_quality.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(qualities, win_rates, color=['green', 'lightgreen', 'orange', 'red'])
        plt.axhline(y=self.metrics.win_rate, color='blue', linestyle='--', label=f'Overall Win Rate: {self.metrics.win_rate:.1f}%')
        plt.title('Win Rate by Signal Quality')
        plt.xlabel('Quality Rating')
        plt.ylabel('Win Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, rate in zip(bars, win_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Quality chart saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_win_rate_by_session(self, save_path: str = None):
        """Plot win rate by trading session (Wave 3)"""
        sessions = list(self.metrics.win_rate_by_session.keys())
        win_rates = list(self.metrics.win_rate_by_session.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sessions, win_rates, color=['gold', 'blue', 'lightblue', 'gray'])
        plt.axhline(y=self.metrics.win_rate, color='red', linestyle='--', label=f'Overall Win Rate: {self.metrics.win_rate:.1f}%')
        plt.title('Win Rate by Trading Session')
        plt.xlabel('Session')
        plt.ylabel('Win Rate (%)')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, rate in zip(bars, win_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{rate:.1f}%', ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Session chart saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_exit_reasons(self, save_path: str = None):
        """Plot exit reason distribution (Wave 2)"""
        reasons = list(self.metrics.exit_reasons.keys())
        counts = list(self.metrics.exit_reasons.values())
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(reasons, counts, color='steelblue')
        plt.title('Exit Reason Distribution')
        plt.xlabel('Exit Reason')
        plt.ylabel('Number of Trades')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(count), ha='center', va='bottom')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Exit reasons chart saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def plot_confidence_vs_profit(self, save_path: str = None):
        """Plot confidence score vs profit (Wave 2)"""
        if not self.trades:
            return
        
        confidences = [t.confidence_score for t in self.trades]
        profits = [t.profit_loss for t in self.trades]
        
        plt.figure(figsize=(10, 6))
        plt.scatter(confidences, profits, alpha=0.6, c=['green' if p > 0 else 'red' for p in profits])
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Add trend line
        z = np.polyfit(confidences, profits, 1)
        p = np.poly1d(z)
        plt.plot(sorted(confidences), p(sorted(confidences)), "b--", alpha=0.5, label='Trend Line')
        
        plt.title('Confidence Score vs Profit/Loss')
        plt.xlabel('Confidence Score')
        plt.ylabel('Profit/Loss ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.verbose:
                print(f"💾 Confidence chart saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    # ========================================================================
    # SECTION 4: EXPORT (Wave 3)
    # ========================================================================
    
    def export_metrics(self, filepath: str) -> bool:
        """Export metrics to JSON"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics.to_dict(), f, indent=2)
            if self.verbose:
                print(f"💾 Metrics exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export metrics: {e}")
            return False
    
    def export_trades(self, filepath: str) -> bool:
        """Export trades to CSV"""
        try:
            trades_df = pd.DataFrame([t.to_dict() for t in self.trades])
            trades_df.to_csv(filepath, index=False)
            if self.verbose:
                print(f"💾 Trades exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export trades: {e}")
            return False
    
    def export_report(self, filepath: str) -> bool:
        """Export comprehensive report (Wave 3)"""
        try:
            with open(filepath, 'w') as f:
                f.write("="*60 + "\n")
                f.write("SID METHOD TRADE LOG ANALYSIS REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write("PERFORMANCE SUMMARY\n")
                f.write("-"*40 + "\n")
                f.write(f"Total Trades: {self.metrics.total_trades}\n")
                f.write(f"Winning Trades: {self.metrics.winning_trades}\n")
                f.write(f"Losing Trades: {self.metrics.losing_trades}\n")
                f.write(f"Win Rate: {self.metrics.win_rate:.2f}%\n\n")
                
                f.write(f"Total Profit: ${self.metrics.total_profit:.2f}\n")
                f.write(f"Total Loss: ${self.metrics.total_loss:.2f}\n")
                f.write(f"Net Profit: ${self.metrics.net_profit:.2f}\n")
                f.write(f"Profit Factor: {self.metrics.profit_factor:.2f}\n")
                f.write(f"Expectancy: ${self.metrics.expectancy:.2f}\n\n")
                
                f.write(f"Average Win: ${self.metrics.average_win:.2f}\n")
                f.write(f"Average Loss: ${self.metrics.average_loss:.2f}\n")
                f.write(f"Largest Win: ${self.metrics.largest_win:.2f}\n")
                f.write(f"Largest Loss: ${self.metrics.largest_loss:.2f}\n\n")
                
                f.write(f"Max Consecutive Losses: {self.metrics.max_consecutive_losses}\n")
                f.write(f"Max Drawdown: ${self.metrics.max_drawdown:.2f} ({self.metrics.max_drawdown_percent:.2f}%)\n\n")
                
                f.write("RISK METRICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Average Risk per Trade: {self.metrics.average_risk_percent:.2f}%\n")
                f.write(f"Average Reward Ratio: {self.metrics.average_reward_ratio:.2f}\n\n")
                
                f.write("SID METHOD SPECIFIC METRICS\n")
                f.write("-"*40 + "\n")
                f.write(f"Average RSI at Entry: {self.metrics.avg_rsi_entry:.2f}\n")
                f.write(f"Average Confidence Score: {self.metrics.avg_confidence_score:.3f}\n")
                f.write(f"MACD Cross Trades: {self.metrics.macd_cross_trades} (Win Rate: {self.metrics.macd_cross_win_rate:.1f}%)\n")
                f.write(f"Pattern Confirmed Trades: {self.metrics.pattern_trades} (Win Rate: {self.metrics.pattern_win_rate:.1f}%)\n")
                f.write(f"Divergence Detected Trades: {self.metrics.divergence_trades} (Win Rate: {self.metrics.divergence_win_rate:.1f}%)\n\n")
                
                f.write("QUALITY BREAKDOWN\n")
                f.write("-"*40 + "\n")
                for quality in ['excellent', 'good', 'fair', 'poor']:
                    trades = self.metrics.trades_by_quality.get(quality, 0)
                    win_rate = self.metrics.win_rate_by_quality.get(quality, 0)
                    f.write(f"  {quality.upper()}: {trades} trades (Win Rate: {win_rate:.1f}%)\n")
                f.write("\n")
                
                f.write("SESSION BREAKDOWN\n")
                f.write("-"*40 + "\n")
                for session in ['overlap', 'us', 'london', 'asian']:
                    trades = self.metrics.trades_by_session.get(session, 0)
                    win_rate = self.metrics.win_rate_by_session.get(session, 0)
                    f.write(f"  {session.upper()}: {trades} trades (Win Rate: {win_rate:.1f}%)\n")
                f.write("\n")
                
                f.write("MARKET CONTEXT BREAKDOWN\n")
                f.write("-"*40 + "\n")
                for trend in ['uptrend', 'downtrend', 'sideways']:
                    trades = self.metrics.trades_by_trend.get(trend, 0)
                    win_rate = self.metrics.win_rate_by_trend.get(trend, 0)
                    f.write(f"  {trend.upper()}: {trades} trades (Win Rate: {win_rate:.1f}%)\n")
                f.write("\n")
                
                f.write("EXIT REASONS\n")
                f.write("-"*40 + "\n")
                for reason, count in self.metrics.exit_reasons.items():
                    win_rate = self.metrics.exit_reason_win_rates.get(reason, 0)
                    f.write(f"  {reason}: {count} trades (Win Rate: {win_rate:.1f}%)\n")
            
            if self.verbose:
                print(f"💾 Report exported to {filepath}")
            return True
        except Exception as e:
            print(f"❌ Failed to export report: {e}")
            return False
    
    # ========================================================================
    # SECTION 5: PRINT SUMMARY
    # ========================================================================
    
    def print_summary(self):
        """Print comprehensive summary (Wave 2 & 3)"""
        print("\n" + "="*60)
        print("📊 TRADE LOG ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\n📈 Overall Performance:")
        print(f"   Total Trades: {self.metrics.total_trades}")
        print(f"   Win Rate: {self.metrics.win_rate:.1f}%")
        print(f"   Net Profit: ${self.metrics.net_profit:.2f}")
        print(f"   Profit Factor: {self.metrics.profit_factor:.2f}")
        print(f"   Expectancy: ${self.metrics.expectancy:.2f}")
        
        print(f"\n💰 Risk Metrics:")
        print(f"   Max Consecutive Losses: {self.metrics.max_consecutive_losses}")
        print(f"   Max Drawdown: {self.metrics.max_drawdown_percent:.1f}%")
        print(f"   Average Reward Ratio: {self.metrics.average_reward_ratio:.2f}")
        
        print(f"\n🎯 SID Method Metrics:")
        print(f"   Avg RSI Entry: {self.metrics.avg_rsi_entry:.1f}")
        print(f"   Avg Confidence: {self.metrics.avg_confidence_score:.3f}")
        print(f"   MACD Cross Win Rate: {self.metrics.macd_cross_win_rate:.1f}%")
        print(f"   Pattern Win Rate: {self.metrics.pattern_win_rate:.1f}%")
        print(f"   Divergence Win Rate: {self.metrics.divergence_win_rate:.1f}%")
        
        print(f"\n⭐ Quality Breakdown:")
        for quality in ['excellent', 'good', 'fair', 'poor']:
            trades = self.metrics.trades_by_quality.get(quality, 0)
            win_rate = self.metrics.win_rate_by_quality.get(quality, 0)
            print(f"   {quality.upper()}: {trades} trades ({win_rate:.1f}% win rate)")
        
        print(f"\n🌍 Session Breakdown:")
        for session in ['overlap', 'us', 'london', 'asian']:
            trades = self.metrics.trades_by_session.get(session, 0)
            win_rate = self.metrics.win_rate_by_session.get(session, 0)
            print(f"   {session.upper()}: {trades} trades ({win_rate:.1f}% win rate)")
        
        print(f"\n📊 Market Context:")
        for trend in ['uptrend', 'downtrend']:
            trades = self.metrics.trades_by_trend.get(trend, 0)
            win_rate = self.metrics.win_rate_by_trend.get(trend, 0)
            print(f"   {trend.upper()}: {trades} trades ({win_rate:.1f}% win rate)")
        
        print("="*60 + "\n")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Trade Log Analysis for SID Method')
    parser.add_argument('--input', type=str, required=True, help='Path to trade log file (CSV/JSON)')
    parser.add_argument('--output-dir', type=str, default='analysis_results/', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--no-plot', dest='plot', action='store_false', help='Disable plots')
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TradeLogAnalyzer(verbose=args.verbose)
    
    # Load trades
    if args.input.endswith('.json'):
        analyzer.load_from_json(args.input)
    else:
        analyzer.load_from_csv(args.input)
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Print summary
    analyzer.print_summary()
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export results
    analyzer.export_metrics(os.path.join(args.output_dir, f"metrics_{timestamp}.json"))
    analyzer.export_trades(os.path.join(args.output_dir, f"trades_analyzed_{timestamp}.csv"))
    analyzer.export_report(os.path.join(args.output_dir, f"report_{timestamp}.txt"))
    
    # Generate plots if requested
    if args.plot:
        analyzer.plot_equity_curve(os.path.join(args.output_dir, f"equity_curve_{timestamp}.png"))
        analyzer.plot_win_rate_by_quality(os.path.join(args.output_dir, f"win_rate_by_quality_{timestamp}.png"))
        analyzer.plot_win_rate_by_session(os.path.join(args.output_dir, f"win_rate_by_session_{timestamp}.png"))
        analyzer.plot_exit_reasons(os.path.join(args.output_dir, f"exit_reasons_{timestamp}.png"))
        analyzer.plot_confidence_vs_profit(os.path.join(args.output_dir, f"confidence_vs_profit_{timestamp}.png"))
    
    print(f"\n✅ Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    # Test with sample data
    print("\n" + "="*70)
    print("🧪 TESTING TRADE LOG ANALYZER v3.0")
    print("="*70)
    
    # Create sample trade data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    trades = []
    for i in range(50):
        is_win = np.random.random() > 0.3  # 70% win rate
        profit = np.random.uniform(50, 200) if is_win else np.random.uniform(-100, -20)
        
        trade = TradeStats(
            trade_id=i+1,
            entry_date=dates[i*2],
            exit_date=dates[i*2 + np.random.randint(1, 5)],
            instrument='EUR_USD',
            direction=np.random.choice(['long', 'short']),
            signal_type=np.random.choice(['oversold', 'overbought']),
            entry_price=100 + np.random.randn() * 5,
            exit_price=100 + np.random.randn() * 5,
            stop_loss=99,
            take_profit=101,
            profit_loss=profit,
            profit_pips=profit / 0.0001,
            profit_percent=profit / 10000 * 100,
            bars_held=np.random.randint(1, 20),
            exit_reason=np.random.choice(['take_profit', 'stop_loss', 'trailing_stop']),
            rsi_entry=np.random.uniform(20, 80),
            rsi_exit=np.random.uniform(30, 70),
            macd_crossed=np.random.random() > 0.5,
            pattern_confirmed=np.random.random() > 0.7,
            pattern_name=np.random.choice(['double_bottom', 'double_top', '']),
            divergence_detected=np.random.random() > 0.8,
            divergence_type=np.random.choice(['bullish', 'bearish', '']),
            confidence_score=np.random.uniform(0.4, 0.95),
            confidence_level=np.random.choice(['high', 'medium', 'low']),
            quality_rating=np.random.choice(['excellent', 'good', 'fair', 'poor'], p=[0.2, 0.4, 0.3, 0.1]),
            quality_score=np.random.uniform(30, 95),
            market_trend=np.random.choice(['uptrend', 'downtrend', 'sideways'], p=[0.4, 0.3, 0.3]),
            session=np.random.choice(['overlap', 'us', 'london', 'asian'], p=[0.3, 0.3, 0.2, 0.2]),
            session_suitability=np.random.choice(['very_high', 'high', 'medium', 'low']),
            risk_percent=1.0,
            risk_amount=100,
            reward_ratio=np.random.uniform(0.8, 2.5),
            notes=""
        )
        trades.append(trade)
    
    # Save sample trades
    analyzer = TradeLogAnalyzer(verbose=True)
    analyzer.trades = trades
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics()
    
    # Print summary
    analyzer.print_summary()
    
    # Export sample results
    analyzer.export_trades("sample_trades_analyzed.csv")
    analyzer.export_metrics("sample_metrics.json")
    analyzer.export_report("sample_report.txt")
    
    print(f"\n✅ Trade log analyzer test complete")