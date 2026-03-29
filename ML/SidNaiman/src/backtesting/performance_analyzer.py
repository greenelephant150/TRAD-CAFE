"""
Performance analysis for backtesting results
Calculates win rates, profit factors, and other metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyzes backtesting performance metrics
    Calculates win rates, profit factors, returns by pattern type, etc.
    """
    
    def __init__(self):
        self.trades = []
        
    def add_trades(self, trades: List[Dict]):
        """Add trades for analysis"""
        self.trades.extend(trades)
        
    def analyze(self) -> Dict[str, Any]:
        """Run complete performance analysis"""
        if not self.trades:
            return {'total_trades': 0}
            
        df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(df)
        winners = df[df['pnl_pct'] > 0]
        losers = df[df['pnl_pct'] <= 0]
        win_count = len(winners)
        loss_count = len(losers)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = df['pnl_pct'].sum()
        avg_win = winners['pnl_pct'].mean() if win_count > 0 else 0
        avg_loss = losers['pnl_pct'].mean() if loss_count > 0 else 0
        
        # Profit factor
        gross_profit = winners['pnl_pct'].sum() if win_count > 0 else 0
        gross_loss = abs(losers['pnl_pct'].sum()) if loss_count > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Risk/reward metrics
        avg_risk_reward = (avg_win / abs(avg_loss)) if avg_loss != 0 else float('inf')
        
        # Max drawdown
        cumulative = df['pnl_pct'].cumsum()
        running_max = cumulative.cummax()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()
        
        # Analysis by pattern type
        by_pattern = {}
        if 'pattern_type' in df.columns:
            for ptype in df['pattern_type'].unique():
                subset = df[df['pattern_type'] == ptype]
                p_winners = subset[subset['pnl_pct'] > 0]
                by_pattern[ptype] = {
                    'trades': len(subset),
                    'wins': len(p_winners),
                    'win_rate': len(p_winners) / len(subset) * 100,
                    'total_pnl': subset['pnl_pct'].sum(),
                    'avg_pnl': subset['pnl_pct'].mean()
                }
        
        # Analysis by stop loss type
        by_stop = {}
        if 'stop_loss_type' in df.columns:
            for stype in df['stop_loss_type'].unique():
                subset = df[df['stop_loss_type'] == stype]
                p_winners = subset[subset['pnl_pct'] > 0]
                by_stop[stype] = {
                    'trades': len(subset),
                    'wins': len(p_winners),
                    'win_rate': len(p_winners) / len(subset) * 100,
                    'total_pnl': subset['pnl_pct'].sum(),
                    'avg_pnl': subset['pnl_pct'].mean()
                }
        
        # Analysis by day of week
        by_day = {}
        if 'entry_time' in df.columns:
            df['entry_time'] = pd.to_datetime(df['entry_time'])
            df['day_of_week'] = df['entry_time'].dt.day_name()
            for day in df['day_of_week'].unique():
                subset = df[df['day_of_week'] == day]
                p_winners = subset[subset['pnl_pct'] > 0]
                by_day[day] = {
                    'trades': len(subset),
                    'wins': len(p_winners),
                    'win_rate': len(p_winners) / len(subset) * 100,
                    'total_pnl': subset['pnl_pct'].sum()
                }
        
        return {
            'total_trades': total_trades,
            'winners': win_count,
            'losers': loss_count,
            'win_rate': win_rate,
            'total_pnl_pct': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_risk_reward': avg_risk_reward,
            'max_drawdown': max_drawdown,
            'by_pattern': by_pattern,
            'by_stop_type': by_stop,
            'by_day': by_day
        }
    
    def compare_stop_strategies(self, trades_by_stop: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Compare different stop loss strategies
        Like Simon's analysis showing aggressive stops yield higher returns
        """
        results = {}
        for stop_type, trades in trades_by_stop.items():
            self.trades = trades
            results[stop_type] = self.analyze()
        
        # Find best strategy by total return
        if results:
            best = max(results.items(), key=lambda x: x[1]['total_pnl_pct'])
            results['best_strategy'] = {
                'stop_type': best[0],
                'return': best[1]['total_pnl_pct'],
                'win_rate': best[1]['win_rate']
            }
        
        return results
    
    def generate_report(self) -> str:
        """Generate a human-readable performance report"""
        results = self.analyze()
        
        report = []
        report.append("="*60)
        report.append("PERFORMANCE ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Total Trades: {results['total_trades']}")
        report.append(f"Winners: {results['winners']}")
        report.append(f"Losers: {results['losers']}")
        report.append(f"Win Rate: {results['win_rate']:.2f}%")
        report.append(f"Total P&L: {results['total_pnl_pct']:.2f}%")
        report.append(f"Avg Win: {results['avg_win']:.2f}%")
        report.append(f"Avg Loss: {results['avg_loss']:.2f}%")
        report.append(f"Profit Factor: {results['profit_factor']:.2f}")
        report.append(f"Avg Risk/Reward: {results['avg_risk_reward']:.2f}")
        report.append(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        
        if results.get('by_pattern'):
            report.append("\n" + "-"*40)
            report.append("BY PATTERN TYPE")
            report.append("-"*40)
            for ptype, stats in results['by_pattern'].items():
                report.append(f"{ptype}: {stats['wins']}/{stats['trades']} ({stats['win_rate']:.1f}%) P&L: {stats['total_pnl']:.2f}%")
        
        if results.get('by_stop_type'):
            report.append("\n" + "-"*40)
            report.append("BY STOP LOSS TYPE")
            report.append("-"*40)
            for stype, stats in results['by_stop_type'].items():
                report.append(f"{stype}: {stats['wins']}/{stats['trades']} ({stats['win_rate']:.1f}%) P&L: {stats['total_pnl']:.2f}%")
        
        if results.get('by_day'):
            report.append("\n" + "-"*40)
            report.append("BY DAY OF WEEK")
            report.append("-"*40)
            for day, stats in results['by_day'].items():
                report.append(f"{day}: {stats['wins']}/{stats['trades']} ({stats['win_rate']:.1f}%) P&L: {stats['total_pnl']:.2f}%")
        
        return "\n".join(report)
