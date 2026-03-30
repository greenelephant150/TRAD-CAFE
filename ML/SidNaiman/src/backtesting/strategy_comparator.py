"""
Strategy comparison tool for backtesting
Compares different stop loss strategies and pattern variations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StrategyComparator:
    """
    Compares different trading strategy variations
    Like Simon's comparison of conservative vs aggressive stop losses
    """
    
    def __init__(self):
        self.results = {}
        
    def add_result(self, strategy_name: str, trades: List[Dict], metadata: Dict = None):
        """Add results for a strategy variation"""
        self.results[strategy_name] = {
            'trades': trades,
            'metadata': metadata or {}
        }
        
    def compare(self) -> Dict[str, Any]:
        """
        Compare all added strategies
        Returns ranking and key metrics
        """
        if not self.results:
            return {}
            
        comparison = {}
        for name, data in self.results.items():
            trades = data['trades']
            if not trades:
                continue
                
            df = pd.DataFrame(trades)
            
            # Calculate metrics
            total_trades = len(df)
            winners = df[df['pnl_pct'] > 0]
            win_rate = len(winners) / total_trades * 100
            total_pnl = df['pnl_pct'].sum()
            
            # Risk-adjusted metrics
            if 'risk_percent' in df.columns:
                avg_risk = df['risk_percent'].mean()
                risk_adjusted_return = total_pnl / avg_risk if avg_risk > 0 else 0
            else:
                risk_adjusted_return = 0
            
            # Max consecutive wins/losses
            win_streak = 0
            loss_streak = 0
            max_win_streak = 0
            max_loss_streak = 0
            
            for pnl in df['pnl_pct']:
                if pnl > 0:
                    win_streak += 1
                    loss_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                else:
                    loss_streak += 1
                    win_streak = 0
                    max_loss_streak = max(max_loss_streak, loss_streak)
            
            comparison[name] = {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'risk_adjusted_return': risk_adjusted_return,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                'metadata': data['metadata']
            }
        
        # Rank by total P&L
        ranked = sorted(comparison.items(), key=lambda x: x[1]['total_pnl'], reverse=True)
        
        return {
            'rankings': ranked,
            'comparison': comparison,
            'best_strategy': ranked[0][0] if ranked else None,
            'best_return': ranked[0][1]['total_pnl'] if ranked else 0
        }
    
    def compare_stop_strategies(self, trades_by_stop: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        Specifically compare different stop loss strategies
        Like Simon's analysis: conservative (75% win, 1.2 R:R) vs aggressive (64% win, 3.1 R:R)
        """
        for stop_type, trades in trades_by_stop.items():
            self.add_result(stop_type, trades, {'stop_type': stop_type})
        
        return self.compare()
    
    def generate_comparison_report(self) -> str:
        """Generate a human-readable comparison report"""
        comparison = self.compare()
        
        report = []
        report.append("="*60)
        report.append("STRATEGY COMPARISON REPORT")
        report.append("="*60)
        
        if not comparison.get('rankings'):
            report.append("No strategies to compare")
            return "\n".join(report)
        
        report.append("\nRANKINGS:")
        for i, (name, stats) in enumerate(comparison['rankings'], 1):
            report.append(f"{i}. {name}")
            report.append(f"   Trades: {stats['total_trades']}")
            report.append(f"   Win Rate: {stats['win_rate']:.1f}%")
            report.append(f"   Total P&L: {stats['total_pnl']:.2f}%")
            if stats['risk_adjusted_return']:
                report.append(f"   Risk-Adj Return: {stats['risk_adjusted_return']:.2f}")
            if stats['metadata']:
                report.append(f"   Meta: {stats['metadata']}")
            report.append("")
        
        report.append(f"\nBEST STRATEGY: {comparison['best_strategy']} with {comparison['best_return']:.2f}% return")
        
        return "\n".join(report)
    
    def plot_comparison(self, save_path: str = None):
        """Generate comparison plots (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
            
            if not self.results:
                return
                
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            names = []
            win_rates = []
            returns = []
            
            for name, data in self.results.items():
                trades = data['trades']
                if not trades:
                    continue
                    
                df = pd.DataFrame(trades)
                names.append(name)
                win_rates.append((df['pnl_pct'] > 0).sum() / len(df) * 100)
                returns.append(df['pnl_pct'].sum())
            
            # Win rate comparison
            axes[0, 0].bar(names, win_rates)
            axes[0, 0].set_title('Win Rate by Strategy')
            axes[0, 0].set_ylabel('Win Rate %')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Return comparison
            axes[0, 1].bar(names, returns)
            axes[0, 1].set_title('Total Return by Strategy')
            axes[0, 1].set_ylabel('Return %')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Risk/reward scatter
            for name, data in self.results.items():
                trades = data['trades']
                if not trades:
                    continue
                    
                df = pd.DataFrame(trades)
                if 'risk_percent' in df.columns and 'pnl_pct' in df.columns:
                    axes[1, 0].scatter(df['risk_percent'], df['pnl_pct'], alpha=0.5, label=name)
            
            axes[1, 0].set_title('Risk vs Return')
            axes[1, 0].set_xlabel('Risk %')
            axes[1, 0].set_ylabel('Return %')
            axes[1, 0].legend()
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.3)
            axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.3)
            
            # Cumulative returns
            for name, data in self.results.items():
                trades = data['trades']
                if not trades:
                    continue
                    
                df = pd.DataFrame(trades)
                if 'entry_time' in df.columns and 'pnl_pct' in df.columns:
                    df = df.sort_values('entry_time')
                    axes[1, 1].plot(df['entry_time'], df['pnl_pct'].cumsum(), label=name)
            
            axes[1, 1].set_title('Cumulative Returns')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Cumulative Return %')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=150)
                logger.info(f"Comparison plot saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            logger.warning("matplotlib not installed, skipping plots")
        except Exception as e:
            logger.error(f"Error plotting comparison: {e}")
