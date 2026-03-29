#!/usr/bin/env python3
"""
Analyze trade logs to find patterns and improve strategy
Like Simon's analysis that showed aggressive stops yield 64% vs 7% return
"""

import argparse
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.trade_logger import TradeLogger


def main():
    parser = argparse.ArgumentParser(description='Analyze trade logs')
    parser.add_argument('--input', type=str, required=True,
                        help='Input trade log file (JSON or CSV)')
    parser.add_argument('--output', type=str, default='analysis_report.txt',
                        help='Output analysis file')
    parser.add_argument('--plot', action='store_true',
                        help='Generate plots')
    
    args = parser.parse_args()
    
    # Load trade log
    if args.input.endswith('.csv'):
        df = pd.read_csv(args.input)
    else:
        with open(args.input, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    
    if df.empty:
        print("No trades found")
        return
    
    # Convert types
    df['pnl_percent'] = pd.to_numeric(df['pnl_percent'], errors='coerce')
    
    # Basic stats
    total_trades = len(df)
    winners = df[df['pnl_percent'] > 0]
    losers = df[df['pnl_percent'] <= 0]
    win_rate = len(winners) / total_trades * 100
    
    total_pnl = df['pnl_percent'].sum()
    avg_win = winners['pnl_percent'].mean() if not winners.empty else 0
    avg_loss = losers['pnl_percent'].mean() if not losers.empty else 0
    
    # Generate report
    report = []
    report.append("="*60)
    report.append("SIMON PULLEN TRADE LOG ANALYSIS")
    report.append("="*60)
    report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"Total Trades: {total_trades}")
    report.append(f"Winners: {len(winners)}")
    report.append(f"Losers: {len(losers)}")
    report.append(f"Win Rate: {win_rate:.1f}%")
    report.append(f"Total P&L: {total_pnl:.2f}%")
    report.append(f"Avg Win: {avg_win:.2f}%")
    report.append(f"Avg Loss: {avg_loss:.2f}%")
    report.append(f"Profit Factor: {abs(avg_win * len(winners) / (avg_loss * len(losers))) if avg_loss != 0 else 'N/A'}")
    
    # Analysis by pattern type
    if 'pattern_type' in df.columns:
        report.append("\n" + "-"*40)
        report.append("ANALYSIS BY PATTERN TYPE")
        report.append("-"*40)
        
        for ptype in df['pattern_type'].unique():
            subset = df[df['pattern_type'] == ptype]
            p_winners = subset[subset['pnl_percent'] > 0]
            p_win_rate = len(p_winners) / len(subset) * 100
            p_pnl = subset['pnl_percent'].sum()
            
            report.append(f"\n{ptype}:")
            report.append(f"  Trades: {len(subset)}")
            report.append(f"  Win Rate: {p_win_rate:.1f}%")
            report.append(f"  Total P&L: {p_pnl:.2f}%")
            report.append(f"  Avg P&L: {subset['pnl_percent'].mean():.2f}%")
    
    # Analysis by stop loss type
    if 'stop_loss_type' in df.columns:
        report.append("\n" + "-"*40)
        report.append("ANALYSIS BY STOP LOSS TYPE")
        report.append("-"*40)
        report.append("(Like Simon's comparison of conservative vs aggressive)")
        
        for stype in df['stop_loss_type'].unique():
            subset = df[df['stop_loss_type'] == stype]
            p_winners = subset[subset['pnl_percent'] > 0]
            p_win_rate = len(p_winners) / len(subset) * 100
            p_pnl = subset['pnl_percent'].sum()
            
            # Calculate average risk/reward
            if 'take_profit' in df.columns and 'stop_loss' in df.columns and 'entry_price' in df.columns:
                rr_ratios = []
                for _, row in subset.iterrows():
                    if row['direction'] == 'long':
                        risk = row['entry_price'] - row['stop_loss']
                        reward = row['take_profit'] - row['entry_price']
                    else:
                        risk = row['stop_loss'] - row['entry_price']
                        reward = row['entry_price'] - row['take_profit']
                    
                    if risk > 0:
                        rr_ratios.append(reward / risk)
                
                avg_rr = sum(rr_ratios) / len(rr_ratios) if rr_ratios else 0
            else:
                avg_rr = 0
            
            report.append(f"\n{stype.upper()}:")
            report.append(f"  Trades: {len(subset)}")
            report.append(f"  Win Rate: {p_win_rate:.1f}%")
            report.append(f"  Total P&L: {p_pnl:.2f}%")
            report.append(f"  Avg R:R: {avg_rr:.2f}")
    
    # Analysis by day of week
    if 'entry_time' in df.columns:
        df['entry_time'] = pd.to_datetime(df['entry_time'])
        df['day_of_week'] = df['entry_time'].dt.day_name()
        
        report.append("\n" + "-"*40)
        report.append("ANALYSIS BY DAY OF WEEK")
        report.append("-"*40)
        report.append("(Simon's best days: Tue, Wed, Thu)")
        
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
            subset = df[df['day_of_week'] == day]
            if subset.empty:
                continue
                
            p_winners = subset[subset['pnl_percent'] > 0]
            p_win_rate = len(p_winners) / len(subset) * 100
            p_pnl = subset['pnl_percent'].sum()
            
            report.append(f"\n{day}:")
            report.append(f"  Trades: {len(subset)}")
            report.append(f"  Win Rate: {p_win_rate:.1f}%")
            report.append(f"  Total P&L: {p_pnl:.2f}%")
    
    # Save report
    with open(args.output, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"\nAnalysis saved to {args.output}")
    
    # Generate plots if requested
    if args.plot:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Cumulative P&L
        df_sorted = df.sort_values('entry_time')
        df_sorted['cumulative_pnl'] = df_sorted['pnl_percent'].cumsum()
        
        axes[0, 0].plot(df_sorted['entry_time'], df_sorted['cumulative_pnl'])
        axes[0, 0].set_title('Cumulative P&L')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('P&L %')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rate by day
        if 'day_of_week' in df.columns:
            day_stats = df.groupby('day_of_week')['pnl_percent'].agg(['count', 'mean', lambda x: (x > 0).sum()])
            day_stats.columns = ['count', 'avg_pnl', 'wins']
            day_stats['win_rate'] = day_stats['wins'] / day_stats['count'] * 100
            
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            day_stats = day_stats.reindex(days_order).dropna()
            
            axes[0, 1].bar(day_stats.index, day_stats['win_rate'])
            axes[0, 1].set_title('Win Rate by Day')
            axes[0, 1].set_ylabel('Win Rate %')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].grid(True, alpha=0.3)
        
        # Win rate by pattern
        if 'pattern_type' in df.columns:
            pattern_stats = df.groupby('pattern_type')['pnl_percent'].agg(['count', 'mean', lambda x: (x > 0).sum()])
            pattern_stats.columns = ['count', 'avg_pnl', 'wins']
            pattern_stats['win_rate'] = pattern_stats['wins'] / pattern_stats['count'] * 100
            
            axes[1, 0].bar(pattern_stats.index, pattern_stats['win_rate'])
            axes[1, 0].set_title('Win Rate by Pattern')
            axes[1, 0].set_ylabel('Win Rate %')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Risk/Reward distribution
        if 'pnl_percent' in df.columns:
            axes[1, 1].hist(df['pnl_percent'], bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].axvline(x=0, color='red', linestyle='--')
            axes[1, 1].set_title('P&L Distribution')
            axes[1, 1].set_xlabel('P&L %')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('trade_analysis.png', dpi=150)
        print("Plot saved to trade_analysis.png")
        plt.show()


if __name__ == '__main__':
    main()
