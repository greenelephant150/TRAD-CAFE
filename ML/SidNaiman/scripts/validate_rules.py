#!/usr/bin/env python3
"""
Validate Simon Pullen's rules against historical data
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.core.mw_pattern import MWPatternDetector
from src.core.head_shoulders import HeadShouldersDetector
from src.confluence.divergence import DivergenceDetector
from config.pattern_rules import MW_RULES, HS_RULES


def main():
    parser = argparse.ArgumentParser(description='Validate Simon Pullen rules')
    parser.add_argument('--instrument', type=str, default='EUR/USD')
    parser.add_argument('--timeframe', type=str, default='1h')
    parser.add_argument('--days', type=int, default=90)
    
    args = parser.parse_args()
    
    # Generate sample data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    # Use 'h' instead of 'H' for hourly frequency (pandas >= 2.0)
    freq_map = {
        '1h': 'h',
        '4h': '4h',
        '15m': '15min',
        '1d': 'D'
    }
    freq = freq_map.get(args.timeframe, 'h')
    
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    np.random.seed(42)
    
    # Generate random walk
    returns = np.random.randn(len(dates)) * 0.001
    price = 1.1000 * np.exp(np.cumsum(returns))
    
    # Add some noise for OHLC
    df = pd.DataFrame({
        'open': price * (1 + np.random.randn(len(dates)) * 0.0002),
        'high': price * (1 + np.abs(np.random.randn(len(dates)) * 0.0005)),
        'low': price * (1 - np.abs(np.random.randn(len(dates)) * 0.0005)),
        'close': price,
        'volume': np.random.randint(100, 1000, len(dates))
    }, index=dates)
    
    # Ensure high >= open/close and low <= open/close
    df['high'] = df[['high', 'open', 'close']].max(axis=1)
    df['low'] = df[['low', 'open', 'close']].min(axis=1)
    
    print(f"Validating rules on {len(df)} bars of {args.instrument} {args.timeframe}")
    print("="*60)
    
    # Test M&W detection
    mw_detector = MWPatternDetector(MW_RULES)
    
    m_patterns = mw_detector.detect_m_top(df, args.instrument, args.timeframe)
    w_patterns = mw_detector.detect_w_bottom(df, args.instrument, args.timeframe)
    
    print(f"\nM-Tops detected: {len(m_patterns)}")
    print(f"W-Bottoms detected: {len(w_patterns)}")
    
    # Show first few patterns
    for i, p in enumerate(m_patterns[:3]):
        print(f"\nM-Top {i+1}:")
        print(f"  Candle count: {p.candle_count} (rules: {MW_RULES['min_candles']}-{MW_RULES['max_candles']})")
        print(f"  Entry: {p.entry_price:.5f}")
        print(f"  Stop: {p.stop_loss_price:.5f}")
        print(f"  TP: {p.take_profit_price:.5f}")
        print(f"  Valid: {p.valid}")
        if p.validation_errors:
            print(f"  Errors: {p.validation_errors}")
    
    # Test H&S detection
    hs_detector = HeadShouldersDetector(HS_RULES)
    
    hs_patterns = hs_detector.detect_normal(df, args.instrument, args.timeframe)
    inv_patterns = hs_detector.detect_inverted(df, args.instrument, args.timeframe)
    
    print(f"\n\nHead & Shoulders detected: {len(hs_patterns)}")
    print(f"Inverted H&S detected: {len(inv_patterns)}")
    
    for i, p in enumerate(hs_patterns[:3]):
        print(f"\nH&S {i+1}:")
        print(f"  Candle count: {p.candle_count} (rules: {HS_RULES['min_candles']}-{HS_RULES['max_candles']})")
        print(f"  Break idx: {p.break_idx}")
        print(f"  Retest idx: {p.retest_idx}")
        print(f"  Entry candle: {p.entry_candle_type}")
        print(f"  Valid: {p.valid}")
    
    # Test divergence
    div_detector = DivergenceDetector({})
    divergences = div_detector.detect_all(df)
    
    print(f"\n\nDivergences detected: {len(divergences)}")
    for i, d in enumerate(divergences[:5]):
        print(f"  {d.divergence_type}: strength {d.strength:.2f}")
    
    print("\n" + "="*60)
    print("Rule validation complete")


if __name__ == '__main__':
    main()
