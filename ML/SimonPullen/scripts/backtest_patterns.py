#!/usr/bin/env python3
"""
Backtest Simon Pullen patterns on historical data
Supports CPU/GPU processing with automatic fallback
Colorful verbose output with tqdm progress bars
"""

import argparse
import sys
import os
import json
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm

# Import modules
try:
    from src.backtesting.bar_replay import BarReplay
    from src.data.data_manager import DataManager
    from config.pattern_rules import MW_RULES, HS_RULES, RISK_RULES
    from src.utils.device_manager import get_device_manager, set_preferred_device
    from src.utils.color_logger import get_logger, Colors, EMOJIS
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are created")
    sys.exit(1)


def parse_args():
    parser = argparse.ArgumentParser(description='Backtest Simon Pullen patterns')
    parser.add_argument('--instrument', type=str, default='EUR_USD',
                        help='Instrument to backtest (use underscore format, e.g., EUR_USD)')
    parser.add_argument('--timeframe', type=str, default='1h',
                        choices=['1h', '4h', '15m', '1d', '5s'],
                        help='Timeframe (1h, 4h, 15m, 1d, 5s)')
    parser.add_argument('--days', type=int, default=365,
                        help='Days of historical data')
    parser.add_argument('--start-year', type=int, default=None,
                        help='Start year for data loading (overrides --days)')
    parser.add_argument('--end-year', type=int, default=None,
                        help='End year for data loading')
    parser.add_argument('--balance', type=float, default=10000,
                        help='Initial account balance')
    parser.add_argument('--stop-type', type=str, default='moderate',
                        choices=['conservative', 'moderate', 'aggressive'],
                        help='Stop loss type')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda', 'mps'],
                        help='Processing device (auto, cpu, cuda, mps)')
    parser.add_argument('--parallel', action='store_true',
                        help='Use parallel processing across multiple GPUs/cores')
    parser.add_argument('--output', type=str, default='backtest_results.json',
                        help='Output file')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--progress', action='store_true', default=True,
                        help='Show progress bars')
    
    return parser.parse_args()


def load_data(args, logger):
    """Load data for backtesting with progress indication"""
    logger.section(f"{EMOJIS['data']} Loading Data", 'data')
    
    dm = DataManager()
    
    if args.start_year and args.end_year:
        # Load by year range
        logger.info(f"Loading data from {args.start_year} to {args.end_year}")
        
        # Show year progress
        years = list(range(args.start_year, args.end_year + 1))
        if args.progress:
            pbar = tqdm(years, desc="Loading years", unit="year", colour="green")
            all_data = []
            for year in pbar:
                df_year = dm.load_pair_data(args.instrument, year, year)
                if not df_year.empty:
                    all_data.append(df_year)
                pbar.set_postfix({"rows": sum(len(d) for d in all_data) if all_data else 0})
            
            if all_data:
                df = pd.concat(all_data)
            else:
                df = pd.DataFrame()
        else:
            df = dm.load_pair_data(args.instrument, args.start_year, args.end_year)
    else:
        # Load by days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        logger.info(f"Loading data from {start_date.date()} to {end_date.date()}")
        
        if args.progress:
            with tqdm(total=1, desc="Loading recent data", unit="file", colour="green") as pbar:
                df = dm.get_recent_data(args.instrument, args.days, args.timeframe)
                pbar.update(1)
                pbar.set_postfix({"bars": len(df)})
        else:
            df = dm.get_recent_data(args.instrument, args.days, args.timeframe)
    
    if df.empty:
        logger.warning(f"No data found for {args.instrument}, generating sample data", 'warning')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        if args.progress:
            with tqdm(total=1, desc="Generating sample data", unit="sample", colour="yellow") as pbar:
                df = dm.generate_sample_data(args.instrument, start_date.year, end_date.year)
                pbar.update(1)
                if args.timeframe != '5s':
                    df = dm.resample_to_timeframe(df, args.timeframe)
                pbar.set_postfix({"bars": len(df)})
        else:
            df = dm.generate_sample_data(args.instrument, start_date.year, end_date.year)
            if args.timeframe != '5s':
                df = dm.resample_to_timeframe(df, args.timeframe)
    
    logger.success(f"Loaded {len(df):,} bars of {args.timeframe} data")
    return df


def main():
    args = parse_args()
    
    # Initialize logger
    logger = get_logger(args.verbose)
    
    # Initialize device manager
    device_manager = set_preferred_device(args.device)
    device_info = device_manager.get_device_info()
    
    # Print header
    logger.header("SIMON PULLEN BACKTEST ENGINE", "rocket")
    
    # Device info
    logger.device_info(device_info['device'], device_info['name'], device_info['type'])
    logger.info(f"Parallel mode: {Colors.BOLD}{'Enabled' if args.parallel else 'Disabled'}{Colors.END}", 'cpu' if not args.parallel else 'gpu')
    logger.info(f"Instrument: {Colors.BOLD}{args.instrument}{Colors.END}", 'chart')
    logger.info(f"Timeframe: {Colors.BOLD}{args.timeframe}{Colors.END}", 'time')
    logger.info(f"Stop type: {Colors.BOLD}{args.stop_type}{Colors.END}", 'stop')
    logger.info(f"Initial balance: {Colors.BOLD}${args.balance:,.2f}{Colors.END}", 'money')
    
    if args.progress:
        logger.info(f"Progress bars: {Colors.GREEN}Enabled{Colors.END}", 'check')
    
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    # Load config
    config = {
        **MW_RULES,
        **HS_RULES,
        **RISK_RULES,
        'default_stop_type': args.stop_type,
        'device': args.device,
        'parallel': args.parallel,
        'num_workers': device_manager.get_optimal_workers() if args.parallel else 1,
        'verbose': args.verbose,
        'progress': args.progress
    }
    
    # Load data
    df = load_data(args, logger)
    
    if df.empty:
        logger.error("No data available for backtesting", 'error')
        sys.exit(1)
    
    # Show data sample
    if args.verbose:
        logger.section(f"{EMOJIS['chart']} Data Sample", 'chart')
        logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")
        logger.debug(f"Columns: {list(df.columns)}")
        logger.debug(f"Price range: {df['low'].min():.5f} - {df['high'].max():.5f}")
    
    # Run backtest with progress
    logger.section(f"{EMOJIS['trade']} Running Backtest", 'trade')
    
    replay = BarReplay(config, args.balance)
    
    if args.progress:
        # Show progress during backtest
        total_bars = len(df)
        with tqdm(total=total_bars, desc="Processing bars", unit="bar", 
                  colour="cyan", dynamic_ncols=True) as pbar:
            
            # Monkey patch the step method to update progress
            original_step = replay.step
            def step_with_progress(steps=1):
                result = original_step(steps)
                pbar.update(steps)
                if replay.current_idx % 100 == 0:
                    pbar.set_postfix({
                        "patterns": sum(len(v) for v in replay.patterns.values()),
                        "positions": len(replay.positions)
                    })
                return result
            replay.step = step_with_progress
            
            # Run backtest
            replay.load_data(df, args.instrument, args.timeframe)
            replay.step_to_end()
    else:
        # Run without progress
        replay.run_full_backtest(df, args.instrument, args.timeframe)
    
    # Get results
    results = replay.get_results()
    
    # Print summary
    logger.summary(results)
    
    # Save results
    output_file = args.output.replace('.json', f'_{args.device}.json')
    with open(output_file, 'w') as f:
        json.dump({
            'results': results,
            'config': config,
            'device_info': device_info,
            'args': vars(args)
        }, f, indent=2, default=str)
    
    logger.success(f"Results saved to {output_file}", 'save')
    
    # Show trade log if verbose
    if args.verbose and results.get('trades'):
        logger.section(f"{EMOJIS['list']} Recent Trades", 'list')
        
        # Show last 5 trades
        for i, trade in enumerate(results['trades'][-5:]):
            exit_status = trade.get('exit_status', {})
            pnl = exit_status.get('pnl_pct', 0)
            color = Colors.GREEN if pnl > 0 else Colors.RED
            emoji = EMOJIS['profit'] if pnl > 0 else EMOJIS['loss']
            
            print(f"  {emoji} Trade #{i+1}: {color}{pnl:+.2f}%{Colors.END} - "
                  f"{exit_status.get('exit_reason', 'unknown')} "
                  f"({exit_status.get('bars_held', 0)} bars)")


if __name__ == '__main__':
    main()
