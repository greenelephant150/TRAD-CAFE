#!/usr/bin/env python3
"""
SID Method Trading System - Main Entry Point
=============================================================================
Complete trading system incorporating ALL THREE WAVES of strategies:

- Wave 1: Core Quick Win (RSI 30/70, MACD, stop loss, take profit)
- Wave 2: Live Sessions (market context, divergence, patterns)
- Wave 3: Academy Support (precision, pip buffers, session filters)

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules
from sid_method import SidMethod
from oanda_client import OANDAClient
from src.risk.position_sizer import PositionSizer, PositionSizerConfig
from src.risk.correlation_manager import CorrelationManager, CorrelationConfig
from src.risk.news_filter import NewsFilter, NewsFilterConfig
from src.risk.time_filter import TimeFilter, TimeFilterConfig
from src.execution.entry_rules import EntryRules, EntryRulesConfig
from src.execution.exit_rules import ExitRules, ExitRulesConfig
from src.execution.stop_loss_calculator import StopLossCalculator, StopLossConfig
from src.backtesting.bar_replay import BarReplay, BarReplayConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/sid_method.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class SidTradingSystem:
    """
    Main trading system integrating all SID Method components
    """
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """
        Initialize trading system
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self._init_core()
        self._init_risk_management()
        self._init_execution()
        self._init_backtesting()
        
        # Initialize OANDA client (if configured)
        if self.config.get('broker', {}).get('api_key'):
            self.oanda = OANDAClient()
        else:
            self.oanda = None
            logger.warning("OANDA API key not configured - using paper trading mode")
        
        logger.info("✅ SID Method Trading System initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    def _init_core(self):
        """Initialize core SID Method components"""
        sid_config = self.config.get('sid_method', {})
        self.sid = SidMethod(
            account_balance=10000,
            verbose=True,
            prefer_macd_cross=sid_config.get('prefer_macd_cross', True),
            use_pattern_confirmation=self.config.get('live_sessions', {}).get('use_pattern_confirmation', True),
            use_divergence=self.config.get('live_sessions', {}).get('use_divergence', True),
            use_market_context=self.config.get('live_sessions', {}).get('use_market_context', True)
        )
    
    def _init_risk_management(self):
        """Initialize risk management components"""
        risk_config = self.config.get('risk_management', {})
        
        # Position Sizer
        pos_config = PositionSizerConfig(
            default_risk_percent=risk_config.get('default_risk_percent', 1.0),
            min_risk_percent=risk_config.get('min_risk_percent', 0.5),
            max_risk_percent=risk_config.get('max_risk_percent', 2.0),
            max_daily_loss_percent=risk_config.get('max_daily_loss_percent', 5.0)
        )
        self.position_sizer = PositionSizer(pos_config, verbose=False)
        
        # Correlation Manager
        corr_config = CorrelationConfig(
            max_active_trades=risk_config.get('max_active_trades', 5),
            max_correlated_trades=risk_config.get('max_correlated_trades', 2),
            correlation_threshold=risk_config.get('correlation_threshold', 0.7),
            max_sector_exposure_percent=risk_config.get('max_sector_exposure_percent', 25.0)
        )
        self.correlation_manager = CorrelationManager(corr_config, verbose=False)
        
        # News Filter
        news_config = NewsFilterConfig(
            earnings_buffer_hours=risk_config.get('earnings_buffer_hours', 24),
            high_impact_buffer_hours=risk_config.get('high_impact_buffer_hours', 4),
            medium_impact_buffer_hours=risk_config.get('medium_impact_buffer_hours', 2)
        )
        self.news_filter = NewsFilter(news_config, verbose=False)
        
        # Time Filter
        time_config = TimeFilterConfig(
            allowed_sessions=risk_config.get('allowed_sessions', ['overlap', 'us', 'london']),
            blocked_sessions=risk_config.get('blocked_sessions', ['asian']),
            allow_weekends=False
        )
        self.time_filter = TimeFilter(time_config, verbose=False)
    
    def _init_execution(self):
        """Initialize execution components"""
        # Entry Rules
        entry_config = EntryRulesConfig(
            rsi_oversold=self.config['sid_method']['rsi_oversold'],
            rsi_overbought=self.config['sid_method']['rsi_overbought'],
            prefer_macd_cross=self.config['sid_method']['prefer_macd_cross'],
            require_pattern_confirmation=self.config['live_sessions'].get('use_pattern_confirmation', True),
            require_market_context=self.config['live_sessions'].get('use_market_context', True),
            session_filter_enabled=self.config['academy_support'].get('use_session_filter', True)
        )
        self.entry_rules = EntryRules(entry_config, verbose=False)
        
        # Exit Rules
        exit_config = ExitRulesConfig(
            use_rsi_50_exit=True,
            use_sma_50_exit=True,
            use_point_target_exit=True,
            use_trailing_stop=False,
            use_reversal_exit=True,
            use_time_stop=True,
            max_hold_bars=50
        )
        self.exit_rules = ExitRules(exit_config, verbose=False)
        
        # Stop Loss Calculator
        sl_config = StopLossConfig(
            use_sid_method_stops=True,
            use_pip_buffer=self.config['academy_support'].get('use_zone_quality', True),
            pip_buffer_default=self.config['academy_support'].get('stop_pips_default', 5),
            pip_buffer_yen=self.config['academy_support'].get('stop_pips_yen', 10)
        )
        self.stop_loss_calc = StopLossCalculator(sl_config, verbose=False)
    
    def _init_backtesting(self):
        """Initialize backtesting components"""
        bt_config = BarReplayConfig(
            initial_balance=self.config['backtesting'].get('initial_balance', 10000),
            risk_percent=self.config['sid_method']['default_risk_percent'],
            max_open_trades=self.config['backtesting'].get('max_open_trades', 3),
            use_pattern_confirmation=self.config['live_sessions'].get('use_pattern_confirmation', True),
            use_divergence=self.config['live_sessions'].get('use_divergence', True),
            use_market_context=self.config['live_sessions'].get('use_market_context', True),
            use_session_filter=self.config['academy_support'].get('use_session_filter', True)
        )
        self.backtest = BarReplay(bt_config)
    
    def scan_for_signals(self, df, instrument: str) -> List[Dict]:
        """
        Scan for trade signals
        
        Args:
            df: Price DataFrame
            instrument: Instrument symbol
        
        Returns:
            List of trade signals
        """
        signals = []
        
        # Calculate indicators
        df = df.copy()
        df['rsi'] = self.sid.calculate_rsi(df)
        macd_df = self.sid.calculate_macd(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # Scan for signals
        for i in range(50, len(df) - 1):
            current_date = df.index[i]
            rsi_value = df['rsi'].iloc[i]
            
            # Check market trend
            market_trend = self.sid.analyze_market_trend(df)
            
            # Check session
            session = self.time_filter.get_trading_session(current_date).value
            session_valid, _ = self.time_filter.validate_session(current_date)
            
            if not session_valid:
                continue
            
            # Check news
            news_valid, _, _ = self.news_filter.check_trade(instrument, current_date)
            if not news_valid:
                continue
            
            # Check RSI signal
            signal_type, is_exact = self.entry_rules.validate_rsi_signal(rsi_value)
            if signal_type == 'neutral':
                continue
            
            # Check MACD
            macd_aligned = self.entry_rules.validate_macd(df, i, signal_type)[0]
            if not macd_aligned:
                continue
            
            macd_crossed = self.entry_rules.validate_macd(df, i, signal_type)[1]
            
            # Check pattern confirmation
            pattern_confirmed = False
            pattern_name = ''
            if self.entry_rules.config.require_pattern_confirmation:
                pattern_result = self.entry_rules.validate_pattern_confirmation(df, i, signal_type)
                pattern_confirmed = pattern_result[0]
                pattern_name = pattern_result[1] if pattern_confirmed else ''
            
            # Build signal
            direction = 'long' if signal_type == 'oversold' else 'short'
            entry_price = df['close'].iloc[i]
            
            # Find signal date
            signal_date = self._find_signal_date(df, i, signal_type)
            
            # Calculate stop loss
            stop_loss = self.stop_loss_calc.calculate_sid_stop_loss(
                df, signal_date, current_date, signal_type, instrument
            )
            
            if stop_loss.stop_price <= 0:
                continue
            
            # Calculate take profit
            tp_result = self.exit_rules.calculate_take_profit(
                entry_price, stop_loss.stop_price, direction, df['sma_50'].iloc[i]
            )
            
            signals.append({
                'date': current_date,
                'instrument': instrument,
                'direction': direction,
                'signal_type': signal_type,
                'rsi_value': rsi_value,
                'entry_price': entry_price,
                'stop_loss': stop_loss.stop_price,
                'take_profit': tp_result['primary_tp'],
                'take_profit_alt': tp_result['alternative_tp'],
                'reward_ratio': tp_result['reward_ratio'],
                'macd_crossed': macd_crossed,
                'pattern_confirmed': pattern_confirmed,
                'pattern_name': pattern_name,
                'market_trend': market_trend.value if hasattr(market_trend, 'value') else str(market_trend),
                'session': session
            })
        
        return signals
    
    def _find_signal_date(self, df, idx, signal_type):
        """Find the date when RSI first triggered"""
        rsi_values = df['rsi'].iloc[:idx+1]
        threshold = 30 if signal_type == 'oversold' else 70
        
        if signal_type == 'oversold':
            mask = rsi_values < threshold
        else:
            mask = rsi_values > threshold
        
        if mask.any():
            return mask[mask].index[-1]
        return df.index[idx]
    
    def run_backtest(self, df, instrument: str) -> Dict:
        """
        Run backtest on historical data
        
        Args:
            df: Price DataFrame
            instrument: Instrument symbol
        
        Returns:
            Backtest metrics
        """
        logger.info(f"Running backtest for {instrument}...")
        
        metrics = self.backtest.run(df)
        
        logger.info(f"Backtest complete: {metrics.total_trades} trades, Win Rate: {metrics.win_rate:.1f}%, Net Profit: ${metrics.net_profit:.2f}")
        
        return metrics.to_dict()
    
    def get_system_status(self) -> Dict:
        """Get system status"""
        return {
            'version': self.config['system']['version'],
            'environment': self.config['system']['environment'],
            'components': {
                'sid_method': 'active',
                'oanda_client': 'connected' if self.oanda else 'not_configured',
                'risk_management': 'active',
                'backtesting': 'ready'
            },
            'config_summary': {
                'rsi_thresholds': f"{self.config['sid_method']['rsi_oversold']}/{self.config['sid_method']['rsi_overbought']}",
                'prefer_macd_cross': self.config['sid_method']['prefer_macd_cross'],
                'allowed_sessions': self.config['risk_management']['allowed_sessions'],
                'max_risk': f"{self.config['sid_method']['max_risk_percent']}%"
            }
        }
    
    def print_summary(self):
        """Print system summary"""
        status = self.get_system_status()
        
        print("\n" + "="*60)
        print(f"📊 SID METHOD TRADING SYSTEM v{status['version']}")
        print("="*60)
        print(f"Environment: {status['environment']}")
        print(f"RSI Thresholds: {status['config_summary']['rsi_thresholds']}")
        print(f"MACD Cross Preferred: {status['config_summary']['prefer_macd_cross']}")
        print(f"Allowed Sessions: {status['config_summary']['allowed_sessions']}")
        print(f"Max Risk: {status['config_summary']['max_risk']}")
        print("\nComponents:")
        for name, status in status['components'].items():
            print(f"  {name}: {status}")
        print("="*60)


def main():
    parser = argparse.ArgumentParser(description='SID Method Trading System')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--mode', type=str, choices=['scan', 'backtest', 'status'], default='status', help='Operation mode')
    parser.add_argument('--data', type=str, help='Data file for backtest')
    parser.add_argument('--instrument', type=str, default='EUR_USD', help='Instrument symbol')
    parser.add_argument('--output', type=str, default='backtest_results/', help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize system
    system = SidTradingSystem(args.config)
    
    if args.mode == 'status':
        system.print_summary()
    
    elif args.mode == 'scan':
        if not args.data:
            print("Error: --data required for scan mode")
            sys.exit(1)
        
        # Load data
        if args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        
        # Scan for signals
        signals = system.scan_for_signals(df, args.instrument)
        
        print(f"\n📊 Found {len(signals)} signals for {args.instrument}:")
        for s in signals[:10]:
            print(f"  {s['date']}: {s['direction'].upper()} @ {s['entry_price']:.5f}")
            print(f"    RSI: {s['rsi_value']:.1f} | Reward: {s['reward_ratio']:.2f}R")
            if s['pattern_confirmed']:
                print(f"    Pattern: {s['pattern_name']}")
    
    elif args.mode == 'backtest':
        if not args.data:
            print("Error: --data required for backtest mode")
            sys.exit(1)
        
        # Load data
        if args.data.endswith('.parquet'):
            df = pd.read_parquet(args.data)
        else:
            df = pd.read_csv(args.data, index_col=0, parse_dates=True)
        
        # Run backtest
        metrics = system.run_backtest(df, args.instrument)
        
        # Save results
        os.makedirs(args.output, exist_ok=True)
        import json
        with open(f"{args.output}/backtest_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\n✅ Backtest results saved to {args.output}")


if __name__ == "__main__":
    import pandas as pd
    main()