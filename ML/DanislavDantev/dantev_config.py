#!/usr/bin/env python3
"""
Danislav Dantev Trading System Configuration
Core Concepts:
- Institutional Order Flow
- Liquidity Sweeps & Stop Hunts
- Order Blocks (Bullish/Bearish)
- Fair Value Gaps (FVG)
- Mitigation Blocks
- Break of Structure (BOS)
- Change of Character (CHoCH)
- Premium/Discount Arrays
- Optimal Trade Entry (OTE)
- Fibonacci Retracements (61.8%, 70.5%, 79%)
"""

from pathlib import Path
import os
from datetime import datetime

# =============================================================================
# PATHS - UPDATE THESE TO MATCH YOUR SYSTEM
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.absolute()

# Model paths
MODEL_PATH = Path("/mnt2/Trading-Cafe/ML/DDantev/ai/trained_models/")
AI_PATH = PROJECT_ROOT / "ai"

# Data paths
CSV_PATH = Path("/home/grct/Forex")
PARQUET_PATH = Path("/home/grct/Forex_Parquet")

# Create directories
def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [MODEL_PATH, AI_PATH, CSV_PATH, PARQUET_PATH]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Ensured directory: {directory}")

ensure_directories()

# =============================================================================
# DANISLAV DANTEV INSTITUTIONAL CONCEPTS
# =============================================================================

# Order Block Configuration
ORDER_BLOCK_CONFIG = {
    'min_candles': 3,                    # Minimum candles in order block
    'max_candles': 8,                    # Maximum candles in order block
    'mitigation_zone_pct': 0.005,        # 0.5% mitigation zone
    'aggressive_mitigation': 0.003,      # 0.3% for aggressive entries
    'conservative_mitigation': 0.01,     # 1% for conservative entries
    'require_breakout': True,            # Must break previous structure
    'require_retest': True,              # Must retest order block
    'consecutive_blocks': 3,             # Look for 3+ order blocks in sequence
    'volume_spike_threshold': 1.5,       # Volume must be 1.5x average
    'strength_threshold': 0.6,           # Minimum order block strength
}

# Fair Value Gap (FVG) Configuration
FVG_CONFIG = {
    'min_gap_pips': 10,                  # Minimum gap size in pips
    'max_age_bars': 50,                  # Maximum age of FVG to consider
    'require_imbalance': True,           # Must show volume imbalance
    'gap_fill_probability': 0.68,        # 68% of gaps get filled
    'institutional_gap_multiplier': 2.5,  # Institutional gaps are larger
    'mitigation_levels': [0.382, 0.5, 0.618, 0.705, 0.79],  # OTE levels
    'fvg_strength_factors': {
        'size': 0.4,                     # Gap size contributes 40%
        'volume': 0.3,                   # Volume contributes 30%
        'age': 0.3                       # Age contributes 30%
    }
}

# Liquidity Configuration
LIQUIDITY_CONFIG = {
    'swing_high_lookback': 20,           # Bars to look for swing highs
    'swing_low_lookback': 20,            # Bars to look for swing lows
    'equal_highs_tolerance': 0.001,      # 0.1% tolerance for equal highs
    'equal_lows_tolerance': 0.001,       # 0.1% tolerance for equal lows
    'trendline_liquidity_bars': 50,      # Bars for trendline liquidity
    'double_top_bottom_bars': 30,        # Bars for double top/bottom detection
    'liquidity_sweep_confirmation': 2,   # Candles to confirm sweep
    'stop_hunt_distance': 0.002,         # 0.2% beyond level for stop hunt
}

# Break of Structure (BOS) Configuration
BOS_CONFIG = {
    'confirmation_candles': 2,           # Candles to confirm break
    'bos_threshold_pips': 15,            # Minimum break in pips
    'require_volume_spike': True,        # Volume spike confirms BOS
    'bos_multiplier': 1.5,               # BOS must exceed previous swing by 1.5x
    'bos_confirmation_close': True,      # Must close beyond level
    'bos_retest_allowed': True,          # Allow retest after BOS
}

# Change of Character (CHoCH) Configuration
CHOCH_CONFIG = {
    'confirmation_candles': 3,           # Candles to confirm CHoCH
    'require_break_structure': True,     # Must break previous structure
    'retracement_threshold': 0.382,      # 38.2% retracement confirms CHoCH
    'momentum_shift_bars': 5,            # Bars to detect momentum shift
    'volume_divergence': True,           # Check for volume divergence
    'choch_strength_threshold': 0.5,     # Minimum CHoCH strength
}

# Fibonacci/OTE Configuration
FIBONACCI_CONFIG = {
    'golden_ratio': 0.618,               # 61.8% key level
    'deep_retracement': 0.705,           # 70.5% deep retracement
    'extreme_retracement': 0.79,         # 79% extreme retracement
    'ote_zones': [0.618, 0.705, 0.79],   # Optimal Trade Entry zones
    'premium_zone': 0.382,               # Premium zone (above 38.2%)
    'discount_zone': 0.618,              # Discount zone (below 61.8%)
    'fib_levels': [0.236, 0.382, 0.5, 0.618, 0.705, 0.786, 0.886],
}

# Premium/Discount Array Configuration
PD_ARRAY_CONFIG = {
    'range_bars': 50,                    # Bars to determine range
    'premium_threshold': 0.618,          # Above 61.8% = premium
    'discount_threshold': 0.382,         # Below 38.2% = discount
    'fair_value': 0.5,                  # 50% = fair value
    'premium_discount_smoothing': 5,     # Smoothing period
}

# Market Structure Configuration
MARKET_STRUCTURE_CONFIG = {
    'trend_following_bars': 20,          # Bars for trend detection
    'consolidation_bars': 15,            # Bars to identify consolidation
    'breakout_volume_multiplier': 1.5,   # Volume must be 1.5x average
    'structural_support_resistance': 10, # Key levels lookback
    'trend_strength_threshold': 0.6,     # Minimum trend strength
}

# Risk Management (Danislav's Rules)
RISK_CONFIG = {
    'default_risk_percent': 0.5,         # 0.5% per trade (more conservative)
    'max_risk_percent': 1.0,             # 1% maximum with high confluence
    'risk_reward_min': 2.0,              # Minimum 2:1 risk/reward
    'risk_reward_target': 3.0,           # Target 3:1
    'risk_reward_aggressive': 5.0,       # Aggressive 5:1 setups
    'max_daily_risk': 2.0,               # Max 2% daily loss
    'max_consecutive_losses': 3,         # Stop after 3 consecutive losses
    'position_scaling': True,            # Scale into positions
    'pyramiding': False,                 # Pyramiding only in strong trends
    'partial_profit_levels': [0.5, 0.75], # Take partial at 50%, 75% of target
}

# AI Model Configuration
AI_CONFIG = {
    'lookback': 50,                      # 50 bars lookback for features
    'train_size': 100000,                # Training samples
    'validation_split': 0.2,             # 20% validation
    'test_size': 0.1,                    # 10% test
    'features': [
        'order_block_strength',
        'order_block_type',
        'fvg_size',
        'fvg_age',
        'fvg_strength',
        'liquidity_swept_highs',
        'liquidity_swept_lows',
        'equal_highs_count',
        'equal_lows_count',
        'bos_confirmed',
        'bos_volume_spike',
        'bos_direction',
        'choch_count',
        'premium_discount_value',
        'is_premium',
        'is_discount',
        'ote_alignment',
        'confluence_score',
        'trend_strength',
        'momentum_shift',
        'institutional_pressure',
        'volume_imbalance',
        'market_structure'
    ]
}

# =============================================================================
# OANDA CONFIGURATION
# =============================================================================
PRACTICE_API_KEY = "fceae94e861642af6c9d9de1bf6a319d-52074b19e787811450bc44152cb71e78"
LIVE_API_KEY = "c4f1b1d0739e88bcc604b26115db4787-1abf7348cce8c429b76a9fcf97b0b97a"
PRACTICE_ACCOUNT_ID = "101-004-35778624-001"
LIVE_ACCOUNT_ID = "001-004-17934933-001"
PRACTICE_URL = "https://api-fxpractice.oanda.com/v3"
LIVE_URL = "https://api-fxtrade.oanda.com/v3"

# Valid OANDA pairs
VALID_OANDA_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "EUR_CHF", "AUD_NZD",
    "NZD_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD", "CAD_JPY", "CHF_JPY",
    "EUR_CAD", "AUD_CAD", "NZD_CAD", "EUR_NZD", "USD_NOK", "USD_SEK", "USD_TRY",
    "EUR_NOK", "EUR_SEK", "EUR_TRY", "GBP_NOK", "GBP_SEK", "GBP_TRY"
]

# Granularity mapping
GRANULARITY_MAP = {
    '1m': 'M1', '5m': 'M5', '15m': 'M15', '30m': 'M30',
    '1h': 'H1', '2h': 'H2', '3h': 'H3', '4h': 'H4',
    '6h': 'H6', '8h': 'H8', '12h': 'H12',
    '1d': 'D', '1w': 'W', '1M': 'M',
}

# =============================================================================
# STREAMLIT UI SETTINGS
# =============================================================================
APP_TITLE = "Danislav Dantev Institutional Trading System"
APP_VERSION = "v1.0 - Smart Money Concepts"
RISK_WARNING = "⚠️ Risk Warning: Trading carries substantial risk. Only trade with risk capital."

print(f"✅ Dantev configuration loaded")
print(f"   Model path: {MODEL_PATH}")
print(f"   CSV path: {CSV_PATH}")
print(f"   Parquet path: {PARQUET_PATH}")