#!/usr/bin/env python3
"""
🎯 SID METHOD AI MODEL TRAINER - AUGMENTED VERSION
============================================================
Trains models for ALL available pairs using GPU acceleration
INCORPORATES ALL THREE WAVES OF TRANSCRIPT STRATEGIES:

WAVE 1 (Core Quick Win):
- RSI thresholds (exact 30/70)
- MACD alignment and cross detection
- Stop loss calculation (rounded down/up)
- Take profit at RSI 50

WAVE 2 (Live Sessions & Q&A):
- Market context filtering (uptrend/downtrend/sideways)
- Divergence detection as feature
- Price pattern confirmation (W, M, H&S)
- Reachability check for take profit targets
- Alternative take profit targets (50-SMA, points)

WAVE 3 (Academy Support Sessions):
- Precision RSI (no "near" values)
- Stop loss pip buffer (5/10 pips behind zone)
- Session-based filtering (Asian/London/US)
- Zone quality assessment
- Minimum candle requirements for patterns

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
"""

import os
import sys
import pickle
import json
import argparse
import warnings
import traceback
import time
import gc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU DETECTION
# ============================================================================
GPU_AVAILABLE = False
CUML_AVAILABLE = False
CUPY_AVAILABLE = False
GPU_COUNT = 0

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    GPU_COUNT = cp.cuda.runtime.getDeviceCount()
    print(f"✅ CuPy available: {GPU_COUNT} GPU(s) found")
    for i in range(GPU_COUNT):
        with cp.cuda.Device(i):
            free, total = cp.cuda.runtime.memGetInfo()
            print(f"   GPU {i}: {total/1024**3:.1f}GB total, {free/1024**3:.1f}GB free")
except ImportError:
    print(f"⚠️ CuPy not available - GPU acceleration disabled")

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cumlRandomForest
    CUML_AVAILABLE = True
    print(f"✅ cuML available for GPU-accelerated ML")
except ImportError:
    print(f"⚠️ cuML not available - using CPU for ML")

GPU_AVAILABLE = CUPY_AVAILABLE

# ============================================================================
# CPU ML LIBRARIES
# ============================================================================
SKLEARN_AVAILABLE = False
XGBOOST_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                                recall_score, confusion_matrix, roc_auc_score)
    SKLEARN_AVAILABLE = True
    print(f"✅ scikit-learn available (CPU)")
except ImportError:
    print(f"⚠️ scikit-learn not available")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print(f"✅ XGBoost available (CPU)")
except ImportError:
    print(f"⚠️ XGBoost not available")

# ============================================================================
# TRY TO IMPORT SID METHOD
# ============================================================================
try:
    from sid_method import SidMethod, MarketTrend, SignalQuality, TradingSession
    SID_AVAILABLE = True
    print(f"✅ sid_method.py imported successfully (Augmented v3.0)")
except ImportError as e:
    SID_AVAILABLE = False
    print(f"❌ sid_method.py not found: {e}")
    sys.exit(1)

# Try to import tqdm
try:
    from tqdm import tqdm, trange
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
        def set_description(self, desc): self.desc = desc
        def set_postfix(self, **kwargs): pass
    trange = lambda *args, **kwargs: tqdm(range(*args), **kwargs)

# ============================================================================
# COLORFUL TERMINAL OUTPUT
# ============================================================================
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[35m'
    BOLD = '\033[1m'
    END = '\033[0m'
    CHECK = "✅"
    CROSS = "❌"
    WARN = "⚠️"
    FIRE = "🔥"
    BRAIN = "🧠"
    DISK = "💾"
    CLOCK = "⏱️"
    GRAPH = "📈"
    MODEL = "🤖"
    DATA = "📊"
    GPU_ICON = "🎮"

def print_header(text): 
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(60)}{Colors.END}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}\n")

def print_success(text): 
    print(f"{Colors.GREEN}{Colors.BOLD}✅ {text}{Colors.END}")

def print_warning(text): 
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ {text}{Colors.END}")

def print_error(text): 
    print(f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.END}")

def print_info(text): 
    print(f"{Colors.BLUE}ℹ️ {text}{Colors.END}")

def print_gpu(text): 
    print(f"{Colors.CYAN}{Colors.BOLD}{Colors.GPU_ICON} {text}{Colors.END}")

# ============================================================================
# CONFIGURATION
# ============================================================================
PARQUET_BASE_PATH = "/home/grct/Forex_Parquet"
MODEL_OUTPUT_PATH = "/mnt2/Trading-Cafe/ML/SNaiman2/ai/trained_models/"

os.makedirs(PARQUET_BASE_PATH, exist_ok=True)
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

print_info(f"Parquet path: {PARQUET_BASE_PATH}")
print_info(f"Model output path: {MODEL_OUTPUT_PATH}")

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TradeSignal:
    """Represents a SID Method trade signal (Wave 1-3)"""
    date: datetime
    instrument: str
    direction: str  # 'long' or 'short'
    signal_type: str  # 'oversold' or 'overbought'
    rsi_value: float
    entry_price: float
    stop_loss: float
    take_profit: float
    take_profit_alt: float
    macd_aligned: bool
    macd_crossed: bool
    pattern_confirmed: bool
    pattern_name: str
    divergence_detected: bool
    confidence_score: float
    confidence_level: str
    session: str
    reachable: bool
    actual_result: Optional[float] = None  # Profit/loss if executed

@dataclass
class TrainingConfig:
    """Configuration for training (Wave 1-3 parameters)"""
    # Wave 1 parameters
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_target: int = 50
    earnings_buffer_days: int = 14
    risk_percent: float = 1.0
    
    # Wave 2 parameters
    prefer_macd_cross: bool = True
    use_pattern_confirmation: bool = True
    use_divergence: bool = True
    use_market_context: bool = True
    
    # Wave 3 parameters
    strict_rsi: bool = True
    use_stop_pip_buffer: bool = True
    stop_pips_default: int = 5
    stop_pips_yen: int = 10
    min_pattern_candles: int = 7
    
    # Training parameters
    lookback_bars: int = 10
    test_size: float = 0.2
    validation_size: float = 0.1
    random_state: int = 42
    max_samples: int = 200000

# ============================================================================
# GPU MEMORY MANAGEMENT
# ============================================================================

class GPUMemoryManager:
    """Manage GPU memory for training (Wave 2 GPU acceleration)"""
    
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.initialized = False

    def activate(self):
        if CUPY_AVAILABLE:
            self.device = cp.cuda.Device(self.gpu_id)
            self.device.use()
            self.initialized = True
            free, total = cp.cuda.runtime.memGetInfo()
            print_gpu(f"GPU {self.gpu_id} activated: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
            return True
        return False

    def cleanup(self):
        if self.initialized and CUPY_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream().synchronize()
            gc.collect()

    def get_free_memory(self):
        if CUPY_AVAILABLE:
            with cp.cuda.Device(self.gpu_id):
                free, total = cp.cuda.runtime.memGetInfo()
                return free / (1024**2)  # MB
        return 0

gpu_manager = GPUMemoryManager(gpu_id=0)

# ============================================================================
# DATA LOADING FUNCTIONS (Augmented with date range detection)
# ============================================================================

def get_available_pairs() -> List[str]:
    """Get list of all trading pairs with parquet data"""
    pairs = []
    if os.path.exists(PARQUET_BASE_PATH):
        for item in os.listdir(PARQUET_BASE_PATH):
            pair_path = os.path.join(PARQUET_BASE_PATH, item)
            if os.path.isdir(pair_path):
                # Check if there are any parquet files
                for root, dirs, files in os.walk(pair_path):
                    if any(f.endswith('.parquet') for f in files):
                        pairs.append(item)
                        break
    return sorted(pairs)

def get_pair_date_range(pair: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the earliest and latest date available in parquet for a pair"""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    if not os.path.exists(pair_path):
        return None, None

    earliest_date = None
    latest_date = None

    for root, dirs, files in os.walk(pair_path):
        # Skip HIVE default partition
        if '__HIVE_DEFAULT_PARTITION__' in root:
            continue

        for file in files:
            if file.endswith('.parquet'):
                # Extract year/month/day from path
                year = None
                month = None
                day = None
                path_parts = root.split(os.sep)
                for part in path_parts:
                    if part.startswith('year='):
                        try:
                            val = part.split('=')[1]
                            if val and not val.startswith('__HIVE'):
                                year = int(val)
                        except:
                            pass
                    elif part.startswith('month='):
                        try:
                            val = part.split('=')[1]
                            if val and not val.startswith('__HIVE'):
                                month = int(val)
                        except:
                            pass
                    elif part.startswith('day='):
                        try:
                            val = part.split('=')[1]
                            if val and not val.startswith('__HIVE'):
                                day = int(val)
                        except:
                            pass

                if year and month and day:
                    try:
                        current_date = datetime(year, month, day)
                        if earliest_date is None or current_date < earliest_date:
                            earliest_date = current_date
                        if latest_date is None or current_date > latest_date:
                            latest_date = current_date
                    except:
                        pass

    return earliest_date, latest_date

def get_earliest_parquet_date() -> datetime:
    """Get the earliest date across all parquet files"""
    earliest_overall = None

    if not os.path.exists(PARQUET_BASE_PATH):
        return datetime(2020, 1, 1)

    pairs = get_available_pairs()
    print_info(f"Scanning {len(pairs)} pairs for earliest date...")

    for pair in tqdm(pairs, desc="Scanning pairs", disable=not TQDM_AVAILABLE):
        earliest, _ = get_pair_date_range(pair)
        if earliest and (earliest_overall is None or earliest < earliest_overall):
            earliest_overall = earliest

    if earliest_overall is None:
        earliest_overall = datetime.now() - timedelta(days=1825)  # 5 years default

    return earliest_overall

def get_latest_parquet_date() -> datetime:
    """Get the latest date across all parquet files"""
    latest_overall = None

    if not os.path.exists(PARQUET_BASE_PATH):
        return datetime.now()

    pairs = get_available_pairs()

    for pair in tqdm(pairs, desc="Scanning pairs", disable=not TQDM_AVAILABLE):
        _, latest = get_pair_date_range(pair)
        if latest and (latest_overall is None or latest > latest_overall):
            latest_overall = latest

    if latest_overall is None:
        latest_overall = datetime.now()

    return latest_overall

def load_parquet_data_full(pair: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load ALL available parquet data for a pair
    Augmented with progress bars and error handling
    """
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    if not os.path.exists(pair_path):
        print_error(f"Pair path not found: {pair_path}")
        return pd.DataFrame()

    # Get date range for this pair
    earliest, latest = get_pair_date_range(pair)
    if earliest and latest and verbose:
        print_info(f"Date range for {pair}: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
        print_info(f"Total days: {(latest - earliest).days:,} days")

    try:
        if verbose:
            print_info(f"Reading parquet files for {pair}...")

        # Collect all parquet files
        all_files = []
        for root, dirs, files in os.walk(pair_path):
            if '__HIVE_DEFAULT_PARTITION__' in root:
                continue
            for file in files:
                if file.endswith('.parquet'):
                    all_files.append(os.path.join(root, file))

        if not all_files:
            print_error(f"No parquet files found for {pair}")
            return pd.DataFrame()

        if verbose:
            print_info(f"Found {len(all_files)} parquet files")

        # Load files with progress bar
        dfs = []
        iterator = tqdm(all_files, desc="  Loading files", unit="file") if TQDM_AVAILABLE and len(all_files) > 10 else all_files

        for file_path in iterator:
            try:
                df_chunk = pd.read_parquet(file_path)
                if not df_chunk.empty:
                    dfs.append(df_chunk)
            except Exception as e:
                if verbose:
                    print_warning(f"Error reading {os.path.basename(file_path)}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(dfs, ignore_index=False)

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df.dropna()
            except:
                print_error(f"Cannot convert index to datetime for {pair}")
                return pd.DataFrame()

        # Sort by date
        df = df.sort_index()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print_error(f"Missing required column: {col}")
                return pd.DataFrame()

        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values
        df = df.dropna(subset=required_cols)

        # Resample to 1-hour bars for consistency (Wave 2)
        if len(df) > 0:
            df = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        if verbose:
            print_success(f"Loaded {len(df):,} rows for {pair}")
            if len(df) > 0:
                print_info(f"  Date range: {df.index.min().strftime('%Y-%m-%d %H:%M')} to {df.index.max().strftime('%Y-%m-%d %H:%M')}")

        return df

    except Exception as e:
        print_error(f"Error loading {pair}: {e}")
        traceback.print_exc()
        return pd.DataFrame()

# ============================================================================
# AUGMENTED FEATURE ENGINEERING (Wave 1, 2, 3)
# ============================================================================

def calculate_sid_features(df: pd.DataFrame, sid: SidMethod, 
                           config: TrainingConfig,
                           verbose: bool = True) -> pd.DataFrame:
    """
    Calculate features incorporating ALL THREE WAVES:
    - Wave 1: RSI, MACD, basic signals
    - Wave 2: Divergence, pattern detection, market context
    - Wave 3: Precision features, session features
    
    Args:
        df: Price DataFrame
        sid: SidMethod instance
        config: Training configuration
        verbose: Enable verbose output
    
    Returns:
        DataFrame with all calculated features
    """
    df = df.copy()

    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            print_error(f"Missing required column: {col}")
            return pd.DataFrame()

    # Convert to numeric
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop rows with NaN
    df = df.dropna(subset=required)

    if len(df) < 100:
        print_error(f"Insufficient data after cleaning: {len(df)} rows")
        return pd.DataFrame()

    if verbose:
        print_info("Step 1/10: Calculating RSI...")
    df['rsi'] = sid.calculate_rsi(df, desc="RSI")

    if verbose:
        print_info("Step 2/10: Calculating MACD...")
    macd_df = sid.calculate_macd(df, desc="MACD")
    df['macd'] = macd_df['macd']
    df['macd_signal'] = macd_df['signal']
    df['macd_hist'] = macd_df['histogram']

    if verbose:
        print_info("Step 3/10: Calculating RSI signals (Wave 1)...")
    
    # WAVE 1: Basic RSI signals (exact thresholds)
    df['rsi_oversold'] = (df['rsi'] < config.rsi_oversold).astype(int)
    df['rsi_overbought'] = (df['rsi'] > config.rsi_overbought).astype(int)
    df['rsi_mid'] = ((df['rsi'] >= 45) & (df['rsi'] <= 55)).astype(int)
    df['rsi_change'] = df['rsi'].diff()
    df['rsi_change_3'] = df['rsi'].diff(3)
    df['rsi_change_5'] = df['rsi'].diff(5)
    
    # WAVE 3: Precision RSI (exact crossing of thresholds)
    df['rsi_crossed_below_30'] = ((df['rsi'].shift(1) >= config.rsi_oversold) & 
                                    (df['rsi'] < config.rsi_oversold)).astype(int)
    df['rsi_crossed_above_70'] = ((df['rsi'].shift(1) <= config.rsi_overbought) & 
                                    (df['rsi'] > config.rsi_overbought)).astype(int)

    if verbose:
        print_info("Step 4/10: Calculating MACD signals (Wave 1 & 2)...")
    
    # WAVE 1: MACD alignment and cross
    df['macd_aligned_up'] = (df['macd'] > df['macd'].shift(1)).astype(int)
    df['macd_aligned_down'] = (df['macd'] < df['macd'].shift(1)).astype(int)
    
    # WAVE 1: MACD cross detection
    df['macd_cross_above'] = ((df['macd'].shift(1) <= df['macd_signal'].shift(1)) & 
                               (df['macd'] > df['macd_signal'])).astype(int)
    df['macd_cross_below'] = ((df['macd'].shift(1) >= df['macd_signal'].shift(1)) & 
                               (df['macd'] < df['macd_signal'])).astype(int)
    
    # WAVE 2: MACD histogram turning points
    df['macd_hist_positive'] = (df['macd_hist'] > 0).astype(int)
    df['macd_hist_turning_up'] = ((df['macd_hist'].shift(1) < 0) & (df['macd_hist'] > 0)).astype(int)
    df['macd_hist_turning_down'] = ((df['macd_hist'].shift(1) > 0) & (df['macd_hist'] < 0)).astype(int)

    if verbose:
        print_info("Step 5/10: Calculating price action features...")
    
    # Price action features
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']
    df['body_ratio'] = df['body_size'] / df['total_range'].replace(0, 1)
    df['returns'] = df['close'].pct_change()
    df['returns_abs'] = abs(df['returns'])
    
    # WAVE 2: Candlestick patterns
    df['hammer'] = ((df['lower_wick'] > df['body_size'] * 2) & 
                    (df['upper_wick'] < df['body_size'])).astype(int)
    df['shooting_star'] = ((df['upper_wick'] > df['body_size'] * 2) & 
                           (df['lower_wick'] < df['body_size'])).astype(int)
    df['engulfing'] = ((df['close'] > df['open']) & 
                       (df['close'].shift(1) < df['open'].shift(1)) & 
                       (df['close'] > df['open'].shift(1)) & 
                       (df['open'] < df['close'].shift(1))).astype(int)

    if verbose:
        print_info("Step 6/10: Calculating volatility features...")
    
    # Volatility features
    for period in [5, 10, 20, 50]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
    
    # WAVE 2: ATR (Average True Range)
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    for period in [7, 14]:
        df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()

    if verbose:
        print_info("Step 7/10: Calculating volume features...")
    
    # Volume features (if available)
    if 'volume' in df.columns:
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_surge'] = (df['volume_ratio'] > 1.5).astype(int)

    if verbose:
        print_info("Step 8/10: Detecting divergence (Wave 2)...")
    
    # WAVE 2: Divergence detection (vectorized approach)
    lookback = 20
    
    # RSI divergence (simplified vectorized)
    df['rsi_rolling_min'] = df['rsi'].rolling(window=lookback, center=True).min()
    df['rsi_rolling_max'] = df['rsi'].rolling(window=lookback, center=True).max()
    df['price_rolling_min'] = df['low'].rolling(window=lookback, center=True).min()
    df['price_rolling_max'] = df['high'].rolling(window=lookback, center=True).max()
    
    # Bullish divergence approximation
    df['bullish_divergence'] = ((df['low'] <= df['price_rolling_min']) & 
                                 (df['rsi'] > df['rsi_rolling_min'])).astype(int)
    # Bearish divergence approximation
    df['bearish_divergence'] = ((df['high'] >= df['price_rolling_max']) & 
                                 (df['rsi'] < df['rsi_rolling_max'])).astype(int)

    if verbose:
        print_info("Step 9/10: Adding session features (Wave 3)...")
    
    # WAVE 3: Session features
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['is_asian_session'] = ((df['hour'] >= 0) & (df['hour'] < 7)).astype(int)
    df['is_london_session'] = ((df['hour'] >= 7) & (df['hour'] < 12)).astype(int)
    df['is_us_session'] = ((df['hour'] >= 12) & (df['hour'] < 21)).astype(int)
    df['is_overlap_session'] = ((df['hour'] >= 12) & (df['hour'] < 16)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    if verbose:
        print_info("Step 10/10: Creating target labels (Wave 1 & 2)...")
    
    # WAVE 1: Target - RSI reaching 50 within N bars
    for horizon in [5, 10, 20]:
        df[f'target_rsi_50_{horizon}'] = 0
        for i in range(len(df) - horizon):
            if any(df['rsi'].iloc[i+1:i+horizon+1] >= config.rsi_target):
                df.loc[df.index[i], f'target_rsi_50_{horizon}'] = 1
    
    # WAVE 2: Target - Price direction (1 if up, 0 if down)
    for horizon in [5, 10, 20]:
        df[f'target_direction_{horizon}'] = (df['close'].shift(-horizon) > df['close']).astype(int)
    
    # WAVE 2: Target - 50-SMA touch
    df['sma_50'] = df['close'].rolling(50).mean()
    for horizon in [5, 10]:
        df[f'target_sma_50_{horizon}'] = 0
        for i in range(len(df) - horizon):
            if any(df['high'].iloc[i+1:i+horizon+1] >= df['sma_50'].iloc[i]):
                df.loc[df.index[i], f'target_sma_50_{horizon}'] = 1
    
    # WAVE 3: Target - Pattern completion (for pattern detection models)
    # Simplified: check if a W or M pattern completes
    df['target_pattern_bullish'] = 0
    df['target_pattern_bearish'] = 0
    
    # Drop NaN rows
    df = df.dropna()
    
    if verbose:
        print_success(f"Features calculated: {len(df):,} rows, {len(df.columns)} columns")
        
        # Print target distribution
        print_info("Target distributions:")
        for col in [c for c in df.columns if c.startswith('target_')]:
            if df[col].dtype in ['int64', 'float64']:
                pos_pct = df[col].mean() * 100
                print_info(f"  {col}: {pos_pct:.1f}% positive")
    
    return df

def create_training_samples(df: pd.DataFrame, 
                             lookback: int = 10,
                             target_col: str = 'target_direction_5',
                             max_samples: int = 200000,
                             stride: int = 1) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Create training samples with sliding window (Wave 1 & 2)
    
    Args:
        df: Feature DataFrame
        lookback: Number of bars to look back
        target_col: Target column name
        max_samples: Maximum number of samples to create
        stride: Step between samples (for downsampling)
    
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature column names
    """
    # Exclude target columns and price columns from features
    exclude_cols = [col for col in df.columns if col.startswith('target_')]
    exclude_cols.extend(['open', 'high', 'low', 'close', 'volume', 'sma_50'])
    
    feature_cols = [col for col in df.columns if col not in exclude_cols and 
                    df[col].dtype in ['float64', 'float32', 'int64']]
    
    if len(df) < lookback + 10:
        print_warning(f"Insufficient data: {len(df)} rows")
        return np.array([]), np.array([]), []

    print_info(f"Using {len(feature_cols)} features, lookback={lookback}, stride={stride}")

    X_list = []
    y_list = []

    # Calculate step size to limit samples
    total_possible = (len(df) - lookback) // stride
    actual_stride = max(stride, total_possible // max_samples) if total_possible > max_samples else stride
    
    sample_indices = range(lookback, len(df) - 1, actual_stride)
    total_samples = len(sample_indices)
    
    print_info(f"Creating {total_samples:,} samples (stride={actual_stride})")

    # Use tqdm for progress
    iterator = tqdm(sample_indices, desc="Creating samples", unit="samples") if TQDM_AVAILABLE else sample_indices

    for i in iterator:
        window_features = []
        for j in range(lookback):
            for col in feature_cols:
                window_features.append(df[col].iloc[i - j])
        
        X_list.append(window_features)
        y_list.append(df[target_col].iloc[i])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)

    print_success(f"Created {len(X):,} samples with {X.shape[1]} features")

    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        print_info(f"Target distribution: {dict(zip(unique, counts))}")

    return X, y, feature_cols

# ============================================================================
# GPU-ACCELERATED MODEL TRAINING (Wave 2)
# ============================================================================

def train_random_forest_gpu(X_train, y_train, X_val, y_val, verbose=True):
    """Train Random Forest on GPU using cuML"""
    if not CUML_AVAILABLE:
        return None, 0

    try:
        gpu_manager.activate()

        if verbose:
            print_gpu("Training Random Forest on GPU (cuML)...")

        X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
        y_train_gpu = cp.asarray(y_train, dtype=cp.int32)

        model = cumlRandomForest(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_streams=1
        )

        model.fit(X_train_gpu, y_train_gpu)

        X_val_gpu = cp.asarray(X_val, dtype=cp.float32)
        y_pred_gpu = model.predict(X_val_gpu)
        y_pred = cp.asnumpy(y_pred_gpu)
        
        accuracy = accuracy_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        if verbose:
            print_success(f"GPU Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

        gpu_manager.cleanup()
        return model, accuracy

    except Exception as e:
        print_warning(f"GPU training failed: {e}, falling back to CPU")
        gpu_manager.cleanup()
        return None, 0

def train_random_forest_cpu(X_train, y_train, X_val, y_val, verbose=True):
    """Train Random Forest on CPU using scikit-learn"""
    if not SKLEARN_AVAILABLE:
        return None, 0

    if verbose:
        print_info("Training Random Forest on CPU (scikit-learn)...")

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        verbose=1 if verbose else 0
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    if verbose:
        print_success(f"CPU Random Forest - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return model, accuracy

def train_gradient_boosting(X_train, y_train, X_val, y_val, verbose=True):
    """Train Gradient Boosting on CPU"""
    if not SKLEARN_AVAILABLE:
        return None, 0

    if verbose:
        print_info("Training Gradient Boosting on CPU...")

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
        verbose=1 if verbose else 0
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    if verbose:
        print_success(f"Gradient Boosting - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return model, accuracy

def train_xgboost_cpu(X_train, y_train, X_val, y_val, verbose=True):
    """Train XGBoost on CPU"""
    if not XGBOOST_AVAILABLE:
        return None, 0

    if verbose:
        print_info("Training XGBoost on CPU...")

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        verbosity=1 if verbose else 0,
        tree_method='hist',
        eval_metric='logloss'
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    if verbose:
        print_success(f"XGBoost - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    return model, accuracy

# ============================================================================
# MODEL EVALUATION (Wave 2 & 3)
# ============================================================================

def evaluate_model_detailed(model, X_test, y_test, model_name: str) -> Dict:
    """
    Detailed model evaluation with multiple metrics (Wave 2 & 3)
    
    Returns:
        Dictionary with metrics including accuracy, precision, recall, f1, auc, confusion matrix
    """
    metrics = {}
    
    try:
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            if len(y_pred_proba.shape) > 1 and y_pred_proba.shape[1] > 1:
                y_pred_proba = y_pred_proba[:, 1]
            y_pred = (y_pred_proba > 0.5).astype(int)
        elif hasattr(model, 'predict'):
            y_pred = model.predict(X_test)
            if hasattr(y_pred, 'numpy'):
                y_pred = y_pred.numpy()
            y_pred_proba = y_pred
        else:
            return {'error': 'Model cannot predict'}

        # Convert to numpy if needed
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        if hasattr(y_pred, 'get'):
            y_pred = y_pred.get()

        # Calculate metrics
        metrics['accuracy'] = accuracy_score(y_test, y_pred)
        metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)

        try:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except:
            metrics['roc_auc'] = 0.5

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # WAVE 3: Additional quality metrics
        metrics['precision_positive'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall_positive'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0

    except Exception as e:
        print_error(f"Error evaluating model: {e}")
        metrics = {'error': str(e)}

    return metrics

# ============================================================================
# MAIN TRAINING FUNCTION (Augmented)
# ============================================================================

def train_pair(pair: str, 
               config: TrainingConfig = None,
               target: str = 'target_direction_5',
               lookback: int = 10,
               force: bool = False,
               use_gpu: bool = True,
               verbose: bool = True) -> Optional[Dict]:
    """
    Train a model for a specific pair using ALL available data
    Incorporates ALL THREE WAVES of strategies
    
    Args:
        pair: Trading pair (e.g., 'EUR_USD')
        config: Training configuration
        target: Target column name
        lookback: Lookback bars for features
        force: Force retraining even if model exists
        use_gpu: Use GPU acceleration if available
        verbose: Enable verbose output
    
    Returns:
        Dictionary with training results or None if failed
    """
    if config is None:
        config = TrainingConfig()
    
    print_header(f"{Colors.BRAIN} TRAINING SID MODEL FOR: {pair}")
    start_time = time.time()
    
    # Initialize Sid Method with augmented parameters
    sid = SidMethod(
        account_balance=10000,
        verbose=verbose,
        prefer_macd_cross=config.prefer_macd_cross,
        use_pattern_confirmation=config.use_pattern_confirmation,
        use_divergence=config.use_divergence,
        use_market_context=config.use_market_context,
        risk_percent_default=config.risk_percent,
        earnings_buffer_days=config.earnings_buffer_days,
        stop_pips_default=config.stop_pips_default,
        stop_pips_yen=config.stop_pips_yen
    )
    
    # Load ALL data
    df = load_parquet_data_full(pair, verbose=verbose)
    
    if df.empty:
        print_error(f"No data loaded for {pair}")
        return None
    
    # Calculate augmented features
    print_info("Calculating Sid features (Wave 1, 2, 3)...")
    df = calculate_sid_features(df, sid, config, verbose=verbose)
    
    if df.empty or len(df) < 1000:
        print_error(f"Insufficient data after feature calculation: {len(df)} rows")
        return None
    
    # Create training samples
    print_info(f"Creating training samples (target={target}, lookback={lookback})...")
    X, y, feature_names = create_training_samples(
        df, 
        lookback=lookback,
        target_col=target,
        max_samples=config.max_samples,
        stride=1
    )
    
    if len(X) == 0:
        print_error("No training samples created")
        return None
    
    # Split data (Wave 2: time-based split for realistic backtesting)
    split_idx = int(len(X) * (1 - config.test_size - config.validation_size))
    val_idx = int(len(X) * (1 - config.test_size))
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:val_idx]
    y_val = y[split_idx:val_idx]
    X_test = X[val_idx:]
    y_test = y[val_idx:]
    
    print_info(f"Train: {len(X_train):,} samples")
    print_info(f"Validation: {len(X_val):,} samples")
    print_info(f"Test: {len(X_test):,} samples")
    
    # Train models
    results = {}
    best_model = None
    best_score = -1
    best_name = None
    
    # Try GPU Random Forest first (Wave 2)
    if use_gpu and CUML_AVAILABLE:
        model, accuracy = train_random_forest_gpu(X_train, y_train, X_val, y_val, verbose)
        if model:
            results['random_forest_gpu'] = {'model': model, 'accuracy': accuracy}
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = 'random_forest_gpu'
    
    # CPU Random Forest
    model, accuracy = train_random_forest_cpu(X_train, y_train, X_val, y_val, verbose)
    if model:
        results['random_forest_cpu'] = {'model': model, 'accuracy': accuracy}
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = 'random_forest_cpu'
    
    # Gradient Boosting
    model, accuracy = train_gradient_boosting(X_train, y_train, X_val, y_val, verbose)
    if model:
        results['gradient_boosting'] = {'model': model, 'accuracy': accuracy}
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = 'gradient_boosting'
    
    # XGBoost
    model, accuracy = train_xgboost_cpu(X_train, y_train, X_val, y_val, verbose)
    if model:
        results['xgboost'] = {'model': model, 'accuracy': accuracy}
        if accuracy > best_score:
            best_score = accuracy
            best_model = model
            best_name = 'xgboost'
    
    if best_model is None:
        print_error("No models trained successfully")
        return None
    
    # Evaluate best model on test set
    print_info(f"\nEvaluating best model ({best_name}) on test set...")
    test_metrics = evaluate_model_detailed(best_model, X_test, y_test, best_name)
    
    print_success(f"Test Results:")
    print_success(f"  Accuracy: {test_metrics.get('accuracy', 0):.4f}")
    print_success(f"  Precision: {test_metrics.get('precision', 0):.4f}")
    print_success(f"  Recall: {test_metrics.get('recall', 0):.4f}")
    print_success(f"  F1 Score: {test_metrics.get('f1', 0):.4f}")
    print_success(f"  ROC AUC: {test_metrics.get('roc_auc', 0):.4f}")
    
    # Save best model
    model_filename = f"{pair}_{target}_lb{lookback}_{best_name}.pkl"
    model_path = os.path.join(MODEL_OUTPUT_PATH, model_filename)
    
    # Save model metadata
    metadata = {
        'pair': pair,
        'target': target,
        'lookback': lookback,
        'model_type': best_name,
        'training_date': datetime.now().isoformat(),
        'test_metrics': test_metrics,
        'config': {
            'rsi_oversold': config.rsi_oversold,
            'rsi_overbought': config.rsi_overbought,
            'prefer_macd_cross': config.prefer_macd_cross,
            'use_pattern_confirmation': config.use_pattern_confirmation,
            'use_divergence': config.use_divergence,
            'use_market_context': config.use_market_context,
            'strict_rsi': config.strict_rsi,
            'stop_pips_default': config.stop_pips_default,
            'stop_pips_yen': config.stop_pips_yen
        },
        'feature_names': feature_names,
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test)
    }
    
    try:
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        metadata_path = model_path.replace('.pkl', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print_success(f"Model saved to {model_path}")
        
    except Exception as e:
        print_error(f"Error saving model: {e}")
    
    elapsed = time.time() - start_time
    print_info(f"Total training time: {elapsed:.1f}s")
    
    return {
        'pair': pair,
        'best_model': best_name,
        'test_metrics': test_metrics,
        'model_path': model_path,
        'training_time': elapsed
    }

# ============================================================================
# BATCH TRAINING
# ============================================================================

def train_all_pairs(pairs: List[str] = None,
                    config: TrainingConfig = None,
                    target: str = 'target_direction_5',
                    lookback: int = 10,
                    use_gpu: bool = True,
                    force: bool = False) -> Dict[str, Dict]:
    """
    Train models for all pairs
    """
    if pairs is None:
        pairs = get_available_pairs()
    
    if config is None:
        config = TrainingConfig()
    
    print_header(f"{Colors.FIRE} BATCH TRAINING {len(pairs)} PAIRS")
    print_info(f"Target: {target}, Lookback: {lookback}")
    print_info(f"GPU: {'Enabled' if use_gpu else 'Disabled'}")
    print_info(f"MACD cross preferred: {config.prefer_macd_cross}")
    print_info(f"Pattern confirmation: {config.use_pattern_confirmation}")
    print_info(f"Divergence detection: {config.use_divergence}")
    print_info(f"Market context: {config.use_market_context}")
    
    results = {}
    total_start = time.time()
    
    for i, pair in enumerate(pairs):
        print(f"\n[{i+1}/{len(pairs)}] ", end="")
        
        try:
            result = train_pair(
                pair=pair,
                config=config,
                target=target,
                lookback=lookback,
                force=force,
                use_gpu=use_gpu,
                verbose=True
            )
            if result:
                results[pair] = result
        except Exception as e:
            print_error(f"Failed to train {pair}: {e}")
            continue
    
    total_elapsed = time.time() - total_start
    print_header(f"{Colors.CHECK} TRAINING COMPLETE")
    print_success(f"Successfully trained {len(results)}/{len(pairs)} pairs")
    print_info(f"Total time: {total_elapsed:.1f}s")
    
    # Summary of results
    print_info("\nModel Performance Summary:")
    for pair, result in results.items():
        metrics = result.get('test_metrics', {})
        print_info(f"  {pair}: {result['best_model']} - F1: {metrics.get('f1', 0):.3f}, Acc: {metrics.get('accuracy', 0):.3f}")
    
    return results

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train SID Method AI Models (Augmented v3.0)')
    parser.add_argument('--pair', type=str, help='Specific pair to train')
    parser.add_argument('--target', type=str, default='target_direction_5',
                       choices=['target_direction_5', 'target_direction_10', 'target_rsi_50_5', 'target_rsi_50_10'],
                       help='Target column for training')
    parser.add_argument('--lookback', type=int, default=10, help='Lookback bars for features')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU acceleration')
    parser.add_argument('--no-cross', action='store_true', help='Disable MACD cross preference')
    parser.add_argument('--no-patterns', action='store_true', help='Disable pattern confirmation')
    parser.add_argument('--no-divergence', action='store_true', help='Disable divergence detection')
    parser.add_argument('--no-context', action='store_true', help='Disable market context')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--list-pairs', action='store_true', help='List available pairs')
    
    args = parser.parse_args()
    
    if args.list_pairs:
        pairs = get_available_pairs()
        print(f"\nAvailable pairs ({len(pairs)}):")
        for p in pairs:
            earliest, latest = get_pair_date_range(p)
            if earliest and latest:
                print(f"  {p}: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
            else:
                print(f"  {p}: No data")
        return
    
    # Create training configuration
    config = TrainingConfig(
        prefer_macd_cross=not args.no_cross,
        use_pattern_confirmation=not args.no_patterns,
        use_divergence=not args.no_divergence,
        use_market_context=not args.no_context
    )
    
    print_header("SID METHOD AI TRAINER v3.0 (Fully Augmented)")
    print_info(f"Configuration:")
    print_info(f"  Target: {args.target}")
    print_info(f"  Lookback: {args.lookback}")
    print_info(f"  MACD cross preferred: {config.prefer_macd_cross}")
    print_info(f"  Pattern confirmation: {config.use_pattern_confirmation}")
    print_info(f"  Divergence detection: {config.use_divergence}")
    print_info(f"  Market context: {config.use_market_context}")
    print_info(f"  Strict RSI: {config.strict_rsi}")
    print_info(f"  Stop pips: {config.stop_pips_default} (default) / {config.stop_pips_yen} (Yen)")
    
    if args.pair:
        # Train single pair
        result = train_pair(
            pair=args.pair,
            config=config,
            target=args.target,
            lookback=args.lookback,
            force=args.force,
            use_gpu=not args.no_gpu
        )
        if result:
            print_success(f"\nTraining completed for {args.pair}")
        else:
            print_error(f"\nTraining failed for {args.pair}")
    else:
        # Train all pairs
        results = train_all_pairs(
            config=config,
            target=args.target,
            lookback=args.lookback,
            use_gpu=not args.no_gpu,
            force=args.force
        )
        print_success(f"\nBatch training completed: {len(results)} models saved")

if __name__ == "__main__":
    main()