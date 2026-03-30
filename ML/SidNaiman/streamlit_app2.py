"""
Streamlit App for Sid Naiman's SID Method Trading System
Complete AI-Augmented Trading Interface with GPU/CPU Support
Following Sid's SID Method rules with ML enhancement
Version: 6.0 - DYNAMIC FILE SELECTION + OANDA API KEY MANAGEMENT
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import logging
import sys
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import json
import warnings
import glob
from pathlib import Path
import pickle
import base64

# Suppress all warnings
warnings.filterwarnings('ignore')

# Set TensorFlow logging to minimum
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Import core modules with error handling
try:
    from oanda_client import OANDAClient
    from oanda_trader import OANDATrader
    from supply_demand import SupplyDemand
    from sid_method import SidMethod
    CORE_IMPORTS_SUCCESS = True
except ImportError as e:
    st.error(f"Failed to import core modules: {e}")
    CORE_IMPORTS_SUCCESS = False

# ============================================================================
# AI Module Imports with graceful fallback
# ============================================================================
try:
    from ai.ai_accelerator import AIAccelerator
    from ai.feature_engineering import FeatureEngineer
    from ai.signal_predictor import SignalPredictor
    from ai.model_trainer import ModelTrainer
    from ai.model_manager import ModelManager
    from ai.training_pipeline import TrainingPipeline
    try:
        from ai.parquet_converter import ParquetConverter
        PARQUET_AVAILABLE = True
    except ImportError:
        PARQUET_AVAILABLE = False
    AI_AVAILABLE = True
except ImportError as e:
    AI_AVAILABLE = False
    PARQUET_AVAILABLE = False
    print(f"AI modules not available: {e}")

# ============================================================================
# DEFAULT PATHS (will be overridden by user input)
# ============================================================================
DEFAULT_CSV_PATH = "/home/grct/Forex"
DEFAULT_PARQUET_PATH = "/home/grct/Forex_Parquet"
DEFAULT_MODEL_PATH = "/mnt2/Trading-Cafe/ML/SNaiman2/ai/trained_models/"

# ============================================================================
# VALID OANDA PAIRS (Verified working pairs)
# ============================================================================
VALID_OANDA_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "EUR_CHF", "AUD_NZD",
    "NZD_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD", "CAD_JPY", "CHF_JPY",
    "EUR_CAD", "AUD_CAD", "NZD_CAD", "EUR_NZD", "USD_NOK", "USD_SEK", "USD_TRY",
    "EUR_NOK", "EUR_SEK", "EUR_TRY", "GBP_NOK", "GBP_SEK", "GBP_TRY"
]

# ============================================================================
# OANDA API KEY MANAGEMENT
# ============================================================================

def save_api_key_to_env(environment: str, api_key: str):
    """Save API key to environment variable and .env file"""
    env_var = f"OANDA_API_KEY_{environment.upper()}"
    os.environ[env_var] = api_key
    
    # Also save to .env file for persistence
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    
    try:
        # Read existing .env file if it exists
        env_vars = {}
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key] = value
        
        # Update with new key
        env_vars[env_var] = api_key
        
        # Write back
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        return True
    except Exception as e:
        logger.warning(f"Could not save to .env file: {e}")
        return False


def load_api_key_from_env(environment: str) -> Optional[str]:
    """Load API key from environment variable"""
    env_var = f"OANDA_API_KEY_{environment.upper()}"
    return os.environ.get(env_var)


def test_oanda_connection(environment: str, api_key: str, account_id: str = None) -> Tuple[bool, str]:
    """Test OANDA API connection with provided credentials"""
    try:
        # Set temporary environment
        old_api_key = os.environ.get('OANDA_API_KEY')
        old_account_id = os.environ.get('OANDA_ACCOUNT_ID')
        
        os.environ['OANDA_API_KEY'] = api_key
        if account_id:
            os.environ['OANDA_ACCOUNT_ID'] = account_id
        
        client = OANDAClient()
        
        # Try to get account summary
        summary = client.get_account_summary()
        
        # Restore old values
        if old_api_key:
            os.environ['OANDA_API_KEY'] = old_api_key
        else:
            del os.environ['OANDA_API_KEY']
        if old_account_id:
            os.environ['OANDA_ACCOUNT_ID'] = old_account_id
        elif 'OANDA_ACCOUNT_ID' in os.environ:
            del os.environ['OANDA_ACCOUNT_ID']
        
        if summary and 'account' in summary:
            balance = summary['account'].get('balance', 'N/A')
            return True, f"Connected! Balance: ${balance}"
        else:
            return False, "Connection failed: No account data"
            
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


# ============================================================================
# DYNAMIC PATH MANAGEMENT FUNCTIONS
# ============================================================================

def get_user_paths():
    """Get file paths from session state or defaults"""
    return {
        'csv_path': st.session_state.get('user_csv_path', DEFAULT_CSV_PATH),
        'parquet_path': st.session_state.get('user_parquet_path', DEFAULT_PARQUET_PATH),
        'model_path': st.session_state.get('user_model_path', DEFAULT_MODEL_PATH)
    }


def set_user_paths(csv_path: str = None, parquet_path: str = None, model_path: str = None):
    """Set file paths in session state"""
    if csv_path:
        st.session_state.user_csv_path = csv_path
    if parquet_path:
        st.session_state.user_parquet_path = parquet_path
    if model_path:
        st.session_state.user_model_path = model_path


def validate_path(path: str, path_type: str = "directory") -> Tuple[bool, str]:
    """Validate that a path exists and is accessible"""
    if not path:
        return False, "Path is empty"
    
    try:
        if path_type == "directory":
            if os.path.isdir(path):
                return True, "Directory exists"
            else:
                return False, f"Directory not found: {path}"
        else:
            if os.path.isfile(path):
                return True, "File exists"
            else:
                return False, f"File not found: {path}"
    except Exception as e:
        return False, f"Error accessing path: {e}"


def scan_directory_for_files(directory: str, extensions: List[str]) -> List[str]:
    """Scan directory for files with given extensions"""
    if not directory or not os.path.exists(directory):
        return []
    
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    return sorted(files)


def get_available_pairs_with_parquet(parquet_path: str = None):
    """Get list of pairs that have Parquet files available in partitioned structure."""
    if parquet_path is None:
        parquet_path = st.session_state.get('user_parquet_path', DEFAULT_PARQUET_PATH)
    
    pairs_with_parquet = []
    
    if not os.path.exists(parquet_path):
        return pairs_with_parquet
    
    for item in os.listdir(parquet_path):
        pair_path = os.path.join(parquet_path, item)
        if os.path.isdir(pair_path):
            for subdir in os.listdir(pair_path):
                subdir_path = os.path.join(pair_path, subdir)
                if os.path.isdir(subdir_path) and subdir.startswith('year='):
                    pairs_with_parquet.append(item)
                    break
    
    return sorted(pairs_with_parquet)


def get_parquet_stats(parquet_path: str = None):
    """Get statistics about Parquet files in partitioned structure."""
    if parquet_path is None:
        parquet_path = st.session_state.get('user_parquet_path', DEFAULT_PARQUET_PATH)
    
    stats = {
        'total_pairs': 0,
        'total_files': 0,
        'total_size_gb': 0,
        'pairs': []
    }
    
    if os.path.exists(parquet_path):
        for pair in os.listdir(parquet_path):
            pair_path = os.path.join(parquet_path, pair)
            if os.path.isdir(pair_path):
                all_files = []
                years = []
                
                for root, dirs, files in os.walk(pair_path):
                    if os.path.basename(root).startswith('year='):
                        year = os.path.basename(root).replace('year=', '')
                        if year not in years:
                            years.append(year)
                    
                    for file in files:
                        if file.endswith('.parquet'):
                            all_files.append(os.path.join(root, file))
                
                if all_files:
                    pair_size = sum(os.path.getsize(f) for f in all_files) / (1024**3)
                    
                    stats['pairs'].append({
                        'pair': pair,
                        'files': len(all_files),
                        'size_gb': pair_size,
                        'years': sorted(years)
                    })
                    stats['total_pairs'] += 1
                    stats['total_files'] += len(all_files)
                    stats['total_size_gb'] += pair_size
    
    return stats


def get_available_models(model_path: str = None):
    """Get list of available trained models."""
    if model_path is None:
        model_path = st.session_state.get('user_model_path', DEFAULT_MODEL_PATH)
    
    if not os.path.exists(model_path):
        return []
    
    models = []
    for file in os.listdir(model_path):
        if file.endswith('.pkl') or file.endswith('.joblib'):
            models.append(file)
    return sorted(models)


def load_data_from_parquet(pair, parquet_path, start_date=None, end_date=None, progress_callback=None):
    """Load data from partitioned Parquet files."""
    import pandas as pd
    import pyarrow.parquet as pq
    
    pair_path = os.path.join(parquet_path, pair)
    
    if not os.path.exists(pair_path):
        if progress_callback:
            progress_callback(0, f"⚠️ Parquet path not found: {pair_path}")
        return pd.DataFrame()
    
    try:
        if progress_callback:
            progress_callback(0.1, "Reading Parquet dataset...")
        
        dataset = pq.ParquetDataset(pair_path, partitioning='hive')
        
        if progress_callback:
            progress_callback(0.3, "Reading data...")
        
        table = dataset.read()
        
        if progress_callback:
            progress_callback(0.8, "Converting to pandas...")
        
        df = table.to_pandas()
        
        if progress_callback:
            progress_callback(0.9, "Normalizing columns...")
        
        df = normalize_column_names(df)
        
        if 'Date' not in df.columns:
            if progress_callback:
                progress_callback(1.0, "Warning: No date column found")
            return pd.DataFrame()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        if progress_callback:
            progress_callback(1.0, f"Loaded {len(df)} rows")
        
        return df
        
    except Exception as e:
        if progress_callback:
            progress_callback(0.2, f"Dataset reading failed: {str(e)[:50]}")
        return load_data_from_parquet_fallback(pair, parquet_path, start_date, end_date, progress_callback)


def load_data_from_parquet_fallback(pair, parquet_path, start_date=None, end_date=None, progress_callback=None):
    """Fallback method: recursively find and load parquet files."""
    import pandas as pd
    
    data_frames = []
    pair_path = os.path.join(parquet_path, pair)
    
    if not os.path.exists(pair_path):
        return pd.DataFrame()
    
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None
    
    all_files = []
    for root, dirs, files in os.walk(pair_path):
        for file in files:
            if file.endswith('.parquet'):
                all_files.append(os.path.join(root, file))
    
    total_files = len(all_files)
    if total_files == 0:
        return pd.DataFrame()
    
    if progress_callback:
        progress_callback(0.1, f"Found {total_files} parquet files")
    
    loaded_files = 0
    for i, file_path in enumerate(sorted(all_files)):
        progress = 0.1 + (0.8 * (i / total_files))
        if progress_callback and i % max(1, total_files // 20) == 0:
            progress_callback(progress, f"Loading file {i+1}/{total_files}")
        
        try:
            path_parts = Path(file_path).parts
            
            year = None
            month = None
            day = None
            
            for part in path_parts:
                if part.startswith('year='):
                    year = int(part.replace('year=', ''))
                elif part.startswith('month='):
                    month = int(part.replace('month=', ''))
                elif part.startswith('day='):
                    day = int(part.replace('day=', ''))
            
            if start_dt and year is not None:
                if year < start_dt.year:
                    continue
                if year == start_dt.year and month is not None and month < start_dt.month:
                    continue
                if year == start_dt.year and month == start_dt.month and day is not None and day < start_dt.day:
                    continue
            
            if end_dt and year is not None:
                if year > end_dt.year:
                    continue
                if year == end_dt.year and month is not None and month > end_dt.month:
                    continue
                if year == end_dt.year and month == end_dt.month and day is not None and day > end_dt.day:
                    continue
            
            df = pd.read_parquet(file_path)
            
            if not df.empty:
                df = normalize_column_names(df)
                data_frames.append(df)
                loaded_files += 1
                
        except Exception as e:
            if progress_callback:
                progress_callback(progress, f"Error loading {os.path.basename(file_path)}")
            continue
    
    if not data_frames:
        return pd.DataFrame()
    
    if progress_callback:
        progress_callback(0.95, f"Combining {len(data_frames)} data chunks...")
    
    df = pd.concat(data_frames, ignore_index=True)
    
    if 'Date' not in df.columns:
        return pd.DataFrame()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    if progress_callback:
        progress_callback(1.0, f"Loaded {len(df)} rows from {loaded_files} files")
    
    return df


def normalize_column_names(df):
    """Normalize column names to expected format (Date, open, high, low, close)."""
    column_mapping = {}
    
    date_variants = ['Date', 'date', 'timestamp', 'time', 'datetime', 'ds']
    for variant in date_variants:
        if variant in df.columns:
            column_mapping[variant] = 'Date'
            break
    
    price_mapping = {
        'open': ['open', 'Open', 'OPEN', 'opening'],
        'high': ['high', 'High', 'HIGH'],
        'low': ['low', 'Low', 'LOW'],
        'close': ['close', 'Close', 'CLOSE', 'closing'],
        'volume': ['volume', 'Volume', 'VOLUME', 'vol']
    }
    
    for target, variants in price_mapping.items():
        for variant in variants:
            if variant in df.columns:
                column_mapping[variant] = target
                break
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df


def load_data_from_parquet_with_progress(pair, parquet_path, start_date=None, end_date=None):
    """Wrapper function to load parquet data with a progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
    
    df = load_data_from_parquet(pair, parquet_path, start_date, end_date, update_progress)
    
    status_text.empty()
    progress_bar.empty()
    
    return df


# ============================================================================
# Page Configuration
# ============================================================================
st.set_page_config(
    page_title="Sid Naiman SID Method Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# Custom CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 500;
        color: #1E88E5;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.5rem;
    }
    .signal-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .buy-signal-card {
        border-left: 8px solid #00C853;
    }
    .sell-signal-card {
        border-left: 8px solid #D32F2F;
    }
    .buy-signal {
        background-color: #00C853;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .sell-signal {
        background-color: #D32F2F;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .neutral-signal {
        background-color: #FFA000;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .rsi-oversold {
        background-color: #00C853;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .rsi-overbought {
        background-color: #D32F2F;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .rsi-neutral {
        background-color: #FFA000;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .gpu-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .cpu-status {
        background: linear-gradient(135deg, #757F9A 0%, #D7DDE8 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .parquet-status {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# SID METHOD FUNCTIONS
# ============================================================================

def calculate_sid_indicators(df: pd.DataFrame, sid: SidMethod) -> pd.DataFrame:
    """Calculate Sid Method indicators for a DataFrame."""
    df = df.copy()
    
    if len(df) < 50:
        return df
    
    df['rsi'] = sid.calculate_rsi(df)
    
    macd_df = sid.calculate_macd(df)
    df['macd'] = macd_df['macd']
    df['macd_signal'] = macd_df['signal']
    df['macd_hist'] = macd_df['histogram']
    
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    
    df['macd_aligned_up'] = (df['macd'] > df['macd'].shift(1)).astype(int)
    df['macd_aligned_down'] = (df['macd'] < df['macd'].shift(1)).astype(int)
    df['macd_cross_above'] = ((df['macd'].shift(1) <= df['macd_signal'].shift(1)) & 
                               (df['macd'] > df['macd_signal'])).astype(int)
    df['macd_cross_below'] = ((df['macd'].shift(1) >= df['macd_signal'].shift(1)) & 
                               (df['macd'] < df['macd_signal'])).astype(int)
    
    return df


def detect_sid_signals(df: pd.DataFrame, sid: SidMethod) -> List[Dict]:
    """Detect Sid Method trade signals from DataFrame."""
    signals = []
    
    if df.empty or len(df) < 50:
        return signals
    
    df = calculate_sid_indicators(df, sid)
    
    for i in range(max(0, len(df) - 20), len(df) - 1):
        current_rsi = df['rsi'].iloc[i]
        
        if current_rsi < 30:
            if df['macd_aligned_up'].iloc[i]:
                macd_crossed = df['macd_cross_above'].iloc[i]
                
                signal = {
                    'type': 'BUY',
                    'direction': 'long',
                    'signal_type': 'oversold',
                    'rsi_value': current_rsi,
                    'price': df['close'].iloc[i],
                    'date': df.index[i],
                    'macd_aligned': True,
                    'macd_crossed': macd_crossed,
                    'confidence': 70 + (10 if macd_crossed else 0)
                }
                signals.append(signal)
        
        elif current_rsi > 70:
            if df['macd_aligned_down'].iloc[i]:
                macd_crossed = df['macd_cross_below'].iloc[i]
                
                signal = {
                    'type': 'SELL',
                    'direction': 'short',
                    'signal_type': 'overbought',
                    'rsi_value': current_rsi,
                    'price': df['close'].iloc[i],
                    'date': df.index[i],
                    'macd_aligned': True,
                    'macd_crossed': macd_crossed,
                    'confidence': 70 + (10 if macd_crossed else 0)
                }
                signals.append(signal)
    
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals


def get_signal_strength(signal: Dict) -> str:
    """Get signal strength label."""
    conf = signal.get('confidence', 50)
    
    if conf >= 85:
        return 'VERY_STRONG'
    elif conf >= 75:
        return 'STRONG'
    elif conf >= 65:
        return 'MODERATE'
    elif conf >= 55:
        return 'WEAK'
    else:
        return 'VERY_WEAK'


def get_rsi_status(rsi_value: float) -> Tuple[str, str]:
    """Get RSI status and CSS class."""
    if rsi_value < 30:
        return 'OVERSOLD', 'rsi-oversold'
    elif rsi_value > 70:
        return 'OVERBOUGHT', 'rsi-overbought'
    else:
        return 'NEUTRAL', 'rsi-neutral'


def create_sid_chart(df: pd.DataFrame, signals: List[Dict], pair: str) -> go.Figure:
    """Create interactive candlestick chart with Sid Method indicators."""
    
    n_rows = 3 if len(signals) == 0 else 4
    row_heights = [0.5, 0.25, 0.25][:n_rows]
    
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[f"{pair} - Price with Signals", "RSI", "MACD"]
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index if 'Date' not in df.columns else df['Date'],
            open=df['open'] if 'open' in df.columns else df['Open'],
            high=df['high'] if 'high' in df.columns else df['High'],
            low=df['low'] if 'low' in df.columns else df['Low'],
            close=df['close'] if 'close' in df.columns else df['Close'],
            name='Price',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add signal markers
    for signal in signals:
        if signal.get('trading_pair') == pair:
            color = 'green' if signal['type'] == 'BUY' else 'red'
            symbol = 'triangle-up' if signal['type'] == 'BUY' else 'triangle-down'
            
            fig.add_trace(
                go.Scatter(
                    x=[signal['date']],
                    y=[signal['price']],
                    mode='markers',
                    marker=dict(
                        symbol=symbol,
                        size=15,
                        color=color,
                        line=dict(width=1, color='white')
                    ),
                    name=signal['type'],
                    showlegend=False,
                    hovertemplate=f"{signal['type']}<br>RSI: {signal['rsi_value']:.1f}<br>Confidence: {signal['confidence']:.0f}<extra></extra>"
                ),
                row=1, col=1
            )
    
    # RSI
    fig.add_trace(
        go.Scatter(
            x=df.index if 'Date' not in df.columns else df['Date'],
            y=df['rsi'],
            line=dict(color='purple', width=2),
            name='RSI'
        ),
        row=2, col=1
    )
    
    # RSI levels
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="orange", opacity=0.3, row=2, col=1)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index if 'Date' not in df.columns else df['Date'],
                y=df['macd'],
                line=dict(color='blue', width=2),
                name='MACD'
            ),
            row=3, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index if 'Date' not in df.columns else df['Date'],
                y=df['macd_signal'],
                line=dict(color='red', width=2),
                name='Signal'
            ),
            row=3, col=1
        )
        
        if 'macd_hist' in df.columns:
            colors = ['green' if val >= 0 else 'red' for val in df['macd_hist']]
            fig.add_trace(
                go.Bar(
                    x=df.index if 'Date' not in df.columns else df['Date'],
                    y=df['macd_hist'],
                    marker_color=colors,
                    name='Histogram'
                ),
                row=3, col=1
            )
    
    fig.update_layout(
        title=f"{pair} - SID Method Analysis",
        height=800,
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    
    return fig


def fetch_market_data(pairs: List[str], oanda_client, timeframe: str = "1h", bars: int = 200) -> Dict[str, pd.DataFrame]:
    """Fetch market data for selected pairs with error handling."""
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_pairs = len(pairs)
    failed_pairs = []
    
    for i, pair in enumerate(pairs):
        status_text.text(f"📡 Fetching {pair}... ({i+1}/{total_pairs})")
        try:
            oanda_pair = pair.replace('/', '_')
            
            if oanda_pair not in VALID_OANDA_PAIRS:
                status_text.text(f"⚠️ Skipping unsupported pair: {pair}")
                failed_pairs.append(pair)
                continue
            
            df = oanda_client.fetch_candles(
                instrument=oanda_pair,
                granularity=timeframe,
                count=bars
            )
            
            if not df.empty and len(df) > 20:
                data[pair] = df
            else:
                failed_pairs.append(pair)
                
        except Exception as e:
            failed_pairs.append(pair)
            logger.warning(f"Failed to fetch {pair}: {e}")
        
        progress_bar.progress((i + 1) / total_pairs)
    
    status_text.empty()
    progress_bar.empty()
    
    if failed_pairs:
        st.caption(f"⚠️ Failed to fetch: {', '.join(failed_pairs[:5])}" + 
                   (f" and {len(failed_pairs)-5} more" if len(failed_pairs) > 5 else ""))
    
    return data


def detect_sid_signals_with_ai(data: Dict[str, pd.DataFrame], sid: SidMethod, ai_enabled: bool = False, signal_predictor=None) -> List[Dict]:
    """Detect Sid Method signals and augment with AI predictions."""
    all_signals = []
    
    for pair, df in data.items():
        if df.empty or len(df) < 50:
            continue
        
        signals = detect_sid_signals(df, sid)
        
        for signal in signals:
            signal['trading_pair'] = pair
            signal['data'] = df
            signal['detected_at'] = datetime.now()
            
            if ai_enabled and signal_predictor:
                try:
                    idx = signal.get('index', -1)
                    if idx >= 0 and idx < len(df):
                        features = {}
                        for col in df.columns:
                            features[col] = df[col].iloc[idx]
                        
                        ai_pred = signal_predictor.predict_zone_success(
                            features=features,
                            model_name='auto',
                            pair=pair
                        )
                        
                        signal['ai_confidence'] = ai_pred.get('confidence', 0)
                        signal['ai_success_probability'] = ai_pred.get('success_probability', 0.5)
                        signal['ai_signal_strength'] = get_signal_strength(signal)
                        
                        base_conf = signal['confidence']
                        ai_conf = ai_pred.get('confidence', 0) / 100
                        signal['confidence'] = (base_conf * 0.6 + ai_conf * 100 * 0.4)
                        
                except Exception as e:
                    signal['ai_confidence'] = 0
            else:
                signal['ai_confidence'] = 0
            
            all_signals.append(signal)
    
    all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    return all_signals


def display_signals(signals: List[Dict]):
    """Display detected Sid Method signals with AI information."""
    if not signals:
        st.info("📊 No trading signals detected in current data")
        return
    
    with st.expander("🔍 Filter Signals", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_confidence = st.slider("Min Confidence", 0, 100, 50)
        
        with col2:
            signal_type_options = ["All", "BUY", "SELL"]
            signal_type = st.selectbox("Signal Type", signal_type_options)
        
        with col3:
            show_ai_only = st.checkbox("AI Enhanced Only", False)
    
    filtered = [s for s in signals if s.get('confidence', 0) >= min_confidence]
    
    if signal_type != "All":
        filtered = [s for s in filtered if s['type'] == signal_type]
    
    if show_ai_only:
        filtered = [s for s in filtered if s.get('ai_confidence', 0) > 0]
    
    st.caption(f"Showing {len(filtered)} of {len(signals)} signals")
    
    for signal in filtered:
        pair = signal['trading_pair']
        signal_type = signal['type']
        signal_class = "buy-signal-card" if signal_type == 'BUY' else "sell-signal-card"
        rsi_status, rsi_class = get_rsi_status(signal['rsi_value'])
        strength = get_signal_strength(signal)
        
        with st.container():
            st.markdown(f'<div class="signal-card {signal_class}">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown(f"### {pair} - {signal_type} Signal")
                st.markdown(f"<span class='{rsi_class}'>RSI: {signal['rsi_value']:.1f} ({rsi_status})</span>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"**Price:** {signal['price']:.5f}")
                st.markdown(f"**Date:** {signal['date'].strftime('%Y-%m-%d')}")
            
            with col3:
                badge_class = f"ai-{strength.lower().replace('_', '-')}" if 'ai-' not in strength.lower() else strength.lower()
                st.markdown(f"**Confidence:** {signal['confidence']:.0f}%")
                if signal.get('ai_confidence', 0) > 0:
                    st.markdown(f"**AI Confidence:** {signal['ai_confidence']:.0f}%")
            
            st.markdown(f"**MACD Aligned:** {'✅' if signal['macd_aligned'] else '❌'} | **Crossed:** {'✅' if signal['macd_crossed'] else '❌'}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button(f"📊 View Chart", key=f"chart_{id(signal)}", use_container_width=False):
                    st.session_state[f"show_chart_{id(signal)}"] = signal
            
            with col2:
                if st.button(f"💰 Execute Trade", key=f"trade_{id(signal)}", type="primary", use_container_width=False):
                    st.session_state[f"confirm_signal_{id(signal)}"] = signal
            
            if st.session_state.get(f"show_chart_{id(signal)}", False):
                if 'data' in signal:
                    fig = create_sid_chart(signal['data'], [signal], pair)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("Close Chart", key=f"close_chart_{id(signal)}"):
                        st.session_state[f"show_chart_{id(signal)}"] = False
            
            st.markdown('</div>', unsafe_allow_html=True)


def render_trade_confirmation(signal: Dict, oanda_trader, account_summary):
    """Render trade confirmation dialog."""
    pair = signal['trading_pair']
    signal_type = signal['type']
    
    st.markdown("""
    <div class="warning-box">
        <h3 style="text-align: center;">⚠️ Confirm Trade</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Trade Details")
        st.markdown(f"""
        - **Instrument:** `{pair}`
        - **Direction:** `{'BUY' if signal_type == 'BUY' else 'SELL'}`
        - **Signal:** `{signal['signal_type'].upper()}`
        - **RSI:** `{signal['rsi_value']:.1f}`
        - **Price:** `{signal['price']:.5f}`
        - **Confidence:** `{signal['confidence']:.0f}%`
        """)
        
        if signal.get('ai_confidence', 0) > 0:
            st.markdown(f"""
            **🤖 AI Analysis**
            - Confidence: `{signal['ai_confidence']:.1f}%`
            - Strength: `{signal.get('ai_signal_strength', 'neutral')}`
            """)
    
    with col2:
        st.markdown("### 💰 Position Sizing (SID Method)")
        
        account = account_summary.get('account', {})
        balance = float(account.get('balance', 10000))
        
        risk_percent = st.slider("Risk %", 0.5, 2.0, 1.0, 0.1)
        risk_amount = balance * (risk_percent / 100)
        
        instrument = pair.replace('/', '_')
        
        if signal_type == 'BUY':
            stop_loss = signal['price'] * 0.99
            take_profit = signal['price'] * 1.02
        else:
            stop_loss = signal['price'] * 1.01
            take_profit = signal['price'] * 0.98
        
        risk_per_unit = abs(signal['price'] - stop_loss)
        units = int(risk_amount / risk_per_unit) if risk_per_unit > 0 else 0
        
        st.markdown(f"""
        **Risk Management:**
        - Balance: `${balance:,.2f}`
        - Risk Amount: `${risk_amount:.2f}` ({risk_percent}%)
        - Stop Loss: `{stop_loss:.5f}`
        - Take Profit: `{take_profit:.5f}`
        - Position Size: `{units:,}` units
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ CONFIRM", type="primary", use_container_width=False):
            try:
                result = oanda_trader.place_order(
                    instrument=instrument,
                    units=units if signal_type == 'BUY' else -units,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    order_type="MARKET"
                )
                
                if result.get('success'):
                    st.success("✅ Trade executed!")
                    st.session_state[f"confirm_signal_{id(signal)}"] = None
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=False):
            st.session_state[f"confirm_signal_{id(signal)}"] = None
            st.rerun()


def display_open_positions(oanda_trader):
    """Display open positions."""
    positions = oanda_trader.get_open_trades()
    st.session_state.open_positions = positions
    
    if not positions:
        st.info("📭 No open positions")
        return
    
    for position in positions:
        with st.container():
            st.markdown('<div class="signal-card">', unsafe_allow_html=True)
            
            current = position.get('current_price', position['price'])
            entry = position['price']
            
            if position['units'] > 0:
                pl = (current - entry) * position['units']
                pl_pct = ((current - entry) / entry) * 100
            else:
                pl = (entry - current) * abs(position['units'])
                pl_pct = ((entry - current) / entry) * 100
            
            pl_color = "green" if pl > 0 else "red"
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**{position['instrument']}**")
                st.markdown(f"Units: `{position['units']}`")
            
            with col2:
                st.markdown(f"Entry: `{position['price']:.5f}`")
                st.markdown(f"Current: `{current:.5f}`")
            
            with col3:
                st.markdown(f"P/L: <span style='color:{pl_color};'>${pl:.2f} ({pl_pct:.2f}%)</span>", unsafe_allow_html=True)
            
            if st.button(f"Close Position", key=f"close_{position['id']}", use_container_width=False):
                result = oanda_trader.close_trade(position['id'])
                if result.get('success'):
                    if result.get('pl', 0) < 0:
                        st.session_state.daily_loss += abs(result['pl'])
                        st.session_state.consecutive_losses += 1
                    else:
                        st.session_state.consecutive_losses = 0
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)


def display_trade_history(oanda_trader):
    """Display trade history."""
    history = oanda_trader.get_trade_history()
    
    if not history:
        st.info("📭 No trade history")
        return
    
    total_trades = len(history)
    wins = sum(1 for t in history if t.get('outcome') == 'win')
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Wins", wins)
    with col3:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    
    trade_data = []
    for trade in history[-20:]:
        trade_data.append({
            'Date': str(trade.get('entry_time', ''))[:10],
            'Pair': str(trade.get('instrument', '')),
            'Direction': str(trade.get('direction', '')),
            'Entry': f"{trade.get('entry_price', 0):.5f}",
            'Risk %': str(trade.get('risk_percent', 0)),
            'Outcome': str(trade.get('outcome', 'pending'))
        })
    
    if trade_data:
        df_trades = pd.DataFrame(trade_data)
        for col in df_trades.columns:
            df_trades[col] = df_trades[col].astype(str)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)


# ============================================================================
# AI TRAINING FUNCTIONS
# ============================================================================

def display_training_data_tab():
    """Display the Training Data tab content."""
    st.markdown("#### 📊 Training Data Management (SID Method)")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.metric("Current Training Samples", len(st.session_state.training_data))
    
    with col2:
        if st.button("🗑️ Clear All Data", use_container_width=False):
            st.session_state.training_data = []
            st.success("Training data cleared")
            st.rerun()
    
    st.markdown("---")
    st.markdown("#### 📂 Load ALL Data from Forex Files")
    
    parquet_pairs = get_available_pairs_with_parquet(st.session_state.user_parquet_path)
    if parquet_pairs:
        st.success(f"✅ Parquet available for {len(parquet_pairs)} pairs (10x faster)")
    else:
        st.info("ℹ️ No Parquet files found. Will use CSV (slower). Use Parquet Management tab to convert.")
    
    all_pairs = sorted(set(VALID_OANDA_PAIRS + parquet_pairs))
    
    if all_pairs:
        selected_pair = st.selectbox(
            "Select Trading Pair",
            all_pairs,
            index=0 if all_pairs else None
        )
    else:
        st.warning("No Forex pairs found")
        selected_pair = None
    
    if selected_pair:
        has_parquet = selected_pair in parquet_pairs
        if has_parquet:
            use_parquet = st.checkbox("🚀 Use Parquet (10x faster)", value=True)
        else:
            use_parquet = False
            st.info("ℹ️ No Parquet for this pair yet. Will use CSV. Use Parquet Management tab to convert.")
    
    st.markdown("##### Select Time Range (Loads ALL data in this range)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        range_preset = st.selectbox(
            "Quick Select",
            ["MAX", "Custom", "Last Week", "Last 2 Weeks", "Last Month", 
             "Last 3 Months", "Last 6 Months", "Last Year", 
             "Last 2 Years", "Last 5 Years", "Last 10 Years"],
            index=0
        )
    
    with col2:
        if range_preset == "Custom":
            start_date = st.date_input(
                "Start Date",
                datetime.now() - timedelta(days=365)
            )
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = None
            if range_preset != "MAX":
                days_map = {
                    "Last Week": 7, "Last 2 Weeks": 14, "Last Month": 30,
                    "Last 3 Months": 90, "Last 6 Months": 180, "Last Year": 365,
                    "Last 2 Years": 730, "Last 5 Years": 1825, "Last 10 Years": 3650
                }
                days = days_map.get(range_preset, 365)
                start_date = datetime.now() - timedelta(days=days)
                st.info(f"Start: {start_date.strftime('%Y-%m-%d')}")
            else:
                st.info("MAX: All available data")
    
    with col3:
        if range_preset == "Custom":
            end_date = st.date_input("End Date", datetime.now())
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = None
            if range_preset != "MAX":
                st.info(f"End: {datetime.now().strftime('%Y-%m-%d')}")
            else:
                st.info("Loading EVERYTHING...")
    
    st.markdown("##### Data Resolution")
    timeframe = st.selectbox(
        "Timeframe for Analysis",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=4,
        help="Data will be averaged to this timeframe"
    )
    
    if st.button("📥 LOAD ALL DATA", type="primary", use_container_width=False):
        if not selected_pair:
            st.error("Please select a trading pair")
        else:
            try:
                sid = st.session_state.sid_method
                
                if use_parquet:
                    df = load_data_from_parquet_with_progress(
                        selected_pair,
                        st.session_state.user_parquet_path,
                        start_date=start_str,
                        end_date=end_str
                    )
                    
                    if not df.empty and 'Date' in df.columns:
                        df = df.set_index('Date')
                        df = calculate_sid_indicators(df, sid)
                        
                        X = df.dropna()
                        
                        new_samples = 0
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(min(len(X), 100000)):
                            if i % 1000 == 0:
                                progress = i / min(len(X), 100000)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing sample {i}/{min(len(X), 100000)}")
                            
                            sample = {
                                'timestamp': df.index[i] if i < len(df.index) else datetime.now(),
                                'pair': selected_pair,
                                'timeframe': timeframe,
                                'date_range': range_preset,
                                'rsi': df['rsi'].iloc[i] if 'rsi' in df.columns else 50,
                                'macd': df['macd'].iloc[i] if 'macd' in df.columns else 0,
                                'macd_signal': df['macd_signal'].iloc[i] if 'macd_signal' in df.columns else 0,
                                'rsi_oversold': 1 if df['rsi'].iloc[i] < 30 else 0,
                                'rsi_overbought': 1 if df['rsi'].iloc[i] > 70 else 0,
                                'macd_aligned_up': df['macd_aligned_up'].iloc[i] if 'macd_aligned_up' in df.columns else 0,
                                'macd_aligned_down': df['macd_aligned_down'].iloc[i] if 'macd_aligned_down' in df.columns else 0,
                                'outcome': 1 if i < len(df) - 1 and df['close'].iloc[i+1] > df['close'].iloc[i] else 0
                            }
                            st.session_state.training_data.append(sample)
                            new_samples += 1
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        st.success(f"✅ Loaded {new_samples} training samples for {selected_pair}")
                    else:
                        st.warning(f"No data available for {selected_pair}")
                
                else:
                    st.warning("CSV loading requires implementation")
                    
            except Exception as e:
                st.error(f"Error loading data: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    st.markdown("---")
    st.markdown("#### 📋 Current Training Data")
    
    if st.session_state.training_data:
        df_summary = pd.DataFrame(st.session_state.training_data)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df_summary))
        with col2:
            if 'outcome' in df_summary.columns:
                win_rate = (df_summary['outcome'].sum() / len(df_summary)) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
        with col3:
            if 'pair' in df_summary.columns:
                unique_pairs = df_summary['pair'].nunique()
                st.metric("Unique Pairs", unique_pairs)
        with col4:
            st.metric("Timeframes", df_summary['timeframe'].nunique() if 'timeframe' in df_summary.columns else 1)
        
        st.markdown("##### Recent Samples")
        try:
            display_cols = ['timestamp', 'pair', 'timeframe', 'rsi', 'outcome']
            existing_cols = [col for col in display_cols if col in df_summary.columns]
            
            if existing_cols:
                display_df = df_summary[existing_cols].tail(10).copy()
                
                for col in display_df.columns:
                    if col == 'timestamp':
                        try:
                            ts = pd.to_datetime(display_df[col], errors='coerce')
                            display_df[col] = ts.dt.strftime('%Y-%m-%d %H:%M').where(ts.notna(), 'Invalid')
                        except:
                            display_df[col] = display_df[col].astype(str)
                    else:
                        display_df[col] = display_df[col].astype(str)
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
            else:
                display_df = df_summary.tail(10).copy()
                for col in display_df.columns:
                    display_df[col] = display_df[col].astype(str)
                st.dataframe(display_df, use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Could not display samples: {e}")
        
        if st.button("💾 Save Training Data to CSV"):
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_summary.to_csv(filename, index=False)
            st.success(f"Saved to {filename}")
    else:
        st.info("No training data loaded yet. Click 'LOAD ALL DATA' above.")


def display_parquet_management():
    """Display Parquet management interface."""
    st.markdown("### 💾 Parquet Management")
    st.markdown("Convert CSV to Parquet for 10x faster data loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats = get_parquet_stats(st.session_state.user_parquet_path)
        
        st.metric("Pairs with Parquet", stats['total_pairs'])
        st.metric("Total Files", stats['total_files'])
        st.metric("Total Size", f"{stats['total_size_gb']:.2f} GB")
        
        if stats['pairs']:
            st.markdown("##### Available Parquet Pairs")
            for p in stats['pairs'][:10]:
                years_str = f"{p['years'][0]}-{p['years'][-1]}" if p['years'] else "Unknown"
                st.caption(f"• {p['pair']}: {p['files']} files, {p['size_gb']:.2f}GB ({years_str})")
    
    with col2:
        csv_pairs = []
        if os.path.exists(st.session_state.user_csv_path):
            for item in os.listdir(st.session_state.user_csv_path):
                pair_path = os.path.join(st.session_state.user_csv_path, item)
                if os.path.isdir(pair_path):
                    csv_pairs.append(item)
        
        if csv_pairs:
            convert_pair = st.selectbox(
                "Select Pair to Convert",
                sorted(csv_pairs)
            )
            
            use_gpu = st.checkbox("🚀 Use GPU for conversion", value=AI_AVAILABLE and st.session_state.ai_accelerator and st.session_state.ai_accelerator.has_gpu)
            
            mode = st.selectbox(
                "Conversion Mode",
                ["append", "initial", "overwrite", "update"],
                index=0,
                help="append: Add new dates only, initial: First time setup, overwrite: Replace all, update: Refresh existing"
            )
            
            if st.button("🔄 Convert to Parquet", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    if st.session_state.parquet_converter is None:
                        try:
                            converter = ParquetConverter(
                                csv_dir=st.session_state.user_csv_path,
                                parquet_dir=st.session_state.user_parquet_path,
                                use_gpu=use_gpu
                            )
                        except TypeError:
                            try:
                                converter = ParquetConverter(
                                    st.session_state.user_csv_path, 
                                    st.session_state.user_parquet_path,
                                    use_gpu
                                )
                            except TypeError:
                                try:
                                    converter = ParquetConverter(use_gpu=use_gpu)
                                except TypeError:
                                    converter = ParquetConverter()
                    else:
                        converter = st.session_state.parquet_converter
                    
                    if hasattr(converter, 'convert_pair'):
                        success = converter.convert_pair(
                            pair=convert_pair,
                            mode=mode,
                            progress_callback=update_progress
                        )
                    elif hasattr(converter, 'convert'):
                        success = converter.convert(
                            pair=convert_pair,
                            mode=mode
                        )
                    else:
                        st.error("❌ Converter doesn't have convert_pair or convert method")
                        return
                    
                    if success:
                        st.success(f"✅ {convert_pair} converted successfully!")
                        st.rerun()
                    else:
                        st.error(f"❌ Conversion failed")
                        
                except Exception as e:
                    st.error(f"Conversion error: {e}")
        else:
            st.warning("No CSV files found")


def display_model_management():
    """Display AI model management interface for Sid Method models."""
    st.markdown("### 🤖 AI Model Management (SID Method)")
    
    if not AI_AVAILABLE or st.session_state.model_manager is None:
        st.warning("⚠️ AI modules not available")
        return
    
    models_df = st.session_state.model_manager.get_latest_models_summary()
    
    if models_df.empty:
        st.info("No trained models found. Run training pipeline to create models.")
        
        if st.button("🚀 Train Default Model", type="primary"):
            with st.spinner("Training model..."):
                result = st.session_state.training_pipeline.train_pair(
                    pair="EUR_USD",
                    force=True,
                    samples=50000
                )
                if result.get('status') == 'success':
                    st.success(f"✅ Model trained! Accuracy: {result['accuracy']:.3f}")
                    st.rerun()
                else:
                    st.error(f"Training failed: {result.get('message', 'Unknown error')}")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(models_df))
    with col2:
        unique_pairs = models_df['Pair'].nunique()
        st.metric("Unique Pairs", unique_pairs)
    with col3:
        accuracy_values = []
        for acc in models_df['Accuracy']:
            if acc != 'N/A' and isinstance(acc, (int, float)):
                accuracy_values.append(acc)
        
        if accuracy_values:
            avg_accuracy = sum(accuracy_values) / len(accuracy_values)
            st.metric("Avg Accuracy", f"{avg_accuracy:.1%}")
        else:
            st.metric("Avg Accuracy", "N/A")
    with col4:
        if not models_df.empty and 'To' in models_df.columns:
            latest_date = models_df['To'].iloc[0]
            if hasattr(latest_date, 'strftime'):
                latest_date_str = latest_date.strftime('%Y-%m-%d')
            else:
                latest_date_str = str(latest_date)
            st.metric("Latest Model", latest_date_str)
        else:
            st.metric("Latest Model", "N/A")
    
    selected_pair = st.selectbox(
        "Filter by Pair",
        ["All"] + sorted(models_df['Pair'].unique().tolist())
    )
    
    if selected_pair != "All":
        display_df = models_df[models_df['Pair'] == selected_pair].copy()
    else:
        display_df = models_df.copy()
    
    display_cols = ['Pair', 'From', 'To', 'Days', 'Accuracy', 'Samples', 'Device', 'Model Name']
    available_cols = [c for c in display_cols if c in display_df.columns]
    
    display_data = pd.DataFrame()
    
    for col in available_cols:
        if col in ['From', 'To']:
            display_data[col] = display_df[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
            )
        elif col == 'Accuracy':
            display_data[col] = display_df[col].apply(
                lambda x: f"{x:.1%}" if isinstance(x, (int, float)) and x != 'N/A' else str(x)
            )
        elif col == 'Samples':
            display_data[col] = display_df[col].apply(
                lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x)
            )
        else:
            display_data[col] = display_df[col].astype(str)
    
    st.dataframe(display_data, use_container_width=True, hide_index=True)
    
    st.markdown("#### 🛠️ Model Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Models", use_container_width=True):
            if st.session_state.signal_predictor:
                st.session_state.signal_predictor.reload_models()
            st.success("Models reloaded")
            st.rerun()
    
    with col2:
        if st.button("📊 Validate All Models", use_container_width=True):
            with st.spinner("Validating models..."):
                st.info("Validation functionality will be implemented")
    
    with col3:
        if st.button("🧹 Cleanup Old Models", use_container_width=True):
            st.info("Cleanup functionality will be implemented")
    
    st.markdown("#### 🚀 Training Pipeline")
    
    col1, col2 = st.columns(2)
    
    with col1:
        train_pair = st.selectbox(
            "Train Specific Pair",
            ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"] + 
            [p for p in VALID_OANDA_PAIRS[:10] if p not in ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]]
        )
        
        if st.button(f"🚀 Train {train_pair}", use_container_width=True):
            with st.spinner(f"Training {train_pair}..."):
                result = st.session_state.training_pipeline.train_pair(
                    pair=train_pair,
                    force=True,
                    samples=100000
                )
                if result.get('status') == 'success':
                    st.success(f"✅ Model trained! Accuracy: {result['accuracy']:.3f}")
                    st.rerun()
                else:
                    st.error(f"Training failed: {result.get('message', 'Unknown error')}")
    
    with col2:
        if st.button("🎯 Train All Pairs", use_container_width=True):
            with st.spinner("Training all pairs... This may take a while"):
                st.info("Batch training will be implemented")


def display_ai_training():
    """Display AI training interface with all tabs."""
    st.markdown("### 🧠 AI Model Training (SID Method)")
    
    if not AI_AVAILABLE:
        st.warning("⚠️ AI modules not available. Run: pip install scikit-learn pandas numpy xgboost")
        return
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Training Data", 
        "🏋️ Train Models", 
        "💾 Parquet Management", 
        "🤖 Model Management"
    ])
    
    with tab1:
        display_training_data_tab()
    
    with tab2:
        st.markdown("#### 🏋️ Train New Models (SID Method)")
        st.info("Training interface will be implemented")
    
    with tab3:
        display_parquet_management()
    
    with tab4:
        display_model_management()


# ============================================================================
# INITIALIZATION
# ============================================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.oanda_client = None
    st.session_state.oanda_trader = None
    st.session_state.sd_detector = None
    st.session_state.sid_method = None
    st.session_state.ai_accelerator = None
    st.session_state.feature_engineer = None
    st.session_state.signal_predictor = None
    st.session_state.model_trainer = None
    st.session_state.model_manager = None
    st.session_state.training_pipeline = None
    st.session_state.parquet_converter = None
    st.session_state.current_data = {}
    st.session_state.detected_signals = []
    st.session_state.trade_history = []
    st.session_state.open_positions = []
    st.session_state.account_summary = {}
    st.session_state.selected_pairs = []
    st.session_state.auto_refresh = False
    st.session_state.refresh_interval = 60
    st.session_state.last_refresh = datetime.now()
    st.session_state.daily_reset_time = None
    st.session_state.consecutive_losses = 0
    st.session_state.daily_loss = 0.0
    st.session_state.ai_enabled = AI_AVAILABLE
    st.session_state.preferred_device = 'auto'
    st.session_state.training_data = []
    st.session_state.training_queue = []
    st.session_state.model_metrics = {}
    st.session_state.show_ai_details = False
    st.session_state.api_error_count = 0
    st.session_state.use_parquet = True
    st.session_state.model_info = {}
    
    # User configuration
    st.session_state.user_csv_path = DEFAULT_CSV_PATH
    st.session_state.user_parquet_path = DEFAULT_PARQUET_PATH
    st.session_state.user_model_path = DEFAULT_MODEL_PATH
    st.session_state.practice_api_key = load_api_key_from_env('practice')
    st.session_state.live_api_key = load_api_key_from_env('live')
    st.session_state.practice_account_id = os.environ.get('OANDA_ACCOUNT_ID_PRACTICE', '')
    st.session_state.live_account_id = os.environ.get('OANDA_ACCOUNT_ID_LIVE', '')
    st.session_state.environment = 'practice'


def initialize_system():
    """Initialize all components with user-configured paths and API keys."""
    if not CORE_IMPORTS_SUCCESS:
        st.error("Core modules not available. Please check installation.")
        return False
    
    try:
        paths = get_user_paths()
        csv_path = paths['csv_path']
        parquet_path = paths['parquet_path']
        model_path = paths['model_path']
        
        if csv_path:
            os.environ['FOREX_CSV_PATH'] = csv_path
        if parquet_path:
            os.environ['FOREX_PARQUET_PATH'] = parquet_path
        if model_path:
            os.environ['MODEL_PATH'] = model_path
        
        with st.spinner("🔄 Connecting to OANDA..."):
            env = st.session_state.environment
            api_key = st.session_state.practice_api_key if env == 'practice' else st.session_state.live_api_key
            account_id = st.session_state.practice_account_id if env == 'practice' else st.session_state.live_account_id
            
            if api_key:
                os.environ['OANDA_API_KEY'] = api_key
                if account_id:
                    os.environ['OANDA_ACCOUNT_ID'] = account_id
                
                st.session_state.oanda_client = OANDAClient()
                st.session_state.oanda_trader = OANDATrader(environment=env)
            else:
                st.warning("⚠️ OANDA API key not configured. Live data will not be available.")
        
        with st.spinner("🔄 Loading Sid Method..."):
            st.session_state.sid_method = SidMethod()
            st.session_state.sd_detector = SupplyDemand()
        
        if AI_AVAILABLE:
            with st.spinner("🔄 Initializing AI engine..."):
                try:
                    st.session_state.ai_accelerator = AIAccelerator(preferred_device=st.session_state.preferred_device)
                    st.session_state.feature_engineer = FeatureEngineer(st.session_state.ai_accelerator)
                    
                    st.session_state.model_manager = ModelManager(
                        model_dir=model_path,
                        parquet_base_path=parquet_path
                    )
                    
                    st.session_state.signal_predictor = SignalPredictor(
                        st.session_state.ai_accelerator,
                        model_dir=model_path
                    )
                    
                    st.session_state.model_trainer = ModelTrainer(
                        st.session_state.ai_accelerator,
                        model_dir=model_path
                    )
                    
                    st.session_state.training_pipeline = TrainingPipeline(
                        model_dir=model_path,
                        parquet_path=parquet_path,
                        csv_path=csv_path
                    )
                    
                    if PARQUET_AVAILABLE:
                        try:
                            st.session_state.parquet_converter = ParquetConverter(
                                csv_dir=csv_path,
                                parquet_dir=parquet_path,
                                use_gpu=st.session_state.ai_accelerator.has_gpu if st.session_state.ai_accelerator else False
                            )
                        except TypeError:
                            try:
                                st.session_state.parquet_converter = ParquetConverter(
                                    csv_path, 
                                    parquet_path,
                                    st.session_state.ai_accelerator.has_gpu if st.session_state.ai_accelerator else False
                                )
                            except TypeError:
                                st.session_state.parquet_converter = ParquetConverter()
                    
                    if st.session_state.model_manager:
                        models_df = st.session_state.model_manager.get_latest_models_summary()
                        if not models_df.empty:
                            st.session_state.model_info = models_df.to_dict('records')
                    
                except Exception as e:
                    st.warning(f"AI initialization partially failed: {e}")
                    st.session_state.ai_enabled = False
        
        if st.session_state.oanda_trader:
            with st.spinner("🔄 Fetching account details..."):
                st.session_state.account_summary = st.session_state.oanda_trader.get_account_summary()
        
        if st.session_state.daily_reset_time is None:
            st.session_state.daily_reset_time = datetime.now()
        
        st.session_state.initialized = True
        
        st.success(f"✅ System initialized with:")
        st.info(f"📁 CSV Path: {csv_path}")
        st.info(f"📊 Parquet Path: {parquet_path}")
        st.info(f"🤖 Model Path: {model_path}")
        
        if st.session_state.oanda_client:
            st.success(f"✅ OANDA {st.session_state.environment.upper()} connected")
        
        if AI_AVAILABLE and st.session_state.ai_accelerator and st.session_state.ai_accelerator.has_gpu:
            st.success(f"✅ GPU: {st.session_state.ai_accelerator.device_name}")
        
        parquet_pairs = get_available_pairs_with_parquet(parquet_path)
        if parquet_pairs:
            st.success(f"✅ Parquet available for {len(parquet_pairs)} pairs")
        
        if st.session_state.model_manager:
            models_df = st.session_state.model_manager.get_latest_models_summary()
            if not models_df.empty:
                st.success(f"✅ {len(models_df)} trained models available")
        
        return True
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        logger.error(f"Initialization error: {e}")
        return False


def check_daily_reset():
    """Check if we need to reset daily counters."""
    now = datetime.now()
    if st.session_state.daily_reset_time.date() < now.date():
        st.session_state.daily_loss = 0.0
        st.session_state.consecutive_losses = 0
        st.session_state.daily_reset_time = now
        if st.session_state.oanda_trader:
            st.session_state.oanda_trader.reset_daily()


# ============================================================================
# MAIN APP LAYOUT
# ============================================================================
st.markdown('<h1 class="main-header">📈 Sid Naiman SID Method Trading System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # ========================================================================
    # SECTION 1: OANDA API KEY MANAGEMENT
    # ========================================================================
    with st.expander("🔑 OANDA API Keys", expanded=True):
        st.markdown("### Practice Account")
        practice_api_key = st.text_input(
            "Practice API Key",
            value=st.session_state.practice_api_key or "",
            type="password",
            placeholder="Enter your practice API key",
            help="Get from OANDA account settings"
        )
        practice_account_id = st.text_input(
            "Practice Account ID",
            value=st.session_state.practice_account_id or "",
            placeholder="e.g., 101-004-35778624-001"
        )
        
        st.markdown("### Live Account (Real Money)")
        live_api_key = st.text_input(
            "Live API Key",
            value=st.session_state.live_api_key or "",
            type="password",
            placeholder="Enter your live API key",
            help="⚠️ This is for real money trading - use with caution"
        )
        live_account_id = st.text_input(
            "Live Account ID",
            value=st.session_state.live_account_id or "",
            placeholder="e.g., 101-004-35778624-001"
        )
        
        environment = st.radio(
            "Trading Environment",
            ["practice", "live"],
            index=0 if st.session_state.environment == 'practice' else 1,
            help="Practice = Demo account, Live = Real money"
        )
        
        if environment != st.session_state.environment:
            st.session_state.environment = environment
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save API Keys", use_container_width=True):
                if practice_api_key:
                    save_api_key_to_env('practice', practice_api_key)
                    st.session_state.practice_api_key = practice_api_key
                if practice_account_id:
                    os.environ['OANDA_ACCOUNT_ID_PRACTICE'] = practice_account_id
                    st.session_state.practice_account_id = practice_account_id
                if live_api_key:
                    save_api_key_to_env('live', live_api_key)
                    st.session_state.live_api_key = live_api_key
                if live_account_id:
                    os.environ['OANDA_ACCOUNT_ID_LIVE'] = live_account_id
                    st.session_state.live_account_id = live_account_id
                st.success("✅ API keys saved")
        
        with col2:
            api_key_to_test = practice_api_key if environment == 'practice' else live_api_key
            account_id_to_test = practice_account_id if environment == 'practice' else live_account_id
            
            if st.button("🔌 Test Connection", use_container_width=True):
                if not api_key_to_test:
                    st.warning("Please enter API key first")
                else:
                    with st.spinner("Testing connection..."):
                        success, message = test_oanda_connection(
                            environment, api_key_to_test, account_id_to_test
                        )
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
    
    # ========================================================================
    # SECTION 2: FILE PATH MANAGEMENT
    # ========================================================================
    with st.expander("📁 File Locations", expanded=True):
        st.markdown("### Data Directories")
        
        csv_path = st.text_input(
            "📄 CSV Files Directory",
            value=st.session_state.user_csv_path,
            placeholder="/path/to/csv/files/",
            help="Directory containing Forex CSV files"
        )
        if csv_path != st.session_state.user_csv_path:
            st.session_state.user_csv_path = csv_path
            set_user_paths(csv_path=csv_path)
        
        csv_valid, csv_msg = validate_path(csv_path, "directory")
        if csv_valid:
            st.success(f"✅ {csv_msg}")
        elif csv_path:
            st.warning(f"⚠️ {csv_msg}")
        
        parquet_path = st.text_input(
            "📊 Parquet Files Directory",
            value=st.session_state.user_parquet_path,
            placeholder="/path/to/parquet/files/",
            help="Directory containing Parquet files (10x faster)"
        )
        if parquet_path != st.session_state.user_parquet_path:
            st.session_state.user_parquet_path = parquet_path
            set_user_paths(parquet_path=parquet_path)
        
        parquet_valid, parquet_msg = validate_path(parquet_path, "directory")
        if parquet_valid:
            st.success(f"✅ {parquet_msg}")
            parquet_pairs = get_available_pairs_with_parquet(parquet_path)
            if parquet_pairs:
                st.caption(f"Found {len(parquet_pairs)} pairs with Parquet")
        elif parquet_path:
            st.warning(f"⚠️ {parquet_msg}")
        
        st.markdown("### Model Directory")
        
        model_path = st.text_input(
            "🤖 Trained Models Directory",
            value=st.session_state.user_model_path,
            placeholder="/path/to/trained/models/",
            help="Directory containing .pkl model files"
        )
        if model_path != st.session_state.user_model_path:
            st.session_state.user_model_path = model_path
            set_user_paths(model_path=model_path)
        
        model_valid, model_msg = validate_path(model_path, "directory")
        if model_valid:
            st.success(f"✅ {model_msg}")
            models = get_available_models(model_path)
            if models:
                st.caption(f"Found {len(models)} trained models")
        elif model_path:
            st.warning(f"⚠️ {model_msg}")
    
    # ========================================================================
    # SECTION 3: SYSTEM INITIALIZATION
    # ========================================================================
    st.markdown("---")
    
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            if initialize_system():
                st.rerun()
    else:
        st.success("✅ System Ready")
        
        if AI_AVAILABLE and st.session_state.ai_accelerator:
            if st.session_state.ai_accelerator.has_gpu:
                st.markdown("""
                <div class="gpu-status">
                    🚀 GPU ACCELERATED
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="cpu-status">
                    💻 CPU MODE
                </div>
                """, unsafe_allow_html=True)
        
        parquet_pairs = get_available_pairs_with_parquet(st.session_state.user_parquet_path)
        if parquet_pairs:
            st.markdown(f"""
            <div class="parquet-status">
                📁 Parquet Ready: {len(parquet_pairs)} pairs
            </div>
            """, unsafe_allow_html=True)
        
        if st.session_state.oanda_trader and st.session_state.account_summary:
            account = st.session_state.account_summary.get('account', {})
            balance = float(account.get('balance', 0))
            st.metric("Account Balance", f"${balance:,.2f}")
        
        st.markdown("---")
        
        st.markdown("### 🤖 AI Settings")
        ai_enabled = st.checkbox("Enable AI", value=st.session_state.ai_enabled, disabled=not AI_AVAILABLE)
        if ai_enabled != st.session_state.ai_enabled:
            st.session_state.ai_enabled = ai_enabled
            st.rerun()
        
        if st.session_state.ai_enabled and st.session_state.model_manager:
            models_df = st.session_state.model_manager.get_latest_models_summary()
            if not models_df.empty:
                st.caption(f"📊 {len(models_df)} models available")
        
        st.markdown("---")
        
        st.markdown("### 📊 Data Settings")
        
        parquet_available = len(parquet_pairs) > 0
        use_parquet = st.checkbox(
            "🚀 Use Parquet (10x faster)", 
            value=st.session_state.use_parquet and parquet_available,
            disabled=not parquet_available
        )
        if use_parquet != st.session_state.use_parquet:
            st.session_state.use_parquet = use_parquet
            st.rerun()
        
        valid_pairs = VALID_OANDA_PAIRS
        if parquet_pairs:
            valid_pairs = parquet_pairs
        
        selected_pairs = st.multiselect(
            "Select Pairs",
            valid_pairs,
            default=valid_pairs[:5] if len(valid_pairs) >= 5 else valid_pairs,
            max_selections=10
        )
        st.session_state.selected_pairs = selected_pairs
        
        col1, col2 = st.columns(2)
        with col1:
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d", "30m", "15m", "5m"], index=0)
        with col2:
            bars = st.number_input("Bars", 50, 500, 200, 50)
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_refresh = datetime.now() - timedelta(seconds=60)
            st.rerun()
        
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Market Scanner", "Open Positions", "Trade History", "AI Training", "AI Models"],
            index=0
        )


# ============================================================================
# MAIN CONTENT
# ============================================================================
if not st.session_state.initialized:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to Sid Naiman SID Method Trading System</h2>
        <p style="font-size: 1.2rem; margin: 2rem 0;">
            📈 SID Method - RSI/MACD Reversal Trading<br>
            ⚡ Real-time Market Analysis with Parquet Acceleration<br>
            💰 Capital Preservation First (0.5-2% Risk Per Trade)
        </p>
        <p>👈 Configure OANDA API keys and file paths in sidebar, then click Initialize System</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if page == "Market Scanner":
        st.markdown('<h2 class="sub-header">🔍 Market Scanner (SID Method)</h2>', unsafe_allow_html=True)
        
        check_daily_reset()
        
        time_since = (datetime.now() - st.session_state.last_refresh).seconds
        if time_since > 60 or not st.session_state.current_data:
            with st.spinner("🔍 Scanning markets..."):
                if st.session_state.oanda_client:
                    st.session_state.current_data = fetch_market_data(
                        st.session_state.selected_pairs,
                        st.session_state.oanda_client,
                        timeframe,
                        bars
                    )
                else:
                    st.warning("OANDA client not available. Using cached data if available.")
                
                if st.session_state.current_data:
                    st.session_state.detected_signals = detect_sid_signals_with_ai(
                        st.session_state.current_data,
                        st.session_state.sid_method,
                        st.session_state.ai_enabled,
                        st.session_state.signal_predictor
                    )
                
                st.session_state.last_refresh = datetime.now()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pairs", len(st.session_state.current_data))
        with col2:
            st.metric("Signals", len(st.session_state.detected_signals))
        with col3:
            st.caption(f"Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        display_signals(st.session_state.detected_signals)
        
        for key in st.session_state:
            if key.startswith('confirm_signal_') and st.session_state[key]:
                render_trade_confirmation(
                    st.session_state[key],
                    st.session_state.oanda_trader,
                    st.session_state.account_summary
                )
                break
    
    elif page == "Open Positions":
        st.markdown('<h2 class="sub-header">📊 Open Positions</h2>', unsafe_allow_html=True)
        if st.session_state.oanda_trader:
            display_open_positions(st.session_state.oanda_trader)
        else:
            st.warning("OANDA trader not available")
    
    elif page == "Trade History":
        st.markdown('<h2 class="sub-header">📝 Trade History</h2>', unsafe_allow_html=True)
        if st.session_state.oanda_trader:
            display_trade_history(st.session_state.oanda_trader)
        else:
            st.warning("OANDA trader not available")
    
    elif page == "AI Training":
        display_ai_training()
    
    elif page == "AI Models":
        display_model_management()


# ============================================================================
# Footer
# ============================================================================
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Sid Naiman SID Method Trading System | v6.0 - DYNAMIC FILE SELECTION + OANDA API KEYS<br>
        Core Rules: RSI &lt;30 Oversold / &gt;70 Overbought | MACD Alignment | RSI 50 Take Profit<br>
        Stop Loss: Lowest Low Rounded Down / Highest High Rounded Up<br>
        Risk: 0.5-2% Per Trade<br>
        ⚠️ Risk Warning: Trading carries substantial risk - Always use proper position sizing
    </div>
    """,
    unsafe_allow_html=True
)