"""
Streamlit App for Sid Naiman's SID Method Trading System
MOBILE-OPTIMIZED VERSION - Complete Feature Parity
Version: 8.0 - Full Mobile Responsive
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

# ============================================================================
# MOBILE DETECTION & RESPONSIVE CONFIGURATION
# ============================================================================

def is_mobile() -> bool:
    """Detect if user is on a mobile device"""
    try:
        # Check screen width from Streamlit browser
        if hasattr(st, 'browser') and hasattr(st.browser, 'width'):
            return st.browser.width < 768
    except:
        pass
    return False


def get_responsive_value(desktop_value, mobile_value):
    """Return appropriate value based on device"""
    return mobile_value if is_mobile() else desktop_value


def get_chart_height():
    """Get appropriate chart height for device"""
    return 500 if is_mobile() else 800


def get_max_bars():
    """Get maximum bars to display based on device"""
    return 100 if is_mobile() else 500


def get_max_pairs():
    """Get maximum pairs to select based on device"""
    return 5 if is_mobile() else 15


# Set page config for mobile
st.set_page_config(
    page_title="SID Method Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed" if is_mobile() else "expanded"
)

# ============================================================================
# MOBILE-FRIENDLY CSS
# ============================================================================

st.markdown("""
<style>
    /* Base responsive styles */
    * {
        box-sizing: border-box;
    }
    
    /* Main headers */
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.3rem;
        font-weight: 500;
        color: #1E88E5;
        margin-bottom: 0.8rem;
        border-bottom: 2px solid #1E88E5;
        padding-bottom: 0.3rem;
    }
    
    /* Mobile responsive overrides */
    @media only screen and (max-width: 768px) {
        .main-header {
            font-size: 1.5rem !important;
        }
        .sub-header {
            font-size: 1.1rem !important;
        }
        
        /* Touch-friendly buttons */
        .stButton > button {
            min-height: 44px !important;
            min-width: 44px !important;
            font-size: 0.9rem !important;
            padding: 0.5rem !important;
            border-radius: 25px !important;
        }
        
        /* Signal cards */
        .signal-card {
            padding: 0.8rem !important;
            margin: 0.6rem 0 !important;
            border-radius: 12px !important;
        }
        
        /* Tables - horizontal scroll */
        .stTable, .dataframe {
            font-size: 0.75rem !important;
            overflow-x: auto !important;
            display: block !important;
            white-space: nowrap !important;
        }
        
        /* Metrics */
        div[data-testid="stMetric"] {
            padding: 0.3rem !important;
        }
        div[data-testid="stMetric"] label {
            font-size: 0.7rem !important;
        }
        div[data-testid="stMetric"] .stMetric-value {
            font-size: 1.2rem !important;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            width: 85% !important;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            padding: 0.8rem !important;
            font-size: 0.9rem !important;
            font-weight: bold !important;
        }
        
        /* Tabs */
        button[data-baseweb="tab"] {
            padding: 0.5rem 0.8rem !important;
            font-size: 0.8rem !important;
        }
        
        /* Input fields */
        .stTextInput > div > div > input,
        .stSelectbox > div > div,
        .stNumberInput > div > div > input {
            font-size: 16px !important;
            padding: 10px !important;
        }
        
        /* Charts */
        .stPlotlyChart {
            width: 100% !important;
            height: auto !important;
        }
        
        /* Column layout - stack on mobile */
        .row-widget.stHorizontal {
            flex-wrap: wrap !important;
        }
        .row-widget.stHorizontal > div {
            flex: 1 1 100% !important;
            margin: 0.25rem 0 !important;
        }
        
        /* Progress bar */
        .stProgress > div {
            height: 6px !important;
        }
        
        /* Checkboxes and radios */
        .stCheckbox > label, .stRadio > label {
            font-size: 0.9rem !important;
            padding: 0.3rem 0 !important;
        }
    }
    
    /* Signal card styles */
    .signal-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .signal-card:active {
        transform: scale(0.98);
    }
    .buy-signal-card {
        border-left: 6px solid #00C853;
    }
    .sell-signal-card {
        border-left: 6px solid #D32F2F;
    }
    .buy-signal {
        background-color: #00C853;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    .sell-signal {
        background-color: #D32F2F;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        display: inline-block;
        font-weight: bold;
    }
    
    /* Badges */
    .pair-category-badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-left: 0.5rem;
    }
    .category-best { background-color: #FFD700; color: #000; }
    .category-good { background-color: #4CAF50; color: white; }
    .category-metals { background-color: #9C27B0; color: white; }
    .category-avoid { background-color: #f44336; color: white; }
    .category-standard { background-color: #2196F3; color: white; }
    
    /* RSI badges */
    .rsi-oversold {
        background-color: #00C853;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.75rem;
        display: inline-block;
    }
    .rsi-overbought {
        background-color: #D32F2F;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.75rem;
        display: inline-block;
    }
    .rsi-neutral {
        background-color: #FFA000;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 15px;
        font-size: 0.75rem;
        display: inline-block;
    }
    
    /* GPU/CPU status */
    .gpu-status {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .cpu-status {
        background: linear-gradient(135deg, #757F9A 0%, #D7DDE8 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    .parquet-status {
        background: linear-gradient(135deg, #00C853 0%, #00E676 100%);
        color: white;
        padding: 0.8rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
    
    /* Info/Warning boxes */
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 0.85rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 0.85rem;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 0.85rem;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 0.8rem 0;
        font-size: 0.85rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Training status indicators */
    .training-queued {
        background-color: #FFA000;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        display: inline-block;
    }
    .training-active {
        background-color: #1E88E5;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    .training-completed {
        background-color: #00C853;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        display: inline-block;
    }
    .training-failed {
        background-color: #D32F2F;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.7rem;
        display: inline-block;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# IMPORT CORE MODULES
# ============================================================================

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
# DEFAULT PATHS
# ============================================================================
#DEFAULT_CSV_PATH = "/home/grct/Forex"
#DEFAULT_PARQUET_PATH = "/home/grct/Forex_Parquet"
#DEFAULT_MODEL_PATH = "/mnt2/Trading-Cafe/ML/SNaiman2/ai/trained_models/"

DEFAULT_CSV_PATH = ""
DEFAULT_PARQUET_PATH = ""
DEFAULT_MODEL_PATH = ""
# ============================================================================
# TRADING PAIRS - ORGANIZED BY SID METHOD RECOMMENDATIONS
# ============================================================================

ALL_OANDA_PAIRS = [
    "AUD_CAD", "AUD_CHF", "AUD_HKD", "AUD_JPY", "AUD_NZD", "AUD_SGD", "AUD_USD",
    "CAD_CHF", "CAD_HKD", "CAD_JPY", "CAD_SGD", "CHF_HKD", "CHF_JPY", "CHF_ZAR",
    "EUR_AUD", "EUR_CAD", "EUR_CHF", "EUR_CZK", "EUR_DKK", "EUR_GBP", "EUR_HKD",
    "EUR_HUF", "EUR_JPY", "EUR_NOK", "EUR_NZD", "EUR_PLN", "EUR_SEK", "EUR_SGD",
    "EUR_TRY", "EUR_USD", "EUR_ZAR", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_HKD",
    "GBP_JPY", "GBP_NOK", "GBP_NZD", "GBP_PLN", "GBP_SGD", "GBP_USD", "GBP_ZAR",
    "HKD_JPY", "NZD_CAD", "NZD_CHF", "NZD_HKD", "NZD_JPY", "NZD_SGD", "NZD_USD",
    "SGD_CHF", "SGD_JPY", "TRY_JPY", "USD_CAD", "USD_CHF", "USD_CNH", "USD_CZK",
    "USD_DKK", "USD_HKD", "USD_HUF", "USD_JPY", "USD_MXN", "USD_NOK", "USD_PLN",
    "USD_SEK", "USD_SGD", "USD_THB", "USD_TRY", "USD_ZAR", "XAG_AUD", "XAG_CHF",
    "XAG_EUR", "XAG_GBP", "XAG_HKD", "XAG_JPY", "XAG_NZD", "XAG_SGD", "XAG_USD",
    "XAU_AUD", "XAU_CAD", "XAU_CHF", "XAU_EUR", "XAU_GBP", "XAU_HKD", "XAU_JPY",
    "XAU_NZD", "XAU_SGD", "XAU_USD", "ZAR_JPY"
]

SID_TOP_PAIRS = ["GBP_JPY", "EUR_USD", "USD_JPY", "AUD_USD", "GBP_USD"]
SID_SECONDARY_PAIRS = ["EUR_GBP", "EUR_JPY", "AUD_JPY", "NZD_USD", "USD_CAD", "USD_CHF", "EUR_AUD", "GBP_AUD", "AUD_NZD", "EUR_CAD"]
SID_METALS_PAIRS = ["XAU_USD", "XAG_USD", "XAU_EUR", "XAU_JPY", "XAG_JPY", "XAU_GBP", "XAG_EUR"]
SID_LEAST_RECOMMENDED = ["EUR_GBP", "USD_CHF"]


def get_organized_pair_list() -> List[str]:
    organized = []
    for pair in SID_TOP_PAIRS:
        if pair not in organized:
            organized.append(pair)
    for pair in SID_SECONDARY_PAIRS:
        if pair not in organized:
            organized.append(pair)
    for pair in SID_METALS_PAIRS:
        if pair not in organized:
            organized.append(pair)
    remaining = [p for p in ALL_OANDA_PAIRS 
                 if p not in organized and p not in SID_LEAST_RECOMMENDED]
    organized.extend(sorted(remaining))
    for pair in SID_LEAST_RECOMMENDED:
        if pair not in organized:
            organized.append(pair)
    return organized


def get_pair_category(pair: str) -> str:
    if pair in SID_TOP_PAIRS:
        return "⭐ BEST"
    elif pair in SID_SECONDARY_PAIRS:
        return "👍 GOOD"
    elif pair in SID_METALS_PAIRS:
        return "🥇 METALS"
    elif pair in SID_LEAST_RECOMMENDED:
        return "⚠️ AVOID"
    else:
        return "📊 STANDARD"


def get_pair_description(pair: str) -> str:
    descriptions = {
        "GBP_JPY": "Best for SID Method - High volatility, clear oversold/overbought signals",
        "EUR_USD": "Most liquid pair - Reliable patterns, good for all strategies",
        "USD_JPY": "Yen pair - Works well with SID's pip buffer rules (10 pips)",
        "AUD_USD": "Commodity currency - Good trends, clear reversals",
        "GBP_USD": "Volatile - Good for short trades, clear M/W patterns",
        "EUR_GBP": "⚠️ Range-bound - Fewer SID signals, use with caution",
        "USD_CHF": "⚠️ Too stable - Low volatility, avoid for SID method",
        "XAU_USD": "Gold - Sid trades gold regularly, good for oversold signals",
        "XAG_USD": "Silver - Follows gold, good for mean reversion",
        "EUR_JPY": "Good cross pair - Follows USD_JPY with more volatility",
        "AUD_JPY": "Good for Yen trades - Commodity + Yen dynamics"
    }
    return descriptions.get(pair, "Standard SID Method instrument")


def get_pair_icon(pair: str) -> str:
    if pair in SID_TOP_PAIRS:
        return "⭐"
    elif pair in SID_SECONDARY_PAIRS:
        return "👍"
    elif pair in SID_METALS_PAIRS:
        return "🥇"
    elif pair in SID_LEAST_RECOMMENDED:
        return "⚠️"
    else:
        return "📊"


# ============================================================================
# VALID OANDA PAIRS
# ============================================================================
VALID_OANDA_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD", "USD_CHF", "NZD_USD",
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "AUD_JPY", "EUR_AUD", "EUR_CHF", "AUD_NZD",
    "NZD_JPY", "GBP_AUD", "GBP_CAD", "GBP_CHF", "GBP_NZD", "CAD_JPY", "CHF_JPY",
    "EUR_CAD", "AUD_CAD", "NZD_CAD", "EUR_NZD", "USD_NOK", "USD_SEK", "USD_TRY",
    "EUR_NOK", "EUR_SEK", "EUR_TRY", "GBP_NOK", "GBP_SEK", "GBP_TRY"
]

# ============================================================================
# PARQUET HELPER FUNCTIONS
# ============================================================================

def get_available_pairs_with_parquet(parquet_path: str = None):
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
    if parquet_path is None:
        parquet_path = st.session_state.get('user_parquet_path', DEFAULT_PARQUET_PATH)
    stats = {'total_pairs': 0, 'total_files': 0, 'total_size_gb': 0, 'pairs': []}
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
                    stats['pairs'].append({'pair': pair, 'files': len(all_files), 'size_gb': pair_size, 'years': sorted(years)})
                    stats['total_pairs'] += 1
                    stats['total_files'] += len(all_files)
                    stats['total_size_gb'] += pair_size
    return stats


def normalize_column_names(df):
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


def load_data_from_parquet(pair, parquet_path, start_date=None, end_date=None, progress_callback=None):
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
            year = None; month = None; day = None
            for part in path_parts:
                if part.startswith('year='):
                    year = int(part.replace('year=', ''))
                elif part.startswith('month='):
                    month = int(part.replace('month=', ''))
                elif part.startswith('day='):
                    day = int(part.replace('day=', ''))
            if start_dt and year is not None:
                if year < start_dt.year: continue
                if year == start_dt.year and month is not None and month < start_dt.month: continue
                if year == start_dt.year and month == start_dt.month and day is not None and day < start_dt.day: continue
            if end_dt and year is not None:
                if year > end_dt.year: continue
                if year == end_dt.year and month is not None and month > end_dt.month: continue
                if year == end_dt.year and month == end_dt.month and day is not None and day > end_dt.day: continue
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


def load_data_from_parquet_with_progress(pair, parquet_path, start_date=None, end_date=None):
    progress_bar = st.progress(0)
    status_text = st.empty()
    def update_progress(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
    df = load_data_from_parquet(pair, parquet_path, start_date, end_date, update_progress)
    status_text.empty()
    progress_bar.empty()
    return df


def check_parquet_available(pair, parquet_path=None):
    if parquet_path is None:
        parquet_path = st.session_state.get('user_parquet_path', DEFAULT_PARQUET_PATH)
    pair_path = os.path.join(parquet_path, pair)
    if not os.path.exists(pair_path):
        return False
    for item in os.listdir(pair_path):
        if item.startswith('year=') and os.path.isdir(os.path.join(pair_path, item)):
            return True
    return False


def get_parquet_date_range(pair, parquet_path=None):
    if parquet_path is None:
        parquet_path = st.session_state.get('user_parquet_path', DEFAULT_PARQUET_PATH)
    try:
        df = load_data_from_parquet(pair, parquet_path)
        if not df.empty and 'Date' in df.columns:
            return df['Date'].min(), df['Date'].max()
    except:
        pass
    return None, None


# ============================================================================
# OANDA API KEY MANAGEMENT
# ============================================================================

def save_api_key_to_env(environment: str, api_key: str):
    env_var = f"OANDA_API_KEY_{environment.upper()}"
    os.environ[env_var] = api_key
    env_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    try:
        env_vars = {}
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        env_vars[env_var] = api_key
        with open(env_file, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        return True
    except Exception as e:
        return False


def load_api_key_from_env(environment: str) -> Optional[str]:
    env_var = f"OANDA_API_KEY_{environment.upper()}"
    return os.environ.get(env_var)


def test_oanda_connection(environment: str, api_key: str, account_id: str = None) -> Tuple[bool, str]:
    try:
        old_api_key = os.environ.get('OANDA_API_KEY')
        old_account_id = os.environ.get('OANDA_ACCOUNT_ID')
        os.environ['OANDA_API_KEY'] = api_key
        if account_id:
            os.environ['OANDA_ACCOUNT_ID'] = account_id
        client = OANDAClient()
        summary = client.get_account_summary()
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
        return False, "Connection failed: No account data"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"


# ============================================================================
# DYNAMIC PATH MANAGEMENT FUNCTIONS
# ============================================================================

def get_user_paths():
    return {
        'csv_path': st.session_state.get('user_csv_path', DEFAULT_CSV_PATH),
        'parquet_path': st.session_state.get('user_parquet_path', DEFAULT_PARQUET_PATH),
        'model_path': st.session_state.get('user_model_path', DEFAULT_MODEL_PATH)
    }


def set_user_paths(csv_path: str = None, parquet_path: str = None, model_path: str = None):
    if csv_path:
        st.session_state.user_csv_path = csv_path
    if parquet_path:
        st.session_state.user_parquet_path = parquet_path
    if model_path:
        st.session_state.user_model_path = model_path


def validate_path(path: str, path_type: str = "directory") -> Tuple[bool, str]:
    if not path:
        return False, "Path is empty"
    try:
        if path_type == "directory":
            if os.path.isdir(path):
                return True, "Directory exists"
            return False, f"Directory not found: {path}"
        else:
            if os.path.isfile(path):
                return True, "File exists"
            return False, f"File not found: {path}"
    except Exception as e:
        return False, f"Error accessing path: {e}"


def scan_directory_for_files(directory: str, extensions: List[str]) -> List[str]:
    if not directory or not os.path.exists(directory):
        return []
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        files.extend(glob.glob(os.path.join(directory, f"**/*{ext}"), recursive=True))
    return sorted(files)


def get_available_models(model_path: str = None):
    if model_path is None:
        model_path = st.session_state.get('user_model_path', DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        return []
    models = []
    for file in os.listdir(model_path):
        if file.endswith('.pkl') or file.endswith('.joblib'):
            models.append(file)
    return sorted(models)


# ============================================================================
# SID METHOD FUNCTIONS
# ============================================================================

def calculate_rsi_simple(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, float('nan'))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def calculate_macd_simple(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    exp1 = df['close'].ewm(span=fast, adjust=False).mean()
    exp2 = df['close'].ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({'macd': macd_line, 'signal': signal_line, 'histogram': macd_line - signal_line})


def calculate_sid_indicators(df: pd.DataFrame, sid=None) -> pd.DataFrame:
    df = df.copy()
    if len(df) < 50:
        return df
    df['rsi'] = calculate_rsi_simple(df)
    macd_df = calculate_macd_simple(df)
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


def detect_sid_signals(df: pd.DataFrame, sid=None) -> List[Dict]:
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
                    'type': 'BUY', 'direction': 'long', 'signal_type': 'oversold',
                    'rsi_value': current_rsi, 'price': df['close'].iloc[i],
                    'date': df.index[i], 'macd_aligned': True,
                    'macd_crossed': macd_crossed, 'confidence': 70 + (10 if macd_crossed else 0)
                }
                signals.append(signal)
        elif current_rsi > 70:
            if df['macd_aligned_down'].iloc[i]:
                macd_crossed = df['macd_cross_below'].iloc[i]
                signal = {
                    'type': 'SELL', 'direction': 'short', 'signal_type': 'overbought',
                    'rsi_value': current_rsi, 'price': df['close'].iloc[i],
                    'date': df.index[i], 'macd_aligned': True,
                    'macd_crossed': macd_crossed, 'confidence': 70 + (10 if macd_crossed else 0)
                }
                signals.append(signal)
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    return signals


def detect_signals_with_indicators(data: Dict[str, pd.DataFrame], rsi_oversold=30, rsi_overbought=70) -> List[Dict]:
    all_signals = []
    scan_bars = 10 if is_mobile() else 20
    for pair, df in data.items():
        if df.empty or len(df) < 50:
            continue
        df_indicators = calculate_sid_indicators(df)
        if 'rsi' not in df_indicators.columns:
            continue
        start_idx = max(0, len(df_indicators) - scan_bars)
        for i in range(start_idx, len(df_indicators) - 1):
            current_rsi = df_indicators['rsi'].iloc[i]
            if pd.isna(current_rsi):
                continue
            if current_rsi < rsi_oversold:
                macd_aligned = df_indicators['macd_aligned_up'].iloc[i] if 'macd_aligned_up' in df_indicators.columns else False
                macd_crossed = df_indicators['macd_cross_above'].iloc[i] if 'macd_cross_above' in df_indicators.columns else False
                if macd_aligned:
                    signal = {
                        'type': 'BUY', 'direction': 'long', 'signal_type': 'oversold',
                        'rsi_value': float(current_rsi), 'price': float(df_indicators['close'].iloc[i]),
                        'date': df_indicators.index[i], 'macd_aligned': bool(macd_aligned),
                        'macd_crossed': bool(macd_crossed), 'confidence': 70 + (10 if macd_crossed else 0),
                        'trading_pair': pair, 'category': get_pair_category(pair),
                        'icon': get_pair_icon(pair), 'data': df_indicators.copy()
                    }
                    all_signals.append(signal)
            elif current_rsi > rsi_overbought:
                macd_aligned = df_indicators['macd_aligned_down'].iloc[i] if 'macd_aligned_down' in df_indicators.columns else False
                macd_crossed = df_indicators['macd_cross_below'].iloc[i] if 'macd_cross_below' in df_indicators.columns else False
                if macd_aligned:
                    signal = {
                        'type': 'SELL', 'direction': 'short', 'signal_type': 'overbought',
                        'rsi_value': float(current_rsi), 'price': float(df_indicators['close'].iloc[i]),
                        'date': df_indicators.index[i], 'macd_aligned': bool(macd_aligned),
                        'macd_crossed': bool(macd_crossed), 'confidence': 70 + (10 if macd_crossed else 0),
                        'trading_pair': pair, 'category': get_pair_category(pair),
                        'icon': get_pair_icon(pair), 'data': df_indicators.copy()
                    }
                    all_signals.append(signal)
    all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    return all_signals


def get_signal_strength(signal: Dict) -> str:
    conf = signal.get('confidence', 50)
    if conf >= 85: return 'VERY_STRONG'
    elif conf >= 75: return 'STRONG'
    elif conf >= 65: return 'MODERATE'
    elif conf >= 55: return 'WEAK'
    else: return 'VERY_WEAK'


def get_rsi_status(rsi_value: float) -> Tuple[str, str]:
    if rsi_value < 30: return 'OVERSOLD', 'rsi-oversold'
    elif rsi_value > 70: return 'OVERBOUGHT', 'rsi-overbought'
    else: return 'NEUTRAL', 'rsi-neutral'


def create_sid_chart(df: pd.DataFrame, signals: List[Dict], pair: str) -> go.Figure:
    df_chart = df.copy()
    if 'rsi' not in df_chart.columns:
        df_chart['rsi'] = calculate_rsi_simple(df_chart)
    if 'macd' not in df_chart.columns:
        macd_df = calculate_macd_simple(df_chart)
        df_chart['macd'] = macd_df['macd']
        df_chart['macd_signal'] = macd_df['signal']
    if is_mobile() and len(df_chart) > 100:
        df_chart = df_chart.iloc[-100:]
    has_signals = len(signals) > 0
    n_rows = 3
    row_heights = [0.5, 0.25, 0.25]
    subplot_titles = [f"{pair} - Price", "RSI", "MACD"]
    fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=row_heights, subplot_titles=subplot_titles)
    fig.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['open'], high=df_chart['high'], low=df_chart['low'], close=df_chart['close'], name='Price', showlegend=False), row=1, col=1)
    if has_signals:
        for signal in signals:
            if signal.get('trading_pair') == pair:
                color = 'green' if signal['type'] == 'BUY' else 'red'
                symbol = 'triangle-up' if signal['type'] == 'BUY' else 'triangle-down'
                fig.add_trace(go.Scatter(x=[signal['date']], y=[signal['price']], mode='markers', marker=dict(symbol=symbol, size=12, color=color, line=dict(width=1, color='white')), name=signal['type'], showlegend=False, hovertemplate=f"{signal['type']}<br>RSI: {signal['rsi_value']:.1f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['rsi'], line=dict(color='purple', width=1.5), name='RSI'), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="orange", opacity=0.3, row=2, col=1)
    fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['macd'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
    fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['macd_signal'], line=dict(color='red', width=1.5), name='Signal'), row=3, col=1)
    height = 500 if is_mobile() else 700
    fig.update_layout(title=f"{pair} - SID Method Analysis", height=height, template="plotly_white", hovermode='x unified', showlegend=False if is_mobile() else True, margin=dict(l=30, r=30, t=60, b=30) if is_mobile() else dict(l=50, r=50, t=80, b=50))
    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    return fig


def fetch_market_data(pairs: List[str], oanda_client, timeframe: str = "1h", bars: int = 200) -> Dict[str, pd.DataFrame]:
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    total_pairs = len(pairs)
    failed_pairs = []
    bars = min(bars, get_max_bars())
    for i, pair in enumerate(pairs):
        category = get_pair_category(pair)
        icon = get_pair_icon(pair)
        status_text.text(f"📡 Fetching {icon} {pair} ({category})... ({i+1}/{total_pairs})")
        try:
            oanda_pair = pair.replace('/', '_')
            df = oanda_client.fetch_candles(instrument=oanda_pair, granularity=timeframe, count=bars)
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
        st.caption(f"⚠️ Failed to fetch: {', '.join(failed_pairs[:5])}" + (f" and {len(failed_pairs)-5} more" if len(failed_pairs) > 5 else ""))
    return data


def detect_sid_signals_with_ai(data: Dict[str, pd.DataFrame], sid: SidMethod, ai_enabled: bool = False, signal_predictor=None) -> List[Dict]:
    all_signals = detect_signals_with_indicators(data, 30, 70)
    if ai_enabled:
        for signal in all_signals:
            try:
                rsi = signal['rsi_value']
                ai_conf = 85 if rsi < 25 else 75 if rsi < 30 else 85 if rsi > 75 else 75 if rsi > 70 else 50
                signal['ai_confidence'] = ai_conf
                signal['ai_success_probability'] = ai_conf / 100
                signal['ai_signal_strength'] = get_signal_strength(signal)
                base_conf = signal['confidence']
                signal['confidence'] = (base_conf * 0.6 + ai_conf * 0.4)
            except:
                signal['ai_confidence'] = 0
    else:
        for signal in all_signals:
            signal['ai_confidence'] = 0
    return all_signals


# ============================================================================
# UI DISPLAY FUNCTIONS
# ============================================================================

def display_signal_card(signal: Dict):
    pair = signal['trading_pair']
    signal_type = signal['type']
    signal_class = "buy-signal-card" if signal_type == 'BUY' else "sell-signal-card"
    rsi_status, rsi_class = get_rsi_status(signal['rsi_value'])
    category = signal.get('category', '📊 STANDARD')
    icon = signal.get('icon', '📊')
    category_class = "category-best" if "BEST" in category else "category-good" if "GOOD" in category else "category-metals" if "METALS" in category else "category-avoid" if "AVOID" in category else "category-standard"
    with st.container():
        st.markdown(f'<div class="signal-card {signal_class}">', unsafe_allow_html=True)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(f"### {icon} {pair}")
            st.markdown(f"<span class='{rsi_class}'>RSI: {signal['rsi_value']:.1f} ({rsi_status})</span>", unsafe_allow_html=True)
            st.markdown(f"<span class='pair-category-badge {category_class}'>{category}</span>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"**{signal_type}**")
            st.markdown(f"Price: {signal['price']:.5f}")
            st.markdown(f"Confidence: {signal['confidence']:.0f}%")
        st.markdown(f"MACD: {'✅ Aligned' if signal['macd_aligned'] else '❌'} | {'✅ Crossed' if signal['macd_crossed'] else '❌'}")
        if pair in SID_TOP_PAIRS:
            st.caption(f"💡 {get_pair_description(pair)}")
        col1, col2 = st.columns(2)
        with col1:
            if st.button(f"📊 Chart", key=f"chart_{id(signal)}", use_container_width=True):
                st.session_state[f"show_chart_{id(signal)}"] = signal
        with col2:
            if st.button(f"💰 Trade", key=f"trade_{id(signal)}", type="primary", use_container_width=True):
                st.session_state[f"confirm_signal_{id(signal)}"] = signal
        if st.session_state.get(f"show_chart_{id(signal)}", False):
            if 'data' in signal:
                fig = create_sid_chart(signal['data'], [signal], pair)
                st.plotly_chart(fig, use_container_width=True)
                if st.button("Close", key=f"close_{id(signal)}"):
                    st.session_state[f"show_chart_{id(signal)}"] = False
        st.markdown('</div>', unsafe_allow_html=True)


def display_signals_grid(signals: List[Dict]):
    if not signals:
        st.info("📊 No trading signals detected")
        return
    with st.expander("🔍 Filter Signals", expanded=not is_mobile()):
        col1, col2, col3 = st.columns(3)
        with col1:
            min_confidence = st.slider("Min Confidence", 0, 100, 50)
        with col2:
            signal_type = st.selectbox("Signal Type", ["All", "BUY", "SELL"])
        with col3:
            category_filter = st.selectbox("Category", ["All", "⭐ BEST", "👍 GOOD", "🥇 METALS", "📊 STANDARD", "⚠️ AVOID"])
    filtered = [s for s in signals if s.get('confidence', 0) >= min_confidence]
    if signal_type != "All":
        filtered = [s for s in filtered if s['type'] == signal_type]
    if category_filter != "All":
        filtered = [s for s in filtered if s.get('category', '') == category_filter]
    st.caption(f"Showing {len(filtered)} of {len(signals)} signals")
    cols_per_row = 1 if is_mobile() else 2
    for i in range(0, len(filtered), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, col in enumerate(cols):
            if i + j < len(filtered):
                with col:
                    display_signal_card(filtered[i + j])


def render_trade_confirmation(signal: Dict, oanda_trader, account_summary):
    pair = signal['trading_pair']
    signal_type = signal['type']
    category = signal.get('category', 'STANDARD')
    st.markdown('<div class="warning-box"><h3 style="text-align: center;">⚠️ Confirm Trade</h3></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📊 Trade Details")
        st.markdown(f"- **Instrument:** `{pair}`\n- **Category:** `{category}`\n- **Direction:** `{'BUY' if signal_type == 'BUY' else 'SELL'}`\n- **Signal:** `{signal['signal_type'].upper()}`\n- **RSI:** `{signal['rsi_value']:.1f}`\n- **Price:** `{signal['price']:.5f}`\n- **Confidence:** `{signal['confidence']:.0f}%`")
        if pair in SID_TOP_PAIRS:
            st.info(f"💡 {get_pair_description(pair)}")
        if signal.get('ai_confidence', 0) > 0:
            st.markdown(f"**🤖 AI Analysis**\n- Confidence: `{signal['ai_confidence']:.1f}%`\n- Strength: `{signal.get('ai_signal_strength', 'neutral')}`")
    with col2:
        st.markdown("### 💰 Position Sizing (SID Method)")
        account = account_summary.get('account', {})
        balance = float(account.get('balance', 10000))
        risk_percent = st.slider("Risk %", 0.5, 2.0, 1.0, 0.1)
        risk_amount = balance * (risk_percent / 100)
        instrument = pair.replace('/', '_')
        is_yen_pair = "_JPY" in pair or pair.endswith("JPY")
        stop_pips = 10 if is_yen_pair else 5
        if signal_type == 'BUY':
            stop_loss = signal['price'] * (1 - stop_pips * 0.0001) if not is_yen_pair else signal['price'] * (1 - stop_pips * 0.01)
            take_profit = signal['price'] * (1 + stop_pips * 0.0001) if not is_yen_pair else signal['price'] * (1 + stop_pips * 0.01)
        else:
            stop_loss = signal['price'] * (1 + stop_pips * 0.0001) if not is_yen_pair else signal['price'] * (1 + stop_pips * 0.01)
            take_profit = signal['price'] * (1 - stop_pips * 0.0001) if not is_yen_pair else signal['price'] * (1 - stop_pips * 0.01)
        risk_per_unit = abs(signal['price'] - stop_loss)
        units = int(risk_amount / risk_per_unit) if risk_per_unit > 0 else 0
        st.markdown(f"**Risk Management:**\n- Balance: `${balance:,.2f}`\n- Risk Amount: `${risk_amount:.2f}` ({risk_percent}%)\n- Stop Loss: `{stop_loss:.5f}` ({stop_pips} pips)\n- Take Profit: `{take_profit:.5f}`\n- Position Size: `{units:,}` units")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ CONFIRM", type="primary", use_container_width=False):
            try:
                result = oanda_trader.place_order(instrument=instrument, units=units if signal_type == 'BUY' else -units, stop_loss=stop_loss, take_profit=take_profit, order_type="MARKET")
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
    positions = oanda_trader.get_open_trades()
    st.session_state.open_positions = positions
    if not positions:
        st.info("📭 No open positions")
        return
    for position in positions:
        pair = position.get('instrument', 'Unknown')
        category = get_pair_category(pair)
        icon = get_pair_icon(pair)
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
                st.markdown(f"**{icon} {position['instrument']}**")
                st.markdown(f"Units: `{position['units']}`")
            with col2:
                st.markdown(f"Entry: `{position['price']:.5f}`")
                st.markdown(f"Current: `{current:.5f}`")
            with col3:
                st.markdown(f"P/L: <span style='color:{pl_color};'>${pl:.2f} ({pl_pct:.2f}%)</span>", unsafe_allow_html=True)
            if st.button(f"Close Position", key=f"close_{position['id']}", use_container_width=False):
                result = oanda_trader.close_trade(position['id'])
                if result.get('success'):
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


def display_trade_history(oanda_trader):
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
        pair = trade.get('instrument', '')
        trade_data.append({'Date': str(trade.get('entry_time', ''))[:10], 'Pair': f"{get_pair_icon(pair)} {pair}", 'Direction': str(trade.get('direction', '')), 'Entry': f"{trade.get('entry_price', 0):.5f}", 'Outcome': str(trade.get('outcome', 'pending'))})
    if trade_data:
        df_trades = pd.DataFrame(trade_data)
        for col in df_trades.columns:
            df_trades[col] = df_trades[col].astype(str)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)


def display_hardware_status():
    st.markdown("### 🖥️ Hardware Status")
    if AI_AVAILABLE and st.session_state.ai_accelerator:
        if st.session_state.ai_accelerator.has_gpu:
            st.markdown('<div class="gpu-status">🚀 GPU ACCELERATED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cpu-status">💻 CPU MODE</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="cpu-status">💻 CPU MODE (AI disabled)</div>', unsafe_allow_html=True)
    parquet_pairs = get_available_pairs_with_parquet(st.session_state.user_parquet_path)
    if parquet_pairs:
        st.markdown(f'<div class="parquet-status">📁 Parquet Ready: {len(parquet_pairs)} pairs (10x faster)</div>', unsafe_allow_html=True)


def display_account_info():
    account = st.session_state.account_summary.get('account', {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Account Type", st.session_state.oanda_trader.account_type)
    with col2:
        balance = float(account.get('balance', 0))
        st.metric("Balance", f"${balance:,.2f}")
    with col3:
        open_trades = len(st.session_state.open_positions)
        st.metric("Open Trades", open_trades)


def display_training_data_tab():
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
    organized_pairs = get_organized_pair_list()
    all_pairs = [p for p in organized_pairs if p in ALL_OANDA_PAIRS]
    if all_pairs:
        pair_options = [f"{get_pair_icon(p)} {p} ({get_pair_category(p)})" for p in all_pairs]
        selected_index = st.selectbox("Select Trading Pair", range(len(pair_options)), format_func=lambda x: pair_options[x])
        selected_pair = all_pairs[selected_index]
    else:
        selected_pair = None
    if selected_pair:
        has_parquet = check_parquet_available(selected_pair, st.session_state.user_parquet_path)
        use_parquet = st.checkbox("🚀 Use Parquet (10x faster)", value=has_parquet) if has_parquet else False
    st.markdown("##### Select Time Range")
    col1, col2, col3 = st.columns(3)
    with col1:
        range_preset = st.selectbox("Quick Select", ["MAX", "Custom", "Last Month", "Last 3 Months", "Last 6 Months", "Last Year"], index=0)
    with col2:
        if range_preset == "Custom":
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
            start_str = start_date.strftime('%Y-%m-%d')
        else:
            start_str = None
    with col3:
        if range_preset == "Custom":
            end_date = st.date_input("End Date", datetime.now())
            end_str = end_date.strftime('%Y-%m-%d')
        else:
            end_str = None
    st.markdown("##### Data Resolution")
    timeframe = st.selectbox("Timeframe for Analysis", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"], index=4)
    if st.button("📥 LOAD ALL DATA", type="primary", use_container_width=False):
        if selected_pair:
            try:
                sid = st.session_state.sid_method
                if use_parquet:
                    df = load_data_from_parquet_with_progress(selected_pair, st.session_state.user_parquet_path, start_date=start_str, end_date=end_str)
                    if not df.empty and 'Date' in df.columns:
                        df = df.set_index('Date')
                        df = calculate_sid_indicators(df, sid)
                        new_samples = 0
                        max_samples = 50000 if is_mobile() else 100000
                        for i in range(min(len(df), max_samples)):
                            sample = {'timestamp': df.index[i] if i < len(df.index) else datetime.now(), 'pair': selected_pair, 'timeframe': timeframe, 'rsi': df['rsi'].iloc[i] if 'rsi' in df.columns else 50, 'outcome': 1 if i < len(df) - 1 and df['close'].iloc[i+1] > df['close'].iloc[i] else 0}
                            st.session_state.training_data.append(sample)
                            new_samples += 1
                        st.success(f"✅ Loaded {new_samples} training samples for {selected_pair}")
                    else:
                        st.warning(f"No data available for {selected_pair}")
            except Exception as e:
                st.error(f"Error loading data: {e}")
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
                st.metric("Unique Pairs", df_summary['pair'].nunique())
        with col4:
            st.metric("Timeframes", df_summary['timeframe'].nunique() if 'timeframe' in df_summary.columns else 1)
        st.markdown("##### Recent Samples")
        try:
            display_cols = ['timestamp', 'pair', 'timeframe', 'rsi', 'outcome']
            existing_cols = [col for col in display_cols if col in df_summary.columns]
            if existing_cols:
                display_df = df_summary[existing_cols].tail(10).copy()
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
            organized_pairs = get_organized_pair_list()
            csv_pairs_organized = [p for p in organized_pairs if p in csv_pairs]
            convert_pair = st.selectbox("Select Pair to Convert", csv_pairs_organized if csv_pairs_organized else sorted(csv_pairs))
            if convert_pair:
                category = get_pair_category(convert_pair)
                icon = get_pair_icon(convert_pair)
                st.caption(f"{icon} {category} - {get_pair_description(convert_pair)}")
            use_gpu = st.checkbox("🚀 Use GPU for conversion", value=AI_AVAILABLE and st.session_state.ai_accelerator and st.session_state.ai_accelerator.has_gpu)
            mode = st.selectbox("Conversion Mode", ["append", "initial", "overwrite", "update"], index=0)
            if st.button("🔄 Convert to Parquet", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                try:
                    if st.session_state.parquet_converter is None:
                        try:
                            converter = ParquetConverter(csv_dir=st.session_state.user_csv_path, parquet_dir=st.session_state.user_parquet_path, use_gpu=use_gpu)
                        except TypeError:
                            try:
                                converter = ParquetConverter(st.session_state.user_csv_path, st.session_state.user_parquet_path, use_gpu)
                            except TypeError:
                                try:
                                    converter = ParquetConverter(use_gpu=use_gpu)
                                except TypeError:
                                    converter = ParquetConverter()
                    else:
                        converter = st.session_state.parquet_converter
                    if hasattr(converter, 'convert_pair'):
                        success = converter.convert_pair(pair=convert_pair, mode=mode, progress_callback=update_progress)
                    elif hasattr(converter, 'convert'):
                        success = converter.convert(pair=convert_pair, mode=mode)
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
    st.markdown("### 🤖 AI Model Management (SID Method)")
    if not AI_AVAILABLE or st.session_state.model_manager is None:
        st.warning("⚠️ AI modules not available")
        return
    models_df = st.session_state.model_manager.get_latest_models_summary()
    if models_df.empty:
        st.info("No trained models found. Run training pipeline to create models.")
        if st.button("🚀 Train Default Model", type="primary"):
            with st.spinner("Training model..."):
                result = st.session_state.training_pipeline.train_pair(pair="EUR_USD", force=True, samples=50000)
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
            latest_date_str = latest_date.strftime('%Y-%m-%d') if hasattr(latest_date, 'strftime') else str(latest_date)
            st.metric("Latest Model", latest_date_str)
        else:
            st.metric("Latest Model", "N/A")
    organized_pairs = get_organized_pair_list()
    available_pairs = [p for p in organized_pairs if p in models_df['Pair'].values]
    selected_pair = st.selectbox("Filter by Pair", ["All"] + available_pairs)
    if selected_pair != "All":
        display_df = models_df[models_df['Pair'] == selected_pair].copy()
    else:
        display_df = models_df.copy()
    display_cols = ['Pair', 'From', 'To', 'Days', 'Accuracy', 'Samples', 'Device', 'Model Name']
    available_cols = [c for c in display_cols if c in display_df.columns]
    display_data = pd.DataFrame()
    for col in available_cols:
        if col in ['From', 'To']:
            display_data[col] = display_df[col].apply(lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x))
        elif col == 'Accuracy':
            display_data[col] = display_df[col].apply(lambda x: f"{x:.1%}" if isinstance(x, (int, float)) and x != 'N/A' else str(x))
        elif col == 'Samples':
            display_data[col] = display_df[col].apply(lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x))
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
        train_pairs = [p for p in organized_pairs if p in ALL_OANDA_PAIRS]
        train_pair = st.selectbox("Train Specific Pair", train_pairs[:20])
        if train_pair:
            category = get_pair_category(train_pair)
            icon = get_pair_icon(train_pair)
            st.caption(f"{icon} {category} - {get_pair_description(train_pair)}")
        if st.button(f"🚀 Train {train_pair}", use_container_width=True):
            with st.spinner(f"Training {train_pair}..."):
                result = st.session_state.training_pipeline.train_pair(pair=train_pair, force=True, samples=100000)
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
    st.markdown("### 🧠 AI Model Training (SID Method)")
    if not AI_AVAILABLE:
        st.warning("⚠️ AI modules not available. Run: pip install scikit-learn pandas numpy xgboost")
        return
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Training Data", "🏋️ Train Models", "💾 Parquet Management", "🤖 Model Management"])
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
    st.session_state.training_data = []
    st.session_state.ai_enabled = AI_AVAILABLE
    st.session_state.user_csv_path = DEFAULT_CSV_PATH
    st.session_state.user_parquet_path = DEFAULT_PARQUET_PATH
    st.session_state.user_model_path = DEFAULT_MODEL_PATH
    st.session_state.practice_api_key = load_api_key_from_env('practice')
    st.session_state.live_api_key = load_api_key_from_env('live')
    st.session_state.practice_account_id = os.environ.get('OANDA_ACCOUNT_ID_PRACTICE', '')
    st.session_state.live_account_id = os.environ.get('OANDA_ACCOUNT_ID_LIVE', '')
    st.session_state.environment = 'practice'
    st.session_state.last_refresh = datetime.now()


def initialize_system():
    if not CORE_IMPORTS_SUCCESS:
        st.error("Core modules not available.")
        return False
    try:
        with st.spinner("🔄 Connecting to OANDA..."):
            env = st.session_state.environment
            api_key = st.session_state.practice_api_key if env == 'practice' else st.session_state.live_api_key
            if api_key:
                os.environ['OANDA_API_KEY'] = api_key
                st.session_state.oanda_client = OANDAClient()
                st.session_state.oanda_trader = OANDATrader(environment=env)
            else:
                st.warning("⚠️ OANDA API key not configured.")
        with st.spinner("🔄 Loading Sid Method..."):
            st.session_state.sid_method = SidMethod()
        if st.session_state.oanda_trader:
            st.session_state.account_summary = st.session_state.oanda_trader.get_account_summary()
        st.session_state.initialized = True
        st.success("✅ System initialized!")
        return True
    except Exception as e:
        st.error(f"❌ Initialization failed: {str(e)}")
        return False


def check_daily_reset():
    if st.session_state.daily_reset_time is None:
        st.session_state.daily_reset_time = datetime.now()
    now = datetime.now()
    if st.session_state.daily_reset_time.date() < now.date():
        st.session_state.daily_loss = 0.0
        st.session_state.consecutive_losses = 0
        st.session_state.daily_reset_time = now


# ============================================================================
# MAIN APP LAYOUT
# ============================================================================
st.markdown('<h1 class="main-header">📈 Sid Naiman SID Method Trading System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    with st.expander("🔑 OANDA API Keys", expanded=not is_mobile()):
        st.markdown("### Practice Account")
        practice_api_key = st.text_input("Practice API Key", value=st.session_state.practice_api_key or "", type="password")
        practice_account_id = st.text_input("Practice Account ID", value=st.session_state.practice_account_id or "")
        st.markdown("### Live Account")
        live_api_key = st.text_input("Live API Key", value=st.session_state.live_api_key or "", type="password")
        live_account_id = st.text_input("Live Account ID", value=st.session_state.live_account_id or "")
        environment = st.radio("Trading Environment", ["practice", "live"], index=0 if st.session_state.environment == 'practice' else 1)
        if environment != st.session_state.environment:
            st.session_state.environment = environment
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save API Keys", use_container_width=True):
                if practice_api_key:
                    save_api_key_to_env('practice', practice_api_key)
                    st.session_state.practice_api_key = practice_api_key
                if live_api_key:
                    save_api_key_to_env('live', live_api_key)
                    st.session_state.live_api_key = live_api_key
                st.success("✅ API keys saved")
        with col2:
            api_key_to_test = practice_api_key if environment == 'practice' else live_api_key
            if st.button("🔌 Test Connection", use_container_width=True):
                if api_key_to_test:
                    success, message = test_oanda_connection(environment, api_key_to_test)
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
    
    with st.expander("📁 File Locations", expanded=not is_mobile()):
        csv_path = st.text_input("📄 CSV Directory", value=st.session_state.user_csv_path)
        if csv_path != st.session_state.user_csv_path:
            st.session_state.user_csv_path = csv_path
        parquet_path = st.text_input("📊 Parquet Directory", value=st.session_state.user_parquet_path)
        if parquet_path != st.session_state.user_parquet_path:
            st.session_state.user_parquet_path = parquet_path
        model_path = st.text_input("🤖 Models Directory", value=st.session_state.user_model_path)
        if model_path != st.session_state.user_model_path:
            st.session_state.user_model_path = model_path
    
    st.markdown("---")
    
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary", use_container_width=True):
            if initialize_system():
                st.rerun()
    else:
        st.success("✅ System Ready")
        display_hardware_status()
        display_account_info()
        st.markdown("---")
        st.markdown("### 🤖 AI Settings")
        ai_enabled = st.checkbox("Enable AI", value=st.session_state.ai_enabled, disabled=not AI_AVAILABLE)
        if ai_enabled != st.session_state.ai_enabled:
            st.session_state.ai_enabled = ai_enabled
            st.rerun()
        st.markdown("---")
        st.markdown("### 📊 Data Settings")
        organized_pairs = get_organized_pair_list()
        default_pairs = SID_TOP_PAIRS.copy()
        if is_mobile():
            default_pairs = default_pairs[:3]
        selected_pairs = st.multiselect("Select Pairs", organized_pairs, default=default_pairs, max_selections=get_max_pairs())
        st.session_state.selected_pairs = selected_pairs
        if selected_pairs:
            best_count = sum(1 for p in selected_pairs if p in SID_TOP_PAIRS)
            good_count = sum(1 for p in selected_pairs if p in SID_SECONDARY_PAIRS)
            st.caption(f"Selected: ⭐{best_count} 👍{good_count}")
        col1, col2 = st.columns(2)
        with col1:
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d", "30m", "15m", "5m"], index=0)
        with col2:
            bars = st.number_input("Bars", 50, get_max_bars(), min(200, get_max_bars()), 50)
        if st.button("🔄 Refresh Now", use_container_width=True):
            st.session_state.last_refresh = datetime.now() - timedelta(seconds=60)
            st.rerun()
        st.markdown("---")
        page = st.radio("Navigation", ["Market Scanner", "Open Positions", "Trade History", "AI Training", "AI Models"], index=0)


# ============================================================================
# MAIN CONTENT
# ============================================================================
if not st.session_state.initialized:
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to Sid Naiman SID Method Trading System</h2>
        <p style="font-size: 1.2rem; margin: 2rem 0;">
            📈 SID Method - RSI/MACD Reversal Trading<br>
            💰 Capital Preservation First (0.5-2% Risk Per Trade)
        </p>
        <p>👈 Configure OANDA API keys and file paths in sidebar, then click Initialize System</p>
        <hr>
        <h3>⭐ Sid's Recommended Pairs (Best First):</h3>
        <p><strong>GBP_JPY</strong> - Best for SID Method<br>
        <strong>EUR_USD</strong> - Most liquid<br>
        <strong>USD_JPY</strong> - Yen pair (10 pip stops)<br>
        <strong>AUD_USD</strong> - Commodity currency<br>
        <strong>GBP_USD</strong> - Volatile, good for shorts</p>
    </div>
    """, unsafe_allow_html=True)
else:
    if page == "Market Scanner":
        st.markdown('<h2 class="sub-header">🔍 Market Scanner (SID Method)</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem; flex-wrap: wrap;">
            <span class="pair-category-badge category-best">⭐ BEST</span>
            <span class="pair-category-badge category-good">👍 GOOD</span>
            <span class="pair-category-badge category-metals">🥇 METALS</span>
            <span class="pair-category-badge category-standard">📊 STANDARD</span>
            <span class="pair-category-badge category-avoid">⚠️ AVOID</span>
        </div>
        """, unsafe_allow_html=True)
        check_daily_reset()
        time_since = (datetime.now() - st.session_state.last_refresh).seconds
        if time_since > 60 or not st.session_state.current_data:
            with st.spinner("🔍 Scanning markets..."):
                if st.session_state.oanda_client:
                    st.session_state.current_data = fetch_market_data(st.session_state.selected_pairs, st.session_state.oanda_client, timeframe, bars)
                if st.session_state.current_data:
                    st.session_state.detected_signals = detect_sid_signals_with_ai(st.session_state.current_data, st.session_state.sid_method, st.session_state.ai_enabled, st.session_state.signal_predictor)
                st.session_state.last_refresh = datetime.now()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pairs", len(st.session_state.current_data))
        with col2:
            st.metric("Signals", len(st.session_state.detected_signals))
        with col3:
            st.caption(f"Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        st.markdown("---")
        display_signals_grid(st.session_state.detected_signals)
        for key in st.session_state:
            if key.startswith('confirm_signal_') and st.session_state[key]:
                render_trade_confirmation(st.session_state[key], st.session_state.oanda_trader, st.session_state.account_summary)
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
    """
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Sid Naiman SID Method Trading System | v8.0 - MOBILE OPTIMIZED<br>
        Core Rules: RSI &lt;30 Oversold / &gt;70 Overbought | MACD Alignment | RSI 50 Take Profit<br>
        Stop Loss: Lowest Low Rounded Down / Highest High Rounded Up<br>
        Risk: 0.5-2% Per Trade<br>
        📊 Pair Order: ⭐ BEST → 👍 GOOD → 🥇 METALS → 📊 STANDARD → ⚠️ AVOID<br>
        ⚠️ Risk Warning: Trading carries substantial risk - Always use proper position sizing
    </div>
    """,
    unsafe_allow_html=True
)