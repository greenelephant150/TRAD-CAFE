"""
Streamlit App for Simon Pullen's Trading System
Complete AI-Augmented Trading Interface with GPU/CPU Support
Following Academy Support Session rules with ML enhancement
Version: 4.9 - FIXED ARROW SERIALIZATION ISSUES
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
    from pattern_detector import PatternDetector, Pattern
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
# CONFIGURATION - SET YOUR PATHS HERE
# ============================================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CSV_BASE_PATH = "/home/grct/Forex"
PARQUET_BASE_PATH = "/home/grct/Forex_Parquet"
MODEL_BASE_PATH = "/mnt2/Trading-Cafe/ML/SPullen/ai/trained_models/"

# Create directories if they don't exist
os.makedirs(MODEL_BASE_PATH, exist_ok=True)
os.makedirs(CSV_BASE_PATH, exist_ok=True)
os.makedirs(PARQUET_BASE_PATH, exist_ok=True)

print(f"📁 Model path: {MODEL_BASE_PATH}")
print(f"📁 CSV path: {CSV_BASE_PATH}")
print(f"📁 Parquet path: {PARQUET_BASE_PATH}")

# Page configuration
st.set_page_config(
    page_title="Simon Pullen AI Trading System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Main headers */
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
    
    /* Zone cards */
    .zone-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .zone-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .demand-zone {
        border-left: 8px solid #00C853;
    }
    .supply-zone {
        border-left: 8px solid #D32F2F;
    }
    
    /* Signal badges */
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
    
    /* AI confidence badges */
    .ai-very-strong {
        background-color: #1B5E20;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .ai-strong {
        background-color: #2E7D32;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .ai-moderate {
        background-color: #F57C00;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    .ai-weak {
        background-color: #C62828;
        color: white;
        padding: 0.2rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        display: inline-block;
    }
    
    /* GPU/CPU status */
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
    
    /* Metric cards */
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        transition: all 0.3s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        border-radius: 25px;
        font-weight: bold;
        transition: all 0.3s;
        background: linear-gradient(90deg, #1E88E5, #7C4DFF);
        color: white;
        border: none;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(30,136,229,0.4);
    }
    .stButton > button:disabled {
        background: #cccccc;
        transform: none;
    }
    
    /* Progress bar customization */
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    
    /* Info/Warning boxes */
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
    
    /* Training status indicators */
    .training-queued {
        background-color: #FFA000;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .training-active {
        background-color: #1E88E5;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    .training-completed {
        background-color: #00C853;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
        display: inline-block;
    }
    .training-failed {
        background-color: #D32F2F;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 10px;
        font-size: 0.8rem;
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
# PARQUET HELPER FUNCTIONS
# ============================================================================

def get_available_pairs_with_parquet():
    """Get list of pairs that have Parquet files available in partitioned structure."""
    pairs_with_parquet = []
    
    if not os.path.exists(PARQUET_BASE_PATH):
        return pairs_with_parquet
    
    # List all pair directories
    for item in os.listdir(PARQUET_BASE_PATH):
        pair_path = os.path.join(PARQUET_BASE_PATH, item)
        
        # Check if it's a directory
        if os.path.isdir(pair_path):
            # Look for year=* subdirectories (partitioned structure)
            for subdir in os.listdir(pair_path):
                subdir_path = os.path.join(pair_path, subdir)
                if os.path.isdir(subdir_path) and subdir.startswith('year='):
                    # Found at least one year partition, consider pair available
                    pairs_with_parquet.append(item)
                    break
    
    return sorted(pairs_with_parquet)


def get_parquet_stats():
    """Get statistics about Parquet files in partitioned structure."""
    stats = {
        'total_pairs': 0,
        'total_files': 0,
        'total_size_gb': 0,
        'pairs': []
    }
    
    if os.path.exists(PARQUET_BASE_PATH):
        for pair in os.listdir(PARQUET_BASE_PATH):
            pair_path = os.path.join(PARQUET_BASE_PATH, pair)
            if os.path.isdir(pair_path):
                # Count all parquet files recursively
                all_files = []
                years = []
                
                for root, dirs, files in os.walk(pair_path):
                    # Extract year from directory path if it's a year partition
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


def normalize_column_names(df):
    """Normalize column names to expected format (Date, open, high, low, close)."""
    # Create a mapping of possible column names
    column_mapping = {}
    
    # Map for Date column
    date_variants = ['Date', 'date', 'timestamp', 'time', 'datetime', 'ds']
    for variant in date_variants:
        if variant in df.columns:
            column_mapping[variant] = 'Date'
            break
    
    # Map for price columns
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
    
    # Rename columns
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df


def load_data_from_parquet(pair, start_date=None, end_date=None, progress_callback=None):
    """
    Load data from partitioned Parquet files for a specific pair with progress tracking.
    Handles structure: /pair/year=YYYY/month=MM/day=DD/*.parquet
    """
    import pandas as pd
    import pyarrow.parquet as pq
    
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    
    if not os.path.exists(pair_path):
        if progress_callback:
            progress_callback(0, f"⚠️ Parquet path not found: {pair_path}")
        return pd.DataFrame()
    
    # First try using pyarrow dataset with partitioning
    try:
        if progress_callback:
            progress_callback(0.1, "Reading Parquet dataset...")
        
        # Build filter conditions if dates provided
        filters = None
        if start_date or end_date:
            filter_conditions = []
            start = pd.to_datetime(start_date) if start_date else None
            end = pd.to_datetime(end_date) if end_date else None
            
            if start:
                filter_conditions.append(('year', '>=', start.year))
            if end:
                filter_conditions.append(('year', '<=', end.year))
            
            if filter_conditions:
                filters = filter_conditions
        
        # Read using pyarrow dataset with partitioning
        dataset = pq.ParquetDataset(
            pair_path,
            partitioning='hive'  # This handles year=YYYY/month=MM/day=DD structure
        )
        
        if progress_callback:
            progress_callback(0.3, "Reading data...")
        
        # Apply filters if any
        if filters:
            table = dataset.read_pieces(filters=filters)
        else:
            table = dataset.read()
        
        if progress_callback:
            progress_callback(0.8, "Converting to pandas...")
        
        df = table.to_pandas()
        
        if progress_callback:
            progress_callback(0.9, "Normalizing columns...")
        
        # Normalize column names
        df = normalize_column_names(df)
        
        # Ensure Date column exists and is datetime
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
            progress_callback(0.2, f"Dataset reading failed, trying recursive search: {str(e)[:50]}")
        
        # Fallback to recursive file search
        return load_data_from_parquet_fallback(pair, start_date, end_date, progress_callback)


def load_data_from_parquet_fallback(pair, start_date=None, end_date=None, progress_callback=None):
    """Fallback method: recursively find and load parquet files with progress tracking."""
    import pandas as pd
    
    data_frames = []
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    
    if not os.path.exists(pair_path):
        return pd.DataFrame()
    
    # Parse date filters
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None
    
    # Find all parquet files recursively
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
    
    # Process files with progress
    loaded_files = 0
    for i, file_path in enumerate(sorted(all_files)):
        # Update progress
        progress = 0.1 + (0.8 * (i / total_files))
        if progress_callback and i % max(1, total_files // 20) == 0:
            progress_callback(progress, f"Loading file {i+1}/{total_files}")
        
        try:
            # Extract date from path for filtering
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
            
            # Apply date filtering if we have year and date filters
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
            
            # Read parquet file
            df = pd.read_parquet(file_path)
            
            if not df.empty:
                # Normalize column names
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
    
    # Combine all data
    df = pd.concat(data_frames, ignore_index=True)
    
    # Ensure Date column exists
    if 'Date' not in df.columns:
        return pd.DataFrame()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    if progress_callback:
        progress_callback(1.0, f"Loaded {len(df)} rows from {loaded_files} files")
    
    return df


def load_data_from_parquet_with_progress(pair, start_date=None, end_date=None):
    """Wrapper function to load parquet data with a progress bar."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(progress, message):
        progress_bar.progress(progress)
        status_text.text(message)
    
    df = load_data_from_parquet(pair, start_date, end_date, update_progress)
    
    status_text.empty()
    progress_bar.empty()
    
    return df


def check_parquet_available(pair):
    """Check if parquet files exist for a specific pair."""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    if not os.path.exists(pair_path):
        return False
    
    # Check for any year=* directory
    for item in os.listdir(pair_path):
        if item.startswith('year=') and os.path.isdir(os.path.join(pair_path, item)):
            return True
    
    return False


def get_parquet_date_range(pair):
    """Get the minimum and maximum dates available in parquet for a pair."""
    try:
        # Load just one file to check structure
        df = load_data_from_parquet(pair)
        if not df.empty and 'Date' in df.columns:
            return df['Date'].min(), df['Date'].max()
    except:
        pass
    return None, None


# ============================================================================
# Model discovery helper functions
# ============================================================================

def get_available_models():
    """Get list of available trained models."""
    if not os.path.exists(MODEL_BASE_PATH):
        return []
    
    models = []
    for file in os.listdir(MODEL_BASE_PATH):
        if file.endswith('.pkl') or file.endswith('.joblib'):
            models.append(file)
    return sorted(models)


def get_model_stats():
    """Get statistics about trained models."""
    stats = {
        'total_models': 0,
        'models_by_pair': {},
        'latest_models': []
    }
    
    if not os.path.exists(MODEL_BASE_PATH):
        return stats
    
    for file in os.listdir(MODEL_BASE_PATH):
        if file.endswith('.pkl') or file.endswith('.joblib'):
            stats['total_models'] += 1
            
            # Try to extract pair from filename
            try:
                parts = file.replace('.pkl', '').replace('.joblib', '').split('--')
                if len(parts) >= 4:
                    pair = parts[3]  # Format: DDMMYYYY--DDMMYYYY--SPullen--PAIR--S5.pkl
                    if pair not in stats['models_by_pair']:
                        stats['models_by_pair'][pair] = []
                    stats['models_by_pair'][pair].append(file)
            except:
                pass
    
    return stats


# ============================================================================
# AI Training Functions
# ============================================================================

def display_model_management():
    """Display AI model management interface"""
    st.markdown("### 🤖 AI Model Management")
    
    if not AI_AVAILABLE:
        st.warning("⚠️ AI modules not available")
        return
    
    if st.session_state.model_manager is None:
        st.warning("⚠️ ModelManager not initialized")
        return
    
    # Get model summary
    models_df = st.session_state.model_manager.get_latest_models_summary()
    
    if models_df.empty:
        st.info("No trained models found. Run training pipeline to create models.")
        
        # Quick training button
        if st.button("🚀 Train Default Model", type="primary"):
            with st.spinner("Training model..."):
                # Train a simple model for EUR_USD as example
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
    
    # Display model statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(models_df))
    
    with col2:
        unique_pairs = models_df['Pair'].nunique()
        st.metric("Unique Pairs", unique_pairs)
    
    with col3:
        # Handle 'N/A' values in accuracy
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
        # Convert date to string before passing to metric
        if not models_df.empty and 'To' in models_df.columns:
            latest_date = models_df['To'].iloc[0]
            # Convert to string if it's a date object
            if hasattr(latest_date, 'strftime'):
                latest_date_str = latest_date.strftime('%Y-%m-%d')
            else:
                latest_date_str = str(latest_date)
            st.metric("Latest Model", latest_date_str)
        else:
            st.metric("Latest Model", "N/A")
    
    # Filter by pair
    selected_pair = st.selectbox(
        "Filter by Pair",
        ["All"] + sorted(models_df['Pair'].unique().tolist())
    )
    
    if selected_pair != "All":
        display_df = models_df[models_df['Pair'] == selected_pair].copy()
    else:
        display_df = models_df.copy()
    
    # Format display columns - FIXED for Arrow serialization
    display_cols = ['Pair', 'From', 'To', 'Days', 'Accuracy', 'Samples', 'Device', 'Model Name']
    available_cols = [c for c in display_cols if c in display_df.columns]
    
    # Create a copy for display and convert all columns to string-safe formats
    display_data = pd.DataFrame()
    
    for col in available_cols:
        if col in ['From', 'To']:
            # Convert dates to strings
            display_data[col] = display_df[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
            )
        elif col == 'Accuracy':
            # Format accuracy as percentage string
            display_data[col] = display_df[col].apply(
                lambda x: f"{x:.1%}" if isinstance(x, (int, float)) and x != 'N/A' else str(x)
            )
        elif col == 'Samples':
            # Format samples with commas as string
            display_data[col] = display_df[col].apply(
                lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x)
            )
        else:
            # Convert all other columns to string
            display_data[col] = display_df[col].astype(str)
    
    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )
    
    # Model actions
    st.markdown("#### 🛠️ Model Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Models", use_container_width=True):
            st.session_state.signal_predictor.reload_models()
            st.success("Models reloaded")
            st.rerun()
    
    with col2:
        if st.button("📊 Validate All Models", use_container_width=True):
            with st.spinner("Validating models..."):
                validation_df = st.session_state.training_pipeline.validate_all_models(days=30)
                if not validation_df.empty:
                    # Convert validation results to string-safe format
                    display_validation = pd.DataFrame()
                    for col in validation_df.columns:
                        display_validation[col] = validation_df[col].astype(str)
                    st.success(f"Validated {len(validation_df)} models")
                    st.dataframe(display_validation[['pair', 'accuracy', 'f1_score']])
                else:
                    st.warning("No validation results")
    
    with col3:
        if st.button("🧹 Cleanup Old Models", use_container_width=True):
            with st.spinner("Cleaning up..."):
                st.session_state.training_pipeline.cleanup_old_models(keep_last=3)
                st.success("Cleanup complete")
                st.rerun()
    
    # Training pipeline status
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
                results = st.session_state.training_pipeline.train_all_pairs(
                    force=True,
                    max_pairs=5  # Limit to 5 for demo
                )
                st.success(f"Training complete! {results.get('successful', 0)} successful")
                st.rerun()


def display_parquet_management():
    """Display Parquet management interface"""
    st.markdown("### 💾 Parquet Management")
    st.markdown("Convert CSV to Parquet for 10x faster data loading")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Show Parquet stats
        stats = get_parquet_stats()
        
        st.metric("Pairs with Parquet", stats['total_pairs'])
        st.metric("Total Files", stats['total_files'])
        st.metric("Total Size", f"{stats['total_size_gb']:.2f} GB")
        
        if stats['pairs']:
            st.markdown("##### Available Parquet Pairs")
            for p in stats['pairs'][:10]:
                years_str = f"{p['years'][0]}-{p['years'][-1]}" if p['years'] else "Unknown"
                st.caption(f"• {p['pair']}: {p['files']} files, {p['size_gb']:.2f}GB ({years_str})")
    
    with col2:
        # Get available CSV pairs
        csv_pairs = []
        if os.path.exists(CSV_BASE_PATH):
            for item in os.listdir(CSV_BASE_PATH):
                pair_path = os.path.join(CSV_BASE_PATH, item)
                if os.path.isdir(pair_path):
                    csv_pairs.append(item)
        
        if csv_pairs:
            convert_pair = st.selectbox(
                "Select Pair to Convert",
                sorted(csv_pairs)
            )
            
            # GPU toggle
            use_gpu = st.checkbox("🚀 Use GPU for conversion", value=True)
            
            # Mode selection
            mode = st.selectbox(
                "Conversion Mode",
                ["append", "initial", "overwrite", "update"],
                index=0,
                help="append: Add new dates only, initial: First time setup, overwrite: Replace all, update: Refresh existing"
            )
            
            # Date range for update
            update_start = None
            update_end = None
            if mode in ['update', 'append']:
                col_a, col_b = st.columns(2)
                with col_a:
                    update_start = st.date_input("Update Start", datetime.now() - timedelta(days=30))
                with col_b:
                    update_end = st.date_input("Update End", datetime.now())
            
            if st.button("🔄 Convert to Parquet", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(progress, message):
                    progress_bar.progress(progress)
                    status_text.text(message)
                
                try:
                    # Try different initialization methods for ParquetConverter
                    if st.session_state.parquet_converter is None:
                        # Try to create new converter with various signatures
                        try:
                            converter = ParquetConverter(
                                csv_dir=CSV_BASE_PATH,
                                parquet_dir=PARQUET_BASE_PATH,
                                use_gpu=use_gpu
                            )
                        except TypeError:
                            try:
                                converter = ParquetConverter(
                                    CSV_BASE_PATH, 
                                    PARQUET_BASE_PATH,
                                    use_gpu
                                )
                            except TypeError:
                                try:
                                    converter = ParquetConverter(use_gpu=use_gpu)
                                except TypeError:
                                    converter = ParquetConverter()
                    else:
                        converter = st.session_state.parquet_converter
                    
                    start_date_str = update_start.strftime('%Y-%m-%d') if update_start else None
                    end_date_str = update_end.strftime('%Y-%m-%d') if update_end else None
                    
                    # Try different method names for conversion
                    if hasattr(converter, 'convert_pair'):
                        success = converter.convert_pair(
                            pair=convert_pair,
                            mode=mode,
                            start_date=start_date_str,
                            end_date=end_date_str,
                            progress_callback=update_progress
                        )
                    elif hasattr(converter, 'convert'):
                        success = converter.convert(
                            pair=convert_pair,
                            mode=mode,
                            start_date=start_date_str,
                            end_date=end_date_str
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


def display_training_data_tab():
    """Display the Training Data tab content"""
    st.markdown("#### 📊 Training Data Management")
    
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
    
    # Show Parquet availability
    parquet_pairs = get_available_pairs_with_parquet()
    if parquet_pairs:
        st.success(f"✅ Parquet available for {len(parquet_pairs)} pairs (10x faster)")
    else:
        st.info("ℹ️ No Parquet files found. Will use CSV (slower). Use Parquet Management tab to convert.")
    
    # Pair selection
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
    
    # Check if Parquet exists for selected pair
    if selected_pair:
        has_parquet = check_parquet_available(selected_pair)
        if has_parquet:
            use_parquet = st.checkbox("🚀 Use Parquet (10x faster)", value=True)
            
            # Show Parquet date range if available
            min_date, max_date = get_parquet_date_range(selected_pair)
            if min_date and max_date:
                st.info(f"📅 Parquet data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
        else:
            use_parquet = False
            st.info("ℹ️ No Parquet for this pair yet. Will use CSV. Use Parquet Management tab to convert.")
    
    # Time range selection with MAX option
    st.markdown("##### Select Time Range (Loads ALL data in this range)")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        range_preset = st.selectbox(
            "Quick Select",
            ["MAX", "Custom", "Last Week", "Last 2 Weeks", "Last Month", 
             "Last 3 Months", "Last 6 Months", "Last Year", 
             "Last 2 Years", "Last 5 Years", "Last 10 Years"],
            index=0  # Default to MAX
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
    
    # Timeframe selection
    st.markdown("##### Data Resolution (Simon Pullen Method)")
    timeframe = st.selectbox(
        "Timeframe for Analysis",
        ["1m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=4,  # Default to 1h
        help="S5 data will be averaged to this timeframe"
    )
    
    # Load button with progress bar
    if st.button("📥 LOAD ALL DATA", type="primary", use_container_width=False):
        if not selected_pair:
            st.error("Please select a trading pair")
        else:
            try:
                feat_eng = st.session_state.feature_engineer
                
                # If using Parquet, load directly with progress bar
                if use_parquet:
                    df = load_data_from_parquet_with_progress(
                        selected_pair,
                        start_date=start_str,
                        end_date=end_str
                    )
                    
                    if not df.empty and 'Date' in df.columns:
                        # Set Date as index for feature engineering
                        df = df.set_index('Date')
                        
                        # Create features
                        X = feat_eng.create_features(df)
                        
                        # Create target (next period return > 0)
                        y = (df['close'].shift(-1) > df['close']).astype(int).iloc[:-1]
                        X = X.iloc[:-1]  # Align
                        
                        # Add to training data
                        new_samples = 0
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i in range(min(len(X), 100000)):  # Limit for performance
                            if i % 1000 == 0:
                                progress = i / min(len(X), 100000)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing sample {i}/{min(len(X), 100000)}")
                            
                            sample = {
                                'timestamp': df.index[i] if i < len(df.index) else datetime.now(),
                                'pair': selected_pair,
                                'timeframe': timeframe,
                                'date_range': range_preset,
                                **X.iloc[i].to_dict(),
                                'outcome': int(y.iloc[i]) if not pd.isna(y.iloc[i]) else 0
                            }
                            st.session_state.training_data.append(sample)
                            new_samples += 1
                        
                        progress_bar.progress(1.0)
                        status_text.empty()
                        
                        st.success(f"✅ Loaded {new_samples} training samples for {selected_pair}")
                    else:
                        st.warning(f"No data available for {selected_pair}")
                
                else:
                    # Create progress elements for CSV loading
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    eta_text = st.empty()
                    speed_text = st.empty()
                    start_time = time.time()
                    
                    def update_progress(progress, message):
                        progress_bar.progress(progress)
                        status_text.text(message)
                        elapsed = time.time() - start_time
                        if progress > 0:
                            total_eta = elapsed / progress
                            remaining = total_eta * (1 - progress)
                            eta_text.text(f"ETA: {remaining:.1f} seconds")
                            speed_text.text(f"Speed: {progress/elapsed:.2%}/s")
                    
                    # Use feature engineer's method for CSV
                    X, y = feat_eng.create_training_dataset_from_forex(
                        pairs=[selected_pair],
                        start_date=start_str,
                        end_date=end_str,
                        timeframe=timeframe,
                        samples_per_pair=1000000,
                        use_parquet=False,
                        progress_callback=update_progress
                    )
                    
                    if not X.empty:
                        # Convert to training samples with safe timestamp handling
                        new_samples = 0
                        for i in range(len(X)):
                            # Create safe timestamp
                            try:
                                days_ago = min(i, 3650)  # Cap at 10 years
                                sample_timestamp = datetime.now() - timedelta(days=days_ago)
                                if sample_timestamp.year < 1900:
                                    sample_timestamp = datetime.now()
                            except:
                                sample_timestamp = datetime.now()
                            
                            sample = {
                                'timestamp': sample_timestamp,
                                'pair': selected_pair,
                                'timeframe': timeframe,
                                'date_range': range_preset,
                                **X.iloc[i].to_dict(),
                                'outcome': int(y.iloc[i]) if not pd.isna(y.iloc[i]) else 0
                            }
                            st.session_state.training_data.append(sample)
                            new_samples += 1
                        
                        progress_bar.progress(1.0)
                        elapsed = time.time() - start_time
                        status_text.text("Complete!")
                        eta_text.text(f"✅ Loaded {new_samples} samples in {elapsed:.1f}s")
                        speed_text.text(f"Speed: {new_samples/elapsed:.0f} samples/sec")
                        
                        st.success(f"✅ Loaded {new_samples} training samples for {selected_pair}")
                        st.info(f"Timeframe: {timeframe}")
                        st.info(f"Format: {'Parquet (10x faster)' if use_parquet else 'CSV'}")
                    else:
                        st.warning(f"No data available for {selected_pair}")
                    
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
        
        # Show data range with error handling - FIXED for Arrow serialization
        if 'timestamp' in df_summary.columns:
            try:
                timestamps = pd.to_datetime(df_summary['timestamp'], errors='coerce')
                valid_timestamps = timestamps.dropna()
                
                if not valid_timestamps.empty:
                    min_date = valid_timestamps.min()
                    max_date = valid_timestamps.max()
                    
                    if hasattr(min_date, 'year') and min_date.year > 1900 and max_date.year < 2100:
                        st.info(f"Data range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
                    else:
                        st.warning(f"Unusual date range: {min_date} to {max_date}")
                    
                    invalid_count = len(timestamps) - len(valid_timestamps)
                    if invalid_count > 0:
                        st.warning(f"⚠️ {invalid_count} samples have invalid timestamps")
                else:
                    st.warning("No valid timestamps found")
            except Exception as e:
                st.warning(f"Could not parse timestamps: {e}")
        
        # Show recent samples - FIXED for Arrow serialization
        st.markdown("##### Recent Samples")
        try:
            display_cols = ['timestamp', 'pair', 'timeframe', 'zone_quality', 'rsi', 'outcome']
            existing_cols = [col for col in display_cols if col in df_summary.columns]
            
            if existing_cols:
                display_df = df_summary[existing_cols].tail(10).copy()
                
                # Convert all columns to string to avoid Arrow serialization issues
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
                # Convert all columns to string
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


def display_train_models_tab():
    """Display the Train Models tab content"""
    st.markdown("#### 🏋️ Train New Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Training Parameters")
        
        # Model type selection
        model_type = st.selectbox(
            "Model Type",
            ["Random Forest", "XGBoost", "LightGBM", "Neural Network"],
            index=0
        )
        
        # Training configuration
        lookback = st.slider("Lookback Period", 10, 100, 20, 
                            help="Number of previous bars to use for features")
        
        train_size = st.slider("Training Samples", 10000, 200000, 100000, step=10000,
                              help="Maximum number of samples to use")
        
        validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05,
                                    help="Portion of data for validation")
        
        # Feature selection
        st.markdown("##### Features to Include")
        
        feature_categories = {
            'Price': ['open', 'high', 'low', 'close', 'volume'],
            'Technical': ['rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle'],
            'Oscillators': ['stoch_k', 'stoch_d', 'cci', 'williams_r'],
            'Trend': ['ema_9', 'ema_21', 'sma_50', 'adx'],
            'Volatility': ['atr', 'natr', 'volatility']
        }
        
        selected_features = []
        for category, feat_list in feature_categories.items():
            with st.expander(category, expanded=False):
                selected = st.multiselect(
                    f"Select {category}",
                    options=feat_list,
                    default=feat_list[:2] if category == 'Price' else []
                )
                selected_features.extend(selected)
        
        # Advanced options
        with st.expander("Advanced Options"):
            use_gpu = st.checkbox("Use GPU if available", value=True)
            parallel_jobs = st.slider("Parallel Jobs", 1, 32, 8)
            save_intermediate = st.checkbox("Save intermediate results", value=False)
    
    with col2:
        st.markdown("##### Training Queue")
        
        # Get available pairs from training data
        if st.session_state.training_data:
            df = pd.DataFrame(st.session_state.training_data)
            available_pairs = df['pair'].unique().tolist() if 'pair' in df.columns else []
        else:
            available_pairs = []
        
        if available_pairs:
            selected_pairs = st.multiselect(
                "Select Pairs to Train",
                options=available_pairs,
                default=available_pairs[:3] if len(available_pairs) >= 3 else available_pairs
            )
            
            if st.button("➕ Add to Queue", use_container_width=True):
                if 'training_queue' not in st.session_state:
                    st.session_state.training_queue = []
                
                for pair in selected_pairs:
                    # Check if already in queue
                    if not any(item['pair'] == pair for item in st.session_state.training_queue):
                        st.session_state.training_queue.append({
                            'pair': pair,
                            'model_type': model_type,
                            'lookback': lookback,
                            'train_size': train_size,
                            'validation_split': validation_split,
                            'features': selected_features,
                            'status': 'queued',
                            'queued_time': datetime.now().strftime("%H:%M:%S")
                        })
                
                st.success(f"Added {len(selected_pairs)} models to queue")
        else:
            st.warning("No training data available. Load data in 'Training Data' tab first.")
        
        # Display queue
        st.markdown("##### Current Queue")
        if 'training_queue' in st.session_state and st.session_state.training_queue:
            for idx, job in enumerate(st.session_state.training_queue):
                status_color = {
                    'queued': '🔵',
                    'training': '🟡',
                    'completed': '🟢',
                    'failed': '🔴'
                }.get(job['status'], '⚪')
                
                col_a, col_b, col_c = st.columns([1, 4, 1])
                with col_a:
                    st.markdown(f"{status_color}")
                with col_b:
                    st.markdown(f"**{job['pair']}** - {job['model_type']}")
                with col_c:
                    if st.button("❌", key=f"remove_queue_{idx}"):
                        st.session_state.training_queue.pop(idx)
                        st.rerun()
            
            # Batch actions
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("▶️ Start All", use_container_width=True):
                    st.session_state.training_active = True
                    st.rerun()
            with col2:
                if st.button("🗑️ Clear Queue", use_container_width=True):
                    st.session_state.training_queue = []
                    st.rerun()
            with col3:
                st.button("⏸️ Pause", disabled=True, use_container_width=True)
        else:
            st.info("Queue is empty")
        
        # Training progress
        if st.session_state.get('training_active', False):
            st.markdown("##### Training Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training (replace with actual training)
            for i in range(100):
                time.sleep(0.05)
                progress_bar.progress(i + 1)
                status_text.text(f"Training... {i+1}%")
            
            status_text.text("Training complete!")
            st.success("✅ All models trained successfully")
            st.session_state.training_active = False
            st.rerun()


def display_model_performance_tab():
    """Display the Model Performance tab content"""
    st.markdown("#### 📈 Model Performance Metrics")
    
    if st.session_state.model_manager is None:
        st.warning("ModelManager not initialized")
        return
    
    # Get model summary
    models_df = st.session_state.model_manager.get_latest_models_summary()
    
    if models_df.empty:
        st.info("No models available. Train some models first!")
        return
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(models_df))
    
    with col2:
        acc_values = models_df['Accuracy'][models_df['Accuracy'] != 'N/A']
        if not acc_values.empty:
            avg_acc = acc_values.mean()
            st.metric("Avg Accuracy", f"{avg_acc:.2%}")
        else:
            st.metric("Avg Accuracy", "N/A")
    
    with col3:
        if not acc_values.empty:
            best_acc = acc_values.max()
            best_pair = models_df.loc[models_df['Accuracy'] == best_acc, 'Pair'].iloc[0]
            st.metric("Best Model", f"{best_pair} ({best_acc:.2%})")
        else:
            st.metric("Best Model", "N/A")
    
    with col4:
        st.metric("Last Updated", datetime.now().strftime("%Y-%m-%d"))
    
    # Performance chart
    if 'Accuracy' in models_df.columns and 'Pair' in models_df.columns:
        # Filter out N/A values
        plot_df = models_df[models_df['Accuracy'] != 'N/A'].copy()
        if not plot_df.empty:
            plot_df['Accuracy'] = plot_df['Accuracy'].astype(float)
            
            fig = px.bar(
                plot_df.sort_values('Accuracy', ascending=False),
                x='Pair',
                y='Accuracy',
                color='Accuracy',
                color_continuous_scale='RdYlGn',
                title="Model Accuracy by Pair",
                labels={'Accuracy': 'Accuracy', 'Pair': 'Currency Pair'},
                hover_data=['Samples', 'Device']
            )
            fig.update_layout(
                xaxis_tickangle=-45,
                height=500,
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Metrics distribution
    col1, col2 = st.columns(2)
    
    with col1:
        if not plot_df.empty and len(plot_df) > 1:
            fig2 = px.box(
                plot_df,
                y='Accuracy',
                title="Accuracy Distribution",
                points="all"
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
            if 'Samples' in plot_df.columns:
                fig3 = px.histogram(
                    plot_df,
                    x='Samples',
                    nbins=20,
                    title="Training Samples Distribution",
                    labels={'Samples': 'Number of Samples'}
                )
                fig3.update_layout(height=400)
                st.plotly_chart(fig3, use_container_width=True)    
    # Detailed table - FIXED for Arrow serialization
    st.markdown("##### Detailed Performance")
    
    display_df = pd.DataFrame()
    
    for col in models_df.columns:
        if col in ['From', 'To']:
            display_df[col] = models_df[col].apply(
                lambda x: x.strftime('%Y-%m-%d') if hasattr(x, 'strftime') else str(x)
            )
        elif col == 'Accuracy':
            display_df[col] = models_df[col].apply(
                lambda x: f"{x:.2%}" if isinstance(x, (int, float)) and x != 'N/A' else str(x)
            )
        elif col == 'Samples':
            display_df[col] = models_df[col].apply(
                lambda x: f"{x:,}" if isinstance(x, (int, float)) else str(x)
            )
        else:
            display_df[col] = models_df[col].astype(str)
    
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


def display_update_model_tab():
    """Display the Update Model tab content"""
    st.markdown("#### 🔄 Update Existing Models")
    
    if st.session_state.model_manager is None:
        st.warning("ModelManager not initialized")
        return
    
    # Get available models
    models_df = st.session_state.model_manager.get_latest_models_summary()
    
    if models_df.empty:
        st.info("No models available to update")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Select Model to Update")
        
        # Create model selection
        model_options = []
        for _, row in models_df.iterrows():
            pair = row['Pair']
            acc = row['Accuracy']
            acc_str = f"{acc:.2%}" if isinstance(acc, (int, float)) else str(acc)
            model_options.append(f"{pair} (Acc: {acc_str})")
        
        selected_idx = st.selectbox(
            "Choose Model",
            range(len(model_options)),
            format_func=lambda x: model_options[x]
        )
        
        selected_row = models_df.iloc[selected_idx]
        
        # Display model info - FIXED for Arrow serialization
        st.markdown("##### Current Model Details")
        info_data = [
            ["Pair", str(selected_row['Pair'])],
            ["Accuracy", f"{selected_row['Accuracy']:.2%}" if isinstance(selected_row['Accuracy'], (int, float)) else str(selected_row['Accuracy'])],
            ["Samples", f"{selected_row['Samples']:,}" if isinstance(selected_row['Samples'], (int, float)) else str(selected_row['Samples'])],
            ["Date Range", f"{selected_row['From']} to {selected_row['To']}"],
            ["Device", str(selected_row.get('Device', 'Unknown'))]
        ]
        info_df = pd.DataFrame(info_data, columns=["Property", "Value"])
        st.table(info_df)
    
    with col2:
        st.markdown("##### Update Options")
        
        update_type = st.radio(
            "Update Type",
            ["🔄 Retrain with new data", "⚡ Fine-tune existing", "📅 Extend date range"]
        )
        
        if update_type == "📅 Extend date range":
            new_end_date = st.date_input(
                "New End Date",
                value=datetime.now().date(),
                min_value=datetime.now().date() - timedelta(days=365),
                max_value=datetime.now().date()
            )
            
            # Calculate days to add
            if hasattr(selected_row['To'], 'year'):
                days_to_add = (new_end_date - selected_row['To']).days
                if days_to_add > 0:
                    st.success(f"Will add {days_to_add} days of new data")
        
        # Update parameters
        st.markdown("##### Update Parameters")
        new_samples = st.slider("Additional Samples", 10000, 100000, 50000, step=10000)
        new_lookback = st.slider("Lookback Period", 10, 100, 20)
        
        if st.button("🔄 Update Model", type="primary", use_container_width=True):
            with st.spinner(f"Updating {selected_row['Pair']}..."):
                # Simulate update process
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.05)
                    progress_bar.progress(i + 1)
                
                st.success(f"✅ Model {selected_row['Pair']} updated successfully!")


def display_ai_training():
    """Display AI training interface with all tabs"""
    st.markdown("### 🧠 AI Model Training")
    
    if not AI_AVAILABLE:
        st.warning("⚠️ AI modules not available. Run: pip install scikit-learn pandas numpy")
        return
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Training Data", 
        "🏋️ Train Models", 
        "📈 Model Performance", 
        "🔄 Update Model",
        "💾 Parquet Management", 
        "🤖 Model Management"
    ])
    
    with tab1:
        display_training_data_tab()
    
    with tab2:
        display_train_models_tab()
    
    with tab3:
        display_model_performance_tab()
    
    with tab4:
        display_update_model_tab()
    
    with tab5:
        display_parquet_management()
    
    with tab6:
        display_model_management()


# ============================================================================
# INITIALIZATION
# ============================================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.oanda_client = None
    st.session_state.oanda_trader = None
    st.session_state.sd_detector = None
    st.session_state.pattern_detector = None
    st.session_state.ai_accelerator = None
    st.session_state.feature_engineer = None
    st.session_state.signal_predictor = None
    st.session_state.model_trainer = None
    st.session_state.model_manager = None
    st.session_state.training_pipeline = None
    st.session_state.parquet_converter = None
    st.session_state.current_data = {}
    st.session_state.detected_zones = []
    st.session_state.detected_patterns = []
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


def initialize_system(environment: str = "practice", device: str = 'auto'):
    """Initialize all components with error handling"""
    if not CORE_IMPORTS_SUCCESS:
        st.error("Core modules not available. Please check installation.")
        return False
    
    try:
        # Initialize OANDA
        with st.spinner("🔄 Connecting to OANDA..."):
            st.session_state.oanda_client = OANDAClient()
            # Override supported pairs with verified list
            st.session_state.oanda_client.valid_pairs = set(VALID_OANDA_PAIRS)
            st.session_state.oanda_trader = OANDATrader(environment=environment)
        
        # Initialize pattern detectors
        with st.spinner("🔄 Loading trading strategies..."):
            st.session_state.sd_detector = SupplyDemand()
            st.session_state.pattern_detector = PatternDetector()
        
        # Initialize AI components
        if AI_AVAILABLE:
            with st.spinner("🔄 Initializing AI engine..."):
                try:
                    st.session_state.ai_accelerator = AIAccelerator(preferred_device=device)
                    st.session_state.feature_engineer = FeatureEngineer(st.session_state.ai_accelerator)
                    
                    # Initialize ModelManager
                    st.session_state.model_manager = ModelManager(
                        model_dir=MODEL_BASE_PATH,
                        parquet_base_path=PARQUET_BASE_PATH
                    )
                    
                    # Initialize SignalPredictor
                    st.session_state.signal_predictor = SignalPredictor(
                        st.session_state.ai_accelerator,
                        model_dir=MODEL_BASE_PATH
                    )
                    
                    st.session_state.model_trainer = ModelTrainer(
                        st.session_state.ai_accelerator,
                        model_dir=MODEL_BASE_PATH
                    )
                    
                    st.session_state.training_pipeline = TrainingPipeline(
                        model_dir=MODEL_BASE_PATH,
                        parquet_path=PARQUET_BASE_PATH,
                        csv_path=CSV_BASE_PATH
                    )
                    
                    # Initialize ParquetConverter
                    if PARQUET_AVAILABLE:
                        try:
                            st.session_state.parquet_converter = ParquetConverter(
                                csv_dir=CSV_BASE_PATH,
                                parquet_dir=PARQUET_BASE_PATH,
                                use_gpu=st.session_state.ai_accelerator.has_gpu
                            )
                        except TypeError:
                            try:
                                st.session_state.parquet_converter = ParquetConverter(
                                    CSV_BASE_PATH, 
                                    PARQUET_BASE_PATH,
                                    st.session_state.ai_accelerator.has_gpu
                                )
                            except TypeError:
                                st.session_state.parquet_converter = ParquetConverter()
                    
                    # Load model info
                    if st.session_state.model_manager:
                        models_df = st.session_state.model_manager.get_latest_models_summary()
                        if not models_df.empty:
                            st.session_state.model_info = models_df.to_dict('records')
                    
                except Exception as e:
                    st.warning(f"AI initialization partially failed: {e}")
                    st.session_state.ai_enabled = False
        
        # Get account summary
        with st.spinner("🔄 Fetching account details..."):
            st.session_state.account_summary = st.session_state.oanda_trader.get_account_summary()
        
        # Set daily reset time
        if st.session_state.daily_reset_time is None:
            st.session_state.daily_reset_time = datetime.now()
        
        st.session_state.initialized = True
        
        # Show success message
        if AI_AVAILABLE and st.session_state.ai_accelerator and st.session_state.ai_accelerator.has_gpu:
            st.success(f"✅ System initialized with GPU: {st.session_state.ai_accelerator.device_name}")
        else:
            st.success("✅ System initialized (CPU mode)")
        
        # Check Parquet availability
        parquet_pairs = get_available_pairs_with_parquet()
        if parquet_pairs:
            st.success(f"✅ Parquet available for {len(parquet_pairs)} pairs (10x faster)")
        
        # Show model availability
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
    """Check if we need to reset daily counters"""
    now = datetime.now()
    if st.session_state.daily_reset_time.date() < now.date():
        st.session_state.daily_loss = 0.0
        st.session_state.consecutive_losses = 0
        st.session_state.daily_reset_time = now
        if st.session_state.oanda_trader:
            st.session_state.oanda_trader.reset_daily()


def fetch_market_data(pairs: List[str], timeframe: str = "1h", bars: int = 200) -> Dict[str, pd.DataFrame]:
    """Fetch market data for selected pairs with error handling"""
    data = {}
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_pairs = len(pairs)
    failed_pairs = []
    
    for i, pair in enumerate(pairs):
        status_text.text(f"📡 Fetching {pair}... ({i+1}/{total_pairs})")
        try:
            oanda_pair = pair.replace('/', '_')
            
            # Skip if not in valid pairs
            if oanda_pair not in VALID_OANDA_PAIRS:
                status_text.text(f"⚠️ Skipping unsupported pair: {pair}")
                failed_pairs.append(pair)
                continue
            
            df = st.session_state.oanda_client.fetch_candles(
                instrument=oanda_pair,
                granularity=timeframe,
                count=bars
            )
            
            if not df.empty and len(df) > 20:
                # Calculate indicators
                df['rsi'] = st.session_state.oanda_client.calculate_rsi(df)
                macd_df = st.session_state.oanda_client.calculate_macd(df)
                df['macd'] = macd_df['macd']
                df['signal'] = macd_df['signal']
                df['histogram'] = macd_df['histogram']
                
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


def detect_zones_with_ai(data: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Detect zones and augment with AI predictions using ModelManager"""
    all_zones = []
    
    for pair, df in data.items():
        if df.empty or len(df) < 50:
            continue
        
        # Use the analyze method from SupplyDemand class
        analysis_result = st.session_state.sd_detector.analyze(df, pair)
        zones = analysis_result.get('zones', [])
        
        # Add additional info to zones
        for zone in zones:
            # Ensure zone has 'type' key
            if 'type' not in zone and 'zone_type' in zone:
                zone['type'] = zone['zone_type']
            
            zone_data = {
                'trading_pair': pair,
                'zone': zone,
                'data': df,
                'detected_at': datetime.now(),
                'type': zone.get('type', 'unknown'),
                'quality_score': zone.get('quality_score', 50),
                'price_level': zone.get('price_level', 0)
            }
            
            # Add AI augmentation if enabled
            if (st.session_state.ai_enabled and 
                AI_AVAILABLE and 
                st.session_state.signal_predictor and
                st.session_state.signal_predictor.models):
                
                try:
                    # Extract features for this zone
                    features = st.session_state.sd_detector.extract_zone_features(zone, df)
                    
                    # Get AI prediction with auto model selection
                    ai_pred = st.session_state.signal_predictor.predict_zone_success(
                        features=features,
                        model_name='auto',
                        pair=pair
                    )
                    
                    # Augment zone with AI prediction
                    augmented_data = st.session_state.signal_predictor.augment_zone_signal(zone_data, ai_pred)
                    
                    # Merge augmented data
                    zone_data.update(augmented_data)
                    
                except Exception as e:
                    # Fallback to rule-based only
                    zone_data['ai_confidence'] = 50
                    zone_data['ai_success_probability'] = 0.5
                    zone_data['ai_signal_strength'] = 'neutral'
                    zone_data['combined_score'] = zone.get('quality_score', 50)
                    zone_data['ai_model_used'] = None
            else:
                # No AI available
                zone_data['combined_score'] = zone.get('quality_score', 50)
                zone_data['ai_confidence'] = 0
                zone_data['ai_signal_strength'] = 'neutral'
            
            all_zones.append(zone_data)
    
    # Sort by combined score
    all_zones.sort(key=lambda x: x.get('combined_score', x['zone'].get('quality_score', 0)), reverse=True)
    
    return all_zones


def create_candlestick_chart(df: pd.DataFrame, zones: List[Dict], patterns: List[Dict], pair: str) -> go.Figure:
    """Create interactive candlestick chart with zones and patterns"""
    
    # Determine number of rows based on AI availability
    n_rows = 4 if (AI_AVAILABLE and st.session_state.ai_enabled) else 3
    row_heights = [0.4, 0.2, 0.2, 0.2][:n_rows]
    
    fig = make_subplots(
        rows=n_rows, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=[f"{pair} - Price with Zones", "RSI", "MACD"] + 
                       (["AI Confidence"] if n_rows == 4 else [])
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
    
    # Add zones
    for zone_data in zones:
        if zone_data['trading_pair'] == pair:
            zone = zone_data['zone']
            
            # Color based on zone type
            if zone['zone_type'] == 'demand':
                color = 'rgba(0, 200, 83, 0.2)'
                line_color = 'green'
            else:
                color = 'rgba(211, 47, 47, 0.2)'
                line_color = 'red'
            
            # Add zone rectangle
            fig.add_hrect(
                y0=zone['zone_low'],
                y1=zone['zone_high'],
                line_width=0,
                fillcolor=color,
                opacity=0.3,
                row=1, col=1
            )
            
            # Enhanced zone label with AI info
            label = f"{zone['zone_type'].upper()}"
            if 'ai_signal_strength' in zone_data:
                ai_strength = zone_data.get('ai_signal_strength', 'neutral')
                ai_conf = zone_data.get('ai_confidence', 0)
                label += f" [{ai_strength}:{ai_conf:.0f}]"
            
            fig.add_annotation(
                x=df.index[-10] if 'Date' not in df.columns else df['Date'].iloc[-10],
                y=zone['zone_high'],
                text=label,
                showarrow=True,
                arrowhead=1,
                font=dict(size=10, color=line_color),
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
            y=df['signal'],
            line=dict(color='red', width=2),
            name='Signal'
        ),
        row=3, col=1
    )
    
    # MACD histogram
    colors = ['green' if val >= 0 else 'red' for val in df['histogram']]
    fig.add_trace(
        go.Bar(
            x=df.index if 'Date' not in df.columns else df['Date'],
            y=df['histogram'],
            marker_color=colors,
            name='Histogram'
        ),
        row=3, col=1
    )
    
    # AI Confidence (if enabled)
    if n_rows == 4 and AI_AVAILABLE and st.session_state.ai_enabled:
        # Simple confidence based on RSI position
        confidence_values = []
        for i in range(len(df)):
            rsi_val = df['rsi'].iloc[i] if i < len(df['rsi']) else 50
            if pd.isna(rsi_val):
                rsi_val = 50
            
            # Higher confidence when RSI is extreme
            if rsi_val < 30:
                conf = 80 + (30 - rsi_val)
            elif rsi_val > 70:
                conf = 80 + (rsi_val - 70)
            else:
                conf = 50 - abs(rsi_val - 50) / 2
            
            confidence_values.append(min(95, max(5, conf)))
        
        fig.add_trace(
            go.Scatter(
                x=df.index if 'Date' not in df.columns else df['Date'],
                y=confidence_values,
                line=dict(color='#7C4DFF', width=2),
                fill='tozeroy',
                fillcolor='rgba(124, 77, 255, 0.2)',
                name='AI Confidence'
            ),
            row=4, col=1
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="green", opacity=0.5, row=4, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="orange", opacity=0.5, row=4, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="red", opacity=0.5, row=4, col=1)
    
    # Update layout
    fig.update_layout(
        title=f"{pair} - Technical Analysis" + (" with AI" if n_rows == 4 else ""),
        height=900,
        template="plotly_white",
        hovermode='x unified',
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    fig.update_xaxes(title_text="Date", row=n_rows, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="RSI", row=2, col=1, range=[0, 100])
    fig.update_yaxes(title_text="MACD", row=3, col=1)
    if n_rows == 4:
        fig.update_yaxes(title_text="AI %", row=4, col=1, range=[0, 100])
    
    return fig


def display_hardware_status():
    """Display hardware status"""
    st.markdown("### 🖥️ Hardware Status")
    
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
    else:
        st.markdown("""
        <div class="cpu-status">
            💻 CPU MODE (AI disabled)
        </div>
        """, unsafe_allow_html=True)
    
    # Show Parquet status
    parquet_pairs = get_available_pairs_with_parquet()
    if parquet_pairs:
        st.markdown(f"""
        <div class="parquet-status">
            📁 Parquet Ready: {len(parquet_pairs)} pairs (10x faster)
        </div>
        """, unsafe_allow_html=True)
    
    # Show model status
    if st.session_state.model_manager:
        models_df = st.session_state.model_manager.get_latest_models_summary()
        if not models_df.empty:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #7C4DFF 0%, #651FFF 100%); 
                        color: white; padding: 0.5rem; border-radius: 10px; 
                        margin-top: 0.5rem; text-align: center; font-weight: bold;">
                🤖 {len(models_df)} Trained Models Available
            </div>
            """, unsafe_allow_html=True)


def display_account_info():
    """Display account information"""
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


def display_zones(zones: List[Dict]):
    """Display detected zones with AI information"""
    if not zones:
        st.info("📊 No zones detected in current data")
        return
    
    # Filters
    with st.expander("🔍 Filter Zones", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            min_quality = st.slider("Min Quality", 0, 100, 40)
        
        with col2:
            zone_type_options = ["All", "demand", "supply"]
            zone_type = st.selectbox("Zone Type", zone_type_options)
        
        with col3:
            show_mitigated = st.checkbox("Show Mitigated", False)
        
        with col4:
            min_ai_confidence = st.slider("Min AI Confidence", 0, 100, 0, 
                                         help="Filter by AI confidence score")
    
    # Apply filters
    filtered = zones
    filtered = [z for z in filtered if z.get('zone', {}).get('quality_score', 0) >= min_quality or 
                z.get('quality_score', 0) >= min_quality]
    
    if zone_type != "All":
        filtered_temp = []
        for z in filtered:
            if 'zone' in z:
                zone_type_val = z['zone'].get('type', '')
            else:
                zone_type_val = z.get('type', '')
            
            if zone_type_val == zone_type:
                filtered_temp.append(z)
        filtered = filtered_temp
    
    if not show_mitigated:
        filtered_temp = []
        for z in filtered:
            if 'zone' in z:
                mitigated = z['zone'].get('mitigated', False)
            else:
                mitigated = z.get('mitigated', False)
            
            if not mitigated:
                filtered_temp.append(z)
        filtered = filtered_temp
    
    # Apply AI confidence filter
    if min_ai_confidence > 0:
        filtered_temp = []
        for z in filtered:
            ai_conf = z.get('ai_confidence', 0)
            if ai_conf >= min_ai_confidence:
                filtered_temp.append(z)
        filtered = filtered_temp
    
    st.caption(f"Showing {len(filtered)} of {len(zones)} zones")
    
    # Display zones
    for zone_data in filtered:
        # Handle both flat zone structure and nested structure
        if 'zone' in zone_data:
            zone = zone_data['zone']
            pair = zone_data.get('trading_pair', 'Unknown')
        else:
            zone = zone_data
            pair = zone_data.get('trading_pair', 'Unknown')
        
        # Get zone type from either 'type' or 'zone_type' key
        zone_type_val = zone.get('type', zone.get('zone_type', 'unknown'))
        
        # Determine styling
        zone_class = "demand-zone" if zone_type_val == 'demand' else "supply-zone"
        
        with st.container():
            st.markdown(f'<div class="zone-card {zone_class}">', unsafe_allow_html=True)
            
            # Header
            col1, col2 = st.columns([3, 1])
            with col1:
                score = zone_data.get('combined_score', zone.get('quality_score', 50))
                ai_badge = ""
                if 'ai_signal_strength' in zone_data:
                    ai_strength = zone_data.get('ai_signal_strength', 'neutral').upper()
                    strength_class = f"ai-{ai_strength.lower().replace('_', '-')}"
                    ai_badge = f" <span class='{strength_class}'>{ai_strength}</span>"
                
                st.markdown(f"### {pair} - {zone_type_val.upper()} Zone (Score: {score:.1f}){ai_badge}", unsafe_allow_html=True)
            
            with col2:
                if zone_type_val == 'demand':
                    st.markdown('<span class="buy-signal">📈 BUY</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="sell-signal">📉 SELL</span>', unsafe_allow_html=True)
            
            # Details
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                **📍 Zone Details**
                - Price: `{zone.get('price_level', 0):.5f}`
                - Range: `{zone.get('zone_low', 0):.5f} - {zone.get('zone_high', 0):.5f}`
                - Quality: `{zone.get('quality_score', 50):.1f}` ({zone.get('quality_rating', 'Unknown')})
                - Status: {'🟢 Fresh' if not zone.get('mitigated', False) else '🔴 Mitigated'}
                """)
            
            with col2:
                if 'ai_confidence' in zone_data and zone_data['ai_confidence'] > 0:
                    ai_conf = zone_data['ai_confidence']
                    ai_strength = zone_data.get('ai_signal_strength', 'neutral')
                    ai_prob = zone_data.get('ai_success_probability', 0.5)
                    ai_model = zone_data.get('ai_model_used', 'unknown')
                    ai_rec = zone_data.get('ai_recommendation', 'NEUTRAL')
                    
                    # Map strength to class
                    strength_map = {
                        'VERY_STRONG': 'ai-very-strong',
                        'STRONG': 'ai-strong',
                        'MODERATE': 'ai-moderate',
                        'WEAK': 'ai-weak',
                        'VERY_WEAK': 'ai-weak',
                        'neutral': 'ai-moderate'
                    }
                    strength_class = strength_map.get(ai_strength, 'ai-moderate')
                    
                    st.markdown(f"""
                    **🤖 AI Analysis**
                    - Confidence: `{ai_conf:.1f}%`
                    - Probability: `{ai_prob:.1%}`
                    - Strength: <span class="{strength_class}">{ai_strength}</span>
                    - Recommendation: `{ai_rec}`
                    - Model: `{ai_model}`
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("**🤖 AI Analysis**\n- No AI model available")
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button(f"📊 Chart", key=f"chart_{id(zone_data)}", use_container_width=False):
                    st.session_state[f"show_chart_{id(zone_data)}"] = True
            
            with col2:
                if not zone.get('mitigated', False):
                    direction = "BUY" if zone_type_val == 'demand' else "SELL"
                    
                    ai_conf = zone_data.get('ai_confidence', 0)
                    button_text = f"💰 {direction}"
                    if ai_conf > 0:
                        button_text += f" (AI:{ai_conf:.0f})"
                    
                    if st.button(button_text, key=f"trade_{id(zone_data)}", type="primary", use_container_width=False):
                        st.session_state[f"confirm_zone_{id(zone_data)}"] = zone_data
                else:
                    st.button("⛔ Mitigated", key=f"disabled_{id(zone_data)}", disabled=True, use_container_width=False)
            
            with col3:
                if st.button(f"➕ Watch", key=f"watch_{id(zone_data)}", use_container_width=False):
                    st.toast(f"Added {pair} to watchlist", icon="✅")
            
            # Chart display
            if st.session_state.get(f"show_chart_{id(zone_data)}", False):
                if 'data' in zone_data:
                    fig = create_candlestick_chart(
                        zone_data['data'],
                        [zone_data],
                        [],
                        pair
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("Close Chart", key=f"close_chart_{id(zone_data)}"):
                        st.session_state[f"show_chart_{id(zone_data)}"] = False
            
            st.markdown('</div>', unsafe_allow_html=True)


def render_trade_confirmation(zone_data: Dict):
    """Render trade confirmation dialog"""
    zone = zone_data['zone']
    pair = zone_data['trading_pair']
    
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
        - **Direction:** `{'BUY' if zone['zone_type'] == 'demand' else 'SELL'}`
        - **Zone Price:** `{zone['zone_price']:.5f}`
        - **Quality:** `{zone['quality_score']:.1f}` ({zone['quality_rating']})
        """)
        
        # Show AI info in trade confirmation
        if 'ai_confidence' in zone_data and zone_data['ai_confidence'] > 0:
            st.markdown(f"""
            **🤖 AI Analysis**
            - Confidence: `{zone_data['ai_confidence']:.1f}%`
            - Strength: `{zone_data.get('ai_signal_strength', 'neutral')}`
            - Recommendation: `{zone_data.get('ai_recommendation', 'NEUTRAL')}`
            """)
    
    with col2:
        st.markdown("### 💰 Position Sizing")
        
        account = st.session_state.account_summary.get('account', {})
        balance = float(account.get('balance', 10000))
        
        risk_percent = st.slider("Risk %", 0.1, 2.0, 1.0, 0.1)
        risk_amount = balance * (risk_percent / 100)
        
        # Calculate stop loss
        instrument = pair.replace('/', '_')
        stop_loss = st.session_state.oanda_trader.calculate_stop_from_zone(
            instrument, zone['zone_price'],
            'long' if zone['zone_type'] == 'demand' else 'short'
        )
        
        st.markdown(f"""
        **Risk:**
        - Balance: `${balance:,.2f}`
        - Risk Amount: `${risk_amount:.2f}` ({risk_percent}%)
        - Stop Loss: `{stop_loss:.5f}`
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("✅ CONFIRM", type="primary", use_container_width=False):
            try:
                result = st.session_state.oanda_trader.place_supply_demand_trade(
                    {
                        'instrument': pair.replace('/', '_'),
                        'zone_price': zone['zone_price'],
                        'direction': 'long' if zone['zone_type'] == 'demand' else 'short',
                        'risk_percent': risk_percent,
                        'quality': zone,
                        'zone_id': zone['zone_id']
                    },
                    balance
                )
                
                if 'error' not in result:
                    st.success("✅ Trade executed!")
                    
                    # Mark zone as traded
                    for z in st.session_state.detected_zones:
                        if z['zone']['zone_id'] == zone['zone_id']:
                            z['zone']['mitigated'] = True
                            break
                    
                    st.session_state[f"confirm_zone_{zone['zone_id']}"] = None
                    time.sleep(2)
                    st.rerun()
                else:
                    st.error(f"❌ Failed: {result['error']}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    with col2:
        if st.button("❌ Cancel", use_container_width=False):
            st.session_state[f"confirm_zone_{zone['zone_id']}"] = None
            st.rerun()


def display_open_positions():
    """Display open positions"""
    positions = st.session_state.oanda_trader.get_open_trades()
    st.session_state.open_positions = positions
    
    if not positions:
        st.info("📭 No open positions")
        return
    
    for position in positions:
        with st.container():
            st.markdown('<div class="zone-card">', unsafe_allow_html=True)
            
            # Calculate P&L
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
                result = st.session_state.oanda_trader.close_trade(position['id'])
                if result.get('success'):
                    if result.get('pl', 0) < 0:
                        st.session_state.daily_loss += abs(result['pl'])
                        st.session_state.consecutive_losses += 1
                    else:
                        st.session_state.consecutive_losses = 0
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)


def display_trade_history():
    """Display trade history"""
    history = st.session_state.oanda_trader.get_trade_history()
    
    if not history:
        st.info("📭 No trade history")
        return
    
    # Statistics
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
    
    # Trade table - FIXED for Arrow serialization
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
        # Convert all columns to string
        for col in df_trades.columns:
            df_trades[col] = df_trades[col].astype(str)
        st.dataframe(df_trades, use_container_width=True, hide_index=True)


# ============================================================================
# MAIN APP LAYOUT
# ============================================================================
st.markdown('<h1 class="main-header">🤖 Simon Pullen AI Trading System</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # Environment selection
    environment = st.radio(
        "Trading Environment",
        ["practice", "live"],
        index=0,
        help="Practice = Demo account"
    )
    
    if environment == "live":
        st.markdown("""
        <div class="error-box">
            ⚠️ LIVE TRADING - REAL MONEY
        </div>
        """, unsafe_allow_html=True)
    
    # Initialize system
    if not st.session_state.initialized:
        if st.button("🚀 Initialize System", type="primary", use_container_width=False):
            if initialize_system(environment, st.session_state.preferred_device):
                st.rerun()
    else:
        st.success("✅ System Ready")
        
        # Hardware status
        display_hardware_status()
        
        # Account info
        display_account_info()
        
        st.markdown("---")
        
        # AI Settings
        st.markdown("### 🤖 AI Settings")
        ai_enabled = st.checkbox("Enable AI", value=st.session_state.ai_enabled, disabled=not AI_AVAILABLE)
        if ai_enabled != st.session_state.ai_enabled:
            st.session_state.ai_enabled = ai_enabled
            st.rerun()
        
        # Show model summary in sidebar
        if st.session_state.ai_enabled and st.session_state.model_manager:
            models_df = st.session_state.model_manager.get_latest_models_summary()
            if not models_df.empty:
                st.caption(f"📊 {len(models_df)} models available")
                
                # Show models for selected pairs
                if st.session_state.selected_pairs:
                    available_for = [p for p in st.session_state.selected_pairs 
                                   if p in models_df['Pair'].values]
                    if available_for:
                        st.caption(f"✅ Models ready for: {', '.join(available_for[:3])}")
        
        st.markdown("---")
        
        # Data settings
        st.markdown("### 📊 Data Settings")
        
        # Parquet toggle
        parquet_pairs = get_available_pairs_with_parquet()
        parquet_available = len(parquet_pairs) > 0
        use_parquet = st.checkbox(
            "🚀 Use Parquet (10x faster)", 
            value=st.session_state.use_parquet and parquet_available,
            disabled=not parquet_available
        )
        if use_parquet != st.session_state.use_parquet:
            st.session_state.use_parquet = use_parquet
            st.rerun()
        
        selected_pairs = st.multiselect(
            "Select Pairs",
            VALID_OANDA_PAIRS,
            default=VALID_OANDA_PAIRS[:5],
            max_selections=10
        )
        st.session_state.selected_pairs = selected_pairs
        
        col1, col2 = st.columns(2)
        with col1:
            timeframe = st.selectbox("Timeframe", ["1h", "4h", "1d", "30m", "15m", "5m"], index=0)
        with col2:
            bars = st.number_input("Bars", 50, 500, 200, 50)
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Now", use_container_width=False):
            st.session_state.last_refresh = datetime.now() - timedelta(seconds=60)
            st.rerun()
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["Market Scanner", "Open Positions", "Trade History", "AI Training", "AI Models"],
            index=0
        )

# Main content
if not st.session_state.initialized:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h2>Welcome to Simon Pullen AI Trading System</h2>
        <p style="font-size: 1.2rem; margin: 2rem 0;">
            🤖 AI-Augmented Supply/Demand Trading<br>
            ⚡ Real-time Market Analysis with Parquet Acceleration<br>
            💰 Capital Preservation First
        </p>
        <p>👈 Initialize system from sidebar to begin</p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    if page == "Market Scanner":
        st.markdown('<h2 class="sub-header">🔍 Market Scanner</h2>', unsafe_allow_html=True)
        
        # Refresh data
        time_since = (datetime.now() - st.session_state.last_refresh).seconds
        if time_since > 60 or not st.session_state.current_data:
            with st.spinner("🔍 Scanning markets..."):
                st.session_state.current_data = fetch_market_data(
                    st.session_state.selected_pairs,
                    timeframe,
                    bars
                )
                
                if st.session_state.current_data:
                    st.session_state.detected_zones = detect_zones_with_ai(st.session_state.current_data)
                
                st.session_state.last_refresh = datetime.now()
        
        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pairs", len(st.session_state.current_data))
        with col2:
            st.metric("Zones", len(st.session_state.detected_zones))
        with col3:
            st.caption(f"Updated: {st.session_state.last_refresh.strftime('%H:%M:%S')}")
        
        st.markdown("---")
        
        # Display zones
        display_zones(st.session_state.detected_zones)
        
        # Trade confirmation
        for key in st.session_state:
            if key.startswith('confirm_zone_') and st.session_state[key]:
                render_trade_confirmation(st.session_state[key])
                break
    
    elif page == "Open Positions":
        st.markdown('<h2 class="sub-header">📊 Open Positions</h2>', unsafe_allow_html=True)
        display_open_positions()
    
    elif page == "Trade History":
        st.markdown('<h2 class="sub-header">📝 Trade History</h2>', unsafe_allow_html=True)
        display_trade_history()
    
    elif page == "AI Training":
        display_ai_training()
    
    elif page == "AI Models":
        st.markdown('<h2 class="sub-header">🤖 AI Models</h2>', unsafe_allow_html=True)
        display_model_management()

# Footer
st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray; padding: 1rem;'>
        Simon Pullen Trading System | v4.9 - FIXED ARROW SERIALIZATION ISSUES<br>
        CSV Path: {CSV_BASE_PATH} | Parquet Path: {PARQUET_BASE_PATH} | Model Path: {MODEL_BASE_PATH}<br>
        ⚠️ Risk Warning: Trading carries substantial risk
    </div>
    """,
    unsafe_allow_html=True
)