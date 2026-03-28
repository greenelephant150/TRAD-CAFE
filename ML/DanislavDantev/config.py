"""
Configuration file for Simon Pullen AI Trading System
Centralizes all paths and settings
"""

from pathlib import Path
import os

# =============================================================================
# PATHS - UPDATE THESE TO MATCH YOUR SYSTEM
# =============================================================================

# Base directories
PROJECT_ROOT = Path(__file__).parent.absolute()

# Model paths
MODEL_PATH = Path("/mnt2/Trading-Cafe/ML/SPullen/ai/trained_models/")
AI_PATH = PROJECT_ROOT / "ai"

# Data paths
CSV_PATH = Path("/home/grcf/Forex")
PARQUET_PATH = Path("/home/grcf/Forex_Parquet")

# =============================================================================
# TRAINING DEFAULTS
# =============================================================================

DEFAULT_LOOKBACK = 20
DEFAULT_TRAIN_SIZE = 100000
DEFAULT_VALIDATION_SPLIT = 0.2
DEFAULT_TEST_SIZE = 0.2
RANDOM_STATE = 42

# =============================================================================
# AVAILABLE PAIRS (Auto-discovered or manual list)
# =============================================================================

# You can set this to None to auto-discover from parquet files
MANUAL_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD",
    "USD_CHF", "NZD_USD", "EUR_GBP", "EUR_JPY", "GBP_JPY",
    "AUD_JPY", "EUR_AUD", "GBP_AUD", "NZD_JPY", "USD_SGD",
    "EUR_CHF", "GBP_CHF", "AUD_CHF", "CAD_JPY", "CHF_JPY"
]

# =============================================================================
# FEATURE CATEGORIES
# =============================================================================

FEATURE_CATEGORIES = {
    'Price Features': ['open', 'high', 'low', 'close', 'volume'],
    'Technical Indicators': ['rsi', 'macd', 'bb_upper', 'bb_lower', 'bb_middle'],
    'Oscillators': ['stoch_k', 'stoch_d', 'cci', 'williams_r'],
    'Trend Indicators': ['ema_9', 'ema_21', 'sma_50', 'adx'],
    'Volatility': ['atr', 'natr', 'volatility']
}

# =============================================================================
# GPU / CPU SETTINGS
# =============================================================================

USE_GPU = True  # Set to False to force CPU
GPU_MEMORY_FRACTION = 0.95  # Use 95% of GPU memory
CPU_THREADS = 32  # Number of threads for CPU processing

# =============================================================================
# STREAMLIT UI SETTINGS
# =============================================================================

APP_TITLE = "Simon Pullen AI Trading System"
APP_VERSION = "v4.7 - Added AI Model Management Integration"
RISK_WARNING = "⚠️ Risk Warning: Trading carries substantial risk"

# =============================================================================
# CREATE DIRECTORIES IF THEY DON'T EXIST
# =============================================================================

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [MODEL_PATH, AI_PATH, CSV_PATH, PARQUET_PATH]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Ensured directory: {directory}")

# Auto-create directories when config is imported
ensure_directories()