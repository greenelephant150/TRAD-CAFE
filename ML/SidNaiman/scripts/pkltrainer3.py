#!/usr/bin/env python3
"""
🎯 SID METHOD AI MODEL TRAINER - GPU ACCELERATED
========================================================
Trains models for ALL available pairs using XGBoost GPU
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Suppress warnings
warnings.filterwarnings('ignore')

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
        def __iter__(self): return iter(self.iterable)
        def update(self, n=1): pass
        def close(self): pass
        def set_description(self, desc): self.desc = desc
        def set_postfix(self, **kwargs): pass
    trange = lambda *args, **kwargs: tqdm(range(*args), **kwargs)

# Import Sid Method
try:
    from sid_method import SidMethod
    SID_AVAILABLE = True
    print(f"✅ sid_method.py imported successfully")
except ImportError as e:
    SID_AVAILABLE = False
    print(f"❌ sid_method.py not found: {e}")
    sys.exit(1)

# ============================================================================
# GPU DETECTION
# ============================================================================
GPU_AVAILABLE = False
CUML_AVAILABLE = False
XGBOOST_AVAILABLE = False
GPU_COUNT = 0

# Set CUDA environment for GPU
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.3'
os.environ['CUDA_PATH'] = '/usr/local/cuda-12.3'
os.environ['CUDA_ROOT'] = '/usr/local/cuda-12.3'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.3/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

try:
    import cupy as cp
    GPU_COUNT = cp.cuda.runtime.getDeviceCount()
    GPU_AVAILABLE = True
    print(f"✅ CuPy available: {GPU_COUNT} GPU(s) found")
    for i in range(GPU_COUNT):
        with cp.cuda.Device(i):
            free, total = cp.cuda.runtime.memGetInfo()
            print(f"   GPU {i}: {total/1024**3:.1f}GB total, {free/1024**3:.1f}GB free")
except ImportError:
    print(f"⚠️ CuPy not available - GPU acceleration disabled")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
    print(f"✅ XGBoost available: {xgb.__version__}")
except ImportError:
    print(f"⚠️ XGBoost not available")

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cumlRandomForest
    CUML_AVAILABLE = True
    print(f"✅ cuML available for GPU-accelerated ML")
except ImportError:
    print(f"⚠️ cuML not available - using CPU for ML")

# ============================================================================
# COLORFUL TERMINAL OUTPUT
# ============================================================================
class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    MAGENTA = '\033[35m'; BOLD = '\033[1m'; END = '\033[0m'
    CHECK = "✅"; CROSS = "❌"; WARN = "⚠️"; FIRE = "🔥"
    BRAIN = "🧠"; DISK = "💾"; CLOCK = "⏱️"; GRAPH = "📈"
    MODEL = "🤖"; DATA = "📊"; GPU_ICON = "🎮"

def print_header(text): print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}\n{Colors.CYAN}{Colors.BOLD}{text.center(60)}{Colors.END}\n{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}\n")
def print_success(text): print(f"{Colors.GREEN}{Colors.BOLD}✅ {text}{Colors.END}")
def print_warning(text): print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ {text}{Colors.END}")
def print_error(text): print(f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.END}")
def print_info(text): print(f"{Colors.BLUE}ℹ️ {text}{Colors.END}")
def print_gpu(text): print(f"{Colors.CYAN}{Colors.BOLD}{Colors.GPU_ICON} {text}{Colors.END}")

# ============================================================================
# CONFIGURATION
# ============================================================================
PARQUET_BASE_PATH = "/home/grct/Forex_Parquet"
MODEL_OUTPUT_PATH = "/mnt2/Trading-Cafe/ML/SNaiman2/ai/trained_models/"

os.makedirs(PARQUET_BASE_PATH, exist_ok=True)
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

print_info(f"Parquet path: {PARQUET_BASE_PATH}")
print_info(f"Model output path: {MODEL_OUTPUT_PATH}")

# CPU ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    SKLEARN_AVAILABLE = True
    print_success("scikit-learn available (CPU)")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print_warning(f"scikit-learn not available: {e}")

# ============================================================================
# GPU MEMORY MANAGEMENT
# ============================================================================
class GPUMemoryManager:
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self.initialized = False
        
    def activate(self):
        if GPU_AVAILABLE:
            self.device = cp.cuda.Device(self.gpu_id)
            self.device.use()
            self.initialized = True
            free, total = cp.cuda.runtime.memGetInfo()
            print_gpu(f"GPU {self.gpu_id} activated: {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
            return True
        return False
    
    def cleanup(self):
        if self.initialized and GPU_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()
            cp.cuda.Stream().synchronize()
            gc.collect()
    
    def get_free_memory(self):
        if GPU_AVAILABLE:
            with cp.cuda.Device(self.gpu_id):
                free, total = cp.cuda.runtime.memGetInfo()
                return free / (1024**2)
        return 0

gpu_manager = GPUMemoryManager(gpu_id=0)

# ============================================================================
# DATA FUNCTIONS
# ============================================================================
def get_available_pairs() -> List[str]:
    """Get list of all trading pairs with parquet data"""
    pairs = []
    if os.path.exists(PARQUET_BASE_PATH):
        for item in os.listdir(PARQUET_BASE_PATH):
            pair_path = os.path.join(PARQUET_BASE_PATH, item)
            if os.path.isdir(pair_path):
                for subdir in os.listdir(pair_path):
                    if subdir.startswith('year=') and not subdir.startswith('__HIVE'):
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
        if '__HIVE_DEFAULT_PARTITION__' in root:
            continue
            
        for file in files:
            if file.endswith('.parquet'):
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
    
    for pair in pairs:
        earliest, _ = get_pair_date_range(pair)
        if earliest and (earliest_overall is None or earliest < earliest_overall):
            earliest_overall = earliest
    
    if earliest_overall is None:
        earliest_overall = datetime.now() - timedelta(days=1825)
    
    return earliest_overall

def get_latest_parquet_date() -> datetime:
    """Get the latest date across all parquet files"""
    latest_overall = None
    
    if not os.path.exists(PARQUET_BASE_PATH):
        return datetime.now()
    
    pairs = get_available_pairs()
    
    for pair in pairs:
        _, latest = get_pair_date_range(pair)
        if latest and (latest_overall is None or latest > latest_overall):
            latest_overall = latest
    
    if latest_overall is None:
        latest_overall = datetime.now()
    
    return latest_overall

def load_parquet_data_full(pair: str, verbose: bool = True) -> pd.DataFrame:
    """Load ALL available parquet data for a pair"""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    if not os.path.exists(pair_path):
        print_error(f"Pair path not found: {pair_path}")
        return pd.DataFrame()
    
    # Get date range for this pair
    earliest, latest = get_pair_date_range(pair)
    if earliest and latest:
        print_info(f"Date range for {pair}: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
        print_info(f"Total days: {(latest - earliest).days:,} days")
    
    try:
        if verbose:
            print_info(f"Reading parquet files for {pair}...")
        
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
        
        dfs = []
        if TQDM_AVAILABLE and len(all_files) > 10:
            iterator = tqdm(all_files, desc="  Loading files", unit="file")
        else:
            iterator = all_files
        
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
        
        df = pd.concat(dfs, ignore_index=False)
        
        # The index is already a datetime from the parquet files
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df.dropna()
            except:
                print_error(f"Cannot convert index to datetime for {pair}")
                return pd.DataFrame()
        
        df = df.sort_index()
        
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                print_error(f"Missing required column: {col}")
                return pd.DataFrame()
        
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=required_cols)
        
        # Resample to 1-hour bars
        if len(df) > 0:
            df = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
        
        if verbose:
            print_info(f"Loaded {len(df):,} rows for {pair}")
            if len(df) > 0:
                print_info(f"  Date range: {df.index.min().strftime('%Y-%m-%d %H:%M')} to {df.index.max().strftime('%Y-%m-%d %H:%M')}")
        
        return df
        
    except Exception as e:
        print_error(f"Error loading {pair}: {e}")
        return pd.DataFrame()

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================
def train_xgboost_gpu(X_train, y_train, X_val, y_val, verbose=True):
    """Train XGBoost with GPU acceleration"""
    try:
        import xgboost as xgb
        
        if verbose:
            print_gpu("Training XGBoost with GPU...")
        
        if hasattr(X_train, 'get'):
            X_train_np = X_train.get()
            X_val_np = X_val.get()
        else:
            X_train_np = np.asarray(X_train)
            X_val_np = np.asarray(X_val)
        
        dtrain = xgb.DMatrix(X_train_np, label=y_train)
        dval = xgb.DMatrix(X_val_np, label=y_val)
        
        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',
            'device': 'cuda:0',
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbosity': 0
        }
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=100,
            evals=[(dval, 'eval')],
            early_stopping_rounds=10,
            verbose_eval=False
        )
        
        y_pred_proba = model.predict(dval)
        y_pred = (y_pred_proba > 0.5).astype(int)
        accuracy = accuracy_score(y_val, y_pred)
        
        if verbose:
            print_success(f"XGBoost GPU accuracy: {accuracy:.4f}")
        
        return model, accuracy
        
    except Exception as e:
        if verbose:
            print_warning(f"XGBoost GPU failed: {e}")
        return None, 0

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
    
    if verbose:
        print_success(f"Gradient Boosting accuracy: {accuracy:.4f}")
    
    return model, accuracy

# ============================================================================
# FEATURE CALCULATION
# ============================================================================
def calculate_sid_features(df: pd.DataFrame, sid: SidMethod, verbose: bool = True) -> pd.DataFrame:
    """Calculate features (vectorized CPU operations)"""
    df = df.copy()
    
    required = ['open', 'high', 'low', 'close']
    for col in required:
        if col not in df.columns:
            print_error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    for col in required:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=required)
    
    if len(df) < 100:
        print_error(f"Insufficient data after cleaning: {len(df)} rows")
        return pd.DataFrame()
    
    if verbose:
        print_info("Step 1/6: Calculating RSI...")
    df['rsi'] = sid.calculate_rsi(df, desc="RSI")
    
    if verbose:
        print_info("Step 2/6: Calculating MACD...")
    macd_df = sid.calculate_macd(df, desc="MACD")
    df['macd'] = macd_df['macd']
    df['macd_signal'] = macd_df['signal']
    df['macd_hist'] = macd_df['histogram']
    
    if verbose:
        print_info("Step 3/6: Calculating RSI signals...")
    
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_mid'] = ((df['rsi'] >= 45) & (df['rsi'] <= 55)).astype(int)
    df['rsi_change'] = df['rsi'].diff()
    df['rsi_change_3'] = df['rsi'].diff(3)
    df['rsi_change_5'] = df['rsi'].diff(5)
    
    if verbose:
        print_info("Step 4/6: Calculating MACD signals...")
    
    df['macd_cross_above'] = ((df['macd'].shift(1) <= df['macd_signal'].shift(1)) & 
                               (df['macd'] > df['macd_signal'])).astype(int)
    df['macd_cross_below'] = ((df['macd'].shift(1) >= df['macd_signal'].shift(1)) & 
                               (df['macd'] < df['macd_signal'])).astype(int)
    df['macd_aligned_up'] = (df['macd'] > df['macd'].shift(1)).astype(int)
    df['macd_aligned_down'] = (df['macd'] < df['macd'].shift(1)).astype(int)
    
    if verbose:
        print_info("Step 5/6: Calculating price action...")
    
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    df['total_range'] = df['high'] - df['low']
    df['body_ratio'] = df['body_size'] / df['total_range'].replace(0, 1)
    df['returns'] = df['close'].pct_change()
    df['returns_abs'] = abs(df['returns'])
    
    if verbose:
        print_info("Step 6/6: Calculating volatility and targets...")
    
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
    
    if 'volume' in df.columns:
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # Targets
    df['target_rsi_50_5'] = 0
    df['target_rsi_50_10'] = 0
    df['target_direction'] = 0
    
    for i in range(len(df) - 10):
        if any(df['rsi'].iloc[i+1:i+6] >= 50):
            df.loc[df.index[i], 'target_rsi_50_5'] = 1
        if any(df['rsi'].iloc[i+1:i+11] >= 50):
            df.loc[df.index[i], 'target_rsi_50_10'] = 1
    
    df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    df = df.dropna()
    
    print_info(f"Features calculated: {len(df):,} rows, {len(df.columns)} columns")
    return df

def create_training_samples(df: pd.DataFrame, lookback: int = 10, 
                            target_col: str = 'target_direction',
                            max_samples: int = 200000) -> Tuple[np.ndarray, np.ndarray]:
    """Create training samples"""
    exclude_cols = ['target_rsi_50_5', 'target_rsi_50_10', 'target_direction']
    feature_cols = [col for col in df.columns if col not in exclude_cols and 
                    col not in ['open', 'high', 'low', 'close', 'volume'] and
                    df[col].dtype in ['float64', 'float32', 'int64']]
    
    if len(df) < lookback + 10:
        print_warning(f"Insufficient data: {len(df)} rows")
        return np.array([]), np.array([])
    
    print_info(f"Using {len(feature_cols)} features, lookback={lookback}")
    
    X_list = []
    y_list = []
    
    step = max(1, (len(df) - lookback) // max_samples)
    total_samples = (len(df) - lookback) // step
    print_info(f"Creating {total_samples:,} samples (step={step})")
    
    for i in range(lookback, len(df) - 1, step):
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
    
    return X, y

# ============================================================================
# MAIN TRAINING FUNCTIONS
# ============================================================================
def train_pair(pair: str, target: str = 'target_direction', 
               lookback: int = 10, force: bool = False, 
               use_gpu: bool = True, verbose: bool = True) -> Optional[Dict]:
    """Train a model for a specific pair using ALL available data"""
    
    print_header(f"{Colors.BRAIN} TRAINING SID MODEL FOR: {pair}")
    start_time = time.time()
    
    sid = SidMethod(verbose=verbose)
    
    df = load_parquet_data_full(pair, verbose=verbose)
    
    if df.empty:
        print_error(f"No data loaded for {pair}")
        return None
    
    print_info("Calculating Sid features...")
    df = calculate_sid_features(df, sid, verbose=verbose)
    
    if df.empty or len(df) < 100:
        print_error("Feature calculation failed")
        return None
    
    print_info(f"Creating training samples (lookback={lookback}, target={target})...")
    X, y = create_training_samples(df, lookback=lookback, target_col=target)
    
    if len(X) == 0:
        print_error("Failed to create training samples")
        return None
    
    print_info("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print_info(f"Training samples: {len(X_train):,}")
    print_info(f"Validation samples: {len(X_val):,}")
    
    models = {}
    accuracies = {}
    
    # Try XGBoost GPU first
    if use_gpu and XGBOOST_AVAILABLE:
        xgb_model, xgb_acc = train_xgboost_gpu(X_train, y_train, X_val, y_val, verbose)
        if xgb_model:
            models['xgboost_gpu'] = xgb_model
            accuracies['xgboost_gpu'] = xgb_acc
            print_success(f"✅ XGBoost GPU trained with accuracy: {xgb_acc:.4f}")
    
    # CPU fallback
    if not models:
        gb_model, gb_acc = train_gradient_boosting(X_train, y_train, X_val, y_val, verbose)
        if gb_model:
            models['gradient_boosting'] = gb_model
            accuracies['gradient_boosting'] = gb_acc
    
    if not models:
        print_error("No models trained successfully")
        return None
    
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    best_accuracy = accuracies[best_model_name]
    
    print_success(f"Best model: {best_model_name} with accuracy {best_accuracy:.4f}")
    
    timestamp = datetime.now().strftime('%d%m%Y')
    model_filename = f"{timestamp}--SidMethod--{pair}--{target}--L{lookback}.pkl"
    model_path = os.path.join(MODEL_OUTPUT_PATH, model_filename)
    
    model_package = {
        'model': best_model,
        'metadata': {
            'pair': pair,
            'training_date': datetime.now().isoformat(),
            'data_start': df.index.min().strftime('%Y-%m-%d'),
            'data_end': df.index.max().strftime('%Y-%m-%d'),
            'total_days': (df.index.max() - df.index.min()).days,
            'total_rows': len(df),
            'target': target,
            'lookback': lookback,
            'accuracy': float(best_accuracy),
            'samples': len(X),
            'method': 'sid_method',
            'best_model': best_model_name,
            'all_accuracies': accuracies,
            'gpu_used': 'gpu' in best_model_name if best_model_name else False
        }
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    file_size = os.path.getsize(model_path) / (1024*1024)
    print_success(f"Model saved: {model_filename} ({file_size:.1f} MB)")
    print_info(f"Total time: {time.time() - start_time:.1f}s")
    
    return {
        'pair': pair,
        'status': 'trained',
        'accuracy': best_accuracy,
        'samples': len(X),
        'best_model': best_model_name,
        'data_start': df.index.min().strftime('%Y-%m-%d'),
        'data_end': df.index.max().strftime('%Y-%m-%d'),
        'total_rows': len(df),
        'time': time.time() - start_time
    }

def train_all_pairs(pairs: List[str], target: str = 'target_direction',
                    lookback: int = 10, force: bool = False, 
                    use_gpu: bool = True, verbose: bool = True) -> Dict:
    """Train models for all available pairs"""
    
    results = {'success': [], 'failed': []}
    
    print_header(f"{Colors.FIRE} TRAINING SID MODELS FOR {len(pairs)} PAIRS")
    if use_gpu and XGBOOST_AVAILABLE:
        print_gpu(f"XGBoost GPU acceleration: ENABLED")
    
    earliest_overall = get_earliest_parquet_date()
    latest_overall = get_latest_parquet_date()
    print_info(f"Overall data range: {earliest_overall.strftime('%Y-%m-%d')} to {latest_overall.strftime('%Y-%m-%d')}")
    print_info(f"Total days: {(latest_overall - earliest_overall).days:,} days")
    
    if TQDM_AVAILABLE:
        pbar = tqdm(pairs, desc="Processing pairs", unit="pair")
        iterator = pbar
    else:
        iterator = pairs
    
    for i, pair in enumerate(iterator):
        if TQDM_AVAILABLE:
            pbar.set_description(f"Processing {pair}")
        
        try:
            result = train_pair(pair, target, lookback, force, use_gpu, verbose)
            if result:
                results['success'].append(result)
                if TQDM_AVAILABLE:
                    pbar.set_postfix({"status": "✓", "acc": f"{result['accuracy']:.3f}"})
            else:
                results['failed'].append(pair)
                if TQDM_AVAILABLE:
                    pbar.set_postfix({"status": "✗"})
        except Exception as e:
            print_error(f"Error training {pair}: {e}")
            traceback.print_exc()
            results['failed'].append(pair)
            if TQDM_AVAILABLE:
                pbar.set_postfix({"status": "✗", "error": str(e)[:20]})
        
        gc.collect()
    
    if TQDM_AVAILABLE:
        pbar.close()
    
    print_header("📊 TRAINING SUMMARY")
    print_success(f"Successful: {len(results['success'])}/{len(pairs)}")
    print_info(f"Data range: {earliest_overall.strftime('%Y-%m-%d')} to {latest_overall.strftime('%Y-%m-%d')}")
    
    total_rows = sum(r.get('total_rows', 0) for r in results['success'])
    avg_accuracy = sum(r.get('accuracy', 0) for r in results['success']) / len(results['success']) if results['success'] else 0
    total_time = sum(r.get('time', 0) for r in results['success'])
    
    print_info(f"Total rows processed: {total_rows:,}")
    print_info(f"Average accuracy: {avg_accuracy:.3f}")
    print_info(f"Total training time: {total_time/60:.1f} minutes")
    
    print_info("\nSuccessful models:")
    for r in results['success']:
        device = "GPU" if "gpu" in r['best_model'] else "CPU"
        print_info(f"  {r['pair']}: acc={r['accuracy']:.3f}, model={r['best_model']} ({device}), rows={r['total_rows']:,}, time={r['time']:.1f}s")
    
    if results['failed']:
        print_warning(f"\nFailed pairs: {len(results['failed'])}")
        for pair in results['failed']:
            print_warning(f"  {pair}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='🎯 Sid Method AI Model Trainer (GPU Accelerated)')
    parser.add_argument('--pair', type=str, help='Train specific pair')
    parser.add_argument('--all', action='store_true', help='Train all pairs')
    parser.add_argument('--list', action='store_true', help='List available pairs')
    parser.add_argument('--target', type=str, default='target_direction',
                       choices=['target_direction', 'target_rsi_50_5', 'target_rsi_50_10'],
                       help='Target variable to predict')
    parser.add_argument('--lookback', type=int, default=10, 
                       help='Lookback periods (default: 10)')
    parser.add_argument('--force', action='store_true', help='Force retrain')
    parser.add_argument('--cpu', action='store_true', help='Force CPU only (no GPU)')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                       help='Verbose output')
    parser.add_argument('--quiet', '-q', action='store_true', 
                       help='Quiet output (minimal)')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    use_gpu = not args.cpu
    
    if not SID_AVAILABLE:
        print_error("sid_method.py not found.")
        return 1
    
    print_header("🎯 SID METHOD AI MODEL TRAINER (GPU ACCELERATED)")
    print_info(f"Target: {args.target}")
    print_info(f"Lookback: {args.lookback}")
    print_info(f"GPU Mode: {'enabled' if use_gpu else 'disabled'}")
    
    if args.list:
        pairs = get_available_pairs()
        print_success(f"Available pairs: {len(pairs)}")
        for pair in pairs[:20]:
            earliest, latest = get_pair_date_range(pair)
            if earliest and latest:
                print(f"  {pair}: {earliest.strftime('%Y-%m-%d')} to {latest.strftime('%Y-%m-%d')}")
            else:
                print(f"  {pair}: date range unknown")
        return 0
    
    if not args.pair and not args.all:
        parser.print_help()
        return 1
    
    all_pairs = get_available_pairs()
    print_info(f"Found {len(all_pairs)} total pairs")
    
    if args.pair:
        if args.pair in all_pairs:
            train_pair(args.pair, args.target, args.lookback, args.force, use_gpu, verbose)
        else:
            print_error(f"Pair '{args.pair}' not found")
            return 1
    elif args.all:
        print_info(f"🚀 Training ALL {len(all_pairs)} pairs with GPU acceleration")
        train_all_pairs(all_pairs, args.target, args.lookback, args.force, use_gpu, verbose)
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_warning("\n\n⚠️ Training interrupted by user")
        sys.exit(1)