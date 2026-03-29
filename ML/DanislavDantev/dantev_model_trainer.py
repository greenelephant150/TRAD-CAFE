#!/usr/bin/env python3
"""
Danislav Dantev - Institutional Model Trainer with FULL GPU Acceleration
Fixed to handle imbalanced classes and improve robustness
"""

import os
import sys
import pickle
import json
import argparse
import warnings
import gc
import glob
import time
import multiprocessing as mp
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# GPU DETECTION AND CONFIGURATION
# ============================================================================

GPU_AVAILABLE = False
GPU_COUNT = 0
GPU_INFO = []
CUML_AVAILABLE = False
CUPY_AVAILABLE = False

# GPU Memory Configuration (85% cap)
GPU_MEMORY_FRACTION = 0.85

# CPU Configuration (80% of cores)
CPU_COUNT = mp.cpu_count()
CPU_WORKERS = max(1, int(CPU_COUNT * 0.8))

print(f"\n{'='*70}")
print(f"🚀 System Configuration")
print(f"{'='*70}")
print(f"CPU Cores: {CPU_COUNT}")
print(f"CPU Workers: {CPU_WORKERS} (80% of cores)")
print(f"GPU Memory Cap: {GPU_MEMORY_FRACTION*100}%")

# Try to import GPU libraries
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    GPU_AVAILABLE = True
    GPU_COUNT = cp.cuda.runtime.getDeviceCount()
    print(f"✅ cuPy {cp.__version__} available")
    
    for i in range(GPU_COUNT):
        with cp.cuda.Device(i):
            free, total = cp.cuda.runtime.memGetInfo()
            total_gb = total / (1024**3)
            free_gb = free / (1024**3)
            GPU_INFO.append({
                'id': i,
                'total_gb': total_gb,
                'free_gb': free_gb,
                'max_memory_gb': total_gb * GPU_MEMORY_FRACTION
            })
            print(f"GPU {i}: {total_gb:.1f}GB total, {free_gb:.1f}GB free (capped at {GPU_MEMORY_FRACTION*100:.0f}% = {GPU_INFO[-1]['max_memory_gb']:.1f}GB)")
except ImportError as e:
    print(f"⚠️ cuPy not available: {e}")
except Exception as e:
    print(f"⚠️ GPU initialization error: {e}")

try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cuRF
    CUML_AVAILABLE = True
    print(f"✅ cuML {cuml.__version__} available")
except ImportError as e:
    print(f"⚠️ cuML not available: {e}")
except Exception as e:
    print(f"⚠️ cuML initialization error: {e}")

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

EMOJIS = {
    'rocket': '🚀', 'chart': '📊', 'gpu': '🎮', 'cpu': '💻',
    'check': '✅', 'cross': '❌', 'warning': '⚠️', 'brain': '🧠',
    'data': '💾', 'time': '⏱️', 'target': '🎯', 'training': '🏋️',
    'model': '🤖', 'success': '✅', 'failure': '❌'
}

def print_header(text):
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}{'='*70}{Colors.END}")

def print_success(text): print(f"{Colors.GREEN}{EMOJIS['check']} {text}{Colors.END}")
def print_warning(text): print(f"{Colors.YELLOW}{EMOJIS['warning']} {text}{Colors.END}")
def print_error(text): print(f"{Colors.RED}{EMOJIS['cross']} {text}{Colors.END}")
def print_info(text): print(f"{Colors.BLUE}ℹ️ {text}{Colors.END}")
def print_gpu(text): print(f"{Colors.CYAN}{EMOJIS['gpu']} {text}{Colors.END}")

# ============================================================================
# CONFIGURATION
# ============================================================================
PARQUET_BASE_PATH = "/home/grct/Forex_Parquet"
MODEL_OUTPUT_PATH = "/mnt2/Trading-Cafe/ML/DDantev/ai/trained_models/"
METADATA_FILE = os.path.join(MODEL_OUTPUT_PATH, "dantev_training_metadata.json")

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

# ============================================================================
# DATA LOADING WITH PARALLEL PROCESSING
# ============================================================================

def load_parquet_files_parallel(file_paths: List[str]) -> List[pd.DataFrame]:
    """Load multiple parquet files in parallel"""
    from concurrent.futures import ThreadPoolExecutor
    
    def load_file(file_path):
        try:
            return pd.read_parquet(file_path)
        except:
            return None
    
    results = []
    with ThreadPoolExecutor(max_workers=CPU_WORKERS) as executor:
        futures = [executor.submit(load_file, fp) for fp in file_paths]
        for future in futures:
            result = future.result()
            if result is not None and not result.empty:
                results.append(result)
    
    return results


def load_parquet_for_pair(pair: str, sample_days: int = None) -> pd.DataFrame:
    """Load parquet data with parallel processing"""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    
    if not os.path.exists(pair_path):
        return pd.DataFrame()
    
    # Collect all parquet files
    file_list = []
    for root, dirs, files in os.walk(pair_path):
        for file in files:
            if file.endswith('.parquet'):
                file_list.append(os.path.join(root, file))
    
    if not file_list:
        return pd.DataFrame()
    
    print_info(f"Found {len(file_list)} parquet files")
    
    # Limit files for testing
    if sample_days:
        file_list.sort()
        max_files = sample_days * 720
        if len(file_list) > max_files:
            file_list = file_list[-max_files:]
            print_info(f"Limited to last {len(file_list)} files (~{sample_days} days)")
    
    # Load files in parallel
    dfs = load_parquet_files_parallel(file_list)
    
    if not dfs:
        return pd.DataFrame()
    
    # Combine
    df = pd.concat(dfs)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep='first')]
    
    # Remove timezone if present
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    print_success(f"Loaded {len(df):,} rows")
    
    return df


def get_all_pairs() -> List[str]:
    """Get all pairs with parquet data"""
    pairs = []
    if os.path.exists(PARQUET_BASE_PATH):
        for item in os.listdir(PARQUET_BASE_PATH):
            pair_path = os.path.join(PARQUET_BASE_PATH, item)
            if os.path.isdir(pair_path):
                for root, dirs, files in os.walk(pair_path):
                    if any(f.endswith('.parquet') for f in files):
                        pairs.append(item)
                        break
    return sorted(pairs)


def get_pair_date_range(pair: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get approximate date range from file paths"""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    if not os.path.exists(pair_path):
        return None, None
    
    dates = []
    for root, dirs, files in os.walk(pair_path):
        for file in files:
            if file.endswith('.parquet'):
                parts = root.split(os.sep)
                year = None
                month = None
                day = None
                
                for part in parts:
                    if '__HIVE_DEFAULT_PARTITION__' in part:
                        continue
                    if part.startswith('year='):
                        try:
                            year = int(part.split('=')[1])
                        except ValueError:
                            continue
                    elif part.startswith('month='):
                        try:
                            month = int(part.split('=')[1])
                        except ValueError:
                            continue
                    elif part.startswith('day='):
                        try:
                            day = int(part.split('=')[1])
                        except ValueError:
                            continue
                
                if year is not None and month is not None and day is not None:
                    try:
                        dates.append(datetime(year, month, day))
                    except ValueError:
                        continue
    
    if not dates:
        return None, None
    
    return min(dates), max(dates)


# ============================================================================
# FEATURE ENGINEERING (CPU for data prep)
# ============================================================================

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate technical indicators"""
    df = df.copy()
    
    # Returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Range features
    df['range'] = df['high'] - df['low']
    df['body'] = abs(df['close'] - df['open'])
    df['body_ratio'] = df['body'] / df['range'].replace(0, 1)
    df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
    df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
    
    # Moving averages
    for period in [10, 20, 50]:
        df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}'].replace(0, 1)
    
    # Volatility
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, 0.001)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # ATR
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift(1))
    tr3 = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    
    # Volume
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
    
    # Premium/Discount
    lookback = 50
    df['range_high'] = df['high'].rolling(window=lookback).max()
    df['range_low'] = df['low'].rolling(window=lookback).min()
    df['premium_discount'] = (df['close'] - df['range_low']) / (df['range_high'] - df['range_low']).replace(0, 1)
    df['premium_discount'] = df['premium_discount'].clip(0, 1)
    
    return df.dropna()


def extract_features(df: pd.DataFrame, idx: int) -> Dict:
    """Extract features at a specific index"""
    features = {
        'rsi': float(df['rsi'].iloc[idx]) if not pd.isna(df['rsi'].iloc[idx]) else 50,
        'atr': float(df['atr'].iloc[idx]) if not pd.isna(df['atr'].iloc[idx]) else 0.001,
        'volatility': float(df['volatility'].iloc[idx]) if not pd.isna(df['volatility'].iloc[idx]) else 0.001,
        'volume_ratio': float(df['volume_ratio'].iloc[idx]) if not pd.isna(df['volume_ratio'].iloc[idx]) else 1.0,
        'body_ratio': float(df['body_ratio'].iloc[idx]) if not pd.isna(df['body_ratio'].iloc[idx]) else 0.5,
        'premium_discount': float(df['premium_discount'].iloc[idx]) if not pd.isna(df['premium_discount'].iloc[idx]) else 0.5,
    }
    
    # Moving average ratios
    for period in [10, 20, 50]:
        col = f'ma_ratio_{period}'
        if col in df and not pd.isna(df[col].iloc[idx]):
            features[f'ma_ratio_{period}'] = float(df[col].iloc[idx])
        else:
            features[f'ma_ratio_{period}'] = 1.0
    
    # Returns (momentum)
    features['return_1'] = df['returns'].iloc[idx] if idx > 0 and not pd.isna(df['returns'].iloc[idx]) else 0
    features['return_5'] = df['close'].iloc[idx] / df['close'].iloc[max(0, idx-5)] - 1
    features['return_10'] = df['close'].iloc[idx] / df['close'].iloc[max(0, idx-10)] - 1
    
    # Time features
    features['hour'] = df.index[idx].hour
    features['day_of_week'] = df.index[idx].weekday()
    
    return features


def create_training_samples(df: pd.DataFrame, max_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Create training samples with class balancing"""
    print_info("Calculating indicators...")
    df = calculate_indicators(df)
    
    if len(df) < 200:
        return np.array([]), np.array([]), []
    
    # Determine step size to achieve max_samples
    step = max(1, len(df) // max_samples)
    indices = list(range(200, len(df) - 720, step))
    
    print_info(f"Processing {len(indices)} samples (step={step})")
    
    X_list = []
    y_list = []
    feature_names = None
    
    # Track class counts for balancing
    class_counts = {0: 0, 1: 0, 2: 0}
    max_per_class = max_samples // 3  # Aim for balanced classes
    
    for idx in indices:
        try:
            features = extract_features(df, idx)
            
            # Target: price movement in next 720 bars (1 hour at 5s)
            future_return = (df['close'].iloc[idx + 720] - df['close'].iloc[idx]) / df['close'].iloc[idx]
            
            if future_return > 0.002:
                target = 1  # Bullish
            elif future_return < -0.002:
                target = 2  # Bearish
            else:
                target = 0  # Neutral
            
            # Skip if we already have enough of this class
            if class_counts[target] >= max_per_class and target != 0:
                continue
            
            X_list.append(list(features.values()))
            y_list.append(target)
            class_counts[target] += 1
            
            if feature_names is None:
                feature_names = list(features.keys())
                
        except Exception as e:
            continue
    
    if not X_list:
        return np.array([]), np.array([]), []
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    print_success(f"Created {len(X)} samples with {X.shape[1]} features")
    print_info(f"Target distribution: Bullish={np.sum(y==1)}, Bearish={np.sum(y==2)}, Neutral={np.sum(y==0)}")
    
    # Check if we have enough samples for training
    if np.sum(y == 1) < 2 or np.sum(y == 2) < 2:
        print_warning("Insufficient samples for bullish/bearish classes. Using simplified binary classification.")
        # Convert to binary: 1 for any movement (bullish or bearish), 0 for neutral
        y_binary = np.where(y != 0, 1, 0)
        return X, y_binary, feature_names
    
    return X, y, feature_names


# ============================================================================
# GPU-ACCELERATED TRAINING WITH IMBALANCED CLASS HANDLING
# ============================================================================

def train_gpu_model(X: np.ndarray, y: np.ndarray, feature_names: List[str], 
                    pair: str, verbose: bool = False) -> Optional[Dict]:
    """Train model using GPU with cuML - handles imbalanced classes"""
    if not CUML_AVAILABLE:
        return None
    
    try:
        import cupy as cp
        from cuml.ensemble import RandomForestClassifier as cuRF
        from cuml.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        print_gpu(f"Training on GPU...")
        
        # Check class distribution
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        class_counts = {c: np.sum(y == c) for c in unique_classes}
        
        print_info(f"Class distribution: {class_counts}")
        
        # Skip GPU training if classes are too imbalanced
        min_class_size = min(class_counts.values())
        if min_class_size < 50:  # Need at least 50 samples per class for meaningful GPU training
            print_warning(f"Min class size ({min_class_size}) too small for GPU training")
            print_info("Falling back to CPU training for better class handling")
            return None
        
        # Skip if we have binary classification with extreme imbalance
        if n_classes == 2:
            ratio = max(class_counts.values()) / min_class_size
            if ratio > 100:  # 100:1 imbalance
                print_warning(f"Class imbalance too extreme ({ratio:.1f}:1) for GPU training")
                print_info("Falling back to CPU training")
                return None
        
        # Get best GPU
        gpu_id = 0
        print_gpu(f"Using GPU {gpu_id} - {GPU_INFO[gpu_id]['total_gb']:.1f}GB total")
        
        # Check data size
        data_size_mb = len(X) * X.shape[1] * 4 / (1024**2)
        print_info(f"Data size: {len(X):,} samples × {X.shape[1]} features = {data_size_mb:.1f}MB")
        
        # Handle binary vs multi-class
        if n_classes == 2:
            print_info("Binary classification mode")
            output_type = 'binary'
        else:
            print_info("Multi-class classification mode")
            output_type = 'multi'
        
        # Clear GPU memory
        if CUPY_AVAILABLE:
            with cp.cuda.Device(gpu_id):
                cp.get_default_memory_pool().free_all_blocks()
        
        with cp.cuda.Device(gpu_id):
            # Move data to GPU
            print_gpu("Moving data to GPU...")
            start_transfer = time.time()
            X_gpu = cp.asarray(X, dtype=cp.float32)
            y_gpu = cp.asarray(y, dtype=cp.int32)
            print_gpu(f"  Transfer completed in {time.time()-start_transfer:.2f}s")
            
            # Use stratified split if possible, otherwise simple split
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_gpu, y_gpu, test_size=0.2, random_state=42, stratify=y_gpu
                )
            except Exception:
                print_warning("Stratified split failed, using simple split...")
                X_train, X_test, y_train, y_test = train_test_split(
                    X_gpu, y_gpu, test_size=0.2, random_state=42
                )
            
            # Calculate optimal max_samples as fraction (0.0 to 1.0)
            # Use 0.8 for datasets > 100k samples, otherwise use 1.0
            n_samples = len(X_train)
            if n_samples > 100000:
                max_samples_frac = 0.8
            elif n_samples > 50000:
                max_samples_frac = 0.9
            else:
                max_samples_frac = 1.0
            
            print_info(f"Using max_samples={max_samples_frac:.1f} ({int(max_samples_frac * n_samples)} samples)")
            
            # Create model
            model = cuRF(
                n_estimators=100,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                n_bins=128,
                random_state=42,
                max_samples=max_samples_frac  # Use fraction, not absolute number
            )
            
            # Train model
            print_gpu("Training model on GPU...")
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Predict
            print_gpu("Evaluating model...")
            y_pred = model.predict(X_test)
            y_pred_cpu = cp.asnumpy(y_pred)
            y_test_cpu = cp.asnumpy(y_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
            
            print_success(f"GPU training completed in {training_time:.1f}s")
            print_info(f"Accuracy: {accuracy:.4f}")
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance = cp.asnumpy(model.feature_importances_)
                feature_importance = dict(zip(feature_names, importance))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
                print_info("Top 10 features:")
                for feat, imp in top_features:
                    print(f"  {feat}: {imp:.4f}")
            else:
                feature_importance = {}
            
            # Clear GPU memory
            del X_gpu, y_gpu, X_train, X_test, y_train, y_test
            if CUPY_AVAILABLE:
                with cp.cuda.Device(gpu_id):
                    cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
            
            return {
                'model': model,
                'model_type': f'random_forest_gpu_{output_type}',
                'accuracy': float(accuracy),
                'training_time': training_time,
                'feature_importance': feature_importance,
                'gpu_id': gpu_id,
                'samples': len(X),
                'classes': n_classes
            }
            
    except Exception as e:
        print_error(f"GPU training failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return None

def train_cpu_model(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                    pair: str, verbose: bool = False) -> Optional[Dict]:
    """Train model using CPU with scikit-learn - handles imbalanced classes"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        print_info(f"Training on CPU with {CPU_WORKERS} workers...")
        
        # Check classes
        unique_classes = np.unique(y)
        n_classes = len(unique_classes)
        
        if n_classes == 1:
            print_warning(f"Only one class found ({unique_classes[0]}). Using dummy model.")
            return None
        
        # Use simple split without stratification if classes are imbalanced
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            print_warning("Stratified split failed, using simple random split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=CPU_WORKERS,
            random_state=42,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print_success(f"CPU training completed in {training_time:.1f}s")
        print_info(f"Accuracy: {accuracy:.4f}")
        
        feature_importance = dict(zip(feature_names, model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print_info("Top 10 features:")
        for feat, imp in top_features:
            print(f"  {feat}: {imp:.4f}")
        
        return {
            'model': model,
            'model_type': f'random_forest_cpu_{n_classes}class',
            'accuracy': float(accuracy),
            'training_time': training_time,
            'feature_importance': feature_importance,
            'samples': len(X),
            'classes': n_classes
        }
        
    except Exception as e:
        print_error(f"CPU training failed: {e}")
        return None


# ============================================================================
# METADATA MANAGEMENT
# ============================================================================

def load_training_metadata() -> Dict:
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_training_metadata(metadata: Dict):
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)


def get_model_info(pair: str) -> Optional[Dict]:
    metadata = load_training_metadata()
    return metadata.get(pair)


def update_model_info(pair: str, info: Dict):
    metadata = load_training_metadata()
    metadata[pair] = info
    save_training_metadata(metadata)


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_model(pair: str, force: bool = False, sample_days: int = None,
                use_gpu: bool = True, verbose: bool = False) -> Optional[Dict]:
    """Train model with GPU/CPU auto-selection"""
    print_header(f"🏦 Training Danislav Dantev Model for {pair}")
    
    # Get existing model info
    existing_info = get_model_info(pair)
    
    if existing_info and not force:
        print_info(f"Model already exists (trained up to {existing_info.get('data_end', 'unknown')[:10]})")
        print_info("Use --force to retrain")
        return {'pair': pair, 'status': 'skipped'}
    
    # Get date range
    min_date, max_date = get_pair_date_range(pair)
    if not min_date or not max_date:
        print_error(f"Cannot determine date range for {pair}")
        return None
    
    print_info(f"Data range: {min_date.date()} to {max_date.date()}")
    
    # Load data
    print_info("Loading data...")
    df = load_parquet_for_pair(pair, sample_days=sample_days)
    
    if df.empty or len(df) < 1000:
        print_error(f"Insufficient data: {len(df)} rows")
        return None
    
    print_success(f"Loaded {len(df):,} rows")
    
    # Create training samples
    X, y, feature_names = create_training_samples(df)
    
    if len(X) == 0:
        print_error("No training samples created")
        return None
    
    # Train model
    result = None
    if use_gpu and CUML_AVAILABLE:
        print_gpu("Attempting GPU training...")
        result = train_gpu_model(X, y, feature_names, pair, verbose)
        if result:
            print_success("Successfully trained on GPU!")
    
    if result is None:
        print_info("Falling back to CPU training...")
        result = train_cpu_model(X, y, feature_names, pair, verbose)
    
    if result is None:
        print_error("Training failed")
        return None
    
    # Save model
    model_filename = f"Dantev_{pair}_{max_date.strftime('%Y%m%d')}.pkl"
    model_path = os.path.join(MODEL_OUTPUT_PATH, model_filename)
    
    model_package = {
        'model': result['model'],
        'model_type': result['model_type'],
        'pair': pair,
        'training_date': datetime.now().isoformat(),
        'training_time': result['training_time'],
        'data_range': {
            'start': min_date.isoformat(),
            'end': max_date.isoformat()
        },
        'performance': {
            'accuracy': result['accuracy'],
            'samples': len(X),
            'features': len(feature_names),
            'classes': result.get('classes', 2)
        },
        'feature_names': feature_names,
        'feature_importance': result['feature_importance'],
        'version': '2.1'
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    # Update metadata
    metadata = {
        'pair': pair,
        'model_type': result['model_type'],
        'training_date': datetime.now().isoformat(),
        'training_time': result['training_time'],
        'accuracy': result['accuracy'],
        'samples': len(X),
        'classes': result.get('classes', 2),
        'data_end': max_date.isoformat(),
        'filename': model_filename
    }
    update_model_info(pair, metadata)
    
    file_size = os.path.getsize(model_path) / (1024 * 1024)
    print_success(f"Model saved: {model_filename} ({file_size:.1f} MB)")
    
    return {
        'pair': pair,
        'status': 'trained',
        'accuracy': result['accuracy'],
        'samples': len(X),
        'training_time': result['training_time'],
        'model_type': result['model_type'],
        'file_size_mb': file_size
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Danislav Dantev GPU-Accelerated Model Trainer')
    parser.add_argument('--pair', type=str, help='Train specific pair')
    parser.add_argument('--all', action='store_true', help='Train all pairs')
    parser.add_argument('--force', action='store_true', help='Force retrain')
    parser.add_argument('--list', action='store_true', help='List available pairs')
    parser.add_argument('--status', action='store_true', help='Show model status')
    parser.add_argument('--sample-days', type=int, help='Use last N days only')
    parser.add_argument('--max-pairs', type=int, help='Max pairs to train')
    parser.add_argument('--cpu-only', action='store_true', help='Force CPU only')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    print_header("🏦 DANISLAV DANTEV INSTITUTIONAL MODEL TRAINER")
    print_info(f"CPU Workers: {CPU_WORKERS}")
    if GPU_AVAILABLE and CUML_AVAILABLE:
        print_success(f"GPU Acceleration: ENABLED ({GPU_COUNT} GPUs)")
    else:
        print_warning("GPU Acceleration: DISABLED (using CPU)")
    
    if args.list:
        pairs = get_all_pairs()
        print_header(f"📋 AVAILABLE PAIRS ({len(pairs)})")
        for pair in pairs:
            min_date, max_date = get_pair_date_range(pair)
            if min_date and max_date:
                days = (max_date - min_date).days + 1
                print(f"  {pair}: {min_date.date()} to {max_date.date()} ({days} days)")
            else:
                print(f"  {pair}: No data")
        return 0
    
    if args.status:
        metadata = load_training_metadata()
        print_header(f"📊 MODEL STATUS ({len(metadata)} models)")
        for pair, info in metadata.items():
            acc = info.get('accuracy', 0)
            samples = info.get('samples', 0)
            classes = info.get('classes', 2)
            data_end = info.get('data_end', 'unknown')[:10] if info.get('data_end') else 'unknown'
            gpu_icon = "🎮" if 'gpu' in info.get('model_type', '') else "💻"
            print(f"  {gpu_icon} {pair}: acc={acc:.3f}, samples={samples:,}, classes={classes}, data_to={data_end}")
        return 0
    
    if args.pair:
        use_gpu = not args.cpu_only
        result = train_model(args.pair, force=args.force, sample_days=args.sample_days,
                            use_gpu=use_gpu, verbose=args.verbose)
        if result and result.get('status') == 'trained':
            print_header("✅ TRAINING COMPLETE")
            print_success(f"Model for {args.pair} trained!")
            print_info(f"  Accuracy: {result['accuracy']:.4f}")
            print_info(f"  Samples: {result['samples']:,}")
            print_info(f"  Time: {result['training_time']:.1f}s")
            print_info(f"  Model: {result['model_type']}")
        elif result and result.get('status') == 'skipped':
            print_info(f"Model already exists for {args.pair}")
        else:
            print_error(f"Failed to train {args.pair}")
        return 0
    
    if args.all:
        pairs = get_all_pairs()
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
            print_info(f"Limited to first {args.max_pairs} pairs")
        
        print_header(f"🚀 TRAINING {len(pairs)} PAIRS")
        
        use_gpu = not args.cpu_only
        results = {'successful': [], 'failed': [], 'skipped': []}
        
        for i, pair in enumerate(pairs):
            print_info(f"\n[{i+1}/{len(pairs)}] {pair}")
            result = train_model(pair, force=args.force, sample_days=args.sample_days,
                                use_gpu=use_gpu, verbose=args.verbose)
            if result:
                if result.get('status') == 'trained':
                    results['successful'].append(result)
                elif result.get('status') == 'skipped':
                    results['skipped'].append(pair)
                else:
                    results['failed'].append(pair)
            else:
                results['failed'].append(pair)
            gc.collect()
        
        print_header("📊 TRAINING SUMMARY")
        print_success(f"Successful: {len(results['successful'])}")
        print_info(f"Skipped: {len(results['skipped'])}")
        print_error(f"Failed: {len(results['failed'])}")
        
        if results['successful']:
            print("\n✅ Trained models:")
            for r in results['successful']:
                gpu_icon = "🎮" if 'gpu' in r.get('model_type', '') else "💻"
                print(f"  {gpu_icon} {r['pair']}: {r['accuracy']:.3f} ({r['samples']:,} samples, {r['training_time']:.1f}s)")
        
        if results['failed']:
            print("\n❌ Failed pairs:")
            for pair in results['failed']:
                print(f"  {pair}")
        
        return 0
    
    parser.print_help()
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_warning("\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ Unexpected error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            import traceback
            traceback.print_exc()
        sys.exit(1)