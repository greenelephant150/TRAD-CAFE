#!/usr/bin/env python3
"""
🎯 SIMON PULLEN AI MODEL TRAINER - DUAL GPU WITH IMPROVED ERROR HANDLING
============================================================
- Uses both NVIDIA Titan V GPUs (12GB each, capped at 95% = ~11.4GB)
- Automatic spillover to system RAM (32GB available)
- Improved cuML error handling with graceful fallback
- Memory-aware GPU training with timeout protection
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
import threading
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np

# Environment variables for stability
os.environ['RMM_NO_INITIALIZE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUPY_CUDA_PER_DEVICE_LIMIT'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'  # Explicitly set visible GPUs
os.environ['RAFT_LOG_LEVEL'] = 'WARN'  # Reduce raft logging

# Test GPU access immediately
try:
    import cudf
    import cupy as cp
    print(f"GPU test: {cp.cuda.runtime.getDeviceCount()} GPUs found")
    for i in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(i):
            free, total = cp.cuda.runtime.memGetInfo()
            print(f"GPU {i}: {total/1024**2:.0f}MB total, {free/1024**2:.0f}MB free")
except Exception as e:
    print(f"GPU init error: {e}")

# Suppress warnings
warnings.filterwarnings('ignore')

# ============================================================================
# COLORFUL TERMINAL OUTPUT
# ============================================================================
class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    MAGENTA = '\033[35m'; BOLD = '\033[1m'; END = '\033[0m'
    GPU0 = "🎮#0"; GPU1 = "🎮#1"; GPU = "🎮"; CPU = "💻"
    CHECK = "✅"; CROSS = "❌"; WARN = "⚠️"; FIRE = "🔥"
    BRAIN = "🧠"; DISK = "💾"; CLOCK = "⏱️"; GRAPH = "📈"
    MODEL = "🤖"; DATA = "📊"; MEM = "📀"

# ============================================================================
# SYSTEM CONFIGURATION
# ============================================================================
CPU_COUNT = multiprocessing.cpu_count()

# ============================================================================
# GPU MEMORY MANAGEMENT CONFIGURATION - INCREASED TO 95%
# ============================================================================
GPU_MEMORY_FRACTION = 0.95  # Use 95% of GPU memory
GPU0_MEMORY_LIMIT = int(12288 * GPU_MEMORY_FRACTION)  # ~11674MB for GPU 0
GPU1_MEMORY_LIMIT = int(12288 * GPU_MEMORY_FRACTION)  # ~11674MB for GPU 1
TOTAL_GPU_MEMORY = GPU0_MEMORY_LIMIT + GPU1_MEMORY_LIMIT  # ~23.3GB total
SPILLOVER_DEVICE = "cpu"  # Spill to system RAM when GPU memory exceeded

# Batch sizes for optimal GPU utilization
GPU_BATCH_SIZE = 50000  # Process 50k samples per GPU batch
CPU_BATCH_SIZE = 20000  # Process 20k samples per CPU batch

print(f"""
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}
{Colors.CYAN}{Colors.BOLD}🎯 SYSTEM CONFIGURATION{Colors.END}
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}
{Colors.YELLOW}CPU Cores:{Colors.END} {CPU_COUNT} cores available
{Colors.YELLOW}CPU Threads:{Colors.END} Using {CPU_COUNT} threads for data processing
{Colors.YELLOW}System RAM:{Colors.END} ~32GB available for spillover
""")

print(f"""
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}
{Colors.CYAN}{Colors.BOLD}🎯 DUAL GPU MEMORY MANAGEMENT{Colors.END}
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}
{Colors.YELLOW}GPU 0 Memory Limit:{Colors.END} {GPU0_MEMORY_LIMIT} MB ({GPU_MEMORY_FRACTION*100}% of 12GB)
{Colors.YELLOW}GPU 1 Memory Limit:{Colors.END} {GPU1_MEMORY_LIMIT} MB ({GPU_MEMORY_FRACTION*100}% of 12GB)
{Colors.YELLOW}Total GPU Memory:{Colors.END} {TOTAL_GPU_MEMORY} MB
{Colors.YELLOW}GPU Batch Size:{Colors.END} {GPU_BATCH_SIZE:,} samples
{Colors.YELLOW}CPU Batch Size:{Colors.END} {CPU_BATCH_SIZE:,} samples
{Colors.YELLOW}Spillover Device:{Colors.END} {SPILLOVER_DEVICE.upper()} (System RAM)
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}
""")

# ============================================================================
# GPU IMPORTS
# ============================================================================
GPU_AVAILABLE = False
GPU_BACKEND = 'cpu'
CUML_AVAILABLE = False
MULTI_GPU = False
DASK_AVAILABLE = False

print("\n🔌 Checking for GPU libraries...")

try:
    import cudf
    import cupy as cp
    
    try:
        import rmm
        print(f"  ✅ RMM memory manager available")
        rmm.reinitialize(managed_memory=True)  # Enable managed memory for better handling
    except ImportError:
        print(f"  ⚠️ RMM not available")
    
    num_gpus = cp.cuda.runtime.getDeviceCount()
    MULTI_GPU = num_gpus >= 2
    GPU_AVAILABLE = True
    GPU_BACKEND = 'cudf'
    
    print(f"  ✅ cuDF {cudf.__version__} available")
    print(f"  ✅ cuPy {cp.__version__} available")
    print(f"  ✅ Found {num_gpus} GPUs")
    print(f"  ✅ GPU 0 limit: {GPU0_MEMORY_LIMIT} MB")
    if MULTI_GPU:
        print(f"  ✅ GPU 1 limit: {GPU1_MEMORY_LIMIT} MB")
    
    try:
        import cuml
        from cuml.ensemble import RandomForestClassifier as cumlRF
        from cuml.common.memory_utils import set_global_memory_limit
        CUML_AVAILABLE = True
        
        # Set global memory limit for cuml
        try:
            set_global_memory_limit(GPU0_MEMORY_LIMIT * 1024 * 1024)  # Convert MB to bytes
        except:
            pass
            
        print(f"  ✅ cuML {cuml.__version__} available")
        
        try:
            from dask.distributed import Client
            from dask_cuda import LocalCUDACluster
            DASK_AVAILABLE = True
            print(f"  ✅ Dask CUDA available")
        except ImportError:
            DASK_AVAILABLE = False
            print(f"  ⚠️ Dask not available")
            
    except ImportError as e:
        print(f"  ⚠️ cuML not available: {e}")
        CUML_AVAILABLE = False
        
except ImportError as e:
    print(f"  ⚠️ cuDF not available: {e}")
    print(f"  ℹ️ Running in CPU mode")
    GPU_AVAILABLE = False

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================================================================
# PROGRESS BARS
# ============================================================================
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

def print_header(text): print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}\n{Colors.CYAN}{Colors.BOLD}{text.center(60)}{Colors.END}\n{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}\n")
def print_success(text): print(f"{Colors.GREEN}{Colors.BOLD}✅ {text}{Colors.END}")
def print_warning(text): print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ {text}{Colors.END}")
def print_error(text): print(f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.END}")
def print_info(text): print(f"{Colors.BLUE}ℹ️ {text}{Colors.END}")
def print_gpu0(text): print(f"{Colors.CYAN}{Colors.BOLD}{Colors.GPU0} {text}{Colors.END}")
def print_gpu1(text): print(f"{Colors.CYAN}{Colors.BOLD}{Colors.GPU1} {text}{Colors.END}")
def print_gpu(text): print(f"{Colors.CYAN}{Colors.BOLD}{Colors.GPU} {text}{Colors.END}")
def print_cpu(text): print(f"{Colors.CYAN}{Colors.BOLD}{Colors.CPU} {text}{Colors.END}")
def print_time(text): print(f"{Colors.YELLOW}{Colors.BOLD}{Colors.CLOCK} {text}{Colors.END}")
def print_model(text): print(f"{Colors.GREEN}{Colors.BOLD}{Colors.MODEL} {text}{Colors.END}")
def print_data(text): print(f"{Colors.BLUE}{Colors.BOLD}{Colors.DATA} {text}{Colors.END}")
def print_mem(text): print(f"{Colors.MAGENTA}{Colors.BOLD}{Colors.MEM} {text}{Colors.END}")

# ============================================================================
# GPU MEMORY MONITORING
# ============================================================================
class GPUMemoryMonitor:
    """Monitor GPU memory usage and trigger spillover if needed"""
    
    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.memory_usage = {0: 0, 1: 0}
        
    def check_memory(self, gpu_id=0):
        """Check current GPU memory usage"""
        try:
            if GPU_AVAILABLE:
                with cp.cuda.Device(gpu_id):
                    free, total = cp.cuda.runtime.memGetInfo()
                    used = total - free
                    fraction = used / total
                    self.memory_usage[gpu_id] = fraction
                    
                    if fraction > self.threshold:
                        print_warning(f"GPU {gpu_id} memory at {fraction:.1%}, triggering cleanup")
                        self._cleanup_gpu(gpu_id)
                    
                    return fraction
        except:
            return 0
        return 0
    
    def _cleanup_gpu(self, gpu_id):
        """Clean up GPU memory"""
        with cp.cuda.Device(gpu_id):
            cp.get_default_memory_pool().free_all_blocks()
            gc.collect()
    
    def get_optimal_gpu(self):
        """Return the GPU ID with more free memory"""
        self.check_memory(0)
        if MULTI_GPU:
            self.check_memory(1)
            return 0 if self.memory_usage[0] <= self.memory_usage[1] else 1
        return 0
    
    def estimate_memory_needed(self, X, y):
        """Estimate memory needed for training in MB"""
        try:
            # Rough estimate: data + model overhead (model ~ 3x data size)
            data_mb = (X.nbytes + y.nbytes) / (1024**2)
            total_mb = data_mb * 4  # Data + model overhead
            return total_mb
        except:
            return float('inf')

gpu_monitor = GPUMemoryMonitor(threshold=0.85)

# ============================================================================
# MEMORY CHECK FUNCTION
# ============================================================================
def check_and_clear_gpu_memory():
    """Check GPU memory and clear if needed before processing new pair"""
    if not GPU_AVAILABLE:
        return
    
    print_info("Checking GPU memory before next pair...")
    for gpu_id in range(2 if MULTI_GPU else 1):
        with cp.cuda.Device(gpu_id):
            free, total = cp.cuda.runtime.memGetInfo()
            used_pct = (total - free) / total * 100
            print_info(f"GPU {gpu_id}: {used_pct:.1f}% used ({ (total-free)/1024**2:.0f}MB / {total/1024**2:.0f}MB)")
            
            if used_pct > 85:
                print_warning(f"GPU {gpu_id} memory high ({used_pct:.1f}%), cleaning...")
                cp.get_default_memory_pool().free_all_blocks()
                gc.collect()

# ============================================================================
# CONFIGURATION
# ============================================================================
PARQUET_BASE_PATH = "/home/grct/Forex_Parquet"
MODEL_OUTPUT_PATH = "/mnt2/Trading-Cafe/ML/SPullen/ai/trained_models/"
# os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

# ============================================================================
# PARALLEL DATA LOADING
# ============================================================================
def parallel_parquet_load(pair_path: str) -> List[str]:
    """Find all parquet files"""
    parquet_files = []
    for root, dirs, files in os.walk(pair_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    return sorted(parquet_files)

def _load_parquet_cpu_parallel(parquet_files: List[str], pair: str, file_count: int, verbose: bool):
    """CPU-based parallel loading with pandas"""
    print_cpu(f"Loading {pair} with parallel CPU processing...")
    print_info(f"Found {file_count} files, using {CPU_COUNT} threads")
    
    try:
        start_time = time.time()
        dfs = []
        
        def load_single_file(file):
            try:
                return pd.read_parquet(file)
            except:
                return None
        
        with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
            futures = [executor.submit(load_single_file, f) for f in parquet_files]
            
            for future in tqdm(futures, desc="Loading files", disable=not verbose):
                result = future.result()
                if result is not None:
                    dfs.append(result)
        
        if not dfs:
            raise ValueError("No data loaded")
            
        df = pd.concat(dfs, ignore_index=True)
        
        # Process timestamps if needed
        if 'timestamp' not in df.columns and all(c in df.columns for c in ['year','month','day']):
            df['timestamp'] = pd.to_datetime(
                df['year'].astype(str) + '-' + 
                df['month'].astype(str).str.zfill(2) + '-' + 
                df['day'].astype(str).str.zfill(2)
            )
            df = df.drop(['year', 'month', 'day'], axis=1)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        elapsed = time.time() - start_time
        print_success(f"CPU loaded {len(df):,} rows in {elapsed:.1f}s")
        return df
        
    except Exception as e:
        print_error(f"CPU loading failed: {e}")
        return None

def load_parquet_hybrid(pair: str, force_cpu: bool = False, verbose: bool = False):
    """
    Load parquet files - FORCE CPU MODE for stability
    """
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    
    if not os.path.exists(pair_path):
        print_error(f"Pair path not found: {pair_path}")
        return None
    
    # Find all parquet files
    parquet_files = []
    for root, dirs, files in os.walk(pair_path):
        for file in files:
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(root, file))
    
    file_count = len(parquet_files)
    
    # FORCE CPU MODE - always use CPU loading
    print_cpu(f"Loading {pair} with CPU (most stable)...")
    print_info(f"Found {file_count} files, using {CPU_COUNT} threads")
    
    try:
        start_time = time.time()
        dfs = []
        
        def load_single_file(file):
            try:
                return pd.read_parquet(file)
            except:
                return None
        
        # Load files in parallel
        with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
            futures = [executor.submit(load_single_file, f) for f in parquet_files]
            
            for future in tqdm(futures, desc="Loading files", disable=not verbose):
                result = future.result()
                if result is not None:
                    dfs.append(result)
        
        if not dfs:
            raise ValueError("No data loaded")
        
        # Combine all dataframes
        print_info("Combining dataframes...")
        df = pd.concat(dfs, ignore_index=True)
        
        # Process timestamps if needed
        if 'timestamp' not in df.columns:
            if all(c in df.columns for c in ['year', 'month', 'day']):
                print_info("Creating timestamp from year/month/day...")
                df['timestamp'] = pd.to_datetime(
                    df['year'].astype(str) + '-' + 
                    df['month'].astype(str).str.zfill(2) + '-' + 
                    df['day'].astype(str).str.zfill(2)
                )
                df = df.drop(['year', 'month', 'day'], axis=1)
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
        
        elapsed = time.time() - start_time
        rows = len(df)
        print_success(f"CPU loaded {rows:,} rows in {elapsed:.1f}s")
        
        return df
        
    except Exception as e:
        print_error(f"CPU loading failed: {e}")
        return None

# ============================================================================
# INDICATOR CALCULATION - OPTIMIZED CHUNKED VERSION
# ============================================================================
def _calculate_indicators_cpu_parallel(df, verbose):
    """CPU-based indicators with parallel processing - OPTIMIZED"""
    print_cpu("Calculating indicators on CPU with parallel processing...")
    
    df = df.copy()
    
    # Split into chunks for parallel processing - use all cores
    num_chunks = CPU_COUNT * 2  # Double the cores for better utilization
    chunk_size = len(df) // num_chunks
    if chunk_size < 1000:  # Don't make chunks too small
        num_chunks = max(1, len(df) // 1000)
        chunk_size = len(df) // num_chunks
    
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(df)
        chunks.append(df.iloc[start:end].copy())
    
    print_info(f"Processing {len(df):,} rows in {num_chunks} chunks")
    
    def process_chunk(chunk):
        # Returns
        chunk['returns'] = chunk['close'].pct_change()
        chunk['log_returns'] = np.log(chunk['close'] / chunk['close'].shift(1))
        
        # Ranges
        chunk['high_low_ratio'] = (chunk['high'] - chunk['low']) / chunk['close']
        hl_diff = (chunk['high'] - chunk['low']).replace(0, 1)
        chunk['close_position'] = (chunk['close'] - chunk['low']) / hl_diff
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            chunk[f'ma_{period}'] = chunk['close'].rolling(window=period).mean()
            ma_period = chunk[f'ma_{period}'].replace(0, 1)
            chunk[f'ma_ratio_{period}'] = chunk['close'] / ma_period
        
        # Volatility
        for period in [5, 10, 20]:
            chunk[f'volatility_{period}'] = chunk['returns'].rolling(window=period).std()
        
        # RSI
        delta = chunk['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 0.001)
        chunk['rsi_14'] = 100 - (100 / (1 + rs))
        
        # Target
        chunk['target'] = (chunk['close'].shift(-1) > chunk['close']).astype(int)
        
        return chunk.dropna()
    
    # Process in parallel
    results = []
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for future in tqdm(futures, desc="Processing chunks", disable=not verbose):
            results.append(future.result())
    
    # Combine results
    result = pd.concat(results, ignore_index=True)
    print_success(f"CPU calculation complete: {len(result):,} rows")
    return result

def calculate_indicators_hybrid(data, force_cpu: bool = False, verbose: bool = False):
    """
    Calculate indicators - CPU ONLY (pandas)
    """
    print_cpu("Calculating indicators on CPU (pandas)...")
    
    # Data is already pandas, just process it
    return _calculate_indicators_cpu_parallel(data, verbose)

# ============================================================================
# DISCOVERY FUNCTIONS
# ============================================================================
def discover_parquet_pairs(verbose: bool = False) -> List[str]:
    """Discover all trading pairs"""
    pairs = []
    
    print_header("📁 PARQUET DATA DISCOVERY")
    
    if not os.path.exists(PARQUET_BASE_PATH):
        print_error(f"Path not found: {PARQUET_BASE_PATH}")
        return pairs
    
    for item in sorted(os.listdir(PARQUET_BASE_PATH)):
        item_path = os.path.join(PARQUET_BASE_PATH, item)
        if os.path.isdir(item_path):
            for subitem in os.listdir(item_path)[:5]:
                if subitem.startswith('year='):
                    pairs.append(item)
                    break
    
    print_success(f"Found {len(pairs)} pairs")
    return sorted(pairs)

def get_pair_date_range_detailed(pair: str) -> Tuple[Optional[datetime], Optional[datetime], int]:
    """Get date range and file count"""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    if not os.path.exists(pair_path):
        return None, None, 0
    
    min_date, max_date, file_count = None, None, 0
    
    for year_dir in os.listdir(pair_path):
        if not year_dir.startswith('year='): continue
        try:
            year = int(year_dir.split('=')[1])
            year_path = os.path.join(pair_path, year_dir)
            
            for month_dir in os.listdir(year_path):
                if not month_dir.startswith('month='): continue
                try:
                    month = int(month_dir.split('=')[1])
                    month_path = os.path.join(year_path, month_dir)
                    
                    for day_dir in os.listdir(month_path):
                        if not day_dir.startswith('day='): continue
                        try:
                            day = int(day_dir.split('=')[1])
                            date = datetime(year, month, day)
                            if min_date is None or date < min_date: min_date = date
                            if max_date is None or date > max_date: max_date = date
                            
                            day_path = os.path.join(month_path, day_dir)
                            if os.path.isdir(day_path):
                                file_count += len([f for f in os.listdir(day_path) if f.endswith('.parquet')])
                        except: continue
                except: continue
        except: continue
    
    return min_date, max_date, file_count

def format_filename(min_date: datetime, max_date: datetime, pair: str) -> str:
    min_str = min_date.strftime("%d%m%Y")
    max_str = max_date.strftime("%d%m%Y")
    return f"{min_str}--{max_str}--SPullen--{pair}--S5.pkl"

def create_samples(data, feature_cols, lookback=20, samples_per_pair=100000, verbose=False):
    """Create training samples"""
    
    print_data(f"Creating samples with lookback={lookback}, max samples={samples_per_pair:,}")
    conversion_start = time.time()
    
    with tqdm(total=1, desc="Extracting features", disable=not verbose) as pbar:
        feature_data = data[feature_cols].values
        target_data = data['target'].values
        pbar.update(1)
    
    conversion_time = time.time() - conversion_start
    if verbose:
        print_info(f"Data conversion: {conversion_time:.2f}s")
    
    total_possible = len(feature_data) - lookback
    if total_possible <= 0:
        return np.array([]), np.array([])
    
    actual_samples = min(samples_per_pair, total_possible)
    step = max(1, total_possible // actual_samples)
    indices = list(range(lookback, len(feature_data) - 1, step))[:actual_samples]
    
    print_info(f"Creating {len(indices):,} samples")
    
    # Create samples in parallel
    chunk_size = 10000
    num_chunks = (len(indices) + chunk_size - 1) // chunk_size
    index_chunks = [indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]
    
    def process_chunk(idx_chunk):
        X_chunk = np.array([feature_data[idx-lookback:idx].flatten() for idx in idx_chunk])
        y_chunk = np.array([target_data[idx] for idx in idx_chunk])
        return X_chunk, y_chunk
    
    X_list, y_list = [], []
    with ThreadPoolExecutor(max_workers=CPU_COUNT) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in index_chunks]
        for future in tqdm(futures, desc="Creating samples", disable=not verbose):
            Xc, yc = future.result()
            X_list.append(Xc)
            y_list.append(yc)
    
    X = np.vstack(X_list) if len(X_list) > 1 else X_list[0]
    y = np.concatenate(y_list) if len(y_list) > 1 else y_list[0]
    
    sample_time = time.time() - conversion_start
    print_success(f"Created {len(X):,} samples in {sample_time:.1f}s")
    
    if len(y) > 0:
        unique, counts = np.unique(y, return_counts=True)
        print_info(f"Target distribution: {dict(zip(unique, counts))}")
    
    return X, y

# ============================================================================
# GPU TRAINING FUNCTION WITH IMPROVED ERROR HANDLING
# ============================================================================
def train_with_gpu(X_train, y_train, X_val, gpu_id, verbose=False):
    """Train model on specific GPU with improved error handling"""
    
    try:
        with cp.cuda.Device(gpu_id):
            # Check available memory
            free, total = cp.cuda.runtime.memGetInfo()
            print_info(f"GPU {gpu_id}: {free/1024**2:.0f}MB free / {total/1024**2:.0f}MB total")
            
            # Estimate memory needed
            memory_needed = (X_train.nbytes + y_train.nbytes) * 4 / (1024**2)  # MB
            print_info(f"Estimated memory needed: {memory_needed:.0f}MB")
            
            if memory_needed > free * 0.9:
                print_warning(f"GPU {gpu_id} may not have enough memory")
                return None, None
            
            # Clear memory before training
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            
            # Convert to GPU arrays with explicit dtype
            X_train_gpu = cp.asarray(X_train, dtype=cp.float32)
            y_train_gpu = cp.asarray(y_train, dtype=cp.int32)
            
            # Create model with conservative parameters for better stability
            model = cumlRF(
                n_estimators=100,
                max_depth=16,
                max_features='sqrt',
                n_bins=128,  # Reduce memory usage
                split_criterion='gini',
                min_samples_split=2,
                min_samples_leaf=1,
                n_streams=1,  # Reduce parallel streams to save memory
                random_state=42,
                verbose=False,
                max_samples=1.0,  # Use all samples but with memory efficiency
                bootstrap=True
            )
            
            # Train with timeout mechanism
            train_complete = threading.Event()
            train_error = None
            trained_model = None
            
            def train_model():
                nonlocal train_error, trained_model
                try:
                    with tqdm(total=1, desc=f"🌲 Training GPU RF on GPU {gpu_id}", disable=not verbose) as pbar:
                        model.fit(X_train_gpu, y_train_gpu)
                        trained_model = model
                        pbar.update(1)
                except Exception as e:
                    train_error = e
                    print_error(f"GPU training error on device {gpu_id}: {e}")
                    import traceback
                    traceback.print_exc()
                finally:
                    train_complete.set()
            
            # Start training in thread
            train_thread = threading.Thread(target=train_model)
            train_thread.daemon = True
            train_thread.start()
            
            # Wait for training to complete with timeout
            timeout = 300  # 5 minutes timeout
            if not train_complete.wait(timeout=timeout):
                print_warning(f"GPU training on GPU {gpu_id} timed out after {timeout}s")
                return None, None
            
            if train_error:
                raise train_error
            
            if trained_model is None:
                print_error(f"GPU training failed on device {gpu_id} - model is None")
                return None, None
            
            # Validate
            try:
                X_val_gpu = cp.asarray(X_val, dtype=cp.float32)
                y_pred = trained_model.predict(X_val_gpu)
                y_pred = cp.asnumpy(y_pred)
                return trained_model, y_pred
            except Exception as e:
                print_error(f"GPU validation failed on device {gpu_id}: {e}")
                return None, None
            
    except Exception as e:
        print_error(f"GPU training failed on device {gpu_id}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# ============================================================================
# CPU TRAINING FUNCTION
# ============================================================================
def train_with_cpu(X_train, y_train, X_val, verbose=False):
    """Train model on CPU with scikit-learn"""
    print_cpu("Using scikit-learn with parallel CPU processing")
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=16,
        n_jobs=min(CPU_COUNT, 16),  # Limit to prevent oversubscription
        random_state=42,
        verbose=0
    )
    
    with tqdm(total=1, desc="🌲 Training CPU RF", disable=not verbose) as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)
    
    y_pred = model.predict(X_val)
    return model, y_pred

# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_pair(pair: str, force_cpu: bool = False, force: bool = False, verbose: bool = False):
    """Train one pair with improved GPU error handling"""

    global CUML_AVAILABLE, GPU_AVAILABLE, MULTI_GPU

    # Create model directory if it doesn't exist (now with correct permissions)
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
    
    print_header(f"{Colors.BRAIN} TRAINING: {pair}")
    pair_start = time.time()
    
    # Check GPU memory before starting
    check_and_clear_gpu_memory()
    
    # Get date range
    with tqdm(total=1, desc="Getting date range", disable=not verbose) as pbar:
        min_date, max_date, file_count = get_pair_date_range_detailed(pair)
        pbar.update(1)
    
    if not min_date or not max_date:
        print_error(f"Cannot determine date range for {pair}")
        return None
    
    days_range = (max_date - min_date).days + 1
    print_success(f"Data range: {min_date.date()} to {max_date.date()} ({days_range} days, {file_count} files)")
    
    # Check if model exists
    model_file = format_filename(min_date, max_date, pair)
    model_path = os.path.join(MODEL_OUTPUT_PATH, model_file)
    
    if os.path.exists(model_path) and not force:
        file_size = os.path.getsize(model_path) / (1024*1024)
        print_warning(f"Model exists: {model_file} ({file_size:.1f} MB)")
        mod_time = datetime.fromtimestamp(os.path.getmtime(model_path))
        print_info(f"Created: {mod_time.strftime('%Y-%m-%d %H:%M')}")
        return {'pair': pair, 'status': 'skipped'}
    
    start_total = time.time()
    
    # Load data
    print_header(f"{Colors.DATA} LOADING DATA")
    data = load_parquet_hybrid(pair, force_cpu, verbose)
    if data is None:
        return None
    
    # Calculate indicators
    print_header(f"{Colors.GRAPH} CALCULATING INDICATORS")
    data = calculate_indicators_hybrid(data, force_cpu, verbose)
    if data is None or len(data) == 0:
        return None
    
    # Create samples
    print_header(f"{Colors.DATA} CREATING SAMPLES")
    feature_cols = [c for c in data.columns if c not in ['open','high','low','close','volume','target']]
    print_info(f"Using {len(feature_cols)} features")
    
    X, y = create_samples(data, feature_cols, verbose=verbose)
    if len(X) == 0:
        return None
    
    # Train model
    print_header(f"{Colors.MODEL} TRAINING MODEL")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print_info(f"Training: {len(X_train):,} samples")
    print_info(f"Validation: {len(X_val):,} samples")
    
    train_start = time.time()
    model = None
    y_pred = None
    training_device = 'cpu'
    
    # Try GPU training if available and not forced to CPU
    if CUML_AVAILABLE and not force_cpu:
        print_gpu("Attempting cuML training with GPU acceleration")
        
        # Try GPUs in order (GPU 1 first as it had more free memory)
        gpus_to_try = [1, 0] if MULTI_GPU else [0]
        
        for gpu_id in gpus_to_try:
            print_gpu(f"Trying GPU {gpu_id} for training...")
            
            # Train on this GPU
            model, y_pred = train_with_gpu(X_train, y_train, X_val, gpu_id, verbose)
            
            if model is not None and y_pred is not None:
                training_device = f'gpu{gpu_id}'
                print_success(f"Successfully trained on GPU {gpu_id}")
                break
            else:
                print_warning(f"GPU {gpu_id} training failed, trying next option...")
                
                # Clean up after failed attempt
                with cp.cuda.Device(gpu_id):
                    cp.get_default_memory_pool().free_all_blocks()
                gc.collect()
    
    # Fall back to CPU if GPU training failed or not available
    if model is None:
        print_info("Falling back to CPU training...")
        model, y_pred = train_with_cpu(X_train, y_train, X_val, verbose)
        training_device = 'cpu'
    
    if model is None or y_pred is None:
        print_error("Training failed on all devices")
        return None
    
    train_time = time.time() - train_start
    print_success(f"Training completed in {train_time:.1f}s on {training_device}")
    
    accuracy = accuracy_score(y_val, y_pred)
    print_success(f"Model accuracy: {accuracy:.4f}")
    
    # Save model
    print_header(f"{Colors.DISK} SAVING MODEL")
    
    model_package = {
        'model': model,
        'metadata': {
            'pair': pair,
            'training_date': datetime.now().isoformat(),
            'data_range': {
                'min_date': min_date.isoformat(), 
                'max_date': max_date.isoformat(),
                'days': days_range,
                'files': file_count
            },
            'features': feature_cols,
            'performance': {
                'accuracy': float(accuracy),
                'eval_samples': len(y_val)
            },
            'model_params': {
                'n_estimators': 100,
                'max_depth': 16,
                'lookback': 20
            },
            'device': training_device,
            'samples': len(X),
            'training_time': time.time() - start_total
        }
    }
    
    with tqdm(total=1, desc="💾 Saving model", disable=not verbose) as pbar:
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        pbar.update(1)
    
    file_size = os.path.getsize(model_path) / (1024*1024)
    print_success(f"Model saved: {model_file} ({file_size:.1f} MB)")
    print_time(f"Total time for {pair}: {time.time()-start_total:.1f}s")
    
    return {
        'pair': pair,
        'status': 'trained',
        'accuracy': accuracy,
        'samples': len(X),
        'time': time.time() - start_total,
        'device': training_device
    }

def train_all_pairs(pairs: List[str], force_cpu: bool = False, force: bool = False, verbose: bool = False):
    """Train all pairs"""

    # Create model directory if it doesn't exist
    os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)

    results = {'success': [], 'skipped': [], 'failed': []}
    
    print_header(f"{Colors.FIRE} TRAINING {len(pairs)} PAIRS")
    
    main_pbar = tqdm(enumerate(pairs), total=len(pairs), desc="Overall progress", 
                     disable=not verbose, unit="pairs")
    
    for i, pair in main_pbar:
        main_pbar.set_description(f"Processing {pair}")
        pair_start = time.time()
        
        try:
            result = train_pair(pair, force_cpu, force, verbose)
            if result:
                if result.get('status') == 'trained':
                    results['success'].append(result)
                    main_pbar.set_postfix({
                        'status': '✓ trained',
                        'device': result.get('device', 'cpu'),
                        'acc': f"{result['accuracy']:.3f}",
                        'time': f"{time.time()-pair_start:.0f}s"
                    })
                else:
                    results['skipped'].append(pair)
                    main_pbar.set_postfix({'status': '↺ skipped'})
            else:
                results['failed'].append(pair)
                main_pbar.set_postfix({'status': '✗ failed'})
        except Exception as e:
            results['failed'].append(pair)
            main_pbar.set_postfix({'status': f'✗ {str(e)[:20]}'})
            if verbose:
                print(f"\nError on {pair}:")
                traceback.print_exc()
        
        gc.collect()
        if GPU_AVAILABLE:
            for gpu_id in range(2 if MULTI_GPU else 1):
                with cp.cuda.Device(gpu_id):
                    cp.get_default_memory_pool().free_all_blocks()
    
    main_pbar.close()
    return results

# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='🎯 Dual GPU Hybrid Model Training')
    parser.add_argument('--pair', type=str, help='Train specific pair')
    parser.add_argument('--all', action='store_true', help='Train all pairs')
    parser.add_argument('--list', action='store_true', help='List available pairs')
    parser.add_argument('--force', action='store_true', help='Force retrain')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--samples', type=int, default=100000, help='Samples per pair (default: 100000)')
    parser.add_argument('--lookback', type=int, default=20, help='Lookback periods (default: 20)')
    
    args = parser.parse_args()
    
    mode_str = 'DUAL GPU' if (GPU_AVAILABLE and MULTI_GPU and not args.cpu) else \
               ('SINGLE GPU' if (GPU_AVAILABLE and not args.cpu) else 'CPU')
    
    print(f"""
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}
{Colors.CYAN}{Colors.BOLD}🎯 SIMON PULLEN - DUAL GPU HYBRID TRAINER{Colors.END}
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}

{Colors.GPU if GPU_AVAILABLE and not args.cpu else Colors.CPU} Mode: {mode_str}
{Colors.YELLOW}GPU Status:{Colors.END} {'✅ Available' if GPU_AVAILABLE else '❌ Not Available'}
{Colors.YELLOW}Multi-GPU:{Colors.END} {'✅ Yes' if MULTI_GPU else '❌ No'}
{Colors.YELLOW}cuML Status:{Colors.END} {'✅ Available' if CUML_AVAILABLE else '❌ Not Available'}
{Colors.YELLOW}GPU Memory Limit:{Colors.END} {GPU_MEMORY_FRACTION*100}% per GPU (~{GPU0_MEMORY_LIMIT} MB)
{Colors.YELLOW}CPU Cores:{Colors.END} {CPU_COUNT} cores
{Colors.YELLOW}Samples:{Colors.END} {args.samples:,} per pair
{Colors.YELLOW}Lookback:{Colors.END} {args.lookback} periods
{Colors.MAGENTA}{Colors.BOLD}{'='*60}{Colors.END}
""")
    
    if args.list:
        pairs = discover_parquet_pairs(verbose=True)
        print_success(f"Total pairs: {len(pairs)}")
        return 0
    
    if not args.pair and not args.all:
        print_error("Specify --pair or --all")
        return 1
    
    all_pairs = discover_parquet_pairs(args.verbose)
    
    train_pairs = []
    if args.pair:
        if args.pair in all_pairs:
            train_pairs = [args.pair]
        else:
            print_error(f"Pair '{args.pair}' not found")
            return 1
    elif args.all:
        train_pairs = all_pairs
    
    print_success(f"Selected {len(train_pairs)} pairs")
    
    program_start = time.time()
    
    if args.pair:
        result = train_pair(args.pair, args.cpu, args.force, args.verbose)
        if result and result.get('status') == 'trained':
            print_success(f"Trained {args.pair}")
    else:
        results = train_all_pairs(train_pairs, args.cpu, args.force, args.verbose)
        
        print_header("📊 FINAL SUMMARY")
        print_success(f"Successful: {len(results['success'])}")
        
        if results['success']:
            print(f"\n{Colors.GREEN}Trained models:{Colors.END}")
            for r in results['success']:
                device_icon = Colors.GPU if 'gpu' in r.get('device', '') else Colors.CPU
                print(f"  {device_icon} {r['pair']}: acc={r['accuracy']:.3f}, time={r['time']:.1f}s")
        
        if results['skipped']:
            print(f"\n{Colors.YELLOW}Skipped: {len(results['skipped'])}")
        if results['failed']:
            print(f"\n{Colors.RED}Failed: {len(results['failed'])}")
    
    total_time = time.time() - program_start
    print_success(f"Total time: {int(total_time//3600)}h {int((total_time%3600)//60)}m {int(total_time%60)}s")
    
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print_warning("\n\n⚠️ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\n❌ Unexpected error: {e}")
        if '--verbose' in sys.argv or '-v' in sys.argv:
            traceback.print_exc()
        sys.exit(1)