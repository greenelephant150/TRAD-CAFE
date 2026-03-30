"""
GPU-Accelerated Data Loader for Forex CSV files
Uses cuDF for parallel processing on NVIDIA GPUs
"""

import os
import glob
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from pathlib import Path
import time
import concurrent.futures
from multiprocessing import cpu_count

# Try to import GPU libraries
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    GPU_BACKEND = 'cudf'
    print("✅ cuDF GPU acceleration available")
except ImportError:
    try:
        import torch
        GPU_AVAILABLE = True
        GPU_BACKEND = 'torch'
        print("✅ PyTorch GPU acceleration available")
    except ImportError:
        GPU_AVAILABLE = False
        GPU_BACKEND = 'cpu'
        print("⚠️ GPU acceleration not available, using CPU")

logger = logging.getLogger(__name__)


class GPUDataLoader:
    """
    GPU-accelerated data loader for Forex CSV files
    Uses parallel processing for faster data loading
    """
    
    def __init__(self, base_path: str = "/home/grct/Forex", use_gpu: bool = True):
        """
        Args:
            base_path: Base directory containing Forex data
            use_gpu: Whether to use GPU acceleration if available
        """
        self.base_path = Path(base_path)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.backend = GPU_BACKEND if self.use_gpu else 'cpu'
        
        # CPU thread pool for parallel file loading
        self.cpu_cores = cpu_count()
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=self.cpu_cores)
        
        # GPU stream for asynchronous operations
        if self.use_gpu and self.backend == 'cudf':
            self.gpu_stream = cp.cuda.Stream()
            self.gpu_memory_pool = cp.get_default_memory_pool()
            
            # Get GPU info
            self.gpu_count = len(cudf.cuda.gpus)
            self.gpu_memory = []
            for i in range(self.gpu_count):
                with cudf.cuda.Device(i):
                    mem_info = cp.cuda.runtime.memGetInfo()
                    free_mem = mem_info[0] / 1e9
                    total_mem = mem_info[1] / 1e9
                    self.gpu_memory.append({
                        'gpu_id': i,
                        'total': total_mem,
                        'free': free_mem
                    })
        
        print(f"🚀 GPUDataLoader initialized with {self.backend.upper()} backend")
        print(f"   CPU cores available: {self.cpu_cores}")
        if self.use_gpu:
            if self.backend == 'cudf':
                print(f"   GPU count: {self.gpu_count}")
                for i, mem in enumerate(self.gpu_memory):
                    print(f"   GPU {i}: {mem['total']:.1f}GB total, {mem['free']:.1f}GB free")
    
    def load_files_parallel(self, file_paths: List[Path], 
                              progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Load multiple CSV files in parallel using CPU threads
        
        Args:
            file_paths: List of file paths to load
            progress_callback: Function to call with progress updates
        
        Returns:
            Combined DataFrame
        """
        total_files = len(file_paths)
        results = [None] * total_files
        
        def load_single_file(index: int, file_path: Path):
            try:
                df = self._load_single_csv(file_path)
                if progress_callback:
                    progress_callback((index + 1) / total_files, f"Loading file {index+1}/{total_files}")
                return index, df
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
                return index, None
        
        # Submit all tasks
        futures = []
        for i, file_path in enumerate(file_paths):
            future = self.thread_pool.submit(load_single_file, i, file_path)
            futures.append(future)
        
        # Collect results
        for future in concurrent.futures.as_completed(futures):
            index, df = future.result()
            results[index] = df
        
        # Filter out None results and combine
        valid_dfs = [df for df in results if df is not None]
        
        if not valid_dfs:
            return pd.DataFrame()
        
        # Combine using pandas (CPU)
        combined = pd.concat(valid_dfs)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined.sort_index(inplace=True)
        
        return combined
    
    def _load_single_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single CSV file (CPU operation)"""
        try:
            df = pd.read_csv(file_path)
            
            required = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                return None
            
            df['datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True)
            df.set_index('datetime', inplace=True)
            df.drop('time', axis=1, inplace=True)
            
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def resample_to_timeframe_gpu(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data using GPU acceleration if available
        
        Args:
            df: DataFrame with S5 data
            timeframe: Target timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
        
        Returns:
            Resampled DataFrame
        """
        if df.empty:
            return df
        
        # Define resample rules
        resample_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D'
        }
        
        rule = resample_map.get(timeframe, '1H')
        
        # Use GPU if available
        if self.use_gpu and self.backend == 'cudf' and len(df) > 10000:
            return self._resample_with_cudf(df, rule)
        else:
            # CPU fallback
            return df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
    
    def _resample_with_cudf(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Resample using cuDF for GPU acceleration"""
        try:
            # Convert pandas to cuDF
            gdf = cudf.from_pandas(df)
            
            # Reset index to make datetime a column
            gdf = gdf.reset_index()
            
            # Convert to datetime if needed
            if not isinstance(gdf['index'], cudf.core.column.DatetimeColumn):
                gdf['index'] = cudf.to_datetime(gdf['index'])
            
            # Set index back
            gdf = gdf.set_index('index')
            
            # Resample (cuDF has limited resampling, so we'll use a workaround)
            # For now, fall back to CPU for resampling
            # This can be optimized further
            return df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
        except Exception as e:
            logger.warning(f"GPU resampling failed, falling back to CPU: {e}")
            return df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
    
    def calculate_indicators_gpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators using GPU acceleration
        
        Args:
            df: DataFrame with OHLC data
        
        Returns:
            DataFrame with indicators added
        """
        if df.empty or len(df) < 50:
            return df
        
        if self.use_gpu and self.backend == 'cudf' and len(df) > 10000:
            return self._calculate_indicators_with_cudf(df)
        elif self.use_gpu and self.backend == 'torch' and len(df) > 10000:
            return self._calculate_indicators_with_torch(df)
        else:
            return self._calculate_indicators_cpu(df)
    
    def _calculate_indicators_cpu(self, df: pd.DataFrame) -> pd.DataFrame:
        """CPU-based indicator calculation"""
        # RSI
        df['rsi'] = self._calculate_rsi_cpu(df)
        
        # MACD
        macd_df = self._calculate_macd_cpu(df)
        df['macd'] = macd_df['macd']
        df['signal'] = macd_df['signal']
        df['histogram'] = macd_df['histogram']
        
        # Moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # ATR
        df['atr_14'] = self._calculate_atr_cpu(df)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        # Price position
        df['price_vs_sma20'] = (df['close'] / df['sma_20'] - 1) * 100
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df
    
    def _calculate_indicators_with_cudf(self, df: pd.DataFrame) -> pd.DataFrame:
        """cuDF-based indicator calculation"""
        try:
            # Convert to cuDF
            gdf = cudf.from_pandas(df)
            
            # RSI (simplified for cuDF)
            delta = gdf['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            gdf['rsi'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = gdf['close'].ewm(span=12, adjust=False).mean()
            exp2 = gdf['close'].ewm(span=26, adjust=False).mean()
            gdf['macd'] = exp1 - exp2
            gdf['signal'] = gdf['macd'].ewm(span=9, adjust=False).mean()
            gdf['histogram'] = gdf['macd'] - gdf['signal']
            
            # Moving averages
            gdf['sma_20'] = gdf['close'].rolling(window=20).mean()
            gdf['sma_50'] = gdf['close'].rolling(window=50).mean()
            
            # Convert back to pandas
            result_df = gdf.to_pandas()
            
            # Fill NaN values
            result_df = result_df.fillna(method='bfill').fillna(method='ffill')
            
            return result_df
            
        except Exception as e:
            logger.warning(f"GPU indicator calculation failed, falling back to CPU: {e}")
            return self._calculate_indicators_cpu(df)
    
    def _calculate_indicators_with_torch(self, df: pd.DataFrame) -> pd.DataFrame:
        """PyTorch-based indicator calculation"""
        try:
            # Convert to PyTorch tensors
            closes = torch.tensor(df['close'].values, dtype=torch.float32, device='cuda')
            highs = torch.tensor(df['high'].values, dtype=torch.float32, device='cuda')
            lows = torch.tensor(df['low'].values, dtype=torch.float32, device='cuda')
            volumes = torch.tensor(df['volume'].values, dtype=torch.float32, device='cuda')
            
            # RSI calculation on GPU
            rsi = self._calculate_rsi_torch(closes)
            
            # MACD on GPU
            macd, signal, hist = self._calculate_macd_torch(closes)
            
            # Convert back to numpy
            df['rsi'] = rsi.cpu().numpy()
            df['macd'] = macd.cpu().numpy()
            df['signal'] = signal.cpu().numpy()
            df['histogram'] = hist.cpu().numpy()
            
            # CPU-based rolling calculations for now
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['atr_14'] = self._calculate_atr_cpu(df)
            df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
            df['price_vs_sma20'] = (df['close'] / df['sma_20'] - 1) * 100
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            
            return df
            
        except Exception as e:
            logger.warning(f"PyTorch indicator calculation failed, falling back to CPU: {e}")
            return self._calculate_indicators_cpu(df)
    
    def _calculate_rsi_torch(self, closes: 'torch.Tensor', period: int = 14) -> 'torch.Tensor':
        """PyTorch-accelerated RSI calculation"""
        delta = closes[1:] - closes[:-1]
        
        gain = torch.where(delta > 0, delta, torch.zeros_like(delta))
        loss = torch.where(delta < 0, -delta, torch.zeros_like(delta))
        
        gain = torch.cat([torch.zeros(1, device=closes.device), gain])
        loss = torch.cat([torch.zeros(1, device=closes.device), loss])
        
        # Simple moving average using convolution
        kernel = torch.ones(period, device=closes.device) / period
        gain_avg = torch.nn.functional.conv1d(
            gain.view(1, 1, -1), 
            kernel.view(1, 1, -1), 
            padding=period-1
        ).view(-1)[:len(closes)]
        
        loss_avg = torch.nn.functional.conv1d(
            loss.view(1, 1, -1), 
            kernel.view(1, 1, -1), 
            padding=period-1
        ).view(-1)[:len(closes)]
        
        rs = gain_avg / (loss_avg + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_torch(self, closes: 'torch.Tensor', fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple['torch.Tensor', 'torch.Tensor', 'torch.Tensor']:
        """PyTorch-accelerated MACD calculation"""
        # EMA calculation (simplified)
        alpha_fast = 2 / (fast + 1)
        alpha_slow = 2 / (slow + 1)
        
        fast_ema = torch.zeros_like(closes)
        slow_ema = torch.zeros_like(closes)
        
        fast_ema[0] = closes[0]
        slow_ema[0] = closes[0]
        
        for i in range(1, len(closes)):
            fast_ema[i] = alpha_fast * closes[i] + (1 - alpha_fast) * fast_ema[i-1]
            slow_ema[i] = alpha_slow * closes[i] + (1 - alpha_slow) * slow_ema[i-1]
        
        macd_line = fast_ema - slow_ema
        
        # Signal line (EMA of MACD)
        alpha_signal = 2 / (signal + 1)
        signal_line = torch.zeros_like(macd_line)
        signal_line[0] = macd_line[0]
        
        for i in range(1, len(macd_line)):
            signal_line[i] = alpha_signal * macd_line[i] + (1 - alpha_signal) * signal_line[i-1]
        
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_rsi_cpu(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """CPU-based RSI calculation"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_macd_cpu(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """CPU-based MACD calculation"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        })
    
    def _calculate_atr_cpu(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """CPU-based ATR calculation"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.fillna(0)
    
    def get_file_list(self, pair: str, start_date: Optional[str] = None, 
                       end_date: Optional[str] = None) -> List[Path]:
        """
        Get list of CSV files for a pair within date range
        
        Args:
            pair: Trading pair
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
        
        Returns:
            List of file paths
        """
        pair_path = self.base_path / pair
        
        if not pair_path.exists():
            return []
        
        # Find all CSV files
        all_files = []
        for year_dir in sorted(pair_path.glob("*")):
            if year_dir.is_dir() and year_dir.name.isdigit():
                all_files.extend(year_dir.glob("*.csv"))
        
        if not all_files:
            return []
        
        # Filter by date range
        if start_date or end_date:
            filtered = []
            start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.min
            end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.max
            
            for file_path in all_files:
                try:
                    file_date_str = file_path.stem
                    file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                    
                    if file_date < start or file_date > end:
                        continue
                    
                    filtered.append(file_path)
                except:
                    continue
            
            return filtered
        else:
            return all_files
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs"""
        pairs = []
        if self.base_path.exists():
            for item in self.base_path.iterdir():
                if item.is_dir() and '_' in item.name:
                    pairs.append(item.name)
        return sorted(pairs)
    
    def get_date_range_for_pair(self, pair: str) -> Tuple[Optional[str], Optional[str]]:
        """Get available date range for a pair"""
        pair_path = self.base_path / pair
        if not pair_path.exists():
            return None, None
        
        all_dates = []
        for year_dir in pair_path.glob("*"):
            if year_dir.is_dir() and year_dir.name.isdigit():
                for csv_file in year_dir.glob("*.csv"):
                    try:
                        date_str = csv_file.stem
                        all_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                    except:
                        continue
        
        if all_dates:
            return min(all_dates).strftime('%Y-%m-%d'), max(all_dates).strftime('%Y-%m-%d')
        return None, None
    
    def get_gpu_memory_info(self) -> List[Dict]:
        """Get GPU memory information"""
        if not self.use_gpu or self.backend != 'cudf':
            return []
        
        memory_info = []
        for i in range(self.gpu_count):
            with cudf.cuda.Device(i):
                mem_info = cp.cuda.runtime.memGetInfo()
                free_mem = mem_info[0] / 1e9
                total_mem = mem_info[1] / 1e9
                used_mem = total_mem - free_mem
                memory_info.append({
                    'gpu_id': i,
                    'total': total_mem,
                    'used': used_mem,
                    'free': free_mem,
                    'utilization': (used_mem / total_mem) * 100
                })
        
        return memory_info
    
    def shutdown(self):
        """Clean up resources"""
        self.thread_pool.shutdown(wait=True)
        if self.use_gpu and self.backend == 'cudf':
            self.gpu_stream.synchronize()
            cp.get_default_memory_pool().free_all_blocks()