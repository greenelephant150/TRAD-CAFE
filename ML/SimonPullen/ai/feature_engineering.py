"""
Feature Engineering for AI Model Training
Creates features from supply/demand zones and patterns using Forex data
Supports loading from CSV or Parquet (10x faster) formats
Includes GPU-accelerated timeframe conversion (S5 to 1m, 5m, 15m, 30m, 1h, 4h, 1d)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime, timedelta
import logging
import os
import glob
from pathlib import Path
import time
import gc

# GPU imports with fallback
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


class FeatureEngineer:
    """
    Creates features for AI models based on supply/demand zones and patterns
    Features augment the rule-based strategy, don't replace it
    Supports loading from local CSV or Parquet Forex files
    Includes GPU-accelerated timeframe conversion
    """
    
    def __init__(self, accelerator=None, use_gpu: bool = True):
        """
        Args:
            accelerator: AIAccelerator instance for GPU/CPU ops (optional)
            use_gpu: Whether to use GPU acceleration if available
        """
        self.acc = accelerator
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_backend = GPU_BACKEND if self.use_gpu else 'cpu'
        
        if accelerator:
            self.np = accelerator.get_numpy()
            self.pd = accelerator.get_pandas()
        else:
            self.np = np
            self.pd = pd
        
        self.supported_timeframes = ['5s', '1m', '5m', '15m', '30m', '1h', '4h', '1d']
        self.csv_base_path = Path("/home/grct/Forex")
        self.parquet_base_path = Path("/home/grct/Forex_Parquet")
        
        print(f"📊 FeatureEngineer initialized")
        print(f"   CSV path: {self.csv_base_path}")
        print(f"   Parquet path: {self.parquet_base_path}")
        print(f"   Processing mode: {'GPU (' + self.gpu_backend + ')' if self.use_gpu else 'CPU'}")
    
    # ========================================================================
    # GPU-ACCELERATED TIMEFRAME CONVERSION
    # ========================================================================
    
    def resample_s5_to_timeframe_gpu(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample S5 data to target timeframe using GPU acceleration
        
        Args:
            df: DataFrame with S5 OHLCV data
            target_timeframe: Target timeframe ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
        
        Returns:
            Resampled DataFrame at target timeframe
        """
        if df.empty:
            return df
        
        if self.gpu_backend == 'cudf':
            return self._resample_with_cudf(df, target_timeframe)
        elif self.gpu_backend == 'torch':
            return self._resample_with_torch(df, target_timeframe)
        else:
            return self._resample_with_pandas(df, target_timeframe)
    
    def _resample_with_cudf(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample using cuDF (GPU)"""
        try:
            # Convert to cuDF
            gdf = cudf.from_pandas(df)
            
            # Add time columns for grouping
            gdf['date'] = gdf.index.date
            gdf['hour'] = gdf.index.hour
            gdf['minute'] = gdf.index.minute
            
            # Define resampling logic based on timeframe
            if target_timeframe == '1m':
                # Group by date, hour, minute
                gdf['time_key'] = gdf['date'].astype(str) + ' ' + \
                                   gdf['hour'].astype(str).str.zfill(2) + ':' + \
                                   gdf['minute'].astype(str).str.zfill(2)
            elif target_timeframe == '5m':
                # Group by 5-minute blocks
                gdf['minute_block'] = (gdf['minute'] // 5) * 5
                gdf['time_key'] = gdf['date'].astype(str) + ' ' + \
                                   gdf['hour'].astype(str).str.zfill(2) + ':' + \
                                   gdf['minute_block'].astype(str).str.zfill(2)
            elif target_timeframe == '15m':
                gdf['minute_block'] = (gdf['minute'] // 15) * 15
                gdf['time_key'] = gdf['date'].astype(str) + ' ' + \
                                   gdf['hour'].astype(str).str.zfill(2) + ':' + \
                                   gdf['minute_block'].astype(str).str.zfill(2)
            elif target_timeframe == '30m':
                gdf['minute_block'] = (gdf['minute'] // 30) * 30
                gdf['time_key'] = gdf['date'].astype(str) + ' ' + \
                                   gdf['hour'].astype(str).str.zfill(2) + ':' + \
                                   gdf['minute_block'].astype(str).str.zfill(2)
            elif target_timeframe == '1h':
                gdf['time_key'] = gdf['date'].astype(str) + ' ' + \
                                   gdf['hour'].astype(str).str.zfill(2) + ':00'
            elif target_timeframe == '4h':
                gdf['hour_block'] = (gdf['hour'] // 4) * 4
                gdf['time_key'] = gdf['date'].astype(str) + ' ' + \
                                   gdf['hour_block'].astype(str).str.zfill(2) + ':00'
            elif target_timeframe == '1d':
                gdf['time_key'] = gdf['date'].astype(str)
            else:
                # Default to pandas resample
                return self._resample_with_pandas(df, target_timeframe)
            
            # Group by time_key and aggregate
            resampled = gdf.groupby('time_key').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index()
            
            # Convert back to datetime index
            resampled['datetime'] = cudf.to_datetime(resampled['time_key'])
            resampled = resampled.set_index('datetime')
            resampled = resampled.drop('time_key', axis=1)
            
            # Convert back to pandas
            result = resampled.to_pandas()
            
            return result
            
        except Exception as e:
            print(f"   GPU resampling failed: {e}, falling back to CPU")
            return self._resample_with_pandas(df, target_timeframe)
    
    def _resample_with_torch(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample using PyTorch (GPU) - falls back to pandas for now"""
        print(f"   PyTorch GPU detected, using CPU for resampling")
        return self._resample_with_pandas(df, target_timeframe)
    
    def _resample_with_pandas(self, df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample using pandas (CPU)"""
        # Define resample rules
        resample_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '1H',
            '4h': '4H',
            '1d': '1D',
            '1min': '1min',
            '5min': '5min',
            '15min': '15min',
            '30min': '30min',
            '1H': '1H',
            '4H': '4H',
            '1D': '1D'
        }
        
        rule = resample_map.get(target_timeframe, '1H')
        
        # Resample with proper OHLC aggregation
        resampled = df.resample(rule).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return resampled
    
    def load_and_resample(self, pair: str, 
                           target_timeframe: str = '1h',
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           use_parquet: bool = True,
                           use_gpu: bool = True,
                           progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Load S5 data and resample to target timeframe in one step
        User can choose GPU or CPU processing
        
        Args:
            pair: Trading pair
            target_timeframe: Desired output timeframe
            start_date: Start date
            end_date: End date
            use_parquet: Use Parquet if available
            use_gpu: Whether to use GPU for resampling
            progress_callback: Progress callback
        
        Returns:
            Resampled DataFrame at target timeframe with indicators
        """
        # Temporarily override GPU setting for this operation
        original_gpu = self.use_gpu
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        try:
            # Load the raw S5 data
            df = self.load_data(
                pair=pair,
                use_parquet=use_parquet,
                start_date=start_date,
                end_date=end_date,
                target_timeframe='5s',  # Keep as S5 for resampling
                progress_callback=progress_callback
            )
            
            if df.empty:
                return df
            
            # Resample to target timeframe with selected backend
            if progress_callback:
                progress_callback(0.7, f"Resampling to {target_timeframe}...")
            
            resampled = self.resample_s5_to_timeframe_gpu(df, target_timeframe)
            
            if progress_callback:
                progress_callback(0.9, "Calculating indicators...")
            
            # Calculate indicators on the resampled data
            resampled = self._calculate_all_indicators(resampled)
            
            return resampled
            
        finally:
            # Restore original GPU setting
            self.use_gpu = original_gpu
    
    def create_multi_timeframe_dataset(self, pair: str,
                                         timeframes: List[str] = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                                         start_date: Optional[str] = None,
                                         end_date: Optional[str] = None,
                                         use_parquet: bool = True,
                                         use_gpu: bool = True,
                                         progress_callback: Optional[Callable] = None) -> Dict[str, pd.DataFrame]:
        """
        Create multiple timeframe datasets from S5 data
        
        Args:
            pair: Trading pair
            timeframes: List of timeframes to generate
            start_date: Start date
            end_date: End date
            use_parquet: Use Parquet if available
            use_gpu: Use GPU acceleration
            progress_callback: Progress callback
        
        Returns:
            Dictionary of {timeframe: DataFrame}
        """
        result = {}
        total_timeframes = len(timeframes)
        
        # Load S5 data once
        if progress_callback:
            progress_callback(0.1, f"Loading S5 data for {pair}...")
        
        s5_data = self.load_data(
            pair=pair,
            use_parquet=use_parquet,
            start_date=start_date,
            end_date=end_date,
            target_timeframe='5s',
            progress_callback=lambda p, msg: None
        )
        
        if s5_data.empty:
            return result
        
        # Generate each timeframe
        for i, tf in enumerate(timeframes):
            if progress_callback:
                progress = 0.2 + (0.8 * i / total_timeframes)
                progress_callback(progress, f"Resampling to {tf}...")
            
            # Temporarily set GPU for this operation
            original_gpu = self.use_gpu
            self.use_gpu = use_gpu and GPU_AVAILABLE
            
            try:
                resampled = self.resample_s5_to_timeframe_gpu(s5_data, tf)
                if not resampled.empty:
                    resampled = self._calculate_all_indicators(resampled)
                    result[tf] = resampled
            finally:
                self.use_gpu = original_gpu
        
        return result
    
    def save_resampled_parquet(self, pair: str,
                                 timeframes: List[str] = ['1m', '5m', '15m', '30m', '1h', '4h', '1d'],
                                 start_date: Optional[str] = None,
                                 end_date: Optional[str] = None,
                                 use_gpu: bool = True,
                                 mode: str = 'append',
                                 progress_callback: Optional[Callable] = None) -> Dict[str, bool]:
        """
        Create and save multiple timeframe Parquet files
        
        Args:
            pair: Trading pair
            timeframes: List of timeframes to generate
            start_date: Start date
            end_date: End date
            use_gpu: Use GPU acceleration
            mode: 'append', 'overwrite', or 'update'
            progress_callback: Progress callback
        
        Returns:
            Dictionary of {timeframe: success}
        """
        from ai.parquet_converter import ParquetConverter
        
        results = {}
        
        # Create multi-timeframe dataset
        datasets = self.create_multi_timeframe_dataset(
            pair=pair,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            use_parquet=True,
            use_gpu=use_gpu,
            progress_callback=progress_callback
        )
        
        # Save each timeframe
        converter = ParquetConverter(use_gpu=use_gpu)
        
        for tf, df in datasets.items():
            if not df.empty:
                # Save to timeframe-specific subdirectory
                tf_pair_path = self.parquet_base_path / f"{tf}" / pair
                tf_pair_path.mkdir(parents=True, exist_ok=True)
                
                # Add partition columns
                df_copy = df.copy()
                df_copy['year'] = df_copy.index.year
                df_copy['month'] = df_copy.index.month
                df_copy['day'] = df_copy.index.day
                
                # Save
                df_copy.to_parquet(
                    tf_pair_path,
                    partition_cols=['year', 'month', 'day'],
                    compression='snappy'
                )
                
                results[tf] = True
                print(f"   Saved {tf} data for {pair}: {len(df)} rows")
            else:
                results[tf] = False
        
        return results
    
    # ========================================================================
    # PARQUET LOADING (10x FASTER)
    # ========================================================================
    
    def load_parquet_data(self, pair: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           target_timeframe: str = '1h',
                           progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Load pre-converted Parquet data (MUCH faster than CSV)
        
        Args:
            pair: Trading pair (e.g., 'GBP_USD')
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            target_timeframe: Desired timeframe for resampling
            progress_callback: Function to call with progress updates
        
        Returns:
            DataFrame with OHLC data at specified timeframe
        """
        # First try loading from timeframe-specific directory
        tf_parquet_path = self.parquet_base_path / target_timeframe / pair
        
        if tf_parquet_path.exists():
            return self._load_from_tf_parquet(tf_parquet_path, start_date, end_date, progress_callback)
        
        # Fall back to raw S5 parquet
        parquet_path = self.parquet_base_path / pair
        
        if not parquet_path.exists():
            logger.warning(f"Parquet data not found for {pair}, falling back to CSV")
            return self.load_csv_data(pair, start_date, end_date, target_timeframe, progress_callback)
        
        # Load S5 data and resample on the fly
        df = self._load_from_s5_parquet(parquet_path, start_date, end_date, progress_callback)
        
        if df.empty:
            return df
        
        # Resample to target timeframe
        if target_timeframe != '5s':
            df = self.resample_s5_to_timeframe_gpu(df, target_timeframe)
        
        # Calculate indicators
        df = self._calculate_all_indicators(df)
        
        return df
    
    def _load_from_tf_parquet(self, parquet_path: Path,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """Load from pre-resampled timeframe-specific parquet"""
        parquet_files = sorted(parquet_path.glob("**/*.parquet"))
        
        if not parquet_files:
            return pd.DataFrame()
        
        dfs = []
        total_files = len(parquet_files)
        
        for i, file_path in enumerate(parquet_files):
            if progress_callback:
                progress_callback(i / total_files, f"Loading parquet {file_path.name}")
            
            try:
                df = pd.read_parquet(file_path)
                
                # Filter by date range if needed
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined.sort_index(inplace=True)
        
        return combined
    
    def _load_from_s5_parquet(self, parquet_path: Path,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """Load raw S5 parquet data"""
        # Determine which years to load
        start_year = int(start_date[:4]) if start_date else 1900
        end_year = int(end_date[:4]) if end_date else 2100
        
        # Find relevant parquet files
        parquet_files = []
        for year_dir in parquet_path.glob("year=*"):
            try:
                year = int(year_dir.name.split('=')[1])
                if start_year <= year <= end_year:
                    for month_dir in year_dir.glob("month=*"):
                        parquet_files.extend(month_dir.glob("*.parquet"))
            except:
                continue
        
        if not parquet_files:
            return pd.DataFrame()
        
        # Load parquet files
        dfs = []
        total_files = len(parquet_files)
        
        for i, file_path in enumerate(sorted(parquet_files)):
            if progress_callback:
                progress_callback(i / total_files, f"Loading S5 parquet {file_path.name}")
            
            try:
                df = pd.read_parquet(file_path)
                
                # Filter by exact date range if needed
                if start_date:
                    df = df[df.index >= start_date]
                if end_date:
                    df = df[df.index <= end_date]
                
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined.sort_index(inplace=True)
        
        return combined
    
    # ========================================================================
    # CSV LOADING (Original, with your exact folder structure)
    # ========================================================================
    
    def load_csv_data(self, pair: str, start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       target_timeframe: str = '1h',
                       progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Load Forex data from CSV files with your exact folder structure:
        /home/grct/Forex/{pair}/{year}/{YYYY-MM-DD}.csv
        
        Args:
            pair: Trading pair (e.g., 'GBP_USD')
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            target_timeframe: Desired timeframe for resampling
            progress_callback: Function to call with progress updates
        
        Returns:
            DataFrame with OHLC data at specified timeframe
        """
        pair_path = self.csv_base_path / pair
        
        if not pair_path.exists():
            logger.error(f"Pair directory not found: {pair_path}")
            return pd.DataFrame()
        
        # Find all CSV files within date range
        all_files = []
        for year_dir in sorted(pair_path.glob("*")):
            if year_dir.is_dir() and year_dir.name.isdigit():
                all_files.extend(year_dir.glob("*.csv"))
        
        if not all_files:
            logger.warning(f"No CSV files found for {pair}")
            return pd.DataFrame()
        
        # Filter by date range
        filtered_files = self._filter_files_by_date(all_files, start_date, end_date)
        
        if not filtered_files:
            logger.warning(f"No files in date range for {pair}")
            return pd.DataFrame()
        
        logger.info(f"Loading {len(filtered_files)} CSV files for {pair}")
        
        # Load files
        dfs = []
        total_files = len(filtered_files)
        
        for i, file_path in enumerate(sorted(filtered_files)):
            if progress_callback:
                progress = (i + 1) / total_files
                progress_callback(progress, f"Loading CSV {file_path.name}")
            
            df = self._load_single_csv(file_path)
            if df is not None and not df.empty:
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Combine all data
        combined = pd.concat(dfs)
        combined = combined[~combined.index.duplicated(keep='first')]
        combined.sort_index(inplace=True)
        
        # Resample to target timeframe
        if target_timeframe != '5s':
            combined = self.resample_s5_to_timeframe_gpu(combined, target_timeframe)
        
        # Calculate indicators
        combined = self._calculate_all_indicators(combined)
        
        logger.info(f"✅ Loaded {len(combined)} {target_timeframe} candles for {pair} from CSV")
        return combined
    
    def load_data(self, pair: str, use_parquet: bool = True,
                   start_date: Optional[str] = None,
                   end_date: Optional[str] = None,
                   target_timeframe: str = '1h',
                   progress_callback: Optional[Callable] = None) -> pd.DataFrame:
        """
        Universal data loader - chooses fastest available method
        
        Args:
            pair: Trading pair
            use_parquet: Whether to try Parquet first
            start_date: Start date
            end_date: End date
            target_timeframe: Desired timeframe
            progress_callback: Progress callback
        
        Returns:
            DataFrame with OHLC data
        """
        if use_parquet:
            # Try Parquet first
            df = self.load_parquet_data(pair, start_date, end_date, target_timeframe, progress_callback)
            if not df.empty:
                return df
        
        # Fall back to CSV
        return self.load_csv_data(pair, start_date, end_date, target_timeframe, progress_callback)
    
    def _filter_files_by_date(self, files: List[Path], start_date: Optional[str], 
                                end_date: Optional[str]) -> List[Path]:
        """Filter CSV files by date range"""
        if start_date is None and end_date is None:
            return files  # Return all files for MAX duration
        
        filtered = []
        
        start = datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.min
        end = datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.max
        
        for file_path in files:
            try:
                file_date_str = file_path.stem
                file_date = datetime.strptime(file_date_str, '%Y-%m-%d')
                
                if file_date < start or file_date > end:
                    continue
                
                filtered.append(file_path)
            except:
                continue
        
        return filtered
    
    def _load_single_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Load a single S5 granularity CSV file"""
        try:
            df = pd.read_csv(file_path)
            
            required = ['time', 'open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                logger.warning(f"Missing columns in {file_path}")
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
    
    # ========================================================================
    # INDICATOR CALCULATIONS
    # ========================================================================
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for Simon Pullen's method"""
        if df.empty or len(df) < 50:
            return df
        
        # RSI (14 period)
        df['rsi'] = self.calculate_rsi(df, 14)
        
        # MACD (12,26,9)
        macd_df = self.calculate_macd(df, 12, 26, 9)
        df['macd'] = macd_df['macd']
        df['signal'] = macd_df['signal']
        df['histogram'] = macd_df['histogram']
        
        # Moving averages (for trend detection)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # ATR for volatility
        df['atr_14'] = self.calculate_atr(df, 14)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        
        # Price position relative to MAs
        df['price_vs_sma20'] = (df['close'] / df['sma_20'] - 1) * 100
        df['price_vs_sma50'] = (df['close'] / df['sma_50'] - 1) * 100
        
        # Volume analysis
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Support/Resistance proximity (simplified)
        df['support_proximity'] = self._calculate_support_proximity(df)
        df['resistance_proximity'] = self._calculate_resistance_proximity(df)
        
        return df
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        if df.empty or len(df) < period + 1:
            return pd.Series(index=df.index, data=50)
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD"""
        if df.empty or len(df) < slow + signal:
            return pd.DataFrame({
                'macd': pd.Series(index=df.index, data=0),
                'signal': pd.Series(index=df.index, data=0),
                'histogram': pd.Series(index=df.index, data=0)
            })
        
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        })
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr.fillna(0)
    
    def _calculate_support_proximity(self, df: pd.DataFrame, lookback: int = 50) -> pd.Series:
        """Calculate distance to nearest support level"""
        result = pd.Series(index=df.index, data=100.0)
        
        for i in range(lookback, len(df)):
            recent = df.iloc[i-lookback:i]
            current_price = df.iloc[i]['close']
            
            lows = recent['low'].values
            supports = lows[lows < current_price]
            
            if len(supports) > 0:
                nearest_support = max(supports)
                result.iloc[i] = ((current_price - nearest_support) / current_price) * 100
        
        return result
    
    def _calculate_resistance_proximity(self, df: pd.DataFrame, lookback: int = 50) -> pd.Series:
        """Calculate distance to nearest resistance level"""
        result = pd.Series(index=df.index, data=100.0)
        
        for i in range(lookback, len(df)):
            recent = df.iloc[i-lookback:i]
            current_price = df.iloc[i]['close']
            
            highs = recent['high'].values
            resistances = highs[highs > current_price]
            
            if len(resistances) > 0:
                nearest_resistance = min(resistances)
                result.iloc[i] = ((nearest_resistance - current_price) / current_price) * 100
        
        return result
    
    # ========================================================================
    # TRAINING DATASET CREATION
    # ========================================================================
    
    def _extract_training_samples_to_xy(self, df: pd.DataFrame, pair: str,
                                           max_samples: int = 1000000) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract training samples from DataFrame and return as X, y"""
        if df.empty or len(df) < 100:
            return pd.DataFrame(), pd.Series()
        
        X_list = []
        y_list = []
        
        available_positions = len(df) - 100
        num_samples = min(max_samples, available_positions)
        
        step = max(1, available_positions // num_samples)
        indices = list(range(100, len(df) - 50, step))[:num_samples]
        
        for idx in indices:
            zone_start = max(0, idx - np.random.randint(10, 30))
            zone_end = idx
            
            zone_low = df['low'].iloc[zone_start:zone_end+1].min()
            zone_high = df['high'].iloc[zone_start:zone_end+1].max()
            zone_price = df['close'].iloc[idx]
            
            if df['close'].iloc[idx] > df['close'].iloc[idx-10]:
                zone_type = 'demand'
            else:
                zone_type = 'supply'
            
            future_bars = min(20, len(df) - idx - 1)
            if future_bars > 10:
                future_return = (df['close'].iloc[idx+future_bars-1] - zone_price) / zone_price * 100
                
                if zone_type == 'demand':
                    outcome = 1 if future_return > 1.5 else 0
                else:
                    outcome = 1 if future_return < -1.5 else 0
            else:
                outcome = 0
            
            consolidation_bars = zone_end - zone_start + 1
            price_range = (zone_high - zone_low) / zone_low * 100
            quality = max(30, min(95, 70 - price_range))
            
            features = {
                'zone_quality': quality,
                'zone_type': 1 if zone_type == 'demand' else 0,
                'consolidation_candles': consolidation_bars,
                'volatility': df['volatility'].iloc[idx] if 'volatility' in df.columns else 1.0,
                'rsi': df['rsi'].iloc[idx] if 'rsi' in df.columns else 50,
                'macd': df['macd'].iloc[idx] if 'macd' in df.columns else 0,
                'macd_signal': df['signal'].iloc[idx] if 'signal' in df.columns else 0,
                'price_vs_sma20': df['price_vs_sma20'].iloc[idx] if 'price_vs_sma20' in df.columns else 0,
                'atr': df['atr_14'].iloc[idx] if 'atr_14' in df.columns else 0.001,
                'volume_ratio': df['volume_ratio'].iloc[idx] if 'volume_ratio' in df.columns else 1,
                'support_proximity': df['support_proximity'].iloc[idx] if 'support_proximity' in df.columns else 50,
                'resistance_proximity': df['resistance_proximity'].iloc[idx] if 'resistance_proximity' in df.columns else 50,
                'hour': df.index[idx].hour,
                'day_of_week': df.index[idx].weekday(),
                'month': df.index[idx].month,
                'year': df.index[idx].year
            }
            
            X_list.append(features)
            y_list.append(outcome)
        
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        return X, y
    
    def create_training_dataset_from_forex(self, pairs: List[str],
                                             start_date: Optional[str] = None,
                                             end_date: Optional[str] = None,
                                             timeframe: str = '1h',
                                             samples_per_pair: int = 100000,
                                             use_parquet: bool = True,
                                             use_gpu: bool = True,
                                             progress_callback: Optional[Callable] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training dataset from Forex data with GPU-accelerated resampling
        
        Args:
            pairs: List of trading pairs
            start_date: Start date or None for MAX
            end_date: End date or None for MAX
            timeframe: Target timeframe
            samples_per_pair: Number of samples per pair
            use_parquet: Whether to use Parquet (faster)
            use_gpu: Whether to use GPU for resampling
            progress_callback: Function for progress updates
        
        Returns:
            X (features), y (labels) for training
        """
        X_list = []
        y_list = []
        
        total_pairs = len(pairs)
        
        for pair_idx, pair in enumerate(pairs):
            if progress_callback:
                progress_callback(
                    pair_idx / total_pairs,
                    f"Loading data for {pair}..."
                )
            
            # Load and resample data (GPU accelerated if selected)
            df = self.load_and_resample(
                pair=pair,
                target_timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                use_parquet=use_parquet,
                use_gpu=use_gpu,
                progress_callback=lambda p, msg: progress_callback(
                    (pair_idx + p) / total_pairs,
                    f"{pair}: {msg}"
                ) if progress_callback else None
            )
            
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {pair}")
                continue
            
            X_pair, y_pair = self._extract_training_samples_to_xy(
                df, pair, max_samples=samples_per_pair
            )
            
            if not X_pair.empty:
                X_pair['pair_code'] = hash(pair) % 10000
                X_pair['timeframe_code'] = hash(timeframe) % 100
                
                X_list.append(X_pair)
                y_list.append(y_pair)
        
        if not X_list:
            logger.warning("No training samples generated")
            return pd.DataFrame(), pd.Series()
        
        X = pd.concat(X_list, ignore_index=True)
        y = pd.concat(y_list, ignore_index=True)
        
        logger.info(f"✅ Created training dataset with {len(X)} samples, {len(X.columns)} features")
        return X, y
    
    # ========================================================================
    # PARQUET UTILITY METHODS
    # ========================================================================
    
    def save_to_parquet(self, df: pd.DataFrame, pair: str, mode: str = 'append') -> bool:
        """
        Save DataFrame to Parquet with partition by date
        
        Args:
            df: DataFrame to save
            pair: Trading pair
            mode: 'append', 'overwrite', or 'update'
        
        Returns:
            True if successful
        """
        parquet_path = self.parquet_base_path / pair
        parquet_path.mkdir(parents=True, exist_ok=True)
        
        # Add partition columns
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        df_copy['month'] = df_copy.index.month
        df_copy['day'] = df_copy.index.day
        
        if mode == 'append':
            df_copy.to_parquet(
                parquet_path,
                partition_cols=['year', 'month', 'day'],
                compression='snappy'
            )
            logger.info(f"Appended {len(df)} rows to Parquet for {pair}")
        elif mode == 'overwrite':
            df_copy.to_parquet(
                parquet_path,
                partition_cols=['year', 'month', 'day'],
                compression='snappy'
            )
            logger.info(f"Overwrote Parquet for {pair} with {len(df)} rows")
        elif mode == 'update':
            df_copy.to_parquet(
                parquet_path,
                partition_cols=['year', 'month', 'day'],
                compression='snappy'
            )
            logger.info(f"Updated Parquet for {pair} with {len(df)} rows")
        
        return True
    
    def update_parquet_from_csv(self, pair: str, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  use_gpu: bool = True,
                                  progress_callback: Optional[Callable] = None) -> bool:
        """
        Update Parquet files with new/modified CSV data
        
        Args:
            pair: Trading pair
            start_date: Start date for update
            end_date: End date for update
            use_gpu: Whether to use GPU acceleration
            progress_callback: Progress callback
        
        Returns:
            True if successful
        """
        from ai.parquet_converter import ParquetConverter
        
        converter = ParquetConverter(
            csv_base_path=str(self.csv_base_path),
            parquet_base_path=str(self.parquet_base_path),
            use_gpu=use_gpu
        )
        
        return converter.update_pair(
            pair=pair,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback
        )
    
    def get_parquet_info(self, pair: str) -> Dict:
        """Get information about Parquet files for a pair"""
        from ai.parquet_converter import ParquetConverter
        
        converter = ParquetConverter(
            csv_base_path=str(self.csv_base_path),
            parquet_base_path=str(self.parquet_base_path)
        )
        
        return converter.get_pair_info(pair)
    
    def get_parquet_pairs(self) -> List[str]:
        """Get list of pairs that have been converted to Parquet"""
        pairs = []
        if self.parquet_base_path.exists():
            for item in self.parquet_base_path.iterdir():
                if item.is_dir() and '_' in item.name:
                    pairs.append(item.name)
        return sorted(pairs)
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs from CSV directory"""
        pairs = []
        if self.csv_base_path.exists():
            for item in self.csv_base_path.iterdir():
                if item.is_dir() and '_' in item.name:
                    pairs.append(item.name)
        return sorted(pairs)
    
    def get_date_range_for_pair(self, pair: str, use_parquet: bool = True) -> Tuple[Optional[str], Optional[str]]:
        """Get available date range for a pair"""
        if use_parquet:
            parquet_path = self.parquet_base_path / pair
            if parquet_path.exists():
                parquet_files = list(parquet_path.glob("**/*.parquet"))
                if parquet_files:
                    try:
                        first_file = pd.read_parquet(parquet_files[0])
                        last_file = pd.read_parquet(parquet_files[-1])
                        min_date = first_file.index.min().strftime('%Y-%m-%d')
                        max_date = last_file.index.max().strftime('%Y-%m-%d')
                        return min_date, max_date
                    except:
                        pass
        
        # Fall back to CSV method
        pair_path = self.csv_base_path / pair
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