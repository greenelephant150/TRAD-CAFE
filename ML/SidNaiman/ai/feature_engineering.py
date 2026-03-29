"""
Feature Engineering for Sid Naiman's SID Method
Creates features based on RSI, MACD, and price action
Supports loading from CSV or Parquet formats
Includes GPU-accelerated processing
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
    Creates features for AI models based on Sid Naiman's SID Method
    Features: RSI, MACD, price action, volatility
    Much simpler than Simon's complex feature engineering
    """
    
    def __init__(self, accelerator=None, use_gpu: bool = True):
        """
        Args:
            accelerator: AIAccelerator instance (optional)
            use_gpu: Whether to use GPU acceleration
        """
        self.acc = accelerator
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_backend = GPU_BACKEND if self.use_gpu else 'cpu'
        
        # Paths
        self.csv_base_path = Path("/home/grct/Forex")
        self.parquet_base_path = Path("/home/grct/Forex_Parquet")
        
        print(f"📊 FeatureEngineer (SID Method) initialized")
        print(f"   Processing mode: {'GPU (' + self.gpu_backend + ')' if self.use_gpu else 'CPU'}")
    
    # ========================================================================
    # CORE SID METHOD FEATURES
    # ========================================================================
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI - Sid's primary indicator"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, float('nan'))
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def calculate_macd(self, df: pd.DataFrame, fast: int = 12, 
                        slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD - Sid's secondary indicator"""
        exp1 = df['close'].ewm(span=fast, adjust=False).mean()
        exp2 = df['close'].ewm(span=slow, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return pd.DataFrame({
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        })
    
    def calculate_all_sid_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all features for Sid's SID Method
        """
        df = df.copy()
        
        # RSI and derived features
        df['rsi'] = self.calculate_rsi(df)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        df['rsi_mid'] = ((df['rsi'] >= 45) & (df['rsi'] <= 55)).astype(int)
        df['rsi_change'] = df['rsi'].diff()
        df['rsi_change_3'] = df['rsi'].diff(3)
        df['rsi_change_5'] = df['rsi'].diff(5)
        df['rsi_distance_to_50'] = abs(df['rsi'] - 50)
        
        # RSI signal zones
        df['rsi_zone'] = 0
        df.loc[df['rsi'] < 30, 'rsi_zone'] = -1  # Oversold
        df.loc[df['rsi'] > 70, 'rsi_zone'] = 1   # Overbought
        
        # MACD features
        macd_df = self.calculate_macd(df)
        df['macd'] = macd_df['macd']
        df['macd_signal'] = macd_df['signal']
        df['macd_hist'] = macd_df['histogram']
        
        # MACD signals
        df['macd_above_signal'] = (df['macd'] > df['macd_signal']).astype(int)
        df['macd_cross_above'] = ((df['macd'].shift(1) <= df['macd_signal'].shift(1)) & 
                                   (df['macd'] > df['macd_signal'])).astype(int)
        df['macd_cross_below'] = ((df['macd'].shift(1) >= df['macd_signal'].shift(1)) & 
                                   (df['macd'] < df['macd_signal'])).astype(int)
        df['macd_trend'] = (df['macd'] > df['macd'].shift(1)).astype(int)
        
        # Price action
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body_size'] / df['total_range'].replace(0, 1)
        df['wick_ratio'] = (df['upper_wick'] + df['lower_wick']) / df['body_size'].replace(0, 1)
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['returns_abs'] = abs(df['returns'])
        df['returns_cum_3'] = df['returns'].rolling(3).sum()
        df['returns_cum_5'] = df['returns'].rolling(5).sum()
        
        # Volatility
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
            df[f'volume_ma_{period}'] = df['volume'].rolling(window=period).mean()
        
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # Moving averages (for context only)
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['close_vs_sma20'] = (df['close'] / df['sma_20'] - 1) * 100
        df['close_vs_sma50'] = (df['close'] / df['sma_50'] - 1) * 100
        
        # Time features
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['month'] = df.index.month
        
        # Target variables (for training)
        df['target_rsi_50_5'] = 0
        df['target_rsi_50_10'] = 0
        df['target_direction'] = 0
        
        # Calculate targets
        for i in range(len(df) - 5):
            if any(df['rsi'].iloc[i+1:i+6] >= 50):
                df.loc[df.index[i], 'target_rsi_50_5'] = 1
        
        for i in range(len(df) - 10):
            if any(df['rsi'].iloc[i+1:i+11] >= 50):
                df.loc[df.index[i], 'target_rsi_50_10'] = 1
        
        for i in range(len(df) - 1):
            if df['close'].iloc[i+1] > df['close'].iloc[i]:
                df.loc[df.index[i], 'target_direction'] = 1
        
        return df
    
    # ========================================================================
    # DATA LOADING FUNCTIONS
    # ========================================================================
    
    def load_parquet_data(self, pair: str, start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> pd.DataFrame:
        """Load parquet data for a pair"""
        import pyarrow.parquet as pq
        
        pair_path = self.parquet_base_path / pair
        if not pair_path.exists():
            logger.warning(f"Parquet path not found: {pair_path}")
            return pd.DataFrame()
        
        try:
            dataset = pq.ParquetDataset(pair_path, partitioning='hive')
            table = dataset.read()
            df = table.to_pandas()
            
            # Normalize date column
            if 'Date' not in df.columns:
                date_variants = ['date', 'timestamp', 'time', 'datetime']
                for var in date_variants:
                    if var in df.columns:
                        df = df.rename(columns={var: 'Date'})
                        break
            
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date')
                df = df.sort_index()
            
            # Filter by date
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading parquet for {pair}: {e}")
            return pd.DataFrame()
    
    def create_training_dataset(self, pairs: List[str],
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None,
                                  samples_per_pair: int = 100000,
                                  progress_callback: Optional[Callable] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Create training dataset for Sid's method
        """
        X_list = []
        y_list = []
        
        total_pairs = len(pairs)
        
        for pair_idx, pair in enumerate(pairs):
            if progress_callback:
                progress_callback(pair_idx / total_pairs, f"Loading {pair}")
            
            # Load data
            df = self.load_parquet_data(pair, start_date, end_date)
            
            if df.empty or len(df) < 100:
                logger.warning(f"Insufficient data for {pair}")
                continue
            
            # Calculate features
            df = self.calculate_all_sid_features(df)
            df = df.dropna()
            
            if len(df) < 100:
                continue
            
            # Create samples (simpler than Simon's)
            feature_cols = [c for c in df.columns if c not in 
                           ['target_rsi_50_5', 'target_rsi_50_10', 'target_direction',
                            'open', 'high', 'low', 'close', 'volume']]
            
            # Take random samples
            sample_size = min(samples_per_pair, len(df) - 50)
            indices = np.random.choice(range(50, len(df) - 1), sample_size, replace=False)
            
            for idx in indices:
                features = {}
                for col in feature_cols:
                    features[col] = df[col].iloc[idx]
                
                # Add pair code
                features['pair_code'] = hash(pair) % 1000
                
                X_list.append(features)
                y_list.append(df['target_direction'].iloc[idx])
        
        if not X_list:
            logger.warning("No training samples generated")
            return pd.DataFrame(), pd.Series()
        
        X = pd.DataFrame(X_list)
        y = pd.Series(y_list)
        
        logger.info(f"Created training dataset with {len(X)} samples, {len(X.columns)} features")
        return X, y
    
    def get_available_pairs(self) -> List[str]:
        """Get list of available trading pairs"""
        pairs = []
        if self.parquet_base_path.exists():
            for item in self.parquet_base_path.iterdir():
                if item.is_dir() and '_' in item.name:
                    pairs.append(item.name)
        return sorted(pairs)