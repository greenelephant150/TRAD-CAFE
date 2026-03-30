"""
Data management for Simon Pullen trading system
Handles loading and preprocessing of forex data
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data loading and preprocessing for trading system
    """
    
    def __init__(self, data_path: str = "/home/grct/Forex/"):
        self.data_path = data_path
        self.cache = {}
        
    def load_pair_data(self, pair: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Load data for a currency pair over a range of years
        """
        all_dfs = []
        
        for year in range(start_year, end_year + 1):
            file_path = os.path.join(self.data_path, pair, f"{year}.csv")
            
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
                
            try:
                df = pd.read_csv(file_path)
                df = self.standardize_columns(df)
                all_dfs.append(df)
                logger.info(f"Loaded {len(df)} rows from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not all_dfs:
            raise ValueError(f"No data found for {pair} from {start_year} to {end_year}")
        
        # Combine all years
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('time')
        combined = combined.drop_duplicates(subset=['time'], keep='first')
        
        # Set time as index
        combined.set_index('time', inplace=True)
        
        logger.info(f"Total data for {pair}: {len(combined)} rows from {combined.index[0]} to {combined.index[-1]}")
        return combined
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to common format
        """
        # Handle time column
        if 'time' in df.columns:
            # Convert time to datetime with error handling
            df['time'] = pd.to_datetime(df['time'], errors='coerce')
            # Drop rows with invalid time
            df = df.dropna(subset=['time'])
        
        # Rename columns to standard format
        column_map = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Ensure all required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to specified timeframe
        """
        # Define aggregation rules
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Resample based on timeframe
        if timeframe == '1h':
            resampled = df.resample('1H').agg(ohlc_dict)
        elif timeframe == '4h':
            resampled = df.resample('4H').agg(ohlc_dict)
        elif timeframe == '15m':
            resampled = df.resample('15T').agg(ohlc_dict)
        elif timeframe == '1d':
            resampled = df.resample('1D').agg(ohlc_dict)
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Drop NaN rows
        resampled = resampled.dropna()
        
        return resampled
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators for pattern detection
        """
        # Price-based features
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
        df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
        df['total_range'] = df['high'] - df['low']
        df['body_ratio'] = df['body_size'] / df['total_range'].replace(0, np.nan)
        
        # Momentum features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling features
        for window in [5, 10, 20]:
            df[f'high_{window}'] = df['high'].rolling(window).max()
            df[f'low_{window}'] = df['low'].rolling(window).min()
            df[f'volatility_{window}'] = df['returns'].rolling(window).std()
        
        return df
