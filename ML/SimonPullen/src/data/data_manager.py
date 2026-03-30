"""
Data management for Simon Pullen trading system
Handles loading and preprocessing of forex data from daily CSV files
Data format: S5 (5-second) granularity with ISO timestamps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import os
import glob
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataManager:
    """
    Manages data loading and preprocessing for trading system
    Data format: /home/grct/Forex/{pair}/{year}/YYYY-MM-DD.csv
    File contains S5 (5-second) data with columns: time,open,high,low,close,volume
    Time format: 2023-12-28T00:00:00.000000000Z
    """
    
    def __init__(self, data_path: str = "/home/grct/Forex/"):
        self.data_path = data_path
        self.cache = {}
        
    def load_pair_data(self, pair: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Load data for a currency pair over a range of years
        Data is stored in daily CSV files: /home/grct/Forex/{pair}/{year}/YYYY-MM-DD.csv
        """
        all_dfs = []
        
        # Convert pair format (EUR_USD -> EUR_USD)
        pair_file = pair
        
        for year in range(start_year, end_year + 1):
            year_path = os.path.join(self.data_path, pair_file, str(year))
            
            if not os.path.exists(year_path):
                logger.warning(f"Year directory not found: {year_path}")
                continue
                
            # Find all CSV files for this year
            csv_files = glob.glob(os.path.join(year_path, "*.csv"))
            
            if not csv_files:
                logger.warning(f"No CSV files found in {year_path}")
                continue
                
            # Sort files by date
            csv_files.sort()
            
            for file_path in csv_files:
                try:
                    df = pd.read_csv(file_path)
                    df = self.standardize_columns(df)
                    all_dfs.append(df)
                    logger.debug(f"Loaded {len(df)} rows from {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        if not all_dfs:
            # Generate sample data if no real data found
            logger.warning(f"No data found for {pair}, generating sample data")
            return self.generate_sample_data(pair, start_year, end_year)
        
        # Combine all files
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('time')
        combined = combined.drop_duplicates(subset=['time'], keep='first')
        
        # Set time as index
        combined.set_index('time', inplace=True)
        
        logger.info(f"Total data for {pair}: {len(combined)} rows from {combined.index[0]} to {combined.index[-1]}")
        return combined
    
    def load_daily_file(self, pair: str, date_str: str) -> Optional[pd.DataFrame]:
        """
        Load a single daily CSV file
        Format: /home/grct/Forex/{pair}/{year}/{date}.csv
        Returns S5 (5-second) data
        """
        pair_file = pair
        year = date_str[:4]
        file_path = os.path.join(self.data_path, pair_file, year, f"{date_str}.csv")
        
        if not os.path.exists(file_path):
            logger.debug(f"File not found: {file_path}")
            return None
            
        try:
            df = pd.read_csv(file_path)
            df = self.standardize_columns(df)
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def load_date_range(self, pair: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load data for a date range
        """
        all_dfs = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            df = self.load_daily_file(pair, date_str)
            
            if df is not None:
                all_dfs.append(df)
            
            current_date += timedelta(days=1)
        
        if not all_dfs:
            return pd.DataFrame()
            
        combined = pd.concat(all_dfs, ignore_index=True)
        combined = combined.sort_values('time')
        combined.set_index('time', inplace=True)
        
        return combined
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to common format
        Handles ISO timestamp format: 2023-12-28T00:00:00.000000000Z
        """
        # Handle time column - ISO format with Z suffix
        if 'time' in df.columns:
            # Convert time to datetime - handles ISO format with Z
            df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
            # Drop rows with invalid time
            df = df.dropna(subset=['time'])
        elif 'Date' in df.columns:
            df['time'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['time'])
        elif 'DateTime' in df.columns:
            df['time'] = pd.to_datetime(df['DateTime'], errors='coerce')
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
            'Volume': 'volume',
            'OpenPrice': 'open',
            'HighPrice': 'high',
            'LowPrice': 'low',
            'ClosePrice': 'close'
        }
        
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})
        
        # Ensure all required columns exist
        required = ['open', 'high', 'low', 'close']
        for col in required:
            if col not in df.columns:
                if col == 'volume':
                    df['volume'] = 0
                else:
                    raise ValueError(f"Missing required column: {col}")
        
        return df
    
    def resample_to_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample S5 (5-second) data to specified timeframe
        Uses correct pandas frequency strings
        """
        # Define aggregation rules
        ohlc_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        # Ensure we have a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'time' in df.columns:
                df = df.set_index('time')
            else:
                raise ValueError("DataFrame must have datetime index or 'time' column")
        
        # Resample based on timeframe - using correct pandas 2.0+ frequency strings
        if timeframe == '1h':
            resampled = df.resample('h').agg(ohlc_dict)  # Use 'h' not 'H'
        elif timeframe == '4h':
            resampled = df.resample('4h').agg(ohlc_dict)
        elif timeframe == '15m':
            resampled = df.resample('15min').agg(ohlc_dict)
        elif timeframe == '1d':
            resampled = df.resample('D').agg(ohlc_dict)
        elif timeframe == '5s':
            # Already S5, just return
            return df
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        # Drop NaN rows
        resampled = resampled.dropna()
        
        return resampled
    
    def get_recent_data(self, pair: str, days: int = 30, timeframe: str = '1h') -> pd.DataFrame:
        """
        Get recent data for a pair and resample to desired timeframe
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load S5 data for date range
        df = self.load_date_range(pair, start_date, end_date)
        
        if df.empty:
            logger.warning(f"No data found for {pair}, generating sample data")
            df = self.generate_sample_data(pair, start_date.year, end_date.year)
            # Set index for sample data
            if 'time' in df.columns:
                df.set_index('time', inplace=True)
        
        # Resample to desired timeframe if not S5
        if timeframe != '5s' and not df.empty:
            df = self.resample_to_timeframe(df, timeframe)
        
        return df
    
    def get_multiple_pairs(self, pairs: List[str], start_year: int, end_year: int) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple currency pairs
        """
        result = {}
        for pair in pairs:
            try:
                result[pair] = self.load_pair_data(pair, start_year, end_year)
            except Exception as e:
                logger.error(f"Error loading {pair}: {e}")
        
        return result
    
    def generate_sample_data(self, pair: str, start_year: int, end_year: int) -> pd.DataFrame:
        """
        Generate sample S5 data for testing when real data is not available
        """
        logger.info(f"Generating sample S5 data for {pair} from {start_year} to {end_year}")
        
        start_date = datetime(start_year, 1, 1)
        end_date = datetime(end_year, 12, 31)
        
        # Generate S5 (5-second) data - limit to 30 days for sample to avoid huge files
        if (end_date - start_date).days > 30:
            logger.info("Limiting sample data to 30 days to avoid huge dataset")
            start_date = end_date - timedelta(days=30)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='5s')
        
        # Set seed for reproducibility
        np.random.seed(hash(pair) % 2**32)
        
        # Generate random walk
        returns = np.random.randn(len(dates)) * 0.00001  # Very small for 5-second data
        base_price = 1.1000 if 'USD' in pair else 1.0000
        price = base_price * np.exp(np.cumsum(returns))
        
        # Create OHLC data with realistic S5 variations
        df = pd.DataFrame({
            'time': dates,
            'open': price * (1 + np.random.randn(len(dates)) * 0.000001),
            'high': price * (1 + np.abs(np.random.randn(len(dates)) * 0.000002)),
            'low': price * (1 - np.abs(np.random.randn(len(dates)) * 0.000002)),
            'close': price,
            'volume': np.random.randint(1, 100, len(dates))
        })
        
        # Ensure high >= open/close and low <= open/close
        df['high'] = df[['high', 'open', 'close']].max(axis=1)
        df['low'] = df[['low', 'open', 'close']].min(axis=1)
        
        logger.info(f"Generated {len(df)} rows of sample S5 data")
        return df
    
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
        
        # Rolling features - adjust windows for S5 data
        # For 1h (720 periods of 5s), 4h (2880 periods), etc.
        df['high_720'] = df['high'].rolling(720).max()  # ~1 hour
        df['low_720'] = df['low'].rolling(720).min()
        df['high_2880'] = df['high'].rolling(2880).max()  # ~4 hours
        df['low_2880'] = df['low'].rolling(2880).min()
        
        # Volatility
        df['volatility_720'] = df['returns'].rolling(720).std()
        
        return df
    
    def get_hourly_data(self, pair: str, days: int = 30) -> pd.DataFrame:
        """Get hourly data for a pair"""
        return self.get_recent_data(pair, days, '1h')
    
    def get_4h_data(self, pair: str, days: int = 90) -> pd.DataFrame:
        """Get 4-hour data for a pair"""
        return self.get_recent_data(pair, days, '4h')
    
    def get_daily_data(self, pair: str, years: int = 2) -> pd.DataFrame:
        """Get daily data for a pair"""
        return self.get_recent_data(pair, years * 365, '1d')
    
    def check_data_available(self, pair: str, year: int) -> bool:
        """Check if data is available for a given pair and year"""
        pair_file = pair
        year_path = os.path.join(self.data_path, pair_file, str(year))
        return os.path.exists(year_path) and len(glob.glob(os.path.join(year_path, "*.csv"))) > 0
    
    def list_available_pairs(self) -> List[str]:
        """List all pairs with available data"""
        pairs = []
        if not os.path.exists(self.data_path):
            logger.warning(f"Data path {self.data_path} does not exist")
            return pairs
            
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            if os.path.isdir(item_path):
                # Check if it has year subdirectories with CSV files
                has_data = False
                for year_dir in os.listdir(item_path):
                    year_path = os.path.join(item_path, year_dir)
                    if os.path.isdir(year_path) and len(glob.glob(os.path.join(year_path, "*.csv"))) > 0:
                        has_data = True
                        break
                if has_data:
                    pairs.append(item)
        return pairs
