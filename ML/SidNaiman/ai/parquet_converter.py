"""
Parquet Converter for Forex CSV Files
Converts CSV to Parquet format with append/update capability
Supports GPU acceleration for faster processing
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import time
from typing import List, Optional, Dict, Tuple, Callable
import os
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

from ai.gpu_data_loader import GPUDataLoader

logger = logging.getLogger(__name__)


class ParquetConverter:
    """
    Converts CSV Forex files to Parquet format with append/update capability
    Supports GPU acceleration and partition-based updates
    """
    
    def __init__(self, csv_base_path: str = "/home/grct/Forex", 
                 parquet_base_path: str = "/home/grct/Forex_Parquet",
                 use_gpu: bool = True):
        """
        Args:
            csv_base_path: Path to original CSV files
            parquet_base_path: Path to store converted Parquet files
            use_gpu: Whether to use GPU acceleration if available
        """
        self.csv_base_path = Path(csv_base_path)
        self.parquet_base_path = Path(parquet_base_path)
        self.loader = GPUDataLoader(csv_base_path)
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.gpu_backend = GPU_BACKEND if self.use_gpu else 'cpu'
        
        # Create parquet directory if it doesn't exist
        self.parquet_base_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 ParquetConverter initialized")
        print(f"   CSV source: {self.csv_base_path}")
        print(f"   Parquet destination: {self.parquet_base_path}")
        print(f"   Processing mode: {'GPU (' + self.gpu_backend + ')' if self.use_gpu else 'CPU'}")
    
    def get_existing_partitions(self, pair: str) -> Dict[Tuple[int, int, int], bool]:
        """
        Get existing date partitions for a pair
        
        Args:
            pair: Trading pair
        
        Returns:
            Dictionary of {(year, month, day): exists}
        """
        pair_path = self.parquet_base_path / pair
        if not pair_path.exists():
            return {}
        
        partitions = {}
        
        # Look for year/month/day partition structure
        for year_dir in pair_path.glob("year=*"):
            try:
                year = int(year_dir.name.split('=')[1])
                for month_dir in year_dir.glob("month=*"):
                    try:
                        month = int(month_dir.name.split('=')[1])
                        # Look for day files
                        day_files = list(month_dir.glob("*.parquet"))
                        for day_file in day_files:
                            # Try to extract day from filename
                            if 'day=' in day_file.name:
                                day = int(day_file.name.split('=')[1].split('.')[0])
                                partitions[(year, month, day)] = True
                            else:
                                # If no day in filename, assume it contains all days
                                # This is a fallback - we'll mark the whole month
                                for day in range(1, 32):
                                    partitions[(year, month, day)] = True
                    except Exception as e:
                        logger.debug(f"Error processing month dir {month_dir}: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error processing year dir {year_dir}: {e}")
                continue
        
        return partitions
    
    def get_existing_years(self, pair: str) -> List[int]:
        """Get list of years that already have Parquet files"""
        pair_path = self.parquet_base_path / pair
        if not pair_path.exists():
            return []
        
        years = []
        for year_dir in pair_path.glob("year=*"):
            try:
                year = int(year_dir.name.split('=')[1])
                years.append(year)
            except:
                continue
        
        return sorted(years)
    
    def get_existing_dates(self, pair: str) -> List[datetime]:
        """Get list of all dates that exist in Parquet"""
        pair_path = self.parquet_base_path / pair
        if not pair_path.exists():
            return []
        
        dates = []
        for year_dir in pair_path.glob("year=*"):
            try:
                year = int(year_dir.name.split('=')[1])
                for month_dir in year_dir.glob("month=*"):
                    try:
                        month = int(month_dir.name.split('=')[1])
                        for day_file in month_dir.glob("*.parquet"):
                            if 'day=' in day_file.name:
                                day = int(day_file.name.split('=')[1].split('.')[0])
                                dates.append(datetime(year, month, day))
                            else:
                                # If no day partition, we can't know exact days
                                # We'll skip for now
                                pass
                    except:
                        continue
            except:
                continue
        
        return sorted(dates)
    
    def convert_pair_gpu(self, df: pd.DataFrame, pair: str, 
                           mode: str = 'append') -> bool:
        """
        Convert DataFrame to Parquet using GPU acceleration
        
        Args:
            df: DataFrame to convert
            pair: Trading pair
            mode: 'initial', 'append', or 'overwrite'
        
        Returns:
            True if successful
        """
        if self.gpu_backend == 'cudf':
            return self._convert_with_cudf(df, pair, mode)
        elif self.gpu_backend == 'torch':
            return self._convert_with_torch(df, pair, mode)
        else:
            return self._convert_with_pandas(df, pair, mode)
    
    def _convert_with_cudf(self, df: pd.DataFrame, pair: str, mode: str) -> bool:
        """Convert using cuDF (GPU)"""
        try:
            # Convert pandas to cuDF
            gdf = cudf.from_pandas(df)
            
            # Add partition columns
            gdf['year'] = gdf.index.year
            gdf['month'] = gdf.index.month
            gdf['day'] = gdf.index.day
            
            pair_path = self.parquet_base_path / pair
            
            if mode == 'initial':
                # First time - write all data
                gdf.to_parquet(
                    str(pair_path),
                    partition_cols=['year', 'month', 'day'],
                    compression='snappy'
                )
                print(f"   GPU: Initial write complete")
                
            elif mode == 'append':
                # Get existing partitions to avoid duplicates
                existing = self.get_existing_partitions(pair)
                
                # Filter out dates that already exist
                if existing:
                    # Create a boolean mask for rows to keep
                    mask = []
                    for i in range(len(gdf)):
                        year = int(gdf['year'].iloc[i])
                        month = int(gdf['month'].iloc[i])
                        day = int(gdf['day'].iloc[i])
                        key = (year, month, day)
                        mask.append(key not in existing)
                    
                    # Apply mask
                    gdf = gdf[mask]
                    
                    if len(gdf) == 0:
                        print(f"   GPU: No new dates to append")
                        return True
                
                # Append new partitions
                if len(gdf) > 0:
                    gdf.to_parquet(
                        str(pair_path),
                        partition_cols=['year', 'month', 'day'],
                        compression='snappy'
                    )
                    print(f"   GPU: Appended {len(gdf)} new rows")
                
            elif mode == 'overwrite':
                # Replace everything
                gdf.to_parquet(
                    str(pair_path),
                    partition_cols=['year', 'month', 'day'],
                    compression='snappy'
                )
                print(f"   GPU: Overwrite complete")
            
            return True
            
        except Exception as e:
            print(f"   GPU conversion failed: {e}, falling back to CPU")
            return self._convert_with_pandas(df, pair, mode)
    
    def _convert_with_torch(self, df: pd.DataFrame, pair: str, mode: str) -> bool:
        """Convert using PyTorch (GPU) - simplified, falls back to pandas for Parquet"""
        print(f"   PyTorch GPU detected, using CPU for Parquet writing")
        return self._convert_with_pandas(df, pair, mode)
    
    def _convert_with_pandas(self, df: pd.DataFrame, pair: str, mode: str) -> bool:
        """Convert using pandas (CPU)"""
        # Add partition columns
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        df_copy['month'] = df_copy.index.month
        df_copy['day'] = df_copy.index.day
        
        pair_path = self.parquet_base_path / pair
        
        if mode == 'initial':
            # First time - write all data
            df_copy.to_parquet(
                pair_path,
                partition_cols=['year', 'month', 'day'],
                compression='snappy'
            )
            print(f"   CPU: Initial write complete")
            
        elif mode == 'append':
            # Get existing partitions to avoid duplicates
            existing = self.get_existing_partitions(pair)
            
            # Filter out dates that already exist
            if existing:
                # Create a boolean mask
                mask = []
                for idx in df_copy.index:
                    key = (idx.year, idx.month, idx.day)
                    mask.append(key not in existing)
                
                df_copy = df_copy[mask]
                
                if len(df_copy) == 0:
                    print(f"   CPU: No new dates to append")
                    return True
            
            # Append new partitions
            if len(df_copy) > 0:
                df_copy.to_parquet(
                    pair_path,
                    partition_cols=['year', 'month', 'day'],
                    compression='snappy'
                )
                print(f"   CPU: Appended {len(df_copy)} new rows")
            
        elif mode == 'overwrite':
            # Replace everything
            df_copy.to_parquet(
                pair_path,
                partition_cols=['year', 'month', 'day'],
                compression='snappy'
            )
            print(f"   CPU: Overwrite complete")
        
        return True
    
    def convert_pair(self, pair: str, 
                      mode: str = 'append',
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      chunk_size: int = 100000,
                      progress_callback: Optional[Callable] = None) -> bool:
        """
        Convert CSV files for a pair to Parquet with update capability
        
        Args:
            pair: Trading pair
            mode: 'initial' (first time), 'append' (add new dates), 
                  'overwrite' (replace everything), 'update' (refresh existing)
            start_date: Optional start date filter
            end_date: Optional end date filter
            chunk_size: Rows per chunk for memory management
            progress_callback: Progress update function
        
        Returns:
            True if successful
        """
        print(f"\n🔄 {'GPU' if self.use_gpu else 'CPU'} Converting {pair} to Parquet (mode: {mode})...")
        start_time = time.time()
        
        # Get all CSV files for this pair
        file_list = self.loader.get_file_list(pair, start_date, end_date)
        
        if not file_list:
            print(f"❌ No files found for {pair}")
            return False
        
        print(f"   Found {len(file_list)} CSV files")
        
        # Create pair directory
        pair_path = self.parquet_base_path / pair
        pair_path.mkdir(parents=True, exist_ok=True)
        
        # Check what data already exists
        existing_years = self.get_existing_years(pair)
        if existing_years:
            print(f"   Existing Parquet years: {existing_years}")
        
        # Process files in chunks
        all_dfs = []
        total_rows = 0
        chunk_count = 0
        
        for i, file_path in enumerate(file_list):
            try:
                # Update progress
                if progress_callback:
                    progress = (i + 1) / len(file_list)
                    progress_callback(progress, f"Processing {file_path.name}")
                
                # Load CSV
                df = pd.read_csv(file_path)
                
                # Parse datetime
                df['datetime'] = pd.to_datetime(df['time'], format='%Y-%m-%dT%H:%M:%S.%fZ', utc=True)
                df.set_index('datetime', inplace=True)
                df.drop('time', axis=1, inplace=True)
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df.dropna(inplace=True)
                
                all_dfs.append(df)
                total_rows += len(df)
                
                # If we've accumulated enough rows, process chunk
                if len(all_dfs) >= 10 or i == len(file_list) - 1:
                    chunk_count += 1
                    print(f"   Processing chunk {chunk_count} ({len(all_dfs)} files)...")
                    
                    # Combine current chunk
                    chunk_df = pd.concat(all_dfs)
                    chunk_df.sort_index(inplace=True)
                    
                    # Convert chunk to Parquet
                    success = self.convert_pair_gpu(chunk_df, pair, mode)
                    
                    if not success:
                        print(f"   Chunk {chunk_count} failed")
                    
                    # Clear chunk and free memory
                    all_dfs = []
                    del chunk_df
                    gc.collect()
                    
            except Exception as e:
                print(f"   Error processing {file_path}: {e}")
                continue
        
        elapsed = time.time() - start_time
        print(f"✅ {'GPU' if self.use_gpu else 'CPU'} conversion complete for {pair}:")
        print(f"   {total_rows:,} rows processed in {elapsed:.1f}s")
        print(f"   Speed: {total_rows/elapsed:.0f} rows/sec")
        
        return True
    
    def update_pair(self, pair: str, 
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     progress_callback: Optional[Callable] = None) -> bool:
        """
        Update existing Parquet files with new/modified data
        
        Args:
            pair: Trading pair
            start_date: Start date for update
            end_date: End date for update
            progress_callback: Progress callback
        
        Returns:
            True if successful
        """
        print(f"\n🔄 Updating {pair} Parquet files...")
        
        # For update mode, we want to replace data for specified dates
        # but keep existing data for other dates
        return self.convert_pair(
            pair=pair,
            mode='overwrite',  # Overwrite the specified date range
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback
        )
    
    def convert_all_pairs(self, pairs: Optional[List[str]] = None,
                           mode: str = 'append',
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None,
                           progress_callback: Optional[Callable] = None) -> Dict[str, bool]:
        """
        Convert multiple pairs to Parquet
        
        Args:
            pairs: List of pairs to convert (None = all available)
            mode: 'initial', 'append', or 'overwrite'
            start_date: Optional start date filter
            end_date: Optional end date filter
            progress_callback: Progress callback
        
        Returns:
            Dictionary of {pair: success_status}
        """
        if pairs is None:
            pairs = self.loader.get_available_pairs()
        
        results = {}
        total_start = time.time()
        
        print(f"\n{'='*60}")
        print(f"🔄 {'GPU' if self.use_gpu else 'CPU'} Converting {len(pairs)} pairs to Parquet (mode: {mode})")
        print(f"{'='*60}")
        
        for i, pair in enumerate(pairs):
            print(f"\n[{i+1}/{len(pairs)}] ", end="")
            
            # Update progress for overall
            if progress_callback:
                progress = (i + 1) / len(pairs)
                progress_callback(progress, f"Converting {pair}")
            
            success = self.convert_pair(
                pair=pair,
                mode=mode,
                start_date=start_date,
                end_date=end_date,
                progress_callback=lambda p, msg: None  # Inner progress handled separately
            )
            results[pair] = success
        
        total_elapsed = time.time() - total_start
        print(f"\n{'='*60}")
        print(f"✅ Conversion complete! {sum(results.values())}/{len(pairs)} pairs successful")
        print(f"   Total time: {total_elapsed:.1f}s")
        print(f"   Mode used: {mode}")
        print(f"   Processing: {'GPU' if self.use_gpu else 'CPU'}")
        
        return results
    
    def get_pair_info(self, pair: str) -> Dict:
        """Get detailed information about converted Parquet files"""
        pair_dir = self.parquet_base_path / pair
        
        if not pair_dir.exists():
            return {'exists': False}
        
        # Get partition information
        years = []
        months = []
        days = []
        total_size = 0
        file_count = 0
        
        for year_dir in pair_dir.glob("year=*"):
            try:
                year = int(year_dir.name.split('=')[1])
                years.append(year)
                
                for month_dir in year_dir.glob("month=*"):
                    try:
                        month = int(month_dir.name.split('=')[1])
                        months.append((year, month))
                        
                        for day_file in month_dir.glob("*.parquet"):
                            file_count += 1
                            total_size += day_file.stat().st_size
                            
                            # Try to extract day
                            if 'day=' in day_file.name:
                                day = int(day_file.name.split('=')[1].split('.')[0])
                                days.append((year, month, day))
                    except Exception as e:
                        logger.debug(f"Error in month dir: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error in year dir: {e}")
                continue
        
        # Get date range
        min_date = None
        max_date = None
        if days:
            try:
                # Sort days to get min/max
                sorted_days = sorted(days)
                min_date = datetime(sorted_days[0][0], sorted_days[0][1], sorted_days[0][2])
                max_date = datetime(sorted_days[-1][0], sorted_days[-1][1], sorted_days[-1][2])
            except:
                pass
        
        return {
            'exists': True,
            'files': file_count,
            'size_gb': total_size / (1024**3),
            'years': sorted(years),
            'months': sorted(months),
            'days': sorted(days),
            'min_date': min_date,
            'max_date': max_date,
            'partitions': {
                'year': len(years),
                'month': len(months),
                'day': len(days)
            }
        }
    
    def get_update_needed_dates(self, pair: str, 
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> List[str]:
        """
        Determine which dates need to be updated
        
        Args:
            pair: Trading pair
            start_date: Start date
            end_date: End date
        
        Returns:
            List of date strings that need updating
        """
        # Get existing partitions
        existing = self.get_existing_partitions(pair)
        
        # Get all CSV files
        csv_files = self.loader.get_file_list(pair, start_date, end_date)
        
        need_update = []
        for file_path in csv_files:
            try:
                date_str = file_path.stem
                date = datetime.strptime(date_str, '%Y-%m-%d')
                key = (date.year, date.month, date.day)
                
                if key not in existing:
                    need_update.append(date_str)
            except:
                continue
        
        return need_update