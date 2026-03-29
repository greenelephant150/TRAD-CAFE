#!/usr/bin/env python3
"""
Diagnostic script to check parquet file structure
"""

import os
import pandas as pd

PARQUET_BASE_PATH = "/home/grct/Forex_Parquet"

def check_parquet_structure(pair="AUD_CAD"):
    """Check what columns are in the parquet files"""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    
    if not os.path.exists(pair_path):
        print(f"❌ Path not found: {pair_path}")
        return
    
    print(f"\n📁 Checking {pair} at: {pair_path}")
    print("=" * 60)
    
    # Find first parquet file
    all_files = []
    for root, dirs, files in os.walk(pair_path):
        for file in files:
            if file.endswith('.parquet'):
                all_files.append(os.path.join(root, file))
                if len(all_files) >= 1:
                    break
        if all_files:
            break
    
    if not all_files:
        print("❌ No parquet files found")
        return
    
    print(f"\n📄 Sample file: {os.path.basename(all_files[0])}")
    print(f"📂 Path: {os.path.dirname(all_files[0])}")
    
    # Read the file
    try:
        df = pd.read_parquet(all_files[0])
        print(f"\n✅ Loaded {len(df)} rows")
        print(f"\n📊 Columns in parquet file:")
        for i, col in enumerate(df.columns, 1):
            print(f"   {i:2d}. {col} ({df[col].dtype})")
        
        print(f"\n📋 First 3 rows:")
        print(df.head(3))
        
        # Check if we have year/month/day columns
        if 'year' in df.columns and 'month' in df.columns and 'day' in df.columns:
            print(f"\n✅ Found year/month/day columns")
            print(f"   Sample date: {df['year'].iloc[0]}-{df['month'].iloc[0]:02d}-{df['day'].iloc[0]:02d}")
        
        # Check for timestamp columns
        ts_cols = [col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()]
        if ts_cols:
            print(f"\n✅ Found timestamp-like columns: {ts_cols}")
        
    except Exception as e:
        print(f"❌ Error reading file: {e}")

if __name__ == "__main__":
    check_parquet_structure("AUD_CAD")
    print("\n" + "=" * 60)
    check_parquet_structure("EUR_USD")