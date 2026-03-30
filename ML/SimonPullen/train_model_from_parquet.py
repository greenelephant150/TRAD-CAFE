#!/usr/bin/env python3
"""
Standalone script to train ML models from parquet data with GPU acceleration.
Creates models for each trading pair and saves them with timestamp.

Usage: python train_model_from_parquet.py [--pair PAIR] [--all] [--gpu] [--force]
"""

import os
import sys
import glob
import pickle
import argparse
import logging
import warnings
import gc
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Suppress warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================
PARQUET_BASE_PATH = "/home/grct/Forex_Parquet"
MODEL_OUTPUT_PATH = "/mnt2/Trading-Cafe/ML/SPullen/ai/trained_models"
CACHE_PATH = "/tmp/parquet_cache"

# Create directories if they don't exist
os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(CACHE_PATH, exist_ok=True)

# ============================================================================
# GPU DETECTION AND SETUP
# ============================================================================
def setup_gpu():
    """Setup GPU acceleration with fallback to CPU."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"✅ GPU detected: {device_name} ({device_count} devices)")
            logger.info(f"   CUDA Version: {torch.version.cuda}")
            return {
                'available': True,
                'device': 'cuda',
                'device_count': device_count,
                'name': device_name,
                'torch': torch,
                'nn': nn,
                'optim': optim,
                'DataLoader': DataLoader,
                'TensorDataset': TensorDataset
            }
        else:
            logger.warning("⚠️ No GPU detected, using CPU mode")
            return {
                'available': False,
                'device': 'cpu',
                'torch': torch,
                'nn': nn,
                'optim': optim,
                'DataLoader': DataLoader,
                'TensorDataset': TensorDataset
            }
    except ImportError:
        logger.warning("⚠️ PyTorch not installed, falling back to scikit-learn")
        return {
            'available': False,
            'device': 'cpu',
            'torch': None,
            'framework': 'sklearn'
        }

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================
def get_available_pairs() -> List[str]:
    """Get list of all trading pairs with parquet data."""
    pairs = []
    if os.path.exists(PARQUET_BASE_PATH):
        for item in os.listdir(PARQUET_BASE_PATH):
            pair_path = os.path.join(PARQUET_BASE_PATH, item)
            if os.path.isdir(pair_path):
                # Check if there's at least one year directory
                for subdir in os.listdir(pair_path):
                    if subdir.startswith('year='):
                        pairs.append(item)
                        break
    return sorted(pairs)


def get_latest_parquet_date(pair: str) -> Optional[datetime]:
    """Get the most recent date available in parquet for a pair."""
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    if not os.path.exists(pair_path):
        return None
    
    latest_date = None
    
    # Find all year directories
    for year_dir in os.listdir(pair_path):
        if not year_dir.startswith('year='):
            continue
        
        year_path = os.path.join(pair_path, year_dir)
        year = int(year_dir.replace('year=', ''))
        
        # Check month directories
        for month_dir in os.listdir(year_path):
            if not month_dir.startswith('month='):
                continue
            
            month_path = os.path.join(year_path, month_dir)
            month = int(month_dir.replace('month=', ''))
            
            # Check day directories
            for day_dir in os.listdir(month_path):
                if not day_dir.startswith('day='):
                    continue
                
                day = int(day_dir.replace('day=', ''))
                current_date = datetime(year, month, day)
                
                if latest_date is None or current_date > latest_date:
                    latest_date = current_date
    
    return latest_date


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to standard format."""
    column_mapping = {}
    
    # Map for date/time columns
    date_variants = ['Date', 'date', 'timestamp', 'time', 'datetime', 'ds', 'Time']
    for variant in date_variants:
        if variant in df.columns:
            column_mapping[variant] = 'timestamp'
            break
    
    # Map for price columns
    price_mapping = {
        'open': ['open', 'Open', 'OPEN', 'opening'],
        'high': ['high', 'High', 'HIGH'],
        'low': ['low', 'Low', 'LOW'],
        'close': ['close', 'Close', 'CLOSE', 'closing', 'price'],
        'volume': ['volume', 'Volume', 'VOLUME', 'vol', 'tickvolume']
    }
    
    for target, variants in price_mapping.items():
        for variant in variants:
            if variant in df.columns:
                column_mapping[variant] = target
                break
    
    if column_mapping:
        df = df.rename(columns=column_mapping)
    
    return df


def load_parquet_data(pair: str, start_date: Optional[str] = None, 
                      end_date: Optional[str] = None, 
                      max_files: Optional[int] = None) -> pd.DataFrame:
    """
    Load parquet data for a specific pair with optional date filtering.
    """
    data_frames = []
    pair_path = os.path.join(PARQUET_BASE_PATH, pair)
    
    if not os.path.exists(pair_path):
        logger.error(f"Pair path not found: {pair_path}")
        return pd.DataFrame()
    
    # Parse date filters
    start_dt = pd.to_datetime(start_date) if start_date else None
    end_dt = pd.to_datetime(end_date) if end_date else None
    
    # Collect all parquet files
    all_files = []
    for root, dirs, files in os.walk(pair_path):
        for file in files:
            if file.endswith('.parquet'):
                all_files.append(os.path.join(root, file))
    
    all_files.sort()  # Ensure chronological order
    total_files = len(all_files)
    
    if total_files == 0:
        logger.warning(f"No parquet files found for {pair}")
        return pd.DataFrame()
    
    logger.info(f"Found {total_files} parquet files for {pair}")
    
    # Apply file limit if specified
    if max_files and len(all_files) > max_files:
        all_files = all_files[-max_files:]  # Take most recent files
        logger.info(f"Limited to {max_files} most recent files")
    
    # Load files with progress tracking
    loaded_count = 0
    for i, file_path in enumerate(all_files):
        try:
            # Extract date from path for filtering
            path_parts = Path(file_path).parts
            year = month = day = None
            
            for part in path_parts:
                if part.startswith('year='):
                    year = int(part.replace('year=', ''))
                elif part.startswith('month='):
                    month = int(part.replace('month=', ''))
                elif part.startswith('day='):
                    day = int(part.replace('day=', ''))
            
            # Apply date filters
            if start_dt and year is not None:
                if year < start_dt.year:
                    continue
                if year == start_dt.year and month is not None and month < start_dt.month:
                    continue
                if year == start_dt.year and month == start_dt.month and day is not None and day < start_dt.day:
                    continue
            
            if end_dt and year is not None:
                if year > end_dt.year:
                    continue
                if year == end_dt.year and month is not None and month > end_dt.month:
                    continue
                if year == end_dt.year and month == end_dt.month and day is not None and day > end_dt.day:
                    continue
            
            # Read parquet file
            df_chunk = pd.read_parquet(file_path)
            
            if not df_chunk.empty:
                df_chunk = normalize_columns(df_chunk)
                data_frames.append(df_chunk)
                loaded_count += 1
            
            # Log progress every 100 files
            if (i + 1) % 100 == 0:
                logger.info(f"  Progress: {i + 1}/{total_files} files processed")
                
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {str(e)[:100]}")
            continue
    
    if not data_frames:
        logger.warning(f"No data loaded for {pair}")
        return pd.DataFrame()
    
    logger.info(f"Successfully loaded {loaded_count} files for {pair}")
    
    # Combine all data
    df = pd.concat(data_frames, ignore_index=True)
    
    # Ensure timestamp column exists and is datetime
    if 'timestamp' not in df.columns:
        logger.error(f"No timestamp column found in {pair} data")
        return pd.DataFrame()
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp'])
    
    logger.info(f"Total data points: {len(df)}")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create technical indicators and features from raw price data."""
    df = df.copy()
    
    # Ensure we have required columns
    required_cols = ['open', 'high', 'low', 'close']
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing required column: {col}")
            return pd.DataFrame()
    
    # Price-based features
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Price ranges
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
    
    # Moving averages
    for period in [5, 10, 20, 50, 200]:
        df[f'ma_{period}'] = df['close'].rolling(window=period).mean()
        df[f'ma_ratio_{period}'] = df['close'] / df[f'ma_{period}']
    
    # Volatility
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
    
    # RSI
    for period in [14, 28]:
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, 0.001)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['close'].ewm(span=12, adjust=False).mean()
    exp2 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = exp1 - exp2
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # Bollinger Bands
    for period in [20]:
        ma = df['close'].rolling(window=period).mean()
        std = df['close'].rolling(window=period).std()
        df[f'bb_upper_{period}'] = ma + (std * 2)
        df[f'bb_lower_{period}'] = ma - (std * 2)
        df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / ma
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = true_range.rolling(window=14).mean()
    
    # Target variable (next period direction)
    df['target_direction'] = (df['close'].shift(-1) > df['close']).astype(int)
    df['target_return'] = df['close'].pct_change().shift(-1)
    
    return df


def prepare_training_data(df: pd.DataFrame, lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for training by creating feature matrix and target vector.
    Uses lookback window to create sequence features.
    """
    # Remove rows with NaN values
    df = df.dropna()
    
    if len(df) < lookback + 10:
        logger.warning(f"Insufficient data: {len(df)} rows")
        return np.array([]), np.array([])
    
    # Select feature columns (exclude timestamp and original price columns)
    exclude_cols = ['timestamp', 'target_direction', 'target_return', 'open', 'high', 'low', 'close']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    logger.info(f"Using {len(feature_cols)} features")
    
    # Create feature matrix with lookback
    X_list = []
    y_list = []
    
    for i in range(lookback, len(df) - 1):
        # Features from lookback window
        window_features = []
        for j in range(lookback):
            window_features.append(df[feature_cols].iloc[i - j].values)
        
        # Flatten window features
        X_list.append(np.concatenate(window_features))
        y_list.append(df['target_direction'].iloc[i])
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    
    logger.info(f"Training data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class balance: {np.bincount(y)}")
    
    return X, y


# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================
def train_torch_model(X, y, gpu_resources):
    """Train a PyTorch neural network model with GPU acceleration."""
    torch = gpu_resources['torch']
    nn = gpu_resources['nn']
    optim = gpu_resources['optim']
    DataLoader = gpu_resources['DataLoader']
    TensorDataset = gpu_resources['TensorDataset']
    device = gpu_resources['device']
    
    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)
    
    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    
    # Define model architecture
    class TradingModel(nn.Module):
        def __init__(self, input_dim, hidden_dims=[256, 128, 64]):
            super().__init__()
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 2))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    # Initialize model
    model = TradingModel(X.shape[1]).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    epochs = 50
    batch_size = 1024
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 10
    
    logger.info(f"Starting PyTorch training on {device.upper()}...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(loader)
        accuracy = correct / total
        
        # Validation on a subset (use training data as validation for simplicity)
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_tensor[:min(10000, len(X_tensor))].to(device))
            val_loss = criterion(val_outputs, y_tensor[:min(10000, len(y_tensor))].to(device))
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 5 == 0:
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= max_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Clear cache periodically
        if device == 'cuda' and (epoch + 1) % 10 == 0:
            torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor.to(device))
        probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
    accuracy = (predictions == y).mean()
    
    logger.info(f"Final training accuracy: {accuracy:.4f}")
    
    return {
        'model': model,
        'accuracy': accuracy,
        'predictions': predictions,
        'probabilities': probabilities,
        'feature_count': X.shape[1],
        'framework': 'pytorch',
        'device': device
    }


def train_sklearn_model(X, y):
    """Train scikit-learn models as fallback."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    
    logger.info("Training scikit-learn models...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train Random Forest
    logger.info("Training Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_score = rf_model.score(X_test, y_test)
    logger.info(f"Random Forest accuracy: {rf_score:.4f}")
    
    # Train Gradient Boosting
    logger.info("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    gb_score = gb_model.score(X_test, y_test)
    logger.info(f"Gradient Boosting accuracy: {gb_score:.4f}")
    
    # Select best model
    if rf_score >= gb_score:
        best_model = rf_model
        best_score = rf_score
        model_type = 'random_forest'
    else:
        best_model = gb_model
        best_score = gb_score
        model_type = 'gradient_boosting'
    
    logger.info(f"Best model: {model_type} with accuracy {best_score:.4f}")
    
    return {
        'model': best_model,
        'accuracy': best_score,
        'framework': 'sklearn',
        'model_type': model_type,
        'feature_count': X.shape[1]
    }


def train_model(pair: str, gpu_resources: Dict, force: bool = False):
    """Main training function for a single pair."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Training model for {pair}")
    logger.info(f"{'='*60}")
    
    # Check if model already exists
    latest_date = get_latest_parquet_date(pair)
    if not latest_date:
        logger.error(f"No data found for {pair}")
        return None
    
    model_filename = f"{latest_date.strftime('%d%m%Y')}--SimonPullen--{pair}--S5.pkl"
    model_path = os.path.join(MODEL_OUTPUT_PATH, model_filename)
    
    if os.path.exists(model_path) and not force:
        logger.info(f"Model already exists: {model_filename} (use --force to overwrite)")
        return model_path
    
    # Load data (last 2 years for training)
    end_date = latest_date.strftime('%Y-%m-%d')
    start_date = (latest_date - timedelta(days=730)).strftime('%Y-%m-%d')  # 2 years of data
    
    logger.info(f"Loading data from {start_date} to {end_date}")
    df = load_parquet_data(pair, start_date=start_date, end_date=end_date)
    
    if df.empty:
        logger.error(f"No data loaded for {pair}")
        return None
    
    # Create features
    logger.info("Creating technical features...")
    df = create_features(df)
    
    if df.empty:
        logger.error("Feature creation failed")
        return None
    
    # Prepare training data
    logger.info("Preparing training data...")
    X, y = prepare_training_data(df, lookback=20)
    
    if len(X) == 0:
        logger.error("Failed to prepare training data")
        return None
    
    # Train model
    logger.info(f"Training with {len(X)} samples, {X.shape[1]} features")
    
    if gpu_resources.get('torch') and gpu_resources.get('available', False):
        result = train_torch_model(X, y, gpu_resources)
    else:
        result = train_sklearn_model(X, y)
    
    # Prepare model package
    model_package = {
        'pair': pair,
        'training_date': datetime.now(),
        'data_range': {
            'start': df['timestamp'].min(),
            'end': df['timestamp'].max()
        },
        'latest_date': latest_date,
        'features': {
            'count': X.shape[1],
            'lookback': 20,
            'names': ['feature_' + str(i) for i in range(X.shape[1])]  # Generic feature names
        },
        'performance': {
            'accuracy': result['accuracy'],
            'samples': len(X)
        },
        'model': result['model'],
        'framework': result['framework'],
        'metadata': {
            'version': '1.0',
            'type': 'direction_prediction',
            'timeframe': 'S5'
        }
    }
    
    if 'device' in result:
        model_package['device'] = result['device']
    if 'model_type' in result:
        model_package['model_type'] = result['model_type']
    
    # Save model
    logger.info(f"Saving model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump(model_package, f)
    
    # Also save a metadata file for easy reference
    metadata_path = model_path.replace('.pkl', '_metadata.json')
    import json
    
    metadata = {
        'pair': pair,
        'training_date': model_package['training_date'].isoformat(),
        'data_range_start': model_package['data_range']['start'].isoformat(),
        'data_range_end': model_package['data_range']['end'].isoformat(),
        'latest_date': latest_date.isoformat(),
        'accuracy': result['accuracy'],
        'samples': len(X),
        'features': X.shape[1],
        'framework': result['framework'],
        'file': model_filename
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"✅ Model saved successfully with accuracy: {result['accuracy']:.4f}")
    
    # Clean up to free memory
    del df, X, y
    gc.collect()
    if gpu_resources.get('torch') and gpu_resources['device'] == 'cuda':
        gpu_resources['torch'].cuda.empty_cache()
    
    return model_path


def train_all_pairs(gpu_resources: Dict, force: bool = False):
    """Train models for all available pairs."""
    pairs = get_available_pairs()
    
    if not pairs:
        logger.error("No trading pairs found")
        return
    
    logger.info(f"Found {len(pairs)} trading pairs")
    
    results = []
    for i, pair in enumerate(pairs):
        logger.info(f"\nProcessing pair {i+1}/{len(pairs)}: {pair}")
        
        try:
            model_path = train_model(pair, gpu_resources, force)
            if model_path:
                results.append({'pair': pair, 'status': 'success', 'path': model_path})
            else:
                results.append({'pair': pair, 'status': 'failed'})
        except Exception as e:
            logger.error(f"Error training {pair}: {e}")
            results.append({'pair': pair, 'status': 'error', 'error': str(e)})
        
        # Clear memory between pairs
        gc.collect()
        if gpu_resources.get('torch') and gpu_resources['device'] == 'cuda':
            gpu_resources['torch'].cuda.empty_cache()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TRAINING SUMMARY")
    logger.info("="*60)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    logger.info(f"Successful: {success_count}/{len(results)}")
    
    for r in results:
        status_icon = "✅" if r['status'] == 'success' else "❌"
        logger.info(f"{status_icon} {r['pair']}: {r['status']}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train ML models from parquet data')
    parser.add_argument('--pair', type=str, help='Specific trading pair to train')
    parser.add_argument('--all', action='store_true', help='Train all available pairs')
    parser.add_argument('--gpu', action='store_true', help='Force GPU usage')
    parser.add_argument('--force', action='store_true', help='Force retrain even if model exists')
    
    args = parser.parse_args()
    
    if not args.pair and not args.all:
        parser.print_help()
        sys.exit(1)
    
    # Setup GPU resources
    gpu_resources = setup_gpu()
    
    if args.pair:
        train_model(args.pair, gpu_resources, args.force)
    elif args.all:
        train_all_pairs(gpu_resources, args.force)


if __name__ == "__main__":
    main()
