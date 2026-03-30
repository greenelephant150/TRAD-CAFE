#!/usr/bin/env python3
"""
Training Pipeline Module for SID Method - AUGMENTED VERSION
=============================================================================
Orchestrates the complete training workflow incorporating ALL THREE WAVES:

WAVE 1 (Core Quick Win):
- Data loading from parquet files
- Feature calculation with RSI and MACD
- Target generation (RSI 50, direction)
- Model training and evaluation

WAVE 2 (Live Sessions & Q&A):
- Market context integration
- Divergence and pattern features
- Confidence scoring pipeline
- Multi-timeframe feature aggregation
- Cross-validation with time series split

WAVE 3 (Academy Support Sessions):
- Session-based feature engineering
- Zone quality assessment
- Precision filtering
- Model quality thresholds
- Early stopping and model selection

Author: Sid Naiman / Trading Cafe
Version: 3.0 (Fully Augmented)
=============================================================================
"""

import os
import sys
import json
import pickle
import argparse
import logging
import warnings
import time
import gc
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# Try to import tqdm
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    class tqdm:
        def __init__(self, iterable=None, desc="", **kwargs):
            self.iterable = iterable or []
            self.desc = desc
        def __iter__(self): 
            return iter(self.iterable)
        def update(self, n=1): pass
        def close(self): pass
        def set_description(self, desc): self.desc = desc

# Import local modules
try:
    from sid_method import SidMethod, MarketTrend, SignalQuality
    SID_AVAILABLE = True
except ImportError:
    SID_AVAILABLE = False
    print("⚠️ sid_method.py not available")

try:
    from ai.feature_engineering import FeatureEngineer, FeatureEngineeringConfig
    FEATURE_ENGINEERING_AVAILABLE = True
except ImportError:
    FEATURE_ENGINEERING_AVAILABLE = False
    print("⚠️ feature_engineering.py not available")

try:
    from ai.model_trainer import ModelTrainer, ModelTrainingConfig, ModelMetrics
    MODEL_TRAINER_AVAILABLE = True
except ImportError:
    MODEL_TRAINER_AVAILABLE = False
    print("⚠️ model_trainer.py not available")

try:
    from ai.signal_predictor import SignalPredictor, SignalPredictorConfig
    SIGNAL_PREDICTOR_AVAILABLE = True
except ImportError:
    SIGNAL_PREDICTOR_AVAILABLE = False
    print("⚠️ signal_predictor.py not available")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TrainingPipelineConfig:
    """Complete pipeline configuration (Wave 1, 2, 3)"""
    # Data paths
    parquet_base_path: str = "/home/grct/Forex_Parquet"
    model_output_path: str = "/mnt2/Trading-Cafe/ML/SNaiman2/ai/trained_models/"
    
    # Wave 1: Core parameters
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    rsi_target: int = 50
    prefer_macd_cross: bool = True
    earnings_buffer_days: int = 14
    
    # Wave 2: Feature parameters
    use_divergence: bool = True
    use_pattern_confirmation: bool = True
    use_market_context: bool = True
    divergence_lookback: int = 20
    pattern_lookback: int = 30
    
    # Wave 2: Target parameters
    target_horizons: List[int] = field(default_factory=lambda: [5, 10, 20])
    target_types: List[str] = field(default_factory=lambda: ['direction', 'rsi_50', 'sma_50'])
    
    # Wave 2: Training parameters
    test_size: float = 0.2
    validation_size: float = 0.1
    use_gpu: bool = True
    models_to_train: List[str] = field(default_factory=lambda: [
        'random_forest', 'xgboost', 'lightgbm'
    ])
    
    # Wave 3: Quality thresholds
    min_f1_score: float = 0.5
    min_accuracy: float = 0.55
    min_precision: float = 0.5
    min_recall: float = 0.4
    
    # Wave 3: Session parameters
    use_session_features: bool = True
    session_scores: Dict[str, float] = field(default_factory=lambda: {
        'overlap': 1.0, 'us': 0.9, 'london': 0.7, 'asian': 0.4
    })
    
    # Wave 3: Precision parameters
    strict_rsi: bool = True
    stop_pips_default: int = 5
    stop_pips_yen: int = 10
    min_pattern_candles: int = 7
    
    # Pipeline control
    max_samples_per_pair: int = 500000
    batch_size: int = 100000
    save_intermediate: bool = True
    overwrite_existing: bool = False


@dataclass
class TrainingResult:
    """Container for training results (Wave 2 & 3)"""
    instrument: str
    target: str
    horizon: int
    best_model: str
    metrics: Dict
    training_time: float
    model_path: str
    feature_importance: Optional[Dict] = None
    config_used: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            'instrument': self.instrument,
            'target': self.target,
            'horizon': self.horizon,
            'best_model': self.best_model,
            'metrics': self.metrics,
            'training_time': self.training_time,
            'model_path': self.model_path,
            'feature_importance': self.feature_importance,
            'config_used': self.config_used
        }


# ============================================================================
# COLORFUL OUTPUT
# ============================================================================

class Colors:
    HEADER = '\033[95m'; BLUE = '\033[94m'; CYAN = '\033[96m'
    GREEN = '\033[92m'; YELLOW = '\033[93m'; RED = '\033[91m'
    MAGENTA = '\033[35m'; BOLD = '\033[1m'; END = '\033[0m'
    CHECK = "✅"; CROSS = "❌"; WARN = "⚠️"; FIRE = "🔥"
    BRAIN = "🧠"; DISK = "💾"; CLOCK = "⏱️"; GRAPH = "📈"
    MODEL = "🤖"; DATA = "📊"; GPU_ICON = "🎮"

def print_header(text): 
    print(f"\n{Colors.MAGENTA}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.MAGENTA}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text): 
    print(f"{Colors.GREEN}{Colors.BOLD}✅ {text}{Colors.END}")

def print_warning(text): 
    print(f"{Colors.YELLOW}{Colors.BOLD}⚠️ {text}{Colors.END}")

def print_error(text): 
    print(f"{Colors.RED}{Colors.BOLD}❌ {text}{Colors.END}")

def print_info(text): 
    print(f"{Colors.BLUE}ℹ️ {text}{Colors.END}")

def print_gpu(text): 
    print(f"{Colors.CYAN}{Colors.BOLD}{Colors.GPU_ICON} {text}{Colors.END}")


# ============================================================================
# DATA LOADING FUNCTIONS (Wave 1)
# ============================================================================

def get_available_pairs(parquet_base_path: str) -> List[str]:
    """Get list of all trading pairs with parquet data"""
    pairs = []
    if os.path.exists(parquet_base_path):
        for item in os.listdir(parquet_base_path):
            pair_path = os.path.join(parquet_base_path, item)
            if os.path.isdir(pair_path):
                # Check if there are any parquet files
                for root, dirs, files in os.walk(pair_path):
                    if any(f.endswith('.parquet') for f in files):
                        pairs.append(item)
                        break
    return sorted(pairs)


def get_pair_date_range(parquet_base_path: str, pair: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Get the earliest and latest date available in parquet for a pair"""
    pair_path = os.path.join(parquet_base_path, pair)
    if not os.path.exists(pair_path):
        return None, None

    earliest_date = None
    latest_date = None

    for root, dirs, files in os.walk(pair_path):
        if '__HIVE_DEFAULT_PARTITION__' in root:
            continue

        for file in files:
            if file.endswith('.parquet'):
                # Extract year/month/day from path
                year = None
                month = None
                day = None
                path_parts = root.split(os.sep)
                for part in path_parts:
                    if part.startswith('year='):
                        try:
                            val = part.split('=')[1]
                            if val and not val.startswith('__HIVE'):
                                year = int(val)
                        except:
                            pass
                    elif part.startswith('month='):
                        try:
                            val = part.split('=')[1]
                            if val and not val.startswith('__HIVE'):
                                month = int(val)
                        except:
                            pass
                    elif part.startswith('day='):
                        try:
                            val = part.split('=')[1]
                            if val and not val.startswith('__HIVE'):
                                day = int(val)
                        except:
                            pass

                if year and month and day:
                    try:
                        current_date = datetime(year, month, day)
                        if earliest_date is None or current_date < earliest_date:
                            earliest_date = current_date
                        if latest_date is None or current_date > latest_date:
                            latest_date = current_date
                    except:
                        pass

    return earliest_date, latest_date


def load_parquet_data(parquet_base_path: str, pair: str, 
                       start_date: Optional[datetime] = None,
                       end_date: Optional[datetime] = None,
                       max_rows: int = None,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Load parquet data for a pair with optional date filtering
    
    Args:
        parquet_base_path: Base path to parquet files
        pair: Trading pair
        start_date: Optional start date filter
        end_date: Optional end date filter
        max_rows: Maximum rows to load
        verbose: Enable verbose output
    
    Returns:
        DataFrame with OHLCV data
    """
    pair_path = os.path.join(parquet_base_path, pair)
    if not os.path.exists(pair_path):
        if verbose:
            print_error(f"Pair path not found: {pair_path}")
        return pd.DataFrame()

    try:
        if verbose:
            print_info(f"Loading data for {pair}...")

        # Collect all parquet files
        all_files = []
        for root, dirs, files in os.walk(pair_path):
            if '__HIVE_DEFAULT_PARTITION__' in root:
                continue
            for file in files:
                if file.endswith('.parquet'):
                    # Extract date from path for filtering
                    file_date = None
                    path_parts = root.split(os.sep)
                    for part in path_parts:
                        if part.startswith('year='):
                            try:
                                year = int(part.split('=')[1])
                                month = int([p for p in path_parts if p.startswith('month=')][0].split('=')[1])
                                day = int([p for p in path_parts if p.startswith('day=')][0].split('=')[1])
                                file_date = datetime(year, month, day)
                            except:
                                pass
                    
                    # Apply date filter
                    if start_date and file_date and file_date < start_date:
                        continue
                    if end_date and file_date and file_date > end_date:
                        continue
                    
                    all_files.append(os.path.join(root, file))

        if not all_files:
            if verbose:
                print_warning(f"No files found for {pair}")
            return pd.DataFrame()

        if verbose:
            print_info(f"Found {len(all_files)} parquet files")

        # Load files
        dfs = []
        total_rows = 0
        
        iterator = tqdm(all_files, desc="  Loading files", unit="file") if TQDM_AVAILABLE and len(all_files) > 10 else all_files

        for file_path in iterator:
            try:
                df_chunk = pd.read_parquet(file_path)
                if not df_chunk.empty:
                    dfs.append(df_chunk)
                    total_rows += len(df_chunk)
                    
                    if max_rows and total_rows >= max_rows:
                        break
            except Exception as e:
                if verbose:
                    print_warning(f"Error reading {os.path.basename(file_path)}: {e}")
                continue

        if not dfs:
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(dfs, ignore_index=False)

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index, errors='coerce')
                df = df.dropna()
            except:
                if verbose:
                    print_error(f"Cannot convert index to datetime for {pair}")
                return pd.DataFrame()

        # Sort by date
        df = df.sort_index()

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        for col in required_cols:
            if col not in df.columns:
                if verbose:
                    print_error(f"Missing required column: {col}")
                return pd.DataFrame()

        # Convert to numeric
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Drop rows with NaN values
        df = df.dropna(subset=required_cols)

        # Resample to 1-hour bars for consistency
        if len(df) > 0:
            df = df.resample('1H').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()

        if verbose:
            print_success(f"Loaded {len(df):,} rows for {pair}")
            if len(df) > 0:
                print_info(f"  Date range: {df.index.min()} to {df.index.max()}")

        return df

    except Exception as e:
        if verbose:
            print_error(f"Error loading {pair}: {e}")
        return pd.DataFrame()


# ============================================================================
# TRAINING PIPELINE (Wave 1, 2, 3)
# ============================================================================

class TrainingPipeline:
    """
    Complete training pipeline orchestrating all SID Method components
    Incorporates ALL THREE WAVES of strategies
    """
    
    def __init__(self, config: TrainingPipelineConfig = None, verbose: bool = True):
        """
        Initialize the training pipeline
        
        Args:
            config: TrainingPipelineConfig instance
            verbose: Enable verbose output
        """
        self.config = config or TrainingPipelineConfig()
        self.verbose = verbose
        
        # Create output directories
        os.makedirs(self.config.model_output_path, exist_ok=True)
        
        # Initialize components
        self.feature_engineer = None
        self.model_trainer = None
        self.signal_predictor = None
        
        # Initialize Sid Method
        self.sid = None
        if SID_AVAILABLE:
            self.sid = SidMethod(
                account_balance=10000,
                verbose=False,
                prefer_macd_cross=self.config.prefer_macd_cross,
                use_pattern_confirmation=self.config.use_pattern_confirmation,
                use_divergence=self.config.use_divergence,
                use_market_context=self.config.use_market_context
            )
        
        # Initialize feature engineer
        if FEATURE_ENGINEERING_AVAILABLE:
            feature_config = FeatureEngineeringConfig(
                rsi_oversold=self.config.rsi_oversold,
                rsi_overbought=self.config.rsi_overbought,
                rsi_target=self.config.rsi_target,
                use_session_features=self.config.use_session_features,
                strict_rsi=self.config.strict_rsi,
                min_pattern_candles=self.config.min_pattern_candles
            )
            self.feature_engineer = FeatureEngineer(feature_config, verbose=False)
        
        # Initialize model trainer
        if MODEL_TRAINER_AVAILABLE:
            train_config = ModelTrainingConfig(
                rsi_oversold=self.config.rsi_oversold,
                rsi_overbought=self.config.rsi_overbought,
                rsi_target=self.config.rsi_target,
                prefer_macd_cross=self.config.prefer_macd_cross,
                models_to_train=self.config.models_to_train,
                test_size=self.config.test_size,
                validation_size=self.config.validation_size
            )
            self.model_trainer = ModelTrainer(
                config=train_config,
                model_dir=self.config.model_output_path,
                use_gpu=self.config.use_gpu,
                verbose=False
            )
        
        # Initialize signal predictor
        if SIGNAL_PREDICTOR_AVAILABLE:
            predictor_config = SignalPredictorConfig(
                rsi_oversold=self.config.rsi_oversold,
                rsi_overbought=self.config.rsi_overbought,
                rsi_target=self.config.rsi_target,
                prefer_macd_cross=self.config.prefer_macd_cross
            )
            self.signal_predictor = SignalPredictor(
                config=predictor_config,
                model_dir=self.config.model_output_path,
                verbose=False
            )
        
        if self.verbose:
            print_header("🤖 SID METHOD TRAINING PIPELINE v3.0")
            print_info(f"Parquet path: {self.config.parquet_base_path}")
            print_info(f"Model output: {self.config.model_output_path}")
            print_info(f"GPU enabled: {self.config.use_gpu}")
            print_info(f"Models to train: {self.config.models_to_train}")
            print_info(f"Target horizons: {self.config.target_horizons}")
            print_info(f"Target types: {self.config.target_types}")
            print_info(f"Min F1 threshold: {self.config.min_f1_score}")
    
    # ========================================================================
    # DATA MANAGEMENT
    # ========================================================================
    
    def get_available_pairs(self) -> List[str]:
        """Get all available trading pairs"""
        return get_available_pairs(self.config.parquet_base_path)
    
    def get_pair_info(self, pair: str) -> Dict:
        """Get information about a pair's data"""
        earliest, latest = get_pair_date_range(self.config.parquet_base_path, pair)
        return {
            'pair': pair,
            'earliest_date': earliest,
            'latest_date': latest,
            'days_available': (latest - earliest).days if earliest and latest else 0
        }
    
    def load_pair_data(self, pair: str) -> pd.DataFrame:
        """Load data for a specific pair"""
        return load_parquet_data(
            self.config.parquet_base_path,
            pair,
            max_rows=self.config.max_samples_per_pair,
            verbose=self.verbose
        )
    
    # ========================================================================
    # FEATURE ENGINEERING (Wave 1, 2, 3)
    # ========================================================================
    
    def engineer_features(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Create features for training"""
        if self.feature_engineer:
            return self.feature_engineer.create_all_features(df)
        elif self.sid:
            # Fallback to basic feature calculation
            df = df.copy()
            df['rsi'] = self.sid.calculate_rsi(df, desc="RSI")
            macd_df = self.sid.calculate_macd(df, desc="MACD")
            df['macd'] = macd_df['macd']
            df['macd_signal'] = macd_df['signal']
            return df
        else:
            return df
    
    # ========================================================================
    # TARGET GENERATION (Wave 1 & 2)
    # ========================================================================
    
    def generate_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate multiple target variables for different horizons"""
        df = df.copy()
        
        # RSI 50 targets
        for horizon in self.config.target_horizons:
            df[f'target_rsi_50_{horizon}'] = 0
            for i in range(len(df) - horizon):
                if any(df['rsi'].iloc[i+1:i+horizon+1] >= self.config.rsi_target):
                    df.loc[df.index[i], f'target_rsi_50_{horizon}'] = 1
        
        # Direction targets
        for horizon in self.config.target_horizons:
            df[f'target_direction_{horizon}'] = (df['close'].shift(-horizon) > df['close']).astype(int)
        
        # 50-SMA targets
        df['sma_50'] = df['close'].rolling(50).mean()
        for horizon in self.config.target_horizons:
            df[f'target_sma_50_{horizon}'] = 0
            for i in range(len(df) - horizon):
                if any(df['high'].iloc[i+1:i+horizon+1] >= df['sma_50'].iloc[i]):
                    df.loc[df.index[i], f'target_sma_50_{horizon}'] = 1
        
        return df
    
    # ========================================================================
    # MODEL TRAINING (Wave 1, 2, 3)
    # ========================================================================
    
    def train_for_instrument(self, pair: str, 
                               target_type: str = 'direction',
                               horizon: int = 5) -> Optional[TrainingResult]:
        """
        Train models for a specific instrument
        
        Args:
            pair: Trading instrument
            target_type: Type of target ('direction', 'rsi_50', 'sma_50')
            horizon: Prediction horizon in bars
        
        Returns:
            TrainingResult or None if failed
        """
        if self.verbose:
            print_info(f"\n{'='*50}")
            print_info(f"Training for {pair} (target={target_type}_{horizon})")
            print_info(f"{'='*50}")
        
        start_time = time.time()
        
        try:
            # Load data
            df = self.load_pair_data(pair)
            if df.empty:
                print_error(f"No data loaded for {pair}")
                return None
            
            # Engineer features
            df_features = self.engineer_features(df, pair)
            if df_features.empty:
                print_error(f"Feature engineering failed for {pair}")
                return None
            
            # Generate targets
            df_features = self.generate_targets(df_features)
            
            # Select target column
            target_col = f'target_{target_type}_{horizon}'
            if target_col not in df_features.columns:
                print_error(f"Target column {target_col} not found")
                return None
            
            # Train model
            if self.model_trainer:
                best_name, best_model, metrics = self.model_trainer.train(
                    df_features, target_col, pair
                )
                
                # Check quality thresholds (Wave 3)
                if metrics.f1 < self.config.min_f1_score:
                    print_warning(f"Model F1={metrics.f1:.4f} below threshold {self.config.min_f1_score}")
                    if not self.config.overwrite_existing:
                        return None
                
                # Save result
                result = TrainingResult(
                    instrument=pair,
                    target=target_type,
                    horizon=horizon,
                    best_model=best_name,
                    metrics=metrics.to_dict(),
                    training_time=time.time() - start_time,
                    model_path=os.path.join(self.config.model_output_path, f"{pair}_{best_name}.pkl"),
                    config_used={
                        'rsi_oversold': self.config.rsi_oversold,
                        'rsi_overbought': self.config.rsi_overbought,
                        'prefer_macd_cross': self.config.prefer_macd_cross,
                        'use_pattern_confirmation': self.config.use_pattern_confirmation,
                        'use_divergence': self.config.use_divergence
                    }
                )
                
                if self.verbose:
                    print_success(f"Training complete for {pair}")
                    print_info(f"  Best model: {best_name}")
                    print_info(f"  F1: {metrics.f1:.4f}, Acc: {metrics.accuracy:.4f}")
                    print_info(f"  Training time: {result.training_time:.1f}s")
                
                return result
            else:
                print_error("Model trainer not available")
                return None
                
        except Exception as e:
            print_error(f"Training failed for {pair}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # ========================================================================
    # BATCH TRAINING
    # ========================================================================
    
    def train_all_instruments(self, pairs: List[str] = None,
                                target_types: List[str] = None,
                                horizons: List[int] = None) -> Dict[str, List[TrainingResult]]:
        """
        Train models for all instruments with multiple targets and horizons
        
        Args:
            pairs: List of instruments (None = all available)
            target_types: List of target types (None = all configured)
            horizons: List of horizons (None = all configured)
        
        Returns:
            Dictionary mapping pair to list of TrainingResult
        """
        if pairs is None:
            pairs = self.get_available_pairs()
        
        if target_types is None:
            target_types = self.config.target_types
        
        if horizons is None:
            horizons = self.config.target_horizons
        
        results = {pair: [] for pair in pairs}
        
        print_header(f"🚀 BATCH TRAINING: {len(pairs)} pairs")
        print_info(f"Target types: {target_types}")
        print_info(f"Horizons: {horizons}")
        
        total_start = time.time()
        total_combinations = len(pairs) * len(target_types) * len(horizons)
        completed = 0
        
        for pair in tqdm(pairs, desc="Processing pairs", disable=not TQDM_AVAILABLE):
            for target_type in target_types:
                for horizon in horizons:
                    result = self.train_for_instrument(pair, target_type, horizon)
                    if result:
                        results[pair].append(result)
                    completed += 1
                    
                    if self.verbose and completed % 10 == 0:
                        print_info(f"Progress: {completed}/{total_combinations} ({100*completed/total_combinations:.1f}%)")
        
        total_elapsed = time.time() - total_start
        
        print_header(f"✅ BATCH TRAINING COMPLETE")
        print_success(f"Total time: {total_elapsed:.1f}s")
        print_success(f"Successful models: {sum(len(v) for v in results.values())}")
        
        # Save summary
        self.save_training_summary(results)
        
        return results
    
    def save_training_summary(self, results: Dict[str, List[TrainingResult]]):
        """Save training summary to JSON"""
        summary_path = os.path.join(self.config.model_output_path, "training_summary.json")
        
        summary = {
            'training_date': datetime.now().isoformat(),
            'config': {
                'rsi_oversold': self.config.rsi_oversold,
                'rsi_overbought': self.config.rsi_overbought,
                'prefer_macd_cross': self.config.prefer_macd_cross,
                'target_horizons': self.config.target_horizons,
                'min_f1_score': self.config.min_f1_score
            },
            'results': {}
        }
        
        for pair, pair_results in results.items():
            summary['results'][pair] = [r.to_dict() for r in pair_results]
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        if self.verbose:
            print_info(f"Training summary saved to {summary_path}")
    
    # ========================================================================
    # MODEL EVALUATION (Wave 2 & 3)
    # ========================================================================
    
    def evaluate_model(self, model_path: str, test_df: pd.DataFrame) -> Dict:
        """Evaluate a trained model on test data"""
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Prepare features
            df_features = self.engineer_features(test_df, "test")
            
            # Get features (exclude targets)
            exclude_cols = [col for col in df_features.columns if col.startswith('target_')]
            feature_cols = [col for col in df_features.columns 
                           if col not in exclude_cols 
                           and df_features[col].dtype in ['float64', 'float32', 'int64']]
            
            X_test = df_features[feature_cols].values
            
            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                if len(y_pred_proba.shape) > 1:
                    y_pred_proba = y_pred_proba[:, 1]
                y_pred = (y_pred_proba >= 0.5).astype(int)
            else:
                y_pred = model.predict(X_test)
            
            # Get actual targets
            target_col = [c for c in df_features.columns if c.startswith('target_')][0]
            y_true = df_features[target_col].values
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0))
            }
            
            return metrics
            
        except Exception as e:
            print_error(f"Evaluation failed: {e}")
            return {}
    
    # ========================================================================
    # CROSS-VALIDATION (Wave 2)
    # ========================================================================
    
    def cross_validate(self, pair: str, target_type: str = 'direction',
                        horizon: int = 5, folds: int = 5) -> Dict:
        """
        Perform time series cross-validation
        
        Returns:
            Dictionary with cross-validation results
        """
        if self.verbose:
            print_info(f"Cross-validating {pair} (target={target_type}_{horizon})")
        
        try:
            # Load data
            df = self.load_pair_data(pair)
            if df.empty:
                return {'error': 'No data loaded'}
            
            # Engineer features
            df_features = self.engineer_features(df, pair)
            df_features = self.generate_targets(df_features)
            
            target_col = f'target_{target_type}_{horizon}'
            if target_col not in df_features.columns:
                return {'error': f'Target {target_col} not found'}
            
            # Prepare features
            exclude_cols = [col for col in df_features.columns if col.startswith('target_')]
            feature_cols = [col for col in df_features.columns 
                           if col not in exclude_cols 
                           and df_features[col].dtype in ['float64', 'float32', 'int64']]
            
            X = df_features[feature_cols].values
            y = df_features[target_col].values
            
            # Time series cross-validation
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=folds)
            
            cv_scores = {'accuracy': [], 'f1': [], 'precision': [], 'recall': []}
            
            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train = X[train_idx]
                y_train = y[train_idx]
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                # Train simple model for CV
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_val)
                cv_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                cv_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                cv_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                cv_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
            
            results = {}
            for metric, scores in cv_scores.items():
                if scores:
                    results[metric] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores))
                    }
            
            if self.verbose:
                print_success(f"CV Results - F1: {results.get('f1', {}).get('mean', 0):.4f} (±{results.get('f1', {}).get('std', 0):.4f})")
            
            return results
            
        except Exception as e:
            print_error(f"Cross-validation failed: {e}")
            return {'error': str(e)}


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='SID Method Training Pipeline')
    parser.add_argument('--pair', type=str, help='Specific pair to train')
    parser.add_argument('--target', type=str, default='direction', 
                        choices=['direction', 'rsi_50', 'sma_50'],
                        help='Target type')
    parser.add_argument('--horizon', type=int, default=5, help='Prediction horizon')
    parser.add_argument('--list-pairs', action='store_true', help='List available pairs')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--folds', type=int, default=5, help='Number of CV folds')
    
    args = parser.parse_args()
    
    config = TrainingPipelineConfig()
    pipeline = TrainingPipeline(config, verbose=True)
    
    if args.list_pairs:
        pairs = pipeline.get_available_pairs()
        print(f"\n📋 Available pairs ({len(pairs)}):")
        for pair in pairs:
            info = pipeline.get_pair_info(pair)
            if info['earliest_date']:
                print(f"  {pair}: {info['earliest_date'].strftime('%Y-%m-%d')} to {info['latest_date'].strftime('%Y-%m-%d')}")
            else:
                print(f"  {pair}: No data")
        return
    
    if args.cv:
        if not args.pair:
            print_error("Please specify --pair for cross-validation")
            return
        results = pipeline.cross_validate(args.pair, args.target, args.horizon, args.folds)
        print(f"\n📊 Cross-validation results for {args.pair}:")
        for metric, values in results.items():
            if isinstance(values, dict):
                print(f"  {metric}: {values.get('mean', 0):.4f} (±{values.get('std', 0):.4f})")
        return
    
    if args.pair:
        result = pipeline.train_for_instrument(args.pair, args.target, args.horizon)
        if result:
            print(f"\n✅ Training complete for {args.pair}")
            print(f"  Best model: {result.best_model}")
            print(f"  F1: {result.metrics.get('f1', 0):.4f}")
            print(f"  Model saved: {result.model_path}")
    else:
        # Train all pairs
        pipeline.train_all_instruments()


if __name__ == "__main__":
    main()