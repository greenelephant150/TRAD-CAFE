#!/usr/bin/env python3
"""
Training Pipeline - Integrates pkltrainer3.py with AI module
Handles scheduled training, model updates, and validation
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import subprocess
import glob

# Add parent directory to path for pkltrainer3 imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.model_manager import ModelManager
from ai.feature_engineering import FeatureEngineer
from ai.signal_predictor import SignalPredictor
from ai.ai_accelerator import AIAccelerator

# Import pkltrainer3 functions
try:
    import pkltrainer3
    PKL_TRAINER_AVAILABLE = True
except ImportError as e:
    PKL_TRAINER_AVAILABLE = False
    print(f"Warning: pkltrainer3 not available: {e}")

logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    End-to-end pipeline for model training and management
    """
    
    def __init__(self, 
                 model_dir: str = "/models/ai/trained_models/",
                 parquet_path: str = "/home/grct/Forex_Parquet",
                 csv_path: str = "/home/grct/Forex"):
        """
        Args:
            model_dir: Directory for trained models
            parquet_path: Path to parquet data
            csv_path: Path to CSV data
        """
        self.model_dir = model_dir
        self.parquet_path = parquet_path
        self.csv_path = csv_path
        
        # Initialize components
        self.accelerator = AIAccelerator()
        self.model_manager = ModelManager(model_dir, parquet_path)
        self.feature_engineer = FeatureEngineer(self.accelerator)
        self.predictor = None  # Lazy initialization
        
        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        
        logger.info(f"Training Pipeline initialized")
        logger.info(f"  Model dir: {model_dir}")
        logger.info(f"  Parquet path: {parquet_path}")
        logger.info(f"  Device: {self.accelerator.device_type}")
        
    def _get_predictor(self):
        """Lazy initialize predictor"""
        if self.predictor is None:
            self.predictor = SignalPredictor(self.accelerator, self.model_dir)
        return self.predictor
        
    def train_pair(self, pair: str, force: bool = False, force_cpu: bool = False, 
                   samples: int = 100000, lookback: int = 20) -> Dict[str, Any]:
        """
        Train a model for a specific pair using pkltrainer3.py
        
        Args:
            pair: Trading pair
            force: Force retraining even if model exists
            force_cpu: Force CPU training
            samples: Number of samples per pair
            lookback: Lookback periods
        
        Returns:
            Training results
        """
        if not PKL_TRAINER_AVAILABLE:
            logger.error("pkltrainer3 not available")
            return {'status': 'error', 'message': 'pkltrainer3 not available'}
        
        logger.info(f"🎯 Training model for {pair} (samples={samples}, lookback={lookback})")
        
        try:
            # Call the original trainer function
            result = pkltrainer3.train_pair(
                pair=pair,
                force_cpu=force_cpu,
                force=force,
                verbose=True
            )
            
            if result and result.get('status') == 'trained':
                logger.info(f"✅ Successfully trained {pair}")
                logger.info(f"   Accuracy: {result.get('accuracy', 0):.3f}")
                logger.info(f"   Device: {result.get('device', 'unknown')}")
                logger.info(f"   Time: {result.get('time', 0):.1f}s")
                
                # Reload models in predictor
                if self.predictor:
                    self.predictor.reload_models()
                
                # Validate the new model
                validation = self.validate_model(pair)
                
                return {
                    'status': 'success',
                    'pair': pair,
                    'accuracy': result.get('accuracy', 0),
                    'samples': result.get('samples', 0),
                    'device': result.get('device', 'unknown'),
                    'training_time': result.get('time', 0),
                    'validation': validation
                }
            elif result and result.get('status') == 'skipped':
                logger.info(f"⏭️ Skipped {pair} (model already exists)")
                return {
                    'status': 'skipped',
                    'pair': pair,
                    'message': 'Model already exists'
                }
            else:
                logger.error(f"❌ Failed to train {pair}")
                return {
                    'status': 'failed',
                    'pair': pair,
                    'message': 'Training failed'
                }
                
        except Exception as e:
            logger.error(f"Error training {pair}: {e}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'error',
                'pair': pair,
                'message': str(e)
            }
    
    def train_all_pairs(self, force: bool = False, force_cpu: bool = False,
                        samples: int = 100000, lookback: int = 20,
                        max_pairs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train models for all available pairs
        
        Args:
            force: Force retraining
            force_cpu: Force CPU training
            samples: Samples per pair
            lookback: Lookback periods
            max_pairs: Maximum number of pairs to train
        
        Returns:
            Training results summary
        """
        if not PKL_TRAINER_AVAILABLE:
            logger.error("pkltrainer3 not available")
            return {'status': 'error', 'message': 'pkltrainer3 not available'}
        
        logger.info(f"🎯 Training models for all pairs")
        
        # Get all pairs from parquet
        pairs = self.feature_engineer.get_parquet_pairs()
        if not pairs:
            pairs = self.feature_engineer.get_available_pairs()
        
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        logger.info(f"Found {len(pairs)} pairs")
        
        # Call pkltrainer3's train_all_pairs
        results = pkltrainer3.train_all_pairs(
            pairs=pairs,
            force_cpu=force_cpu,
            force=force,
            verbose=True
        )
        
        # Reload models in predictor
        if self.predictor:
            self.predictor.reload_models()
        
        # Validate all new models
        validation_results = {}
        for success in results.get('success', []):
            pair = success['pair'] if isinstance(success, dict) else success
            validation = self.validate_model(pair)
            if validation:
                validation_results[pair] = validation
        
        summary = {
            'total': len(pairs),
            'successful': len(results.get('success', [])),
            'skipped': len(results.get('skipped', [])),
            'failed': len(results.get('failed', [])),
            'validation': validation_results
        }
        
        logger.info(f"Training complete: {summary['successful']} successful, "
                   f"{summary['skipped']} skipped, {summary['failed']} failed")
        
        return summary
    
    def validate_model(self, pair: str, days: int = 30) -> Dict[str, Any]:
        """
        Validate a trained model on recent data
        
        Args:
            pair: Trading pair
            days: Number of recent days to validate on
        
        Returns:
            Validation metrics
        """
        logger.info(f"🔍 Validating model for {pair} on last {days} days")
        
        # Get the latest model
        model_package = self.model_manager.get_model_for_pair(pair)
        if not model_package:
            logger.warning(f"No model found for {pair}")
            return {}
        
        # Extract metadata
        if isinstance(model_package, dict):
            metadata = model_package.get('metadata', {})
            features = metadata.get('features', [])
            lookback = metadata.get('model_params', {}).get('lookback', 20)
            model = model_package.get('model')
        else:
            logger.warning(f"Invalid model package for {pair}")
            return {}
        
        if not model or not features:
            logger.warning(f"Model missing required components for {pair}")
            return {}
        
        # Load recent data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        df = self.feature_engineer.load_data(
            pair=pair,
            use_parquet=True,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            target_timeframe='1h'
        )
        
        if df.empty or len(df) < lookback + 10:
            logger.warning(f"Insufficient validation data for {pair}: {len(df)} rows")
            return {}
        
        # Calculate indicators
        df = self.feature_engineer._calculate_all_indicators(df)
        
        # Create validation samples using pkltrainer3's function
        try:
            X_val, y_val = pkltrainer3.create_samples(
                df, features, lookback=lookback, 
                samples_per_pair=1000, verbose=False
            )
        except Exception as e:
            logger.error(f"Error creating samples for {pair}: {e}")
            return {}
        
        if len(X_val) == 0:
            logger.warning(f"No validation samples for {pair}")
            return {}
        
        # Get predictions
        try:
            y_pred = model.predict(X_val)
            
            # Convert to numpy if needed
            if hasattr(y_pred, 'get'):
                y_pred = y_pred.get()
            if hasattr(y_val, 'values'):
                y_val = y_val.values
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            
            accuracy = accuracy_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
            precision = precision_score(y_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_val, y_pred, average='weighted', zero_division=0)
            
            # Calculate directional accuracy
            correct_direction = (y_pred == y_val).sum()
            directional_accuracy = correct_direction / len(y_val)
            
            logger.info(f"✅ Validation for {pair}:")
            logger.info(f"   Accuracy: {accuracy:.3f}")
            logger.info(f"   F1 Score: {f1:.3f}")
            logger.info(f"   Samples: {len(X_val)}")
            
            return {
                'pair': pair,
                'accuracy': float(accuracy),
                'f1_score': float(f1),
                'precision': float(precision),
                'recall': float(recall),
                'directional_accuracy': float(directional_accuracy),
                'samples': len(X_val),
                'validation_date': datetime.now().isoformat(),
                'validation_days': days,
                'model_date': metadata.get('training_date', 'unknown')
            }
            
        except Exception as e:
            logger.error(f"Error during validation for {pair}: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def validate_all_models(self, days: int = 30) -> pd.DataFrame:
        """
        Validate all available models
        
        Args:
            days: Number of recent days to validate on
        
        Returns:
            DataFrame with validation results
        """
        logger.info(f"🔍 Validating all models on last {days} days")
        
        # Get all models summary
        models_df = self.model_manager.get_latest_models_summary()
        
        results = []
        for _, row in models_df.iterrows():
            pair = row['Pair']
            validation = self.validate_model(pair, days)
            if validation:
                results.append(validation)
        
        df = pd.DataFrame(results)
        if not df.empty:
            df = df.sort_values('accuracy', ascending=False)
            logger.info(f"Validation complete for {len(df)} models")
            logger.info(f"Average accuracy: {df['accuracy'].mean():.3f}")
        
        return df
    
    def schedule_training(self, pairs: Optional[List[str]] = None, 
                          days_threshold: int = 7,
                          min_accuracy_threshold: float = 0.55) -> Dict[str, Any]:
        """
        Check and update models that are older than threshold or below accuracy threshold
        
        Args:
            pairs: List of pairs to check (None = all)
            days_threshold: Retrain if older than this many days
            min_accuracy_threshold: Retrain if accuracy below this threshold
        
        Returns:
            Training results for updated models
        """
        logger.info(f"🔄 Running scheduled training check")
        logger.info(f"   Age threshold: {days_threshold} days")
        logger.info(f"   Accuracy threshold: {min_accuracy_threshold}")
        
        # Get all models
        models_df = self.model_manager.get_latest_models_summary()
        
        if models_df.empty:
            logger.info("No existing models found, training all pairs")
            return self.train_all_pairs()
        
        # Identify models that need update
        needs_update = []
        today = datetime.now()
        
        for _, row in models_df.iterrows():
            pair = row['Pair']
            
            if pairs and pair not in pairs:
                continue
            
            # Check age
            age_days = (today - row['To']).days
            age_needs_update = age_days > days_threshold
            
            # Check accuracy
            accuracy = row.get('Accuracy', 0)
            if accuracy != 'N/A' and isinstance(accuracy, (int, float)):
                accuracy_needs_update = accuracy < min_accuracy_threshold
            else:
                accuracy_needs_update = False
            
            if age_needs_update or accuracy_needs_update:
                reasons = []
                if age_needs_update:
                    reasons.append(f"age ({age_days} days)")
                if accuracy_needs_update:
                    reasons.append(f"accuracy ({accuracy:.3f})")
                
                logger.info(f"📅 {pair} needs update: {', '.join(reasons)}")
                needs_update.append(pair)
        
        if not needs_update:
            logger.info("✅ All models are up to date")
            return {'status': 'up_to_date', 'updated': []}
        
        logger.info(f"🔄 Updating {len(needs_update)} models")
        
        # Train models that need update
        results = []
        for pair in needs_update:
            result = self.train_pair(pair, force=True)
            results.append(result)
        
        return {
            'status': 'updated',
            'updated_count': len(needs_update),
            'updated_pairs': needs_update,
            'results': results
        }
    
    def get_training_report(self) -> pd.DataFrame:
        """
        Generate a comprehensive training report
        
        Returns:
            DataFrame with model status
        """
        models_df = self.model_manager.get_latest_models_summary()
        
        if models_df.empty:
            return pd.DataFrame()
        
        # Add age and status
        today = datetime.now()
        models_df['Age Days'] = (today - pd.to_datetime(models_df['To'])).dt.days
        models_df['Status'] = models_df.apply(
            lambda row: 'Current' if row['Age Days'] <= 7 else 'Stale', 
            axis=1
        )
        
        # Add validation results if available
        try:
            validation_results = self.validate_all_models(days=30)
            if not validation_results.empty:
                # Merge validation results
                models_df = models_df.merge(
                    validation_results[['pair', 'accuracy', 'f1_score']], 
                    left_on='Pair', 
                    right_on='pair', 
                    how='left'
                )
                models_df = models_df.drop('pair', axis=1)
                models_df = models_df.rename(columns={'accuracy': 'Validation Accuracy'})
        except:
            pass
        
        # Reorder columns
        column_order = ['Pair', 'From', 'To', 'Days', 'Age Days', 'Status', 
                       'Accuracy', 'Validation Accuracy', 'Samples', 'Device', 'Size (MB)']
        available_cols = [c for c in column_order if c in models_df.columns]
        
        return models_df[available_cols]
    
    def export_model_metrics(self, output_file: str = 'model_metrics.json'):
        """
        Export model metrics to JSON file
        
        Args:
            output_file: Path to output JSON file
        """
        models_df = self.get_training_report()
        
        if models_df.empty:
            logger.warning("No models to export")
            return
        
        # Convert to dict
        metrics = {
            'export_date': datetime.now().isoformat(),
            'total_models': len(models_df),
            'models': models_df.to_dict(orient='records'),
            'summary': {
                'avg_accuracy': float(models_df['Accuracy'].mean()) if 'Accuracy' in models_df else 0,
                'avg_age_days': float(models_df['Age Days'].mean()) if 'Age Days' in models_df else 0,
                'current_models': int((models_df['Age Days'] <= 7).sum()) if 'Age Days' in models_df else 0
            }
        }
        
        import json
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        logger.info(f"Metrics exported to {output_file}")
    
    def cleanup_old_models(self, keep_last: int = 3):
        """
        Delete old models, keeping only the most recent ones
        
        Args:
            keep_last: Number of most recent models to keep per pair
        """
        logger.info(f"🧹 Cleaning up old models (keeping last {keep_last} per pair)")
        
        # Get all models
        models_df = self.model_manager.get_latest_models_summary()
        
        if models_df.empty:
            return
        
        # Group by pair
        for pair in models_df['Pair'].unique():
            pair_models = models_df[models_df['Pair'] == pair].sort_values('To', ascending=False)
            
            if len(pair_models) > keep_last:
                # Delete older models
                to_delete = pair_models.iloc[keep_last:]
                for _, row in to_delete.iterrows():
                    filename = row['Filename']
                    logger.info(f"   Deleting {filename}")
                    self.model_manager.delete_model(filename)
        
        logger.info("Cleanup complete")

def main():
    """Command-line interface for training pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Model Training Pipeline')
    parser.add_argument('--pair', type=str, help='Train specific pair')
    parser.add_argument('--all', action='store_true', help='Train all pairs')
    parser.add_argument('--force', action='store_true', help='Force retraining')
    parser.add_argument('--cpu', action='store_true', help='Force CPU mode')
    parser.add_argument('--schedule', action='store_true', 
                       help='Run scheduled training (check age)')
    parser.add_argument('--report', action='store_true', 
                       help='Show training report')
    parser.add_argument('--validate', type=str, 
                       help='Validate model for specific pair')
    parser.add_argument('--validate-all', action='store_true',
                       help='Validate all models')
    parser.add_argument('--samples', type=int, default=100000,
                       help='Samples per pair (default: 100000)')
    parser.add_argument('--lookback', type=int, default=20,
                       help='Lookback periods (default: 20)')
    parser.add_argument('--days-threshold', type=int, default=7,
                       help='Days threshold for scheduled training')
    parser.add_argument('--accuracy-threshold', type=float, default=0.55,
                       help='Accuracy threshold for scheduled training')
    parser.add_argument('--cleanup', type=int, metavar='KEEP',
                       help='Cleanup old models, keeping K most recent per pair')
    parser.add_argument('--export-metrics', type=str,
                       help='Export model metrics to JSON file')
    
    args = parser.parse_args()
    
    pipeline = TrainingPipeline()
    
    if args.export_metrics:
        pipeline.export_model_metrics(args.export_metrics)
        return
    
    if args.cleanup:
        pipeline.cleanup_old_models(keep_last=args.cleanup)
        return
    
    if args.report:
        print("\n📊 Training Report:")
        df = pipeline.get_training_report()
        if not df.empty:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            print(df.to_string(index=False))
        else:
            print("No models found")
        return
    
    if args.validate:
        metrics = pipeline.validate_model(args.validate)
        if metrics:
            print(f"\n🔍 Validation Results for {args.validate}:")
            for key, value in metrics.items():
                if key != 'pair':
                    print(f"  {key}: {value}")
        else:
            print(f"No validation data for {args.validate}")
        return
    
    if args.validate_all:
        df = pipeline.validate_all_models()
        if not df.empty:
            print("\n📊 Validation Results (sorted by accuracy):")
            print(df[['pair', 'accuracy', 'f1_score', 'samples']].to_string(index=False))
        else:
            print("No validation results")
        return
    
    if args.schedule:
        results = pipeline.schedule_training(
            days_threshold=args.days_threshold,
            min_accuracy_threshold=args.accuracy_threshold
        )
        print(f"\n🔄 Scheduled training results:")
        print(f"  Status: {results.get('status', 'unknown')}")
        if 'updated_count' in results:
            print(f"  Updated: {results['updated_count']} models")
            for pair in results.get('updated_pairs', []):
                print(f"    - {pair}")
        return
    
    if args.pair:
        result = pipeline.train_pair(
            args.pair, 
            force=args.force, 
            force_cpu=args.cpu,
            samples=args.samples,
            lookback=args.lookback
        )
        print(f"\n🎯 Training result for {args.pair}:")
        for key, value in result.items():
            print(f"  {key}: {value}")
    elif args.all:
        results = pipeline.train_all_pairs(
            force=args.force,
            force_cpu=args.cpu,
            samples=args.samples,
            lookback=args.lookback
        )
        print(f"\n📊 Training summary:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()