#!/usr/bin/env python3
"""
Scheduled training script - meant to be run by cron or task scheduler
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.training_pipeline import TrainingPipeline

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def main():
    """Run scheduled training"""
    logger.info("="*60)
    logger.info("Starting scheduled training")
    logger.info("="*60)
    
    pipeline = TrainingPipeline()
    
    # Run scheduled training with default thresholds
    results = pipeline.schedule_training(
        days_threshold=7,      # Retrain models older than 7 days
        accuracy_threshold=0.55 # Retrain models with accuracy below 55%
    )
    
    # Log results
    if results.get('status') == 'up_to_date':
        logger.info("All models are up to date")
    elif results.get('status') == 'updated':
        logger.info(f"Updated {results['updated_count']} models")
        for pair in results.get('updated_pairs', []):
            logger.info(f"  - {pair}")
    
    # Validate all models
    logger.info("\nValidating all models...")
    validation_df = pipeline.validate_all_models(days=30)
    
    if not validation_df.empty:
        logger.info(f"Validation complete. Average accuracy: {validation_df['accuracy'].mean():.3f}")
        
        # Log low accuracy models
        low_accuracy = validation_df[validation_df['accuracy'] < 0.55]
        if not low_accuracy.empty:
            logger.warning("Models with low accuracy:")
            for _, row in low_accuracy.iterrows():
                logger.warning(f"  {row['pair']}: {row['accuracy']:.3f}")
    
    # Export metrics
    pipeline.export_model_metrics(f'model_metrics_{datetime.now().strftime("%Y%m%d")}.json')
    
    # Cleanup old models
    logger.info("\nCleaning up old models...")
    pipeline.cleanup_old_models(keep_last=3)
    
    logger.info("="*60)
    logger.info("Scheduled training complete")
    logger.info("="*60)

if __name__ == "__main__":
    main()