"""
AI Module for SID Method Trading System
"""

from .feature_engineering import FeatureEngineer, FeatureEngineeringConfig
from .model_trainer import ModelTrainer, ModelTrainingConfig
from .signal_predictor import SignalPredictor, SignalPredictorConfig
from .training_pipeline import TrainingPipeline, TrainingPipelineConfig

# Optional imports (if files exist)
try:
    from .ai_accelerator import AIAccelerator
except ImportError:
    AIAccelerator = None
    print("⚠️ ai_accelerator module not found - GPU acceleration disabled")

try:
    from .gpu_data_loader import GPUDataLoader
except ImportError:
    GPUDataLoader = None
    print("⚠️ gpu_data_loader module not found")

__all__ = [
    'FeatureEngineer',
    'FeatureEngineeringConfig',
    'ModelTrainer',
    'ModelTrainingConfig',
    'SignalPredictor',
    'SignalPredictorConfig',
    'TrainingPipeline',
    'TrainingPipelineConfig',
    'AIAccelerator',
    'GPUDataLoader'
]