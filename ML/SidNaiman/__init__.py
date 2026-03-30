"""
Sid Naiman's SID Method Trading System

A complete trading system based on Sid Naiman's SID Method:
- RSI signals (<30 oversold, >70 overbought)
- MACD alignment confirmation
- Stop loss at swing low/high rounded to whole numbers
- Take profit at RSI 50
- Position size calculator with 0.5-2% risk
- Daily charts only
"""

"""
AI Augmentation Module for Sid Naiman's SID Method
"""

from .ai_accelerator import AIAccelerator
from .feature_engineering import FeatureEngineer
from .signal_predictor import SignalPredictor
from .model_trainer import ModelTrainer
from .model_manager import ModelManager
from .training_pipeline import TrainingPipeline

__all__ = [
    'AIAccelerator', 
    'FeatureEngineer', 
    'SignalPredictor', 
    'ModelTrainer',
    'ModelManager',
    'TrainingPipeline'
]