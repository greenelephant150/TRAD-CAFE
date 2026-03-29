"""
GPU-accelerated ML predictor for Simon Pullen patterns
Uses PyTorch for faster processing of large datasets
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import pickle
from pathlib import Path

from src.utils.device_manager import get_device_manager

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, GPU acceleration disabled")


class SimpleNN(nn.Module):
    """Simple neural network for pattern classification"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32]):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class GPUPredictor:
    """
    GPU-accelerated predictor using PyTorch
    Falls back to CPU if GPU not available
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device_manager = get_device_manager(config.get('device', 'auto'))
        self.device = self.device_manager.get_device()
        self.model = None
        self.scaler = None
        self.input_size = None
        
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, using CPU fallback")
            self.device = 'cpu'
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.001):
        """Train the model with GPU acceleration"""
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for training")
            return self
        
        # Convert to tensors and move to device
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        
        train_dataset = TensorDataset(
            X_tensor[:split_idx], 
            y_tensor[:split_idx].view(-1, 1)
        )
        val_dataset = TensorDataset(
            X_tensor[split_idx:], 
            y_tensor[split_idx:].view(-1, 1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create model
        self.input_size = X.shape[1]
        self.model = SimpleNN(self.input_size).to(self.device)
        
        # Use DataParallel if multiple GPUs and parallel requested
        if self.config.get('parallel', False) and self.device == 'cuda':
            self.model = self.device_manager.parallelize_if_possible(self.model)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            if epoch % 10 == 0:
                self.model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        predicted = (outputs > 0.5).float()
                        total += batch_y.size(0)
                        correct += (predicted == batch_y).sum().item()
                
                accuracy = 100 * correct / total
                logger.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, "
                          f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {accuracy:.2f}%")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities with GPU acceleration"""
        if self.model is None:
            logger.error("Model not trained")
            return np.zeros(len(X))
        
        if not TORCH_AVAILABLE:
            logger.error("PyTorch not available for prediction")
            return np.zeros(len(X))
        
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'module'):  # DataParallel
                outputs = self.model.module(X_tensor)
            else:
                outputs = self.model(X_tensor)
        
        return outputs.cpu().numpy().flatten()
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classes"""
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)
    
    def save(self, filepath: str):
        """Save model to file"""
        if self.model is None:
            logger.error("No model to save")
            return
        
        # Move model to CPU for saving
        if hasattr(self.model, 'module'):  # DataParallel
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        save_dict = {
            'model_state': model_state,
            'input_size': self.input_size,
            'config': self.config,
            'scaler': self.scaler
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from file"""
        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.input_size = save_dict['input_size']
        self.config = save_dict['config']
        self.scaler = save_dict['scaler']
        
        # Recreate model
        self.model = SimpleNN(self.input_size).to(self.device)
        self.model.load_state_dict(save_dict['model_state'])
        
        if self.config.get('parallel', False) and self.device == 'cuda':
            self.model = self.device_manager.parallelize_if_possible(self.model)
        
        logger.info(f"Model loaded from {filepath}")
        return self
