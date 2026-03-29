"""
Device management for CPU/GPU processing
Handles device selection and fallback
"""

import os
import logging
from typing import Tuple, Optional
import platform

logger = logging.getLogger(__name__)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, using CPU only")

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pass


class DeviceManager:
    """
    Manages device selection (CPU/GPU) for processing
    Provides fallback mechanisms when GPU is requested but not available
    """
    
    def __init__(self, preferred_device: str = 'auto'):
        """
        Initialize device manager
        
        Args:
            preferred_device: 'cpu', 'cuda', 'mps' (Apple Silicon), or 'auto'
        """
        self.preferred_device = preferred_device
        self.device, self.device_name, self.device_type = self._detect_device()
        
    def _detect_device(self) -> Tuple[str, str, str]:
        """
        Detect available devices and return the best one based on preference
        
        Returns:
            Tuple of (device_string, device_name, device_type)
        """
        # Check for CUDA (NVIDIA GPUs)
        cuda_available = False
        cuda_device_count = 0
        cuda_device_name = ""
        
        if TORCH_AVAILABLE:
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_device_count = torch.cuda.device_count()
                cuda_device_name = torch.cuda.get_device_name(0) if cuda_device_count > 0 else ""
        
        # Check for MPS (Apple Silicon)
        mps_available = False
        if TORCH_AVAILABLE and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            mps_available = True
        
        # Determine best device based on preference
        if self.preferred_device == 'cuda' and cuda_available:
            return ('cuda', f'CUDA ({cuda_device_name})', 'gpu')
        elif self.preferred_device == 'mps' and mps_available:
            return ('mps', 'MPS (Apple Silicon)', 'gpu')
        elif self.preferred_device == 'cpu':
            return ('cpu', 'CPU', 'cpu')
        elif self.preferred_device == 'auto':
            if cuda_available:
                return ('cuda', f'CUDA ({cuda_device_name})', 'gpu')
            elif mps_available:
                return ('mps', 'MPS (Apple Silicon)', 'gpu')
            else:
                cpu_name = platform.processor() or "CPU"
                return ('cpu', f'{cpu_name}', 'cpu')
        else:
            # Fallback to CPU
            logger.warning(f"Preferred device {self.preferred_device} not available, falling back to CPU")
            return ('cpu', 'CPU (fallback)', 'cpu')
    
    def get_device(self) -> str:
        """Get the current device string"""
        return self.device
    
    def get_device_info(self) -> dict:
        """Get detailed device information"""
        return {
            'device': self.device,
            'name': self.device_name,
            'type': self.device_type,
            'preferred': self.preferred_device,
            'torch_available': TORCH_AVAILABLE
        }
    
    def to_device(self, data):
        """Move data to the selected device if applicable"""
        if self.device_type == 'cpu' or not TORCH_AVAILABLE:
            return data
            
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.to_device(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self.to_device(item) for item in data)
        else:
            return data
    
    def parallelize_if_possible(self, model, num_gpus=None):
        """Parallelize model across multiple GPUs if available"""
        if self.device_type != 'gpu' or not TORCH_AVAILABLE:
            return model
            
        if self.device == 'cuda':
            if num_gpus is None:
                num_gpus = torch.cuda.device_count()
            
            if num_gpus > 1:
                logger.info(f"Using DataParallel across {num_gpus} GPUs")
                model = torch.nn.DataParallel(model)
            
            return model.to(self.device)
        else:
            return model.to(self.device)
    
    @staticmethod
    def get_optimal_workers() -> int:
        """Get optimal number of worker threads based on CPU cores"""
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use 75% of available cores, min 1, max 8
        return max(1, min(8, int(cpu_count * 0.75)))


# Singleton instance
_device_manager = None


def get_device_manager(preferred_device: str = 'auto') -> DeviceManager:
    """Get or create the global device manager instance"""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager(preferred_device)
    return _device_manager


def set_preferred_device(device: str) -> DeviceManager:
    """Change the preferred device"""
    global _device_manager
    _device_manager = DeviceManager(device)
    return _device_manager
