"""
AI Accelerator Module
Manages GPU/CPU initialization for AI model training and inference
Supports:
- CPU: pandas, numpy, tensorflow-cpu, scikit-learn
- GPU: cuml, cupy, cudf, tensorflow-gpu, cudnn
"""

import os
import logging
import platform
from typing import Dict, Optional, Tuple, Any
import warnings

logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class AIAccelerator:
    """
    Centralized GPU/CPU management for AI operations
    Automatically detects and initializes available hardware
    """
    
    def __init__(self, preferred_device: str = 'auto'):
        """
        Args:
            preferred_device: 'auto', 'cpu', 'gpu', 'cuda', 'rocm'
        """
        self.preferred_device = preferred_device
        self.device_type = 'cpu'  # Default fallback
        self.device_name = 'CPU'
        self.has_gpu = False
        self.gpu_libraries = {}
        self.cpu_libraries = {}
        
        # Initialize and detect hardware
        self._detect_hardware()
        self._initialize_libraries()
        
        logger.info(f"AI Accelerator initialized with device: {self.device_type}")
        if self.has_gpu:
            logger.info(f"  GPU detected: {self.device_name}")
    
    def _detect_hardware(self):
        """Detect available GPU hardware"""
        # Check for NVIDIA CUDA
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.device_name = pynvml.nvmlDeviceGetName(handle).decode()
                self.has_gpu = True
                self.device_type = 'cuda' if self.preferred_device in ['auto', 'gpu', 'cuda'] else 'cpu'
            pynvml.nvmlShutdown()
        except:
            pass
        
        # Check for ROCm (AMD)
        if not self.has_gpu:
            try:
                import amdsmi
                amdsmi.amdsmi_init()
                if amdsmi.amdsmi_get_processor_handle_count() > 0:
                    self.has_gpu = True
                    self.device_name = 'AMD ROCm'
                    self.device_type = 'rocm' if self.preferred_device in ['auto', 'gpu', 'rocm'] else 'cpu'
                amdsmi.amdsmi_shutdown()
            except:
                pass
    
    def _initialize_libraries(self):
        """Initialize appropriate libraries based on hardware"""
        
        # CPU Libraries (always available)
        self.cpu_libraries = {
            'pandas': self._import_cpu_pandas,
            'numpy': self._import_cpu_numpy,
            'sklearn': self._import_cpu_sklearn,
            'tensorflow': self._import_cpu_tensorflow
        }
        
        # GPU Libraries (if available and selected)
        if self.has_gpu and self.device_type != 'cpu':
            self.gpu_libraries = {
                'cudf': self._import_gpu_cudf,
                'cuml': self._import_gpu_cuml,
                'cupy': self._import_gpu_cupy,
                'tensorflow_gpu': self._import_gpu_tensorflow,
                'torch': self._import_gpu_torch
            }
    
    def _import_cpu_pandas(self):
        """Import pandas (CPU)"""
        import pandas as pd
        return pd
    
    def _import_cpu_numpy(self):
        """Import numpy (CPU)"""
        import numpy as np
        return np
    
    def _import_cpu_sklearn(self):
        """Import scikit-learn (CPU)"""
        import sklearn
        return sklearn
    
    def _import_cpu_tensorflow(self):
        """Import TensorFlow CPU version"""
        try:
            import tensorflow as tf
            # Force CPU only
            tf.config.set_visible_devices([], 'GPU')
            return tf
        except ImportError:
            return None
    
    def _import_gpu_cudf(self):
        """Import RAPIDS cuDF (GPU)"""
        try:
            import cudf
            return cudf
        except ImportError:
            logger.warning("cuDF not available, falling back to pandas")
            return None
    
    def _import_gpu_cuml(self):
        """Import RAPIDS cuML (GPU)"""
        try:
            import cuml
            return cuml
        except ImportError:
            logger.warning("cuML not available, falling back to scikit-learn")
            return None
    
    def _import_gpu_cupy(self):
        """Import CuPy (GPU NumPy)"""
        try:
            import cupy as cp
            return cp
        except ImportError:
            logger.warning("CuPy not available, falling back to NumPy")
            return None
    
    def _import_gpu_tensorflow(self):
        """Import TensorFlow GPU version"""
        try:
            import tensorflow as tf
            # Verify GPU is visible
            if len(tf.config.list_physical_devices('GPU')) > 0:
                return tf
            else:
                logger.warning("TensorFlow GPU requested but no GPU devices found")
                return None
        except ImportError:
            return None
    
    def _import_gpu_torch(self):
        """Import PyTorch with CUDA support"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch
            else:
                logger.warning("PyTorch CUDA requested but not available")
                return None
        except ImportError:
            return None
    
    def get_pandas(self):
        """Get pandas (always CPU)"""
        return self._import_cpu_pandas()
    
    def get_numpy(self):
        """Get numpy or cupy based on device"""
        if self.device_type != 'cpu' and self.gpu_libraries.get('cupy'):
            cupy = self.gpu_libraries['cupy']()
            if cupy:
                return cupy
        return self._import_cpu_numpy()
    
    def get_sklearn_or_cuml(self):
        """Get scikit-learn (CPU) or cuML (GPU)"""
        if self.device_type != 'cpu' and self.gpu_libraries.get('cuml'):
            cuml = self.gpu_libraries['cuml']()
            if cuml:
                return cuml
        return self._import_cpu_sklearn()
    
    def get_tensorflow(self):
        """Get appropriate TensorFlow version"""
        if self.device_type != 'cpu' and self.gpu_libraries.get('tensorflow_gpu'):
            tf = self.gpu_libraries['tensorflow_gpu']()
            if tf:
                return tf
        return self._import_cpu_tensorflow()
    
    def get_dataframe_library(self):
        """Get pandas (CPU) or cudf (GPU) for DataFrame operations"""
        if self.device_type != 'cpu' and self.gpu_libraries.get('cudf'):
            cudf = self.gpu_libraries['cudf']()
            if cudf:
                return cudf
        return self._import_cpu_pandas()
    
    def to_device(self, data, library='numpy'):
        """Move data to appropriate device (CPU/GPU)"""
        if self.device_type == 'cpu':
            return data
        
        if library == 'numpy' and self.gpu_libraries.get('cupy'):
            cupy = self.gpu_libraries['cupy']()
            if cupy:
                return cupy.asarray(data)
        
        return data
    
    def to_host(self, data, library='numpy'):
        """Move data from GPU back to CPU"""
        if self.device_type == 'cpu':
            return data
        
        if library == 'cupy' and hasattr(data, 'get'):
            return data.get()
        
        return data
    
    def get_device_info(self) -> Dict:
        """Get detailed device information"""
        info = {
            'device_type': self.device_type,
            'device_name': self.device_name,
            'has_gpu': self.has_gpu,
            'gpu_libraries_available': {},
            'cpu_libraries_available': {}
        }
        
        # Check GPU libraries
        for name, importer in self.gpu_libraries.items():
            lib = importer()
            info['gpu_libraries_available'][name] = lib is not None
        
        # Check CPU libraries
        for name, importer in self.cpu_libraries.items():
            lib = importer()
            info['cpu_libraries_available'][name] = lib is not None
        
        return info
    
    def get_memory_info(self) -> Dict:
        """Get memory usage information"""
        info = {'device': self.device_type}
        
        if self.device_type == 'cpu':
            import psutil
            mem = psutil.virtual_memory()
            info['total'] = mem.total / 1e9
            info['available'] = mem.available / 1e9
            info['percent_used'] = mem.percent
        else:
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info['total'] = pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1e9
                info['used'] = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1e9
                info['free'] = pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1e9
                pynvml.nvmlShutdown()
            except:
                pass
        
        return info