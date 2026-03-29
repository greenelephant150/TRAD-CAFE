#!/usr/bin/env python3
"""
Test XGBoost GPU with correct parameters for version 3.2.0
"""

import os
import numpy as np

# Set CUDA environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.3'
os.environ['CUDA_PATH'] = '/usr/local/cuda-12.3'

print("=" * 60)
print("🎯 XGBoost GPU Test (v3.2.0)")
print("=" * 60)

print("\n1. Testing XGBoost...")
try:
    import xgboost as xgb
    print(f"   ✅ XGBoost version: {xgb.__version__}")
    
    # Check GPU build
    print("\n2. Checking GPU build...")
    try:
        # Create sample data
        X = np.random.randn(10000, 20).astype(np.float32)
        y = np.random.randint(0, 2, 10000)
        
        dtrain = xgb.DMatrix(X, label=y)
        
        # Test with device parameter (new in 3.2.0)
        params = {
            'max_depth': 6,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'tree_method': 'hist',  # Use hist with device
            'device': 'cuda:0',  # This enables GPU
            'verbosity': 1
        }
        
        print("   Training XGBoost with GPU (device='cuda:0')...")
        model_gpu = xgb.train(params, dtrain, num_boost_round=20, verbose_eval=False)
        print("   ✅ XGBoost GPU training successful!")
        
        # Test prediction
        X_test = np.random.randn(1000, 20).astype(np.float32)
        dtest = xgb.DMatrix(X_test)
        pred = model_gpu.predict(dtest)
        print(f"   ✅ Prediction successful: {pred.shape}")
        
    except Exception as e:
        print(f"   ❌ GPU test failed: {e}")
        
except Exception as e:
    print(f"   ❌ XGBoost error: {e}")

print("\n" + "=" * 60)
print("✅ Test complete!")
print("=" * 60)