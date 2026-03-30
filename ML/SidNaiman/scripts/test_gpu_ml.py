#!/usr/bin/env python3
"""
Test GPU ML with proper CUDA 12.3 environment
"""

import os
import sys

# Set CUDA 12.3 environment
os.environ['CUDA_HOME'] = '/usr/local/cuda-12.3'
os.environ['CUDA_PATH'] = '/usr/local/cuda-12.3'
os.environ['CUDA_ROOT'] = '/usr/local/cuda-12.3'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.3/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['NUMBA_CUDA_DRIVER'] = '/usr/lib/x86_64-linux-gnu/libcuda.so'

print("=" * 60)
print("🎯 GPU ML Test - CUDA 12.3")
print("=" * 60)

# Test 1: CuPy
print("\n1. Testing CuPy...")
try:
    import cupy as cp
    print(f"   ✅ CuPy version: {cp.__version__}")
    print(f"   GPU count: {cp.cuda.runtime.getDeviceCount()}")
    
    # Get GPU info
    for i in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(i):
            props = cp.cuda.runtime.getDeviceProperties(i)
            free, total = cp.cuda.runtime.memGetInfo()
            print(f"   GPU {i}: {props['name'].decode()} - {free/1024**3:.1f}GB free / {total/1024**3:.1f}GB total")
    
    # Simple test
    a = cp.array([1, 2, 3])
    b = cp.array([4, 5, 6])
    c = a + b
    print(f"   ✅ GPU array works: {c.get()}")
    
except Exception as e:
    print(f"   ❌ CuPy error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: cuML
print("\n2. Testing cuML...")
try:
    import cuml
    from cuml.ensemble import RandomForestClassifier as cumlRF
    print(f"   ✅ cuML version: {cuml.__version__}")
    
    # Create sample data on GPU
    X = cp.random.randn(10000, 20, dtype=cp.float32)
    y = cp.random.randint(0, 2, 10000, dtype=cp.int32)
    print(f"   Sample data: X shape {X.shape}, y shape {y.shape}")
    
    # Train model
    print("   Training cuML Random Forest...")
    model = cumlRF(
        n_estimators=50,
        max_depth=10,
        random_state=42,
        n_streams=1
    )
    model.fit(X, y)
    print("   ✅ cuML training successful!")
    
    # Test prediction
    X_test = cp.random.randn(2000, 20, dtype=cp.float32)
    y_pred = model.predict(X_test)
    print(f"   ✅ Prediction successful: {y_pred.shape}")
    
except Exception as e:
    print(f"   ❌ cuML error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: XGBoost GPU
print("\n3. Testing XGBoost GPU...")
try:
    import xgboost as xgb
    import numpy as np
    print(f"   ✅ XGBoost version: {xgb.__version__}")
    
    # Create sample data on CPU
    X_np = np.random.randn(10000, 20).astype(np.float32)
    y_np = np.random.randint(0, 2, 10000)
    
    dtrain = xgb.DMatrix(X_np, label=y_np)
    
    # Try GPU
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'gpu_id': 0,
        'verbosity': 0
    }
    
    print("   Training XGBoost with GPU...")
    model = xgb.train(params, dtrain, num_boost_round=50, verbose_eval=False)
    print("   ✅ XGBoost GPU training successful!")
    
except Exception as e:
    print(f"   ⚠️ XGBoost GPU not available: {e}")

print("\n" + "=" * 60)
print("✅ Test complete!")
print("=" * 60)