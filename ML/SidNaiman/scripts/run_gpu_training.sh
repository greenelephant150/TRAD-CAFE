#!/bin/bash
#
# Run GPU-accelerated training with proper CUDA 12.3 environment
#

# Set CUDA 12.3 environment
export CUDA_HOME=/usr/local/cuda-12.3
export CUDA_PATH=/usr/local/cuda-12.3
export CUDA_ROOT=/usr/local/cuda-12.3
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-12.3/bin:$PATH

# Set XGBoost GPU environment
export XGBOOST_USE_CUDA=1

# Set memory limits
export TF_GPU_ALLOCATOR=cuda_malloc_async
export CUDA_MEMORY_LIMIT=10737418240

echo "=================================================="
echo "🎯 GPU-ACCELERATED SID METHOD TRAINING"
echo "=================================================="
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_PATH: $CUDA_PATH"
echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
echo ""

# Test GPU libraries
python -c "
import os
print('Testing GPU libraries...')
try:
    import xgboost as xgb
    print(f'✅ XGBoost {xgb.__version__}')
except Exception as e:
    print(f'❌ XGBoost: {e}')
try:
    import cupy as cp
    print(f'✅ CuPy {cp.__version__} - {cp.cuda.runtime.getDeviceCount()} GPUs')
except Exception as e:
    print(f'❌ CuPy: {e}')
"

echo ""
echo "=================================================="
echo "Starting GPU-accelerated training..."
echo "=================================================="

# Change to project root
cd /mnt2/Trading-Cafe/ML/SNaiman2

# Run training with GPU
python scripts/pkltrainer3.py --all --target target_direction --lookback 10 2>&1 | tee -a training_gpu.log

echo ""
echo "=================================================="
echo "Training complete!"
echo "=================================================="

# Show summary
echo ""
echo "📊 Model Summary:"
if [ -d "ai/trained_models" ]; then
    count=$(find ai/trained_models -name "*SidMethod*--target_direction--*.pkl" -type f 2>/dev/null | wc -l)
    echo "  Total models trained: $count"
    echo ""
    echo "  Latest models:"
    ls -lt ai/trained_models/*SidMethod*.pkl 2>/dev/null | head -10
fi