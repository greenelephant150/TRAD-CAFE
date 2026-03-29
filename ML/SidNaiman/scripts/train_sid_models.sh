#!/bin/bash
#
# Train Sid Method AI Models with GPU Acceleration
#

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "🎯 SID METHOD AI MODEL TRAINING (GPU ACCELERATED)"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"

# Change to project root directory
cd "$PROJECT_ROOT"

# Create model directory if it doesn't exist
mkdir -p ai/trained_models/

# Check if sid_method.py exists
if [ ! -f "sid_method.py" ]; then
    echo "❌ ERROR: sid_method.py not found in $PROJECT_ROOT"
    exit 1
else
    echo "✅ sid_method.py found"
fi

# Check GPU libraries
echo ""
echo "🔍 Checking GPU libraries..."
python -c "
import sys
sys.path.insert(0, '.')
try:
    import cupy as cp
    gpu_count = cp.cuda.runtime.getDeviceCount()
    print(f'  ✅ CuPy available: {gpu_count} GPU(s)')
except ImportError:
    print('  ⚠️ CuPy not available')
try:
    import xgboost as xgb
    print(f'  ✅ XGBoost {xgb.__version__}')
    # Test GPU availability
    import numpy as np
    dtrain = xgb.DMatrix(np.random.randn(100, 10))
    params = {'tree_method': 'hist', 'device': 'cuda:0', 'max_depth': 2}
    model = xgb.train(params, dtrain, num_boost_round=2, verbose_eval=False)
    print('  ✅ XGBoost GPU acceleration available')
except Exception as e:
    print(f'  ⚠️ XGBoost GPU not available: {e}')
"

# Train direction prediction models with GPU
echo ""
echo "📊 Training DIRECTION prediction models (GPU accelerated)..."
echo "   This will train models for ALL 87 pairs"
echo "   Estimated time: 3-4 hours with GPU"
echo ""

# Run GPU training
bash scripts/run_gpu_training.sh

echo ""
echo "=================================================="
echo "✅ Sid Method model training complete!"
echo "=================================================="

# Show GPU memory usage after training
echo ""
echo "📊 Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv