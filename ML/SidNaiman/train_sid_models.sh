#!/bin/bash
#
# Train Sid Method AI Models with GPU Acceleration
# Uses cuML and XGBoost GPU for faster training
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
    print(f'  ✅ CuPy available: {cp.cuda.runtime.getDeviceCount()} GPU(s)')
except ImportError:
    print('  ⚠️ CuPy not available')
try:
    import cuml
    print('  ✅ cuML available (GPU Random Forest)')
except ImportError:
    print('  ⚠️ cuML not available')
try:
    import xgboost as xgb
    print('  ✅ XGBoost available')
except ImportError:
    print('  ⚠️ XGBoost not available')
"

# Train direction prediction models (default) with GPU
echo ""
echo "📊 Training DIRECTION prediction models (GPU accelerated)..."
echo "   This may take several minutes per pair..."
python scripts/pkltrainer3.py --all --target target_direction --lookback 10 2>&1 | tee -a training_direction.log

# Check if training was successful
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Direction prediction training completed"
else
    echo "❌ Direction prediction training failed. Check training_direction.log"
fi

# Train RSI 50 (5-bar) prediction models
echo ""
echo "📊 Training RSI 50 (5-bar) prediction models (GPU accelerated)..."
python scripts/pkltrainer3.py --all --target target_rsi_50_5 --lookback 10 2>&1 | tee -a training_rsi5.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ RSI 50 (5-bar) training completed"
else
    echo "❌ RSI 50 (5-bar) training failed. Check training_rsi5.log"
fi

# Train RSI 50 (10-bar) prediction models
echo ""
echo "📊 Training RSI 50 (10-bar) prediction models (GPU accelerated)..."
python scripts/pkltrainer3.py --all --target target_rsi_50_10 --lookback 15 2>&1 | tee -a training_rsi10.log

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ RSI 50 (10-bar) training completed"
else
    echo "❌ RSI 50 (10-bar) training failed. Check training_rsi10.log"
fi

echo ""
echo "=================================================="
echo "✅ Sid Method model training complete!"
echo "=================================================="

# Show GPU memory usage after training
echo ""
echo "📊 GPU Memory Usage After Training:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv

# Show model summary
echo ""
echo "📊 Model Summary:"
if [ -d "ai/trained_models" ]; then
    echo ""
    echo "  Models by target:"
    for target in target_direction target_rsi_50_5 target_rsi_50_10; do
        count=$(find ai/trained_models -name "*SidMethod*--${target}--*.pkl" -type f 2>/dev/null | wc -l)
        echo "    $target: $count models"
    done
    
    echo ""
    echo "  Latest models:"
    ls -lt ai/trained_models/*SidMethod*.pkl 2>/dev/null | head -10
fi