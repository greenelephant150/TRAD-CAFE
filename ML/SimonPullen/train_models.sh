#!/bin/bash
# train_models.sh - Wrapper script for model training

# Set up environment
export PYTHONPATH=/mnt2/Trading-Cafe/ML/SPullen:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}🚀 Simon Pullen AI Model Training${NC}"
echo -e "${BLUE}================================================================${NC}"

# Check if Python script exists
if [ ! -f "train_model_from_parquet.py" ]; then
    echo -e "${RED}Error: train_model_from_parquet.py not found${NC}"
    exit 1
fi

# Function to show usage
show_usage() {
    echo -e "${YELLOW}Usage:${NC}"
    echo "  ./train_models.sh --pair GBP_USD        # Train single pair"
    echo "  ./train_models.sh --all                 # Train all pairs"
    echo "  ./train_models.sh --pair GBP_USD --force # Force retrain"
    echo "  ./train_models.sh --all --force         # Force retrain all"
    echo "  ./train_models.sh --list                 # List available pairs"
}

# Parse arguments
FORCE=""
PAIR=""
ALL=""
LIST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --pair)
            PAIR="$2"
            shift 2
            ;;
        --all)
            ALL="--all"
            shift
            ;;
        --force)
            FORCE="--force"
            shift
            ;;
        --list)
            LIST="true"
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# List available pairs
if [ "$LIST" = "true" ]; then
    echo -e "${YELLOW}Available trading pairs:${NC}"
    python3 -c "
import os
path = '/home/grct/Forex_Parquet'
if os.path.exists(path):
    pairs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for p in sorted(pairs):
        print(f'  • {p}')
else:
    print('No pairs found')
"
    exit 0
fi

# Check GPU availability
echo -e "${YELLOW}Checking GPU status...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    GPU_FLAG=""
else
    echo -e "${RED}No NVIDIA GPU detected, falling back to CPU${NC}"
    GPU_FLAG=""
fi

echo -e "${BLUE}================================================================${NC}"

# Run training
if [ -n "$PAIR" ]; then
    echo -e "${GREEN}Training model for: $PAIR${NC}"
    python3 train_model_from_parquet.py --pair "$PAIR" $FORCE
elif [ -n "$ALL" ]; then
    echo -e "${GREEN}Training models for ALL pairs${NC}"
    python3 train_model_from_parquet.py --all $FORCE
else
    show_usage
    exit 1
fi

# Check result
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Training completed successfully${NC}"
    
    # Show latest models
    echo -e "${YELLOW}Latest models in /mnt2/Trading-Cafe/ML/SPullen/ai/trained_models/:${NC}"
    ls -lt /mnt2/Trading-Cafe/ML/SPullen/ai/trained_models/*.pkl 2>/dev/null | head -5
else
    echo -e "${RED}❌ Training failed${NC}"
    exit 1
fi
