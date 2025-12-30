#!/bin/bash

# Parameters
CONFIG=${1:-configs/train_HUM.yaml}
LOAD_PATH=$2
DATASET=${3:-mIndustrial_and_Scientific}

if [ -z "$LOAD_PATH" ]; then
    echo "Usage: $0 <config_path> <load_path> [dataset]"
    exit 1
fi

echo "========================================================================"
echo "Stage 1: Analyzing Layer Importance (EffiR)"
echo "========================================================================"
echo "Config: $CONFIG"
echo "Load Path: $LOAD_PATH"
echo "Dataset: $DATASET"
echo "========================================================================"

# Set validation data path if provided by user (server specific)
export VALID_DATA_PATH="/llm-reco-ssd-share/baohonghui/Baselines/HUM/local_dataset/${DATASET}-1.0-5-5/valid_data.pkl"

python calculate_importance.py \
    --config "$CONFIG" \
    --load "$LOAD_PATH" \
    --dataset "$DATASET"

echo "Analysis completed. Check layer_importance_*.png/json."
