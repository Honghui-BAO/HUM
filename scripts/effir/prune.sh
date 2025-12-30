#!/bin/bash

# Parameters
CONFIG=${1:-configs/train_HUM.yaml}
SCORES_PATH=$2
LOAD_PATH=$3
TOP_K=${4:-8}
SAVE_PATH=${5:-ckp/pruned_model.pth}

if [ -z "$SCORES_PATH" ] || [ -z "$LOAD_PATH" ]; then
    echo "Usage: $0 <config_path> <scores_json_path> <load_path> [top_k] [save_path]"
    exit 1
fi

echo "========================================================================"
echo "Stage 2: Pruning Redundant MLP Layers (EffiR)"
echo "========================================================================"
echo "Config: $CONFIG"
echo "Scores Path: $SCORES_PATH"
echo "Load Path: $LOAD_PATH"
echo "Top K: $TOP_K"
echo "Save Path: $SAVE_PATH"
echo "========================================================================"

python prune_model.py \
    --config "$CONFIG" \
    --scores_path "$SCORES_PATH" \
    --load_path "$LOAD_PATH" \
    --top_k "$TOP_K" \
    --save_path "$SAVE_PATH"

echo "Pruning completed. Pruned model saved to $SAVE_PATH."
