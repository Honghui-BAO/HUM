#!/bin/bash

# Parameters
PRUNED_PATH=$1
PRUNED_LAYERS=$2
CONFIG=${3:-configs/train_HUM.yaml}
DATASET=${4:-mIndustrial_and_Scientific}
NUM_GPUS=${5:-1}

if [ -z "$PRUNED_PATH" ] || [ -z "$PRUNED_LAYERS" ]; then
    echo "Usage: $0 <pruned_model_path> <pruned_layers_comma_separated> [config] [dataset] [num_gpus]"
    echo "Example: $0 ckp/pruned_model.pth '20,21,22,23,24,25,26,27'"
    exit 1
fi

echo "========================================================================"
echo "Stage 3: Validating Pruned Model (EffiR)"
echo "========================================================================"
echo "Pruned Model Path: $PRUNED_PATH"
echo "Pruned Layers: $PRUNED_LAYERS"
echo "Config: $CONFIG"
echo "Dataset: $DATASET"
echo "========================================================================"

# Use torchrun to launch validation with unbuffered output
PYTHONUNBUFFERED=1 torchrun --nproc_per_node=$NUM_GPUS run_hum.py \
    --config "$CONFIG" \
    --load "$PRUNED_PATH" \
    --dataset "$DATASET" \
    --pruned_mlp_layers "$PRUNED_LAYERS" \
    --test_only true

echo "Validation completed."
