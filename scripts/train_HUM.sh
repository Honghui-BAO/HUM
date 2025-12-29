#!/bin/bash
export NCCL_P2P_LEVEL=NVL
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080 
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Parameters
OUTPUT_PATH=${1:-./ckp/hum_v1.pth}
DATASET=${2:-mIndustrial_and_Scientific}
LR=${3:-5e-5}
CONFIG=${4:-configs/train_HUM.yaml}
NUM_GPUS=${5:-8}
PORT=${6:-29500}

echo "========================================================================"
echo "Training HUM v1"
echo "========================================================================"
echo "Output: $OUTPUT_PATH"
echo "Dataset: $DATASET"
echo "Learning Rate: $LR"
echo "Config: $CONFIG"
echo "GPUs: $NUM_GPUS"
echo "Port: $PORT"
echo "========================================================================"
echo ""

# Run training
torchrun --nproc_per_node=$NUM_GPUS --master_port=$PORT run_hum.py \
    --output "$OUTPUT_PATH" \
    --dataset "$DATASET" \
    --lr "$LR" \
    --config "$CONFIG"

echo ""
echo "========================================================================"
echo "Training Completed!"
echo "========================================================================"
echo "Model saved to: $OUTPUT_PATH"
echo "========================================================================"


