#!/bin/bash
export NCCL_P2P_LEVEL=NVL
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080 
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Parameters
CONFIG=${1:-configs/train_HUM.yaml}
LOAD_PATH=${2:-/llm-reco-ssd-share/baohonghui/Baselines/HUM/ckp/hum_v1_qwen2.pth}
DATASET=${3:-mIndustrial_and_Scientific}


echo "========================================================================"
echo "Stage 1: Analyzing Layer Importance (EffiR)"
echo "========================================================================"
echo "Config: $CONFIG"
echo "Load Path: $LOAD_PATH"
echo "Dataset: $DATASET"
echo "========================================================================"

# Set validation data path if provided by user (server specific)
export VALID_DATA_PATH="/llm-reco-ssd-share/baohonghui/Baselines/HUM/local_dataset/${DATASET}-1.0-5-5/valid_data.pkl"

python3 -u calculate_importance.py \
    --config "$CONFIG" \
    --load "$LOAD_PATH" \
    --dataset "$DATASET"

echo "Analysis completed. Check layer_importance_*.png/json."
