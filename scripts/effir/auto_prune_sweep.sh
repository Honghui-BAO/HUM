#!/bin/bash
export NCCL_P2P_LEVEL=NVL
export http_proxy=http://oversea-squid1.jp.txyun:11080 
export https_proxy=http://oversea-squid1.jp.txyun:11080 
export no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com

# Parameters
CONFIG=${1:-configs/train_HUM.yaml}
SCORES_PATH=${2:-/llm-reco-ssd-share/baohonghui/Baselines/HUM/layer_importance_mIndustrial_and_Scientific-1.0-5-5.json}
LOAD_PATH=${3:-/llm-reco-ssd-share/baohonghui/Baselines/HUM/ckp/hum_v1_qwen2.pth}
DATASET=${4:-mIndustrial_and_Scientific}

if [ -z "$SCORES_PATH" ] || [ -z "$LOAD_PATH" ]; then
    echo "Usage: $0 <config_path> <scores_json_path> <load_path> [dataset]"
    echo "Example: $0 configs/train_HUM.yaml layer_importance_mIndustrial_and_Scientific.json ckp/hum_v1_qwen2.pth"
    exit 1
fi

ks=(4 8 16)

echo "========================================================================"
echo "Automated EffiR Pruning Sweep (K = 4, 8, 16)"
echo "========================================================================"
echo "Config: $CONFIG"
echo "Scores: $SCORES_PATH"
echo "Model:  $LOAD_PATH"
echo "Dataset: $DATASET"
echo "========================================================================"

RESULTS_LOG="pruning_sweep_results.log"
echo "Pruning Sweep Log - $(date)" > $RESULTS_LOG
echo "-------------------------------------------" >> $RESULTS_LOG

for k in "${ks[@]}"; do
    SAVE_PATH="ckp/pruned_model_k${k}.pth"
    INFO_PATH="${SAVE_PATH}.info.json"
    
    echo ""
    echo ">>> Step 1: Pruning with K=$k"
    python3 prune_model.py \
        --config "$CONFIG" \
        --scores_path "$SCORES_PATH" \
        --load_path "$LOAD_PATH" \
        --top_k "$k" \
        --save_path "$SAVE_PATH"
    
    # Extract pruned indices from info file
    PRUNED_LAYERS=$(python3 -c "import json; print(','.join(map(str, json.load(open('$INFO_PATH'))['pruned_mlp_layers'])))")
    
    echo ">>> Step 2: Validating K=$k (Pruned Layers: $PRUNED_LAYERS)"
    # Run validation and capture NDCG@10 (or similar) if possible, for now just run it
    # We use a subshell to capture the output and append to log
    bash scripts/effir/validate_pruned.sh "$SAVE_PATH" "$PRUNED_LAYERS" "$CONFIG" "$DATASET" 1 | tee -a $RESULTS_LOG
    
    echo "------------------------------------------------------------------------" >> $RESULTS_LOG
done

echo "Sweep completed. Check $RESULTS_LOG for summarized metrics."
