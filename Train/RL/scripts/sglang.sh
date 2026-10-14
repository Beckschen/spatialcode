
#!/bin/bash
# SGLang server for PA eval (OpenAI-compatible chat API).
# Usage: bash scripts/sglang.sh <gpu_ids> <model_name> <port>
#   e.g. bash scripts/sglang.sh 0,1 qwen3-4b-instruct 22002
#
# 单节点 4 卡同时跑两个服务：必须在同一终端跑，先后台 0,1 再跑 2,3，否则 2,3 可能只看到 1 卡：
#   bash scripts/sglang.sh 0,1 model1 22002 &
#   bash scripts/sglang.sh 2,3 model2 22003
#
# Higher concurrency (client BATCH_SIZE): set before running, e.g.
#   MAX_TOTAL_TOKENS=81920 MAX_RUNNING_REQUESTS=64 bash scripts/sglang.sh 0,1 qwen3-4b-instruct 22002
# If OOM, lower MAX_RUNNING_REQUESTS or MAX_TOTAL_TOKENS, or set MEM_FRACTION_STATIC=0.8

# python -m sglang.launch_server \
#  --model-path Qwen/Qwen3-VL-4B-Instruct \
#  --host 0.0.0.0 \
#  --port 22002 \
#  --max-total-tokens 60000 \
#  --tp 2

if [ -n "${SPATIALCODE_MODULE:-}" ] && command -v module >/dev/null 2>&1; then
    module load "$SPATIALCODE_MODULE"
fi
if [ -n "${SPATIALCODE_CONDA_ENV:-}" ]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "Error: SPATIALCODE_CONDA_ENV is set, but conda is not available in PATH."
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "$SPATIALCODE_CONDA_ENV"
fi
export CUDA_VISIBLE_DEVICES=$1
# 使用 2,3 时避免 SGLang 对每个 rank 设单卡 CUDA_VISIBLE_DEVICES，否则 rank1 会报 device=1 num_gpus=1
export SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=0
# 检查可见 GPU 数量（CUDA_VISIBLE_DEVICES 会重映射为 0,1,...）
NUM_VISIBLE=$(echo "$1" | tr ',' '\n' | wc -l)
echo "CUDA_VISIBLE_DEVICES=$1 -> 可见 GPU 数: $NUM_VISIBLE (在进程内编号为 0..$((NUM_VISIBLE-1)))"

declare -A MODEL_PATH_MAP=(
    ["qwen3-4b"]="Qwen/Qwen3-4B"
    ["qwen3-4b-instruct"]="Qwen/Qwen3-4B-Instruct-2507"
)
MODEL_NAME=$2
MODEL_PATH=${MODEL_PATH_MAP[$2]}

if [ -z "$MODEL_PATH" ]; then
    if [ -d "$2" ]; then
        MODEL_PATH=$2
        IS_HF_MODEL=false
    elif [[ "$2" == *"/"* ]] && [[ "$2" != *"/"*"/"* ]]; then
        MODEL_PATH=$2
        IS_HF_MODEL=true
    else
        echo "Error: Model path not found for '$2' and '$2' is not a valid directory or HuggingFace model name"
        echo "Available models: ${!MODEL_PATH_MAP[@]}"
        exit 1
    fi
else
    if [ -d "$MODEL_PATH" ]; then
        IS_HF_MODEL=false
    elif [[ "$MODEL_PATH" == *"/"* ]] && [[ "$MODEL_PATH" != *"/"*"/"* ]] && [[ ! "$MODEL_PATH" =~ ^/ ]]; then
        IS_HF_MODEL=true
    else
        IS_HF_MODEL=false
        if [ ! -d "$MODEL_PATH" ]; then
            echo "Error: Model path does not exist: $MODEL_PATH"
            exit 1
        fi
    fi
fi

if [ "$IS_HF_MODEL" = false ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path does not exist: $MODEL_PATH"
    exit 1
fi

IFS=',' read -ra GPUS <<< "$1"
REQUESTED_TP=${#GPUS[@]}
# 实际可见 GPU 数（CUDA_VISIBLE_DEVICES 后 PyTorch 看到的数量，可能少于请求的）
ACTUAL_GPUS=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "1")
if [ "$REQUESTED_TP" -le "$ACTUAL_GPUS" ]; then
    GPU_COUNT=$REQUESTED_TP
else
    GPU_COUNT=$ACTUAL_GPUS
fi
if [ "$GPU_COUNT" -lt "$REQUESTED_TP" ]; then
    echo "注意: 请求 --tp $REQUESTED_TP 但当前仅 $ACTUAL_GPUS 张 GPU 可见，使用 --tp $GPU_COUNT"
    echo "若为单节点4卡且要同时用 0,1 和 2,3，请在同一终端先后台运行: bash scripts/sglang.sh 0,1 ... & 再运行本命令。"
fi

echo "使用 GPU 数量 (--tp): $GPU_COUNT"
echo "模型名称: $MODEL_NAME"
echo "模型路径: $MODEL_PATH"
echo "模型类型: $([ "$IS_HF_MODEL" = true ] && echo "HuggingFace Model" || echo "Local Path")"

export SGLANG_DISABLE_CUDNN_CHECK=1

# Concurrency / batch: increase for higher throughput (client sends more concurrent requests).
# - max-total-tokens: total token capacity (prompt+decode); larger = more concurrent requests.
# - max-running-requests: max concurrent requests in flight; raise to allow higher client BATCH_SIZE.
# - mem-fraction-static: KV cache (0.9 = aggressive); lower if OOM.
# Override with env, e.g.: MAX_TOTAL_TOKENS=81920 MAX_RUNNING_REQUESTS=64 ./scripts/sglang.sh 0,1 qwen3-4b-instruct 22002
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-81920}
MAX_RUNNING_REQUESTS=${MAX_RUNNING_REQUESTS:-512}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.9}
GRAMMAR_BACKEND=${GRAMMAR_BACKEND:-outlines}

SGLANG_CMD="python -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --tp $GPU_COUNT \
    --dtype bfloat16 \
    --max-total-tokens $MAX_TOTAL_TOKENS \
    --max-running-requests $MAX_RUNNING_REQUESTS \
    --mem-fraction-static $MEM_FRACTION_STATIC \
    --grammar-backend $GRAMMAR_BACKEND \
    --host 0.0.0.0 \
    --port ${3}"

if [[ "$MODEL_NAME" == *"vl"* ]] || [[ "$MODEL_NAME" == *"VL"* ]]; then
    echo "检测到 VL 模型..."
fi

$SGLANG_CMD
