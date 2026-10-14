#!/bin/bash
set -e
#SBATCH --job-name=sam2              # 作业名字
#SBATCH --nodes=2                    # 节点数
#SBATCH --gres=gpu:8                 # 每个节点 8 张 GPU
#SBATCH --cpus-per-task=16           # 每个 GPU 分配 16 个 CPU
#SBATCH --time=72:00:00              # 最长时间
#SBATCH --partition=a100             # 队列分区
#SBATCH --qos=normal                 # QOS，可根据集群改
#SBATCH -o logs/%x-%j.out            # 标准输出
#SBATCH -e logs/%x-%j.err            # 错误输出

# conda activate spatialcode
# export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export PYTHONPATH=$(pwd)

# m-cut3r / m-box
WANDB_MODE=disabled \
python training/train.py \
    -c training/configs/train_merged_iterable_da3.yaml \
    --use-cluster 0 \
    --num-gpus 4

# =================================
# 获取参数，设置默认值

# export NCCL_SOCKET_IFNAME=ens0

# NUM_NODES="${1:-1}"  # 默认为 1 个节点
# NUM_GPUS="${2:-8}"   # 默认为 8 张卡
# MASTER_ADDR="${3:-$(hostname -I | awk '{print $2}')}"  # 默认为第一个节点的地址
# MASTER_ADDR=$(echo "${3:-$DEFAULT_IP}" | tr -d '[:space:]')
# RANK="${4:-0}"        # 默认为 0

# # # 执行训练脚本
# python training/train.py -c training/configs/train_merged_iterable_da3.yaml \
#     --use-cluster=0 \
#     --num-nodes="$NUM_NODES" \
#     --num-gpus="$NUM_GPUS" \
#     --master-addr="$MASTER_ADDR" \
#     --master-port=29503 \
#     --node-rank="$RANK" 
# # ==================================

# # module load cuda/12.8
# python training/train.py \
#     -c training/configs/train_ca1m_iterable_da3.yaml \
#     --use-cluster 1 \
#     --num-gpus 4 \
#     --num-nodes 2 \
#     --partition h100,nvl \
#     --qos normal \
#     --account dwirtz1
