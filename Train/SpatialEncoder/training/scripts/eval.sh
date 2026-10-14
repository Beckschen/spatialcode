
export PYTHONPATH=$(pwd)


NUM_NODES="${1:-1}"
NUM_GPUS="${2:-8}"
MASTER_ADDR="${3:-"0.0.0.0"}"
RANK="${4:-0}"

WANDB_MODE=disabled python training/train.py \
    -c training/configs/val_ca1m_base_da3.yaml \
    --use-cluster=0 \
    --num-nodes="$NUM_NODES" \
    --num-gpus="$NUM_GPUS" \
    --master-addr="$MASTER_ADDR" \
    --master-port=29510 \
    --node-rank="$RANK" 