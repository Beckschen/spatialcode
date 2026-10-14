#!/usr/bin/env bash
set -euo pipefail

# Usage: bash run_inference_4_shards.sh [DATASET] [PART_ID] [SAVE_DIR_NAME] [SHARDS]
#   DATASET       arkitscenes | scannet | scannetpp (default: arkitscenes)
#   PART_ID       scene list index: reads temp/{DATASET}_remaining{PART_ID}.txt (default: 1)
#   SAVE_DIR_NAME output subdirectory under BoxDet/pred/ (default: same as DATASET)
#   SHARDS        number of parallel GPUs (default: 4)
DATASET="${1:-arkitscenes}"
PART_ID="${2:-1}"
SAVE_DIR_NAME="${3:-${DATASET}}"
SHARDS="${4:-4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${BOX_DATA_PATH:?BOX_DATA_PATH is not set}"
: "${BOX_WEIGHTS_PATH:?BOX_WEIGHTS_PATH is not set}"
export BOX_OUTPUT_PATH="${BOX_OUTPUT_PATH:-$(pwd)}"

LIST_FILE="temp/${DATASET}_remaining${PART_ID}.txt"
PREFIX="temp/${DATASET}_remaining${PART_ID}_part_"
LOG_DIR="logs"

if [[ ! -f "${LIST_FILE}" ]]; then
    echo "Error: Input file ${LIST_FILE} not found!"
    exit 1
fi

mkdir -p temp "${LOG_DIR}"

# 清理可能残留的旧 split 文件
rm -f "${PREFIX}"*

# 按行平均切分，避免截断行
split -n "l/${SHARDS}" -d "${LIST_FILE}" "${PREFIX}"

declare -a pids
shard_idx=0

for shard_file in "${PREFIX}"*; do
  if [[ $shard_idx -ge $SHARDS ]]; then
      echo "Warning: More split files found than SHARDS count. Skipping ${shard_file}"
      continue
  fi

  log_file="${LOG_DIR}/inference_shard_${DATASET}_${PART_ID}_${shard_idx}.log"
  line_count=$(wc -l < "${shard_file}")
  echo "[Shard ${shard_idx}] GPU=${shard_idx} | ${line_count} scenes | Log -> ${log_file}"

  (
    export CUDA_VISIBLE_DEVICES=$shard_idx
    python "${SCRIPT_DIR}/scene_inference.py" "${shard_file}" "${SAVE_DIR_NAME}" "${DATASET}" \
      || { echo "ERROR processing shard ${shard_idx}"; exit 1; }
  ) > "${log_file}" 2>&1 &

  pids+=($!)
  shard_idx=$((shard_idx + 1))
done

echo "Waiting for ${#pids[@]} shards to complete..."
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "Shard ${i} completed successfully"
  else
    echo "Shard ${i} FAILED (see ${LOG_DIR}/inference_shard_${DATASET}_${PART_ID}_${i}.log)"
    failed=$((failed + 1))
  fi
done

rm -f "${PREFIX}"*

echo "All shards finished. Failed: ${failed}/${#pids[@]}"
echo "Results saved to: ${BOX_OUTPUT_PATH}/BoxDet/pred/${SAVE_DIR_NAME}"
[[ $failed -eq 0 ]]
