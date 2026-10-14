#!/usr/bin/env bash
set -euo pipefail

# Usage: bash merge.sh [DATASET] [PRED_DIR] [EXP_NAME]
#   DATASET  arkitscenes | scannet | scannetpp (default: arkitscenes)
#   PRED_DIR output subdirectory under BoxDet/pred/ (default: same as DATASET)
#   EXP_NAME output filename without .json (default: merged_bbox_3dconf_top5meanwhl1r_timestamp)
dataset="${1:-arkitscenes}"
pred_dir="${2:-${dataset}}"
exp_name="${3:-merged_bbox_3dconf_top5meanwhl1r_timestamp}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${BOX_DATA_PATH:?BOX_DATA_PATH is not set}"
export BOX_OUTPUT_PATH="${BOX_OUTPUT_PATH:-$(pwd)}"

pred_root="${BOX_OUTPUT_PATH}/BoxDet/pred/${pred_dir}"

if [[ ! -d "${pred_root}" ]]; then
    echo "Error: pred directory not found: ${pred_root}"
    exit 1
fi

total_num_files=$(ls "${pred_root}" | wc -l)
idx=0

for scene_path in "${pred_root}"/*; do
    idx=$((idx + 1))
    scene_id=$(basename "${scene_path}")
    echo "[${idx}/${total_num_files}] Merging scene: ${scene_id}"
    "${PYTHON_BIN}" "${SCRIPT_DIR}/merge_json.py" --scene_id "${scene_id}" --pred_dir "${pred_dir}" --exp_name "${exp_name}"
done

echo "Done. Merged ${idx} scenes -> ${pred_root}/*/${exp_name}.json"
