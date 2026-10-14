# Inference

This directory runs Spatial Encoder inference on scene videos and merges
per-frame predictions into one 3D bounding-box file per scene.

Pipeline:

1. Create a scene-id list.
2. Run sharded scene inference.
3. Merge per-frame JSONL predictions into scene-level boxes.

## Prerequisites

Run from this directory:

```bash
cd Inference

export BOX_DATA_PATH=/path/to/datasets
export BOX_WEIGHTS_PATH=/path/to/checkpoints
export BOX_OUTPUT_PATH=/path/to/output
```

`BOX_OUTPUT_PATH` is optional. If it is not set, outputs are written under the
current working directory.

If metadata JSON files contain machine-specific absolute paths, remap them
without editing the dataset files:

```bash
export SPATIALCODE_PATH_REMAPS="/old/data/root=/new/data/root"
```

Known legacy SpatialCode paths are remapped automatically from `BOX_DATA_PATH`.

Expected checkpoint files:

```text
$BOX_WEIGHTS_PATH/
|-- sam2.1_hiera_large.pt
`-- Exps/BoxDet/CA1M_Hyperism/checkpoints/checkpoint.pt
```

To evaluate another Spatial Encoder checkpoint:

```bash
export BOX_EVAL_CKPT=/path/to/checkpoint.pt
```

Released model weights are available on Hugging Face:
[qicq1c/SpatialCode](https://huggingface.co/qicq1c/SpatialCode).

## Supported Datasets

`scene_inference.py` supports:

```text
arkitscenes
scannet
scannetpp
```

Expected dataset layout:

```text
$BOX_DATA_PATH/
|-- arkitscenes/val-json/{scene_id}.json
|-- vsibench-arkitscenes/
|-- vsibench-scannet/val-json/{scene_id}.json
|-- vsibench-scannet/val-json/val_frame_meta/{scene_id}.json    # Also supported
|-- vsibench-scannet/
|-- scannetpp_data/processing/val_frame_meta/{scene_id}.json
`-- scannetpp_data/processed_data/
```

## 1. Create Scene Lists

`run_inference_4_shards.sh` reads one scene id per line from:

```text
temp/{DATASET}_remaining{PART_ID}.txt
```

Example:

```bash
DATASET=scannet
PART_ID=1
mkdir -p temp

cat > "temp/${DATASET}_remaining${PART_ID}.txt" <<'EOF'
scene0025_01
scene0153_00
EOF
```

If you already have metadata JSON files, one convenient way to build the list is:

```bash
DATASET=scannet
PART_ID=1
META_DIR="$BOX_DATA_PATH/vsibench-scannet/val-json"

mkdir -p temp
find "$META_DIR" -maxdepth 1 -name '*.json' -printf '%f\n' \
  | sed 's/\.json$//' \
  | sort \
  > "temp/${DATASET}_remaining${PART_ID}.txt"
```

For multi-node runs, split this file manually and use different `PART_ID` values.

## 2. Run Sharded Inference

```bash
# Usage:
#   bash run_inference_4_shards.sh [DATASET] [PART_ID] [SAVE_DIR_NAME] [SHARDS]

bash run_inference_4_shards.sh scannet 1 scannet 4
```

Arguments:

- `DATASET`: `arkitscenes`, `scannet`, or `scannetpp`.
- `PART_ID`: reads `temp/{DATASET}_remaining{PART_ID}.txt`.
- `SAVE_DIR_NAME`: output subdirectory under `BoxDet/pred/`.
- `SHARDS`: number of parallel GPU workers.

Per-frame predictions are written to:

```text
$BOX_OUTPUT_PATH/BoxDet/pred/{SAVE_DIR_NAME}/{scene_id}/json/{clip_idx}/{frame_name}.jsonl
```

Each completed scene also gets:

```text
$BOX_OUTPUT_PATH/BoxDet/pred/{SAVE_DIR_NAME}/{scene_id}/done.flag
```

To retry scenes without outputs:

```bash
DATASET=scannet
SAVE_DIR=scannet

comm -23 \
  <(sort "temp/${DATASET}_remaining1.txt") \
  <(find "$BOX_OUTPUT_PATH/BoxDet/pred/${SAVE_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort) \
  > "temp/${DATASET}_remaining1_retry.txt"

bash run_inference_4_shards.sh "$DATASET" 1_retry "$SAVE_DIR" 4
```

## 3. Merge Predictions

```bash
# Usage:
#   bash merge.sh [DATASET] [PRED_DIR] [EXP_NAME]

bash merge.sh scannet scannet merged_bbox_3dconf_top5meanwhl1r_timestamp
```

`PRED_DIR` must match the `SAVE_DIR_NAME` used during inference.

The merge step:

1. Loads all per-frame JSONL files for each scene.
2. Converts camera-frame boxes into world coordinates.
3. Sorts detections by 3D confidence.
4. Merges the top detections for each object.

Merged output:

```text
$BOX_OUTPUT_PATH/BoxDet/pred/{PRED_DIR}/{scene_id}/{EXP_NAME}.json
```

The merged file is JSONL-style text, one object per line:

```json
{"label": "42", "bbox_3d": [x, y, z, w, h, l, qx, qy, qz, qw]}
```

## End-to-End Example

```bash
cd Inference
export BOX_DATA_PATH=/path/to/datasets
export BOX_WEIGHTS_PATH=/path/to/checkpoints
export BOX_OUTPUT_PATH=/path/to/output

DATASET=scannet
PART_ID=1
SAVE_DIR=scannet
EXP_NAME=merged_bbox_3dconf_top5meanwhl1r_timestamp

mkdir -p temp
find "$BOX_DATA_PATH/vsibench-scannet/val-json" -maxdepth 1 -name '*.json' -printf '%f\n' \
  | sed 's/\.json$//' \
  | sort \
  > "temp/${DATASET}_remaining${PART_ID}.txt"

bash run_inference_4_shards.sh "$DATASET" "$PART_ID" "$SAVE_DIR" 4
bash merge.sh "$DATASET" "$SAVE_DIR" "$EXP_NAME"
```

## Path Note

The repository was reorganized into an uppercase `Inference/` directory. If a
local script still references `inference/...`, update that path to this directory
before release, or keep a compatibility symlink in your local workspace.
