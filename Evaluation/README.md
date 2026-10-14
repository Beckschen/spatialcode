# Evaluation

This directory evaluates VSIBench questions with bbox-only scene inputs. The
main workflow is:

1. Serve an LLM with SGLang.
2. Convert Spatial Encoder merged boxes into the evaluation bbox format.
3. Run `eval.py`.
4. Compute metrics with `acc.py`.

## Directory Layout

```text
Evaluation/
|-- data/
|   |-- vsibench_qa.json
|   |-- gt2d_bbox.jsonl
|   |-- gt3d_bbox.jsonl
|   `-- code_ver_cq.jsonl
`-- evaluation/
    |-- run_deploy.sh
    |-- convert.py
    |-- eval.py
    |-- eval2.py
    |-- acc.py
    `-- utils.py
```

Run commands from `Evaluation/evaluation` unless noted otherwise.

## Released Data and Model

- Model: [qicq1c/SpatialCode](https://huggingface.co/qicq1c/SpatialCode)
- Preprocessed evaluation data: [qicq1c/VSIBench](https://huggingface.co/datasets/qicq1c/VSIBench)

Use the model path downloaded from Hugging Face as the argument to
`run_deploy.sh`. Use the preprocessed VSIBench data for `--qa_json` and
`--bbox_json` inputs when evaluating released data.

## 1. Serve the Model

```bash
cd Evaluation/evaluation
bash run_deploy.sh /path/to/model
```

By default, `run_deploy.sh` starts two SGLang instances:

```text
ports: 23000,23001
TP_SIZE: 1
NUM_INSTANCES: 2
```

If `CUDA_VISIBLE_DEVICES` is set, the script shards those GPUs across instances.
Keep this terminal running during evaluation.

Useful environment overrides:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export NUM_INSTANCES=2
export TP_SIZE=1
export PORT_BASE=23000
export MAX_TOTAL_TOKENS=60000
export MAX_RUNNING_REQUESTS=64
export XDG_CACHE_HOME=/path/to/scratch/xdg_cache
export OUTLINES_CACHE_DIR=/path/to/scratch/outlines_cache
```

Set `GCC_HOME` if your cluster requires a specific GCC toolchain, and set
`CLEAR_JIT_CACHE=1` only when you need to remove stale JIT cache files.

## 2. Convert Spatial Encoder Boxes

`eval.py` expects one JSON dict:

```json
{
  "scene0025_01": [
    {
      "bbox_3d": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
      "label": "chair"
    }
  ]
}
```

Spatial Encoder merge outputs are stored per scene:

```text
BoxDet/pred/{dataset}/
|-- scene0025_01/
|   `-- merged_bbox_3dconf_top5meanwhl1r_timestamp.json
`-- scene0153_00/
    `-- merged_bbox_3dconf_top5meanwhl1r_timestamp.json
```

Convert them with:

```bash
python convert.py \
  --input_dir /path/to/BoxDet/pred/scannet \
  --output_json ../data/scannet_bbox.json
```

The merged files store 10D boxes:

```text
[x, y, z, w, h, length, qx, qy, qz, qw]
```

`convert.py` converts them to the 9D yaw format consumed by `eval.py`:

```text
[x, y, z, w, h, length, 0.0, 0.0, yaw]
```

Release note: `Inference/merge_json.py` writes SciPy quaternions as
`qx, qy, qz, qw`. If you change the merge output or use older merged files,
verify the quaternion order before running `convert.py`.

Useful conversion options:

```bash
# Use a non-default merged filename.
python convert.py \
  --input_dir /path/to/pred_root \
  --output_json ../data/bbox.json \
  --merged_name my_merged_result.json

# Keep labels such as chair_1 instead of converting them to chair.
python convert.py \
  --input_dir /path/to/pred_root \
  --output_json ../data/bbox.json \
  --keep_instance_suffix

# Convert only scenes that contain done.flag.
python convert.py \
  --input_dir /path/to/pred_root \
  --output_json ../data/bbox.json \
  --require_done
```

## 3. Run Evaluation

```bash
mkdir -p results

python eval.py \
  --model_name qwen3 \
  --node localhost \
  --port 23000,23001 \
  --output_path results/scannet_eval.jsonl \
  --qa_json ../data/vsibench_qa.json \
  --bbox_json ../data/scannet_bbox.json \
  --include all \
  --dataset scannet \
  --concurrency 128
```

`eval.py` writes JSONL during inference and converts it to JSON at the end:

```text
results/scannet_eval.json
```

It removes the intermediate JSONL file after conversion.

### Dataset Filtering

```bash
--dataset scannet
--dataset arkitscenes
--dataset scannetpp
--dataset scannet,scannetpp
--dataset all
```

### Task Filtering

Use `--include all` for all question types, or pass comma-separated QA types:

```bash
--include object_counting,object_abs_distance
```

## 4. Compute Metrics

```bash
python acc.py -f results/scannet_eval.json
```

`acc.py` prints:

- overall score for the selected dataset filter
- per-dataset scores for `arkitscenes`, `scannet`, and `scannetpp`
- empty-prediction counts

It also writes:

```text
vsibench_combined_results.csv
```

To score only one dataset:

```bash
python acc.py \
  -f results/scannet_eval.json \
  -d scannet
```
