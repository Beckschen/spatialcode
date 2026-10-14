# Spatial Encoder

This directory contains the Spatial Encoder model, dataset loaders, Hydra
configs, and training entry point used by the SpatialCode 3D bbox pipeline.

Run commands from this directory:

```bash
cd Train/SpatialEncoder
export PYTHONPATH=$(pwd)
```

## Data Preparation

We provide the preprocessed Spatial Encoder training data on Hugging Face:

- [qicq1c/spatialencoder_full](https://huggingface.co/datasets/qicq1c/spatialencoder_full)

Download or mount this dataset and point `BOX_DATA_PATH` to the prepared data
root and `BOX_DATA_VAL_PATH` to the index-json directory before launching
training.

Set the environment variables:

```bash
export BOX_DATA_PATH=/path/to/spatialencoder_full
export BOX_DATA_VAL_PATH=/path/to/spatialencoder_full/dataset
export BOX_WEIGHTS_PATH=./training_output
export PYTHONPATH=$(pwd)
```

If your metadata jsons contain machine-specific absolute paths, remap them
without editing the dataset files:

```bash
export SPATIALCODE_PATH_REMAPS="/old/data/root=/new/data/root;/old/eval/root=/new/eval/root"
```

Known legacy SpatialCode paths are remapped automatically from `BOX_DATA_PATH`.

Expected checkpoint:

```text
$BOX_WEIGHTS_PATH/sam2.1_hiera_large.pt
```

The default experiment directory is configured as:

```text
$BOX_WEIGHTS_PATH/Exps/SpatialEncoder
```

The structure of the dataset is:

```text
$BOX_DATA_PATH/
|-- CA-1M/
|   |-- train/
|   |   `-- ca1m-train-<video_id>.tar
|   |-- val/
|   |   `-- ca1m-val-<video_id>.tar
|   `-- val-unzip/                              # Used by json-based loaders/evaluation
|-- hyperism/
|   `-- hyperism/                               # Base directory for per-frame files
|-- aria_digital_twin/
|   `-- ADT/                                    # Base directory for per-frame files
|-- pickle/
|   `-- CA-1M/
|       `-- *train*.pkl                         # CA-1M iterable training meta
|-- json_wo_pose/
|   `-- val/
|-- val-json/                                   # Optional extrinsics path used when use_extrinsic=true
`-- ...                                        # Meta json files referenced by index

$BOX_DATA_VAL_PATH/
|-- CA-1M.json                                  # Index file: [meta_json_rel_path, num_frames, "train"/"val"]
|-- hyperism.json                               # Index file: [meta_json_rel_path, num_frames, "train"/"val"]
`-- ADT.json                                    # Index file: [meta_json_rel_path, num_frames, "train"/"val"]
```

Each meta json pointed to by the index files should contain:

- `scene_id`
- `frames`: `{frame_id: {"rgb_path", "depth_path", "rgb_intrinsic", optional "camera_pose"/"img_orientation"}}`
- `objects`: object-wise annotations with the 2D/3D box fields used by training

To prepare the data from the original datasets:

- CA-1M can be downloaded following the [Cubify Anything instructions](https://github.com/apple/ml-cubifyanything).
- Hyperism/Hypersim can be downloaded following the [Hypersim instructions](https://github.com/apple/ml-hypersim). Object-level annotations are acquired from [Omni3D](https://github.com/facebookresearch/omni3d).
- Aria Digital Twin can be downloaded from the [Project Aria ADT dataset](https://www.projectaria.com/datasets/adt/).

## Configs

```text
training/configs/
|-- train_ca1m_iterable_da3.yaml       # CA-1M training
|-- train_merged_iterable_da3.yaml     # CA-1M + Hyperism + ADT training
`-- val_ca1m_base_da3.yaml             # CA-1M validation
```

## Training

Train on CA-1M:

```bash
python training/train.py \
  -c training/configs/train_ca1m_iterable_da3.yaml \
  --use-cluster 0 \
  --num-gpus 8
```

Train on CA-1M + Hyperism + ADT:

```bash
python training/train.py \
  -c training/configs/train_merged_iterable_da3.yaml \
  --use-cluster 0 \
  --num-gpus 8
```

The helper script uses the merged config by default:

```bash
bash training/scripts/train.sh
```

## Validation

Validate on CA-1M with 2D box prompts:

```bash
WANDB_MODE=disabled python training/train.py \
  -c training/configs/val_ca1m_base_da3.yaml \
  --use-cluster 0 \
  --num-nodes "${NUM_NODES:-1}" \
  --num-gpus "${NUM_GPUS:-8}" \
  --master-addr "${MASTER_ADDR:-0.0.0.0}" \
  --master-port 29510 \
  --node-rank "${RANK:-0}"
```

Or use the helper script:

```bash
bash training/scripts/eval.sh 1 8 0.0.0.0 0
```

## Model Layout

```text
modeling/
|-- backbones/
|   |-- image_encoder.py
|   |-- hieradet.py
|   `-- spatial/
|       |-- spatial_encoder.py
|       |-- interactive.py
|       |-- depthanythingv3/
|       |-- mapanything/
|       `-- vggt/
|-- memory_attention.py
|-- memory_encoder.py
`-- sam/
```

The training model is assembled by Hydra from `training.model.SAM2Train` and the
targets declared in each config.

## Outputs

Training writes checkpoints and logs under the experiment directory configured
by `scratch.exp_name`, relative to `BOX_WEIGHTS_PATH`:

```text
$BOX_WEIGHTS_PATH/Exps/SpatialEncoder/
```

Use the resulting checkpoint with inference by setting:

```bash
export BOX_EVAL_CKPT=/path/to/checkpoint.pt
```
