# Training

This directory contains the training-side code for SpatialCode.

```text
Train/
|-- SpatialEncoder/     # Main 3D bbox / video spatial encoder training code
`-- RL/                 # Experimental code-to-answer and RL utilities
```

Most users should start with `SpatialEncoder/`.

## Spatial Encoder

The Spatial Encoder training code is a standalone Python package rooted at
`Train/SpatialEncoder`. It trains and validates the model used by the inference
pipeline.

```bash
cd Train/SpatialEncoder
export BOX_DATA_PATH=/path/to/datasets
export BOX_WEIGHTS_PATH=/path/to/checkpoints
export PYTHONPATH=$(pwd)

python training/train.py \
  -c training/configs/train_merged_iterable_da3.yaml \
  --use-cluster 0 \
  --num-gpus 8
```

See [SpatialEncoder/README.md](SpatialEncoder/README.md) for data preparation,
configs, and validation commands.

## RL Utilities

`RL/` contains experimental utilities for code-to-answer inference, evaluation,
visualization export, and GRPO-style training. It now includes compatibility
packages so imports such as `spatialcode.*` and `spatialcode_train.*` resolve to
the current `Train/RL` layout.

See [RL/README.md](RL/README.md) for details.

## Required Paths

The training configs use:

```bash
export BOX_DATA_PATH=/path/to/datasets
export BOX_WEIGHTS_PATH=/path/to/checkpoints
```

`BOX_DATA_PATH` is used for CA-1M, Hyperism, and ADT dataset paths.
`BOX_WEIGHTS_PATH` is used for pretrained and output checkpoints.
