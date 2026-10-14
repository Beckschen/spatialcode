# RL Utilities

This directory contains experimental utilities for code-to-answer inference,
evaluation, visualization export, and GRPO-style training.

## Package Layout

This directory includes compatibility packages for the current release layout:

```text
spatialcode/         # resolves to code/, config/, encoder/, pipeline/, reason/, utils/
spatialcode_train/   # resolves to trainer/, config/, and local reward code
```

That means existing imports such as `from spatialcode import llm` and
`from spatialcode_train.trainer.grpo_trainer import GRPOTrainer` work without
moving the source tree.

For video encoding, see the [Python API example](../../README.md#python-api-for-a-short-rgb-clip).

## Files

```text
RL/
|-- scripts/
|   |-- run_code_to_answer.py        # Run code-to-answer on a small set of samples
|   |-- eval_code_to_answer.py       # Evaluate code-to-answer predictions
|   |-- train_rl.py                  # GRPO-style RL training entry point
|   |-- export_scene_for_viz.py      # Export scene JSON for visualization
|   |-- sglang.sh                    # SGLang launch helper
|   `-- recompute_acc.sh             # Recompute saved accuracy
`-- README.md
```

## Code-to-Answer Inference

Start an SGLang server first:

```bash
cd Train/RL

MAX_TOTAL_TOKENS=60000 MAX_RUNNING_REQUESTS=64 \
bash scripts/sglang.sh 0 qwen3-4b-instruct 22002
```

`scripts/sglang.sh` accepts either a built-in model alias, a Hugging Face model
id, or a local model path. The release aliases are:

```text
qwen3-4b           -> Qwen/Qwen3-4B
qwen3-4b-instruct  -> Qwen/Qwen3-4B-Instruct-2507
```

Optional runtime environment variables:

```bash
export SPATIALCODE_MODULE=anaconda3          # Optional cluster module
export SPATIALCODE_CONDA_ENV=spatialcode     # Optional conda env to activate
export GRAMMAR_BACKEND=outlines
export MEM_FRACTION_STATIC=0.9
```

From this directory:

```bash
cd Train/RL

python scripts/run_code_to_answer.py \
  --qa_json /path/to/data.json \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --backend api \
  --base_url http://localhost:22002/v1 \
  --limit 5
```

Input data should contain scene boxes and questions in a VSI-Bench or
RoboSpatial-like format. The script reads boxes from `gt_3d_bboxes` or `bboxes`.

## Code-to-Answer Evaluation

```bash
python scripts/eval_code_to_answer.py \
  --qa_json /path/to/data.json \
  --model_name Qwen/Qwen3-4B-Instruct-2507 \
  --backend api \
  --base_url http://localhost:22002/v1 \
  --batch_size 32 \
  --output results/code_to_answer.jsonl \
  --output_json results/code_to_answer.json
```

The evaluator reports overall and per-`qa_type` scores. Numeric tasks use MRA
style scoring; choice and yes/no tasks use normalized exact matching.

## RL Training

`scripts/train_rl.py` expects the local GRPO config:

```bash
torchrun --nproc_per_node=4 scripts/train_rl.py \
  --config config/yaml/rl_grpo.yaml
```

## Visualization Export

```bash
python scripts/export_scene_for_viz.py /path/to/data.json \
  --index 0 \
  --out viz/scene_0.json
```

The exported JSON contains the selected scene name, question, QA type, answer,
and `gt_3d_bboxes`.
