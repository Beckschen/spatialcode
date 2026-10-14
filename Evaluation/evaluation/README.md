# Evaluation Scripts

This directory contains the runnable VSIBench evaluation scripts. For the full
workflow, see [../README.md](../README.md).

## Released Resources

- Model: [qicq1c/SpatialCode](https://huggingface.co/qicq1c/SpatialCode)
- Preprocessed evaluation data: [qicq1c/VSIBench](https://huggingface.co/datasets/qicq1c/VSIBench)

## Entry Points

```text
run_deploy.sh    # Launch SGLang model servers
convert.py       # Convert merged Spatial Encoder boxes to eval bbox JSON
eval.py          # Run bbox-only VSIBench inference
acc.py           # Compute VSIBench metrics
eval2.py         # Legacy evaluator for older bbox formats
```

## Typical Run

```bash
cd Evaluation/evaluation

# Terminal 1
bash run_deploy.sh /path/to/model

# Terminal 2
python convert.py \
  --input_dir /path/to/BoxDet/pred/scannet \
  --output_json ../data/scannet_bbox.json

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

python acc.py -f results/scannet_eval.json
```

## Inputs

- `--qa_json`: VSIBench question file, usually `../data/vsibench_qa.json`.
- `--bbox_json`: one JSON dict formatted as `{scene_name: [bbox, ...]}`.
- `--dataset`: `arkitscenes`, `scannet`, `scannetpp`, comma-separated subsets, or `all`.
- `--include`: `all` or comma-separated QA types.

## Outputs

- `eval.py` writes `results/*.json`.
- `acc.py` writes `vsibench_combined_results.csv`.
