#!/usr/bin/env python3
"""
Run code -> answer on VSI-Bench / RoboSpatial format data (e.g. spatial_dataset_all).

Example:
  # Use first sample from test data, with local HF model
  python scripts/run_code_to_answer.py \\
    --qa_json /path/to/pa/parse_anything_trl/data/spatial_dataset_all/test/data.json \\
    --model_name Qwen/Qwen3-4B-Instruct-2507 \\
    --limit 1

  # Use API backend (e.g. vLLM server)
  python scripts/run_code_to_answer.py \\
    --qa_json /path/to/data.json \\
    --model_name Qwen/Qwen3-4B-Instruct-2507 \\
    --backend api --base_url http://localhost:8000/v1 \\
    --limit 5
"""

import argparse
import json
import os
import sys

# project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spatialcode import llm
from spatialcode.code.spatial_code import SceneCode
from spatialcode.pipeline.code_to_answer import code_to_answer
from spatialcode.utils.scoring import score_sample


def _format_options(options) -> str:
    if options is None:
        return "Yes or No"
    if isinstance(options, list):
        return ", ".join(str(o) for o in options)
    return str(options)


def main():
    parser = argparse.ArgumentParser(description="Code -> Answer on spatial_dataset_all format")
    parser.add_argument("--qa_json", type=str, required=True,
                        help="Path to QA JSON (e.g. .../spatial_dataset_all/test/data.json)")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name for LLM")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm", "api"],
                        help="LLM backend")
    parser.add_argument("--base_url", type=str, default=None,
                        help="API base URL (for backend=api)")
    parser.add_argument("--limit", type=int, default=1,
                        help="Max number of samples to run (default 1)")
    parser.add_argument("--prompt_style", type=str, default="vsi", choices=["vsi", "robospatial"],
                        help="Prompt style for dataset")
    args = parser.parse_args()

    if not os.path.isfile(args.qa_json):
        print(f"Error: qa_json not found: {args.qa_json}")
        sys.exit(1)

    with open(args.qa_json) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    samples = data[: args.limit]

    kwargs = {}
    if args.backend == "api" and args.base_url:
        kwargs["base_url"] = args.base_url
    reasoner = llm(args.model_name, backend=args.backend, **kwargs)

    total = 0
    total_score = 0.0
    for i, item in enumerate(samples):
        bboxes = item.get("gt_3d_bboxes") or item.get("bboxes", [])
        if not bboxes:
            print(f"[{i}] Skip: no gt_3d_bboxes")
            continue
        question = item.get("questions") or item.get("question", "")
        options = _format_options(item.get("options"))
        qa_type = item.get("qa_type") or item.get("question_type") or "unknown"
        solution = (item.get("solution") or item.get("ground_truth") or "").strip()

        scene = SceneCode.from_bbox_list(bboxes)
        answer = code_to_answer(
            scene,
            question,
            reasoner,
            options=options,
            qa_type=qa_type,
            prompt_style=args.prompt_style,
        )
        answer = (answer or "").strip()
        score, is_numeric_mra = score_sample(answer, solution, qa_type)
        total += 1
        total_score += score

        print(f"\n[{i}] scene_name={item.get('scene_name', '')} qa_type={qa_type}")
        print(f"  Q: {question[:120]}...")
        print(f"  Solution: {solution}")
        print(f"  Pred:     {answer}")
        metric = "MRA" if is_numeric_mra else "exact match"
        print(f"  Score:    {score:.2f} ({metric})")

    if total:
        print(f"\nAccuracy: {total_score:.1f}/{total} = {total_score / total:.2%}")
    else:
        print("\nAccuracy: N/A (no samples)")


if __name__ == "__main__":
    main()
