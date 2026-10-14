#!/usr/bin/env python3
"""
Evaluate code → answer on VSI-Bench / RoboSpatial format data.

By default uses PA (parse_anything_trl) compatible prompt and answer extraction
built into spatialcode (reason/prompt/pa_eval.py, reason/answer.py), so results
match PA eval without setting PA_ROOT.

Type field alignment: both "qa_type" and "question_type" are read from items (same
as PA evaluation/acc.py), so object_abs_distance, room_size_estimation, etc. are
reported correctly regardless of which key the dataset uses.

Scoring (aligned with parse_anything_trl/evaluation/acc.py):
  - Numeric qa_types (object_abs_distance, object_counting, object_size_estimation,
    room_size_estimation): MRA (Mean Relative Accuracy) with thresholds [0.5..0.95].
  - Yes/no and multi-choice: exact match after normalizing (e.g. A/B/C/D).

Outputs:
  - Overall accuracy (fractional for numeric = sum of MRA scores / total)
  - Per-qa_type accuracy (if multiple types present)
  - Optional: save predictions to JSONL/JSON (--output); each record has "score" (0-1).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

from tqdm import tqdm

# project root
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from spatialcode import llm
from spatialcode.reason.prompt.pa_eval import make_prompt_pa as make_prompt_pa_builtin
from spatialcode.reason.answer import extract_answer as extract_answer_builtin
from spatialcode.utils.scoring import score_sample as _score_sample


def _format_options(options) -> str:
    if options is None:
        return "Yes or No"
    if isinstance(options, list):
        return ", ".join(str(o) for o in options)
    return str(options)


def _setup_pa_prompt_and_extract(pa_root: str):
    """Import make_prompt and extract_answer from parse_anything_trl (optional override)."""
    pa_root = os.path.abspath(pa_root)
    eval_dir = os.path.join(pa_root, "evaluation")
    training_dir = os.path.join(pa_root, "training")
    if not os.path.isdir(eval_dir) or not os.path.isdir(training_dir):
        raise FileNotFoundError(f"PA dirs not found: {eval_dir} and/or {training_dir}")
    sys.path.insert(0, eval_dir)
    sys.path.insert(0, training_dir)
    from prompt import make_prompt  # evaluation/prompt.py
    from rewards import extract_answer  # training/rewards.py
    return make_prompt, extract_answer


def _extract_answer_with_mc_fallback(pred: str, extract_answer_fn, raw_options) -> str:
    """Match eval.py: extract_answer then multi-choice fallback (option text -> letter)."""
    answer = extract_answer_fn(pred)
    if answer is not None:
        return answer
    if not isinstance(raw_options, list) or not raw_options:
        return ""
    pred_l = pred.lower()
    for opt in raw_options:
        if not isinstance(opt, str):
            continue
        m = re.match(r"\s*([A-Da-d])\s*[\.\)]\s*(.+)\s*$", opt)
        if not m:
            continue
        letter, text = m.group(1).upper(), m.group(2).strip().lower()
        if text and text in pred_l:
            return letter
    return ""


# VSI-Bench 8 qa_types (PA acc.py order and short names); object_rel_direction_* -> object_rel_direction
VSI_BENCH_ORDERED_TYPES = [
    "object_counting",
    "object_abs_distance",
    "object_size_estimation",
    "room_size_estimation",
    "object_rel_distance",
    "object_rel_direction",
    "route_planning",
    "obj_appearance_order",
]
VSI_BENCH_TYPE_MAPPING = {
    "object_counting": "Obj. Count",
    "object_abs_distance": "Abs. Dist.",
    "object_size_estimation": "Obj. Size",
    "room_size_estimation": "Room Size",
    "object_rel_distance": "Rel. Dist.",
    "object_rel_direction": "Rel. Dir.",
    "route_planning": "Route Plan",
    "obj_appearance_order": "Appr. Order",
}
# Exclude these (RoboSpatial) when reporting VSI-Bench only
VSI_BENCH_EXCLUDE_QA_TYPES = {"pairwise_compatibility", "pairwise_configuration"}


def _normalize_qa_type_for_vsibench(qa_type: str) -> str:
    """Collapse object_rel_direction_* to object_rel_direction for VSI-Bench table."""
    if (qa_type or "").startswith("object_rel_direction"):
        return "object_rel_direction"
    return qa_type or "unknown"


def _print_vsibench_table(by_qa_type: dict, total: float, correct: float, path: str = ""):
    """Print VSI-Bench 8 qa_types table + Overall (micro-average), PA acc.py style."""
    # Only types that appear in results, in fixed order
    ordered = [t for t in VSI_BENCH_ORDERED_TYPES if t in by_qa_type]
    if not ordered:
        return
    headers = [VSI_BENCH_TYPE_MAPPING.get(t, t) for t in ordered]
    headers.append("Overall")
    col_width = 12
    sep = " | "
    print("\n" + "=" * (len(sep) * (len(headers) - 1) + col_width * len(headers)))
    print("VSI-Bench (micro-average)")
    if path:
        print(f"  Input: {path}")
    print("=" * (len(sep) * (len(headers) - 1) + col_width * len(headers)))
    print(sep.join(f"{h:>{col_width}}" for h in headers))
    print("-" * (len(sep) * (len(headers) - 1) + col_width * len(headers)))
    counts = [f"{by_qa_type[t]['total']:>{col_width}}" for t in ordered]
    counts.append(f"{int(total):>{col_width}}")
    print(sep.join(counts))
    print("-" * (len(sep) * (len(headers) - 1) + col_width * len(headers)))
    accuracies = []
    for t in ordered:
        d = by_qa_type[t]
        acc = (d["correct"] / d["total"] * 100) if d["total"] else 0
        accuracies.append(f"{acc:>{col_width-1}.2f}%")
    overall_acc = (correct / total * 100) if total else 0
    accuracies.append(f"{overall_acc:>{col_width-1}.2f}%")
    print(sep.join(accuracies))
    print("=" * (len(sep) * (len(headers) - 1) + col_width * len(headers)) + "\n")


def _run_recompute_from_jsonl(args):
    """Load JSON or JSONL, recompute score/correct with current metric (MRA for numeric), print metrics."""
    path = args.recompute_from
    if not os.path.isfile(path):
        print(f"Error: --recompute_from file not found: {path}")
        sys.exit(1)
    records = []
    if path.endswith(".jsonl"):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    else:
        with open(path) as f:
            data = json.load(f)
        records = data if isinstance(data, list) else [data]
    # VSI-Bench only: exclude pairwise_* and optionally normalize object_rel_direction_*
    if getattr(args, "vsi_bench_only", False):
        records = [r for r in records if (r.get("qa_type") or r.get("question_type") or "") not in VSI_BENCH_EXCLUDE_QA_TYPES]
    total = 0
    correct = 0.0
    by_qa_type = defaultdict(lambda: {"correct": 0.0, "total": 0})
    for rec in tqdm(records, desc="Recompute scores"):
        pred = (rec.get("pred") or "").strip()
        solution = (rec.get("solution") or rec.get("ground_truth") or "").strip()
        qa_type = rec.get("qa_type") or rec.get("question_type") or "unknown"
        score, _ = _score_sample(pred, solution, qa_type)
        total += 1
        correct += score
        # For VSI-Bench table, collapse object_rel_direction_* -> object_rel_direction
        agg_key = _normalize_qa_type_for_vsibench(qa_type) if getattr(args, "vsi_bench_only", False) else qa_type
        by_qa_type[agg_key]["total"] += 1
        by_qa_type[agg_key]["correct"] += score
        rec["score"] = score
        rec["correct"] = score >= 1.0
    # Print metrics
    print("\n" + "=" * 60)
    print("Code → Answer metrics (recomputed from JSONL)")
    print("=" * 60)
    print(f"  Input:       {path}")
    if getattr(args, "vsi_bench_only", False):
        print(f"  Filter:      VSI-Bench only (excluded pairwise_compatibility, pairwise_configuration)")
    print(f"  N samples:   {total}")
    acc_pct = (correct / total * 100) if total else 0
    print(f"  Accuracy:   {correct:.1f}/{total} = {acc_pct:.2f}%" if total else "  Accuracy:   N/A")
    if getattr(args, "vsi_bench_only", False) and by_qa_type:
        _print_vsibench_table(by_qa_type, total, correct, path)
    else:
        print()
        if by_qa_type:
            print("  Per qa_type (numeric types use MRA as in PA evaluation/acc.py):")
            for qt in sorted(by_qa_type.keys()):
                c, t = by_qa_type[qt]["correct"], by_qa_type[qt]["total"]
                acc = c / t if t else 0.0
                print(f"    {qt}: {c:.1f}/{t} = {acc:.2%}")
    print("=" * 60)
    if args.output:
        with open(args.output, "w") as out_file:
            for rec in records:
                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"\nUpdated JSONL (with score/correct) saved to {args.output}")
    if args.output_json and records:
        with open(args.output_json, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Full JSON saved to {args.output_json}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate code → answer (metrics: accuracy, per qa_type)")
    parser.add_argument("--qa_json", type=str, default=None,
                        help="Path to QA JSON (e.g. .../test/data.json). Not needed if --recompute_from is set.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B-Instruct-2507",
                        help="Model name for LLM")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "vllm", "api"],
                        help="LLM backend")
    parser.add_argument("--base_url", type=str, default=None,
                        help="API base URL (for backend=api)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max samples (default: all)")
    parser.add_argument("--prompt_style", type=str, default="vsi", choices=["vsi", "robospatial"],
                        help="Prompt style")
    parser.add_argument("--output", type=str, default=None,
                        help="Save predictions to JSONL (one JSON object per line)")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Save full results to a single JSON file (list of items)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Only load data and print metrics layout (no LLM calls); accuracy will be 0")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Concurrent API requests (default 16). Larger = faster but server VRAM may grow; reduce if SGLang OOM.")
    parser.add_argument("--max_tokens", type=int, default=30000,
                        help="Max new tokens per completion (default 30000 to match PA eval; truncation can drop 'Final Answer')")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default 0.0 to match PA eval)")
    parser.add_argument("--qa_type", type=str, nargs="*", default=None,
                        help="Only evaluate these qa_type(s), e.g. --qa_type object_rel_direction_hard object_rel_direction_easy. Omit to run on all types.")
    parser.add_argument("--pa_root", type=str, default=None,
                        help="Path to parse_anything_trl repo. When set, use PA evaluation prompt (prompt.make_prompt -> training/prompts.build_spatial_prompt) and rewards.extract_answer, matching eval.py exactly.")
    parser.add_argument("--dataset", type=str, default="vsibench", choices=["vsibench", "robospatial"],
                        help="Dataset name passed to PA make_prompt (used only when --pa_root is set).")
    parser.add_argument("--recompute_from", type=str, default=None,
                        help="Recompute metrics from an existing JSONL (pred/solution/qa_type). Skips LLM; --qa_json not needed. Use --output to write JSONL with updated score/correct.")
    parser.add_argument("--vsi_bench_only", action="store_true",
                        help="Exclude pairwise_compatibility and pairwise_configuration; report VSI-Bench 8 qa_types table + micro-average (for --recompute_from or normal eval).")
    args = parser.parse_args()

    # ----- Recompute-from-JSONL mode -----
    if args.recompute_from:
        _run_recompute_from_jsonl(args)
        return

    if not args.qa_json or not os.path.isfile(args.qa_json):
        print("Error: --qa_json required (path to QA JSON) when not using --recompute_from")
        sys.exit(1)

    with open(args.qa_json) as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    samples = data[: args.limit] if args.limit else data

    # Default: use built-in PA-compatible prompt + extract (no PA_ROOT). Optional: --pa_root to use external PA repo.
    use_external_pa = bool(args.pa_root and os.path.isdir(args.pa_root))
    if use_external_pa:
        make_prompt_pa, extract_answer_pa = _setup_pa_prompt_and_extract(args.pa_root)
        print(f"Using PA prompt + extract_answer from: {args.pa_root} (dataset={args.dataset})")
    else:

        def make_prompt_pa(bboxes, q, opts, qa_type):
            return make_prompt_pa_builtin(bboxes, q, opts, qa_type)

        extract_answer_pa = extract_answer_builtin
        print("Using built-in PA-compatible prompt + extract_answer (reason/prompt/pa_eval.py, reason/answer.py)")

    if not args.dry_run:
        kwargs = {}
        if args.backend == "api" and args.base_url:
            kwargs["base_url"] = args.base_url
        reasoner = llm(args.model_name, backend=args.backend, **kwargs)
    else:
        reasoner = None

    # Optional filter by qa_type
    qa_type_set = set(args.qa_type) if args.qa_type else None
    if qa_type_set:
        print(f"Filtering to qa_type in: {sorted(qa_type_set)}")

    # Build list of (index, item) for samples with bboxes (and matching qa_type if requested)
    to_process = []
    for i, item in enumerate(samples):
        bboxes = item.get("gt_3d_bboxes") or item.get("bboxes", [])
        if not bboxes:
            continue
        if qa_type_set is not None:
            qt = item.get("qa_type") or item.get("question_type") or "unknown"
            if qt not in qa_type_set:
                continue
        if args.vsi_bench_only and (item.get("qa_type") or item.get("question_type") or "") in VSI_BENCH_EXCLUDE_QA_TYPES:
            continue
        to_process.append((i, item))

    total = 0
    correct = 0.0  # sum of per-sample scores (1.0 or MRA in [0,1] for numeric)
    by_qa_type = defaultdict(lambda: {"correct": 0.0, "total": 0})
    results = []
    out_file = open(args.output, "w") if args.output else None
    batch_size = max(1, args.batch_size)
    n_total = len(to_process)

    pbar = tqdm(total=n_total, desc="Eval", unit="sample", dynamic_ncols=True)

    for start in range(0, n_total, batch_size):
        chunk = to_process[start : start + batch_size]
        if reasoner is not None:
            # PA-compatible path (built-in or external): build prompts -> batch_generate -> extract with MC fallback
            prompts = []
            for _, item in chunk:
                bboxes = item.get("gt_3d_bboxes") or item.get("bboxes", [])
                question = item.get("questions") or item.get("question", "")
                options_str = _format_options(item.get("options"))
                qa_type = item.get("qa_type") or item.get("question_type") or ""
                if use_external_pa:
                    prompts.append(make_prompt_pa(bboxes, question, options_str, args.dataset, use_video=False, use_bbox=True, qa_type=qa_type))
                else:
                    prompts.append(make_prompt_pa(bboxes, question, options_str, qa_type))
            raw_outputs = reasoner.backend.batch_generate(
                prompts,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_workers=batch_size,
            )
            answers = []
            for j, (_, item) in enumerate(chunk):
                pred = (raw_outputs[j] or "").strip()
                raw_options = item.get("options")
                a = _extract_answer_with_mc_fallback(pred, extract_answer_pa, raw_options)
                answers.append(a or "")
        else:
            answers = [""] * len(chunk)

        solutions = [item.get("solution") or item.get("ground_truth") or "" for _, item in chunk]
        qa_types = [item.get("qa_type") or item.get("question_type") or "unknown" for _, item in chunk]
        for (i, item), answer, solution, qa_type in zip(chunk, answers, solutions, qa_types):
            answer = (answer or "").strip()
            score, _ = _score_sample(answer, solution, qa_type)
            total += 1
            correct += score
            agg_key = _normalize_qa_type_for_vsibench(qa_type) if args.vsi_bench_only else qa_type
            by_qa_type[agg_key]["total"] += 1
            by_qa_type[agg_key]["correct"] += score
            question = item.get("questions") or item.get("question", "")
            rec = {
                "id": item.get("id", i),
                "scene_name": item.get("scene_name"),
                "qa_type": qa_type,  # also compatible with data that uses "question_type"
                "question": question[:200],
                "solution": solution,
                "pred": answer,
                "correct": score >= 1.0,  # boolean for exact-match; numeric may have 0 < score < 1
                "score": score,
            }
            results.append(rec)
            if out_file:
                out_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        pbar.update(len(chunk))
        pbar.set_postfix(acc=f"{correct:.1f}/{total}" if total else "0/0")

    pbar.close()
    if out_file:
        out_file.close()

    # Print metrics
    print("\n" + "=" * 60)
    print("Code → Answer evaluation metrics")
    print("=" * 60)
    print(f"  Dataset:     {args.qa_json}")
    print(f"  Model:       {args.model_name}")
    print(f"  N samples:   {total}")
    acc_pct = (correct / total * 100) if total else 0
    print(f"  Accuracy:    {correct:.1f}/{total} = {acc_pct:.2f}%" if total else "  Accuracy:    N/A (no samples)")
    if args.vsi_bench_only and by_qa_type:
        _print_vsibench_table(by_qa_type, total, correct, args.qa_json)
    else:
        print()
        if by_qa_type:
            print("  Per qa_type (numeric types use MRA as in PA evaluation/acc.py):")
            for qt in sorted(by_qa_type.keys()):
                c, t = by_qa_type[qt]["correct"], by_qa_type[qt]["total"]
                acc = c / t if t else 0.0
                print(f"    {qt}: {c:.1f}/{t} = {acc:.2%}")
    print("=" * 60)

    if args.output_json and results:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output_json}")


if __name__ == "__main__":
    main()
