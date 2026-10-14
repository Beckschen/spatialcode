import os
import sys
import json
import asyncio
import argparse
from tqdm import tqdm

try:
    from openai import AsyncOpenAI
except ModuleNotFoundError:
    AsyncOpenAI = None

MAX_TOKENS = 8192
"""
Prompt builder for bbox-only spatial reasoning (build_spatial_prompt).
"""

import re
from typing import List, Dict


def build_spatial_prompt(
    bboxes: List[Dict],
    question: str,
    qa_type: str,
    options: str = "Yes or No",
    wobbox: bool = False,
) -> str:
    """
    Build the spatial reasoning prompt with 3D bounding boxes.

    Args:
        bboxes:   List of bbox dicts with 'bbox_3d' and 'label' keys.
        question: The spatial reasoning question.
        qa_type:  Question type string (e.g. 'object_rel_direction').
        options:  Answer options string (default "Yes or No").
        wobbox:   If True, omit 3D coords and only list object labels
                  (used for obj_appearance_order_wobbox tasks).

    Returns:
        Complete prompt string.
    """
    base_prompt = (
        "You are a multimodal reasoning model that interprets structured scene inputs. "
        "You will be provided with a list of bounding boxes representing objects in the scene. "
        "Each bounding box corresponds to an object with a label and spatial coordinates in world coordinates. "
        "Your goal is to build a mental map of the scene and use it to reason about spatial relationships, object interactions, depth estimation, and other spatial tasks. \n\n"
        "**Coordinate System Conventions:**\n"
        "1. World Frame: Z-axis points Up. X and Y are horizontal.\n"
        "2. Yaw: 0 rad is along +X. Positive Yaw is counter-clockwise. All rotations are around the Z-axis.\n"
        "**Measurement Definitions:**\n"
        "1. **Object Size (Longest Dimension):** The longest dimension of an object is strictly defined as max(x_size, y_size, z_size) from the object's 3D oriented bounding box.\n"
        "2. **Absolute Distance:** The distance between two objects is measured as the minimum Euclidean distance between the closest points of their two bounding boxes.\n\n"
        "**IMPORTANT - Reading bbox_3d values:**\n"
        "The bbox_3d array format is: [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]\n"
        "- The YAW is the 9th (last) value - check it carefully for each object!\n"
        "- Do NOT assume yaw=0 without verifying the actual value.\n\n"
    )

    # wobbox variant: only object labels, no 3D coords
    is_wobbox = wobbox or (qa_type and "obj_appearance_order_wobbox" in qa_type.lower())

    if is_wobbox:
        objects = [bbox for bbox in bboxes if bbox.get("label", "").lower() != "wall"]
        object_labels = [bbox.get("label", "unknown") for bbox in objects]

        bboxes_prompt = "The objects in the scene (ordered by their appearance sequence in the video, from first to last):\n"
        for i, label in enumerate(object_labels, 1):
            bboxes_prompt += f"{i}. {label}\n"

        base_prompt += (
            bboxes_prompt
            + "\nNote: Objects are listed in the order they first appear in the video (first appearance to last appearance).\n"
        )
    else:
        walls = [bbox for bbox in bboxes if bbox.get("label", "").lower() == "wall"]
        objects = [bbox for bbox in bboxes if bbox.get("label", "").lower() != "wall"]

        if walls:
            bboxes_prompt = "The walls are:\n"
            for wall in walls:
                bboxes_prompt += json.dumps(wall) + "\n"
            bboxes_prompt += "\n"
        else:
            bboxes_prompt = ""

        bboxes_prompt += "The bounding boxes are:\n"
        for bbox in objects:
            bboxes_prompt += json.dumps(bbox) + "\n"

        base_prompt += (
            bboxes_prompt
            + '\nThe bounding boxes are given in this JSON format: JSON: `[{"bbox_3d":[x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw],"label":"category"}]`. Note: Objects in the list are ordered by their appearance sequence in the video (first appearance to last appearance).'
        )

    base_prompt += (
        "\n\n**Room & Wall Interpretation (CRITICAL):**\n"
        "- **Structure:** Bounding boxes with the same `room_id` belong to the same room. Within each room, `wall_index` provides the sequential order of walls from start to end.\n"
        "- **Data Imprecision:** Walls are thin segments and may be fragmented or have gaps. Use the `wall_index` sequence to infer the most plausible enclosed floor plan. "
        "**Small angle deviations (±5°) and gaps (<1m) are normal scan noise—do not reject calculations for these minor inconsistencies.**\n"
        "- **Area Calculation:** Do NOT rely on global X/Y min/max (AABB), as this overestimates irregular/L-shaped spaces. Instead, follow the wall sequence and **decompose the layout into simple rectangular sub-zones** to sum the area. "
        "**If walls are too fragmented for precise tracing, estimate a 'mental model' from the wall positions and provide a reasonable area based on the overall X/Y spans, accounting for wall thickness.**\n"
        "- **Final Goal:** Always provide a realistic numerical estimate in square meters. **An approximate answer is better than 'N/A'—only report insufficient data if fewer than 2 walls exist.**\n"
    )

    # qa_type-specific task instructions
    if qa_type and "pairwise_configuration" in qa_type:
        base_prompt += (
            "**Task Type: Pairwise Configuration**\n"
            "You are analyzing the spatial configuration between two objects. "
            "Focus on their relative positions, orientations, and how they are arranged with respect to each other.\n"
            "Consider: left/right, front/behind, above/below, facing direction, alignment, etc.\n\n"
        )
    if qa_type and "pairwise_compatibility" in qa_type:
        base_prompt += (
            "**Task Type: Pairwise Compatibility**\n"
            "You are evaluating whether two objects are spatially compatible or could interact. "
            "Consider: collision/overlap, reachability, functional proximity, clearance space, etc.\n\n"
        )
    if qa_type and "object_rel_direction" in qa_type:
        base_prompt += (
            "**Task Type: Object Relative Direction**\n"
            "You are analyzing the Object Relative Direction between two objects. "
            "Focus on their relative positions, orientations, and how they are arranged with respect to each other.\n"
            "Consider: left/right, front/behind, above/below, facing direction, alignment, etc.\n\n"
        )
    if qa_type and "object_counting" in qa_type:
        base_prompt += (
            "**Task Type: Object Counting**\n"
            "Count the number of objects matching the specified category in the scene. "
            "Each bounding box with the matching label counts as one instance. "
            "Be careful to match the exact category name (case-insensitive).\n\n"
        )
    if qa_type and "object_size_estimation" in qa_type:
        base_prompt += (
            "**Task Type: Object Size Estimation**\n"
            "Estimate the size of the specified object using its bounding box dimensions. "
            "The bbox_3d format provides [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]. "
            "The longest dimension is max(x_size, y_size, z_size). "
            "Convert to the requested unit (e.g., meters to centimeters: multiply by 100).\n\n"
        )
    if qa_type and "object_rel_distance" in qa_type:
        base_prompt += (
            "**Task Type: Object Relative Distance**\n"
            "Compare distances from a reference object to multiple candidate objects. "
            "Distance is measured as the minimum Euclidean distance between the closest points of bounding boxes "
            "(not center-to-center). Account for object sizes when computing closest-point distances.\n\n"
        )
    if qa_type and "obj_appearance_order" in qa_type:
        if is_wobbox:
            base_prompt += (
                "**Task Type: Object Appearance Order (Without 3D Bounding Boxes)**\n"
                "Determine the temporal order in which objects first appear in the video. "
                "You are provided with a list of object labels ordered by their first appearance in the video (earliest to latest). "
                "Use the list order to determine which object appears first, second, etc. "
                "You do not need 3D bounding box coordinates for this task - only the appearance sequence matters.\n\n"
            )
        else:
            base_prompt += (
                "**Task Type: Object Appearance Order**\n"
                "Determine the temporal order in which objects first appear in the video. "
                "Objects in the bounding box list are ordered by their first appearance (earliest to latest). "
                "Use the list order to determine which object appears first, second, etc.\n\n"
            )
    if qa_type and "route_planning" in qa_type:
        base_prompt += (
            "**Task Type: Route Planning**\n"
            "Plan navigation between objects by determining required turns. "
            "Consider the starting position, initial facing direction, and intermediate waypoints. "
            "At each waypoint, determine if you need to 'turn left', 'turn right', or 'turn back' to face the next target. "
            "Use object positions and orientations to calculate relative angles between consecutive waypoints.\n\n"
        )
    if qa_type and "room_size_estimation" in qa_type:
        base_prompt += (
            "**Task Type: Room Size Estimation**\n"
            "Estimate the floor area of the room using wall bounding boxes. "
            "Walls define room boundaries - use their positions to determine the room's footprint. "
            "For irregular shapes, decompose into rectangular sub-regions and sum areas. "
            "Report the answer in square meters. An approximate estimate is acceptable.\n\n"
        )
    if qa_type and "object_abs_distance" in qa_type:
        base_prompt += (
            "**Task Type: Object Absolute Distance**\n"
            "Calculate the distance between two specific objects. "
            "Distance is measured as the minimum Euclidean distance between the closest points of their bounding boxes. "
            "Account for object sizes (subtract half-dimensions along the line connecting centers). "
            "Report the answer in the requested unit (typically meters).\n\n"
        )

    base_prompt += "Based on the provided bounding boxes, consider the spatial arrangement of all objects and answer the following question:\n " + question
    if options is not None and options != "":
        base_prompt += f" Choose from the options:\n {options}\n\n"
    else:
        base_prompt += "\n\n"
    base_prompt += (
        "Think step by step to determine the correct spatial relationship.\n"
        "Keep your reasoning concise (under 1000 words).\n"
        "IMPORTANT: You MUST end your response with exactly this format:\n"
    )

    MultiChoice = [
        "object_rel_direction",
        "object_rel_distance",
        "obj_appearance_order",
        "obj_appearance_order_wobbox",
        "route_planning",
    ]
    Numeric = [
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
        "object_abs_distance",
    ]

    if qa_type and any(i in qa_type for i in MultiChoice):
        if options and options != "Yes or No":
            option_letters = re.findall(r"\b([A-Z])\.", options)
            if option_letters:
                answer_lines = [f"Final Answer: {letter}" for letter in option_letters]
                base_prompt += "\n" + "\nor\n".join(answer_lines) + "\n"
            else:
                base_prompt += (
                    "Final Answer: A\nor\nFinal Answer: B\nor\nFinal Answer: C\nor\nFinal Answer: D\n"
                )
        else:
            base_prompt += (
                "Final Answer: A\nor\nFinal Answer: B\nor\nFinal Answer: C\nor\nFinal Answer: D\n"
            )
    elif qa_type and any(i in qa_type for i in Numeric):
        base_prompt += "Final Answer: [Your numeric answer here]\n"
    elif qa_type and (
        "pairwise_configuration" in qa_type or "pairwise_compatibility" in qa_type
    ):
        base_prompt += "Final Answer: Yes\nor\nFinal Answer: No\n"

    return base_prompt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class Qwen3_API:
    def __init__(self, node, port, model):
        if AsyncOpenAI is None:
            raise RuntimeError(
                "The 'openai' package is required to run inference. "
                "Install it in the active environment before starting eval2.py."
            )
        self.client = AsyncOpenAI(
            api_key="EMPTY",
            base_url=f"http://{node}:{port}/v1",
            timeout=3600,
        )
        self.model_name = model

    async def __call__(self, prompt):
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Endpoint parsing
# ---------------------------------------------------------------------------

def parse_endpoints(node_str: str, port_str: str):
    """Parse comma-separated node/port strings into a list of (node, port) tuples.

    Single  : --node gpu01      --port 8000   → [("gpu01", "8000")]
    Multi   : --node gpu01,gpu02 --port 8000,8001 → [("gpu01","8000"),("gpu02","8001")]
    Node broadcast: --node gpu01 --port 8000,8001 → [("gpu01","8000"),("gpu01","8001")]
    Port broadcast: --node gpu01,gpu02 --port 8000 → [("gpu01","8000"),("gpu02","8000")]
    """
    def _strip_scheme(node):
        node = node.strip()
        for prefix in ("http://", "https://"):
            if node.startswith(prefix):
                node = node[len(prefix):]
        return node.split("/")[0]

    nodes = [_strip_scheme(n) for n in node_str.split(",") if n.strip()]
    ports = [p.strip() for p in port_str.split(",") if p.strip()]

    if len(nodes) == 1 and len(ports) > 1:
        nodes = nodes * len(ports)
    elif len(ports) == 1 and len(nodes) > 1:
        ports = ports * len(nodes)
    if len(ports) != len(nodes):
        raise ValueError(
            f"Number of ports ({len(ports)}) must match number of nodes ({len(nodes)})"
        )
    return list(zip(nodes, ports))


# ---------------------------------------------------------------------------
# BBox source
# ---------------------------------------------------------------------------

def load_bbox_lookup(bbox_json_path):
    """Load bbox JSON as {scene_name: [bbox, ...]}.

    eval.py intentionally supports only one bbox file format so bbox ingestion
    stays separate from the evaluation loop:

    {
      "scene_name": [
        {"bbox_3d": [x, y, z, x_size, y_size, z_size, roll, pitch, yaw],
         "label": "category"}
      ]
    }
    """
    with open(bbox_json_path) as f:
        raw_bbox_source = json.load(f)

    if not isinstance(raw_bbox_source, dict):
        raise ValueError(
            "Unsupported bbox_json format: eval.py expects a dict formatted as "
            "{scene_name: [bbox, ...]}. Use eval2.py for legacy list formats."
        )

    bbox_lookup = {}
    for scene_name, bboxes in raw_bbox_source.items():
        if not isinstance(scene_name, str) or not scene_name:
            raise ValueError(f"Invalid scene_name key in bbox_json: {scene_name!r}")
        if bboxes is None:
            bboxes = []
        if not isinstance(bboxes, list):
            raise ValueError(
                f"Invalid bbox list for scene {scene_name!r}: "
                f"expected list, got {type(bboxes).__name__}"
            )
        bbox_lookup[scene_name] = bboxes

    return bbox_lookup


def get_bboxes_for_item(item, bbox_lookup):
    return bbox_lookup.get(item.get("scene_name"), [])


def attach_bboxes_to_item(item, bbox_lookup):
    all_bboxes = get_bboxes_for_item(item, bbox_lookup)
    item["bboxes"] = filter_bboxes_by_key_object(
        all_bboxes,
        item.get("key_object"),
    )
    return item


def parse_dataset_filter(dataset_arg):
    if not dataset_arg or dataset_arg == "all":
        return None
    return {dataset.strip() for dataset in dataset_arg.split(",") if dataset.strip()}


def item_matches_dataset(item, dataset_filter):
    if dataset_filter is None:
        return True
    return item.get("dataset") in dataset_filter


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def _add_to_jsonl(item, path):
    with open(path, "a") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _load_answered_ids(output_path):
    answered_ids = set()
    if not os.path.exists(output_path):
        return answered_ids
    with open(output_path, "r") as f:
        content = f.read().strip()
    if not content:
        return answered_ids
    try:
        data = json.loads(content)
        items = data if isinstance(data, list) else [data]
        for item in items:
            answered_ids.add(item["id"])
    except json.JSONDecodeError:
        for line in content.split("\n"):
            if line.strip():
                answered_ids.add(json.loads(line.strip())["id"])
    return answered_ids


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

async def run_bbox(bboxes, question, options, qa_type, model):
    prompt = build_spatial_prompt(
        bboxes,
        question,
        qa_type,
        options=options,
    )
    output = await model(prompt)
    return output.strip(), prompt


def normalize_options(raw_options):
    if raw_options in (None, "", []):
        return None
    if isinstance(raw_options, list):
        return "\n".join(str(option) for option in raw_options)
    return str(raw_options)


OBJECT_LABEL_ALIASES = {
    "bathroom sink": "sink",
    "coffee table": "table",
    "dining table": "table",
    "mouse": "computer mouse",
    "television": "tv",
    "tv monitor": "tv",
    "washing machine": "washer",
}


def normalize_object_label(label):
    label = str(label).strip().lower()
    label = re.sub(r"[_\s-]+\d+$", "", label)
    label = label.replace("_", " ")
    label = " ".join(label.split())
    return OBJECT_LABEL_ALIASES.get(label, label)


def normalize_key_objects(key_object):
    if key_object in (None, "", []):
        return []
    if isinstance(key_object, str):
        return [normalize_object_label(key_object)]
    if isinstance(key_object, list):
        return [
            normalize_object_label(obj)
            for obj in key_object
            if obj not in (None, "")
        ]
    return [normalize_object_label(key_object)]


def filter_bboxes_by_key_object(bboxes, key_object):
    key_objects = set(normalize_key_objects(key_object))
    if not key_objects:
        return bboxes or []

    return [
        bbox
        for bbox in (bboxes or [])
        if isinstance(bbox, dict)
        and normalize_object_label(bbox.get("label", "")) in key_objects
    ]


async def process_single_item(item, model, output_path, pbar, args):
    options = normalize_options(item.get("options"))
    qa_type = item.get("question_type") or item.get("qa_type", "")

    pred, prompt = await run_bbox(
        item["bboxes"],
        item["question"],
        options,
        qa_type,
        model,
    )

    answer = pred.split("Answer")[-1].strip() if "Answer" in pred else pred
    answer = (
        answer.replace("'", "")
        .replace('"', "")
        .replace("}", "")
        .replace("{", "")
        .replace(":", "")
        .replace("*", "")
        .strip()
    )

    item["pred_answer"] = answer
    item["prompt"] = prompt
    item["model_output"] = pred
    item["model"] = args.model_name
    item["benchmark"] = "vsibench"

    is_correct = item["ground_truth"] in answer.split(".")[0]

    excluded_fields = {
        "bboxes",
        "gt_3d_bboxes",
        "pred_3d_bboxes_from_gt2d",
        "pred_3d_bboxes_from_pred2d",
        "graph",
    }
    item_to_save = {k: v for k, v in item.items() if k not in excluded_fields}
    _add_to_jsonl(item_to_save, output_path)
    pbar.update(1)
    return is_correct


async def process_queue(items, output_path, args, concurrency=128):
    """Queue-based parallel inference across multiple API endpoints."""
    endpoints = parse_endpoints(args.node, args.port)
    model_pool = [Qwen3_API(node, port, args.model_name) for node, port in endpoints]

    print(
        f"[process_queue] {len(model_pool)} endpoint(s): "
        + ", ".join(f"{n}:{p}" for n, p in endpoints)
    )

    pbar = tqdm(total=len(items), desc="Processing questions")
    input_q: asyncio.Queue = asyncio.Queue(maxsize=max(1, concurrency * 2))
    correct = 0
    lock = asyncio.Lock()

    async def producer():
        for idx, itm in enumerate(items):
            await input_q.put((idx, itm))
        for _ in range(concurrency):
            await input_q.put(None)

    async def worker():
        nonlocal correct
        while True:
            payload = await input_q.get()
            if payload is None:
                break
            idx, itm = payload
            model = model_pool[idx % len(model_pool)]
            is_correct = await process_single_item(itm, model, output_path, pbar, args)
            if is_correct:
                async with lock:
                    correct += 1

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await asyncio.gather(producer(), *workers)
    pbar.close()
    return correct


# ---------------------------------------------------------------------------
# VSIBench runner
# ---------------------------------------------------------------------------

def run_vsibench(output_path, qa_json, bbox_json, args):
    print("Running VSIBench evaluation...")

    answered_ids = _load_answered_ids(output_path)
    if not answered_ids:
        alt = output_path.replace(".jsonl", ".json")
        if alt != output_path:
            answered_ids = _load_answered_ids(alt)
    print(f"Found {len(answered_ids)} already answered questions")

    with open(qa_json) as f:
        raw = json.load(f)
    print(f"Total QA items: {len(raw)}")

    bbox_lookup = load_bbox_lookup(bbox_json)
    print(f"Loaded bboxes for {len(bbox_lookup)} scenes")

    include_set = set(args.include.split(",")) if args.include and args.include != "all" else None
    dataset_filter = parse_dataset_filter(args.dataset)
    if dataset_filter is None:
        print("Dataset filter: all")
    else:
        print("Dataset filter: " + ",".join(sorted(dataset_filter)))

    qas_to_process = []
    selected_answered = 0
    selected_total = 0
    for idx, item in enumerate(raw):
        qa_type = item.get("question_type") or item.get("qa_type", "")
        item["qa_index"] = idx
        item["id"] = f"vsibench_{idx}"
        if include_set and qa_type not in include_set:
            continue
        if not item_matches_dataset(item, dataset_filter):
            continue
        selected_total += 1
        if item["id"] in answered_ids:
            selected_answered += 1
            continue
        item["question"] = item.get("questions") or item.get("question", "")
        item["ground_truth"] = item.get("solution") or item.get("ground_truth", "")
        item["question_type"] = qa_type
        attach_bboxes_to_item(item, bbox_lookup)
        qas_to_process.append(item)

    print(f"Selected QA items: {selected_total}")
    print(f"Found {selected_answered} already answered selected questions")
    print(f"Processing {len(qas_to_process)} remaining questions")

    if not qas_to_process:
        print("No remaining questions to process; keeping existing results unchanged.")
        return

    correct = asyncio.run(
        process_queue(qas_to_process, output_path, args, concurrency=args.concurrency)
    )

    total = selected_answered + len(qas_to_process)
    if total > 0:
        print(f"VSIBench: {correct}/{total} = {correct / total:.4f}")

    # merge jsonl → json
    lines = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                if line.strip():
                    lines.append(json.loads(line.strip()))
    json_path = output_path.replace(".jsonl", ".json")
    with open(json_path, "w") as f:
        json.dump(lines, f, indent=2, ensure_ascii=False)
    if os.path.exists(output_path):
        os.remove(output_path)
    print(f"Results saved to: {json_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSIBench evaluation with Qwen3 API")

    parser.add_argument("--output_path", type=str, required=True,
                        help="Output .jsonl path (will be converted to .json at the end)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name served by vLLM")
    parser.add_argument("--node", type=str, required=True,
                        help="Node address(es), comma-separated (e.g. gpu01 or gpu01,gpu02)")
    parser.add_argument("--port", type=str, required=True,
                        help="Port(s), comma-separated; single port broadcasts to all nodes")
    parser.add_argument("--qa_json", type=str, required=True,
                        help="Path to QA JSON file")
    parser.add_argument("--bbox_json", type=str, required=True,
                        help="Path to bbox JSON dict file formatted as {scene_name: [bboxes]}")
    parser.add_argument("--bbox_field", type=str, default="pred_3d_bboxes_from_gt2d",
                        help="Kept for eval2.py CLI compatibility; eval.py reads only --bbox_json dict files")
    parser.add_argument("--concurrency", type=int, default=128,
                        help="Number of concurrent async workers")
    parser.add_argument("--include", type=str, default="all",
                        help="Comma-separated qa_types to include, or 'all'")
    parser.add_argument("--dataset", type=str, default="all",
                        help="Comma-separated datasets to include, e.g. scannet, arkitscenes, scannetpp, or 'all'")
    args = parser.parse_args()
    run_vsibench(args.output_path, args.qa_json, args.bbox_json, args)
