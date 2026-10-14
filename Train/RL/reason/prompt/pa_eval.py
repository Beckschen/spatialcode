"""
PA (parse_anything_trl) eval-compatible prompt and option formatting.

Copied from parse_anything_trl/training/prompts.build_spatial_prompt so that
code→answer evaluation matches PA eval.py without requiring PA_ROOT.
"""

from __future__ import annotations

import json
import re
from typing import List, Dict, Any, Optional


def _format_options(options: Any) -> str:
    """Align with PA: None -> 'Yes or No'; list -> join with ', '."""
    if options is None:
        return "Yes or No"
    if isinstance(options, list):
        return ", ".join(str(o) for o in options)
    return str(options)


def build_spatial_prompt_pa(
    bboxes: List[Dict],
    question: str,
    qa_type: str,
    options: str = "Yes or No",
    wobbox: bool = False,
) -> str:
    """
    Build the spatial reasoning prompt (PA training/prompts.build_spatial_prompt).

    Args:
        bboxes: List of dicts with 'bbox_3d' and 'label' (and optionally room_id, wall_index).
        question: The spatial reasoning question.
        qa_type: e.g. 'object_rel_direction_hard', 'pairwise_configuration'.
        options: Answer options string (default "Yes or No").
        wobbox: If True, only object labels (no 3D bbox); for obj_appearance_order_wobbox.

    Returns:
        Full prompt string for the LLM (matches PA eval).
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

    is_wobbox = wobbox or (qa_type and "obj_appearance_order_wobbox" in (qa_type or "").lower())

    if is_wobbox:
        objects = [b for b in bboxes if b.get("label", "").lower() != "wall"]
        object_labels = [b.get("label", "unknown") for b in objects]
        bboxes_prompt = "The objects in the scene (ordered by their appearance sequence in the video, from first to last):\n"
        for i, label in enumerate(object_labels, 1):
            bboxes_prompt += f"{i}. {label}\n"
        base_prompt += (
            bboxes_prompt
            + "\nNote: Objects are listed in the order they first appear in the video (first appearance to last appearance).\n"
        )
    else:
        walls = [b for b in bboxes if b.get("label", "").lower() == "wall"]
        objects = [b for b in bboxes if b.get("label", "").lower() != "wall"]
        bboxes_prompt = ""
        if walls:
            bboxes_prompt = "The walls are:\n"
            for w in walls:
                bboxes_prompt += json.dumps(w) + "\n"
            bboxes_prompt += "\n"
        bboxes_prompt += "The bounding boxes are:\n"
        for b in objects:
            bboxes_prompt += json.dumps(b) + "\n"
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

    qa_type = qa_type or ""
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

    base_prompt += f"Based on the provided bounding boxes, consider the spatial arrangement of all objects and answer the following question:\n {question}"
    if options is not None and options != "":
        base_prompt += f" Choose from the options:\n {options}\n\n"
    else:
        base_prompt += "\n\n"
    base_prompt += (
        "Think step by step to determine the correct spatial relationship.\n"
        "Keep your reasoning concise (under 1000 words).\n"
        "IMPORTANT: You MUST end your response with exactly this format:\n"
    )

    MultiChoice = ["object_rel_direction", "object_rel_distance", "obj_appearance_order", "obj_appearance_order_wobbox", "route_planning"]
    Numeric = ["object_counting", "object_size_estimation", "room_size_estimation", "object_abs_distance"]
    Yes_No = ["pairwise_configuration", "pairwise_compatibility"]

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
    elif qa_type and ("pairwise_configuration" in qa_type or "pairwise_compatibility" in qa_type):
        base_prompt += "Final Answer: Yes\nor\nFinal Answer: No\n"

    return base_prompt


def make_prompt_pa(
    bboxes: List[Dict],
    question: str,
    options: Any,
    qa_type: Optional[str] = None,
) -> str:
    """
    PA eval-style prompt for bbox-only (no video). Same as PA evaluation/prompt.make_prompt
    with use_video=False, use_bbox=True.
    """
    options_str = _format_options(options)
    return build_spatial_prompt_pa(bboxes, question, qa_type or "", options_str)
