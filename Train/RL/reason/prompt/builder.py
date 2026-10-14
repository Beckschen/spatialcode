"""
Prompt builder: assembles full LLM prompts from spatial codes + questions.
"""

from __future__ import annotations

from typing import Optional

from spatialcode.code.spatial_code import SceneCode
from spatialcode.reason.prompt.templates import get_template, SYSTEM_PROMPT, TEMPLATES


def build_prompt(
    scene_code: SceneCode,
    question: str,
    task_type: Optional[str] = None,
    options: Optional[str] = None,
    max_reasoning_words: int = 1000,
) -> str:
    """Build a full prompt for LLM spatial reasoning.

    Args:
        scene_code:  SceneCode with detected objects.
        question:    Natural language spatial question.
        task_type:   Task type for specialized instructions (auto-detected if None).
        options:     Answer options string (e.g., "A. left, B. right, C. front, D. back").
        max_reasoning_words: Max words for chain-of-thought.

    Returns:
        Complete prompt string ready for LLM.
    """
    # Auto-detect task type from question if not specified
    if task_type is None:
        task_type = _detect_task_type(question)

    # Get template
    if task_type and task_type in TEMPLATES:
        template = get_template(task_type)
        system = template.system
        task_instr = template.task_instructions
        answer_fmt = template.answer_format
    else:
        system = SYSTEM_PROMPT
        task_instr = "Analyze the spatial relationships in the scene."
        answer_fmt = "Final Answer: [your answer]"

    # Serialize scene code
    bbox_str = scene_code.to_prompt_str()

    # Assemble prompt
    parts = [
        system,
        "",
        "The bounding boxes are:",
        bbox_str,
        "",
        task_instr,
        "",
        f"Based on the provided bounding boxes, answer the following question:",
        question,
    ]

    if options:
        parts.append(f"Options: {options}")

    parts.extend([
        "",
        f"Think step by step. Keep your reasoning concise (under {max_reasoning_words} words).",
        f"IMPORTANT: End your response with exactly:",
        answer_fmt,
    ])

    return "\n".join(parts)


def _detect_task_type(question: str) -> Optional[str]:
    """Heuristic task type detection from question text."""
    q = question.lower()
    if any(w in q for w in ["left", "right", "front", "behind", "direction"]):
        if "perspective" in q or "facing" in q or "standing" in q:
            return "object_rel_direction"
        return "pairwise_configuration"
    if "distance" in q or "how far" in q or "closest" in q or "nearest" in q:
        if "which" in q:
            return "object_rel_distance"
        return "object_abs_distance"
    if "how many" in q or "count" in q:
        return "object_counting"
    if "size" in q or "how big" in q or "how tall" in q:
        return "object_size_estimation"
    if "room" in q and ("area" in q or "size" in q):
        return "room_size_estimation"
    if "appear" in q and ("first" in q or "order" in q):
        return "obj_appearance_order"
    if "route" in q or "navigate" in q or "path" in q:
        return "route_planning"
    return None
