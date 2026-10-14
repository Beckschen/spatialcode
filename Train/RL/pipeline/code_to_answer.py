"""
Code -> Answer pipeline: SceneCode + question -> LLM -> answer.

Used for VSI-Bench / RoboSpatial evaluation when scene code (e.g. gt_3d_bboxes)
is already available (no encoder).
"""

from __future__ import annotations

from typing import Optional, List

from spatialcode.code.spatial_code import SceneCode
from spatialcode.reason.reasoner import SpatialReasoner


def code_to_answer(
    scene_code: SceneCode,
    question: str,
    reasoner: SpatialReasoner,
    options: Optional[str] = None,
    qa_type: Optional[str] = None,
    prompt_style: str = "vsi",
    return_full: bool = False,
) -> str:
    """
    Run spatial reasoning from existing scene code (e.g. from_bbox_list).

    Args:
        scene_code: SceneCode (e.g. SceneCode.from_bbox_list(gt_3d_bboxes)).
        question: Natural language question.
        reasoner: Reasoner instance (e.g. from llm(...)).
        options: Answer options string (for VSI-Bench often "Yes or No").
        qa_type: e.g. "pairwise_configuration", "pairwise_compatibility".
        prompt_style: "vsi" or "robospatial" for dataset-compatible prompt.
        return_full: If True, return full model output instead of extracted answer.

    Returns:
        Extracted answer string, or full output if return_full=True.
    """
    return reasoner.reason(
        scene_code,
        question,
        task_type=qa_type,
        options=options,
        return_full=return_full,
        prompt_style=prompt_style,
    )


def code_to_answer_batch(
    scene_codes: List[SceneCode],
    questions: List[str],
    reasoner: SpatialReasoner,
    options_list: Optional[List[Optional[str]]] = None,
    qa_types: Optional[List[Optional[str]]] = None,
    prompt_style: str = "vsi",
    **kwargs,
) -> List[str]:
    """Batch code -> answer."""
    return reasoner.batch_reason(
        scene_codes,
        questions,
        options_list=options_list,
        task_types=qa_types,
        prompt_style=prompt_style,
        **kwargs,
    )
