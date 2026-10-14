"""
Task-specific prompt templates for spatial reasoning.

Each template defines the system instructions, coordinate conventions,
and answer format for a particular spatial reasoning task type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class PromptTemplate:
    """A prompt template for a spatial reasoning task."""
    task_type: str
    system: str
    task_instructions: str
    answer_format: str


# ── System prompt (shared across all tasks) ───────────────────────────

SYSTEM_PROMPT = """\
You are a multimodal reasoning model that interprets structured scene inputs. \
You will be provided with a list of bounding boxes representing objects in the scene.

Coordinate System Conventions:
1. World Frame: Z-axis points Up. X and Y are horizontal.
2. Yaw: 0 rad is along +X. Positive Yaw is counter-clockwise.

Measurement Definitions:
1. Object Size: max(x_size, y_size, z_size)
2. Absolute Distance: minimum Euclidean distance between closest points

IMPORTANT - Reading bbox_3d values:
The bbox_3d array format is: [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw]
- The YAW is the 9th (last) value - check it carefully!"""


# ── Task-specific templates ───────────────────────────────────────────

TEMPLATES: Dict[str, PromptTemplate] = {}


def register_template(task_type: str, task_instructions: str, answer_format: str):
    TEMPLATES[task_type] = PromptTemplate(
        task_type=task_type,
        system=SYSTEM_PROMPT,
        task_instructions=task_instructions,
        answer_format=answer_format,
    )


# Relative Direction
register_template(
    "object_rel_direction",
    task_instructions="""\
Task Type: Object Relative Direction
You are analyzing the Object Relative Direction between two objects. \
Focus on their relative positions, orientations, and use the object's \
local coordinate frame to determine direction.""",
    answer_format="Final Answer: A or Final Answer: B or Final Answer: C or Final Answer: D",
)

# Relative Distance
register_template(
    "object_rel_distance",
    task_instructions="""\
Task Type: Object Relative Distance
Compare the closest-point distances from a reference object to multiple candidates. \
Use minimum Euclidean distance between the closest bounding box surface points.""",
    answer_format="Final Answer: A or Final Answer: B or Final Answer: C or Final Answer: D",
)

# Absolute Distance
register_template(
    "object_abs_distance",
    task_instructions="""\
Task Type: Absolute Distance Estimation
Calculate the minimum distance between the closest bounding box points of two objects. \
Report the distance in meters.""",
    answer_format="Final Answer: [numeric value in meters]",
)

# Object Counting
register_template(
    "object_counting",
    task_instructions="""\
Task Type: Object Counting
Count the number of objects matching the specified category in the scene.""",
    answer_format="Final Answer: [integer count]",
)

# Object Size
register_template(
    "object_size_estimation",
    task_instructions="""\
Task Type: Object Size Estimation
Report max(x_size, y_size, z_size) of the specified object in the requested unit.""",
    answer_format="Final Answer: [numeric value]",
)

# Room Size
register_template(
    "room_size_estimation",
    task_instructions="""\
Task Type: Room Size Estimation
Estimate the floor area from wall bounding boxes. \
Compute the rectangular area enclosed by the walls.""",
    answer_format="Final Answer: [numeric value in square meters]",
)

# Pairwise Configuration
register_template(
    "pairwise_configuration",
    task_instructions="""\
Task Type: Pairwise Configuration
Focus on relative positions, orientations, and use the object's perspective \
(not the viewer's) to determine spatial relationships like left/right, front/behind.""",
    answer_format="Final Answer: Yes or Final Answer: No",
)

# Appearance Order
register_template(
    "obj_appearance_order",
    task_instructions="""\
Task Type: Object Appearance Order
Determine the temporal order in which objects first appear in the video. \
Use the frame_ids to determine first appearance.""",
    answer_format="Final Answer: A or Final Answer: B or Final Answer: C or Final Answer: D",
)

# Route Planning
register_template(
    "route_planning",
    task_instructions="""\
Task Type: Route Planning
Plan navigation between waypoints, determining turn directions at each stop. \
Use coordinate geometry to compute heading changes.""",
    answer_format="Final Answer: A or Final Answer: B or Final Answer: C or Final Answer: D",
)


def get_template(task_type: str) -> PromptTemplate:
    """Get a prompt template by task type.

    Args:
        task_type: One of the registered task types.

    Returns:
        PromptTemplate for the task.

    Raises:
        KeyError: If task_type is not registered.
    """
    if task_type not in TEMPLATES:
        available = ", ".join(sorted(TEMPLATES.keys()))
        raise KeyError(f"Unknown task type '{task_type}'. Available: [{available}]")
    return TEMPLATES[task_type]
