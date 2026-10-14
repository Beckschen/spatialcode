from __future__ import annotations

from dataclasses import dataclass

from spatialcode.reason.answer import extract_answer


@dataclass
class RewardBreakdown:
    total: float
    correctness: float


def _normalize(value: str | None) -> str:
    return str(value or "").strip().lower().rstrip(".")


def compute_reward(
    response: str,
    ground_truth: str,
    task_type: str | None = None,
    reward_clip: tuple[float, float] = (-0.5, 1.8),
) -> RewardBreakdown:
    answer = extract_answer(response) or response
    correctness = 1.0 if _normalize(answer) == _normalize(ground_truth) else 0.0
    low, high = reward_clip
    total = max(low, min(high, correctness))
    return RewardBreakdown(total=total, correctness=correctness)
