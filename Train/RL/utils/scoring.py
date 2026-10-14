"""Shared PA-compatible scoring for the code-to-answer scripts."""


MRA_THRESHOLDS = [0.5 + 0.05 * i for i in range(10)]
_NUMERIC_QA_TYPE_KEYWORDS = [
    "object_abs_distance", "object_counting",
    "object_size_estimation", "room_size_estimation",
]


def _is_numeric(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _calculate_mra(pred_val, gt_val: float) -> float:
    """Match PA acc.py: mean of strict relative-error threshold comparisons."""
    try:
        pred_val = float(pred_val)
        gt_val = float(gt_val)
    except (TypeError, ValueError):
        return 0.0
    if gt_val == 0:
        return 1.0 if pred_val == 0 else 0.0
    score = 0
    for theta in MRA_THRESHOLDS:
        relative_error = abs(pred_val - gt_val) / gt_val
        if relative_error < (1 - theta):
            score += 1
    return score / len(MRA_THRESHOLDS)


def _is_numeric_qa_type(qa_type: str) -> bool:
    if not qa_type:
        return False
    return any(k in qa_type.lower() for k in _NUMERIC_QA_TYPE_KEYWORDS)


def _normalize_for_compare(answer: str, solution: str, qa_type: str) -> tuple:
    """Normalize numbers, choice letters and yes/no answers as in PA eval."""
    a = (answer or "").strip()
    s = (solution or "").strip()
    if _is_numeric_qa_type(qa_type):
        try:
            n = float(s)
            s = str(int(n)) if abs(n - int(n)) < 1e-6 else str(n)
        except ValueError:
            pass
        try:
            n = float(a)
            a = str(int(n)) if abs(n - int(n)) < 1e-6 else str(n)
        except ValueError:
            pass
        return (a, s)
    if qa_type and any(k in qa_type for k in [
        "object_rel_direction", "object_rel_distance", "obj_appearance_order", "route_planning"
    ]):
        if len(a) >= 1 and a[0].upper() in "ABCD":
            a = a[0].upper()
        if len(s) >= 1 and s[0].upper() in "ABCD":
            s = s[0].upper()
    else:
        a, s = a.lower(), s.lower()
    return (a, s)


def score_sample(answer: str, solution: str, qa_type: str) -> tuple:
    """Return (score, is_numeric_mra); other tasks use normalized exact match."""
    norm_a, norm_s = _normalize_for_compare(answer, solution, qa_type)
    if _is_numeric_qa_type(qa_type):
        if not _is_numeric(norm_s):
            return (1.0 if (norm_a == norm_s) else 0.0, True)
        return (_calculate_mra(norm_a, norm_s), True)
    is_correct = (norm_a == norm_s) if norm_a else False
    return (1.0 if is_correct else 0.0, False)
