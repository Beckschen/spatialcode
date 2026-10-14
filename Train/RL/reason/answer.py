"""
Answer extraction from model output (VSI-Bench / RoboSpatial compatible).

Matches parse_anything_trl training/rewards.extract_answer for consistent evaluation.
"""

from __future__ import annotations

import re
from typing import Optional


def extract_answer(text: str, debug: bool = False) -> Optional[str]:
    """
    Extract final answer from model output.

    Supports:
      - Yes / No
      - Multiple choice letters (A/B/C/D)
      - Numeric answers

    Returns the first valid "Final Answer:" (or "Answer:") match, or None.
    """
    if not text or not text.strip():
        return None

    pattern_yesno = r"Final Answer:\s*\**\[?([Yy]es|[Nn]o)\]?\**"
    match = re.search(pattern_yesno, text)
    if match:
        if debug:
            print(f"  [extract] Matched Yes/No: {match.group(0)}")
        return match.group(1).capitalize()

    pattern_mc = r"Final Answer:\s*\**\[?([A-Da-d])\]?\**"
    match = re.search(pattern_mc, text)
    if match:
        if debug:
            print(f"  [extract] Matched Multi-Choice: {match.group(0)}")
        return match.group(1).upper()

    pattern_numeric = r"Final Answer:\s*\**\[?(\d+\.?\d*)\]?\**"
    match = re.search(pattern_numeric, text)
    if match:
        if debug:
            print(f"  [extract] Matched Numeric: {match.group(0)}")
        return match.group(1)

    pattern_fallback_yesno = r"(?:^|\n)\s*Answer:\s*([Yy]es|[Nn]o)"
    match = re.search(pattern_fallback_yesno, text, re.IGNORECASE | re.MULTILINE)
    if match:
        if debug:
            print(f"  [extract] Matched Answer Yes/No: {match.group(0)}")
        return match.group(1).capitalize()

    pattern_fallback_mc = r"(?:^|\n)\s*Answer:\s*([A-Da-d])"
    match = re.search(pattern_fallback_mc, text, re.IGNORECASE | re.MULTILINE)
    if match:
        if debug:
            print(f"  [extract] Matched Answer MC: {match.group(0)}")
        return match.group(1).upper()

    pattern_fallback_numeric = r"(?:^|\n)\s*Answer:\s*(\d+\.?\d*)"
    match = re.search(pattern_fallback_numeric, text, re.MULTILINE)
    if match:
        if debug:
            print(f"  [extract] Matched Answer Numeric: {match.group(0)}")
        return match.group(1)

    if debug:
        print("  [extract] No match found")
    return None
