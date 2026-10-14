"""
Thinking with spatial codes: LLM-based reasoning over spatial codes.

    # Config-driven
    cfg = SpatialReasonerConfig(
        backend=VLLMBackendConfig(model_name="Qwen/Qwen3-4B", tensor_parallel=2),
    )
    reasoner = SpatialReasoner(cfg)

    # Quick (infers HF backend)
    reasoner = SpatialReasoner.from_model("Qwen/Qwen3-4B-Instruct-2507")
"""

from __future__ import annotations

import re
import logging
from typing import List, Optional

from spatialcode.config import (
    SpatialReasonerConfig,
    HFBackendConfig,
)
from spatialcode.code.spatial_code import SceneCode
from spatialcode.reason.backend.base import LLMBackend
from spatialcode.reason.prompt.builder import build_prompt
from spatialcode.reason.prompt.spatial_dataset import build_prompt_spatial_dataset
from spatialcode.reason.answer import extract_answer
from spatialcode.utils.registry import LLM_BACKENDS

# Ensure backends are registered
import spatialcode.reason.backend.base  # noqa: F401

logger = logging.getLogger(__name__)


class SpatialReasoner:
    """LLM for reasoning over structured 3D codes (thinking with spatial codes).

    Args:
        cfg: SpatialReasonerConfig (or omit for defaults).
    """

    def __init__(self, cfg: Optional[SpatialReasonerConfig] = None):
        self.cfg = cfg or SpatialReasonerConfig()
        self.backend: LLMBackend = LLM_BACKENDS.build(self.cfg.backend)

    @classmethod
    def from_model(
        cls,
        model_name: str,
        backend: str = "hf",
        **kwargs,
    ) -> "SpatialReasoner":
        """Quick constructor from model name.

        Args:
            model_name: e.g. "Qwen/Qwen3-4B-Instruct-2507"
            backend:    "hf", "vllm", or "api"
            **kwargs:   Extra backend config fields.
        """
        from spatialcode.config import CONFIG_REGISTRY
        backend_cfg_cls = CONFIG_REGISTRY[backend]
        backend_cfg = backend_cfg_cls(model_name=model_name, **kwargs)
        return cls(SpatialReasonerConfig(backend=backend_cfg))

    def reason(
        self,
        scene_code: SceneCode,
        question: str,
        task_type: Optional[str] = None,
        options: Optional[str] = None,
        return_full: bool = False,
        prompt_style: Optional[str] = None,
    ) -> str:
        """Reason over spatial codes to answer a question.

        Args:
            scene_code: SceneCode from the Spatial Encoder (or from_bbox_list for VSI-Bench).
            question:   Natural language spatial question.
            task_type:  Task type / qa_type (e.g. "pairwise_configuration"; auto-detected if None for default style).
            options:    Multiple-choice options string.
            return_full: If True, return full reasoning trace.
            prompt_style: "vsi" or "robospatial" to use VSI-Bench/RoboSpatial prompt format.

        Returns:
            Answer string (or full trace if return_full=True).
        """
        if prompt_style in ("vsi", "robospatial"):
            bbox_list = scene_code.to_bbox_list()
            prompt = build_prompt_spatial_dataset(
                bbox_list, question, options=options, qa_type=task_type
            )
        else:
            prompt = build_prompt(
                scene_code, question, task_type, options,
                max_reasoning_words=self.cfg.max_reasoning_words,
            )
        logger.debug(f"Prompt ({len(prompt)} chars):\n{prompt[:500]}...")

        response = self.backend.generate(
            prompt,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
        )

        if return_full:
            return response
        return self._extract_answer(response, prompt_style=prompt_style)

    def batch_reason(
        self,
        scene_codes: List[SceneCode],
        questions: List[str],
        task_types: Optional[List[str]] = None,
        options_list: Optional[List[str]] = None,
        prompt_style: Optional[str] = None,
        **kwargs,
    ) -> List[str]:
        """Batch reasoning over multiple scenes/questions."""
        prompts = []
        for i, (code, q) in enumerate(zip(scene_codes, questions)):
            tt = task_types[i] if task_types else None
            opts = options_list[i] if options_list else None
            if prompt_style in ("vsi", "robospatial"):
                bl = code.to_bbox_list()
                prompts.append(build_prompt_spatial_dataset(bl, q, options=opts, qa_type=tt))
            else:
                prompts.append(build_prompt(code, q, tt, opts, max_reasoning_words=self.cfg.max_reasoning_words))
        responses = self.backend.batch_generate(prompts, **kwargs)
        return [self._extract_answer(r, prompt_style=prompt_style) for r in responses]

    @staticmethod
    def _extract_answer(response: str, prompt_style: Optional[str] = None) -> str:
        if prompt_style in ("vsi", "robospatial"):
            out = extract_answer(response)
            return out if out is not None else ""
        match = re.search(r"Final Answer:\s*(.+?)(?:\n|$)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        out = extract_answer(response)
        if out is not None:
            return out
        lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
        return lines[-1] if lines else response.strip()
