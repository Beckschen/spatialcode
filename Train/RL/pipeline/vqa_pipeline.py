"""
End-to-end VQA: Video -> Spatial Code -> LLM -> Answer

    # Config-driven
    cfg = SpatialVQAConfig(
        encoder=SpatialEncoderConfig(backbone=SAM2Config(model_name="...")),
        reasoner=SpatialReasonerConfig(backend=VLLMBackendConfig(tensor_parallel=2)),
    )
    vqa = SpatialVQA(cfg)

    # Quick
    vqa = SpatialVQA.from_pretrained("spatial-code/vqa-4b")
    answer = vqa("room.mp4", "Is the lamp left or right of the table?")
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from spatialcode.config import SpatialVQAConfig, SpatialEncoderConfig, SpatialReasonerConfig
from spatialcode.code.spatial_code import SceneCode
# from spatialcode.encoder.spatial_encoder import SpatialEncoder
from spatialcode.reason.reasoner import SpatialReasoner


class SpatialVQA:
    """End-to-end spatial VQA from video.

    Args:
        cfg: SpatialVQAConfig (or omit for defaults).
    """

    def __init__(self, cfg: Optional[SpatialVQAConfig] = None):
        self.cfg = cfg or SpatialVQAConfig()
        # self.encoder = SpatialEncoder(self.cfg.encoder)
        self.reasoner = SpatialReasoner(self.cfg.reasoner)

    def __call__(
        self,
        video: Union[str, Path, List[np.ndarray]],
        question: str,
        task_type: Optional[str] = None,
        options: Optional[str] = None,
        return_code: bool = False,
        return_trace: bool = False,
    ) -> Union[str, dict]:
        """Run full VQA pipeline."""
        from spatialcode.pipeline.video_pipeline import _load_video

        if isinstance(video, (str, Path)):
            frames = _load_video(str(video), max_frames=self.cfg.max_frames)
        else:
            frames = video

        scene_code = self.encoder.encode_video(frames)

        response = self.reasoner.reason(
            scene_code, question,
            task_type=task_type, options=options,
            return_full=return_trace,
        )

        if return_code or return_trace:
            result = {"answer": response if not return_trace else self.reasoner._extract_answer(response)}
            if return_code:
                result["scene_code"] = scene_code
            if return_trace:
                result["reasoning_trace"] = response
            return result

        return response

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs) -> "SpatialVQA":
        """Load encoder + reasoner from a combined checkpoint."""
        cfg = SpatialVQAConfig(
            # encoder=SpatialEncoderConfig(),  # will be overridden by from_pretrained
            reasoner=SpatialReasonerConfig(),
        )
        if kwargs:
            cfg = cfg.merge(kwargs)
        return cls(cfg)
