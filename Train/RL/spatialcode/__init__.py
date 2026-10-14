from __future__ import annotations

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent
_RL_ROOT = _PKG_DIR.parent

# Let imports such as spatialcode.code and spatialcode.reason resolve to the
# current Train/RL layout without moving the source directories.
__path__ = [str(_PKG_DIR), str(_RL_ROOT)]

from spatialcode.code.spatial_code import ObjectCode, SceneCode
from spatialcode.encoder import SpatialEncoder
from spatialcode.config import (
    APIBackendConfig,
    CrossAttentionFusionConfig,
    DepthAnythingConfig,
    DepthHeadConfig,
    Detection3DHeadConfig,
    HFBackendConfig,
    SAM2Config,
    SpatialEncoderConfig,
    SpatialReasonerConfig,
    SpatialVQAConfig,
    TemporalTrackerConfig,
    VLLMBackendConfig,
)
from spatialcode.pipeline.code_to_answer import code_to_answer, code_to_answer_batch
from spatialcode.reason.reasoner import SpatialReasoner
from spatialcode.version import __version__


def llm(model_name: str, backend: str = "hf", **kwargs):
    return SpatialReasoner.from_model(model_name, backend=backend, **kwargs)


try:
    from spatialcode.pipeline.image_pipeline import image_to_code
    from spatialcode.pipeline.video_pipeline import video_to_code
    from spatialcode.pipeline.vqa_pipeline import SpatialVQA
except Exception:
    image_to_code = None
    video_to_code = None
    SpatialVQA = None


__all__ = [
    "ObjectCode",
    "SceneCode",
    "SpatialEncoder",
    "SpatialEncoderConfig",
    "SpatialReasonerConfig",
    "SpatialVQAConfig",
    "SAM2Config",
    "DepthAnythingConfig",
    "CrossAttentionFusionConfig",
    "Detection3DHeadConfig",
    "DepthHeadConfig",
    "TemporalTrackerConfig",
    "HFBackendConfig",
    "VLLMBackendConfig",
    "APIBackendConfig",
    "SpatialReasoner",
    "SpatialVQA",
    "code_to_answer",
    "code_to_answer_batch",
    "image_to_code",
    "video_to_code",
    "llm",
    "__version__",
]
