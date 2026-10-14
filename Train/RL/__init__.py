"""
Spatial Code — Image & Video as Code

    >>> from spatialcode import image_to_code, video_to_code
    >>> code = image_to_code("room.jpg")

    >>> from spatialcode import SpatialVQA
    >>> vqa = SpatialVQA.from_pretrained("spatial-code/vqa-4b")
    >>> answer = vqa("walkthrough.mp4", "Which object is closest to the table?")
"""

# ── Core data structures ──────────────────────────────────────────────
from spatialcode.code.spatial_code import ObjectCode, SceneCode

# ── Configs ───────────────────────────────────────────────────────────
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
from spatialcode.encoder import *
from spatialcode.pipeline.code_to_answer import code_to_answer, code_to_answer_batch

# ── High-level API ────────────────────────────────────────────────────
from spatialcode.pipeline.image_pipeline import image_to_code
from spatialcode.pipeline.video_pipeline import video_to_code
from spatialcode.pipeline.vqa_pipeline import SpatialVQA

# ── Model classes ─────────────────────────────────────────────────────
# from spatialcode.encoder.spatial_encoder import SpatialEncoder
from spatialcode.reason.reasoner import SpatialReasoner
from spatialcode.version import __version__


def llm(
    model_name: str,
    backend: str = "hf",
    **kwargs,
):
    """Convenience helper for creating a reasoner (LLM for spatial codes).

    Example:
        >>> from spatialcode import llm
        >>> reasoner = llm("Qwen/Qwen3-4B-Instruct-2507")
    """
    return SpatialReasoner.from_model(model_name, backend=backend, **kwargs)

__all__ = [
    # Data structures
    "ObjectCode", "SceneCode",
    # Configs
    "SpatialEncoderConfig", "SpatialReasonerConfig", "SpatialVQAConfig",
    "SAM2Config", "DepthAnythingConfig", "CrossAttentionFusionConfig",
    "Detection3DHeadConfig", "DepthHeadConfig", "TemporalTrackerConfig",
    "HFBackendConfig", "VLLMBackendConfig", "APIBackendConfig",
    # High-level API
    "image_to_code", "video_to_code",
    # Models
    "SpatialEncoder", "SpatialReasoner", "SpatialVQA",
    # Code -> Answer (VSI-Bench / RoboSpatial)
    "code_to_answer", "code_to_answer_batch",
    # LLM helper
    "llm",
]
