from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class BaseConfig:
    type: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]):
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merge(self, updates: dict[str, Any]):
        data = self.to_dict()
        data.update(updates)
        return type(self).from_dict(data)


@dataclass
class LLMBackendConfig(BaseConfig):
    model_name: str = "Qwen/Qwen3-4B-Instruct-2507"


@dataclass
class HFBackendConfig(LLMBackendConfig):
    type: str = "hf"
    torch_dtype: str = "bfloat16"
    device: str = "auto"


@dataclass
class VLLMBackendConfig(LLMBackendConfig):
    type: str = "vllm"
    tensor_parallel: int = 1
    max_model_len: int = 32768


@dataclass
class APIBackendConfig(LLMBackendConfig):
    type: str = "api"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"


@dataclass
class SAM2Config(BaseConfig):
    type: str = "sam2"
    model_name: str = "facebook/sam2-hiera-large"


@dataclass
class DepthAnythingConfig(BaseConfig):
    type: str = "depth_anything"
    model_name: str = "depth-anything"


@dataclass
class CrossAttentionFusionConfig(BaseConfig):
    type: str = "cross_attention"


@dataclass
class Detection3DHeadConfig(BaseConfig):
    type: str = "detection_3d"


@dataclass
class DepthHeadConfig(BaseConfig):
    type: str = "depth_head"


@dataclass
class TemporalTrackerConfig(BaseConfig):
    type: str = "temporal_tracker"


@dataclass
class SpatialEncoderConfig(BaseConfig):
    type: str = "spatial_encoder"
    backbone: SAM2Config = field(default_factory=SAM2Config)


@dataclass
class SpatialReasonerConfig(BaseConfig):
    type: str = "spatial_reasoner"
    backend: LLMBackendConfig = field(default_factory=HFBackendConfig)
    max_tokens: int = 2048
    temperature: float = 0.0
    max_reasoning_words: int = 1000


@dataclass
class SpatialVQAConfig(BaseConfig):
    type: str = "spatial_vqa"
    encoder: SpatialEncoderConfig = field(default_factory=SpatialEncoderConfig)
    reasoner: SpatialReasonerConfig = field(default_factory=SpatialReasonerConfig)
    max_frames: int = 64


CONFIG_REGISTRY = {
    "hf": HFBackendConfig,
    "vllm": VLLMBackendConfig,
    "api": APIBackendConfig,
    "sam2": SAM2Config,
    "depth_anything": DepthAnythingConfig,
    "cross_attention": CrossAttentionFusionConfig,
    "detection_3d": Detection3DHeadConfig,
    "depth_head": DepthHeadConfig,
    "temporal_tracker": TemporalTrackerConfig,
    "spatial_encoder": SpatialEncoderConfig,
    "spatial_reasoner": SpatialReasonerConfig,
    "spatial_vqa": SpatialVQAConfig,
}
