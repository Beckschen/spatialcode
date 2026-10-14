"""Adapter for the Spatial Encoder released with this repository."""

from functools import lru_cache

from .spatial_encoder import SpatialEncoder


@lru_cache(maxsize=1)
def get_encoder(model: str, device: str):
    """Share the loaded predictor between image and video entry points."""
    return SpatialEncoder.from_pretrained(model, device=device)


__all__ = ["SpatialEncoder"]
