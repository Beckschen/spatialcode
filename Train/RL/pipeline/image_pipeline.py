"""
Image → Spatial Code pipeline.

The simplest entry point:
    >>> from spatialcode import image_to_code
    >>> code = image_to_code("room.jpg", object_prompts=[
    ...     {"label": "chair", "bbox": [20, 40, 180, 300]},
    ... ])
    >>> print(code)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from spatialcode.code.spatial_code import SceneCode
from spatialcode.encoder import get_encoder as _get_encoder
from spatialcode.encoder.spatial_encoder import DEFAULT_MODEL, as_rgb_frames, validate_prompts


def image_to_code(
    image: Union[str, Path, np.ndarray, "torch.Tensor"],
    model: str = DEFAULT_MODEL,
    object_prompts: Optional[List[dict]] = None,
    device: str = "cuda",
) -> SceneCode:
    """Convert a single image to spatial code.

    Args:
        image:          Path to image file, numpy array (H, W, 3), or tensor (3, H, W).
        model:          Pretrained encoder model name or path.
        object_prompts: Required labeled boxes in original image pixels:
                        [{"label": "sofa", "bbox": [x1,y1,x2,y2]}, ...].
        device:         Compute device.

    Returns:
        SceneCode with prompted objects in camera coordinates, in meters.

    Example:
        >>> code = image_to_code("living_room.jpg",
        ...     object_prompts=[{"label": "chair", "bbox": [20, 40, 180, 300]}])

        >>> for obj in code:
        ...     print(f"{obj.label}: pos={obj.position}, size={obj.size}")
    """
    # Load image if path
    if isinstance(image, (str, Path)):
        image = _load_image(str(image))
    import torch

    frames = image.unsqueeze(0) if isinstance(image, torch.Tensor) else [image]
    frame = as_rgb_frames(frames)[0]
    validate_prompts(object_prompts, *frame.shape[:2])
    encoder = _get_encoder(str(model), str(device))
    return encoder.encode_image(frame, object_prompts)


def _load_image(path: str) -> np.ndarray:
    """Load an image as HWC uint8 numpy array."""
    from PIL import Image
    with Image.open(path) as img:
        return np.array(img.convert("RGB"))
