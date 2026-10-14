"""
Video → Spatial Code pipeline with temporal tracking.

    >>> from spatialcode import video_to_code
    >>> code = video_to_code("walkthrough.mp4", object_prompts=[
    ...     {"label": "chair", "bbox": [20, 40, 180, 300]},
    ... ], max_frames=16)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from spatialcode.code.spatial_code import SceneCode
from spatialcode.encoder import get_encoder as _get_encoder
from spatialcode.encoder.spatial_encoder import DEFAULT_MODEL, as_rgb_frames, validate_prompts


def video_to_code(
    video: Union[str, Path, List[np.ndarray], "torch.Tensor"],
    model: str = DEFAULT_MODEL,
    object_prompts: Optional[List[dict]] = None,
    fps: Optional[float] = None,
    max_frames: int = 64,
    device: str = "cuda",
) -> SceneCode:
    """Convert a video to spatial code with temporal tracking.

    Args:
        video:          Path to video file, list of HWC numpy frames, or (T,3,H,W) tensor.
        model:          Pretrained encoder model name or path.
        object_prompts: Required labeled first-frame boxes, in original pixel
                        coordinates: [{"label": "chair", "bbox": [x1,y1,x2,y2]}].
        fps:            Sample rate (None = use all frames, up to max_frames).
        max_frames:     Maximum number of frames to process.
        device:         Compute device.

    Returns:
        SceneCode with prompted objects, in meters in the first camera frame.
        Camera poses are predicted from RGB. Objects appearing later without a
        first-frame prompt are not discovered automatically.

    Example:
        >>> code = video_to_code("walkthrough.mp4", max_frames=16,
        ...     object_prompts=[{"label": "chair", "bbox": [20, 40, 180, 300]}])

        >>> # Objects have frame_ids showing when they appeared
        >>> for obj in code:
        ...     print(f"{obj.label}: first seen at frame {obj.frame_ids[0]}")
    """
    _validate_sampling(fps, max_frames)

    # Load video if path
    if isinstance(video, (str, Path)):
        frames = _load_video(str(video), fps=fps, max_frames=max_frames)
    else:
        frames = as_rgb_frames(video)[:max_frames]

    frames = as_rgb_frames(frames)
    validate_prompts(object_prompts, *frames[0].shape[:2])
    encoder = _get_encoder(str(model), str(device))

    return encoder.encode_video(frames, object_prompts)


def _validate_sampling(fps, max_frames):
    if not isinstance(max_frames, int) or isinstance(max_frames, bool) or max_frames <= 0:
        raise ValueError("max_frames must be a positive integer.")
    if fps is not None and (not np.isfinite(fps) or fps <= 0):
        raise ValueError("fps must be a positive finite number.")


def _load_video(
    path: str,
    fps: Optional[float] = None,
    max_frames: int = 64,
) -> List[np.ndarray]:
    """Decode RGB frames; apply fps sampling consistently with OpenCV."""
    import cv2

    _validate_sampling(fps, max_frames)
    if not Path(path).is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is not None and (not np.isfinite(native_fps) or native_fps <= 0):
            raise ValueError("Video has no valid frame rate; omit fps to read consecutive frames.")
        step = max(1.0, native_fps / fps) if fps is not None else 1.0
        frame_index, next_sample = 0, 0.0
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index >= next_sample - 1e-9:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                next_sample += step
            frame_index += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError(f"Video contains no decodable frames: {path}")
    return frames
