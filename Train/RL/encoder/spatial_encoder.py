"""RGB + labeled first-frame boxes -> SceneCode using the released predictor.

The checkpoint tracks prompted objects; it does not discover semantic labels.
All frames are processed together so predicted camera poses share a coordinate
system. Output boxes use the first frame's camera axes, with distances in meters.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from spatialcode.code.spatial_code import ObjectCode, SceneCode


DEFAULT_MODEL = "qicq1c/SpatialCode"
# The released decoder normalizes both bbox coordinates and camera translations
# by 2.5 (Train/SpatialEncoder/modeling/sam/mask_decoder.py).
METRIC_SCALE = 2.5


def as_rgb_frames(frames):
    """Accept HWC numpy frames or a TCHW torch tensor, without normalization."""
    import torch

    if isinstance(frames, torch.Tensor):
        if frames.ndim != 4 or frames.shape[1] != 3:
            raise ValueError("Video tensors must have shape (T, 3, H, W).")
        frames = frames.detach().cpu()
        if frames.is_floating_point():
            frames = frames.float()
        frames = frames.permute(0, 2, 3, 1).numpy()
    if (
        not isinstance(frames, (list, tuple, np.ndarray))
        or (isinstance(frames, np.ndarray) and frames.ndim == 0)
        or len(frames) == 0
    ):
        raise ValueError("Provide at least one RGB frame.")
    result = []
    for frame in frames:
        frame = np.asarray(frame)
        if frame.ndim != 3 or frame.shape[-1] != 3 or min(frame.shape[:2]) == 0:
            raise ValueError("Each RGB frame must have shape (H, W, 3).")
        if not np.isfinite(frame).all():
            raise ValueError("RGB frames must contain finite values.")
        if np.issubdtype(frame.dtype, np.floating):
            if frame.min() < 0 or frame.max() > 1:
                raise ValueError("Floating-point RGB frames must be in [0, 1].")
            frame = np.rint(frame * 255).astype(np.uint8)
        elif frame.dtype != np.uint8:
            raise ValueError("Integer RGB frames must have dtype uint8.")
        if result and frame.shape != result[0].shape:
            raise ValueError("All video frames must have the same resolution.")
        result.append(np.ascontiguousarray(frame))
    return result


def validate_prompts(object_prompts, height, width):
    """Validate boxes before downloading weights or initializing a GPU model."""
    if not isinstance(object_prompts, (list, tuple)) or not object_prompts:
        raise ValueError(
            "object_prompts is required: provide labeled first-frame boxes, "
            "e.g. [{'label': 'chair', 'bbox': [x1, y1, x2, y2]}]. "
            "The released encoder does not automatically detect or label objects."
        )
    prompts = []
    for prompt in object_prompts:
        if (
            not isinstance(prompt, dict)
            or not isinstance(prompt.get("label"), str)
            or not prompt["label"].strip()
        ):
            raise ValueError("Every object prompt needs a nonempty string label.")
        box = np.asarray(prompt.get("bbox", []), dtype=np.float32)
        if box.shape != (4,) or not np.isfinite(box).all():
            raise ValueError("Every bbox must contain four finite xyxy coordinates.")
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("Prompt boxes must lie inside the original first frame and have positive area.")
        prompts.append({"label": prompt["label"].strip(), "bbox": box})
    return prompts


class SpatialEncoder:
    """Wrap the checkpoint-backed SAM2/DA3 predictor used by Inference/.

    Use ``from_pretrained`` to load weights. The constructor also accepts an
    initialized predictor, allowing applications to manage model lifetime.
    """

    def __init__(self, predictor):
        self.predictor = predictor

    @classmethod
    def from_pretrained(cls, model=DEFAULT_MODEL, device="cuda"):
        """Load checkpoint.pt and sam2.1_hiera_large.pt from a directory or Hub.

        A checkpoint file is also accepted when the SAM checkpoint is beside it.
        The DA3 backbone uses the normal Hugging Face cache (HF_HOME).
        """
        import torch

        device = torch.device(device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; run in an environment with a visible NVIDIA GPU.")
        path = Path(model).expanduser()
        if path.exists():
            checkpoint = path / "checkpoint.pt" if path.is_dir() else path
            sam_checkpoint = checkpoint.parent / "sam2.1_hiera_large.pt"
            for required in (checkpoint, sam_checkpoint):
                if not required.is_file():
                    raise FileNotFoundError(f"Required checkpoint not found: {required}")
        else:
            if path.is_absolute() or str(model).startswith(".") or path.suffix == ".pt":
                raise FileNotFoundError(f"Checkpoint path not found: {path}")
            from huggingface_hub import hf_hub_download

            checkpoint = hf_hub_download(str(model), "checkpoint.pt")
            sam_checkpoint = hf_hub_download(str(model), "sam2.1_hiera_large.pt")

        # The training tree currently uses absolute imports (modeling, utils).
        encoder_root = Path(__file__).resolve().parents[2] / "SpatialEncoder"
        if str(encoder_root) not in sys.path:
            sys.path.insert(0, str(encoder_root))
        from modeling.build_sam import build_sam2_video_predictor

        predictor = build_sam2_video_predictor(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            ckpt_path=str(checkpoint),
            sam_checkpoint=str(sam_checkpoint),
            device=str(device),
            # Override the training config's BOX_WEIGHTS_PATH interpolation;
            # the public API already resolved the requested checkpoint path.
            hydra_overrides_extra=[
                "model.sam_weight_path=" + json.dumps(str(sam_checkpoint)),
            ],
        )
        return cls(predictor)

    def encode_image(self, image, object_prompts=None):
        import torch

        frames = image.unsqueeze(0) if isinstance(image, torch.Tensor) else [image]
        return self.encode_video(frames, object_prompts)

    def _prepare_video(self, frames):
        import cv2
        import torch
        from depth_anything_3.utils.io.input_processor import InputProcessor

        size = self.predictor.image_size
        # Match the square RGB input used by the released dataset loader. Boxes
        # are scaled with the same transform below; no depth/GT metadata is used.
        square_frames = [cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR) for frame in frames]
        rgb = torch.from_numpy(np.stack(square_frames)).permute(0, 3, 1, 2).float() / 255
        mean = rgb.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = rgb.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        spatial, _, _ = InputProcessor()(square_frames, process_res=504)
        return {"rgb_images": (rgb - mean) / std, "spatial_images": spatial}

    def encode_video(self, frames, object_prompts=None):
        """Track first-frame prompts and merge metric boxes in one camera frame.

        No annotations, depth maps or ground-truth poses are required. Labels
        come from the caller; object IDs follow prompt order. ``frame_ids`` index
        the supplied/sampled frames. This processes one clip, without sliding
        windows; use short clips (e.g. 16 frames) to limit GPU memory.
        """
        import torch

        frames = as_rgb_frames(frames)
        height, width = frames[0].shape[:2]
        prompts = validate_prompts(object_prompts, height, width)
        observations = {i: [] for i in range(len(prompts))}
        first_extrinsic = None
        with torch.inference_mode():
            state = self.predictor.init_state(video_frames=self._prepare_video(frames))
            try:
                size = self.predictor.image_size
                for obj_id, prompt in enumerate(prompts):
                    box = prompt["bbox"] * np.array([size / width, size / height] * 2)
                    self.predictor.add_new_points_or_box(
                        state, frame_idx=0, obj_id=obj_id, box=box,
                        points=[[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]],
                        labels=[1],
                    )
                for output in self.predictor.propagate_in_video(state):
                    extrinsic = np.eye(4)
                    extrinsic[:3] = output["extrinsics"].detach().cpu().numpy().reshape(-1, 3, 4)[0]
                    # Decoder pose translations and bbox centers use the same
                    # 1/2.5 scale. Convert BOTH to meters before transforming.
                    extrinsic[:3, 3] *= METRIC_SCALE
                    if first_extrinsic is None:
                        first_extrinsic = extrinsic.copy()
                    camera_to_first = first_extrinsic @ np.linalg.inv(extrinsic)
                    boxes = output["all_pred_bbox_3d"].detach().cpu().numpy()
                    quats = output["all_pred_bbox_quat"].detach().cpu().numpy()
                    visible = output["has_object"].detach().cpu().numpy().reshape(-1)
                    from scipy.spatial.transform import Rotation

                    for index, obj_id in enumerate(output["obj_ids"]):
                        if not visible[index]:
                            continue
                        bbox = boxes[index].reshape(-1)
                        quat = quats[index].reshape(-1)
                        if (
                            not np.isfinite(bbox).all()
                            or not np.isfinite(quat).all()
                            or np.linalg.norm(quat) == 0
                            or np.any(bbox[3:6] <= 0)
                        ):
                            continue
                        position = camera_to_first[:3, :3] @ (bbox[:3] * METRIC_SCALE) + camera_to_first[:3, 3]
                        rotation = camera_to_first[:3, :3] @ Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
                        confidence = max(float(bbox[-1]), 0.0)
                        observations[obj_id].append((
                            position, bbox[3:6] * METRIC_SCALE,
                            Rotation.from_matrix(rotation).as_quat(),
                            confidence / (confidence + 1), int(output["frame_idx"]),
                        ))
            finally:
                # Spatial features are shared by the entire clip and can be large.
                state.clear()

        objects = []
        for obj_id, samples in observations.items():
            if not samples:
                continue
            # Use the release merge convention: confidence-ranked centers/sizes
            # with robust outlier rejection, and the best detection's rotation.
            best = sorted(samples, key=lambda row: row[3], reverse=True)[:20]
            centers = np.array([row[0] for row in best])
            inliers = np.arange(len(best))
            if len(best) > 5:
                distances = np.linalg.norm(centers - np.median(centers, axis=0), axis=1)
                mad = np.median(np.abs(distances - np.median(distances)))
                threshold = 3 * mad if mad > 1e-6 else distances.max() * 0.1
                candidates = np.flatnonzero(distances < threshold)
                if len(candidates) >= 2:
                    inliers = candidates
            objects.append(ObjectCode(
                label=prompts[obj_id]["label"],
                position=centers[inliers].mean(axis=0),
                size=np.array([row[1] for row in best])[inliers].mean(axis=0),
                orientation=best[0][2], track_id=obj_id,
                confidence=best[0][3], frame_ids=[row[4] for row in samples],
            ))
        return SceneCode(objects, metadata={
            "coordinate_frame": "first_camera", "units": "meters",
            "num_frames": len(frames), "pose_source": "predicted",
        })
